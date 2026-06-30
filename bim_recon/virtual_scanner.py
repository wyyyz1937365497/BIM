"""Virtual 2D laser scanner: render depth from 3DGS to simulate a horizontal LiDAR scan.

A horizontal depth slice through a 3DGS scene at a given height is mathematically
equivalent to a 2D LiDAR scan — each pixel's depth measures the distance to the
nearest visible surface along a horizontal ray.

When semantic features (feat.pt) are loaded, each scan point is also tagged with
its dominant semantic class by rendering a second pass with class-index-encoded
colors. This makes feat.pt an integral part of the scan pipeline: not only is
it used to locate the floor (scan center / up-axis), but every scan ray carries
a semantic label derived from the per-Gaussian language features.

The output can be visualized as a radar plot (color-coded by semantic class) and
is suitable for 2D SLAM algorithms (split-and-merge, occupancy grid, etc.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from bim_recon.gs_scene import GSScene, look_at_pose

# Palette for semantic classes (matches bim_class_names.txt order).
SEMANTIC_PALETTE: List[Tuple[float, float, float]] = [
    (0.80, 0.80, 0.80),  # wall      — gray
    (0.60, 0.40, 0.30),  # floor     — brown
    (0.95, 0.95, 0.90),  # ceiling   — off-white
    (0.85, 0.25, 0.25),  # door      — red
    (0.25, 0.60, 0.85),  # window    — blue
    (0.50, 0.50, 0.55),  # column    — dark gray
    (0.30, 0.30, 0.35),  # beam      — darker gray
    (0.80, 0.75, 0.25),  # stairs    — yellow-ish
    (0.25, 0.75, 0.35),  # furniture — green
]


@dataclass
class ScanResult:
    """A 360° horizontal laser scan from a 3DGS scene."""

    angles_deg: np.ndarray       # (M,) azimuth angles [0, 360)
    distances: np.ndarray        # (M,) horizontal distance from center (meters)
    points_2d: np.ndarray        # (M, 2) world XY coordinates
    height: float                # scan height (world up-axis coordinate)
    center_2d: np.ndarray        # (2,) scan center in world XY
    up_axis: int                 # which axis is vertical
    semantic_labels: Optional[np.ndarray] = None  # (M,) int class index per point, or None
    view_azimuths: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angles_deg": self.angles_deg.tolist(),
            "distances": self.distances.tolist(),
            "points_2d": self.points_2d.tolist(),
            "height": self.height,
            "center_2d": self.center_2d.tolist(),
            "up_axis": self.up_axis,
            "semantic_labels": self.semantic_labels.tolist() if self.semantic_labels is not None else None,
            "num_points": len(self.angles_deg),
        }


class VirtualScanner:
    """Render virtual 2D laser scans from a 3DGS scene.

    When the scene has semantic features loaded (feat.pt from SceneSplat),
    each scan point is tagged with its dominant semantic class via a second
    render pass with class-index-encoded Gaussian colors.
    """

    def __init__(self, scene: GSScene, up_axis: int = 2):
        self.scene = scene
        self.up_axis = up_axis
        self.h_axes = [i for i in range(3) if i != up_axis]
        self._has_semantics = (
            scene.semantic_querier is not None and scene.feat is not None
        )

    def scan(
        self,
        center_2d: Tuple[float, float],
        height: float,
        num_views: int = 8,
        fov: float = 60.0,
        width: int = 1024,
    ) -> ScanResult:
        """Render a 360° horizontal scan at ``height`` from ``center_2d``.

        For each viewpoint, two render passes are performed when semantics are
        available:
          1. Normal RGB+Depth render (geometry).
          2. Semantic-class render: Gaussian colors replaced by class-index
             encoding, so each pixel's color reveals the dominant feat.pt
             class at the hit surface.

        Args:
            center_2d: (x, y) scan center in world horizontal plane.
            height: World up-axis coordinate of the scan plane.
            num_views: Number of azimuth viewpoints (8 × 60° = 480° coverage).
            fov: Horizontal FOV per view in degrees.
            width: Rendered image width per view.

        Returns:
            ScanResult with angles, distances, world XY points, and optional
            semantic labels per point.
        """
        cx, cy = float(center_2d[0]), float(center_2d[1])
        h0, h1 = self.h_axes

        # Prepare semantic color encoding if feat.pt is loaded.
        semantic_colors: Optional[torch.Tensor] = None
        num_classes = 0
        if self._has_semantics:
            querier = self.scene.semantic_querier
            if querier is not None:
                dominant = querier.get_dominant_labels()  # (N,) numpy int32
                num_classes = querier.num_classes
                N = self.scene.num_gaussians
                # Encode class index in R channel: R = class_idx / (C-1).
                enc = torch.zeros((N, 3), dtype=torch.float32, device=self.scene.device)
                if num_classes > 1:
                    enc[:, 0] = torch.from_numpy(dominant.astype(np.float32)).to(self.scene.device) / (num_classes - 1)
                semantic_colors = enc

        all_angles: List[float] = []
        all_distances: List[float] = []
        all_points: List[Tuple[float, float]] = []
        all_labels: Optional[List[int]] = [] if self._has_semantics else None
        view_azimuths: List[float] = []

        fx = 0.5 * width / math.tan(0.5 * math.radians(fov))
        cx_pix = width / 2.0
        render_h = width  # square image so middle row = camera height
        middle_v = render_h // 2

        for i in range(num_views):
            azimuth_deg = i * (360.0 / num_views)
            azimuth_rad = math.radians(azimuth_deg)
            view_azimuths.append(azimuth_deg)

            eye = [0.0, 0.0, 0.0]
            eye[h0] = cx
            eye[h1] = cy
            eye[self.up_axis] = height

            target = [0.0, 0.0, 0.0]
            target[h0] = cx + math.cos(azimuth_rad)
            target[h1] = cy + math.sin(azimuth_rad)
            target[self.up_axis] = height

            up = [0.0, 0.0, 0.0]
            up[self.up_axis] = 1.0

            pose = look_at_pose(
                (eye[0], eye[1], eye[2]),
                (target[0], target[1], target[2]),
                (up[0], up[1], up[2]),
            )

            # Pass 1: geometry render (depth + alpha).
            result = self.scene.render(pose, width=width, height=render_h, fov_degrees=fov)
            depth = result.depth
            alpha = result.alpha
            depth_row = depth[middle_v].copy()
            alpha_row = alpha[middle_v]

            # Pass 2: semantic render (class-index encoded colors).
            sem_row: Optional[np.ndarray] = None
            if semantic_colors is not None and num_classes > 1:
                orig_colors = self.scene.colors
                try:
                    self.scene.colors = semantic_colors
                    sem_result = self.scene.render(pose, width=width, height=render_h, fov_degrees=fov)
                finally:
                    self.scene.colors = orig_colors
                # Decode class index from R channel: class = round(R * (C-1)).
                sem_r = sem_result.colors[middle_v, :, 0]  # (W,)
                sem_alpha = sem_result.alpha[middle_v]
                sem_row = np.round(sem_r * (num_classes - 1)).astype(np.int32)
                sem_row[sem_alpha < 0.1] = -1  # no hit

            # Unproject middle-row pixels to world XY.
            viewmat = pose.to_viewmat()
            R_w2c = viewmat[:3, :3].astype(np.float64)
            R_c2w = R_w2c.T
            eye_np = np.array(eye, dtype=np.float64)

            for u in range(width):
                if alpha_row[u] < 0.1:
                    continue
                d = float(depth_row[u])
                if d < 0.05 or d > 50.0:
                    continue

                x_cam = (u - cx_pix) / fx * d
                y_cam = (middle_v - render_h / 2.0) / fx * d  # 0 for middle row
                z_cam = d
                p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
                p_world = R_c2w @ p_cam + eye_np

                px = float(p_world[h0])
                py = float(p_world[h1])
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx * dx + dy * dy)
                angle = math.degrees(math.atan2(dy, dx)) % 360.0

                all_angles.append(angle)
                all_distances.append(dist)
                all_points.append((px, py))
                if all_labels is not None and sem_row is not None:
                    all_labels.append(int(sem_row[u]))

        return ScanResult(
            angles_deg=np.array(all_angles, dtype=np.float64),
            distances=np.array(all_distances, dtype=np.float64),
            points_2d=np.array(all_points, dtype=np.float64),
            height=height,
            center_2d=np.array([cx, cy], dtype=np.float64),
            up_axis=self.up_axis,
            semantic_labels=np.array(all_labels, dtype=np.int32) if all_labels else None,
            view_azimuths=view_azimuths,
        )


def save_scan_plot(
    scan: ScanResult,
    output_path: str,
    max_distance: float = 15.0,
    title: Optional[str] = None,
) -> str:
    """Save the scan as a radar-style PNG with semantic color-coding.

    Two panels:
      Left:  Polar radar (angle vs distance), colored by semantic class.
      Right: Top-down XY scatter, colored by semantic class.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_polar, ax_xy) = plt.subplots(
        1, 2, figsize=(16, 7),
        subplot_kw={"projection": "polar"} if False else {},
    )
    # Recreate ax_polar as polar manually since subplot_kw trick is messy.
    fig.delaxes(ax_polar)
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")

    mask = scan.distances <= max_distance
    angles_rad = np.radians(scan.angles_deg[mask])
    dists = scan.distances[mask]
    pts = scan.points_2d[mask]

    # Determine colors: semantic palette or uniform blue.
    if scan.semantic_labels is not None:
        labels = scan.semantic_labels[mask]
        colors = np.array([
            SEMANTIC_PALETTE[l] if 0 <= l < len(SEMANTIC_PALETTE) else (0.5, 0.5, 0.5)
            for l in labels
        ])
    else:
        colors = np.tile([0.3, 0.5, 0.9], (len(dists), 1))

    # --- Polar plot ---
    ax_polar.scatter(angles_rad, dists, s=0.5, c=colors, alpha=0.6)
    ax_polar.set_ylim(0, max_distance)
    ax_polar.set_title("Polar Radar Scan" if title is None else title, pad=20)
    ax_polar.grid(True, alpha=0.3)

    # --- Top-down XY scatter ---
    if len(pts) > 0:
        ax_xy.scatter(pts[:, 0], pts[:, 1], s=0.5, c=colors, alpha=0.6)
    ax_xy.plot(scan.center_2d[0], scan.center_2d[1], "k+", markersize=15, markeredgewidth=2)
    ax_xy.set_aspect("equal")
    margin = max_distance
    ax_xy.set_xlim(scan.center_2d[0] - margin, scan.center_2d[0] + margin)
    ax_xy.set_ylim(scan.center_2d[1] - margin, scan.center_2d[1] + margin)
    h0 = [i for i in range(3) if i != scan.up_axis][0]
    h1 = [i for i in range(3) if i != scan.up_axis][1]
    ax_xy.set_xlabel(f"World {'XYZ'[h0]} (m)")
    ax_xy.set_ylabel(f"World {'XYZ'[h1]} (m)")
    ax_xy.set_title(f"Top-Down (h={scan.height:.2f}m)")
    ax_xy.grid(True, alpha=0.3)

    # Legend for semantic classes (only classes present in scan).
    if scan.semantic_labels is not None:
        present = sorted(set(scan.semantic_labels[mask].tolist()))
        from matplotlib.patches import Patch
        from pathlib import Path
        class_names_path = Path(__file__).resolve().parent.parent / "data" / "bim_class_names.json"
        import json
        try:
            names_map = json.loads(class_names_path.read_text())
            names = [k for k, v in sorted(names_map.items(), key=lambda x: x[1])]
        except Exception:
            names = [f"class_{i}" for i in range(len(SEMANTIC_PALETTE))]
        legend_elems = [
            Patch(
                facecolor=SEMANTIC_PALETTE[l] if 0 <= l < len(SEMANTIC_PALETTE) else (0.5, 0.5, 0.5),
                label=names[l] if l < len(names) else f"class_{l}",
            )
            for l in present
        ]
        fig.legend(handles=legend_elems, loc="lower center", ncol=min(len(present), 9), fontsize=8)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
