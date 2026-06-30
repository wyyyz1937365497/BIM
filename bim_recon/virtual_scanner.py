"""Virtual 2D laser scanner: render depth from 3DGS to simulate a horizontal LiDAR scan.

A horizontal depth slice through a 3DGS scene at a given height is mathematically
equivalent to a 2D LiDAR scan — each pixel's depth measures the distance to the
nearest visible surface along a horizontal ray.

This module renders depth from multiple viewpoints (covering 360° azimuth),
extracts the horizontal middle row from each, and stitches them into a single
polar scan. The output can be visualized as a radar plot and is suitable for
2D SLAM algorithms (split-and-merge, occupancy grid, etc.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bim_recon.gs_scene import GSScene, look_at_pose


@dataclass
class ScanResult:
    """A 360° horizontal laser scan from a 3DGS scene."""

    angles_deg: np.ndarray       # (M,) azimuth angles [0, 360)
    distances: np.ndarray        # (M,) horizontal distance from center (meters)
    points_2d: np.ndarray        # (M, 2) world XY coordinates
    height: float                # scan height (world up-axis coordinate)
    center_2d: np.ndarray        # (2,) scan center in world XY
    up_axis: int                 # which axis is vertical
    raw_depth_rows: List[np.ndarray] = field(default_factory=list)  # per-view middle rows
    view_azimuths: List[float] = field(default_factory=list)         # per-view azimuth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angles_deg": self.angles_deg.tolist(),
            "distances": self.distances.tolist(),
            "points_2d": self.points_2d.tolist(),
            "height": self.height,
            "center_2d": self.center_2d.tolist(),
            "up_axis": self.up_axis,
            "num_points": len(self.angles_deg),
        }


class VirtualScanner:
    """Render virtual 2D laser scans from a 3DGS scene."""

    def __init__(self, scene: GSScene, up_axis: int = 2):
        """Initialize with a loaded scene.

        Args:
            scene: The GSScene to scan.
            up_axis: Vertical axis index (0=x, 1=y, 2=z). Auto-detected
                from floor centroid if not specified.
        """
        self.scene = scene
        self.up_axis = up_axis
        self.h_axes = [i for i in range(3) if i != up_axis]

    def scan(
        self,
        center_2d: Tuple[float, float],
        height: float,
        num_views: int = 8,
        fov: float = 60.0,
        width: int = 1024,
    ) -> ScanResult:
        """Render a 360° horizontal scan at ``height`` from ``center_2d``.

        Places ``num_views`` cameras equally spaced in azimuth around the scan
        center, each at the same height. Each camera renders a depth image;
        the middle row (horizontal plane at camera height) is extracted and
        unprojected to world XY coordinates.

        Args:
            center_2d: (x, y) scan center in world horizontal plane.
            height: World up-axis coordinate of the scan plane (e.g., 1.5m
                above floor for a typical wall-height scan).
            num_views: Number of azimuth viewpoints. More views = denser
                angular sampling. 8 views with 60° FOV gives full 360°
                coverage with overlap.
            fov: Horizontal FOV per view in degrees. Must satisfy
                ``num_views * fov >= 360`` for gap-free coverage.
            width: Rendered image width per view. Higher = more angular
                samples per view.

        Returns:
            ScanResult with angles, distances, and world XY points.
        """
        cx, cy = float(center_2d[0]), float(center_2d[1])
        h0, h1 = self.h_axes

        all_angles: List[float] = []
        all_distances: List[float] = []
        all_points: List[Tuple[float, float]] = []
        raw_rows: List[np.ndarray] = []
        view_azimuths: List[float] = []

        # Focal length in pixels for angle computation.
        fx = 0.5 * width / math.tan(0.5 * math.radians(fov))
        cx_pix = width / 2.0
        # Render square images so middle row is exactly at camera height.
        render_h = width
        cy_pix = render_h / 2.0
        middle_v = render_h // 2

        for i in range(num_views):
            azimuth_deg = i * (360.0 / num_views)
            azimuth_rad = math.radians(azimuth_deg)
            view_azimuths.append(azimuth_deg)

            # Camera position in world (horizontal plane + height).
            eye = [0.0, 0.0, 0.0]
            eye[h0] = cx
            eye[h1] = cy
            eye[self.up_axis] = height

            # Target: same height, looking horizontally in azimuth direction.
            target = [0.0, 0.0, 0.0]
            target[h0] = cx + math.cos(azimuth_rad)
            target[h1] = cy + math.sin(azimuth_rad)
            target[self.up_axis] = height

            # World up vector.
            up = [0.0, 0.0, 0.0]
            up[self.up_axis] = 1.0

            pose = look_at_pose(
                (eye[0], eye[1], eye[2]),
                (target[0], target[1], target[2]),
                (up[0], up[1], up[2]),
            )
            result = self.scene.render(
                pose, width=width, height=render_h, fov_degrees=fov,
            )
            depth = result.depth  # (H, W) float32
            alpha = result.alpha  # (H, W) float32

            depth_row = depth[middle_v].copy()
            alpha_row = alpha[middle_v]
            raw_rows.append(depth_row)

            # Compute camera-to-world rotation from viewmat.
            viewmat = pose.to_viewmat()  # 4x4 world-to-camera
            R_w2c = viewmat[:3, :3].astype(np.float64)
            R_c2w = R_w2c.T
            eye_np = np.array(eye, dtype=np.float64)

            for u in range(width):
                if alpha_row[u] < 0.1:
                    continue
                d = float(depth_row[u])
                if d < 0.05 or d > 50.0:
                    continue

                # Unproject pixel (u, middle_v) with depth d to camera coords.
                # Y_cam = 0 because middle_v = cy_pix (horizontal plane).
                x_cam = (u - cx_pix) / fx * d
                y_cam = (middle_v - cy_pix) / fx * d  # = 0 for middle row
                z_cam = d
                p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

                # Transform to world.
                p_world = R_c2w @ p_cam + eye_np

                px = float(p_world[h0])
                py = float(p_world[h1])

                # Polar coordinates relative to scan center.
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx * dx + dy * dy)
                angle = math.degrees(math.atan2(dy, dx)) % 360.0

                all_angles.append(angle)
                all_distances.append(dist)
                all_points.append((px, py))

        return ScanResult(
            angles_deg=np.array(all_angles, dtype=np.float64),
            distances=np.array(all_distances, dtype=np.float64),
            points_2d=np.array(all_points, dtype=np.float64),
            height=height,
            center_2d=np.array([cx, cy], dtype=np.float64),
            up_axis=self.up_axis,
            raw_depth_rows=raw_rows,
            view_azimuths=view_azimuths,
        )


def save_scan_plot(
    scan: ScanResult,
    output_path: str,
    max_distance: float = 15.0,
    title: Optional[str] = None,
) -> str:
    """Save the scan as a radar-style PNG using matplotlib.

    Renders two subplots:
      1. Polar plot: angle vs distance (radar view from above).
      2. Top-down scatter: world XY coordinates.

    Args:
        scan: The ScanResult to visualize.
        output_path: Path to save the PNG file.
        max_distance: Maximum distance to display (clips far outliers).
        title: Optional plot title.

    Returns:
        The output path.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Polar plot (radar view) ---
    ax_polar = fig.add_subplot(121, projection="polar")
    mask = scan.distances <= max_distance
    angles_rad = np.radians(scan.angles_deg[mask])
    dists = scan.distances[mask]
    ax_polar.scatter(angles_rad, dists, s=0.3, c="blue", alpha=0.5)
    ax_polar.set_ylim(0, max_distance)
    ax_polar.set_title("Polar Radar Scan" if title is None else title, pad=20)
    ax_polar.set_xlabel("Azimuth (deg)")
    ax_polar.grid(True, alpha=0.3)

    # --- Top-down XY scatter ---
    ax_xy = axes[1]
    pts = scan.points_2d[scan.distances <= max_distance]
    if len(pts) > 0:
        ax_xy.scatter(pts[:, 0], pts[:, 1], s=0.3, c="red", alpha=0.5)
    # Mark scan center
    ax_xy.plot(scan.center_2d[0], scan.center_2d[1], "k+", markersize=15, markeredgewidth=2)
    ax_xy.set_aspect("equal")
    margin = max_distance
    ax_xy.set_xlim(scan.center_2d[0] - margin, scan.center_2d[0] + margin)
    ax_xy.set_ylim(scan.center_2d[1] - margin, scan.center_2d[1] + margin)
    # Derive horizontal axis labels from up_axis.
    h0 = [i for i in range(3) if i != scan.up_axis][0]
    h1 = [i for i in range(3) if i != scan.up_axis][1]

    ax_xy.set_xlabel(f"World {'XYZ'[h0]} (m)")
    ax_xy.set_ylabel(f"World {'XYZ'[h1]} (m)")
    ax_xy.set_title(f"Top-Down (h={scan.height:.2f}m)")
    ax_xy.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
