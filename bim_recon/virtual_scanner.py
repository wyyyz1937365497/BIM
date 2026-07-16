"""Virtual laser scanner: render depth from 3DGS to simulate LiDAR scans.

Inspired by gsplat's official spinning-lidar design (angular-space ray
generation + structured scan patterns), this module implements a spherical
scanner that works **without** the gsplat CUDA extension by using batched
pinhole-camera rendering via ``rasterization(render_mode="RGB+ED")``.

Key capabilities beyond the original 2D horizontal scanner:

  * **3D spherical scan** — scans at multiple elevation angles, producing a
    full 3D point cloud with semantic labels.  Floor and ceiling planes are
    detected by RANSAC on the 3D points.

  * **Per-wall panorama** — renders a seamless wide-format elevation image
    of each wall for Falcon/VLM element detection, avoiding the perspective
    distortion of single wide-FOV views.

  * **Backward-compatible horizontal scan** — ``scan()`` still returns a
    ``ScanResult`` for the existing wall-line and candidate extractors.

When semantic features (feat.pt) are loaded, every scan point carries its
dominant semantic class, making feat.pt an integral part of the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from bim_recon.gs_scene import GSScene, look_at_pose

# ---------------------------------------------------------------------------
# Palette (unchanged, kept for backward compat)
# ---------------------------------------------------------------------------

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


def label_palette(n: int) -> List[Tuple[float, float, float]]:
    """Return *n* distinct colours for an arbitrary open-vocabulary label set."""
    base = list(SEMANTIC_PALETTE)
    if n <= len(base):
        return base[:n]
    import colorsys
    extra = []
    for i in range(len(base), n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.90)
        extra.append((r, g, b))
    return base + extra


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """A 360° horizontal laser scan from a 3DGS scene (backward compat)."""

    angles_deg: np.ndarray       # (M,) azimuth angles [0, 360)
    distances: np.ndarray        # (M,) horizontal distance from center (meters)
    points_2d: np.ndarray        # (M, 2) world XY coordinates
    height: float                # scan height (world up-axis coordinate)
    center_2d: np.ndarray        # (2,) scan center in world XY
    up_axis: int
    semantic_labels: Optional[np.ndarray] = None  # (M,) int class index
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


@dataclass
class Scan3DResult:
    """Full 3D point cloud from spherical scanning.

    All arrays are aligned (same length N).  ``heights`` is relative to
    floor_z (0 = floor level).
    """

    points_3d: np.ndarray        # (N, 3) world XYZ
    semantic_labels: Optional[np.ndarray]  # (N,) int class index
    azimuth_deg: np.ndarray      # (N,) horizontal angle from center [0, 360)
    elevation_deg: np.ndarray    # (N,) vertical angle from horizontal
    distance: np.ndarray         # (N,) 3D distance from scan center
    heights: np.ndarray          # (N,) height above floor_z
    floor_z: float
    ceiling_z: float
    center: np.ndarray           # (3,) scan center world XYZ
    up_axis: int

    def slice_at_height(self, height: float, tol: float = 0.15) -> "ScanResult":
        """Extract a horizontal ScanResult at a given height (±tol)."""
        mask = np.abs(self.heights - height) < tol
        h_axes = [i for i in range(3) if i != self.up_axis]
        pts = self.points_3d[mask]
        cx, cy = self.center[h_axes[0]], self.center[h_axes[1]]
        dx = pts[:, h_axes[0]] - cx
        dy = pts[:, h_axes[1]] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dy, dx)) % 360.0
        labels = self.semantic_labels[mask] if self.semantic_labels is not None else None
        return ScanResult(
            angles_deg=angle,
            distances=dist,
            points_2d=pts[:, h_axes],
            height=float(self.floor_z + height),
            center_2d=np.array([cx, cy], dtype=np.float64),
            up_axis=self.up_axis,
            semantic_labels=labels,
        )

    def filter_by_label(self, label_idx: int) -> np.ndarray:
        """Return 3D points whose semantic label == *label_idx*."""
        if self.semantic_labels is None:
            return np.empty((0, 3))
        return self.points_3d[self.semantic_labels == label_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_points": len(self.points_3d),
            "floor_z": self.floor_z,
            "ceiling_z": self.ceiling_z,
            "center": self.center.tolist(),
            "up_axis": self.up_axis,
        }


# ---------------------------------------------------------------------------
# Panoramic view for per-wall element detection
# ---------------------------------------------------------------------------

@dataclass
class WallPanorama:
    """A panoramic elevation image of a wall + camera params for back-mapping."""

    image: np.ndarray              # (H, W, 3) uint8 RGB
    depth: np.ndarray              # (H, W) float32 metric depth
    alpha: np.ndarray              # (H, W) float32
    wall_idx: int
    # Camera params for pixel→world back-projection
    eye: np.ndarray                # (3,) camera position
    forward: np.ndarray            # (3,) camera forward direction
    right: np.ndarray              # (3,) camera right direction
    up_dir: np.ndarray             # (3,) camera up direction
    fx: float                      # focal length in pixels
    cx_pix: float                  # principal point x
    cy_pix: float                  # principal point y
    img_width: int
    img_height: int


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class VirtualScanner:
    """Render virtual laser scans from a 3DGS scene.

    Supports both 2D horizontal scans (backward-compatible ``scan()``)
    and full 3D spherical scans (``scan_3d()``).  Per-wall panoramic
    rendering (``render_wall_panorama()``) produces distortion-free
    elevation images for element detection.
    """

    def __init__(
        self,
        scene: "GSScene",
        up_axis: int = 2,
        labels: Optional[List[str]] = None,
    ):
        self.scene = scene
        self.up_axis = up_axis
        self.h_axes = [i for i in range(3) if i != up_axis]
        self._has_semantics = (
            scene.semantic_querier is not None and scene._has_feat
        )
        self.label_names: List[str] = []
        self._semantic_colors: Optional[torch.Tensor] = None
        self._num_classes = 0
        if self._has_semantics:
            querier = scene.semantic_querier
            if querier is not None:
                if labels is not None:
                    dominant = querier.get_dominant_labels(labels)
                    self.label_names = list(labels)
                    self._num_classes = len(labels)
                else:
                    dominant = querier.get_dominant_labels()
                    self.label_names = list(querier.registered_labels)
                    self._num_classes = querier.num_classes
                N = scene.num_gaussians
                enc = torch.zeros((N, 3), dtype=torch.float32, device=scene.device)
                if self._num_classes > 1:
                    enc[:, 0] = torch.from_numpy(dominant.astype(np.float32)).to(scene.device) / (self._num_classes - 1)
                self._semantic_colors = enc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_camera_at_angle(
        self,
        cx: float, cy: float, height: float,
        azimuth_deg: float, elevation_deg: float = 0.0,
    ) -> Tuple[list, list, list]:
        """Build eye/target/up for a camera at (cx, cy) looking at azimuth/elevation."""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)

        eye = [0.0, 0.0, 0.0]
        eye[self.h_axes[0]] = cx
        eye[self.h_axes[1]] = cy
        eye[self.up_axis] = height

        # Direction vector: horizontal component scaled by cos(el), vertical by sin(el)
        cos_el = math.cos(el)
        sin_el = math.sin(el)
        target = [0.0, 0.0, 0.0]
        target[self.h_axes[0]] = cx + math.cos(az) * cos_el
        target[self.h_axes[1]] = cy + math.sin(az) * cos_el
        target[self.up_axis] = height + sin_el

        up = [0.0, 0.0, 0.0]
        up[self.up_axis] = 1.0

        return eye, target, up

    def _render_with_semantics(
        self, pose, width: int, height: int, fov: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Render geometry + optional semantic pass.

        Returns (depth, alpha, colors, sem_row_func) where sem_row_func
        is a callable that maps row index → semantic labels, or None.
        """
        result = self.scene.render(pose, width=width, height=height, fov_degrees=fov)
        depth = result.depth
        alpha = result.alpha
        colors = result.colors

        sem = None
        if self._semantic_colors is not None and self._num_classes > 1:
            orig_colors = self.scene.colors
            try:
                self.scene.colors = self._semantic_colors
                sem_result = self.scene.render(pose, width=width, height=height, fov_degrees=fov)
            finally:
                self.scene.colors = orig_colors
            sem_r = sem_result.colors[:, :, 0]
            sem_alpha = sem_result.alpha
            sem = np.round(sem_r * (self._num_classes - 1)).astype(np.int32)
            sem[sem_alpha < 0.1] = -1

        return depth, alpha, colors, sem

    def _unproject_row(
        self,
        depth_row: np.ndarray,
        sem_row: Optional[np.ndarray],
        azimuth_deg: float,
        eye: list,
        viewmat: np.ndarray,
        fx: float,
        cx_pix: float,
        middle_v: int,
        render_h: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Unproject a depth row to world coordinates + polar angles."""
        us = np.arange(len(depth_row), dtype=np.float64)
        d = depth_row.astype(np.float64)
        valid = (d >= 0.05) & (d <= 80.0)

        # Camera-space coordinates (middle row → y_cam ≈ 0)
        x_cam = (us - cx_pix) / fx * d
        y_cam = (middle_v - render_h / 2.0) / fx * d
        z_cam = d
        P_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # World coordinates
        R_c2w = viewmat[:3, :3].T.astype(np.float64)
        eye_np = np.array(eye, dtype=np.float64)
        P_world = P_cam @ R_c2w.T + eye_np

        h0, h1 = self.h_axes
        cx, cy = float(eye[h0]), float(eye[h1])
        px = P_world[:, h0]
        py = P_world[:, h1]
        dx = px - cx
        dy = py - cy
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dy, dx)) % 360.0

        idx = valid
        pts = P_world[idx]
        if sem_row is not None:
            labels = sem_row[idx]
        else:
            labels = None

        return pts, angle[idx], dist[idx], labels

    # ------------------------------------------------------------------
    # 2D horizontal scan (backward-compatible)
    # ------------------------------------------------------------------

    def scan(
        self,
        center_2d: Tuple[float, float],
        height: float,
        num_views: int = 8,
        fov: float = 60.0,
        width: int = 1024,
        scan_render_height: int = 16,
    ) -> ScanResult:
        """Render a 360° horizontal scan at *height* from *center_2d*.

        Backward-compatible with the original implementation.
        """
        cx, cy = float(center_2d[0]), float(center_2d[1])

        all_angles: List[float] = []
        all_distances: List[float] = []
        all_points: List[Tuple[float, float]] = []
        all_labels: Optional[List[int]] = [] if self._has_semantics else None
        view_azimuths: List[float] = []

        fx = 0.5 * width / math.tan(0.5 * math.radians(fov))
        cx_pix = width / 2.0
        render_h = scan_render_height
        middle_v = render_h // 2

        for i in range(num_views):
            azimuth_deg = i * (360.0 / num_views)
            view_azimuths.append(azimuth_deg)

            eye, target, up = self._make_camera_at_angle(cx, cy, height, azimuth_deg)
            pose = look_at_pose(tuple(eye), tuple(target), tuple(up))

            depth, alpha, colors, sem = self._render_with_semantics(
                pose, width, render_h, fov)

            depth_row = depth[middle_v]
            sem_row = sem[middle_v] if sem is not None else None

            viewmat = pose.to_viewmat()
            pts, angles, dists, labels = self._unproject_row(
                depth_row, sem_row, azimuth_deg, eye, viewmat,
                fx, cx_pix, middle_v, render_h)

            h0, h1 = self.h_axes
            all_angles.extend(angles.tolist())
            all_distances.extend(dists.tolist())
            all_points.extend(zip(pts[:, h0].tolist(), pts[:, h1].tolist()))
            if all_labels is not None and labels is not None:
                all_labels.extend(labels.tolist())

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

    # ------------------------------------------------------------------
    # 3D spherical scan
    # ------------------------------------------------------------------

    def scan_3d(
        self,
        center_2d: Tuple[float, float],
        floor_z: float,
        ceiling_z: float,
        n_azimuth_views: int = 12,
        n_elevation_bands: int = 5,
        width: int = 512,
        fov: float = 45.0,
        scan_render_height: int = 32,
    ) -> Scan3DResult:
        """Full 3D spherical scan from the room center.

        Renders at *n_elevation_bands* tilt angles (from looking-down at
        the floor through horizontal to looking-up at the ceiling), each
        with *n_azimuth_views* overlapping cameras covering 360°.

        All depth pixels are unprojected to 3D world coordinates, producing
        a unified point cloud with semantic labels, azimuth/elevation angles,
        and distances.  The point cloud enables:

          * Direct floor/ceiling plane fitting (via ``detect_floor_ceiling()``)
          * Multi-height wall-line extraction (via ``slice_at_height()``)
          * Element candidate detection from semantic-tagged 3D points

        Args:
            center_2d: (x, y) scan center in world horizontal plane.
            floor_z: Floor level (up-axis coordinate).
            ceiling_z: Ceiling level.
            n_azimuth_views: Cameras per elevation band (≥ 8 for full 360°).
            n_elevation_bands: Number of tilt angles (≥ 3).
            width: Rendered image width per view.
            fov: Horizontal FOV per view in degrees (45° recommended).
            scan_render_height: Image height (rows) per view.

        Returns:
            Scan3DResult with full 3D point cloud.
        """
        cx, cy = float(center_2d[0]), float(center_2d[1])
        room_height = ceiling_z - floor_z
        mid_z = (floor_z + ceiling_z) / 2.0

        # Elevation angles: from -max_el (looking down at floor) to +max_el (looking up)
        # max_el is chosen so the floor edge at ~3m distance is visible
        max_el = min(35.0, math.degrees(math.atan(room_height / 2.0 / 2.0)))
        elevations = np.linspace(-max_el, max_el, n_elevation_bands)

        # Azimuth step must provide overlap: with fov=45°, need ≥ 9 views for 360°
        az_step = 360.0 / n_azimuth_views

        fx = 0.5 * width / math.tan(0.5 * math.radians(fov))
        cx_pix = width / 2.0
        cy_pix = scan_render_height / 2.0

        all_pts: List[np.ndarray] = []
        all_az: List[np.ndarray] = []
        all_el: List[np.ndarray] = []
        all_dist: List[np.ndarray] = []
        all_labels: Optional[List[np.ndarray]] = [] if self._has_semantics else None

        for el_deg in elevations:
            # Camera height adjusts with elevation: looking down → higher,
            # looking up → lower, to maximise wall/floor coverage.
            if el_deg < -5:
                cam_height = mid_z + room_height * 0.2
            elif el_deg > 5:
                cam_height = mid_z - room_height * 0.2
            else:
                cam_height = mid_z

            for i in range(n_azimuth_views):
                az_deg = i * az_step
                eye, target, up = self._make_camera_at_angle(
                    cx, cy, cam_height, az_deg, el_deg)
                pose = look_at_pose(tuple(eye), tuple(target), tuple(up))

                depth, alpha, colors, sem = self._render_with_semantics(
                    pose, width, scan_render_height, fov)

                viewmat = pose.to_viewmat()
                R_c2w = viewmat[:3, :3].T.astype(np.float64)
                eye_np = np.array(eye, dtype=np.float64)

                # Unproject ALL rows (not just middle) for 3D cloud
                for v in range(scan_render_height):
                    d_row = depth[v].astype(np.float64)
                    valid = (d_row >= 0.05) & (d_row <= 80.0)
                    if valid.sum() < 2:
                        continue
                    us = np.arange(width, dtype=np.float64)
                    x_cam = (us - cx_pix) / fx * d_row
                    y_cam = (v - cy_pix) / fx * d_row
                    z_cam = d_row
                    P_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
                    P_world = P_cam @ R_c2w.T + eye_np

                    # Compute spherical angles from scan center
                    h0, h1 = self.h_axes
                    ua = self.up_axis
                    dx3 = P_world[:, h0] - cx
                    dy3 = P_world[:, h1] - cy
                    dz3 = P_world[:, ua] - cam_height
                    horiz_dist = np.sqrt(dx3**2 + dy3**2)
                    az = np.degrees(np.arctan2(dy3, dx3)) % 360.0
                    el_out = np.degrees(np.arctan2(dz3, np.maximum(horiz_dist, 1e-6)))
                    dist3 = np.sqrt(dx3**2 + dy3**2 + dz3**2)

                    idx = valid
                    all_pts.append(P_world[idx])
                    all_az.append(az[idx])
                    all_el.append(el_out[idx])
                    all_dist.append(dist3[idx])

                    if all_labels is not None and sem is not None:
                        all_labels.append(sem[v][idx])

        points_3d = np.concatenate(all_pts) if all_pts else np.empty((0, 3))
        azimuth = np.concatenate(all_az) if all_az else np.empty(0)
        elevation = np.concatenate(all_el) if all_el else np.empty(0)
        distance = np.concatenate(all_dist) if all_dist else np.empty(0)

        if all_labels is not None and all_labels:
            labels_arr = np.concatenate(all_labels)
        else:
            labels_arr = None

        heights = points_3d[:, self.up_axis] - floor_z if len(points_3d) > 0 else np.empty(0)

        center_3d = np.zeros(3)
        center_3d[self.h_axes[0]] = cx
        center_3d[self.h_axes[1]] = cy
        center_3d[self.up_axis] = mid_z

        return Scan3DResult(
            points_3d=points_3d,
            semantic_labels=labels_arr,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            distance=distance,
            heights=heights,
            floor_z=floor_z,
            ceiling_z=ceiling_z,
            center=center_3d,
            up_axis=self.up_axis,
        )

    # ------------------------------------------------------------------
    # Floor/ceiling detection from 3D point cloud
    # ------------------------------------------------------------------

    @staticmethod
    def detect_floor_ceiling(
        scan_3d: Scan3DResult,
        labels: Optional[List[str]] = None,
        floor_label: str = "floor",
        ceiling_label: str = "ceiling",
    ) -> Tuple[float, float]:
        """Detect floor and ceiling Z from 3D point cloud.

        Uses semantic labels (if available) to pre-filter, then RANSAC
        plane fitting for robustness.  Falls back to percentile of all
        points along the up-axis if semantics are unavailable.

        Returns:
            (floor_z, ceiling_z) in world coordinates.
        """
        up = scan_3d.up_axis
        pts = scan_3d.points_3d

        if len(pts) < 10:
            return scan_3d.floor_z, scan_3d.ceiling_z

        zs = pts[:, up]

        # Try semantic pre-filtering
        floor_pts = None
        ceiling_pts = None
        if scan_3d.semantic_labels is not None and labels is not None:
            try:
                fi = labels.index(floor_label)
                ci = labels.index(ceiling_label)
                floor_pts = pts[scan_3d.semantic_labels == fi]
                ceiling_pts = pts[scan_3d.semantic_labels == ci]
            except (ValueError, IndexError):
                pass

        def _fit_plane_z(points: np.ndarray) -> Optional[float]:
            """RANSAC-fit a horizontal plane, return its Z coordinate."""
            if len(points) < 5:
                return None
            from sklearn.linear_model import RANSACRegressor
            XY = points[:, [i for i in range(3) if i != up]].astype(np.float64)
            Z = points[:, up].astype(np.float64)
            try:
                r = RANSACRegressor(
                    estimator=None, min_samples=5,
                    residual_threshold=0.08, max_trials=80,
                )
                r.fit(XY, Z)
                # Plane Z is approximately constant → use inlier median
                inlier_mask = r.inlier_mask_
                if inlier_mask.sum() < 3:
                    return None
                return float(np.median(Z[inlier_mask]))
            except Exception:
                return float(np.median(Z))

        floor_z = _fit_plane_z(floor_pts) if floor_pts is not None and len(floor_pts) > 5 else None
        ceiling_z = _fit_plane_z(ceiling_pts) if ceiling_pts is not None and len(ceiling_pts) > 5 else None

        # Fallback: use elevation angle to separate floor/ceiling
        if floor_z is None:
            horiz_mask = scan_3d.elevation_deg < -5
            if horiz_mask.sum() > 10:
                floor_z = float(np.percentile(zs[horiz_mask], 10))
            else:
                floor_z = float(np.percentile(zs, 1))
        if ceiling_z is None:
            horiz_mask = scan_3d.elevation_deg > 5
            if horiz_mask.sum() > 10:
                ceiling_z = float(np.percentile(zs[horiz_mask], 90))
            else:
                ceiling_z = float(np.percentile(zs, 99))

        # Sanity: room height must be reasonable
        if ceiling_z - floor_z < 1.0:
            floor_z = float(np.percentile(zs, 1))
            ceiling_z = float(np.percentile(zs, 99))

        return floor_z, ceiling_z

    # ------------------------------------------------------------------
    # Per-wall panoramic rendering
    # ------------------------------------------------------------------

    def render_wall_panorama(
        self,
        wall: Dict[str, Any],
        center_2d: Tuple[float, float],
        floor_z: float,
        ceiling_z: float,
        camera_dist: float = 2.0,
        img_width: int = 1024,
        fov: float = 45.0,
    ) -> Optional[WallPanorama]:
        """Render a panoramic elevation view of a wall.

        Places a camera at *camera_dist* metres in front of the wall
        midpoint, looking perpendicular to the wall surface.  The FOV
        is chosen to cover the full wall width + margin.

        Unlike the old single wide-FOV approach, this uses a moderate
        FOV (≤ 60°) and positions the camera to minimise distortion.
        For walls wider than the FOV allows, multiple overlapping views
        are rendered and the central strip is used.

        Args:
            wall: Dict with x1, y1, x2, y2, length.
            center_2d: Room center (for inward normal computation).
            floor_z, ceiling_z: Room bounds.
            camera_dist: Distance from wall surface to camera (metres).
            img_width: Output image width.
            fov: Horizontal FOV (45° recommended for low distortion).

        Returns:
            WallPanorama with RGB image, depth, and back-projection params.
        """
        ws = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
        we = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
        wlen = float(np.linalg.norm(we - ws))
        wmid = (ws + we) / 2.0
        wall_dir = (we - ws) / max(wlen, 1e-9)

        # Inward normal: from wall midpoint toward room center
        center_np = np.array(center_2d, dtype=np.float64)
        normal = center_np - wmid
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-6:
            normal = np.array([-wall_dir[1], wall_dir[0]])
        else:
            normal = normal / n_norm

        room_height = ceiling_z - floor_z
        mid_z = (floor_z + ceiling_z) / 2.0
        h0, h1 = self.h_axes

        # FOV: cover wall width + 0.5m margin, capped at 75°
        fov_w = min(math.degrees(2 * math.atan((wlen / 2 + 0.5) / max(camera_dist, 0.5))), 75.0)
        # FOV: cover room height + 0.3m margin
        fov_h = min(math.degrees(2 * math.atan((room_height / 2 + 0.3) / max(camera_dist, 0.5))), 75.0)
        fov_use = max(fov_w, fov_h)

        # Camera position: in front of wall midpoint
        eye_xy = wmid + normal * camera_dist
        eye = [0.0, 0.0, 0.0]
        eye[h0] = float(eye_xy[0])
        eye[h1] = float(eye_xy[1])
        eye[self.up_axis] = mid_z

        # Target: wall midpoint at mid-height
        tgt = [0.0, 0.0, 0.0]
        tgt[h0] = float(wmid[0])
        tgt[h1] = float(wmid[1])
        tgt[self.up_axis] = mid_z

        up = [0.0, 0.0, 0.0]
        up[self.up_axis] = 1.0

        pose = look_at_pose(tuple(eye), tuple(tgt), tuple(up))

        # Render
        from bim_recon.gs_scene import GSScene
        result, reason, _ = GSScene.render_validated(
            self.scene, pose, img_width, img_width, fov_use)

        if result is None:
            return None

        fx = 0.5 * img_width / math.tan(0.5 * math.radians(fov_use))

        # Camera basis vectors for back-projection
        viewmat = pose.to_viewmat()
        R_c2w = viewmat[:3, :3].T.astype(np.float64)
        forward = R_c2w[:, 2]
        right = R_c2w[:, 0]
        up_dir = R_c2w[:, 1]

        return WallPanorama(
            image=(result.colors * 255).clip(0, 255).astype(np.uint8),
            depth=result.depth,
            alpha=result.alpha,
            wall_idx=wall.get("wall_idx", -1),
            eye=np.array(eye, dtype=np.float64),
            forward=forward,
            right=right,
            up_dir=up_dir,
            fx=fx,
            cx_pix=img_width / 2.0,
            cy_pix=img_width / 2.0,
            img_width=img_width,
            img_height=img_width,
        )

    @staticmethod
    def panorama_pixel_to_world(
        pan: WallPanorama, u: int, v: int, depth: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Back-project a panorama pixel to world XYZ.

        Uses the camera intrinsics + depth at (u, v) (or the provided
        *depth* override) to compute the 3D world coordinate.
        """
        d = depth if depth is not None else float(pan.depth[v, u])
        if d < 0.05:
            return None
        x_cam = (u - pan.cx_pix) / pan.fx * d
        y_cam = (v - pan.cy_pix) / pan.fx * d
        P_cam = np.array([x_cam, y_cam, d])
        # World = R_c2w @ P_cam + eye
        R_c2w = np.stack([pan.right, pan.up_dir, pan.forward], axis=1)
        return R_c2w @ P_cam + pan.eye


# ---------------------------------------------------------------------------
# Plot helpers (unchanged)
# ---------------------------------------------------------------------------

def save_scan_plot(
    scan: ScanResult,
    output_path: str,
    label_names: Optional[List[str]] = None,
    palette: Optional[List[Tuple[float, float, float]]] = None,
    title: str = "Radar Scan",
    wall_lines: Optional[List[dict]] = None,
    elements: Optional[List[dict]] = None,
) -> str:
    """Save a radar-style polar plot of a scan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_labels = len(label_names) if label_names else 0
    if palette is None:
        palette = label_palette(max(n_labels, 9)) if n_labels > 0 else SEMANTIC_PALETTE

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    angles = np.radians(scan.angles_deg)
    dists = scan.distances

    if scan.semantic_labels is not None and n_labels > 0:
        colors = np.array([(0.5, 0.5, 0.5, 0.8)] * len(angles))
        for i in range(min(n_labels, len(palette))):
            r, g, b = palette[i]
            mask = scan.semantic_labels == i
            colors[mask] = (r, g, b, 0.9)
        ax.scatter(angles, dists, c=colors, s=3, zorder=2)
    else:
        ax.scatter(angles, dists, c="steelblue", s=3, alpha=0.6, zorder=2)

    # Draw wall lines
    if wall_lines:
        cx, cy = scan.center_2d
        for wl in wall_lines:
            x1, y1 = wl["x1"] - cx, wl["y1"] - cy
            x2, y2 = wl["x2"] - cx, wl["y2"] - cy
            a = [math.atan2(y1, x1), math.atan2(y2, x2)]
            r = [math.sqrt(x1**2 + y1**2), math.sqrt(x2**2 + y2**2)]
            ax.plot(a, r, "r-", linewidth=2, zorder=3)

    # Draw element markers
    if elements:
        for el in elements:
            theta = math.radians(el.get("theta_center", 0))
            r = el.get("r_mean", 0)
            ax.plot(theta, r, "g*" if el.get("confirmed") else "y*",
                    markersize=12, zorder=4)

    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)

    # Legend for semantic classes
    if scan.semantic_labels is not None and n_labels > 0:
        from matplotlib.patches import Patch
        handles = []
        for i in range(min(n_labels, len(palette))):
            r, g, b = palette[i]
            name = label_names[i] if i < len(label_names) else f"class_{i}"
            handles.append(Patch(facecolor=(r, g, b), label=name))
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
