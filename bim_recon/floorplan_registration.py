"""Auto-register a 2D FloorPlan to a 3DGS scene.

The FloorPlan is in an arbitrary 2D coordinate system (e.g., hand-measured
room dimensions). This module maps it to the 3DGS horizontal plane using a
similarity transform (translation + rotation + uniform scale).

Registration strategy:
  1. Initial translation: AABB center of the floor footprint (robust to uneven
     floor coverage).
  2. Rotation: PCA of the floorplan wall samples vs PCA of the floor footprint
     (cleaner than noisy wall Gaussians), with a 90° multi-candidate search.
  3. Scale: default 1.0 (both floorplan and metric 3DGS are in meters).
  4. Refinement: a coarse grid search over translations, scored by how many
     floor Gaussians fall inside the floorplan polygon plus how many wall
     Gaussians lie near the floorplan wall segments.
"""
from __future__ import annotations

from math import atan2
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, Point

from bim_recon.floorplan import FloorPlan, WallSegment


def _sample_wall_points_2d(floorplan: FloorPlan) -> np.ndarray:
    """Sample points densely along wall segments for shape-aware PCA."""
    samples: list[np.ndarray] = []
    for wall in floorplan.walls:
        seg_len = wall.length()
        if seg_len < 1e-6:
            continue
        n = max(10, int(seg_len * 10))  # 10 samples per meter, at least 10
        t = np.linspace(0.0, 1.0, n)
        xs = wall.x1 + t * (wall.x2 - wall.x1)
        ys = wall.y1 + t * (wall.y2 - wall.y1)
        samples.append(np.column_stack([xs, ys]))
    if not samples:
        return np.array([[1.0, 0.0]], dtype=np.float64)
    return np.concatenate(samples, axis=0)


def _floorplan_center_2d(floorplan: FloorPlan) -> np.ndarray:
    """Mean of all wall segment midpoints."""
    if not floorplan.walls:
        return np.zeros(2, dtype=np.float64)
    mids = np.array(
        [[(w.x1 + w.x2) / 2.0, (w.y1 + w.y2) / 2.0] for w in floorplan.walls],
        dtype=np.float64,
    )
    return mids.mean(axis=0)


def _pca_axis(points: np.ndarray) -> np.ndarray:
    """Return the first principal component of a (N, 2) point set."""
    if points.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1].astype(np.float64)
    # Make deterministic: prefer pointing toward +x, then +y.
    if abs(axis[0]) > 1e-6:
        if axis[0] < 0:
            axis = -axis
    else:
        if axis[1] < 0:
            axis = -axis
    return axis


def _rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    """2D rotation matrix for a counter-clockwise rotation by angle_rad."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _point_to_segment_distance_2d(
    pts: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Return 2D point-to-segment distances for (N, 2) points and segment a->b."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    ap = pts - a
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(pts - closest, axis=1)


def _transform_floorplan(
    floorplan: FloorPlan,
    fp_center: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    angle_rad: float,
) -> list[WallSegment]:
    """Apply similarity transform to all floorplan wall segments."""
    rot = _rotation_matrix_2d(angle_rad)

    def transform_endpoint(x: float, y: float) -> tuple[float, float]:
        local = np.array([x - fp_center[0], y - fp_center[1]], dtype=np.float64)
        rotated = rot @ local
        scaled = rotated * scale
        final = scaled + centroid
        return float(final[0]), float(final[1])

    registered: list[WallSegment] = []
    for wall in floorplan.walls:
        x1, y1 = transform_endpoint(wall.x1, wall.y1)
        x2, y2 = transform_endpoint(wall.x2, wall.y2)
        registered.append(
            WallSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                thickness=wall.thickness,
                type=wall.type,
            )
        )
    return registered


def _score_registration(
    floorplan: FloorPlan,
    wall_means_2d: np.ndarray,
    fp_center: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    angle_rad: float,
    corridor_width: float,
    floor_means_2d: Optional[np.ndarray] = None,
) -> float:
    """Score a candidate registration.

    Combines:
      - Floor Gaussians inside the floorplan polygon (primary signal).
      - Wall Gaussians within corridor_width of floorplan wall segments.
    """
    registered = _transform_floorplan(
        floorplan, fp_center, centroid, scale, angle_rad,
    )

    # Wall-corridor score
    corridor_score = 0.0
    for wall in registered:
        a = np.array([wall.x1, wall.y1], dtype=np.float64)
        b = np.array([wall.x2, wall.y2], dtype=np.float64)
        dists = _point_to_segment_distance_2d(wall_means_2d, a, b)
        corridor_score += float(np.sum(dists <= corridor_width))

    if floor_means_2d is None or len(floor_means_2d) == 0:
        return corridor_score

    # Floor-in-polygon score: the floorplan walls define the room boundary,
    # so a good registration should have most floor Gaussians inside.
    poly_pts = [(w.x1, w.y1) for w in registered]
    # If the polygon is not closed, add the last point to close it
    if len(poly_pts) > 0 and (poly_pts[0] != poly_pts[-1]):
        poly_pts.append(poly_pts[0])
    try:
        polygon = Polygon(poly_pts)
        if not polygon.is_valid or polygon.is_empty:
            return corridor_score
        # Vectorized point-in-polygon using shapely prepared geometry is
        # possible but for ~300k points a loop is too slow. Use a vectorized
        # ray-casting fallback for the floor points.
        inside = _points_in_polygon(floor_means_2d, poly_pts)
        floor_score = float(inside.sum())
    except Exception:
        return corridor_score

    # Weight floor-in-polygon higher than corridor (it is the stronger signal).
    return 2.0 * floor_score + corridor_score


def _points_in_polygon(pts: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon for a (N, 2) point array."""
    n = len(polygon)
    if n < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if edge straddles the horizontal line at y
        cond = ((y1 > y) != (y2 > y))
        # Compute x intersection of edge with horizontal line at y
        x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (x < x_intersect)
    return inside


def register_floorplan(
    floorplan: FloorPlan,
    wall_means_2d: np.ndarray,
    floor_centroid_2d: Optional[np.ndarray] = None,
    floor_means_2d: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    corridor_width: float = 0.5,
    num_rotations: int = 4,
    translation_search_radius: float = 3.0,
    translation_grid_steps: int = 7,
) -> FloorPlan:
    """Map a 2D FloorPlan into the 3DGS horizontal plane.

    Args:
        floorplan: The hand-measured 2D floorplan (arbitrary origin/rotation).
            Wall coordinates are expected to be in meters.
        wall_means_2d: (N, 2) wall-classified Gaussian means on the horizontal
            plane. Used for PCA rotation and corridor scoring.
        floor_centroid_2d: Optional (2,) horizontal centroid used as the initial
            translation. If None, the AABB center of ``floor_means_2d`` is used
            if provided, otherwise the AABB center of ``wall_means_2d``.
        floor_means_2d: Optional (M, 2) floor-classified Gaussian means. If
            provided, registration is scored primarily by floor points inside
            the floorplan polygon, which is much more robust than wall-only
            corridor scoring.
        scale: Optional uniform scale override. Default 1.0 (meters).
        corridor_width: Width for wall corridor scoring.
        num_rotations: Number of 90° rotation candidates.
        translation_search_radius: Radius (meters) for the translation grid.
        translation_grid_steps: Grid steps per axis. ``steps=1`` disables search.

    Returns:
        A new FloorPlan registered to the 3DGS horizontal plane.
    """
    if len(floorplan.walls) == 0:
        return FloorPlan(walls=[], openings=floorplan.openings, meta=floorplan.meta)

    wall_means_2d = np.asarray(wall_means_2d, dtype=np.float64)
    if wall_means_2d.ndim != 2 or wall_means_2d.shape[1] != 2:
        raise ValueError("wall_means_2d must be (N, 2)")

    if floor_means_2d is not None:
        floor_means_2d = np.asarray(floor_means_2d, dtype=np.float64)

    # Initial translation: prefer floor AABB center, fall back to wall AABB center.
    if floor_centroid_2d is None:
        if floor_means_2d is not None and floor_means_2d.shape[0] > 0:
            centroid = (floor_means_2d.min(axis=0) + floor_means_2d.max(axis=0)) / 2.0
        elif wall_means_2d.shape[0] > 0:
            centroid = (wall_means_2d.min(axis=0) + wall_means_2d.max(axis=0)) / 2.0
        else:
            centroid = np.zeros(2, dtype=np.float64)
    else:
        centroid = np.asarray(floor_centroid_2d, dtype=np.float64).reshape(2)

    fp_center = _floorplan_center_2d(floorplan)

    # Rotation: use floor footprint PCA if available (cleaner), else wall PCA.
    fp_samples = _sample_wall_points_2d(floorplan)
    fp_axis = _pca_axis(fp_samples)
    if floor_means_2d is not None and floor_means_2d.shape[0] > 0:
        gs_axis = _pca_axis(floor_means_2d)
    else:
        gs_axis = _pca_axis(wall_means_2d)
    base_angle = float(atan2(gs_axis[1], gs_axis[0]) - atan2(fp_axis[1], fp_axis[0]))

    if scale is None:
        scale = 1.0

    # Translation search grid.
    if translation_grid_steps <= 1:
        tx_offsets = np.array([0.0])
        ty_offsets = np.array([0.0])
    else:
        tx_offsets = np.linspace(
            -translation_search_radius, translation_search_radius, translation_grid_steps
        )
        ty_offsets = np.linspace(
            -translation_search_radius, translation_search_radius, translation_grid_steps
        )

    rotation_candidates = [base_angle + k * (np.pi / 2.0) for k in range(num_rotations)]
    best_score = -1.0
    best_angle = base_angle
    best_centroid = centroid.copy()

    for angle in rotation_candidates:
        for tx in tx_offsets:
            for ty in ty_offsets:
                candidate_centroid = centroid + np.array([tx, ty], dtype=np.float64)
                score = _score_registration(
                    floorplan,
                    wall_means_2d,
                    fp_center,
                    candidate_centroid,
                    scale,
                    angle,
                    corridor_width,
                    floor_means_2d,
                )
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_centroid = candidate_centroid

    registered_walls = _transform_floorplan(
        floorplan, fp_center, best_centroid, scale, best_angle,
    )
    return FloorPlan(
        walls=registered_walls,
        openings=floorplan.openings,
        meta=floorplan.meta,
    )
