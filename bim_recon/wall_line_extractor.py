"""Wall line extraction from multi-height semantic laser scans.

Scans a 3DGS scene at multiple heights (floor to ceiling), collects all
wall-tagged points across heights, and extracts wall line segments via
split-and-merge — the classic 2D SLAM line extraction algorithm.

Key insight: furniture occluding walls is typically shorter than the wall.
Scanning at multiple heights means higher scans see wall surfaces above
furniture, recovering wall geometry that low scans miss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bim_recon.virtual_scanner import ScanResult, VirtualScanner


@dataclass
class WallLine:
    """A wall line segment extracted from scan data."""

    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    num_points: int  # number of scan points that formed this segment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 4),
            "y1": round(self.y1, 4),
            "x2": round(self.x2, 4),
            "y2": round(self.y2, 4),
            "length": round(self.length, 4),
            "num_points": self.num_points,
        }


def multi_height_scan(
    scanner: VirtualScanner,
    center_2d: Tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    num_heights: int = 8,
    num_views: int = 8,
    fov: float = 60.0,
    width: int = 1024,
) -> List[ScanResult]:
    """Scan at multiple heights from floor to ceiling.

    Args:
        scanner: The VirtualScanner instance.
        center_2d: (x, y) scan center.
        floor_z: Floor level (up-axis coordinate).
        ceiling_z: Ceiling level.
        num_heights: Number of evenly-spaced scan heights.
        num_views, fov, width: Passed to VirtualScanner.scan().

    Returns:
        List of ScanResult, one per height.
    """
    heights = np.linspace(floor_z + 0.15, ceiling_z - 0.15, num_heights)
    scans: List[ScanResult] = []
    for h in heights:
        scan = scanner.scan(
            center_2d=center_2d,
            height=float(h),
            num_views=num_views,
            fov=fov,
            width=width,
        )
        scans.append(scan)
    return scans


def extract_wall_points(
    scans: List[ScanResult],
    wall_class_idx: int = 0,
    exclude_classes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect wall-tagged points from multi-height scans.

    By default, only ``wall_class_idx`` (class 0) is used. When
    ``exclude_classes`` is given, all points whose label is NOT in the
    exclusion list are kept — this captures misclassified wall surfaces
    (SceneSplat often labels wall areas as "door" or "window").

    Args:
        scans: List of ScanResult with semantic_labels.
        wall_class_idx: The class index for "wall" (used when exclude_classes is None).
        exclude_classes: If given, keep all points except these classes.
            Typical: [1, 2, 8] = floor, ceiling, furniture.

    Returns:
        (points_2d, heights) — (N, 2) XY coords and (N,) scan heights.
    """
    all_pts: List[np.ndarray] = []
    all_h: List[float] = []
    exclude_set = set(exclude_classes) if exclude_classes is not None else None
    for scan in scans:
        if scan.semantic_labels is None:
            continue
        labels = scan.semantic_labels
        if exclude_set is not None:
            # Vectorized: build boolean mask without Python-level iteration.
            mask = np.ones(len(labels), dtype=bool)
            for cls in exclude_set:
                mask &= (labels != cls)
        else:
            mask = labels == wall_class_idx
        if mask.sum() == 0:
            continue
        all_pts.append(scan.points_2d[mask])
        all_h.extend([scan.height] * int(mask.sum()))
    if not all_pts:
        return np.empty((0, 2)), np.empty(0)
    return np.concatenate(all_pts), np.array(all_h)


def _polar_sort(
    points: np.ndarray,
    center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort 2D points by azimuth angle around center.

    Returns (angles_rad, distances, sorted_points).
    """
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    angles = np.arctan2(dy, dx)
    dists = np.sqrt(dx ** 2 + dy ** 2)
    order = np.argsort(angles)
    return angles[order], dists[order], points[order]


def _bin_polar(
    angles: np.ndarray,
    dists: np.ndarray,
    bin_deg: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin polar data into uniform angle bins, taking median distance per bin.

    Vectorized via np.digitize for speed.

    Returns (bin_angles_rad, bin_distances) for bins with data.
    """
    if len(angles) == 0:
        return np.empty(0), np.empty(0)
    bin_rad = math.radians(bin_deg)
    bin_edges = np.arange(angles.min(), angles.max() + bin_rad, bin_rad)
    if len(bin_edges) < 2:
        return np.empty(0), np.empty(0)
    # Assign each point to a bin.
    bin_idx = np.digitize(angles, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    n_bins = len(bin_edges) - 1
    # Vectorized median per bin.
    bin_centers = []
    bin_medians = []
    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        if count > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_medians.append(float(np.median(dists[mask])))
    return np.array(bin_centers), np.array(bin_medians)


def _polar_to_cartesian(
    angles: np.ndarray,
    dists: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Convert polar (angle, distance) to Cartesian (x, y) around center."""
    x = center[0] + dists * np.cos(angles)
    y = center[1] + dists * np.sin(angles)
    return np.column_stack([x, y])


def _point_line_distance(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Perpendicular distance from point to line segment a-b."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(point - a))
    ap = point - a
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _split_and_merge(
    points: np.ndarray,
    dist_threshold: float = 0.08,
    min_points: int = 5,
    merge_angle_deg: float = 5.0,
) -> List[Tuple[int, int]]:
    """Split-and-merge line extraction on ordered 2D points.

    Args:
        points: (N, 2) array, ordered (e.g., by angle).
        dist_threshold: Max perpendicular distance for splitting.
        min_points: Minimum points per segment.
        merge_angle_deg: Max angle between adjacent segments to merge.

    Returns:
        List of (start_idx, end_idx) pairs into the points array.
    """
    n = len(points)
    if n < min_points:
        return [(0, n - 1)] if n >= 2 else []

    def split_recursive(start: int, end: int) -> List[Tuple[int, int]]:
        if end - start < min_points:
            return [(start, end)] if end > start else []
        a = points[start]
        b = points[end]
        max_dist = 0.0
        max_idx = -1
        for k in range(start + 1, end):
            d = _point_line_distance(points[k], a, b)
            if d > max_dist:
                max_dist = d
                max_idx = k
        if max_dist > dist_threshold and max_idx > 0:
            left = split_recursive(start, max_idx)
            right = split_recursive(max_idx, end)
            return left + right
        return [(start, end)]

    segments = split_recursive(0, n - 1)

    # Merge adjacent collinear segments.
    merged: List[Tuple[int, int]] = []
    merge_rad = math.radians(merge_angle_deg)
    for seg in segments:
        if merged:
            prev = merged[-1]
            # Check if prev and seg are collinear enough to merge.
            if prev[1] == seg[0]:
                a = points[prev[0]]
                b = points[prev[1]]
                c = points[seg[1]]
                v1 = b - a
                v2 = c - b
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    angle = abs(np.arccos(np.clip(
                        np.dot(v1, v2) / (n1 * n2), -1, 1
                    )))
                    if angle < merge_rad:
                        merged[-1] = (prev[0], seg[1])
                        continue
        merged.append(seg)

    # Filter segments with too few points.
    return [(s, e) for s, e in merged if e - s >= min_points - 1]


def _merge_wall_lines(
    lines: List[WallLine],
    angle_thresh_deg: float = 10.0,
    gap_thresh: float = 1.0,
) -> List[WallLine]:
    """Merge collinear or near-collinear WallLine segments.

    Two segments are merged if:
      - Direction angle difference < angle_thresh_deg.
      - Endpoint gap < gap_thresh meters.
      - They are roughly coplanar (perpendicular offset < 0.15m).

    The merged segment spans the full extent of both.
    """
    if len(lines) <= 1:
        return lines

    def seg_angle(wl: WallLine) -> float:
        return math.degrees(math.atan2(wl.y2 - wl.y1, wl.x2 - wl.x1))

    def seg_dir(wl: WallLine) -> np.ndarray:
        d = np.array([wl.x2 - wl.x1, wl.y2 - wl.y1])
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else d

    def seg_normal(wl: WallLine) -> np.ndarray:
        d = seg_dir(wl)
        return np.array([-d[1], d[0]])

    def perp_offset(a: WallLine, b: WallLine) -> float:
        n = seg_normal(a)
        mid_b = np.array([(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2])
        mid_a = np.array([(a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2])
        return abs(np.dot(n, mid_b - mid_a))

    def endpoint_gap(a: WallLine, b: WallLine) -> float:
        gaps = [
            math.hypot(a.x2 - b.x1, a.y2 - b.y1),
            math.hypot(a.x1 - b.x2, a.y1 - b.y2),
            math.hypot(a.x2 - b.x2, a.y2 - b.y2),
            math.hypot(a.x1 - b.x1, a.y1 - b.y1),
        ]
        return min(gaps)

    def merge_two(a: WallLine, b: WallLine) -> WallLine:
        pts = np.array([
            [a.x1, a.y1], [a.x2, a.y2],
            [b.x1, b.y1], [b.x2, b.y2],
        ])
        d = seg_dir(a)
        projections = pts @ d
        i_min = np.argmin(projections)
        i_max = np.argmax(projections)
        p0 = pts[i_min]
        p1 = pts[i_max]
        length = float(np.linalg.norm(p1 - p0))
        return WallLine(
            x1=float(p0[0]), y1=float(p0[1]),
            x2=float(p1[0]), y2=float(p1[1]),
            length=length,
            num_points=a.num_points + b.num_points,
        )

    merged = list(lines)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                ang_diff = abs(seg_angle(a) - seg_angle(b))
                ang_diff = min(ang_diff, 180 - ang_diff)
                if ang_diff > angle_thresh_deg:
                    continue
                if perp_offset(a, b) > 0.15:
                    continue
                if endpoint_gap(a, b) > gap_thresh:
                    continue
                # Merge
                merged[i] = merge_two(a, b)
                merged.pop(j)
                changed = True
                break
            if changed:
                break
    return merged


def _snap_endpoints_to_loop(
    lines: List[WallLine],
    snap_threshold: float = 0.8,
) -> List[WallLine]:
    """Snap nearby wall endpoints together to form a closed loop.

    For each endpoint, find the nearest endpoint from a different wall.
    If within ``snap_threshold``, replace both with their midpoint.
    Iterates until no more snaps occur.
    """
    if len(lines) <= 1:
        return lines

    def endpoints(wl: WallLine) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([wl.x1, wl.y1]), np.array([wl.x2, wl.y2])

    changed = True
    while changed:
        changed = False
        # Collect all endpoints with wall index and which end (0=p0, 1=p1).
        eps: List[Tuple[int, int, np.ndarray]] = []
        for i, wl in enumerate(lines):
            p0, p1 = endpoints(wl)
            eps.append((i, 0, p0))
            eps.append((i, 1, p1))

        best_dist = snap_threshold
        best_pair = None
        for a in range(len(eps)):
            for b in range(a + 1, len(eps)):
                wi_a, ei_a, pa = eps[a]
                wi_b, ei_b, pb = eps[b]
                if wi_a == wi_b:
                    continue
                d = float(np.linalg.norm(pa - pb))
                if d < best_dist and d > 1e-3:
                    best_dist = d
                    best_pair = (wi_a, ei_a, pa, wi_b, ei_b, pb)

        if best_pair is not None:
            wi_a, ei_a, pa, wi_b, ei_b, pb = best_pair
            midpoint = (pa + pb) / 2.0
            # Update both walls
            for wi, ei, _ in [(wi_a, ei_a, pa), (wi_b, ei_b, pb)]:
                wl = lines[wi]
                if ei == 0:
                    lines[wi] = WallLine(
                        x1=float(midpoint[0]), y1=float(midpoint[1]),
                        x2=wl.x2, y2=wl.y2,
                        length=wl.length, num_points=wl.num_points,
                    )
                else:
                    lines[wi] = WallLine(
                        x1=wl.x1, y1=wl.y1,
                        x2=float(midpoint[0]), y2=float(midpoint[1]),
                        length=wl.length, num_points=wl.num_points,
                    )
            # Recompute lengths
            for wi in [wi_a, wi_b]:
                wl = lines[wi]
                wl.length = float(math.hypot(wl.x2 - wl.x1, wl.y2 - wl.y1))
            changed = True
    return lines


def _flatten_deviations(
    lines: List[WallLine],
    short_threshold: float = 1.0,
    deviation_threshold: float = 0.2,
) -> List[WallLine]:
    """Snap short wall segments that deviate slightly onto nearby long walls.

    For each wall shorter than ``short_threshold``, find the nearest longer
    wall. If the short wall's midpoint is within ``deviation_threshold``
    perpendicular distance of the long wall's line, project the short segment
    onto the long wall's line direction.
    """
    if len(lines) <= 1:
        return lines

    result: List[WallLine] = []
    for wl in lines:
        if wl.length >= short_threshold:
            result.append(wl)
            continue
        # Find nearest longer wall.
        mid = np.array([(wl.x1 + wl.x2) / 2, (wl.y1 + wl.y2) / 2])
        best_long: Optional[WallLine] = None
        best_dist = float("inf")
        for other in lines:
            if other is wl or other.length < short_threshold:
                continue
            d = _point_line_distance(mid, np.array([other.x1, other.y1]), np.array([other.x2, other.y2]))
            if d < best_dist:
                best_dist = d
                best_long = other
        if best_long is not None and best_dist < deviation_threshold:
            # Project short segment onto the long wall's line.
            a = np.array([best_long.x1, best_long.y1])
            b = np.array([best_long.x2, best_long.y2])
            ab = b - a
            ab_len = float(np.linalg.norm(ab))
            if ab_len < 1e-6:
                result.append(wl)
                continue
            ab_dir = ab / ab_len
            # Project short wall endpoints onto the long line.
            p0 = np.array([wl.x1, wl.y1])
            p1 = np.array([wl.x2, wl.y2])
            t0 = np.clip(np.dot(p0 - a, ab_dir) / ab_len, 0.0, 1.0)
            t1 = np.clip(np.dot(p1 - a, ab_dir) / ab_len, 0.0, 1.0)
            proj0 = a + t0 * ab
            proj1 = a + t1 * ab
            new_len = float(np.linalg.norm(proj1 - proj0))
            if new_len > 0.05:
                result.append(WallLine(
                    x1=float(proj0[0]), y1=float(proj0[1]),
                    x2=float(proj1[0]), y2=float(proj1[1]),
                    length=new_len, num_points=wl.num_points,
                ))
            # else: segment collapses, drop it
        else:
            result.append(wl)
    return result


def _remove_isolated_walls(
    lines: List[WallLine],
    isolation_threshold: float = 2.0,
) -> List[WallLine]:
    """Remove walls whose midpoint is far from all other walls.

    A wall is isolated if its minimum distance to any other wall's
    nearest point exceeds ``isolation_threshold`` meters.
    """
    if len(lines) <= 1:
        return lines

    result: List[WallLine] = []
    for i, wl in enumerate(lines):
        mid = np.array([(wl.x1 + wl.x2) / 2, (wl.y1 + wl.y2) / 2])
        min_dist = float("inf")
        for j, other in enumerate(lines):
            if i == j:
                continue
            # Distance from mid to the other wall segment.
            d = _point_line_distance(
                mid,
                np.array([other.x1, other.y1]),
                np.array([other.x2, other.y2]),
            )
            min_dist = min(min_dist, d)
        if min_dist <= isolation_threshold:
            result.append(wl)
    return result


def _close_loop_via_intersections(
    lines: List[WallLine],
    max_corner_dist: float = 1.5,
) -> List[WallLine]:
    """Snap each wall endpoint to the nearest wall-wall line intersection.

    For each endpoint of each wall, finds the nearest OTHER wall, computes
    the infinite-line intersection, and if it's within ``max_corner_dist``
    of the current endpoint, snaps the endpoint to that intersection.

    This creates proper corners where walls meet, forming a closed loop.
    """
    if len(lines) <= 1:
        return lines

    def line_intersection(
        p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Intersection of two infinite 2D lines. Returns None if parallel."""
        det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
        if abs(det) < 1e-8:
            return None
        diff = p2 - p1
        t = (diff[0] * (-d2[1]) - (-d2[0]) * diff[1]) / det
        return p1 + t * d1

    result: List[WallLine] = []
    for i, wl in enumerate(lines):
        p0 = np.array([wl.x1, wl.y1])
        p1 = np.array([wl.x2, wl.y2])
        wall_dir = p1 - p0
        dir_norm = float(np.linalg.norm(wall_dir))
        if dir_norm < 1e-6:
            result.append(wl)
            continue
        wall_dir_unit = wall_dir / dir_norm

        # For each endpoint, find best intersection with another wall.
        new_p0 = p0.copy()
        new_p1 = p1.copy()

        for endpoint, other_end, label in [
            (p0, p1, "p0"), (p1, p0, "p1"),
        ]:
            best_pt = endpoint.copy()
            best_dist = max_corner_dist
            for j, other in enumerate(lines):
                if i == j:
                    continue
                op0 = np.array([other.x1, other.y1])
                op1 = np.array([other.x2, other.y2])
                other_dir = op1 - op0
                other_norm = float(np.linalg.norm(other_dir))
                if other_norm < 1e-6:
                    continue
                other_dir_unit = other_dir / other_norm
                intersection = line_intersection(
                    endpoint, wall_dir_unit, op0, other_dir_unit,
                )
                if intersection is None:
                    continue
                dist = float(np.linalg.norm(intersection - endpoint))
                if dist < best_dist:
                    best_dist = dist
                    best_pt = intersection
            if label == "p0":
                new_p0 = best_pt
            else:
                new_p1 = best_pt

        new_len = float(np.linalg.norm(new_p1 - new_p0))
        if new_len >= 0.1:
            result.append(WallLine(
                x1=float(new_p0[0]), y1=float(new_p0[1]),
                x2=float(new_p1[0]), y2=float(new_p1[1]),
                length=new_len, num_points=wl.num_points,
            ))
    return result


def extract_wall_lines(
    scans: List[ScanResult],
    wall_class_idx: int = 0,
    exclude_classes: Optional[List[int]] = None,
    center: Optional[np.ndarray] = None,
    # New pipeline parameters
    grid_resolution: float = 0.05,
    morph_kernel_size: int = 7,
    dbscan_eps: float = 0.15,
    dbscan_min_samples: int = 10,
    dp_epsilon_factor: float = 0.012,
    min_wall_length: float = 0.3,
    # Legacy parameters (ignored, kept for API compat)
    split_threshold: float = 0.08,
    min_segment_points: int = 8,
    angle_bin_deg: float = 0.5,
) -> Tuple[List[WallLine], np.ndarray]:
    """Extract wall lines via grid rasterization + morphology + contour + Douglas-Peucker.

    Replaces the previous polar split-and-merge pipeline. The new pipeline:

      1. Collect wall-surface points (exclude floor/ceiling/furniture).
      2. DBSCAN clustering to remove noise outliers.
      3. Rasterize points to an occupancy grid (``grid_resolution`` m/px).
      4. Morphological closing (dilate+erode) to bridge gaps from occlusion.
      5. Extract the largest external contour (guaranteed closed polygon).
      6. Douglas-Peucker simplification → corner vertices.
      7. Convert consecutive corners to WallLine segments (closed loop).

    Args:
        scans: Multi-height scan results with semantic labels.
        wall_class_idx: Class index for "wall" (used when exclude_classes is None).
        exclude_classes: Keep all points except these classes.
            Default [1, 2, 8] = floor, ceiling, furniture.
        grid_resolution: Occupancy grid resolution in meters per pixel.
        morph_kernel_size: Closing kernel size in pixels (bridges gaps of
            approximately ``morph_kernel_size * grid_resolution`` meters).
        dbscan_eps: DBSCAN neighborhood radius for noise removal.
        dbscan_min_samples: DBSCAN minimum cluster size.
        dp_epsilon_factor: Douglas-Peucker tolerance as fraction of perimeter.
        min_wall_length: Minimum wall segment length in meters.

    Returns:
        (wall_lines, all_wall_points) — extracted walls and raw points.
    """
    import cv2
    from sklearn.cluster import DBSCAN

    # --- Step 1: Collect wall points ---
    if exclude_classes is None:
        exclude_classes = [1, 2, 8]  # floor, ceiling, furniture
    wall_pts, _ = extract_wall_points(scans, wall_class_idx, exclude_classes)
    if len(wall_pts) < 10:
        return [], wall_pts

    # --- Step 2: DBSCAN noise removal ---
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(wall_pts)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return [], wall_pts
    # Keep clusters with ≥10% of the largest cluster's size.
    size_threshold = counts.max() * 0.1
    valid_labels = set(unique_labels[counts >= size_threshold].tolist())
    mask = np.isin(labels, list(valid_labels))
    clean_pts = wall_pts[mask]
    if len(clean_pts) < 10:
        return [], wall_pts

    # --- Step 3: Rasterize to occupancy grid ---
    x_min, y_min = clean_pts[:, 0].min(), clean_pts[:, 1].min()
    x_max, y_max = clean_pts[:, 0].max(), clean_pts[:, 1].max()
    grid_w = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    grid_h = int(np.ceil((y_max - y_min) / grid_resolution)) + 1

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    px = ((clean_pts[:, 0] - x_min) / grid_resolution).astype(np.int32)
    py = ((clean_pts[:, 1] - y_min) / grid_resolution).astype(np.int32)
    px = np.clip(px, 0, grid_w - 1)
    py = np.clip(py, 0, grid_h - 1)
    grid[py, px] = 255

    # --- Step 4: Morphological closing ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size),
    )
    grid_closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

    # --- Step 5: Extract largest contour ---
    contours, _ = cv2.findContours(grid_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], wall_pts
    largest = max(contours, key=cv2.contourArea)

    # --- Step 6: Douglas-Peucker simplification ---
    perimeter = cv2.arcLength(largest, True)
    epsilon = max(dp_epsilon_factor * perimeter, 2.0)  # at least 2 pixels
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    # --- Step 7: Convert pixel corners → world coords → WallLine ---
    corners_px = simplified.reshape(-1, 2)
    world_corners = np.zeros_like(corners_px, dtype=np.float64)
    world_corners[:, 0] = corners_px[:, 0] * grid_resolution + x_min
    world_corners[:, 1] = corners_px[:, 1] * grid_resolution + y_min

    n = len(world_corners)
    wall_lines: List[WallLine] = []
    for i in range(n):
        p0 = world_corners[i]
        p1 = world_corners[(i + 1) % n]  # closed loop
        length = float(np.linalg.norm(p1 - p0))
        if length >= min_wall_length:
            wall_lines.append(WallLine(
                x1=float(p0[0]), y1=float(p0[1]),
                x2=float(p1[0]), y2=float(p1[1]),
                length=length,
                num_points=0,
            ))

    return wall_lines, wall_pts


def wall_lines_to_json(
    wall_lines: List[WallLine],
    scans: List[ScanResult],
    center: np.ndarray,
) -> Dict[str, Any]:
    """Serialize wall lines + scan metadata to JSON-serializable dict."""
    return {
        "walls": [w.to_dict() for w in wall_lines],
        "num_walls": len(wall_lines),
        "scan_info": {
            "center": center.tolist(),
            "heights": [s.height for s in scans],
            "num_heights": len(scans),
        },
    }


def save_wall_lines_plot(
    wall_lines: List[WallLine],
    wall_points: np.ndarray,
    center: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """Save a top-down wall line visualization as PNG.

    Shows:
      - Wall scan points (light blue, small dots)
      - Extracted wall lines (red, thick) with endpoints marked (black)
      - Scan center (black +)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot wall scan points
    if len(wall_points) > 0:
        ax.scatter(
            wall_points[:, 0], wall_points[:, 1],
            s=0.2, c="steelblue", alpha=0.3, label="Wall scan pts",
        )

    # Plot extracted wall lines
    for i, wl in enumerate(wall_lines):
        ax.plot(
            [wl.x1, wl.x2], [wl.y1, wl.y2],
            "r-", linewidth=2.5, alpha=0.8,
            label="Wall lines" if i == 0 else None,
        )
        ax.plot(
            [wl.x1, wl.x2], [wl.y1, wl.y2],
            "ko", markersize=5,
        )

    # Scan center
    ax.plot(center[0], center[1], "k+", markersize=20, markeredgewidth=3)

    ax.set_aspect("equal")
    # Auto-scale to wall points + lines
    all_x = list(wall_points[:, 0]) if len(wall_points) > 0 else [center[0]]
    all_y = list(wall_points[:, 1]) if len(wall_points) > 0 else [center[1]]
    for wl in wall_lines:
        all_x.extend([wl.x1, wl.x2])
        all_y.extend([wl.y1, wl.y2])
    margin = max(
        np.std(all_x) * 3 if len(all_x) > 1 else 5,
        np.std(all_y) * 3 if len(all_y) > 1 else 5,
        3.0,
    )
    cx_data, cy_data = float(np.mean(all_x)), float(np.mean(all_y))
    ax.set_xlim(float(cx_data - margin), float(cx_data + margin))
    ax.set_ylim(float(cy_data - margin), float(cy_data + margin))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or f"Extracted Wall Lines ({len(wall_lines)} walls)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
