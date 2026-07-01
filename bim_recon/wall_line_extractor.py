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
        if exclude_set is not None:
            mask = np.array([
                int(l) not in exclude_set
                for l in scan.semantic_labels
            ])
        else:
            mask = scan.semantic_labels == wall_class_idx
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

    Returns (bin_angles_rad, bin_distances) for bins with data.
    """
    if len(angles) == 0:
        return np.empty(0), np.empty(0)
    bin_rad = math.radians(bin_deg)
    bins = np.arange(angles.min(), angles.max() + bin_rad, bin_rad)
    bin_centers = []
    bin_dists = []
    for i in range(len(bins) - 1):
        mask = (angles >= bins[i]) & (angles < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_dists.append(np.median(dists[mask]))
    return np.array(bin_centers), np.array(bin_dists)


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


def extract_wall_lines(
    scans: List[ScanResult],
    wall_class_idx: int = 0,
    exclude_classes: Optional[List[int]] = None,
    center: Optional[np.ndarray] = None,
    split_threshold: float = 0.08,
    min_segment_points: int = 8,
    angle_bin_deg: float = 0.5,
) -> Tuple[List[WallLine], np.ndarray]:
    """Extract wall line segments from multi-height semantic scans.

    Algorithm:
      1. Collect wall-surface points (broadened: exclude floor/ceiling/furniture).
      2. Sort by azimuth angle around scan center.
      3. Bin into uniform angle bins (median distance per bin).
      4. Apply split-and-merge on the ordered Cartesian points.
      5. Convert segments to WallLine with endpoints.

    Args:
        scans: Multi-height scan results with semantic labels.
        wall_class_idx: Class index for "wall" (used when exclude_classes is None).
        exclude_classes: If given, keep all points except these classes.
            Default [1, 2, 8] = floor, ceiling, furniture (broadens wall
            detection to include misclassified surfaces like door/window).
        center: Scan center (auto from first scan if None).
        split_threshold: Max point-line distance for splitting.
        min_segment_points: Min points per wall segment.
        angle_bin_deg: Angular resolution for polar binning.

    Returns:
        (wall_lines, all_wall_points) — extracted walls and raw points.
    """
    if exclude_classes is None:
        exclude_classes = [1, 2, 8]  # floor, ceiling, furniture
    wall_pts, _ = extract_wall_points(scans, wall_class_idx, exclude_classes)
    if len(wall_pts) < 10:
        return [], wall_pts

    if center is None:
        center = scans[0].center_2d

    # Polar sort + bin.
    angles, dists, _ = _polar_sort(wall_pts, center)
    bin_angles, bin_dists = _bin_polar(angles, dists, angle_bin_deg)
    if len(bin_angles) < 4:
        return [], wall_pts

    bin_pts = _polar_to_cartesian(bin_angles, bin_dists, center)

    # Split-and-merge.
    seg_indices = _split_and_merge(
        bin_pts,
        dist_threshold=split_threshold,
        min_points=min_segment_points,
    )

    # Convert to WallLine, handling wraparound (0°/360° boundary).
    wall_lines: List[WallLine] = []
    for s, e in seg_indices:
        p0 = bin_pts[s]
        p1 = bin_pts[e]
        length = float(np.linalg.norm(p1 - p0))
        if length < 0.3:  # skip very short segments
            continue
        wall_lines.append(WallLine(
            x1=float(p0[0]), y1=float(p0[1]),
            x2=float(p1[0]), y2=float(p1[1]),
            length=length,
            num_points=e - s + 1,
        ))

    # Post-process: merge fragmented collinear segments.
    wall_lines = _merge_wall_lines(wall_lines, angle_thresh_deg=12.0, gap_thresh=1.5)

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
