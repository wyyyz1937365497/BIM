"""Wall line extraction from multi-height semantic laser scans.

Scans a 3DGS scene at multiple heights (floor to ceiling), collects all
wall-surface points across heights, and extracts wall line segments via
grid rasterization + morphological closing + contour extraction +
Douglas-Peucker simplification.

Key insight: furniture occluding walls is typically shorter than the wall.
Scanning at multiple heights means higher scans see wall surfaces above
furniture, recovering wall geometry that low scans miss.

Pipeline:
  1. DBSCAN clustering to remove noise outliers.
  2. Rasterize wall points to an occupancy grid.
  3. Morphological closing (dilate+erode) to bridge gaps from occlusion.
  4. Extract the largest external contour (guaranteed closed polygon).
  5. Douglas-Peucker simplification to corner vertices.
  6. Convert consecutive corners to WallLine segments (closed loop).
"""
from __future__ import annotations

from dataclasses import dataclass
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


def extract_wall_lines(
    scans: List[ScanResult],
    wall_class_idx: int = 0,
    exclude_classes: Optional[List[int]] = None,
    center: Optional[np.ndarray] = None,
    grid_resolution: float = 0.05,
    morph_kernel_size: int = 7,
    dbscan_eps: float = 0.15,
    dbscan_min_samples: int = 10,
    dp_epsilon_factor: float = 0.012,
    min_wall_length: float = 0.3,
) -> Tuple[List[WallLine], np.ndarray]:
    """Extract wall lines via grid rasterization + morphology + contour + Douglas-Peucker.

    Pipeline:
      1. Collect wall-surface points (exclude floor/ceiling/furniture).
      2. DBSCAN clustering to remove noise outliers.
      3. Rasterize points to an occupancy grid (``grid_resolution`` m/px).
      4. Morphological closing (dilate+erode) to bridge gaps from occlusion.
      5. Extract the largest external contour (guaranteed closed polygon).
      6. Douglas-Peucker simplification to corner vertices.
      7. Convert consecutive corners to WallLine segments (closed loop).

    Args:
        scans: Multi-height scan results with semantic labels.
        wall_class_idx: Class index for "wall" (used when exclude_classes is None).
        exclude_classes: Keep all points except these classes.
            Default [1, 2, 8] = floor, ceiling, furniture.
        center: Unused (kept for API compatibility with callers).
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
    epsilon = max(dp_epsilon_factor * perimeter, 2.0)
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    # --- Step 7: Convert pixel corners to world coords, build closed loop ---
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

    Shows wall scan points (light blue), extracted wall lines (red) with
    endpoints marked (black), and scan center (black +).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if len(wall_points) > 0:
        ax.scatter(
            wall_points[:, 0], wall_points[:, 1],
            s=0.2, c="steelblue", alpha=0.3, label="Wall scan pts",
        )

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

    ax.plot(center[0], center[1], "k+", markersize=20, markeredgewidth=3)

    ax.set_aspect("equal")
    all_x = list(wall_points[:, 0]) if len(wall_points) > 0 else [center[0]]
    all_y = list(wall_points[:, 1]) if len(wall_points) > 0 else [center[1]]
    for wl in wall_lines:
        all_x.extend([wl.x1, wl.x2])
        all_y.extend([wl.y1, wl.y2])
    margin = max(
        float(np.std(all_x)) * 3 if len(all_x) > 1 else 5.0,
        float(np.std(all_y)) * 3 if len(all_y) > 1 else 5.0,
        3.0,
    )
    cx_data, cy_data = float(np.mean(all_x)), float(np.mean(all_y))
    ax.set_xlim(cx_data - margin, cx_data + margin)
    ax.set_ylim(cy_data - margin, cy_data + margin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or f"Extracted Wall Lines ({len(wall_lines)} walls)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
