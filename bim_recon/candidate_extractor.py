"""Element candidate extraction from multi-height semantic scans.

Given ScanResults with semantic labels and a set of wall lines, extracts
candidate element locations (doors, windows, furniture, ...) by:

  1. Filtering scan points by target semantic class.
  2. Projecting structural elements onto the nearest wall line.
  3. Clustering projected positions into discrete openings.
  4. Computing polar coordinates (θ, r) for each candidate.

This module is pure numpy — no torch, no GPU. It operates on the output
of :mod:`bim_recon.virtual_scanner` and :mod:`bim_recon.wall_line_extractor`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bim_recon.virtual_scanner import ScanResult


@dataclass
class Candidate:
    """A candidate element location detected from radar scan + feat.pt."""

    element_class: str          # "door", "window", "furniture", ...
    class_idx: int              # semantic label index
    wall_idx: Optional[int]     # wall line index (None for free-standing)
    t_min: float                # parameter along wall [0, 1]
    t_max: float
    theta_center: float         # polar azimuth from scan center (degrees)
    theta_span: float           # angular extent (degrees)
    r_mean: float               # mean distance from scan center (meters)
    h_min: float                # min height above floor (meters)
    h_max: float                # max height above floor (meters)
    width_m: float              # estimated width (meters)
    num_points: int             # scan point count
    world_x: float              # candidate center world XY
    world_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_class": self.element_class,
            "class_idx": self.class_idx,
            "wall_idx": self.wall_idx,
            "t_min": round(self.t_min, 4),
            "t_max": round(self.t_max, 4),
            "theta_center": round(self.theta_center, 2),
            "theta_span": round(self.theta_span, 2),
            "r_mean": round(self.r_mean, 4),
            "h_min": round(self.h_min, 4),
            "h_max": round(self.h_max, 4),
            "width_m": round(self.width_m, 4),
            "num_points": self.num_points,
            "world_x": round(self.world_x, 4),
            "world_y": round(self.world_y, 4),
        }


def project_point_to_wall(
    pt: np.ndarray, wall_start: np.ndarray, wall_end: np.ndarray,
) -> Tuple[float, float]:
    """Project a 2D point onto a wall segment.

    Returns ``(t, dist)`` where *t* is the parameter along the wall [0, 1]
    (clamped) and *dist* is the perpendicular distance from the point to
    the wall segment (meters).
    """
    seg = wall_end - wall_start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return 0.0, float(np.linalg.norm(pt - wall_start))
    t = float(np.dot(pt - wall_start, seg) / seg_len_sq)
    t_clamped = max(0.0, min(1.0, t))
    closest = wall_start + t_clamped * seg
    dist = float(np.linalg.norm(pt - closest))
    return t_clamped, dist


def _cluster_openings(
    ts: List[float],
    hs: List[float],
    wall_length: float,
    min_gap: float = 0.3,
    min_pts: int = 5,
) -> List[Dict[str, Any]]:
    """Cluster 1-D t-parameters into groups separated by > *min_gap* meters."""
    if len(ts) < min_pts:
        return []
    order = np.argsort(ts)
    ts_arr = np.array(ts)[order]
    hs_arr = np.array(hs)[order]

    clusters: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, len(ts_arr)):
        gap_m = (ts_arr[i] - ts_arr[i - 1]) * wall_length
        if gap_m > min_gap:
            clusters.append((ts_arr[start:i], hs_arr[start:i]))
            start = i
    clusters.append((ts_arr[start:], hs_arr[start:]))

    openings: List[Dict[str, Any]] = []
    for cluster_ts, cluster_hs in clusters:
        if len(cluster_ts) < min_pts:
            continue
        t_min = float(np.min(cluster_ts))
        t_max = float(np.max(cluster_ts))
        openings.append({
            "t_center": float(np.mean(cluster_ts)),
            "t_min": t_min,
            "t_max": t_max,
            "width_m": (t_max - t_min) * wall_length,
            "h_min": float(np.min(cluster_hs)),
            "h_max": float(np.max(cluster_hs)),
            "num_points": len(cluster_ts),
        })
    return openings


def extract_candidates(
    scans: List[ScanResult],
    walls: List[Dict[str, Any]],
    floor_z: float,
    scan_center: Tuple[float, float],
    element_class: str = "door",
    class_idx: int = 3,
    project_to_walls: bool = True,
    max_wall_dist: float = 0.5,
    cluster_min_gap: float = 0.3,
    cluster_min_pts: int = 5,
) -> List[Candidate]:
    """Extract element candidates from multi-height scan data.

    Args:
        scans: List of :class:`ScanResult` with ``semantic_labels``.
        walls: List of wall dicts with keys ``x1, y1, x2, y2, length``.
        floor_z: Floor level world coordinate (up-axis).
        scan_center: (cx, cy) scan center in world XY.
        element_class: Human-readable label (e.g. ``"door"``).
        class_idx: Semantic class index for this element.
        project_to_walls: If True, project points onto nearest wall line.
            If False, cluster freely in XY (for furniture).
        max_wall_dist: Max perpendicular distance for wall projection.
        cluster_min_gap: Min gap (meters) to split clusters.
        cluster_min_pts: Min points per cluster.

    Returns:
        List of :class:`Candidate` sorted by wall index then position.
    """
    cx, cy = scan_center

    if project_to_walls:
        # Structural elements: project onto walls, cluster per-wall.
        wall_data: List[Dict[str, Any]] = []
        wall_starts_list: List[np.ndarray] = []
        wall_ends_list: List[np.ndarray] = []
        for w in walls:
            ws = np.array([w["x1"], w["y1"]])
            we = np.array([w["x2"], w["y2"]])
            wall_data.append({
                "start": ws, "end": we,
                "length": w["length"],
                "ts": [], "hs": [], "world_xs": [], "world_ys": [],
            })
            wall_starts_list.append(ws)
            wall_ends_list.append(we)

        # Pre-stack wall geometry for vectorized projection
        W = len(wall_data)
        if W > 0:
            wall_starts = np.stack(wall_starts_list)  # (W, 2)
            wall_ends = np.stack(wall_ends_list)      # (W, 2)
            segs = wall_ends - wall_starts             # (W, 2)
            seg_len_sq = np.sum(segs ** 2, axis=1)     # (W,)
            seg_len_sq = np.clip(seg_len_sq, 1e-12, None)

            for scan in scans:
                if scan.semantic_labels is None:
                    continue
                mask = scan.semantic_labels == class_idx
                if mask.sum() == 0:
                    continue
                rel_h = scan.height - floor_z
                pts = scan.points_2d[mask]  # (P, 2)

                # Vectorized projection of P points onto W walls
                diffs = pts[:, None, :] - wall_starts[None, :, :]  # (P, W, 2)
                t = np.einsum("pwd,wd->pw", diffs, segs) / seg_len_sq[None, :]  # (P, W)
                t_clamped = np.clip(t, 0.0, 1.0)
                closest = wall_starts[None, :, :] + t_clamped[:, :, None] * segs[None, :, :]
                dist = np.linalg.norm(pts[:, None, :] - closest, axis=2)  # (P, W)

                # Best wall per point
                best_wi = np.argmin(dist, axis=1)
                P = len(pts)
                best_dist = dist[np.arange(P), best_wi]
                best_t = t_clamped[np.arange(P), best_wi]

                for pi in range(P):
                    wi = int(best_wi[pi])
                    if best_dist[pi] < max_wall_dist:
                        wd = wall_data[wi]
                        wd["ts"].append(float(best_t[pi]))
                        wd["hs"].append(rel_h)
                        wd["world_xs"].append(float(pts[pi, 0]))
                        wd["world_ys"].append(float(pts[pi, 1]))

        candidates: List[Candidate] = []
        for wi, wd in enumerate(wall_data):
            if not wd["ts"]:
                continue
            openings = _cluster_openings(
                wd["ts"], wd["hs"], wd["length"],
                cluster_min_gap, cluster_min_pts,
            )
            for op in openings:
                pos = wd["start"] + op["t_center"] * (wd["end"] - wd["start"])
                wx = float(pos[0])
                wy = float(pos[1])
                dx, dy = wx - cx, wy - cy
                r = float(np.hypot(dx, dy))
                theta = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
                candidates.append(Candidate(
                    element_class=element_class,
                    class_idx=class_idx,
                    wall_idx=wi,
                    t_min=op["t_min"], t_max=op["t_max"],
                    theta_center=theta,
                    theta_span=op["width_m"] / r * 57.2958
                    if r > 1e-6 else 0.0,
                    r_mean=r,
                    h_min=op["h_min"], h_max=op["h_max"],
                    width_m=op["width_m"],
                    num_points=op["num_points"],
                    world_x=wx, world_y=wy,
                ))
        candidates.sort(key=lambda c: (c.wall_idx or 0, c.t_min))
        return candidates

    else:
        # Free-standing elements: DBSCAN cluster in XY.
        from sklearn.cluster import DBSCAN
        all_pts: List[np.ndarray] = []
        all_hs: List[float] = []
        for scan in scans:
            if scan.semantic_labels is None:
                continue
            mask = scan.semantic_labels == class_idx
            if mask.sum() == 0:
                continue
            rel_h = scan.height - floor_z
            for pt in scan.points_2d[mask]:
                all_pts.append(pt)
                all_hs.append(rel_h)
        if len(all_pts) < cluster_min_pts:
            return []
        pts_arr = np.array(all_pts)
        hs_arr = np.array(all_hs)
        clustering = DBSCAN(eps=0.5, min_samples=cluster_min_pts, algorithm="kd_tree").fit(pts_arr)
        candidates = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            mask = clustering.labels_ == label
            cluster_pts = pts_arr[mask]
            cluster_hs = hs_arr[mask]
            wx = float(np.mean(cluster_pts[:, 0]))
            wy = float(np.mean(cluster_pts[:, 1]))
            dx, dy = wx - cx, wy - cy
            r = float(np.hypot(dx, dy))
            theta = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
            candidates.append(Candidate(
                element_class=element_class,
                class_idx=class_idx,
                wall_idx=None,
                t_min=0.0, t_max=0.0,
                theta_center=theta,
                theta_span=0.0,
                r_mean=r,
                h_min=float(np.min(cluster_hs)),
                h_max=float(np.max(cluster_hs)),
                width_m=float(np.max(np.std(cluster_pts, axis=0))) * 4,
                num_points=int(mask.sum()),
                world_x=wx, world_y=wy,
            ))
        return candidates


def prefilter_candidates(
    candidates: List[Candidate],
    min_width: float = 0.0,
    min_points: int = 1,
) -> List[Candidate]:
    """Filter candidates by physical constraints to reduce VLM calls.

    Defaults are permissive (min_width=0, min_points=1) — the VLM is the
    primary filter. Override via ElementConfig if stricter filtering is needed.
    """
    return [
        c for c in candidates
        if c.width_m >= min_width and c.num_points >= min_points
    ]


# Open-vocabulary label → index resolution. The active label set is built by
# the pipeline (structural defaults + element semantic labels) and passed to
# the VirtualScanner; scan points carry argmax indices into THAT set, so every
# consumer must resolve a label name to its position in the same set.
def resolve_class_index(semantic_label: str, labels: Sequence[str]) -> int:
    """Return the position of *semantic_label* in the active *labels* set.

    Raises ValueError if absent.
    """
    if semantic_label not in labels:
        raise ValueError(
            f"semantic_label '{semantic_label}' not in active label set {list(labels)}"
        )
    return labels.index(semantic_label)


# Classic 9-class BIM vocabulary (order matches data/bim_class_names.json).
# Kept as an EXPLICIT fallback for offline tools that re-plot saved scans
# (which store integer indices, not label names) — not used by the live
# open-vocabulary pipeline.
CLASSIC_BIM_VOCAB: Tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "window",
    "column", "beam", "stairs", "furniture",
)

BIM_STRUCTURAL_CLASSES = {"door", "window", "column"}
