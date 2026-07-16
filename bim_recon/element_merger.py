"""Merge element detections from multiple 360° views into unified components.

The problem: when rendering N panoramic views around a room, the same door or
window is often detected from 2–3 adjacent viewpoints.  Without merging, each
view creates a separate Revit element, resulting in scattered duplicates.

The solution (as requested in the design): project **all** detections from all
views onto a shared polar coordinate system (θ, r) centered on the room center,
then cluster detections of the same type that are within a threshold distance.

Clustering uses DBSCAN on the (θ, r) plane with an angular-aware metric:
two detections are "close" if their angular separation × distance is less than
*merge_threshold* metres AND their height ranges overlap.

Usage::

    from bim_recon.element_merger import merge_detections

    merged = merge_detections(
        all_detections,       # list of detection dicts from all views
        center=(cx, cy),
        up_axis=2,
        merge_threshold=0.5,  # metres
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MergedElement:
    """A single element after merging multi-view detections."""

    element_class: str              # "door", "window", ...
    world_x: float                  # merged center in world XY
    world_y: float
    theta_center: float             # polar azimuth from room center (degrees)
    r_mean: float                   # mean distance from center (metres)
    width_m: float                  # merged width estimate
    sill_height: float              # height above floor (metres)
    header_height: float
    element_height: float
    wall_idx: Optional[int]         # assigned wall (if structural)
    num_sources: int                # how many raw detections were merged
    source_indices: List[int]       # indices into the original detection list
    confidence: float               # mean of source confidences (0–1)
    best_image_path: Optional[str]  # the clearest source image (for VLM/TRELLIS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_class": self.element_class,
            "world_x": round(self.world_x, 4),
            "world_y": round(self.world_y, 4),
            "theta_center": round(self.theta_center, 2),
            "r_mean": round(self.r_mean, 4),
            "width_m": round(self.width_m, 4),
            "sill_height": round(self.sill_height, 4),
            "header_height": round(self.header_height, 4),
            "element_height": round(self.element_height, 4),
            "wall_idx": self.wall_idx,
            "num_sources": self.num_sources,
            "confidence": round(self.confidence, 3),
            "image_path": self.best_image_path,
        }


def _world_to_polar(
    wx: float, wy: float, cx: float, cy: float,
) -> Tuple[float, float]:
    """World XY → (theta_deg, distance) from room center."""
    dx = wx - cx
    dy = wy - cy
    r = math.hypot(dx, dy)
    theta = math.degrees(math.atan2(dy, dx)) % 360.0
    return theta, r


def _angular_distance(
    theta1: float, r1: float, theta2: float, r2: float,
) -> float:
    """Approximate arc distance between two polar points (metres).

    Uses the chord length on the smaller-radius circle for the angular
    component, which is exact when r1 ≈ r2.
    """
    dt = math.radians(abs(theta1 - theta2) % 360.0)
    dt = min(dt, 2 * math.pi - dt)
    r_min = min(r1, r2)
    arc = r_min * dt
    drad = abs(r1 - r2)
    return math.hypot(arc, drad)


def _height_overlap(
    h1_min: float, h1_max: float, h2_min: float, h2_max: float,
    tol: float = 0.3,
) -> bool:
    """Check if two height ranges overlap (within *tol* metres)."""
    return (h1_min - tol) <= h2_max and (h2_min - tol) <= h1_max


import math  # noqa: E402 — needed by _world_to_polar / _angular_distance


def merge_detections(
    detections: List[Dict[str, Any]],
    center: Tuple[float, float],
    up_axis: int = 2,
    merge_threshold: float = 0.5,
    height_tolerance: float = 0.3,
    walls: Optional[List[Dict[str, Any]]] = None,
) -> List[MergedElement]:
    """Merge nearby element detections in polar coordinates.

    All detections from all 360° views are projected to (θ, r) and
    clustered.  Two detections are neighbours if:

      * Their arc distance in polar space ≤ *merge_threshold* (metres)
      * Their height ranges overlap within *height_tolerance* (metres)
      * They have the same ``element_class`` (label)
      * **They are on the same wall** (when *walls* is provided)

    The wall constraint prevents transitive-chaining across corners:
    without it, a chain of nearby detections can merge windows on
    different walls into one giant cluster.
    """
    if not detections:
        return []

    cx, cy = float(center[0]), float(center[1])

    # --- Wall assignment: project each detection to its nearest wall ---
    if walls:
        wall_segments = [
            (np.array([w["x1"], w["y1"]], dtype=np.float64),
             np.array([w["x2"], w["y2"]], dtype=np.float64))
            for w in walls
        ]
    else:
        wall_segments = None

    def _nearest_wall(wx: float, wy: float) -> Optional[int]:
        if not wall_segments:
            return None
        pt = np.array([wx, wy], dtype=np.float64)
        best_idx, best_dist = 0, 1e18
        for wi, (ws, we) in enumerate(wall_segments):
            seg = we - ws
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                d = float(np.linalg.norm(pt - ws))
            else:
                t = max(0.0, min(1.0, float(np.dot(pt - ws, seg) / seg_len_sq)))
                closest = ws + t * seg
                d = float(np.linalg.norm(pt - closest))
            if d < best_dist:
                best_dist = d
                best_idx = wi
        return best_idx

    # Normalise each detection into a common format
    normals: List[Dict[str, Any]] = []
    for i, det in enumerate(detections):
        wx = det.get("world_x")
        wy = det.get("world_y")
        if wx is None or wy is None:
            wp = det.get("world_pos")
            if wp and len(wp) >= 2:
                wx, wy = float(wp[0]), float(wp[1])
            else:
                continue

        ec = det.get("element_class") or det.get("label") or "unknown"
        sill = float(det.get("sill_height", 0.0))
        header = float(det.get("header_height", sill))
        if header <= sill:
            header = sill + float(det.get("element_height", 0.0))
        theta, r = _world_to_polar(float(wx), float(wy), cx, cy)
        wi = _nearest_wall(float(wx), float(wy))

        normals.append({
            "idx": i,
            "element_class": ec,
            "wx": float(wx),
            "wy": float(wy),
            "theta": theta,
            "r": r,
            "sill": sill,
            "header": header,
            "width": float(det.get("width_m", 0.0)),
            "wall_idx": wi,
            "confidence": float(det.get("confidence", 0.5)),
            "image_path": det.get("image_path"),
            "depth": float(det.get("depth", 0.0)),
        })

    if not normals:
        return []

    # Group by element class first, then DBSCAN within each group
    classes = set(n["element_class"] for n in normals)
    merged_all: List[MergedElement] = []

    for ec in classes:
        group = [n for n in normals if n["element_class"] == ec]
        if not group:
            continue

        # Build adjacency: two detections are neighbours if close in polar + height
        n = len(group)
        labels = -np.ones(n, dtype=np.int32)  # -1 = unassigned
        cluster_id = 0

        for seed in range(n):
            if labels[seed] != -1:
                continue
            # BFS from this seed
            queue = [seed]
            labels[seed] = cluster_id
            while queue:
                cur = queue.pop(0)
                for j in range(n):
                    if labels[j] != -1:
                        continue
                    # Wall constraint: only merge detections on the same wall
                    if wall_segments is not None:
                        if group[cur]["wall_idx"] != group[j]["wall_idx"]:
                            continue
                    # Check polar distance
                    dist = _angular_distance(
                        group[cur]["theta"], group[cur]["r"],
                        group[j]["theta"], group[j]["r"],
                    )
                    if dist > merge_threshold:
                        continue
                    # Check height overlap
                    if not _height_overlap(
                        group[cur]["sill"], group[cur]["header"],
                        group[j]["sill"], group[j]["header"],
                        tol=height_tolerance,
                    ):
                        continue
                    labels[j] = cluster_id
                    queue.append(j)
            cluster_id += 1

        # Merge each cluster
        for cid in range(cluster_id):
            members = [group[i] for i in range(n) if labels[i] == cid]
            if not members:
                continue

            # Weighted average by confidence
            confs = np.array([m["confidence"] for m in members])
            confs = np.maximum(confs, 0.01)
            weights = confs / confs.sum()

            wx = float(np.average([m["wx"] for m in members], weights=weights))
            wy = float(np.average([m["wy"] for m in members], weights=weights))
            theta, r = _world_to_polar(wx, wy, cx, cy)

            # Width: take the max of member widths (widest detection wins,
            # since narrow detections often miss the frame edges)
            width = float(np.max([m["width"] for m in members]))

            # Height: take the union (min sill, max header)
            sill = float(np.min([m["sill"] for m in members]))
            header = float(np.max([m["header"] for m in members]))
            elem_h = max(0.0, header - sill)

            # Wall index: most common among members
            wall_idxs = [m["wall_idx"] for m in members if m["wall_idx"] is not None]
            wall_idx = max(set(wall_idxs), key=wall_idxs.count) if wall_idxs else None

            # Best image: highest-confidence member's image
            best = max(members, key=lambda m: m["confidence"])
            best_img = best.get("image_path")

            # Confidence: mean of member confidences, boosted by consensus
            conf = float(np.mean(confs)) * min(1.0, 0.7 + 0.3 * len(members))

            merged_all.append(MergedElement(
                element_class=ec,
                world_x=wx,
                world_y=wy,
                theta_center=theta,
                r_mean=r,
                width_m=width,
                sill_height=sill,
                header_height=header,
                element_height=elem_h,
                wall_idx=wall_idx,
                num_sources=len(members),
                source_indices=[m["idx"] for m in members],
                confidence=min(conf, 1.0),
                best_image_path=best_img,
            ))

    # Sort by theta for deterministic output
    merged_all.sort(key=lambda e: (e.element_class, e.theta_center))
    return merged_all


def merge_falcon_ring_detections(
    falcon_dets: List[Dict[str, Any]],
    center: Tuple[float, float],
    merge_threshold: float = 0.5,
) -> List[MergedElement]:
    """Convenience wrapper for Falcon ring-scan detections.

    Falcon detections have ``world_pos``, ``label``, ``sill_height``,
    ``header_height``, ``width_m``, ``wall_idx``, ``image_path``.
    """
    return merge_detections(
        falcon_dets,
        center=center,
        merge_threshold=merge_threshold,
    )


def merge_vlm_results(
    vlm_results: List[Dict[str, Any]],
    center: Tuple[float, float],
    element_class: str,
    merge_threshold: float = 0.5,
) -> List[MergedElement]:
    """Convenience wrapper for VLM-verified detection results.

    VLM results have ``candidate.world_x/world_y``, ``height_detection``,
    and ``confirmed`` flag.  Only confirmed results are merged.
    """
    dets = []
    for r in vlm_results:
        if not r.get("confirmed"):
            continue
        c = r.get("candidate", {})
        hd = r.get("height_detection", {}) or {}
        dets.append({
            "element_class": element_class,
            "world_x": c.get("world_x", 0.0),
            "world_y": c.get("world_y", 0.0),
            "sill_height": hd.get("sill_height", c.get("h_min", 0.0)),
            "header_height": hd.get("header_height", c.get("h_max", 0.0)),
            "width_m": hd.get("width_m", c.get("width_m", 0.5)),
            "confidence": 1.0,
            "image_path": r.get("image_path"),
        })

    return merge_detections(dets, center=center, merge_threshold=merge_threshold)
