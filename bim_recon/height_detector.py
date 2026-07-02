"""Height detection for wall-mounted BIM elements.

After VLM confirms an element (door, window), this module refines its
vertical extent (sill height, header height) using targeted depth probing
from the 3DGS scene.

Two-phase approach:
1. Coarse: probe at ``coarse_step`` (0.2 m) intervals across the full wall
   height to find approximate opening boundaries.
2. Fine: linear scan at ``fine_step`` (0.02 m) around each coarse boundary
   to pin the exact transition height.

Dual signal at each probe height:
- **Depth discontinuity**: ``|depth - median_wall_depth| > threshold``
  indicates the ray passed through an opening rather than hitting the wall
  surface.
- **Semantic match**: the dominant Gaussian class at the probe pixel matches
  the element's ``class_idx``.

An opening is detected when *either* signal fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from bim_recon.candidate_extractor import Candidate
from bim_recon.gs_scene import GSScene, look_at_pose


@dataclass
class HeightResult:
    """Detected vertical extent of an element."""

    sill_height: float       # metres above floor
    header_height: float     # metres above floor
    element_height: float    # header - sill
    confidence: float        # 0.0 – 1.0
    method: str              # "depth+semantic" or "fallback"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _inward_normal(
    wall: Dict[str, Any],
    scan_center: Tuple[float, float],
) -> np.ndarray:
    """Wall unit-normal pointing toward the room interior (*scan_center*)."""
    start = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
    end = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return np.array([1.0, 0.0])
    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    mid = (start + end) * 0.5
    to_center = np.array(scan_center, dtype=np.float64) - mid
    if float(np.dot(normal, to_center)) < 0.0:
        normal = -normal
    return normal


# ---------------------------------------------------------------------------
# Single-height probe
# ---------------------------------------------------------------------------

def _probe_height(
    scene: GSScene,
    target_xy: np.ndarray,
    height: float,
    inward_normal: np.ndarray,
    up_axis: int,
    class_idx: Optional[int],
    camera_dist: float,
    img_size: int,
    fov: float,
) -> Tuple[float, Optional[int]]:
    """Render a tiny image looking at the wall at *height*.

    Returns ``(center_depth, center_semantic_label)``.
    ``center_depth < 0`` means no geometry was hit.
    """
    h_axes = [i for i in range(3) if i != up_axis]

    eye_xy = target_xy + inward_normal * camera_dist
    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = float(eye_xy[0])
    eye[h_axes[1]] = float(eye_xy[1])
    eye[up_axis] = height

    tgt = [0.0, 0.0, 0.0]
    tgt[h_axes[0]] = float(target_xy[0])
    tgt[h_axes[1]] = float(target_xy[1])
    tgt[up_axis] = height

    up = [0.0, 0.0, 0.0]
    up[up_axis] = 1.0

    pose = look_at_pose(
        (eye[0], eye[1], eye[2]),
        (tgt[0], tgt[1], tgt[2]),
        (up[0], up[1], up[2]),
    )

    # --- geometry pass --------------------------------------------------
    result = scene.render(pose, width=img_size, height=img_size, fov_degrees=fov)
    c = img_size // 2
    depth = float(result.depth[c, c])
    alpha = float(result.alpha[c, c])
    if alpha < 0.1:
        return -1.0, None

    # --- semantic pass (optional) --------------------------------------
    label: Optional[int] = None
    if class_idx is not None and scene.semantic_querier is not None and scene.feat is not None:
        querier = scene.semantic_querier
        dominant = querier.get_dominant_labels()
        num_classes = querier.num_classes
        n = scene.num_gaussians

        enc = torch.zeros((n, 3), dtype=torch.float32, device=scene.device)
        if num_classes > 1:
            enc[:, 0] = (
                torch.from_numpy(dominant.astype(np.float32))
                .to(scene.device)
                / (num_classes - 1)
            )

        orig = scene.colors
        try:
            scene.colors = enc
            sem = scene.render(pose, width=img_size, height=img_size, fov_degrees=fov)
        finally:
            scene.colors = orig

        sem_alpha = float(sem.alpha[c, c])
        if sem_alpha > 0.1:
            label = int(round(float(sem.colors[c, c, 0]) * (num_classes - 1)))

    return depth, label


def _is_opening(
    depth: float,
    label: Optional[int],
    class_idx: int,
    ref_depth: float,
    depth_threshold: float,
) -> bool:
    """Decide whether a probe hit an opening (void or element semantics)."""
    sem_match = label is not None and label == class_idx
    if depth < 0.0:
        return sem_match
    is_void = abs(depth - ref_depth) > depth_threshold
    return is_void or sem_match


# ---------------------------------------------------------------------------
# Fine boundary search
# ---------------------------------------------------------------------------

def _fine_boundary(
    scene: GSScene,
    target_xy: np.ndarray,
    inward_normal: np.ndarray,
    up_axis: int,
    class_idx: int,
    ref_depth: float,
    depth_threshold: float,
    camera_dist: float,
    img_size: int,
    fov: float,
    h_lo: float,
    h_hi: float,
    step: float,
    find_bottom: bool,
) -> float:
    """Linear fine scan to pin the exact opening boundary height."""
    heights = np.arange(h_lo, h_hi + step * 0.5, step)
    result_h = h_lo if find_bottom else h_hi
    for h in heights:
        depth, label = _probe_height(
            scene, target_xy, float(h), inward_normal, up_axis,
            class_idx, camera_dist, img_size, fov,
        )
        opening = _is_opening(depth, label, class_idx, ref_depth, depth_threshold)
        if find_bottom:
            if opening:
                return float(h)
        elif opening:
            result_h = float(h)
    return result_h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_element_heights(
    scene: GSScene,
    candidate: Candidate,
    wall: Dict[str, Any],
    floor_z: float,
    ceiling_z: float,
    scan_center: Tuple[float, float],
    class_idx: int,
    up_axis: int = 2,
    coarse_step: float = 0.2,
    fine_step: float = 0.02,
    camera_dist: float = 1.0,
    depth_threshold: float = 0.15,
    img_size: int = 64,
    fov: float = 30.0,
) -> HeightResult:
    """Detect precise sill/header heights for a confirmed wall element.

    Args:
        scene: Renderable 3DGS scene (with optional semantics).
        candidate: VLM-confirmed element candidate.
        wall: Wall dict with ``x1, y1, x2, y2`` keys.
        floor_z: Floor level (up-axis world coordinate, metres).
        ceiling_z: Ceiling level (metres).
        scan_center: Room centre used to orient the inward normal.
        class_idx: Semantic class index for this element type.
        up_axis: Up axis (0=x, 1=y, 2=z).
        coarse_step: Coarse scan interval (metres).
        fine_step: Fine scan interval (metres).
        camera_dist: Probing camera distance from wall (metres).
        depth_threshold: Depth change to classify as void (metres).
        img_size: Render resolution per probe (square).
        fov: Probe camera FOV (degrees).

    Returns:
        :class:`HeightResult` with sill/header above floor and confidence.
    """
    inward_n = _inward_normal(wall, scan_center)
    target_xy = np.array([candidate.world_x, candidate.world_y], dtype=np.float64)

    # --- Phase 1: coarse scan -------------------------------------------
    h_lo = floor_z + 0.05
    h_hi = ceiling_z - 0.05
    coarse_zs = np.arange(h_lo, h_hi + coarse_step * 0.5, coarse_step)

    probes: List[Tuple[float, float, Optional[int]]] = []
    wall_depths: List[float] = []

    for z in coarse_zs:
        depth, label = _probe_height(
            scene, target_xy, float(z), inward_n, up_axis,
            class_idx, camera_dist, img_size, fov,
        )
        probes.append((float(z), depth, label))
        sem_match = label is not None and label == class_idx
        if not sem_match and 0.05 < depth < 50.0:
            wall_depths.append(depth)

    # Wall surface = closest geometry.  Using minimum (not median) because
    # a tall opening (e.g. full-height door) means >50% of probes hit the
    # void behind the wall, which would skew the median toward the void
    # depth and invert the classification.
    ref_depth = float(np.min(wall_depths)) if wall_depths else camera_dist

    flags = [
        _is_opening(d, lb, class_idx, ref_depth, depth_threshold)
        for _, d, lb in probes
    ]
    opening_zs = [probes[i][0] for i in range(len(flags)) if flags[i]]

    if not opening_zs:
        return HeightResult(
            sill_height=candidate.h_min,
            header_height=candidate.h_max,
            element_height=candidate.h_max - candidate.h_min,
            confidence=0.3,
            method="fallback",
        )

    coarse_sill = min(opening_zs)
    coarse_header = max(opening_zs)

    # --- Phase 2: fine boundary search ----------------------------------
    fine_sill = _fine_boundary(
        scene, target_xy, inward_n, up_axis, class_idx,
        ref_depth, depth_threshold, camera_dist, img_size, fov,
        h_lo=max(h_lo, coarse_sill - coarse_step),
        h_hi=min(h_hi, coarse_sill + coarse_step),
        step=fine_step,
        find_bottom=True,
    )
    fine_header = _fine_boundary(
        scene, target_xy, inward_n, up_axis, class_idx,
        ref_depth, depth_threshold, camera_dist, img_size, fov,
        h_lo=max(h_lo, coarse_header - coarse_step),
        h_hi=min(h_hi, coarse_header + coarse_step),
        step=fine_step,
        find_bottom=False,
    )

    if fine_header <= fine_sill:
        fine_sill = coarse_sill
        fine_header = coarse_header

    opening_ratio = sum(flags) / max(1, len(flags))
    confidence = round(min(1.0, opening_ratio * 2.0), 2)

    return HeightResult(
        sill_height=round(fine_sill - floor_z, 3),
        header_height=round(fine_header - floor_z, 3),
        element_height=round(fine_header - fine_sill, 3),
        confidence=confidence,
        method="depth+semantic",
    )
