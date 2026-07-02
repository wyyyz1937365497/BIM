"""Spatial extraction via Falcon-Perception segmentation.

After VLM confirms an element exists at a candidate location, this module
renders a head-on (perpendicular) elevation view of the wall, sends it to
the Falcon inference server for segmentation, and maps the resulting
pixel-space bounding box back to wall-local metric coordinates
(sill height, header height, width along wall).

The perpendicular camera setup ensures a *linear* pixel-to-meter mapping:
for a flat wall at constant depth ``d``, each normalized image coordinate
maps directly to wall-local meters via simple trigonometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from bim_recon.falcon_client import FalconClient, FalconDetection

if TYPE_CHECKING:
    from bim_recon.candidate_extractor import Candidate
    from bim_recon.gs_scene import GSScene


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SpatialResult:
    """Detected spatial extent of a wall-mounted element."""

    sill_height: float        # metres above floor
    header_height: float      # metres above floor
    element_height: float     # header - sill
    width_m: float            # along-wall width
    t_min: float              # wall parameter [0, 1]
    t_max: float
    world_x: float            # element centre world XY
    world_y: float
    confidence: float         # 0.0 – 1.0
    method: str               # "falcon_segmentation" | "falcon_detection"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wall_inward_normal(
    wall: Dict[str, Any],
    scan_center: Tuple[float, float],
) -> np.ndarray:
    """Unit normal of the wall pointing toward the room interior."""
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


def _wall_direction(wall: Dict[str, Any]) -> np.ndarray:
    """Unit direction vector along the wall (start → end)."""
    start = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
    end = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
    d = end - start
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


# ---------------------------------------------------------------------------
# Elevation rendering
# ---------------------------------------------------------------------------

@dataclass
class ElevationParams:
    """Camera parameters needed for pixel-to-meter back-mapping."""

    camera_dist: float
    fov_degrees: float
    img_size: int
    wall_length: float
    wall_dir: np.ndarray          # unit vector along wall (horizontal plane)
    wall_start: np.ndarray        # [x, y] horizontal
    target_along: float           # target position along wall (metres from start)
    cam_h_above_floor: float      # camera height above floor (metres)
    extent_h: float               # horizontal metres visible at wall surface
    extent_v: float               # vertical metres visible at wall surface


def render_elevation(
    scene: "GSScene",
    candidate: "Candidate",
    wall: Dict[str, Any],
    floor_z: float,
    ceiling_z: float,
    scan_center: Tuple[float, float],
    up_axis: int = 2,
    camera_dist: float = 2.5,
    img_size: int = 800,
) -> Tuple[Image.Image, ElevationParams]:
    """Render a perpendicular (head-on) view of the wall at the candidate.

    Returns the rendered PIL image and the camera parameters needed to
    map normalized image coordinates back to wall-local metres.
    """
    from bim_recon.gs_scene import look_at_pose

    wall_height = ceiling_z - floor_z
    normal = _wall_inward_normal(wall, scan_center)
    wall_dir = _wall_direction(wall)
    wall_start = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
    wall_length = float(wall["length"])

    target_xy = np.array([candidate.world_x, candidate.world_y], dtype=np.float64)
    target_along = float(np.dot(target_xy - wall_start, wall_dir))

    # FOV to cover full wall height + margin
    margin = 0.3  # metres
    fov_rad = 2.0 * math.atan((wall_height / 2.0 + margin) / camera_dist)
    fov_deg = math.degrees(fov_rad)

    # Horizontal / vertical extent at wall surface (fy = fx in fov_to_intrinsics)
    extent_h = 2.0 * camera_dist * math.tan(fov_rad / 2.0)
    extent_v = extent_h  # square image → square extent

    cam_h = wall_height / 2.0  # mid-height above floor

    h_axes = [i for i in range(3) if i != up_axis]

    eye_xy = target_xy + normal * camera_dist
    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = float(eye_xy[0])
    eye[h_axes[1]] = float(eye_xy[1])
    eye[up_axis] = floor_z + cam_h

    tgt = [0.0, 0.0, 0.0]
    tgt[h_axes[0]] = float(target_xy[0])
    tgt[h_axes[1]] = float(target_xy[1])
    tgt[up_axis] = floor_z + cam_h

    up = [0.0, 0.0, 0.0]
    up[up_axis] = 1.0

    pose = look_at_pose(
        (eye[0], eye[1], eye[2]),
        (tgt[0], tgt[1], tgt[2]),
        (up[0], up[1], up[2]),
    )

    result = scene.render(
        pose, width=img_size, height=img_size, fov_degrees=fov_deg,
    )
    img = Image.fromarray(
        (result.colors * 255).clip(0, 255).astype(np.uint8)
    )

    params = ElevationParams(
        camera_dist=camera_dist,
        fov_degrees=fov_deg,
        img_size=img_size,
        wall_length=wall_length,
        wall_dir=wall_dir,
        wall_start=wall_start,
        target_along=target_along,
        cam_h_above_floor=cam_h,
        extent_h=extent_h,
        extent_v=extent_v,
    )
    return img, params


# ---------------------------------------------------------------------------
# Normalized bbox → wall-local metres
# ---------------------------------------------------------------------------

def bbox_to_wall_coords(
    norm_bbox: Dict[str, float],
    params: ElevationParams,
    floor_z: float,
    ceiling_z: float,
) -> Optional[SpatialResult]:
    """Map a normalized [0,1] bbox to wall-local metric coordinates.

    For a perpendicular camera on a flat wall, the mapping is linear:

    * Horizontal: ``along = target_along + (norm_x - 0.5) * extent_h``
    * Vertical:   ``height = cam_h + (0.5 - norm_y) * extent_v``

    (``norm_y`` is inverted because image y increases downward.)

    Args:
        norm_bbox: ``{"x","y","w","h"}`` in [0, 1].
        params: Camera parameters from :func:`render_elevation`.
        floor_z: Floor world Z.
        ceiling_z: Ceiling world Z (for clamping header height).

    Returns:
        :class:`SpatialResult` or ``None`` if bbox is degenerate.
    """
    cx = float(norm_bbox["x"])
    cy = float(norm_bbox["y"])
    bw = float(norm_bbox["w"])
    bh = float(norm_bbox["h"])

    if bw <= 0 or bh <= 0:
        return None

    # Normalized edges
    norm_x1 = cx - bw / 2.0   # left
    norm_x2 = cx + bw / 2.0   # right
    norm_y1 = cy - bh / 2.0   # top (higher on wall)
    norm_y2 = cy + bh / 2.0   # bottom (lower on wall)

    # Map to wall-local metres
    along_left = params.target_along + (norm_x1 - 0.5) * params.extent_h
    along_right = params.target_along + (norm_x2 - 0.5) * params.extent_h
    height_top = params.cam_h_above_floor + (0.5 - norm_y1) * params.extent_v
    height_bot = params.cam_h_above_floor + (0.5 - norm_y2) * params.extent_v

    # Clamp to wall extent
    along_left = max(0.0, min(params.wall_length, along_left))
    along_right = max(0.0, min(params.wall_length, along_right))

    # Actual wall height (ceiling - floor), NOT the rendered extent which includes margin
    wall_height = ceiling_z - floor_z
    sill = max(0.0, min(wall_height, height_bot))
    header = max(sill + 0.01, min(wall_height, height_top))

    if along_right <= along_left or header <= sill:
        return None

    width_m = along_right - along_left
    element_height = header - sill
    t_min = along_left / params.wall_length if params.wall_length > 0 else 0.0
    t_max = along_right / params.wall_length if params.wall_length > 0 else 1.0

    # World XY at element centre
    mid_along = (along_left + along_right) / 2.0
    centre_xy = params.wall_start + params.wall_dir * mid_along

    return SpatialResult(
        sill_height=round(sill, 3),
        header_height=round(header, 3),
        element_height=round(element_height, 3),
        width_m=round(width_m, 3),
        t_min=round(t_min, 4),
        t_max=round(t_max, 4),
        world_x=round(float(centre_xy[0]), 4),
        world_y=round(float(centre_xy[1]), 4),
        confidence=0.0,  # set by caller
        method="",       # set by caller
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_spatial(
    falcon: FalconClient,
    scene: "GSScene",
    candidate: "Candidate",
    wall: Dict[str, Any],
    floor_z: float,
    ceiling_z: float,
    scan_center: Tuple[float, float],
    element_name: str,
    up_axis: int = 2,
    camera_dist: float = 2.5,
    img_size: int = 800,
    save_image_path: Optional[str] = None,
) -> Optional[SpatialResult]:
    """Extract precise spatial extent of a confirmed element via Falcon.

    1. Render a perpendicular elevation view of the wall.
    2. Send to Falcon server for segmentation.
    3. Map the tightest bounding box back to wall-local metres.

    Args:
        falcon: Connected :class:`FalconClient`.
        scene: Renderable 3DGS scene.
        candidate: VLM-confirmed element candidate.
        wall: Wall dict with ``x1, y1, x2, y2, length``.
        floor_z, ceiling_z: Floor/ceiling world Z.
        scan_center: Room centre (for inward normal).
        element_name: Query for Falcon (e.g. ``"window"``).
        up_axis: Up axis (0/1/2).
        camera_dist: Camera distance from wall (metres).
        img_size: Render resolution (square).
        save_image_path: If set, save the elevation render here.

    Returns:
        :class:`SpatialResult` or ``None`` if Falcon finds nothing.
    """
    # 1. Render elevation
    img, params = render_elevation(
        scene, candidate, wall, floor_z, ceiling_z,
        scan_center, up_axis, camera_dist, img_size,
    )

    if save_image_path:
        img.save(save_image_path)

    # 2. Falcon segmentation
    detections = falcon.segment(img, element_name, task="segmentation")

    if not detections:
        return None

    # 2a. Draw overlay for debugging
    if save_image_path:
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)
        w_px, h_px = overlay.size
        for d in detections:
            # Detection bbox (green)
            bx, by, bw, bh = d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"]
            x1 = int((bx - bw / 2) * w_px)
            y1 = int((by - bh / 2) * h_px)
            x2 = int((bx + bw / 2) * w_px)
            y2 = int((by + bh / 2) * h_px)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            # Mask bbox (red) if available
            if d.mask_bbox:
                mx, my, mw, mh = (
                    d.mask_bbox["x"], d.mask_bbox["y"],
                    d.mask_bbox["w"], d.mask_bbox["h"],
                )
                mx1 = int((mx - mw / 2) * w_px)
                my1 = int((my - mh / 2) * h_px)
                mx2 = int((mx + mw / 2) * w_px)
                my2 = int((my + mh / 2) * h_px)
                draw.rectangle([mx1, my1, mx2, my2], outline="red", width=2)
        overlay_path = str(Path(save_image_path).with_suffix("")) + "_overlay.png"
        overlay.save(overlay_path)

    # 3. Pick best detection — largest mask area, or largest bbox
    best = max(
        detections,
        key=lambda d: d.mask_area_ratio if d.mask_area_ratio is not None
        else d.bbox.get("w", 0) * d.bbox.get("h", 0),
    )

    # Use mask_bbox if available (tighter), else detection bbox
    norm_bbox = best.mask_bbox if best.mask_bbox else best.bbox
    method = "falcon_segmentation" if best.mask_bbox is not None else "falcon_detection"

    # 4. Map to wall-local metres
    result = bbox_to_wall_coords(norm_bbox, params, floor_z, ceiling_z)
    if result is None:
        return None

    result.method = method
    result.confidence = round(
        min(1.0, (best.mask_area_ratio or 0.0) * 5.0 + 0.3), 2,
    )

    return result
