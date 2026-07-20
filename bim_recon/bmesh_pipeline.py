"""Deterministic B-class extraction: ElementResult → Falcon cutout + 3D anchor.

This module bridges the pipeline's B-class detections (furniture, etc.) and the
TRELLIS mesh workflow. For each confirmed element:

1. Render a targeted view from the 3DGS scene (front-facing, square 800×800).
2. Run Falcon segmentation on the rendered view using the element's label.
3. Pick the detection whose mask centre is closest to the image centre.
4. Apply the RLE mask as alpha → clean RGBA cutout (TRELLIS input).
5. Backproject the mask's depth median to world coordinates.

The output is a :class:`BClassExtraction` ready to feed into
:class:`~bim_recon.trellis_workflow.ApprovedMeshObject`.

Designed for the Gradio deterministic workflow (⑤) so B-class objects use the
same Falcon → TRELLIS → DirectShape chain as the manual extraction tab.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bim_recon.falcon_client import FalconClient
from bim_recon.gs_scene import GSScene, look_at_pose
from bim_recon.pipeline_api import ElementResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BClassExtraction:
    """One B-class element prepared for TRELLIS mesh generation.

    Attributes:
        element: The source pipeline detection (for element_class, width, etc.).
        cutout_path: Path to a clean RGBA PNG ready for TRELLIS.
        position_3d: (x, y, z) world position of the object centre.
        up_axis: World up axis (0/1/2) — drives TRELLIS axis remap.
        render_path: The rendered RGB view used for segmentation.
        width_m / height_m: Estimated footprint for placement scaling.
        detail: Human-readable status for Gradio logging.
    """

    element: ElementResult
    cutout_path: Path
    position_3d: tuple[float, float, float]
    up_axis: int
    render_path: Path
    width_m: float
    height_m: float
    detail: str


def render_element_front_view(
    scene: GSScene,
    world_x: float,
    world_y: float,
    *,
    scan_center: tuple[float, float] = (0.0, 0.0),
    floor_z: float = 0.0,
    ceiling_z: float = 3.0,
    up_axis: int = 2,
    img_size: int = 800,
    fov: float = 45.0,
    eye_distance: float | None = None,
    name_prefix: str = "bmesh",
    output_dir: Path | None = None,
    index: int = 0,
) -> tuple[Path, np.ndarray, np.ndarray, int, int]:
    """Render a single front-facing view centred on (world_x, world_y).

    Camera is placed along the line from the element toward ``scan_center`` at
    ``eye_distance`` metres (or the distance to the room centre if None), at
    mid-height between floor and ceiling.

    Returns:
        ``(image_path, rgb_array, depth_array, eye_h_axis_0, eye_h_axis_1)``
        where the last two values are the indices of the horizontal axes in
        world coordinates for later backprojection.
    """
    cx, cy = scan_center
    dx = cx - world_x
    dy = cy - world_y
    if eye_distance is None:
        eye_distance = math.sqrt(dx * dx + dy * dy)
        eye_distance = max(1.0, min(eye_distance, 5.0))

    primary_angle = math.atan2(dy, dx)
    h_axes = [i for i in range(3) if i != up_axis]
    eye_h_x = world_x + eye_distance * math.cos(primary_angle)
    eye_h_y = world_y + eye_distance * math.sin(primary_angle)

    mid_z = (floor_z + ceiling_z) / 2.0
    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = eye_h_x
    eye[h_axes[1]] = eye_h_y
    eye[up_axis] = floor_z + mid_z

    target = [0.0, 0.0, 0.0]
    target[h_axes[0]] = world_x
    target[h_axes[1]] = world_y
    target[up_axis] = floor_z + mid_z

    up_vec = [0.0, 0.0, 0.0]
    up_vec[up_axis] = 1.0

    pose = look_at_pose(
        (eye[0], eye[1], eye[2]),
        (target[0], target[1], target[2]),
        up=(up_vec[0], up_vec[1], up_vec[2]),
    )
    render = scene.render(pose, width=img_size, height=img_size, fov_degrees=fov)
    rgb = (render.colors * 255).clip(0, 255).astype(np.uint8)

    if output_dir is None:
        raise ValueError("output_dir is required to persist the render")
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"{name_prefix}_{index:03d}.png"
    Image.fromarray(rgb).save(str(img_path))

    return img_path, rgb, render.depth, h_axes[0], h_axes[1]


def segment_and_cutout(
    falcon: FalconClient,
    rgb: np.ndarray,
    label: str,
    *,
    output_dir: Path,
    name_prefix: str = "bmesh",
    index: int = 0,
    debug: bool = False,
) -> tuple[Path, Any, Any] | None:
    """Run Falcon segmentation and build a clean RGBA cutout.

    Picks the detection whose mask centre is closest to the image centre (a
    front-facing render should place the target near the middle).

    Args:
        falcon: Initialised FalconClient.
        rgb: H×W×3 uint8 RGB image.
        label: Referring expression / class name (e.g. ``"chair"``).
        output_dir: Where to save the cutout.
        name_prefix, index: For naming the output PNG.
        debug: When True, also save an annotated overlay PNG alongside.

    Returns:
        ``(cutout_path, detection, norm_bbox)`` or ``None`` if Falcon produced
        no usable mask.
    """
    pil = Image.fromarray(rgb).convert("RGB")
    try:
        detections = falcon.segment(pil, label, task="segmentation")
    except Exception as exc:
        logger.warning("Falcon segmentation failed for %s: %s", label, exc)
        return None
    if not detections:
        logger.info("Falcon returned no detections for %s", label)
        return None

    best = None
    best_dist = float("inf")
    for det in detections:
        bbox = det.mask_bbox or det.bbox
        if not bbox:
            continue
        dx = bbox["x"] - 0.5
        dy = bbox["y"] - 0.5
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best = det
    if best is None:
        return None

    norm_bbox = best.mask_bbox or best.bbox
    cutout = _build_rgba_cutout(rgb, best, padding=0.08)
    if cutout is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    cutout_path = output_dir / f"{name_prefix}_{index:03d}_cutout.png"
    cutout.save(str(cutout_path))

    if debug:
        overlay = _draw_bbox_overlay(rgb, norm_bbox)
        Image.fromarray(overlay).save(
            str(output_dir / f"{name_prefix}_{index:03d}_overlay.png")
        )

    return cutout_path, best, norm_bbox


def _build_rgba_cutout(
    rgb: np.ndarray, detection, *, padding: float = 0.08,
) -> Image.Image | None:
    """Crop rgb to detection bbox + padding, apply RLE mask as alpha."""
    norm_bbox = detection.mask_bbox or detection.bbox
    if not norm_bbox:
        return None
    h_img, w_img = rgb.shape[:2]
    x0 = max(0, int((norm_bbox["x"] - norm_bbox["w"] / 2 - padding) * w_img))
    y0 = max(0, int((norm_bbox["y"] - norm_bbox["h"] / 2 - padding) * h_img))
    x1 = min(w_img, int((norm_bbox["x"] + norm_bbox["w"] / 2 + padding) * w_img))
    y1 = min(h_img, int((norm_bbox["y"] + norm_bbox["h"] / 2 + padding) * h_img))
    if x1 <= x0 or y1 <= y0:
        return None

    cropped = Image.fromarray(rgb).crop((x0, y0, x1, y1)).convert("RGBA")
    alpha = _rle_to_alpha(detection, x0, y0, x1, y1, cropped.size)
    if alpha is not None:
        cropped.putalpha(alpha)
    return cropped


def _rle_to_alpha(detection, x0: int, y0: int, x1: int, y1: int, crop_size):
    """Decode a Falcon RLE mask to a cropped PIL ``L`` mode alpha channel."""
    import base64

    rle = getattr(detection, "mask_rle", None)
    size = getattr(detection, "mask_size", None)
    if not rle or not size:
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        logger.warning("pycocotools not available; falling back to opaque alpha")
        return None
    counts = base64.b64decode(rle) if isinstance(rle, str) else rle
    try:
        full_mask = mask_utils.decode({"counts": counts, "size": list(size)})
    except Exception as exc:
        logger.warning("RLE decode failed (%s); using opaque alpha", exc)
        return None
    mask_crop = full_mask[y0:y1, x0:x1]
    if mask_crop.shape != (crop_size[1], crop_size[0]):
        return None
    return Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")


def _draw_bbox_overlay(rgb: np.ndarray, norm_bbox: dict) -> np.ndarray:
    """Draw a green rectangle on a copy of rgb for debug overlay."""
    from PIL import ImageDraw

    overlay = rgb.copy()
    h_img, w_img = overlay.shape[:2]
    x0 = int((norm_bbox["x"] - norm_bbox["w"] / 2) * w_img)
    y0 = int((norm_bbox["y"] - norm_bbox["h"] / 2) * h_img)
    x1 = int((norm_bbox["x"] + norm_bbox["w"] / 2) * w_img)
    y1 = int((norm_bbox["y"] + norm_bbox["h"] / 2) * h_img)
    pil = Image.fromarray(overlay)
    ImageDraw.Draw(pil).rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
    return np.array(pil)


def backproject_mask_centre(
    depth: np.ndarray,
    norm_bbox: dict,
    *,
    h_axes: tuple[int, int],
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up_axis: int,
    fov: float,
    img_size: int,
) -> tuple[float, float, float]:
    """Unproject the mask centre pixel + median depth to world coordinates.

    Uses the same look-at convention as :func:`render_element_front_view`.
    """
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_world = np.zeros(3, dtype=np.float64)
    up_world[up_axis] = 1.0

    px = (norm_bbox["x"] + 0.0) * img_size  # bbox x is already centre-x
    py = (norm_bbox["y"] + 0.0) * img_size

    half = img_size / 2.0
    focal = 0.5 * img_size / math.tan(math.radians(fov) / 2.0)

    x0 = max(0, min(img_size - 1, int((norm_bbox["x"] - norm_bbox["w"] / 2) * img_size)))
    x1 = max(0, min(img_size - 1, int((norm_bbox["x"] + norm_bbox["w"] / 2) * img_size)))
    y0 = max(0, min(img_size - 1, int((norm_bbox["y"] - norm_bbox["h"] / 2) * img_size)))
    y1 = max(0, min(img_size - 1, int((norm_bbox["y"] + norm_bbox["h"] / 2) * img_size)))
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0.05]
    d = float(np.median(valid)) if valid.size else float(depth[
        max(0, min(img_size - 1, int(py))),
        max(0, min(img_size - 1, int(px)))
    ])
    if d <= 0.05:
        return (float(eye_arr[0]), float(eye_arr[1]), float(eye_arr[2]))

    x_cam = (px - half) / focal * d
    y_cam = (py - half) / focal * d
    z_cam = d

    forward = target_arr - eye_arr
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)

    world = eye_arr + right * x_cam + down * y_cam + forward * z_cam
    return (float(world[0]), float(world[1]), float(world[2]))


def extract_bclass_element(
    element: ElementResult,
    scene: GSScene,
    falcon: FalconClient,
    *,
    output_dir: Path,
    scan_center: tuple[float, float] = (0.0, 0.0),
    floor_z: float = 0.0,
    ceiling_z: float = 3.0,
    up_axis: int = 2,
    fov: float = 45.0,
    img_size: int = 800,
    default_height_m: float = 1.0,
    debug: bool = False,
) -> BClassExtraction | None:
    """Run the full deterministic chain for one B-class element.

    Returns:
        :class:`BClassExtraction` on success, ``None`` if rendering or
        segmentation produced nothing usable.
    """
    label = element.element_class
    render_path, rgb, depth, ha0, ha1 = render_element_front_view(
        scene, element.world_x, element.world_y,
        scan_center=scan_center,
        floor_z=floor_z, ceiling_z=ceiling_z,
        up_axis=up_axis, img_size=img_size, fov=fov,
        name_prefix=label, output_dir=output_dir,
        index=element.result_index,
    )

    seg = segment_and_cutout(
        falcon, rgb, label,
        output_dir=output_dir / "cutouts",
        name_prefix=label, index=element.result_index,
        debug=debug,
    )
    if seg is None:
        return None
    cutout_path, detection, norm_bbox = seg

    # Estimate eye + target for backprojection (mirror render_element_front_view)
    cx, cy = scan_center
    dx = cx - element.world_x
    dy = cy - element.world_y
    eye_distance = math.sqrt(dx * dx + dy * dy)
    eye_distance = max(1.0, min(eye_distance, 5.0))
    primary_angle = math.atan2(dy, dx)
    h_axes = [ha0, ha1]
    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = element.world_x + eye_distance * math.cos(primary_angle)
    eye[h_axes[1]] = element.world_y + eye_distance * math.sin(primary_angle)
    eye[up_axis] = floor_z + (floor_z + ceiling_z) / 2.0
    target = [0.0, 0.0, 0.0]
    target[h_axes[0]] = element.world_x
    target[h_axes[1]] = element.world_y
    target[up_axis] = floor_z + (floor_z + ceiling_z) / 2.0

    position_3d = backproject_mask_centre(
        depth, norm_bbox,
        h_axes=(ha0, ha1),
        eye=(eye[0], eye[1], eye[2]),
        target=(target[0], target[1], target[2]),
        up_axis=up_axis,
        fov=fov, img_size=img_size,
    )

    hd = element.height_detection or {}
    width_m = float(hd.get("width_m") or max(norm_bbox["w"] * 5.0, 0.3))
    height_m = float(hd.get("element_height") or default_height_m)

    return BClassExtraction(
        element=element,
        cutout_path=cutout_path,
        position_3d=position_3d,
        up_axis=up_axis,
        render_path=render_path,
        width_m=width_m,
        height_m=height_m,
        detail=f"Falcon segmented {label} ({norm_bbox['w']:.2f}×{norm_bbox['h']:.2f})",
    )


__all__ = [
    "BClassExtraction",
    "render_element_front_view",
    "segment_and_cutout",
    "backproject_mask_centre",
    "extract_bclass_element",
]
