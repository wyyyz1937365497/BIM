"""Focused B-class registration pipeline shared by the dedicated Gradio page.

The implementation intentionally mirrors the proven main-page path:

3DGS viewer camera → high-resolution 3DGS RGB/depth render → rough brush box
→ VLM referring expression → Falcon instance mask → RGBA cutout → TRELLIS GLB
→ depth backprojection for world position and dimensions → silhouette yaw search
→ deterministic mesh placement → auditable dataset manifest.

No Revit calls are made here. The output is the registration foundation for
the constrained render-compare alignment, with every intermediate asset retained.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bim_recon.mesh_registrar import (
    MeshPlacement,
    compute_placement_transform,
    find_best_yaw_silhouette,
    serialize_placement_diagnostics,
)
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest


@dataclass(frozen=True, slots=True)
class RegistrationInputs:
    """World and camera metadata for one automatic GLB registration."""

    world_position: tuple[float, float, float]
    floor_z: float
    ceiling_z: float
    element_width_m: float
    element_height_m: float
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    camera_fov_deg: float
    image_size: tuple[int, int]
    up_axis: int = 2
    bbox: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0)


def safe_stem(value: str, default: str = "trellis_object") -> str:
    """Return a filesystem-safe, human-readable artifact stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return stem[:96] or default


def _rgba_alpha(image_path: Path) -> np.ndarray:
    """Load a cutout alpha mask; opaque cropped objects are valid too."""
    with Image.open(image_path) as image:
        alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
    if int(np.count_nonzero(alpha)) == 0:
        raise ValueError("输入图像的 alpha 通道为空")
    return alpha


def generate_mesh(
    client: TrellisClient,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    name: str = "trellis_object",
    seed: int = 1,
    simplify: float = 0.95,
    texture_size: int = 1024,
):
    """Generate one TRELLIS mesh through the configured HTTP bridge."""
    source = Path(image_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not client.health():
        raise RuntimeError("TRELLIS 服务不可达或模型尚未加载")
    return client.generate_mesh(TrellisMeshRequest(
        image_path=source,
        output_dir=destination,
        name=safe_stem(name),
        seed=int(seed),
        simplify=float(simplify),
        texture_size=int(texture_size),
    ))


def _unproject_center(
    depth: np.ndarray,
    bbox: dict[str, float],
    camera: dict[str, Any],
) -> tuple[float, float, float]:
    """Backproject the mask-center median depth using the viewer camera."""
    height, width = depth.shape[:2]
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up_raw = camera.get("up")
    if up_raw is not None:
        up = np.asarray(up_raw, dtype=np.float64)
    else:
        up_axis = int(camera.get("up_axis", 2))
        up = np.zeros(3, dtype=np.float64)
        up[up_axis] = 1.0
    x = float(bbox["x"]) * width
    y = float(bbox["y"]) * height
    x0 = max(0, min(width - 1, int((bbox["x"] - bbox["w"] / 2) * width)))
    x1 = max(x0 + 1, min(width, int((bbox["x"] + bbox["w"] / 2) * width)))
    y0 = max(0, min(height - 1, int((bbox["y"] - bbox["h"] / 2) * height)))
    y1 = max(y0 + 1, min(height, int((bbox["y"] + bbox["h"] / 2) * height)))
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0.05]
    if valid.size:
        distance = float(np.median(valid))
    else:
        distance = float(depth[min(height - 1, int(y)), min(width - 1, int(x))])
    if distance <= 0.05:
        raise ValueError("目标 bbox 内没有有效 3DGS 深度")

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    hfov = math.radians(float(camera.get("fov_degrees", camera.get("fov", 45.0))))
    # FOV is horizontal (matches gs_scene.fov_to_intrinsics): focal from width.
    focal_x = 0.5 * width / math.tan(hfov / 2.0)
    focal_y = focal_x
    x_cam = (x - width / 2.0) / focal_x * distance
    y_cam = (y - height / 2.0) / focal_y * distance
    world = eye + right * x_cam + down * y_cam + forward * distance
    return tuple(float(value) for value in world)


def _estimate_dimensions(
    bbox: dict[str, float],
    distance: float,
    image_size: tuple[int, int],
    fov_deg: float,
) -> tuple[float, float]:
    """Convert normalized mask extent and metric depth into meters."""
    width, height = image_size
    hfov = math.radians(float(fov_deg))
    aspect = width / max(height, 1)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
    physical_width = float(bbox["w"]) * 2.0 * distance * math.tan(hfov / 2.0)
    physical_height = float(bbox["h"]) * 2.0 * distance * math.tan(vfov / 2.0)
    return max(physical_width, 0.1), max(physical_height, 0.1)


def _unproject_mask_points(
    depth: np.ndarray,
    mask: np.ndarray,
    camera: dict[str, Any],
    *,
    max_points: int = 12_000,
) -> np.ndarray:
    """Vectorize valid Falcon-mask pixels into sampled 3DGS world points."""
    if mask.shape != depth.shape:
        raise ValueError("Falcon mask/depth dimensions do not match")
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.05)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if rows.size > max_points:
        indices = np.linspace(0, rows.size - 1, max_points, dtype=np.int64)
        rows, cols = rows[indices], cols[indices]

    distances = depth[rows, cols].astype(np.float64)
    height, width = depth.shape
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up_raw = camera.get("up")
    if up_raw is not None:
        up = np.asarray(up_raw, dtype=np.float64)
    else:
        up_axis = int(camera.get("up_axis", 2))
        up = np.zeros(3, dtype=np.float64)
        up[up_axis] = 1.0
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    hfov = math.radians(
        float(camera.get("fov_degrees", camera.get("fov", 45.0)))
    )
    # FOV is horizontal (matches gs_scene.fov_to_intrinsics): focal from width.
    focal_x = 0.5 * width / math.tan(hfov / 2.0)
    focal_y = focal_x
    x_camera = (cols.astype(np.float64) - width / 2.0) / focal_x * distances
    y_camera = (rows.astype(np.float64) - height / 2.0) / focal_y * distances
    return (
        eye[None, :]
        + right[None, :] * x_camera[:, None]
        + down[None, :] * y_camera[:, None]
        + forward[None, :] * distances[:, None]
    )


def _robust_anchor_from_points(
    world_points: np.ndarray, eye: np.ndarray
) -> np.ndarray | None:
    """Robust visible-surface anchor: median of the [10, 70] camera-distance
    percentile band of mask surface points.

    Unlike the bbox-center + median-depth anchor, this rejects background bleed
    and leg/seat gaps that contaminate the bbox median, yielding a stable point
    on the object's visible surface. Returns ``None`` when no points survive.
    """
    if world_points.size == 0:
        return None
    distances = np.linalg.norm(world_points - eye[None, :], axis=1)
    lo, hi = np.percentile(distances, [10.0, 70.0])
    keep = (distances >= lo) & (distances <= hi)
    if not np.any(keep):
        keep = distances <= hi
    if not np.any(keep):
        return None
    return np.asarray(np.median(world_points[keep], axis=0), dtype=np.float64)


def robust_mask_anchor(
    depth: np.ndarray, mask: np.ndarray, camera: dict[str, Any]
) -> np.ndarray | None:
    """Public robust visible-surface anchor from a Falcon mask + 3DGS depth.

    Unprojects the mask pixels and takes the median of the [10, 70] camera-
    distance percentile band. Shared by the registration lab and the
    render-compare optimizer. Returns ``None`` when the mask is empty."""
    world_points = _unproject_mask_points(depth, mask, camera)
    eye = np.asarray(camera["eye"], dtype=np.float64)
    return _robust_anchor_from_points(world_points, eye)


def backproject_observation(
    render_state: dict[str, Any],
    detection_info: dict[str, Any],
) -> dict[str, Any]:
    """Backproject one segmented observation without requiring a GLB.

    Returns JSON-safe camera, world-position, distance, and estimated-size data
    shared by the Gradio observation radar and final mesh registration.
    """
    camera = dict(render_state["cam"])
    depth = np.asarray(render_state["depth"], dtype=np.float32)
    rgb = np.asarray(render_state["rgb"], dtype=np.uint8)
    bbox = {
        key: float(value)
        for key, value in detection_info["norm_bbox"].items()
    }
    image_size = (int(render_state["width"]), int(render_state["height"]))
    expected_shape = (image_size[1], image_size[0])
    if rgb.shape[:2] != expected_shape or depth.shape != expected_shape:
        raise ValueError("RGB/depth/render dimensions do not match")

    world_position = _unproject_center(depth, bbox, camera)
    camera_eye = np.asarray(camera["eye"], dtype=np.float64)
    distance = float(
        np.linalg.norm(np.asarray(world_position, dtype=np.float64) - camera_eye)
    )
    fov = float(camera.get("fov_degrees", camera.get("fov", 45.0)))
    element_width_m, element_height_m = _estimate_dimensions(
        bbox, distance, image_size, fov
    )
    up_axis = int(camera.get("up_axis", 2))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    mask_points_xy: list[list[float]] = []
    visible_anchor: np.ndarray | None = None
    mask_path_value = detection_info.get("mask_path") or render_state.get("mask_path")
    if mask_path_value:
        mask_path = Path(mask_path_value)
        if mask_path.is_file():
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            world_points = _unproject_mask_points(depth, mask, camera)
            if world_points.size:
                mask_points_xy = world_points[:, horizontal_axes].tolist()
                visible_anchor = _robust_anchor_from_points(world_points, camera_eye)
    radar_anchor = (visible_anchor if visible_anchor is not None
                    else np.asarray(world_position, dtype=np.float64))
    return {
        "camera": camera,
        "image_size": list(image_size),
        "norm_bbox": bbox,
        "backprojected_center": list(world_position),
        "visible_anchor": list(visible_anchor) if visible_anchor is not None else list(world_position),
        "horizontal_position": [
            float(radar_anchor[horizontal_axes[0]]),
            float(radar_anchor[horizontal_axes[1]]),
        ],
        "camera_horizontal_position": [
            float(camera_eye[horizontal_axes[0]]),
            float(camera_eye[horizontal_axes[1]]),
        ],
        "up_axis": up_axis,
        "mask_points_horizontal": mask_points_xy,
        "distance_m": distance,
        "estimated_dimensions_m": {
            "width": element_width_m,
            "height": element_height_m,
        },
    }


def capture_viewer_observation(
    scene_name: str,
    viewer_session: dict[str, Any],
    output_dir: str | Path,
    *,
    width: int = 2048,
    height: int = 1536,
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Capture the live viewer camera and persist an RGB-D 3DGS observation."""
    from bim_recon.gradio_helpers import fetch_camera_state, _get_scene
    from bim_recon.gs_scene import look_at_pose

    status, camera_data = fetch_camera_state(viewer_session)
    if not camera_data:
        raise RuntimeError(status)
    scene = _get_scene(scene_name)
    eye = tuple(float(value) for value in camera_data["position"])
    target = tuple(float(value) for value in camera_data["look_at"])
    up = tuple(float(value) for value in camera_data.get("up", [0.0, 0.0, 1.0]))
    fov = float(camera_data.get("fov_degrees", 45.0))
    pose = look_at_pose(eye, target, up=up)
    render = scene.render(pose, width=int(width), height=int(height), fov_degrees=fov)
    rgb = (np.clip(render.colors, 0.0, 1.0) * 255).astype(np.uint8)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rgb_path = destination / "00_observation_rgb.png"
    depth_path = destination / "00_observation_depth.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, render.depth.astype(np.float32))
    state = {
        "rgb": rgb,
        "depth": render.depth.astype(np.float32),
        "cam": {"eye": eye, "target": target, "up": up, "fov_degrees": fov,
                "up_axis": int(camera_data.get("up_axis", 2))},
        "scene": scene_name, "width": int(width), "height": int(height),
        "rgb_path": str(rgb_path), "depth_path": str(depth_path),
    }
    return rgb, state, f"✅ 已捕获 3DGS RGB-D 观测 ({width}×{height})"


def extract_observation_from_editor(
    mask_editor_value: dict[str, Any] | None,
    render_state: dict[str, Any] | None,
    *,
    output_dir: str | Path,
    vlm_caller,
    falcon_client,
) -> tuple[str, np.ndarray | None, np.ndarray | None, str | None, dict[str, Any] | None, str]:
    """Run brush bbox → VLM referring expression → Falcon cutout."""
    from bim_recon.bmesh_extractor import (
        _extract_user_bbox,
        _select_detection,
        classify_and_segment_from_mask_editor,
    )
    if not render_state:
        return "", None, None, None, None, "⚠️ 请先捕获 3DGS 视角"
    destination = Path(output_dir)
    debug_dir = destination / "segmentation_debug"
    result = classify_and_segment_from_mask_editor(
        mask_editor_value, vlm_caller, falcon_client, debug_dir=debug_dir,
    )
    if result.cutout is None:
        return result.label, result.overlay, None, None, None, result.detail
    extracted = _extract_user_bbox(mask_editor_value)
    detection_info = None
    if extracted and falcon_client is not None:
        base_rgb, user_bbox = extracted
        detections = falcon_client.segment(Image.fromarray(base_rgb), result.label, task="segmentation")
        selected = _select_detection(detections, user_bbox, base_rgb.shape[1], base_rgb.shape[0])
        if selected is not None:
            detection_info = {"norm_bbox": dict(selected.mask_bbox or selected.bbox),
                              "mask_area_ratio": selected.mask_area_ratio}
            render_state["mask_path"] = str(destination / "segmentation_debug" / "03_falcon_mask.png")
            try:
                from bim_recon.bmesh_pipeline import _full_frame_mask
                full_mask = _full_frame_mask(selected, base_rgb.shape[0], base_rgb.shape[1])
                if full_mask is not None:
                    mask_path = destination / "segmentation_debug" / "03_falcon_mask.png"
                    Image.fromarray(full_mask, mode="L").save(mask_path)
                    render_state["mask_path"] = str(mask_path)
                    detection_info["mask_path"] = str(mask_path)
            except Exception:
                pass
    cutout_path = debug_dir / "04_cutout.png"
    if not cutout_path.is_file():
        cutout_path = destination / "04_cutout.png"
        result.cutout.save(cutout_path)
    return result.label, result.overlay, np.asarray(result.cutout), str(cutout_path), detection_info, result.detail


def register_observation(
    glb_path: str | Path,
    cutout_path: str | Path,
    render_state: dict[str, Any],
    detection_info: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str,
    label: str,
    floor_z: float = 0.0,
    ceiling_z: float = 3.0,
    yaw_override: float | None = None,
) -> dict[str, Any]:
    """Run the full observed RGB-D → GLB placement path and write a manifest."""
    glb = Path(glb_path)
    cutout = Path(cutout_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not glb.is_file():
        raise FileNotFoundError(glb)
    if not cutout.is_file():
        raise FileNotFoundError(cutout)

    observation = backproject_observation(render_state, detection_info)
    camera = observation["camera"]
    bbox = observation["norm_bbox"]
    image_size = tuple(observation["image_size"])
    backprojected_center = tuple(observation["backprojected_center"])
    world_position = tuple(observation.get("visible_anchor") or observation["backprojected_center"])
    distance = float(observation["distance_m"])
    element_width_m = float(observation["estimated_dimensions_m"]["width"])
    element_height_m = float(observation["estimated_dimensions_m"]["height"])
    up_axis = int(observation["up_axis"])
    h_axes = [axis for axis in range(3) if axis != up_axis]
    alpha = _rgba_alpha(cutout)
    yaw_result = find_best_yaw_silhouette(
        glb_path=glb,
        cutout_alpha=alpha,
        norm_bbox=bbox,
        camera_eye=tuple(float(v) for v in camera["eye"]),
        camera_target=tuple(float(v) for v in camera["target"]),
        camera_up_axis=up_axis,
        camera_fov=float(camera.get("fov_degrees", camera.get("fov", 45.0))),
        camera_img_w=image_size[0],
        camera_img_h=image_size[1],
        world_pos=world_position,
        element_width_m=element_width_m,
        up_axis=up_axis,
        debug_dir=destination / "yaw_debug",
    )
    resolved_yaw = float(yaw_override) if yaw_override is not None else float(yaw_result["best_yaw"])
    placement = MeshPlacement(
        glb_path=glb,
        world_x=float(world_position[h_axes[0]]),
        world_y=float(world_position[h_axes[1]]),
        floor_z=float(floor_z),
        ceiling_z=float(ceiling_z),
        element_width_m=element_width_m,
        element_height_m=element_height_m,
        up_axis=up_axis,
        yaw_degrees=resolved_yaw,
        category="OST_GenericModel",
        name=safe_stem(name),
    )
    transform = compute_placement_transform(placement)
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "object": {"label": label, "name": name},
        "assets": {
            "glb": str(glb.resolve()),
            "cutout": str(cutout.resolve()),
            "rgb": str(Path(render_state.get("rgb_path", "")).resolve()) if render_state.get("rgb_path") else "",
            "depth": str(Path(render_state.get("depth_path", "")).resolve()) if render_state.get("depth_path") else "",
            "mask": str(Path(detection_info.get("mask_path") or render_state.get("mask_path", "")).resolve()) if detection_info.get("mask_path") or render_state.get("mask_path") else "",
        },
        "observation": {
            "camera": camera,
            "image_size": list(image_size),
            "norm_bbox": bbox,
            "backprojected_center": list(backprojected_center),
            "visible_anchor": list(world_position),
            "distance_m": distance,
            "estimated_dimensions_m": {
                "width": element_width_m,
                "height": element_height_m,
            },
        },
        "registration": {
            "method": "depth_backprojection_plus_silhouette_yaw_search",
            "yaw_search": yaw_result,
            "resolved_yaw_degrees": resolved_yaw,
            "placement": serialize_placement_diagnostics(placement, transform),
        },
    }
    manifest_path = destination / f"{safe_stem(name, 'registration')}_registration.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def register_mesh(
    glb_path: str | Path,
    cutout_path: str | Path,
    output_dir: str | Path,
    inputs: RegistrationInputs,
    *,
    name: str = "trellis_registration",
) -> dict[str, Any]:
    """Compatibility wrapper for the parameter-driven registration tab."""
    return register_observation(
        glb_path,
        cutout_path,
        {
            "cam": {
                "eye": list(inputs.camera_eye),
                "target": list(inputs.camera_target),
                "fov_degrees": inputs.camera_fov_deg,
                "up_axis": inputs.up_axis,
            },
            "depth": np.ones((inputs.image_size[1], inputs.image_size[0]), dtype=np.float32),
            "rgb": np.zeros((inputs.image_size[1], inputs.image_size[0], 3), dtype=np.uint8),
            "width": inputs.image_size[0],
            "height": inputs.image_size[1],
        },
        {"norm_bbox": {"x": inputs.bbox[0], "y": inputs.bbox[1], "w": inputs.bbox[2], "h": inputs.bbox[3]}},
        output_dir,
        name=name,
        label="object",
        floor_z=inputs.floor_z,
        ceiling_z=inputs.ceiling_z,
    )


__all__ = [
    "RegistrationInputs",
    "capture_viewer_observation",
    "extract_observation_from_editor",
    "generate_mesh",
    "register_mesh",
    "register_observation",
    "safe_stem",
]
