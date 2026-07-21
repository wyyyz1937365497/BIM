"""Focused TRELLIS mesh generation and registration helpers.

This module keeps the dedicated Gradio page thin while reusing the project's
existing deterministic mesh registrar. Registration is analysis-by-synthesis:
TRELLIS mesh silhouettes are projected through the captured camera and searched
for the yaw with the highest IoU against the transparent object cutout. The
result is an auditable placement manifest, not an opaque UI-only adjustment.
"""
from __future__ import annotations

import json
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
    """Load the cutout alpha mask; opaque cropped objects are valid too."""
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


def register_mesh(
    glb_path: str | Path,
    cutout_path: str | Path,
    output_dir: str | Path,
    inputs: RegistrationInputs,
    *,
    name: str = "trellis_registration",
) -> dict[str, Any]:
    """Automatically solve yaw, compute placement, and write a manifest."""
    glb = Path(glb_path)
    cutout = Path(cutout_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not glb.is_file():
        raise FileNotFoundError(glb)
    if not cutout.is_file():
        raise FileNotFoundError(cutout)
    if inputs.up_axis not in (0, 1, 2):
        raise ValueError("up_axis 必须是 0、1 或 2")
    if inputs.element_width_m <= 0 or inputs.element_height_m <= 0:
        raise ValueError("物体宽度和高度必须大于 0")

    alpha = _rgba_alpha(cutout)
    width, height = inputs.image_size
    if width <= 0 or height <= 0:
        raise ValueError("相机图像尺寸必须大于 0")
    if alpha.shape != (height, width):
        raise ValueError(
            f"cutout 尺寸 {alpha.shape[1]}x{alpha.shape[0]} 与相机图像尺寸 "
            f"{width}x{height} 不一致"
        )

    bbox = {
        "x": float(inputs.bbox[0]),
        "y": float(inputs.bbox[1]),
        "w": float(inputs.bbox[2]),
        "h": float(inputs.bbox[3]),
    }
    yaw_result = find_best_yaw_silhouette(
        glb_path=glb,
        cutout_alpha=alpha,
        norm_bbox=bbox,
        camera_eye=inputs.camera_eye,
        camera_target=inputs.camera_target,
        camera_up_axis=inputs.up_axis,
        camera_fov=float(inputs.camera_fov_deg),
        camera_img_w=width,
        camera_img_h=height,
        world_pos=inputs.world_position,
        element_width_m=float(inputs.element_width_m),
        up_axis=inputs.up_axis,
        debug_dir=destination / "yaw_debug",
    )

    h_axes = [axis for axis in range(3) if axis != inputs.up_axis]
    placement = MeshPlacement(
        glb_path=glb,
        world_x=float(inputs.world_position[h_axes[0]]),
        world_y=float(inputs.world_position[h_axes[1]]),
        floor_z=float(inputs.floor_z),
        ceiling_z=float(inputs.ceiling_z),
        element_width_m=float(inputs.element_width_m),
        element_height_m=float(inputs.element_height_m),
        up_axis=inputs.up_axis,
        yaw_degrees=float(yaw_result["best_yaw"]),
        name=safe_stem(name),
    )
    transform = compute_placement_transform(placement)
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "silhouette_iou_yaw_search",
        "glb_path": str(glb.resolve()),
        "cutout_path": str(cutout.resolve()),
        "registration": {
            "inputs": {
                "world_position": list(inputs.world_position),
                "floor_z": inputs.floor_z,
                "ceiling_z": inputs.ceiling_z,
                "element_width_m": inputs.element_width_m,
                "element_height_m": inputs.element_height_m,
                "camera_eye": list(inputs.camera_eye),
                "camera_target": list(inputs.camera_target),
                "camera_fov_deg": inputs.camera_fov_deg,
                "image_size": list(inputs.image_size),
                "up_axis": inputs.up_axis,
                "bbox": list(inputs.bbox),
            },
            "yaw_search": yaw_result,
            "placement": serialize_placement_diagnostics(placement, transform),
        },
    }
    manifest_path = destination / f"{safe_stem(name, 'registration')}_registration.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


__all__ = ["RegistrationInputs", "generate_mesh", "register_mesh", "safe_stem"]
