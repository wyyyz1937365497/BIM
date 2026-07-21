# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene validation and JSON export for B-class pose annotations."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector

SCHEMA_VERSION = 1
ROLE_KEY = "bim_role"
ROLE_GS_REFERENCE = "gs_reference"
ROLE_TRELLIS_PROXY = "trellis_gt_proxy"
ROLE_OBSERVATION_CAMERA = "observation_camera"


class AnnotationError(RuntimeError):
    """Raised when a scene cannot produce a reliable annotation manifest."""


def _matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _vector_values(vector: Vector) -> list[float]:
    return [float(value) for value in vector]


def _resolved_path(value: str | None) -> str:
    if not value:
        return ""
    return str(Path(bpy.path.abspath(str(value))).resolve())
def _scene_value(scene: bpy.types.Scene, key: str, default: Any) -> Any:
    """Read an RNA property first while preserving old custom-property files."""
    value = scene.get(key)
    if value is not None:
        return value
    return getattr(scene, key, default)



def _objects_with_role(scene: bpy.types.Scene, role: str) -> list[bpy.types.Object]:
    return [obj for obj in scene.objects if obj.get(ROLE_KEY) == role]


def _single_gs_reference(scene: bpy.types.Scene) -> bpy.types.Object:
    references = _objects_with_role(scene, ROLE_GS_REFERENCE)
    if len(references) != 1:
        raise AnnotationError(
            f"Expected exactly one {ROLE_GS_REFERENCE!r} object, found {len(references)}"
        )
    return references[0]


def _mesh_bbox_center(obj: bpy.types.Object) -> Vector:
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        raise AnnotationError(f"Proxy {obj.name!r} must be a non-empty mesh")
    first = obj.data.vertices[0].co.copy()
    minimum = first.copy()
    maximum = first.copy()
    for vertex in obj.data.vertices[1:]:
        coordinate = vertex.co
        minimum.x = min(minimum.x, coordinate.x)
        minimum.y = min(minimum.y, coordinate.y)
        minimum.z = min(minimum.z, coordinate.z)
        maximum.x = max(maximum.x, coordinate.x)
        maximum.y = max(maximum.y, coordinate.y)
        maximum.z = max(maximum.z, coordinate.z)
    return (minimum + maximum) * 0.5


def _decompose_uniform_transform(
    matrix: Matrix,
    *,
    tolerance: float = 1.0e-4,
) -> tuple[float, Matrix, Vector]:
    linear = matrix.to_3x3()
    columns = [
        Vector((linear[0][index], linear[1][index], linear[2][index]))
        for index in range(3)
    ]
    lengths = [column.length for column in columns]
    scale = sum(lengths) / 3.0
    if not math.isfinite(scale) or scale <= 1.0e-9:
        raise AnnotationError("Transform has a zero or invalid scale")
    if max(abs(length - scale) for length in lengths) > tolerance * max(scale, 1.0):
        raise AnnotationError(
            "Transform contains non-uniform scale; apply a single uniform scale only"
        )

    normalized = [column / scale for column in columns]
    for left in range(3):
        for right in range(left + 1, 3):
            if abs(normalized[left].dot(normalized[right])) > tolerance:
                raise AnnotationError("Transform contains shear")

    rotation = Matrix((
        (normalized[0].x, normalized[1].x, normalized[2].x),
        (normalized[0].y, normalized[1].y, normalized[2].y),
        (normalized[0].z, normalized[1].z, normalized[2].z),
    ))
    determinant = rotation.determinant()
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise AnnotationError("Transform contains reflection or an invalid rotation")
    if abs(determinant - 1.0) > 5.0 * tolerance:
        raise AnnotationError(f"Rotation determinant is {determinant:.6f}, expected 1")
    return float(scale), rotation, matrix.translation.copy()


def _camera_payload(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    gs_to_blender: Matrix,
) -> dict[str, Any]:
    camera_to_gs = gs_to_blender.inverted_safe() @ camera.matrix_world
    eye = camera_to_gs.translation
    rotation = camera_to_gs.to_3x3()
    forward = (rotation @ Vector((0.0, 0.0, -1.0))).normalized()
    up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    target = eye + forward
    render = scene.render
    scale = float(render.resolution_percentage) / 100.0
    width = max(1, int(round(render.resolution_x * scale)))
    height = max(1, int(round(render.resolution_y * scale)))
    bbox = camera.get("bim_norm_bbox", (0.5, 0.5, 0.0, 0.0))
    return {
        "view_id": str(camera.get("bim_view_id", camera.name)),
        "blender_object": camera.name,
        "target_object_id": str(camera.get("bim_target_object_id", "")),
        "rgb_path": _resolved_path(camera.get("bim_rgb_path", "")),
        "depth_path": _resolved_path(camera.get("bim_depth_path", "")),
        "mask_path": _resolved_path(camera.get("bim_mask_path", "")),
        "norm_bbox": [float(value) for value in bbox],
        "camera_to_gs": _matrix_rows(camera_to_gs),
        "camera_eye": _vector_values(eye),
        "camera_target": _vector_values(target),
        "camera_up": _vector_values(up),
        "vertical_fov_deg": math.degrees(float(camera.data.angle_y)),
        "image_size": [width, height],
    }


def _proxy_payload(
    scene: bpy.types.Scene,
    proxy: bpy.types.Object,
    gs_reference: bpy.types.Object,
    cameras: list[bpy.types.Object],
) -> dict[str, Any]:
    gs_to_blender = gs_reference.matrix_world.copy()
    trellis_to_blender = proxy.matrix_world.copy()
    trellis_to_gs = gs_to_blender.inverted_safe() @ trellis_to_blender
    scale, rotation, matrix_translation = _decompose_uniform_transform(trellis_to_gs)
    mesh_center = _mesh_bbox_center(proxy)
    centered_translation = trellis_to_gs @ mesh_center
    object_id = str(proxy.get("bim_object_id", ""))
    observations = [
        _camera_payload(scene, camera, gs_to_blender)
        for camera in cameras
        if str(camera.get("bim_target_object_id", "")) == object_id
    ]
    return {
        "scene_id": str(_scene_value(scene, "bim_scene_id", "")),
        "object_id": object_id,
        "class_name": str(proxy.get("bim_class_name", "")),
        "trellis": {
            "variant_id": str(proxy.get("bim_variant_id", "")),
            "source_view_id": str(proxy.get("bim_source_view_id", "")),
            "source_glb": _resolved_path(proxy.get("bim_source_glb", "")),
            "proxy_source": _resolved_path(proxy.get("bim_proxy_source", "")),
            "proxy_object": proxy.name,
            "raw_to_blender": _matrix_rows(trellis_to_blender),
        },
        "gs": {
            "source_ply": _resolved_path(gs_reference.get("bim_source_ply", "")),
            "blender_object": gs_reference.name,
            "raw_to_blender": _matrix_rows(gs_to_blender),
            "up_axis": int(_scene_value(scene, "bim_up_axis", 2)),
            "metric_scale_validated": bool(
                gs_reference.get("bim_metric_scale_validated", False)
            ),
        },
        "ground_truth": {
            "raw_trellis_to_gs": _matrix_rows(trellis_to_gs),
            "mesh_center": _vector_values(mesh_center),
            "scale": scale,
            "rotation": _matrix_rows(rotation),
            "matrix_translation": _vector_values(matrix_translation),
            "translation": _vector_values(centered_translation),
            "formula": "x_gs = scale * rotation @ (x_trellis - mesh_center) + translation",
        },
        "annotation": {
            "status": str(proxy.get("bim_annotation_status", "draft")),
            "quality": float(proxy.get("bim_annotation_quality", 0.0)),
            "notes": str(proxy.get("bim_annotation_notes", "")),
        },
        "observation_views": observations,
    }


def validate_scene(scene: bpy.types.Scene) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking warnings for the active scene."""
    errors: list[str] = []
    warnings: list[str] = []
    scene_id = str(_scene_value(scene, "bim_scene_id", "")).strip()
    if not scene_id:
        errors.append("Scene ID is empty")

    references = _objects_with_role(scene, ROLE_GS_REFERENCE)
    if len(references) != 1:
        errors.append(f"Expected one GS reference, found {len(references)}")
    elif not references[0].get("bim_source_ply"):
        warnings.append("GS reference has no source PLY path")

    proxies = _objects_with_role(scene, ROLE_TRELLIS_PROXY)
    if not proxies:
        errors.append("No TRELLIS GT proxy is marked")
    seen_variants: set[tuple[str, str]] = set()
    object_ids: set[str] = set()
    for proxy in proxies:
        object_id = str(proxy.get("bim_object_id", "")).strip()
        class_name = str(proxy.get("bim_class_name", "")).strip()
        variant_id = str(proxy.get("bim_variant_id", "")).strip()
        source_glb = str(proxy.get("bim_source_glb", "")).strip()
        if not object_id:
            errors.append(f"Proxy {proxy.name!r} has no object ID")
        if not class_name:
            errors.append(f"Proxy {proxy.name!r} has no class name")
        if not variant_id:
            errors.append(f"Proxy {proxy.name!r} has no variant ID")
        if not source_glb:
            errors.append(f"Proxy {proxy.name!r} has no source GLB")
        elif not Path(_resolved_path(source_glb)).is_file():
            warnings.append(f"Proxy {proxy.name!r} source GLB does not currently exist")
        key = (object_id, variant_id)
        if key in seen_variants:
            errors.append(f"Duplicate object/variant pair: {key}")
        seen_variants.add(key)
        object_ids.add(object_id)
        try:
            _mesh_bbox_center(proxy)
            if len(references) == 1:
                transform = references[0].matrix_world.inverted_safe() @ proxy.matrix_world
                _decompose_uniform_transform(transform)
        except AnnotationError as exc:
            errors.append(f"Proxy {proxy.name!r}: {exc}")

    cameras = _objects_with_role(scene, ROLE_OBSERVATION_CAMERA)
    for camera in cameras:
        target = str(camera.get("bim_target_object_id", "")).strip()
        if camera.type != "CAMERA":
            errors.append(f"Observation {camera.name!r} is not a camera")
        if target not in object_ids:
            errors.append(
                f"Camera {camera.name!r} targets unknown object ID {target!r}"
            )
        for key, label in (
            ("bim_rgb_path", "RGB"),
            ("bim_depth_path", "depth"),
            ("bim_mask_path", "mask"),
        ):
            value = str(camera.get(key, "")).strip()
            if not value:
                warnings.append(f"Camera {camera.name!r} has no {label} path")
    return errors, warnings


def build_scene_manifest(scene: bpy.types.Scene, *, strict: bool = True) -> dict[str, Any]:
    """Build a JSON-safe annotation manifest from marked scene objects."""
    errors, warnings = validate_scene(scene)
    if strict and errors:
        raise AnnotationError("; ".join(errors))

    references = _objects_with_role(scene, ROLE_GS_REFERENCE)
    proxies = _objects_with_role(scene, ROLE_TRELLIS_PROXY)
    cameras = _objects_with_role(scene, ROLE_OBSERVATION_CAMERA)
    annotations: list[dict[str, Any]] = []
    if len(references) == 1:
        gs_reference = references[0]
        for proxy in proxies:
            try:
                annotations.append(
                    _proxy_payload(scene, proxy, gs_reference, cameras)
                )
            except AnnotationError as exc:
                if strict:
                    raise
                warnings.append(f"Skipped proxy {proxy.name!r}: {exc}")

    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "blend_file": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else "",
        "scene_id": str(_scene_value(scene, "bim_scene_id", "")),
        "validation": {"errors": errors, "warnings": warnings},
        "annotations": annotations,
    }


def export_scene_manifest(
    scene: bpy.types.Scene,
    output_path: str | Path,
    *,
    strict: bool = True,
) -> Path:
    """Validate and write the active scene annotation manifest."""
    path = Path(bpy.path.abspath(str(output_path))).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_scene_manifest(scene, strict=strict)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
