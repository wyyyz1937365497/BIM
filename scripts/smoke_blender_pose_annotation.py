#!/usr/bin/env python
"""Blender-native smoke test for the BIM pose annotation extension."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON_ROOT = ROOT / "blender_addons"
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

import bpy
from mathutils import Euler, Matrix, Vector

import bim_pose_annotation
from bim_pose_annotation.annotation_io import (
    ROLE_GS_REFERENCE,
    ROLE_KEY,
    ROLE_OBSERVATION_CAMERA,
    ROLE_TRELLIS_PROXY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def transform_matrix(translation, rotation_euler, scale=1.0) -> Matrix:
    rotation = Euler(rotation_euler, "XYZ").to_matrix().to_4x4()
    scaling = Matrix.Diagonal((scale, scale, scale, 1.0))
    return Matrix.Translation(Vector(translation)) @ rotation @ scaling


def create_mesh_object(name: str, center: tuple[float, float, float]) -> bpy.types.Object:
    cx, cy, cz = center
    vertices = [
        (cx + dx, cy + dy, cz + dz)
        for dx in (-0.5, 0.5)
        for dy in (-0.75, 0.75)
        for dz in (-1.0, 1.0)
    ]
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def assert_close(actual, expected, tolerance=1.0e-5):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (actual, expected)
        for actual_value, expected_value in zip(actual, expected):
            assert_close(actual_value, expected_value, tolerance)
        return
    assert math.isclose(float(actual), float(expected), abs_tol=tolerance), (
        actual,
        expected,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene["bim_scene_id"] = "smoke_scene"
    scene["bim_up_axis"] = 2
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 680
    scene.render.resolution_percentage = 100

    source_ply = output_dir / "smoke_scene.ply"
    source_glb = output_dir / "chair_view_001.glb"
    proxy_source = output_dir / "chair_view_001.proxy.ply"
    rgb_path = output_dir / "view_002.png"
    depth_path = output_dir / "view_002.depth.npy"
    mask_path = output_dir / "view_002.mask.png"
    for path in (source_ply, source_glb, proxy_source, rgb_path, depth_path, mask_path):
        path.write_bytes(b"smoke")

    gs_reference = create_mesh_object("GS_REFERENCE", (0.0, 0.0, 0.0))
    gs_reference.matrix_world = transform_matrix(
        (0.4, -0.2, 0.7),
        (-math.pi / 2.0, 0.0, -math.pi / 2.0),
    )
    gs_reference[ROLE_KEY] = ROLE_GS_REFERENCE
    gs_reference["bim_source_ply"] = str(source_ply)
    gs_reference["bim_metric_scale_validated"] = True

    mesh_center = (1.0, 2.0, 3.0)
    proxy = create_mesh_object("CHAIR_PROXY", mesh_center)
    trellis_to_gs = transform_matrix(
        (1.5, -0.75, 2.25),
        (0.2, -0.1, 0.4),
        1.25,
    )
    proxy.matrix_world = gs_reference.matrix_world @ trellis_to_gs
    proxy[ROLE_KEY] = ROLE_TRELLIS_PROXY
    proxy["bim_object_id"] = "chair_17"
    proxy["bim_class_name"] = "chair"
    proxy["bim_variant_id"] = "view_001_seed_1"
    proxy["bim_source_view_id"] = "view_001"
    proxy["bim_source_glb"] = str(source_glb)
    proxy["bim_proxy_source"] = str(proxy_source)
    proxy["bim_annotation_status"] = "approved"
    proxy["bim_annotation_quality"] = 0.95
    proxy["bim_annotation_notes"] = "synthetic smoke annotation"

    camera_data = bpy.data.cameras.new("ObservationCameraData")
    camera = bpy.data.objects.new("OBS_VIEW_002", camera_data)
    scene.collection.objects.link(camera)
    camera_to_gs = transform_matrix((5.0, -3.0, 4.0), (1.1, 0.0, 0.65))
    camera.matrix_world = gs_reference.matrix_world @ camera_to_gs
    camera_data.angle_y = math.radians(50.0)
    camera[ROLE_KEY] = ROLE_OBSERVATION_CAMERA
    camera["bim_target_object_id"] = "chair_17"
    camera["bim_view_id"] = "view_002"
    camera["bim_rgb_path"] = str(rgb_path)
    camera["bim_depth_path"] = str(depth_path)
    camera["bim_mask_path"] = str(mask_path)
    camera["bim_norm_bbox"] = (0.45, 0.55, 0.25, 0.35)
    scene.camera = camera

    bim_pose_annotation.register()
    assert bpy.ops.bim_pose.validate_scene() == {"FINISHED"}

    blend_path = output_dir / "smoke_annotation.blend"
    manifest_path = output_dir / "smoke_annotation.json"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    assert bpy.ops.bim_pose.export_manifest(filepath=str(manifest_path)) == {"FINISHED"}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["validation"]["errors"] == []
    assert len(payload["annotations"]) == 1
    annotation = payload["annotations"][0]
    ground_truth = annotation["ground_truth"]
    assert_close(ground_truth["raw_trellis_to_gs"], [list(row) for row in trellis_to_gs])
    assert_close(ground_truth["mesh_center"], mesh_center)
    assert_close(ground_truth["scale"], 1.25)
    expected_centered_translation = trellis_to_gs @ Vector(mesh_center)
    assert_close(ground_truth["translation"], list(expected_centered_translation))
    assert annotation["annotation"]["status"] == "approved"
    assert annotation["observation_views"][0]["view_id"] == "view_002"
    assert_close(
        annotation["observation_views"][0]["camera_to_gs"],
        [list(row) for row in camera_to_gs],
    )
    print(f"BIM_POSE_SMOKE_BLEND: {blend_path}")
    print(f"BIM_POSE_SMOKE_MANIFEST: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
