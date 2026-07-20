"""Behavioral tests for learned B-class pose refinement."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from bim_recon.config import load_config
from bim_recon.mesh_registrar import MeshPlacement, compute_placement_transform
from bim_recon.pose_refiner import (
    PoseObservation,
    PoseQuality,
    PoseRefinementResult,
    PoseRefinerNet,
    create_pose_refiner,
    crop_observation_channels,
    load_pose_observation,
    pose_refiner_loss,
    render_mesh_channels,
    sample_mesh_surface,
)
from bim_recon.pose_refiner_synthetic import (
    SyntheticPoseConfig,
    make_synthetic_pose_sample,
)
from bim_recon.trellis_client import TrellisMeshResult
from bim_recon.trellis_workflow import (
    ApprovedMeshObject,
    TrellisRevitWorkflow,
    TrellisWorkflowConfig,
)


def _write_cube_glb(path: Path) -> Path:
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint16)
    vertex_bytes = vertices.tobytes()
    face_bytes = faces.tobytes()
    binary = vertex_bytes + face_bytes
    while len(binary) % 4:
        binary += b"\x00"
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes)},
            {"buffer": 0, "byteOffset": len(vertex_bytes), "byteLength": len(face_bytes)},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    json_bytes = json.dumps(gltf).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with path.open("wb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, total))
        stream.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        stream.write(json_bytes)
        stream.write(struct.pack("<II", len(binary), 0x004E4942))
        stream.write(binary)
    return path


def test_synthetic_sample_and_model_loss_are_finite():
    config = SyntheticPoseConfig(image_size=64, mesh_points=128)
    sample = make_synthetic_pose_sample(7, config)
    batch = {key: value[None] for key, value in sample.items()}
    model = PoseRefinerNet(channels=32)

    outputs = model(
        batch["observed"], batch["candidate"],
        batch["mesh_features"], batch["metadata"],
    )
    loss, parts = pose_refiner_loss(outputs, batch)

    assert sample["observed"].shape == (5, 64, 64)
    assert sample["candidate"].shape == (5, 64, 64)
    assert sample["mesh_features"].shape == (128, 6)
    assert sample["metadata"].shape == (PoseRefinerNet.metadata_dim,)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_crop_observation_channels_uses_bbox_padding():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    depth = np.ones((100, 200), dtype=np.float32)
    mask = np.zeros((100, 200), dtype=np.float32)
    mask[30:70, 80:120] = 1.0
    observation = PoseObservation(
        rgb=rgb,
        depth=depth,
        mask=mask,
        norm_bbox=(0.5, 0.5, 0.2, 0.4),
        camera_eye=(0.0, 0.0, -3.0),
        camera_target=(0.0, 0.0, 0.0),
        camera_up=(0.0, 1.0, 0.0),
        camera_fov=50.0,
        camera_image_size=(200, 100),
    )

    cropped_rgb, cropped_depth, cropped_mask = crop_observation_channels(observation)

    assert cropped_rgb.shape[:2] == cropped_depth.shape == cropped_mask.shape
    assert cropped_rgb.shape[0] > 40
    assert cropped_rgb.shape[1] > 40
    assert cropped_mask.sum() == pytest.approx(1600.0)


def test_load_pose_observation_rejects_misaligned_assets(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.npy"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.zeros((10, 10), dtype=np.float32))

    with pytest.raises(ValueError, match="shapes differ"):
        load_pose_observation(
            rgb_path, depth_path, mask_path,
            (0.5, 0.5, 0.5, 0.5),
            (0.0, 0.0, -3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            50.0, (20, 20), 2,
        )


def test_create_pose_refiner_is_disabled_without_checkpoint(tmp_path):
    config = load_config(tmp_path / "missing.json").pose_refiner
    assert create_pose_refiner(config) is None

def test_checkpoint_backed_refiner_runs_full_geometry_path(tmp_path):
    glb = _write_cube_glb(tmp_path / "runtime_cube.glb")
    checkpoint = tmp_path / "pose_refiner.pt"
    torch.save({"model_state": PoseRefinerNet().state_dict()}, checkpoint)
    placement = MeshPlacement(
        glb_path=glb,
        world_x=0.0,
        world_y=0.0,
        floor_z=0.0,
        ceiling_z=3.0,
        element_width_m=1.0,
        element_height_m=1.0,
        up_axis=2,
    )
    points, normals, _ = sample_mesh_surface(glb, count=512)
    observation_shell = PoseObservation(
        rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        depth=np.zeros((128, 128), dtype=np.float32),
        mask=np.zeros((128, 128), dtype=np.float32),
        norm_bbox=(0.5, 0.5, 0.65, 0.65),
        camera_eye=(0.0, -3.0, 0.5),
        camera_target=(0.0, 0.0, 0.5),
        camera_up=(0.0, 0.0, 1.0),
        camera_fov=50.0,
        camera_image_size=(128, 128),
        up_axis=2,
    )
    rendered_rgb, rendered_depth, rendered_mask = render_mesh_channels(
        points,
        normals,
        compute_placement_transform(placement),
        observation_shell,
        128,
    )
    observation = PoseObservation(
        rgb=rendered_rgb,
        depth=rendered_depth,
        mask=rendered_mask,
        norm_bbox=observation_shell.norm_bbox,
        camera_eye=observation_shell.camera_eye,
        camera_target=observation_shell.camera_target,
        camera_up=observation_shell.camera_up,
        camera_fov=observation_shell.camera_fov,
        camera_image_size=observation_shell.camera_image_size,
        up_axis=2,
    )
    config = SimpleNamespace(
        enabled=True,
        checkpoint=str(checkpoint),
        device="cpu",
        input_size=64,
        iterations=1,
        confidence_threshold=0.65,
        min_quality_score=0.0,
        quality_tolerance=0.02,
        min_mask_pixels=1,
        min_depth_ratio=0.001,
        max_rotation_degrees=20.0,
        max_translation_m=0.25,
        max_log_scale=0.15,
        gravity_locked=True,
        floor_contact=True,
    )

    result = create_pose_refiner(config).refine_placement(placement, observation)

    assert result.iterations == 1
    assert 0.0 <= result.confidence <= 1.0
    assert np.isfinite(result.initial_quality.score)
    assert np.isfinite(result.refined_quality.score)
    assert result.accepted or result.fallback_reason


class _FakeTrellisClient:
    def __init__(self, result: TrellisMeshResult):
        self.result = result

    def health(self):
        return True

    def generate_mesh(self, _request):
        return self.result


class _FakePoseRefiner:
    def __init__(self, accepted: bool):
        self.accepted = accepted
        self.calls = 0

    def refine_placement(self, placement, _observation):
        self.calls += 1
        refined = MeshPlacement(
            glb_path=placement.glb_path,
            world_x=placement.world_x,
            world_y=placement.world_y,
            floor_z=placement.floor_z,
            ceiling_z=placement.ceiling_z,
            element_width_m=placement.element_width_m,
            element_height_m=placement.element_height_m,
            up_axis=placement.up_axis,
            yaw_degrees=placement.yaw_degrees,
            translation_offset=(0.1, 0.0, 0.0),
            category=placement.category,
            name=placement.name,
        )
        quality = PoseQuality(0.7, 0.8, 0.74)
        return PoseRefinementResult(
            placement=refined if self.accepted else placement,
            accepted=self.accepted,
            confidence=0.9 if self.accepted else 0.2,
            iterations=1,
            initial_quality=quality,
            refined_quality=quality,
            fallback_reason=None if self.accepted else "low_confidence",
        )


def _workflow_object(tmp_path: Path) -> ApprovedMeshObject:
    rgb = tmp_path / "rgb.png"
    mask = tmp_path / "mask.png"
    depth = tmp_path / "depth.npy"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(rgb)
    mask_array = np.zeros((32, 32), dtype=np.uint8)
    mask_array[8:24, 8:24] = 255
    Image.fromarray(mask_array).save(mask)
    np.save(depth, np.full((32, 32), 3.0, dtype=np.float32))
    return ApprovedMeshObject(
        object_id="chair_1",
        label="chair",
        image_path=rgb,
        observation_rgb_path=rgb,
        depth_path=depth,
        mask_path=mask,
        norm_bbox=(0.5, 0.5, 0.5, 0.5),
        camera_eye=(0.0, 0.0, -3.0),
        camera_target=(0.0, 0.0, 0.0),
        camera_up=(0.0, 1.0, 0.0),
        camera_fov=50.0,
        camera_image_size=(32, 32),
        position_3d=(0.0, 0.0, 0.5),
    )


@pytest.mark.parametrize("accepted, expected_offset", [(True, 0.1), (False, 0.0)])
def test_workflow_applies_or_falls_back_from_pose_refinement(
    tmp_path, monkeypatch, accepted, expected_offset,
):
    glb = _write_cube_glb(tmp_path / "cube.glb")
    refiner = _FakePoseRefiner(accepted)
    captured: dict[str, MeshPlacement] = {}

    def fake_register(placement, transform):
        captured["placement"] = placement
        return {
            "vertex_count": len(transform.vertices_world),
            "face_count": len(transform.faces),
            "payload_path": str(tmp_path / "payload.json"),
        }

    monkeypatch.setattr(
        "bim_recon.trellis_workflow.register_mesh_in_revit", fake_register,
    )
    workflow = TrellisRevitWorkflow(
        TrellisWorkflowConfig(
            objects=(_workflow_object(tmp_path),),
            output_dir=tmp_path / "output",
            register_in_revit=False,
            pose_refiner=refiner,
        ),
        client_factory=lambda: _FakeTrellisClient(
            TrellisMeshResult(glb, None, None, 1),
        ),
    )

    from bim_recon.workflow_runtime import stream_workflow_sync
    events = list(stream_workflow_sync(workflow))
    manifest = json.loads((tmp_path / "output" / "trellis_workflow.json").read_text())

    assert events
    assert refiner.calls == 1
    assert captured["placement"].translation_offset[0] == pytest.approx(expected_offset)
    assert manifest[0]["pose_refinement"]["accepted"] is accepted
    if not accepted:
        assert manifest[0]["pose_refinement"]["fallback_reason"] == "low_confidence"

def test_workflow_records_missing_observation_fallback(tmp_path, monkeypatch):
    glb = _write_cube_glb(tmp_path / "cube_missing.glb")
    refiner = _FakePoseRefiner(True)
    monkeypatch.setattr(
        "bim_recon.trellis_workflow.register_mesh_in_revit",
        lambda _placement, transform: {
            "vertex_count": len(transform.vertices_world),
            "face_count": len(transform.faces),
            "payload_path": str(tmp_path / "payload.json"),
        },
    )
    object_without_observation = ApprovedMeshObject(
        object_id="chair_missing",
        label="chair",
        image_path=tmp_path / "input.png",
        position_3d=(0.0, 0.0, 0.5),
    )
    object_without_observation.image_path.write_bytes(b"png")
    workflow = TrellisRevitWorkflow(
        TrellisWorkflowConfig(
            objects=(object_without_observation,),
            output_dir=tmp_path / "missing_output",
            register_in_revit=False,
            pose_refiner=refiner,
        ),
        client_factory=lambda: _FakeTrellisClient(
            TrellisMeshResult(glb, None, None, 1),
        ),
    )

    from bim_recon.workflow_runtime import stream_workflow_sync
    list(stream_workflow_sync(workflow))
    manifest = json.loads(
        (tmp_path / "missing_output" / "trellis_workflow.json").read_text()
    )

    assert refiner.calls == 0
    assert manifest[0]["pose_refinement"]["fallback_reason"] == "missing_observation"

def test_workflow_falls_back_when_pose_inference_raises(tmp_path, monkeypatch):
    glb = _write_cube_glb(tmp_path / "cube_error.glb")

    class RaisingRefiner:
        def refine_placement(self, _placement, _observation):
            raise RuntimeError("broken checkpoint")

    monkeypatch.setattr(
        "bim_recon.trellis_workflow.register_mesh_in_revit",
        lambda _placement, transform: {
            "vertex_count": len(transform.vertices_world),
            "face_count": len(transform.faces),
            "payload_path": str(tmp_path / "payload.json"),
        },
    )
    workflow = TrellisRevitWorkflow(
        TrellisWorkflowConfig(
            objects=(_workflow_object(tmp_path),),
            output_dir=tmp_path / "error_output",
            register_in_revit=False,
            pose_refiner=RaisingRefiner(),
        ),
        client_factory=lambda: _FakeTrellisClient(
            TrellisMeshResult(glb, None, None, 1),
        ),
    )

    from bim_recon.workflow_runtime import stream_workflow_sync
    list(stream_workflow_sync(workflow))
    manifest = json.loads(
        (tmp_path / "error_output" / "trellis_workflow.json").read_text()
    )

    assert manifest[0]["status"] == "completed"
    assert manifest[0]["pose_refinement"]["fallback_reason"].startswith(
        "inference_error:RuntimeError"
    )
