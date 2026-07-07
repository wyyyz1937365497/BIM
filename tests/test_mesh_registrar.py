"""Unit tests for mesh_registrar: coordinate transform math + GLB parsing.

Tests cover:
  - Axis remap rotation matrices (Z-up, Y-up, X-up)
  - Placement transform: scale, translation, vertex output shape
  - GLB parsing fallback (when trimesh unavailable)
  - register_mesh_in_revit payload format
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bim_recon.mesh_registrar import (
    MeshPlacement,
    MeshTransform,
    _build_axis_remap_rotation,
    compute_placement_transform,
    parse_glb_vertices_faces,
    register_mesh_in_revit,
)


# ---------------------------------------------------------------------------
# Axis remap rotation
# ---------------------------------------------------------------------------

class TestAxisRemapRotation:
    def test_zup_rotates_y_to_z(self):
        """TRELLIS Y-up → 3DGS Z-up: mesh Y axis maps to world Z."""
        R = _build_axis_remap_rotation(up_axis=2)
        # Mesh up vector [0, 1, 0] should become world up [0, 0, 1]
        result = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-6)

    def test_zup_mesh_z_becomes_negative_y(self):
        """Mesh forward (Z) → world -Y."""
        R = _build_axis_remap_rotation(up_axis=2)
        result = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-6)

    def test_yup_is_identity(self):
        """TRELLIS is already Y-up, so remap is identity."""
        R = _build_axis_remap_rotation(up_axis=1)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_xup_mesh_y_becomes_x(self):
        """Mesh up (Y) → world X when up_axis=0."""
        R = _build_axis_remap_rotation(up_axis=0)
        result = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-6)
    def test_rotation_is_orthonormal(self):
        """R @ R.T = I for all axes."""
        for up_axis in [0, 1, 2]:
            R = _build_axis_remap_rotation(up_axis)
            product = R @ R.T
            np.testing.assert_allclose(product, np.eye(3), atol=1e-6)

    def test_det_is_one(self):
        """Rotation determinant = 1 (proper rotation, no reflection)."""
        for up_axis in [0, 1, 2]:
            R = _build_axis_remap_rotation(up_axis)
            assert abs(np.linalg.det(R) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Placement transform
# ---------------------------------------------------------------------------

class TestComputePlacementTransform:
    def _make_test_glb(self, tmp_path: Path) -> Path:
        """Create a minimal valid GLB file with a unit cube."""
        # Build a minimal GLB with JSON + BIN chunks
        # Vertices: 8 corners of a unit cube, TRELLIS Y-up convention
        vertices = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 5, 1], [0, 4, 5],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [1, 5, 6], [1, 6, 2],  # right
            [0, 3, 7], [0, 7, 4],  # left
        ], dtype=np.uint16)

        # Build buffer: vertices (float32) + faces (uint16)
        vert_bytes = vertices.tobytes()
        face_bytes = faces.tobytes()
        buffer_data = vert_bytes + face_bytes

        # Pad to 4-byte boundary
        while len(buffer_data) % 4 != 0:
            buffer_data += b'\x00'

        # Accessors
        accessors = [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": 8,
                "type": "VEC3",
                "byteOffset": 0,
                "max": [0.5, 0.5, 0.5],
                "min": [-0.5, -0.5, -0.5],
            },
            {
                "bufferView": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": 36,
                "type": "SCALAR",
                "byteOffset": len(vert_bytes),
            },
        ]

        buffer_views = [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(buffer_data),
            },
        ]

        gltf_json = {
            "asset": {"version": "2.0"},
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0},
                    "indices": 1,
                }],
            }],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buffer_data)}],
        }

        json_bytes = json.dumps(gltf_json).encode('utf-8')
        # Pad JSON to 4-byte boundary with spaces
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '

        # GLB structure: header(12) + JSON chunk(8+len) + BIN chunk(8+len)
        total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)

        glb_path = tmp_path / "test_cube.glb"
        with open(glb_path, 'wb') as f:
            # Header
            f.write(struct.pack('<III', 0x46546C67, 2, total_length))
            # JSON chunk
            f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
            f.write(json_bytes)
            # BIN chunk
            f.write(struct.pack('<II', len(buffer_data), 0x004E4942))
            f.write(buffer_data)

        return glb_path

    def test_transform_scales_mesh_to_target_width(self, tmp_path):
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=2.0, world_y=3.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        # Unit cube has horizontal extent = 1.0, target = 1.0m → scale = 1.0
        assert transform.scale == pytest.approx(1.0, abs=0.01)

    def test_transform_translates_to_world_position(self, tmp_path):
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=2.0, world_y=3.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        # Centroid of transformed vertices should be at (world_x, world_y, mid_height)
        centroid = transform.vertices_world.mean(axis=0)
        assert centroid[0] == pytest.approx(2.0, abs=0.01)
        assert centroid[1] == pytest.approx(3.0, abs=0.01)
        # Z should be at floor + half height
        assert centroid[2] == pytest.approx(0.5, abs=0.01)

    def test_transform_output_shapes(self, tmp_path):
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=0.5, element_height_m=0.5,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        assert transform.vertices_world.shape == (8, 3)
        assert transform.faces.shape == (12, 3)
        assert transform.rotation.shape == (3, 3)
        assert transform.translation.shape == (3,)

    def test_transform_clamps_to_ceiling(self, tmp_path):
        glb_path = self._make_test_glb(tmp_path)
        # Unit cube (1m) but room only 0.5m tall → scale should shrink
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=0.5,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        # Scaled mesh height should not exceed room height
        mesh_height = transform.vertices_world[:, 2].max() - transform.vertices_world[:, 2].min()
        assert mesh_height <= 0.5 + 0.01  # tolerance for float


# ---------------------------------------------------------------------------
# GLB parsing
# ---------------------------------------------------------------------------

class TestParseGlb:
    def test_parses_test_cube(self, tmp_path):
        """Verify the minimal GLB parser extracts correct vertex/face counts."""
        # Reuse the cube builder from TestComputePlacementTransform
        builder = TestComputePlacementTransform()
        glb_path = builder._make_test_glb(tmp_path)

        vertices, faces = parse_glb_vertices_faces(glb_path)

        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)

    def test_raises_on_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_glb_vertices_faces(Path("/nonexistent/file.glb"))


# ---------------------------------------------------------------------------
# Revit registration payload
# ---------------------------------------------------------------------------

class TestRegisterMeshInRevit:
    def test_no_runner_returns_formatted_payload(self, tmp_path):
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=1.0, world_y=2.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
            name="Test Mesh",
            category="OST_GenericModel",
        )
        transform = compute_placement_transform(placement)
        result = register_mesh_in_revit(placement, transform)

        assert result["status"] == "formatted"
        assert result["script_name"] == "create_directshape_from_mesh"
        assert result["vertex_count"] == 8
        assert result["face_count"] == 12

        payload = json.loads(result["payload_json"])
        assert payload["name"] == "Test Mesh"
        assert payload["category"] == "OST_GenericModel"
        assert len(payload["vertices"]) == 8 * 3
        assert len(payload["faces"]) == 12 * 3

        # Verify coordinates are in feet (1m ≈ 3.28ft)
        max_x_ft = max(payload["vertices"][i] for i in range(0, len(payload["vertices"]), 3))
        assert max_x_ft > 3.0

    def test_with_mock_runner_calls_revit(self, tmp_path):
        """When a runner with MCP sender is provided, register_mesh_in_revit
        calls runner.run() and returns the Revit result."""
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0, world_y=0,
            floor_z=0, ceiling_z=3,
            element_width_m=1.0, element_height_m=1.0,
        )
        transform = compute_placement_transform(placement)

        class MockRunner:
            def __init__(self):
                self.last_script = None
                self.last_params = None

            def run(self, script_name, parameters=None):
                self.last_script = script_name
                self.last_params = parameters
                return {"elementId": 12345, "name": "test"}

        runner = MockRunner()
        result = register_mesh_in_revit(placement, transform, runner=runner)

        assert result["status"] == "ok"
        assert result["elementId"] == 12345
        assert runner.last_script == "create_directshape_from_mesh"
        assert runner.last_params is not None

    def test_with_runner_no_sender_returns_formatted(self, tmp_path):
        """When runner has no MCP sender (_note in result), returns formatted."""
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0, world_y=0,
            floor_z=0, ceiling_z=3,
            element_width_m=1.0, element_height_m=1.0,
        )
        transform = compute_placement_transform(placement)

        class MockRunnerNoSender:
            def run(self, script_name, parameters=None):
                return {"_note": "No MCP sender configured"}

        result = register_mesh_in_revit(placement, transform, runner=MockRunnerNoSender())

        assert result["status"] == "formatted"
        assert "payload_json" in result
        assert result["status"] == "formatted"
