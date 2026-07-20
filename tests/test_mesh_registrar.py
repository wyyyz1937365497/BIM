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
    _build_yaw_rotation,
    _horizontal_angle_deg,
    _principal_axis,
    compute_placement_transform,
    extract_object_from_render,
    parse_glb_vertices_faces,
    register_mesh_in_revit,
    serialize_placement_diagnostics,
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
# Yaw rotation
# ---------------------------------------------------------------------------

class TestYawRotation:
    def test_zup_90_cw_sends_east_to_south(self):
        """90° CW from above sends world +X (east) → -Y (south) for Z-up."""
        R = _build_yaw_rotation(up_axis=2, yaw_degrees=90.0)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-6)

    def test_zup_preserves_up_axis(self):
        """Yaw around Z leaves the Z component untouched."""
        R = _build_yaw_rotation(up_axis=2, yaw_degrees=90.0)
        result = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-6)

    def test_yup_90_rotates_horizontal_plane(self):
        """For Y-up, 90° yaw maps +X into the horizontal (XZ) plane.

        ``_build_yaw_rotation`` applies ``-yaw_degrees`` around the right-hand
        rule +up axis; for Y-up this sends +X to +Z at 90°. The exact
        horizontal direction is a convention choice; we only assert that the
        up component stays zero and the result lies in the X-Z plane.
        """
        R = _build_yaw_rotation(up_axis=1, yaw_degrees=90.0)
        result = R @ np.array([1, 0, 0])
        assert abs(result[1]) < 1e-6  # Y component unchanged (still 0)
        # Magnitude preserved (pure rotation)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_zero_yaw_is_identity(self):
        for up_axis in [0, 1, 2]:
            R = _build_yaw_rotation(up_axis, 0.0)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_is_proper_rotation(self):
        """Yaw matrices are orthonormal with det=+1 for any axis/angle."""
        for up_axis in [0, 1, 2]:
            for deg in [0, 30, 90, 180, -90]:
                R = _build_yaw_rotation(up_axis, deg)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
                assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_invalid_up_axis_raises(self):
        with pytest.raises(ValueError):
            _build_yaw_rotation(up_axis=5, yaw_degrees=90.0)


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

    def test_default_yaw_composes_with_axis_remap(self, tmp_path):
        """Default yaw_degrees=90 produces rotation = yaw(90) @ axis_remap."""
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        expected = _build_yaw_rotation(2, 90.0) @ _build_axis_remap_rotation(2)
        np.testing.assert_allclose(transform.rotation, expected, atol=1e-6)

    def test_zero_yaw_falls_back_to_axis_remap_only(self, tmp_path):
        """yaw_degrees=0 reproduces the legacy axis-remap-only transform."""
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
            yaw_degrees=0.0,
        )
        transform = compute_placement_transform(placement)

        expected = _build_axis_remap_rotation(2)
        np.testing.assert_allclose(transform.rotation, expected, atol=1e-6)

    def test_default_yaw_preserves_world_up_from_mesh_up(self, tmp_path):
        """Default yaw still maps TRELLIS Y-up to world Z-up (vertical kept)."""
        glb_path = self._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        # Mesh Y-axis [0,1,0] (TRELLIS up) must map to world Z-axis [0,0,1]
        # (the up axis). R @ mesh_up == world_up.
        result = transform.rotation @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Placement diagnostics (PCA + serializer)
# ---------------------------------------------------------------------------

class TestPrincipalAxis:
    def test_long_x_returns_unit_x(self):
        """Vertices spread along mesh X → principal axis ≈ ±X."""
        verts = np.zeros((100, 3), dtype=np.float32)
        verts[:, 0] = np.linspace(-1, 1, 100)
        verts[:, 1] = np.random.uniform(-0.01, 0.01, 100)
        verts[:, 2] = np.random.uniform(-0.01, 0.01, 100)
        result = _principal_axis(verts - verts.mean(axis=0))
        # Sign is arbitrary; the axis line should be X
        assert abs(result[0]) > 0.99
        assert abs(result[1]) < 0.05
        assert abs(result[2]) < 0.05
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_diagonal_returns_diagonal(self):
        """Vertices along the X-Z diagonal → principal axis is diagonal."""
        verts = np.zeros((100, 3), dtype=np.float32)
        t = np.linspace(-1, 1, 100)
        verts[:, 0] = t
        verts[:, 2] = t
        result = _principal_axis(verts - verts.mean(axis=0))
        # Either +diagonal or -diagonal; both should have |x| ≈ |z|
        assert abs(abs(result[0]) - abs(result[2])) < 0.05
        assert abs(result[1]) < 0.05

    def test_single_vertex_falls_back_to_x(self):
        result = _principal_axis(np.zeros((1, 3), dtype=np.float32))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_zero_covariance_falls_back_to_x(self):
        # All identical points → zero covariance
        verts = np.ones((10, 3), dtype=np.float32)
        result = _principal_axis(verts - verts.mean(axis=0))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)


class TestHorizontalAngleDeg:
    def test_east_is_zero(self):
        assert _horizontal_angle_deg(np.array([1, 0, 0]), up_axis=2) == pytest.approx(0.0)

    def test_north_is_90(self):
        assert _horizontal_angle_deg(np.array([0, 1, 0]), up_axis=2) == pytest.approx(90.0)

    def test_south_is_minus_90(self):
        assert _horizontal_angle_deg(np.array([0, -1, 0]), up_axis=2) == pytest.approx(-90.0)

    def test_yup_uses_xz_plane(self):
        """Y-up: horizontal plane is X-Z; +Z is the 'north' direction."""
        assert _horizontal_angle_deg(np.array([0, 0, 1]), up_axis=1) == pytest.approx(90.0)


class TestPlacementDiagnostics:
    def test_transform_carries_diagnostic_fields(self, tmp_path):
        """MeshTransform carries the new diagnostic fields."""
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
        )
        transform = compute_placement_transform(placement)

        # All new fields populated
        assert len(transform.mesh_extents) == 3
        assert len(transform.mesh_center) == 3
        assert len(transform.principal_axis_mesh) == 3
        assert len(transform.principal_axis_world) == 3
        assert isinstance(transform.principal_axis_angle_deg, float)
        # Principal axes are unit vectors
        np.testing.assert_allclose(
            np.linalg.norm(transform.principal_axis_mesh), 1.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(transform.principal_axis_world), 1.0, atol=1e-5,
        )

    def test_serializer_returns_json_safe_dict(self, tmp_path):
        """serialize_placement_diagnostics returns a dict with JSON-safe leaves."""
        import json as _json
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=1.5, world_y=-0.5,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=0.8, element_height_m=1.0,
            up_axis=2,
            yaw_degrees=90.0,
            name="Test Sofa",
        )
        transform = compute_placement_transform(placement)

        result = serialize_placement_diagnostics(placement, transform)

        # Must be JSON-serializable (no numpy arrays)
        _json.dumps(result)  # raises if not serializable

        # Placement inputs echoed back
        assert result["placement_input"]["world_x"] == 1.5
        assert result["placement_input"]["world_y"] == -0.5
        assert result["placement_input"]["yaw_degrees"] == 90.0
        assert result["placement_input"]["up_axis"] == 2

        # Mesh analysis populated
        mesh_ext = result["mesh_analysis_trellis_space"]["extents_x_y_z"]
        assert len(mesh_ext) == 3

        # Transform output populated with row-major 3x3 rotation
        rot = result["transform_output"]["rotation_matrix_row_major"]
        assert len(rot) == 9  # 3x3 = 9 floats
        assert isinstance(result["transform_output"]["scale"], float)
        assert "principal_axis_world_horizontal_angle_deg" in result["transform_output"]

    def test_default_yaw_cube_has_axis_aligned_principal(self, tmp_path):
        """For a unit cube with default yaw, the world principal axis ends up
        axis-aligned (not diagonal). This is the baseline for comparison when
        debugging why a long object lands at 45°."""
        glb_path = TestComputePlacementTransform()._make_test_glb(tmp_path)
        placement = MeshPlacement(
            glb_path=glb_path,
            world_x=0.0, world_y=0.0,
            floor_z=0.0, ceiling_z=3.0,
            element_width_m=1.0, element_height_m=1.0,
            up_axis=2,
        )
        transform = compute_placement_transform(placement)

        angle = transform.principal_axis_angle_deg
        # For a cube with default yaw=90, angle should be a multiple of 90°
        # (i.e., axis-aligned). Allow for any of {0, ±90, 180}.
        remainder = abs(angle) % 90.0
        assert remainder < 1.0 or remainder > 89.0


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
    def test_returns_formatted_payload_file(self, tmp_path):
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

        payload_path = Path(result["payload_path"])
        assert payload_path.is_file()
        payload = json.loads(payload_path.read_text("utf-8"))
        assert payload["name"] == "Test Mesh"
        assert payload["category"] == "OST_GenericModel"
        assert len(payload["vertices"]) == 8 * 3
        assert len(payload["faces"]) == 12 * 3

        # Verify coordinates are in feet (1m ≈ 3.28ft)
        max_x_ft = max(payload["vertices"][i] for i in range(0, len(payload["vertices"]), 3))
        assert max_x_ft > 3.0


# ---------------------------------------------------------------------------
# Object extraction (Falcon mask → clean RGBA)
# ---------------------------------------------------------------------------

class TestExtractObjectFromRender:
    def _make_render(self, size=(400, 300)) -> "Image.Image":
        """Create a test render with a colored rectangle in the center."""
        from PIL import Image, ImageDraw
        img = Image.new("RGB", size, (128, 128, 128))  # gray background
        draw = ImageDraw.Draw(img)
        # Draw a red rectangle in center (the "object")
        draw.rectangle([120, 80, 280, 220], fill=(255, 0, 0))
        return img

    def test_extracts_object_with_mask_bbox(self):
        from PIL import Image
        render = self._make_render()
        detections = [{
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.5},
            "mask_bbox": {"x": 0.5, "y": 0.5, "w": 0.35, "h": 0.45},
            "mask_area_ratio": 0.12,
        }]

        result = extract_object_from_render(render, detections)

        assert result is not None
        assert result.mode == "RGBA"
        # Should be cropped smaller than original
        assert result.size[0] < render.size[0]
        assert result.size[1] < render.size[1]

    def test_returns_none_for_empty_detections(self):
        from PIL import Image
        render = self._make_render()
        result = extract_object_from_render(render, [])
        assert result is None

    def test_picks_largest_mask_area(self):
        from PIL import Image
        render = self._make_render()
        detections = [
            {"bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}, "mask_area_ratio": 0.01},
            {"bbox": {"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.5}, "mask_area_ratio": 0.15},
            {"bbox": {"x": 0.8, "y": 0.8, "w": 0.05, "h": 0.05}, "mask_area_ratio": 0.002},
        ]

        result = extract_object_from_render(render, detections)

        assert result is not None
        # The largest detection (index 1, 0.4×0.5) should be selected
        # Crop with padding: ~(0.4+0.1) * 400 = 200 wide
        assert result.size[0] > 150  # roughly 40% of 400

    def test_alpha_channel_is_transparent_at_edges(self):
        from PIL import Image
        import numpy as np
        render = self._make_render()
        detections = [{"bbox": {"x": 0.5, "y": 0.5, "w": 0.6, "h": 0.6}}]

        result = extract_object_from_render(render, detections)
        arr = np.array(result)
        alpha = arr[:, :, 3]

        # Center should be more opaque than corners
        center_alpha = alpha[alpha.shape[0]//2, alpha.shape[1]//2]
        corner_alpha = alpha[0, 0]
        assert center_alpha > corner_alpha

    def test_falls_back_to_bbox_when_no_mask_bbox(self):
        from PIL import Image
        render = self._make_render()
        detections = [{
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3},
            "mask_bbox": None,
            "mask_area_ratio": None,
        }]

        result = extract_object_from_render(render, detections)

        assert result is not None
        assert result.mode == "RGBA"
