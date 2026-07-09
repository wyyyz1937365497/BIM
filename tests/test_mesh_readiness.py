"""Tests for mesh_readiness: multi-angle rendering + VLM readiness parsing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from bim_recon.mesh_readiness import (
    MeshReadinessResult,
    _build_mesh_readiness_prompt,
    _parse_mesh_readiness,
    render_multi_angle,
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_element_class(self):
        prompt = _build_mesh_readiness_prompt("furniture")
        assert "FURNITURE" in prompt

    def test_prompt_asks_ready_or_not(self):
        prompt = _build_mesh_readiness_prompt("chair")
        assert "READY" in prompt
        assert "NOT_READY" in prompt

    def test_prompt_mentions_3d_reconstruction(self):
        prompt = _build_mesh_readiness_prompt("sofa")
        assert "3D" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseReadiness:
    def test_ready_response(self):
        is_ready, reason = _parse_mesh_readiness(
            "READY\nComplete object visible, good front angle."
        )
        assert is_ready is True
        assert "Complete" in reason

    def test_not_ready_response(self):
        is_ready, reason = _parse_mesh_readiness(
            "NOT_READY\nObject is cut off at right edge."
        )
        assert is_ready is False
        assert "cut off" in reason

    def test_ambiguous_response_is_conservative(self):
        """Vague responses default to NOT ready (conservative)."""
        is_ready, reason = _parse_mesh_readiness("Maybe, it looks okay I guess.")
        assert is_ready is False

    def test_ready_with_space_variant(self):
        """Handle 'NOT READY' (with space) as NOT_READY."""
        is_ready, _ = _parse_mesh_readiness("NOT READY\nToo dark.")
        assert is_ready is False

    def test_empty_response_is_conservative(self):
        is_ready, _ = _parse_mesh_readiness("")
        assert is_ready is False


# ---------------------------------------------------------------------------
# Multi-angle rendering (mock scene)
# ---------------------------------------------------------------------------

class TestRenderMultiAngle:
    def _make_mock_scene(self):
        """Mock GSScene that returns a solid-color image."""
        mock = MagicMock()
        mock.render.return_value = MagicMock()
        mock.render.return_value.colors = np.zeros((100, 100, 3), dtype=np.float32)
        return mock

    def test_renders_three_angles_by_default(self, tmp_path):
        scene = self._make_mock_scene()
        results = render_multi_angle(
            scene, world_x=2.0, world_y=0.0,
            h_min=0.0, h_max=1.0,
            scan_center=(0.0, 0.0), floor_z=0.0,
            output_dir=tmp_path,
            name_prefix="test",
        )
        assert len(results) == 3  # default 3 angles
        for angle, path in results:
            assert Path(path).exists()
            assert isinstance(angle, float)

    def test_custom_angles(self, tmp_path):
        scene = self._make_mock_scene()
        results = render_multi_angle(
            scene, world_x=1.0, world_y=1.0,
            h_min=0.0, h_max=2.0,
            scan_center=(0.0, 0.0), floor_z=0.0,
            angles=[-45.0, 0.0, 45.0],
            output_dir=tmp_path,
            name_prefix="custom",
        )
        assert len(results) == 3
        angles = [a for a, _ in results]
        assert angles == [-45.0, 0.0, 45.0]

    def test_single_angle(self, tmp_path):
        scene = self._make_mock_scene()
        results = render_multi_angle(
            scene, world_x=1.0, world_y=0.0,
            h_min=0.0, h_max=1.0,
            scan_center=(0.0, 0.0), floor_z=0.0,
            num_steps=1,
            output_dir=tmp_path,
            name_prefix="single",
        )
        assert len(results) == 1
        assert results[0][0] == 0.0  # single angle is front

    def test_images_are_square(self, tmp_path):
        """TRELLIS prefers square images for better framing."""
        scene = self._make_mock_scene()
        results = render_multi_angle(
            scene, world_x=1.0, world_y=0.0,
            h_min=0.0, h_max=1.0,
            scan_center=(0.0, 0.0), floor_z=0.0,
            output_dir=tmp_path,
            name_prefix="square",
        )
        for _, path in results:
            img = Image.open(path)
            assert img.size[0] == img.size[1]  # square


# ---------------------------------------------------------------------------
# MeshReadinessResult
# ---------------------------------------------------------------------------

class TestMeshReadinessResult:
    def test_ready_result(self):
        r = MeshReadinessResult(
            is_ready=True,
            best_image_path=Path("/tmp/best.png"),
            best_angle=0.0,
            scores={0.0: "READY: good", -30.0: "NOT_READY: cut off"},
            reason="best angle 0°: good",
        )
        assert r.is_ready is True
        assert r.best_angle == 0.0

    def test_not_ready_result(self):
        r = MeshReadinessResult(
            is_ready=False,
            best_image_path=None,
            best_angle=0.0,
            scores={0.0: "NOT_READY: occluded", 30.0: "NOT_READY: too far"},
            reason="No angle passed",
        )
        assert r.is_ready is False
        assert r.best_image_path is None
