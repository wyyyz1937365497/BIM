"""Unit tests for vlm_verifier — pure math, no GPU/Ollama needed.

Tests:
  - compute_polar: θ/r computation from world coordinates
  - candidate_to_viewpoint: polar → camera pose mapping
  - _parse_vlm_response: CONFIRMED/REJECTED parsing
  - _build_prompt: prompt structure
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bim_recon.vlm_verifier import (
    compute_polar,
    candidate_to_viewpoint,
    _parse_vlm_response,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# compute_polar
# ---------------------------------------------------------------------------

class TestComputePolar:
    def test_east_direction(self):
        # Point to the east (+x) → θ=0
        theta, r = compute_polar(5.0, 0.0, (0.0, 0.0))
        assert theta == pytest.approx(0.0)
        assert r == pytest.approx(5.0)

    def test_north_direction(self):
        # Point to the north (+y) → θ=90
        theta, r = compute_polar(0.0, 3.0, (0.0, 0.0))
        assert theta == pytest.approx(90.0)
        assert r == pytest.approx(3.0)

    def test_west_direction(self):
        # Point to the west (-x) → θ=180
        theta, r = compute_polar(-4.0, 0.0, (0.0, 0.0))
        assert theta == pytest.approx(180.0)
        assert r == pytest.approx(4.0)

    def test_south_direction(self):
        # Point to the south (-y) → θ=270
        theta, r = compute_polar(0.0, -2.0, (0.0, 0.0))
        assert theta == pytest.approx(270.0)
        assert r == pytest.approx(2.0)

    def test_offset_center(self):
        theta, r = compute_polar(3.0, 4.0, (1.0, 1.0))
        # dx=2, dy=3 → θ=atan2(3,2), r=sqrt(13)
        assert theta == pytest.approx(math.degrees(math.atan2(3, 2)) % 360)
        assert r == pytest.approx(math.sqrt(13))

    def test_at_center(self):
        theta, r = compute_polar(1.0, 1.0, (1.0, 1.0))
        assert r == pytest.approx(0.0)

    def test_theta_always_positive(self):
        """θ must be in [0, 360)."""
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315, 359]:
            rad = math.radians(angle_deg)
            x = math.cos(rad) * 5
            y = math.sin(rad) * 5
            theta, r = compute_polar(x, y, (0.0, 0.0))
            assert 0 <= theta < 360
            assert r == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# candidate_to_viewpoint
# ---------------------------------------------------------------------------

class TestCandidateToViewpoint:
    def test_basic_mapping(self):
        """Camera at center, looking at candidate to the east."""
        eye, target, fov = candidate_to_viewpoint(
            world_x=5.0, world_y=0.0,
            h_min=0.0, h_max=2.0,
            scan_center=(0.0, 0.0),
            floor_z=0.0,
        )
        # Camera at origin, eye height 1.5m
        assert eye == [0.0, 0.0, 1.5]
        # Target at candidate position, mid-height
        assert target == [5.0, 0.0, 1.0]
        assert fov == 60.0

    def test_custom_eye_height(self):
        eye, target, fov = candidate_to_viewpoint(
            world_x=3.0, world_y=0.0,
            h_min=0.5, h_max=1.5,
            scan_center=(0.0, 0.0),
            floor_z=-1.0,
            eye_height=1.2,
        )
        # eye at floor_z + 1.2 = -1.0 + 1.2 = 0.2
        assert eye == [0.0, 0.0, pytest.approx(0.2)]
        # target mid-height = 1.0, world z = floor_z + 1.0 = 0.0
        assert target == [3.0, 0.0, pytest.approx(0.0)]

    def test_offset_center(self):
        eye, target, fov = candidate_to_viewpoint(
            world_x=2.0, world_y=3.0,
            h_min=0.0, h_max=2.0,
            scan_center=(1.0, 1.0),
            floor_z=0.0,
        )
        assert eye == [1.0, 1.0, 1.5]
        assert target == [2.0, 3.0, 1.0]

    def test_custom_fov(self):
        _, _, fov = candidate_to_viewpoint(
            world_x=1.0, world_y=0.0,
            h_min=0.0, h_max=2.0,
            scan_center=(0.0, 0.0),
            floor_z=0.0,
            fov=90.0,
        )
        assert fov == 90.0

    def test_y_up_axis(self):
        """When up_axis=1 (Y-up), eye/target use Y for vertical."""
        eye, target, fov = candidate_to_viewpoint(
            world_x=5.0, world_y=0.0,
            h_min=0.0, h_max=2.0,
            scan_center=(0.0, 0.0),
            floor_z=0.0,
            up_axis=1,
        )
        # up_axis=1 → h_axes=[0,2], so eye=[cx, floor_z+eye_h, cy]
        assert eye[0] == pytest.approx(0.0)   # h_axes[0] = x
        assert eye[1] == pytest.approx(1.5)   # up_axis = y → floor_z + eye_height
        assert eye[2] == pytest.approx(0.0)   # h_axes[1] = z
        # target=[world_x, floor_z+h_mid, world_y] → but world_y is h-plane too
        # Actually: target[h_axes[0]]=world_x, target[h_axes[1]]=world_y, target[up]=floor_z+h_mid
        # h_axes=[0,2] → target[0]=5.0, target[2]=0.0(world_y), target[1]=1.0(floor_z+h_mid)
        assert target[0] == pytest.approx(5.0)
        assert target[1] == pytest.approx(1.0)
        assert target[2] == pytest.approx(0.0)

    def test_x_up_axis(self):
        """When up_axis=0 (X-up), eye/target use X for vertical."""
        eye, target, fov = candidate_to_viewpoint(
            world_x=3.0, world_y=4.0,
            h_min=0.0, h_max=2.0,
            scan_center=(1.0, 1.0),
            floor_z=-2.0,
            up_axis=0,
        )
        # up_axis=0 → h_axes=[1,2]
        # eye[h_axes[0]]=cy=1.0, eye[h_axes[1]]=0.0(unused), eye[up]=floor_z+1.5=-0.5
        assert eye[0] == pytest.approx(-0.5)  # up_axis = x
        assert eye[1] == pytest.approx(1.0)   # h_axes[0] = y → cx
        assert eye[2] == pytest.approx(1.0)   # h_axes[1] = z → cy
        # target[h_axes[0]]=world_y=4.0... wait, world_x and world_y are h-plane coords
        # target[h_axes[0]]=world_x=3.0? No...
        # The convention: world_x → h_axes[0] position, world_y → h_axes[1] position
        # h_axes=[1,2] → target[1]=world_x=3.0, target[2]=world_y=4.0
        # target[0]=floor_z+h_mid=-2.0+1.0=-1.0
        assert target[0] == pytest.approx(-1.0)
        assert target[1] == pytest.approx(3.0)
        assert target[2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _parse_vlm_response
# ---------------------------------------------------------------------------

class TestParseVLMResponse:
    def test_confirmed(self):
        confirmed, text = _parse_vlm_response(
            "CONFIRMED\nA wooden door is visible."
        )
        assert confirmed is True
        assert "wooden" in text

    def test_rejected(self):
        confirmed, text = _parse_vlm_response(
            "REJECTED\nNo door visible, only a window."
        )
        assert confirmed is False

    def test_confirmed_case_insensitive(self):
        confirmed, _ = _parse_vlm_response("confirmed - yes there is a door")
        assert confirmed is True

    def test_ambiguous_returns_none(self):
        confirmed, _ = _parse_vlm_response("I'm not sure what this is.")
        assert confirmed is None

    def test_confirmed_in_second_line_not_counted(self):
        """Only first line matters for parsing."""
        confirmed, _ = _parse_vlm_response(
            "I see something.\nCONFIRMED it's a door."
        )
        assert confirmed is None  # first line doesn't have CONFIRMED


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_element_class(self):
        prompt = _build_prompt("door")
        assert "DOOR" in prompt

    def test_contains_instruction(self):
        prompt = _build_prompt("window")
        assert "CONFIRMED" in prompt
        assert "REJECTED" in prompt
        assert "first line" in prompt.lower()


# ---------------------------------------------------------------------------
# verify_candidates (mocked end-to-end)
# ---------------------------------------------------------------------------

class TestVerifyCandidatesMock:
    """Mock scene.render and query_ollama to test the full pipeline logic."""

    def test_confirmed_candidate(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from bim_recon.vlm_verifier import verify_candidates, VerificationResult
        from bim_recon.candidate_extractor import Candidate

        cand = Candidate(
            element_class="door", class_idx=3, wall_idx=0,
            t_min=0.4, t_max=0.6, theta_center=90.0, theta_span=10.0,
            r_mean=3.0, h_min=0.0, h_max=2.0, width_m=1.0,
            num_points=200, world_x=0.0, world_y=3.0,
        )

        mock_scene = MagicMock()
        mock_result = MagicMock()
        mock_result.colors = np.zeros((10, 10, 3), dtype=np.float32)
        mock_scene.render.return_value = mock_result

        with patch("bim_recon.vlm_verifier.query_ollama",
                   return_value="CONFIRMED\nA door is visible."):
            results = verify_candidates(
                [cand], mock_scene, (0.0, 0.0), 0.0, tmp_path,
                element_class="door", skip_vlm=False,
            )

        assert len(results) == 1
        assert results[0].confirmed is True
        assert "door" in results[0].vlm_response
        # Verify image was saved
        assert (tmp_path / results[0].image_path).exists()

    def test_rejected_candidate(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from bim_recon.vlm_verifier import verify_candidates
        from bim_recon.candidate_extractor import Candidate

        cand = Candidate(
            element_class="door", class_idx=3, wall_idx=0,
            t_min=0.1, t_max=0.3, theta_center=0.0, theta_span=5.0,
            r_mean=2.0, h_min=0.0, h_max=2.0, width_m=0.8,
            num_points=150, world_x=2.0, world_y=0.0,
        )

        mock_scene = MagicMock()
        mock_result = MagicMock()
        mock_result.colors = np.zeros((10, 10, 3), dtype=np.float32)
        mock_scene.render.return_value = mock_result

        with patch("bim_recon.vlm_verifier.query_ollama",
                   return_value="REJECTED\nNo door here."):
            results = verify_candidates(
                [cand], mock_scene, (0.0, 0.0), 0.0, tmp_path,
                element_class="door",
            )

        assert len(results) == 1
        assert results[0].confirmed is False

    def test_vlm_error_handled(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from bim_recon.vlm_verifier import verify_candidates
        from bim_recon.candidate_extractor import Candidate

        cand = Candidate(
            element_class="door", class_idx=3, wall_idx=0,
            t_min=0.4, t_max=0.6, theta_center=90.0, theta_span=10.0,
            r_mean=3.0, h_min=0.0, h_max=2.0, width_m=1.0,
            num_points=200, world_x=0.0, world_y=3.0,
        )

        mock_scene = MagicMock()
        mock_result = MagicMock()
        mock_result.colors = np.zeros((10, 10, 3), dtype=np.float32)
        mock_scene.render.return_value = mock_result

        with patch("bim_recon.vlm_verifier.query_ollama",
                   side_effect=ConnectionError("Ollama offline")):
            results = verify_candidates(
                [cand], mock_scene, (0.0, 0.0), 0.0, tmp_path,
                element_class="door",
            )

        assert len(results) == 1
        assert results[0].confirmed is None
        assert "ERROR" in results[0].vlm_response

    def test_empty_candidates(self, tmp_path):
        from unittest.mock import MagicMock
        from bim_recon.vlm_verifier import verify_candidates

        mock_scene = MagicMock()
        results = verify_candidates(
            [], mock_scene, (0.0, 0.0), 0.0, tmp_path,
            element_class="door",
        )
        assert results == []

    def test_up_axis_passed_through(self, tmp_path):
        """Verify up_axis=1 (Y-up) produces correct eye/target in results."""
        from unittest.mock import MagicMock, patch
        from bim_recon.vlm_verifier import verify_candidates
        from bim_recon.candidate_extractor import Candidate

        cand = Candidate(
            element_class="door", class_idx=3, wall_idx=0,
            t_min=0.4, t_max=0.6, theta_center=0.0, theta_span=10.0,
            r_mean=3.0, h_min=0.0, h_max=2.0, width_m=1.0,
            num_points=200, world_x=3.0, world_y=0.0,
        )

        mock_scene = MagicMock()
        mock_result = MagicMock()
        mock_result.colors = np.zeros((10, 10, 3), dtype=np.float32)
        mock_scene.render.return_value = mock_result

        with patch("bim_recon.vlm_verifier.query_ollama",
                   return_value="CONFIRMED"):
            results = verify_candidates(
                [cand], mock_scene, (0.0, 0.0), 0.0, tmp_path,
                element_class="door", up_axis=1,
            )

        # With up_axis=1: eye should have Y as vertical
        eye = results[0].eye
        # h_axes=[0,2], up_axis=1
        # eye[0]=cx=0.0, eye[1]=floor_z+eye_height=1.5, eye[2]=cy=0.0
        assert eye[1] == pytest.approx(1.5)  # Y is vertical
