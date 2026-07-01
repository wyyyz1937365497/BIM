"""Unit tests for vlm_verifier — pure math, no GPU/Ollama needed.

Tests:
  - compute_polar: θ/r computation from world coordinates
  - candidate_to_viewpoint: polar → camera pose mapping
  - _parse_vlm_response: CONFIRMED/REJECTED parsing
  - _build_prompt: prompt structure
"""
from __future__ import annotations

import math

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
