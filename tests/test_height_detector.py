"""Tests for bim_recon.height_detector.

Tests are organised in two tiers:
- **Pure-logic tests** (no GSScene needed): ``_inward_normal``, ``_is_opening``.
- **Mock-scene tests**: a lightweight ``_MockScene`` stand-in for GSScene that
  returns scripted depth/alpha/colour arrays, exercising the two-phase scan
  and fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pytest

from bim_recon.candidate_extractor import Candidate
from bim_recon.gs_scene import GSScene
from bim_recon.height_detector import (
    HeightResult,
    _inward_normal,
    _is_opening,
    detect_element_heights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockRender:
    depth: np.ndarray
    alpha: np.ndarray
    colors: np.ndarray


class _MockScene:
    """Minimal stand-in for GSScene.

    ``depth_fn(height)`` returns the centre-pixel depth at a given world
    height.  This lets each test script a vertical depth profile without
    building a real 3DGS scene.
    """

    def __init__(
        self,
        depth_fn,
        label_fn=None,
        class_idx: int = 4,
        num_classes: int = 9,
        img_size: int = 64,
        device: str = "cpu",
    ):
        self._depth_fn = depth_fn
        self._label_fn = label_fn or (lambda _h: -1)
        self._class_idx = class_idx
        self._num_classes = num_classes
        self._img_size = img_size
        self.device = device
        self.num_gaussians = 100
        # Semantic stubs — height_detector checks these are non-None.
        self.semantic_querier = True
        self.feat = True
        self.colors = np.zeros((100, 3), dtype=np.float32)

    # ---- GSScene API used by height_detector --------------------------

    def render(self, pose, width, height, fov_degrees):
        """Return a scripted RenderResult based on the camera height."""
        # Extract camera height from pose position (up_axis = 2).
        eye_z = float(pose.position[2])
        d = self._depth_fn(eye_z)
        lbl = self._label_fn(eye_z)

        sz = width  # square
        depth_arr = np.full((sz, sz), 1.0, dtype=np.float32)
        alpha_arr = np.ones((sz, sz), dtype=np.float32)
        color_arr = np.zeros((sz, sz, 3), dtype=np.float32)

        c = sz // 2
        if d < 0:
            alpha_arr[c, c] = 0.0       # no hit
        else:
            depth_arr[c, c] = d
            alpha_arr[c, c] = 1.0

        if lbl >= 0 and self._num_classes > 1:
            color_arr[c, c, 0] = lbl / (self._num_classes - 1)

        return _MockRender(depth_arr, alpha_arr, color_arr)

    # ---- Semantic stub ------------------------------------------------

    @property
    def _mock_num_classes(self):
        return self._num_classes

    # height_detector calls querier.get_dominant_labels() + num_classes;
    # we intercept by providing a duck-typed object.
    class _QuerierStub:
        def __init__(self, parent):
            self._parent = parent

        def get_dominant_labels(self):
            return np.zeros(100, dtype=np.int32)

        @property
        def num_classes(self):
            return self._parent._num_classes

    @property
    def semantic_querier(self):
        return self._QuerierStub(self)

    @semantic_querier.setter
    def semantic_querier(self, val):
        pass  # accept any truthy value from __init__


def _make_candidate(
    wx: float = 0.0,
    wy: float = 5.0,
    h_min: float = 0.15,
    h_max: float = 1.86,
) -> Candidate:
    return Candidate(
        element_class="window",
        class_idx=4,
        wall_idx=0,
        t_min=0.3,
        t_max=0.7,
        theta_center=0.0,
        theta_span=10.0,
        r_mean=5.0,
        h_min=h_min,
        h_max=h_max,
        width_m=1.0,
        num_points=100,
        world_x=wx,
        world_y=wy,
    )


def _wall_y5() -> dict:
    """Horizontal wall segment at y=5, from x=-5 to x=5."""
    return {"x1": -5.0, "y1": 5.0, "x2": 5.0, "y2": 5.0, "length": 10.0}


# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------

class TestInwardNormal:
    def test_horizontal_wall_center_below(self):
        """Wall at y=5, room centre at origin → normal points toward -y."""
        wall = _wall_y5()
        n = _inward_normal(wall, (0.0, 0.0))
        assert n[0] == pytest.approx(0.0, abs=1e-9)
        assert n[1] == pytest.approx(-1.0, abs=1e-9)

    def test_horizontal_wall_center_above(self):
        """Wall at y=5, room centre at y=10 → normal points toward +y."""
        wall = _wall_y5()
        n = _inward_normal(wall, (0.0, 10.0))
        assert n[0] == pytest.approx(0.0, abs=1e-9)
        assert n[1] == pytest.approx(1.0, abs=1e-9)

    def test_vertical_wall_center_right(self):
        """Wall along y-axis at x=5, centre at origin → normal toward -x."""
        wall = {"x1": 5.0, "y1": -5.0, "x2": 5.0, "y2": 5.0, "length": 10.0}
        n = _inward_normal(wall, (0.0, 0.0))
        assert n[0] == pytest.approx(-1.0, abs=1e-9)
        assert n[1] == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_wall(self):
        """Zero-length wall → fallback unit vector."""
        wall = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "length": 0.0}
        n = _inward_normal(wall, (1.0, 1.0))
        assert np.linalg.norm(n) == pytest.approx(1.0)

    def test_unit_length(self):
        wall = {"x1": 0.0, "y1": 0.0, "x2": 3.0, "y2": 4.0, "length": 5.0}
        n = _inward_normal(wall, (0.0, 0.0))
        assert np.linalg.norm(n) == pytest.approx(1.0)


class TestIsOpening:
    def test_wall_surface_not_opening(self):
        assert not _is_opening(
            depth=1.0, label=0, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )

    def test_depth_void_is_opening(self):
        assert _is_opening(
            depth=3.0, label=0, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )

    def test_semantic_match_is_opening(self):
        """Even at wall depth, matching semantics counts as opening."""
        assert _is_opening(
            depth=1.0, label=4, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )

    def test_no_hit_no_semantic_not_opening(self):
        assert not _is_opening(
            depth=-1.0, label=None, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )

    def test_no_hit_with_semantic_is_opening(self):
        assert _is_opening(
            depth=-1.0, label=4, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )

    def test_no_hit_wrong_semantic_not_opening(self):
        assert not _is_opening(
            depth=-1.0, label=3, class_idx=4,
            ref_depth=1.0, depth_threshold=0.15,
        )


# ---------------------------------------------------------------------------
# Mock-scene tests
# ---------------------------------------------------------------------------

class TestDetectElementHeights:
    """Tests with scripted depth profiles via _MockScene."""

    def test_window_opening_detected(self):
        """Wall solid except 0.8–1.6 m where depth jumps to 3.0."""
        floor_z, ceil_z = 0.0, 2.5

        def depth_fn(h):
            if 0.8 <= h <= 1.6:
                return 3.0          # void / opening
            if h < 0.05 or h > 2.45:
                return -1.0         # out of bounds
            return 1.0              # solid wall

        scene = cast(GSScene, _MockScene(depth_fn, class_idx=4))
        cand = _make_candidate()
        wall = _wall_y5()

        result = detect_element_heights(
            scene, cand, wall, floor_z, ceil_z,
            scan_center=(0.0, 0.0), class_idx=4,
            coarse_step=0.2, fine_step=0.05,
            camera_dist=1.0, depth_threshold=0.15,
        )

        assert result.method == "depth+semantic"
        # Sill should be near 0.8 m, header near 1.6 m.
        assert 0.6 <= result.sill_height <= 1.0
        assert 1.4 <= result.header_height <= 1.8
        assert result.header_height > result.sill_height

    def test_no_opening_fallback(self):
        """Wall is solid everywhere → fallback to candidate h_min/h_max."""
        floor_z, ceil_z = 0.0, 2.5

        def depth_fn(h):
            return 1.0  # always solid

        scene = cast(GSScene, _MockScene(depth_fn, class_idx=4))
        cand = _make_candidate(h_min=0.3, h_max=1.5)
        wall = _wall_y5()

        result = detect_element_heights(
            scene, cand, wall, floor_z, ceil_z,
            scan_center=(0.0, 0.0), class_idx=4,
            coarse_step=0.2, fine_step=0.05,
        )

        assert result.method == "fallback"
        assert result.sill_height == pytest.approx(0.3)
        assert result.header_height == pytest.approx(1.5)

    def test_door_full_height_opening(self):
        """Door opening from 0 to 2.0 m → sill near floor, header near 2.0."""
        floor_z, ceil_z = 0.0, 2.5

        def depth_fn(h):
            if 0.05 <= h <= 2.0:
                return 3.0      # opening
            return 1.0          # solid (lintel above 2.0)

        scene = cast(GSScene, _MockScene(depth_fn, class_idx=3))
        cand = Candidate(
            element_class="door", class_idx=3, wall_idx=0,
            t_min=0.2, t_max=0.8,
            theta_center=0.0, theta_span=15.0, r_mean=5.0,
            h_min=0.1, h_max=2.0,
            width_m=0.9, num_points=200,
            world_x=0.0, world_y=5.0,
        )
        wall = _wall_y5()

        result = detect_element_heights(
            scene, cand, wall, floor_z, ceil_z,
            scan_center=(0.0, 0.0), class_idx=3,
            coarse_step=0.2, fine_step=0.05,
        )

        assert result.method == "depth+semantic"
        assert result.sill_height < 0.3   # near floor
        assert result.header_height > 1.7  # near lintel

    def test_semantic_only_detection(self):
        """Opening only detectable via semantic label, not depth."""
        floor_z, ceil_z = 0.0, 2.5

        def depth_fn(h):
            return 1.0  # always at wall depth

        def label_fn(h):
            if 0.9 <= h <= 1.8:
                return 4   # window class
            return 0       # wall class

        scene = cast(GSScene, _MockScene(depth_fn, label_fn, class_idx=4))
        cand = _make_candidate()
        wall = _wall_y5()

        result = detect_element_heights(
            scene, cand, wall, floor_z, ceil_z,
            scan_center=(0.0, 0.0), class_idx=4,
            coarse_step=0.2, fine_step=0.05,
        )

        assert result.method == "depth+semantic"
        # Should find the opening via semantic signal.
        assert result.sill_height < result.header_height

    def test_height_result_fields(self):
        """HeightResult has all expected fields with correct types."""
        r = HeightResult(
            sill_height=0.9, header_height=2.1,
            element_height=1.2, confidence=0.8,
            method="depth+semantic",
        )
        assert r.element_height == pytest.approx(r.header_height - r.sill_height)
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.method, str)
