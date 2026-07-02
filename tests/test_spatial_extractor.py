"""Tests for bim_recon.spatial_extractor.

Covers:
- Wall inward normal computation
- Elevation parameter computation (FOV, extents)
- Normalized bbox → wall-local metre mapping (pure math)
- End-to-end extract_spatial with mock Falcon client
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bim_recon.spatial_extractor import (
    ElevationParams,
    SpatialResult,
    _wall_direction,
    _wall_inward_normal,
    bbox_to_wall_coords,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wall(x1, y1, x2, y2, length=None):
    import math
    if length is None:
        length = math.hypot(x2 - x1, y2 - y1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length}


def _elevation_params(extent_h=4.0, target_along=2.0, cam_h=1.0,
                      wall_length=5.0, wall_start=None, wall_dir=None):
    if wall_start is None:
        wall_start = np.array([0.0, 0.0])
    if wall_dir is None:
        wall_dir = np.array([1.0, 0.0])
    return ElevationParams(
        camera_dist=2.5, fov_degrees=50.0, img_size=800,
        wall_length=wall_length, wall_dir=wall_dir, wall_start=wall_start,
        target_along=target_along, cam_h_above_floor=cam_h,
        extent_h=extent_h, extent_v=extent_h,
    )


# ---------------------------------------------------------------------------
# Wall geometry
# ---------------------------------------------------------------------------

class TestWallNormal:
    def test_horizontal_wall_normal_points_up(self):
        """Wall along X axis, center above → normal points +Y."""
        wall = _wall(0, 0, 5, 0)
        n = _wall_inward_normal(wall, (2.5, 3.0))
        assert n[1] > 0  # toward center which is +Y

    def test_vertical_wall_normal_points_right(self):
        """Wall along Y axis, center to the right → normal points +X."""
        wall = _wall(0, 0, 0, 5)
        n = _wall_inward_normal(wall, (3.0, 2.5))
        assert n[0] > 0

    def test_normal_is_unit(self):
        wall = _wall(0, 0, 3, 4)
        n = _wall_inward_normal(wall, (1.5, 2.0))
        assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-9

    def test_wall_direction(self):
        wall = _wall(0, 0, 3, 4)
        d = _wall_direction(wall)
        expected = np.array([3, 4]) / 5.0
        np.testing.assert_allclose(d, expected)


# ---------------------------------------------------------------------------
# bbox_to_wall_coords — pure math
# ---------------------------------------------------------------------------

class TestBboxToWallCoords:
    def test_centre_bbox_at_image_centre(self):
        """Bbox at image centre → element centred at target_along, cam_h."""
        params = _elevation_params(extent_h=4.0, target_along=2.0, cam_h=1.0)
        bbox = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3}
        r = bbox_to_wall_coords(bbox, params, floor_z=-1.0, ceiling_z=2.0)

        assert r is not None
        # Centre along wall = target_along
        mid = (r.t_min + r.t_max) / 2 * params.wall_length
        assert abs(mid - 2.0) < 0.01
        # Sill/header centred on cam_h
        mid_h = (r.sill_height + r.header_height) / 2
        assert abs(mid_h - 1.0) < 0.01

    def test_bbox_at_top_is_higher(self):
        """Bbox near top of image → header near ceiling."""
        params = _elevation_params(extent_h=4.0, cam_h=1.0)
        # Top of image: y ≈ 0.1, height 0.2 → y1=0.0
        bbox_top = {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.2}
        r_top = bbox_to_wall_coords(bbox_top, params, floor_z=0.0, ceiling_z=2.0)
        assert r_top is not None

        bbox_bot = {"x": 0.5, "y": 0.9, "w": 0.3, "h": 0.2}
        r_bot = bbox_to_wall_coords(bbox_bot, params, floor_z=0.0, ceiling_z=2.0)
        assert r_bot is not None

        assert r_top.header_height > r_bot.header_height
        assert r_top.sill_height > r_bot.sill_height

    def test_width_maps_to_along_wall(self):
        params = _elevation_params(extent_h=4.0, target_along=2.0)
        bbox = {"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.2}
        r = bbox_to_wall_coords(bbox, params, floor_z=0.0, ceiling_z=2.0)
        assert r is not None
        # Width 0.5 of 4.0m extent = 2.0m
        assert abs(r.width_m - 2.0) < 0.01

    def test_left_bbox_maps_left(self):
        """Bbox on left side of image → lower t_min."""
        params = _elevation_params(extent_h=4.0, target_along=2.0, wall_length=5.0)
        bbox = {"x": 0.25, "y": 0.5, "w": 0.1, "h": 0.1}
        r = bbox_to_wall_coords(bbox, params, floor_z=0.0, ceiling_z=2.0)
        assert r is not None
        # Centre at norm_x=0.25 → along = 2.0 + (0.25-0.5)*4.0 = 1.0
        mid_along = (r.t_min + r.t_max) / 2 * 5.0
        assert abs(mid_along - 1.0) < 0.01

    def test_degenerate_bbox_returns_none(self):
        params = _elevation_params()
        assert bbox_to_wall_coords({"x": 0.5, "y": 0.5, "w": 0, "h": 0.2}, params, 0.0, 2.0) is None
        assert bbox_to_wall_coords({"x": 0.5, "y": 0.5, "w": 0.2, "h": 0}, params, 0.0, 2.0) is None

    def test_clamping_to_wall_extent(self):
        """Bbox extending beyond wall → clamped to [0, wall_length]."""
        params = _elevation_params(extent_h=10.0, target_along=1.0, wall_length=5.0)
        bbox = {"x": 0.5, "y": 0.5, "w": 1.0, "h": 0.1}  # full width
        r = bbox_to_wall_coords(bbox, params, floor_z=0.0, ceiling_z=2.0)
        assert r is not None
        assert 0.0 <= r.t_min * 5.0
        assert r.t_max * 5.0 <= 5.0 + 0.01


# ---------------------------------------------------------------------------
# End-to-end with mock Falcon client
# ---------------------------------------------------------------------------

class TestExtractSpatial:
    """Mock the FalconClient and GSScene to test the full flow."""

    def _make_mock_candidate(self):
        from bim_recon.candidate_extractor import Candidate
        return Candidate(
            element_class="window", class_idx=4, wall_idx=0,
            t_min=0.3, t_max=0.7, theta_center=90.0, theta_span=30.0,
            r_mean=2.5, h_min=0.5, h_max=1.5, width_m=1.0,
            num_points=100, world_x=2.5, world_y=0.0,
        )

    def _make_mock_falcon(self, detections):
        mock = MagicMock()
        mock.segment.return_value = detections
        return mock

    def _make_mock_scene(self):
        """Minimal mock that returns a render result with colors."""
        from bim_recon.gs_scene import RenderResult
        mock = MagicMock()
        mock.render.return_value = RenderResult(
            colors=np.zeros((800, 800, 3), dtype=np.float32),
            depth=np.zeros((800, 800), dtype=np.float32),
            alpha=np.ones((800, 800), dtype=np.float32),
        )
        return mock

    def test_successful_extraction(self):
        from bim_recon.falcon_client import FalconDetection
        from bim_recon.spatial_extractor import extract_spatial

        cand = self._make_mock_candidate()
        wall = _wall(0, 0, 5, 0)
        scene = self._make_mock_scene()
        falcon = self._make_mock_falcon([
            FalconDetection(
                bbox={"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
                mask_bbox={"x": 0.48, "y": 0.52, "w": 0.19, "h": 0.28},
                mask_area_ratio=0.05,
            )
        ])

        result = extract_spatial(
            falcon, scene, cand, wall,
            floor_z=-1.0, ceiling_z=1.0,
            scan_center=(2.5, 3.0),
            element_name="window",
        )

        assert result is not None
        assert result.method == "falcon_segmentation"
        assert 0.0 < result.element_height < 3.0
        assert 0.0 < result.width_m <= 5.0
        assert result.confidence > 0

    def test_no_detections_returns_none(self):
        from bim_recon.spatial_extractor import extract_spatial

        cand = self._make_mock_candidate()
        wall = _wall(0, 0, 5, 0)
        scene = self._make_mock_scene()
        falcon = self._make_mock_falcon([])

        result = extract_spatial(
            falcon, scene, cand, wall,
            floor_z=-1.0, ceiling_z=1.0,
            scan_center=(2.5, 3.0),
            element_name="window",
        )
        assert result is None

    def test_detection_only_fallback(self):
        """When mask_bbox is None but bbox exists → method=falcon_detection."""
        from bim_recon.falcon_client import FalconDetection
        from bim_recon.spatial_extractor import extract_spatial

        cand = self._make_mock_candidate()
        wall = _wall(0, 0, 5, 0)
        scene = self._make_mock_scene()
        falcon = self._make_mock_falcon([
            FalconDetection(
                bbox={"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
                mask_bbox=None,
                mask_area_ratio=None,
            )
        ])

        result = extract_spatial(
            falcon, scene, cand, wall,
            floor_z=-1.0, ceiling_z=1.0,
            scan_center=(2.5, 3.0),
            element_name="window",
        )
        assert result is not None
        assert result.method == "falcon_detection"

    def test_picks_largest_mask(self):
        """When multiple detections, picks the one with largest mask_area_ratio."""
        from bim_recon.falcon_client import FalconDetection
        from bim_recon.spatial_extractor import extract_spatial

        cand = self._make_mock_candidate()
        wall = _wall(0, 0, 5, 0)
        scene = self._make_mock_scene()
        falcon = self._make_mock_falcon([
            FalconDetection(
                bbox={"x": 0.3, "y": 0.5, "w": 0.1, "h": 0.1},
                mask_bbox={"x": 0.3, "y": 0.5, "w": 0.1, "h": 0.1},
                mask_area_ratio=0.01,
            ),
            FalconDetection(
                bbox={"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.4},
                mask_bbox={"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.4},
                mask_area_ratio=0.12,
            ),
        ])

        result = extract_spatial(
            falcon, scene, cand, wall,
            floor_z=-1.0, ceiling_z=1.0,
            scan_center=(2.5, 3.0),
            element_name="window",
        )
        assert result is not None
        # Larger detection: width 0.3 * extent ≈ larger width_m
        assert result.width_m > 0.5
