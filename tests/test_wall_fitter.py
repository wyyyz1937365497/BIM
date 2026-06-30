"""Unit tests for WallFitter -- synthetic point clouds, TDD.

Synthesizes known wall planes with Gaussian noise, then verifies the
fitter recovers correct geometry. No dependency on real feat.pt.
"""
from __future__ import annotations

import numpy as np
import pytest

from bim_recon.wall_fitter import WallFit, WallFitter
from bim_recon.wall_fitter import wallfit_to_line_based_element, wallfit_to_wall_segment


# ---------------------------------------------------------------------------
# Synthetic wall point cloud generator
# ---------------------------------------------------------------------------

def _make_wall_points(
    center: np.ndarray,
    normal: np.ndarray,
    length: float,
    height: float,
    thickness: float = 0.12,
    n: int = 2000,
    noise: float = 0.02,
    up_axis: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic wall point cloud on a plane with noise.

    Args:
        center: (3,) wall center point (at floor level on the up axis).
        normal: (3,) wall outward normal (will be normalized).
        length: wall length along the horizontal direction.
        height: wall height along the up axis.
        thickness: slab thickness along the normal.
        n: number of points.
        noise: Gaussian noise std (m).
        up_axis: vertical axis (0=x, 1=y, 2=z).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    up = np.zeros(3)
    up[up_axis] = 1.0
    normal = normal / np.linalg.norm(normal)

    # Wall direction = cross(normal, up), normalized
    direction = np.cross(normal, up)
    direction /= np.linalg.norm(direction)

    t = rng.uniform(-length / 2, length / 2, n)
    h = rng.uniform(0, height, n)
    d = rng.uniform(-thickness / 2, thickness / 2, n)

    points = (
        center
        + t[:, None] * direction
        + h[:, None] * up
        + d[:, None] * normal
    )
    points += rng.normal(0, noise, points.shape)
    return points


# ---------------------------------------------------------------------------
# T1: Basic fit tests
# ---------------------------------------------------------------------------

class TestWallFitBasic:
    """Tests for WallFitter.fit() core RANSAC + _plane_to_wall."""

    def test_fit_finds_4_walls(self):
        """4 walls of a 4x4m rectangular room -> fit returns 4 WallFit."""
        rng = np.random.default_rng(42)
        walls_data = [
            (np.array([2.0, 4.0, 0.0]), np.array([0.0, 1.0, 0.0]), 4.0),
            (np.array([2.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 4.0),
            (np.array([4.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0]), 4.0),
            (np.array([0.0, 2.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 4.0),
        ]
        all_points = []
        for center, normal, length in walls_data:
            pts = _make_wall_points(center, normal, length, height=2.8, rng=rng)
            all_points.append(pts)
        points = np.concatenate(all_points)

        fitter = WallFitter(min_inliers=500)
        walls = fitter.fit(points, up_axis=2)

        assert len(walls) == 4, f"Expected 4 walls, got {len(walls)}"

    def test_wall_length_within_5pct(self):
        """Single 4m wall -> length within 5% (3.8-4.2m)."""
        rng = np.random.default_rng(42)
        pts = _make_wall_points(
            np.array([2.0, 4.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            length=4.0,
            height=2.8,
            rng=rng,
        )
        fitter = WallFitter(min_inliers=500)
        walls = fitter.fit(pts, up_axis=2)

        assert len(walls) == 1
        assert abs(walls[0].length - 4.0) < 0.2, \
            f"Length {walls[0].length:.3f}, expected ~4.0"

    def test_wall_height_within_5pct(self):
        """Single 2.8m wall -> height within 5% (2.66-2.94m)."""
        rng = np.random.default_rng(42)
        pts = _make_wall_points(
            np.array([2.0, 4.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            length=4.0,
            height=2.8,
            rng=rng,
        )
        fitter = WallFitter(min_inliers=500)
        walls = fitter.fit(pts, up_axis=2)

        assert len(walls) == 1
        assert abs(walls[0].height - 2.8) < 0.14, \
            f"Height {walls[0].height:.3f}, expected ~2.8"

    def test_fit_empty_input(self):
        """Empty point cloud -> empty list, no crash."""
        fitter = WallFitter()
        walls = fitter.fit(np.zeros((0, 3)), up_axis=2)
        assert walls == []

    def test_min_inliers_filters_small_planes(self):
        """100-point wall (< min_inliers=500) -> filtered out."""
        rng = np.random.default_rng(42)
        pts = _make_wall_points(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            length=4.0,
            height=2.8,
            n=100,
            rng=rng,
        )
        fitter = WallFitter(min_inliers=500)
        walls = fitter.fit(pts, up_axis=2)
        assert len(walls) == 0


# ---------------------------------------------------------------------------
# T2: Merge + gravity alignment tests
# ---------------------------------------------------------------------------

class TestMergeAndAlign:
    """Tests for _merge_walls (occlusion bridging) + _gravity_align."""

    def test_merge_same_wall_split_in_two(self):
        """Two coplanar fragments of the same wall -> merge to 1."""
        rng = np.random.default_rng(42)
        pts1 = _make_wall_points(
            np.array([1.0, 4.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=2.0, height=2.8, rng=rng,
        )
        pts2 = _make_wall_points(
            np.array([4.0, 4.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=2.0, height=2.8, rng=rng,
        )
        w1 = WallFit(
            p0=np.array([0.0, 4.0, 0.0]), p1=np.array([2.0, 4.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=len(pts1), confidence=0.5, inlier_pts=pts1,
        )
        w2 = WallFit(
            p0=np.array([3.0, 4.0, 0.0]), p1=np.array([5.0, 4.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=len(pts2), confidence=0.5, inlier_pts=pts2,
        )
        fitter = WallFitter()
        merged = fitter._merge_walls([w1, w2], up_axis=2)
        assert len(merged) == 1, f"Expected 1 merged wall, got {len(merged)}"
        # Merged wall spans from x~0 to x~5 (both fragments)
        assert abs(merged[0].length - 5.0) < 0.5, \
            f"Merged length {merged[0].length:.2f}, expected ~5.0"

    def test_merge_keeps_parallel_non_coplanar(self):
        """Two parallel walls 5m apart -> keep as 2 (not coplanar)."""
        rng = np.random.default_rng(42)
        pts1 = _make_wall_points(
            np.array([2.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=4.0, height=2.8, rng=rng,
        )
        pts2 = _make_wall_points(
            np.array([2.0, 5.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=4.0, height=2.8, rng=rng,
        )
        w1 = WallFit(
            p0=np.array([0.0, 0.0, 0.0]), p1=np.array([4.0, 0.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=len(pts1), confidence=0.5, inlier_pts=pts1,
        )
        w2 = WallFit(
            p0=np.array([0.0, 5.0, 0.0]), p1=np.array([4.0, 5.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=len(pts2), confidence=0.5, inlier_pts=pts2,
        )
        fitter = WallFitter()
        merged = fitter._merge_walls([w1, w2], up_axis=2)
        assert len(merged) == 2, f"Expected 2 walls (non-coplanar), got {len(merged)}"

    def test_gravity_align_zero_up_component(self):
        """Gravity-aligned normal has zero up-axis component."""
        wall = WallFit(
            p0=np.array([0.0, 4.0, 0.0]), p1=np.array([4.0, 4.0, 0.0]),
            height=2.8, normal=np.array([0.1, 0.9, 0.1]), thickness=0.12,
            num_inliers=1000, confidence=1.0,
        )
        fitter = WallFitter()
        aligned = fitter._gravity_align(wall, up_axis=2)
        assert abs(aligned.normal[2]) < 0.01, \
            f"Up component {aligned.normal[2]:.4f}, expected ~0"
        # Normal should still be unit length (in horizontal plane)
        assert abs(np.linalg.norm(aligned.normal) - 1.0) < 0.01

    def test_merge_wall_with_mid_gap(self):
        """Occlusion completion: 5m wall with 1.5m gap in middle -> 1 wall."""
        rng = np.random.default_rng(42)
        pts = _make_wall_points(
            np.array([2.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=5.0, height=2.8, rng=rng,
        )
        # Remove middle 1.5m (simulating furniture occlusion)
        mask = (pts[:, 0] < 1.75) | (pts[:, 0] > 3.25)
        pts_gapped = pts[mask]

        fitter = WallFitter(min_inliers=300)
        walls = fitter.fit(pts_gapped, up_axis=2)
        assert len(walls) == 1, f"Expected 1 wall spanning gap, got {len(walls)}"
        assert abs(walls[0].length - 5.0) < 0.5, \
            f"Length {walls[0].length:.2f}, expected ~5.0 (spanning occlusion gap)"

    def test_wall_not_split_at_opening(self):
        """Opening continuity: 4m wall with 0.9m door gap -> 1 wall."""
        rng = np.random.default_rng(42)
        pts = _make_wall_points(
            np.array([2.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            length=4.0, height=2.8, rng=rng,
        )
        # Remove middle 0.9m (simulating door opening)
        mask = (pts[:, 0] < 1.55) | (pts[:, 0] > 2.45)
        pts_gapped = pts[mask]

        fitter = WallFitter(min_inliers=300)
        walls = fitter.fit(pts_gapped, up_axis=2)
        assert len(walls) == 1, f"Expected 1 wall (not split at opening), got {len(walls)}"
        assert abs(walls[0].length - 4.0) < 0.4, \
            f"Length {walls[0].length:.2f}, expected ~4.0 (continuous across door)"


# ---------------------------------------------------------------------------
# T3: Endpoint refinement + height extraction tests
# ---------------------------------------------------------------------------

class TestRefineAndHeight:
    """Tests for _refine_endpoints (wall-wall intersection) + _compute_heights."""

    def test_refine_l_shaped_corner(self):
        """L-shaped walls with gaps at corner -> snapped to shared corner."""
        fitter = WallFitter()
        # Wall A: along x-axis, p0 slightly past origin
        w1 = WallFit(
            p0=np.array([0.15, 0.0, 0.0]),
            p1=np.array([4.0, 0.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=1000, confidence=0.5,
        )
        # Wall B: along y-axis, p0 slightly past origin
        w2 = WallFit(
            p0=np.array([0.0, 0.15, 0.0]),
            p1=np.array([0.0, 4.0, 0.0]),
            height=2.8, normal=np.array([1.0, 0.0, 0.0]), thickness=0.12,
            num_inliers=1000, confidence=0.5,
        )
        refined = fitter._refine_endpoints([w1, w2], up_axis=2)

        # Both p0 endpoints should snap to ~[0, 0]
        corner_1 = refined[0].p0[:2]
        corner_2 = refined[1].p0[:2]
        assert np.linalg.norm(corner_1) < 0.05, \
            f"Corner 1 {corner_1} not at origin"
        assert np.linalg.norm(corner_2) < 0.05, \
            f"Corner 2 {corner_2} not at origin"
        assert np.linalg.norm(corner_1 - corner_2) < 0.05, \
            f"Corners don't match: {corner_1} vs {corner_2}"

    def test_refine_isolated_wall_keeps_pca(self):
        """Single wall (no neighbor) -> endpoints unchanged."""
        fitter = WallFitter()
        wall = WallFit(
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([4.0, 0.0, 0.0]),
            height=2.8, normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=1000, confidence=1.0,
        )
        refined = fitter._refine_endpoints([wall], up_axis=2)
        assert len(refined) == 1
        assert np.allclose(refined[0].p0, wall.p0), "p0 changed (should be unchanged)"
        assert np.allclose(refined[0].p1, wall.p1), "p1 changed (should be unchanged)"

    def test_height_from_floor_ceiling(self):
        """Height overridden by floor_z→ceiling_z distance."""
        fitter = WallFitter()
        wall = WallFit(
            p0=np.array([0.0, 0.0, 0.1]),
            p1=np.array([4.0, 0.0, 0.1]),
            height=2.5,  # wrong height from inliers
            normal=np.array([0.0, 1.0, 0.0]), thickness=0.12,
            num_inliers=1000, confidence=1.0,
        )
        refined = fitter._compute_heights(
            [wall], floor_z=0.0, ceiling_z=2.8, up_axis=2,
        )
        assert abs(refined[0].height - 2.8) < 0.01, \
            f"Height {refined[0].height}, expected 2.8"
        assert abs(refined[0].p0[2] - 0.0) < 0.01, "p0 not at floor level"
        assert abs(refined[0].p1[2] - 0.0) < 0.01, "p1 not at floor level"


# ---------------------------------------------------------------------------
# T5: Revit conversion tests
# ---------------------------------------------------------------------------

class TestRevitConversion:
    """Tests for wallfit_to_line_based_element + wallfit_to_wall_segment."""

    def test_to_line_based_element_units(self):
        """WallFit (meters) -> params (millimeters)."""
        wall = WallFit(
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([4.0, 0.0, 0.0]),
            height=2.8,
            normal=np.array([0.0, 1.0, 0.0]),
            thickness=0.24,
            num_inliers=1000,
            confidence=1.0,
        )
        params = wallfit_to_line_based_element(wall)
        assert params["category"] == "OST_Walls"
        assert params["locationLine"]["p0"]["x"] == 0.0
        assert params["locationLine"]["p1"]["x"] == 4000.0  # 4m -> 4000mm
        assert params["thickness"] == 240.0  # 0.24m -> 240mm
        assert params["height"] == 2800.0  # 2.8m -> 2800mm

    def test_to_wall_segment_projection(self):
        """3D WallFit -> 2D WallSegment (up axis dropped)."""
        wall = WallFit(
            p0=np.array([1.0, 2.0, 0.5]),
            p1=np.array([5.0, 2.0, 0.5]),
            height=2.8,
            normal=np.array([0.0, 1.0, 0.0]),
            thickness=0.24,
            num_inliers=1000,
            confidence=1.0,
        )
        seg = wallfit_to_wall_segment(wall, up_axis=2)
        assert seg.x1 == 1.0 and seg.y1 == 2.0
        assert seg.x2 == 5.0 and seg.y2 == 2.0
        # z=0.5 dropped (not in WallSegment)

    def test_to_wall_segment_thickness_preserved(self):
        """Thickness correctly passed through."""
        wall = WallFit(
            p0=np.array([0.0, 0.0, 0.0]),
            p1=np.array([3.0, 0.0, 0.0]),
            height=2.8,
            normal=np.array([0.0, 1.0, 0.0]),
            thickness=0.15,
            num_inliers=1000,
            confidence=1.0,
        )
        seg = wallfit_to_wall_segment(wall)
        assert seg.thickness == 0.15
