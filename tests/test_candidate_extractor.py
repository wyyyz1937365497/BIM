"""Unit tests for candidate_extractor — synthetic scan data, no GPU.

Tests:
  - project_point_to_wall: projection geometry
  - _cluster_openings: clustering by gap threshold
  - extract_candidates: full pipeline with synthetic scans
  - prefilter_candidates: filtering by width/points
"""
from __future__ import annotations

import numpy as np
import pytest

from bim_recon.candidate_extractor import (
    Candidate,
    BIM_CLASS_INDICES,
    extract_candidates,
    prefilter_candidates,
    project_point_to_wall,
    _cluster_openings,
)
from bim_recon.virtual_scanner import ScanResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scan(
    points_2d: np.ndarray,
    semantic_labels: np.ndarray,
    height: float,
    center: tuple[float, float] = (0.0, 0.0),
    up_axis: int = 2,
) -> ScanResult:
    """Build a minimal ScanResult for testing."""
    n = len(points_2d)
    angles = np.array([
        np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) % 360
        for p in points_2d
    ])
    dists = np.array([
        np.hypot(p[0] - center[0], p[1] - center[1]) for p in points_2d
    ])
    return ScanResult(
        angles_deg=angles,
        distances=dists,
        points_2d=points_2d,
        height=height,
        center_2d=np.array(center),
        up_axis=up_axis,
        semantic_labels=semantic_labels,
    )


def _make_wall(x1: float, y1: float, x2: float, y2: float) -> dict:
    length = np.hypot(x2 - x1, y2 - y1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": float(length)}


# ---------------------------------------------------------------------------
# project_point_to_wall
# ---------------------------------------------------------------------------

class TestProjectPointToWall:
    def test_point_on_segment_midpoint(self):
        ws = np.array([0.0, 0.0])
        we = np.array([4.0, 0.0])
        t, dist = project_point_to_wall(np.array([2.0, 0.0]), ws, we)
        assert t == pytest.approx(0.5)
        assert dist == pytest.approx(0.0)

    def test_point_offset_from_wall(self):
        ws = np.array([0.0, 0.0])
        we = np.array([4.0, 0.0])
        t, dist = project_point_to_wall(np.array([2.0, 1.0]), ws, we)
        assert t == pytest.approx(0.5)
        assert dist == pytest.approx(1.0)

    def test_point_beyond_end_clamped(self):
        ws = np.array([0.0, 0.0])
        we = np.array([4.0, 0.0])
        t, dist = project_point_to_wall(np.array([6.0, 0.0]), ws, we)
        assert t == pytest.approx(1.0)
        assert dist == pytest.approx(2.0)

    def test_point_before_start_clamped(self):
        ws = np.array([0.0, 0.0])
        we = np.array([4.0, 0.0])
        t, dist = project_point_to_wall(np.array([-2.0, 0.0]), ws, we)
        assert t == pytest.approx(0.0)
        assert dist == pytest.approx(2.0)

    def test_degenerate_segment(self):
        t, dist = project_point_to_wall(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        assert t == 0.0
        assert dist == pytest.approx(np.sqrt(2))


# ---------------------------------------------------------------------------
# _cluster_openings
# ---------------------------------------------------------------------------

class TestClusterOpenings:
    def test_single_cluster(self):
        ts = [0.1, 0.12, 0.14, 0.11, 0.13]
        hs = [1.0, 1.0, 1.5, 0.5, 1.2]
        openings = _cluster_openings(ts, hs, wall_length=5.0, min_gap=0.3, min_pts=3)
        assert len(openings) == 1
        assert openings[0]["num_points"] == 5
        # t range after sort: 0.1-0.14, width = 0.04 * 5.0 = 0.2
        assert openings[0]["width_m"] == pytest.approx(0.04 * 5.0)

    def test_two_clusters_separated_by_gap(self):
        # Cluster 1: t=0.1-0.14, cluster 2: t=0.5-0.54
        # Gap = 0.36 * 5.0 = 1.8m > 0.3m threshold
        ts = [0.1, 0.12, 0.14, 0.5, 0.52, 0.54]
        hs = [1.0] * 6
        openings = _cluster_openings(ts, hs, wall_length=5.0, min_gap=0.3, min_pts=3)
        assert len(openings) == 2

    def test_too_few_points(self):
        ts = [0.1, 0.12]
        hs = [1.0, 1.0]
        openings = _cluster_openings(ts, hs, wall_length=5.0, min_pts=5)
        assert len(openings) == 0

    def test_empty_input(self):
        openings = _cluster_openings([], [], wall_length=5.0)
        assert openings == []


# ---------------------------------------------------------------------------
# extract_candidates
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_door_on_single_wall(self):
        """A wall along x-axis with door points at t=0.4-0.6."""
        wall = _make_wall(0, 0, 5, 0)  # 5m wall along x
        floor_z = 0.0
        center = (2.5, 2.0)  # scan center 2m away from wall

        # Door points: near x=2-3 (t=0.4-0.6), slightly off wall (y=0.1)
        rng = np.random.default_rng(42)
        door_pts = np.column_stack([
            rng.uniform(2.0, 3.0, 50),  # x: 2-3 (door area)
            rng.uniform(-0.1, 0.1, 50),  # y: near wall
        ])
        door_labels = np.full(50, 3)  # door class

        # Wall points elsewhere (should be ignored)
        wall_pts = np.column_stack([
            rng.uniform(0.5, 1.5, 30),
            rng.uniform(-0.1, 0.1, 30),
        ])
        wall_labels = np.zeros(30, dtype=np.int32)

        all_pts = np.vstack([door_pts, wall_pts])
        all_labels = np.concatenate([door_labels, wall_labels.astype(np.int32)])

        scan = _make_scan(all_pts, all_labels, height=1.0, center=center)

        candidates = extract_candidates(
            [scan], [wall], floor_z, center,
            element_class="door", class_idx=3,
        )
        assert len(candidates) == 1
        c = candidates[0]
        assert c.element_class == "door"
        assert c.wall_idx == 0
        assert 0.8 < c.width_m < 1.5  # roughly 1m
        assert c.num_points >= 5

    def test_no_matching_class(self):
        """No door points → no candidates."""
        wall = _make_wall(0, 0, 5, 0)
        pts = np.array([[1.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0])  # wall, not door
        scan = _make_scan(pts, labels, height=1.0)
        candidates = extract_candidates(
            [scan], [wall], 0.0, (2.5, 2.0),
            element_class="door", class_idx=3,
        )
        assert candidates == []

    def test_multiple_walls(self):
        """Door candidates on two different walls."""
        w0 = _make_wall(0, 0, 5, 0)   # bottom wall
        w1 = _make_wall(5, 0, 5, 4)   # right wall
        floor_z = 0.0
        center = (2.5, 2.0)

        rng = np.random.default_rng(42)
        # Door on wall 0
        pts_w0 = np.column_stack([
            rng.uniform(1.0, 2.0, 20), rng.uniform(-0.1, 0.1, 20)
        ])
        # Door on wall 1
        pts_w1 = np.column_stack([
            rng.uniform(4.9, 5.1, 20), rng.uniform(1.0, 2.0, 20)
        ])
        pts = np.vstack([pts_w0, pts_w1])
        labels = np.full(40, 3)
        scan = _make_scan(pts, labels, height=1.0, center=center)

        candidates = extract_candidates(
            [scan], [w0, w1], floor_z, center,
            element_class="door", class_idx=3,
        )
        assert len(candidates) >= 2
        wall_indices = {c.wall_idx for c in candidates}
        assert 0 in wall_indices
        assert 1 in wall_indices


# ---------------------------------------------------------------------------
# prefilter_candidates
# ---------------------------------------------------------------------------

class TestPrefilter:
    def test_filter_by_width(self):
        candidates = [
            Candidate("door", 3, 0, 0.1, 0.3, 90, 10, 3, 0.5, 2, 0.5, 50, 0, 0),
            Candidate("door", 3, 0, 0.4, 0.6, 90, 10, 3, 0.5, 2, 1.2, 200, 0, 0),
        ]
        result = prefilter_candidates(candidates, min_width=0.7, min_points=100)
        assert len(result) == 1
        assert result[0].width_m == 1.2

    def test_filter_by_points(self):
        candidates = [
            Candidate("door", 3, 0, 0.1, 0.3, 90, 10, 3, 0.5, 2, 1.0, 50, 0, 0),
            Candidate("door", 3, 0, 0.4, 0.6, 90, 10, 3, 0.5, 2, 1.0, 200, 0, 0),
        ]
        result = prefilter_candidates(candidates, min_width=0.7, min_points=100)
        assert len(result) == 1
        assert result[0].num_points == 200

    def test_all_pass(self):
        candidates = [
            Candidate("door", 3, 0, 0.1, 0.3, 90, 10, 3, 0.5, 2, 1.0, 200, 0, 0),
        ]
        result = prefilter_candidates(candidates, min_width=0.7, min_points=100)
        assert len(result) == 1

    def test_all_filtered(self):
        candidates = [
            Candidate("door", 3, 0, 0.1, 0.3, 90, 10, 3, 0.5, 2, 0.5, 50, 0, 0),
        ]
        result = prefilter_candidates(candidates, min_width=0.7, min_points=100)
        assert len(result) == 0
