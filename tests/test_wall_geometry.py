"""Contracts for removing Gaussian-blur chamfers from wall corners."""
from __future__ import annotations

import pytest

from bim_recon.pipeline_runner import snap_wall_endpoints
from bim_recon.wall_geometry import fit_short_corner_walls, merge_overlapping_walls


def _chamfered_corner():
    return [
        {"x1": 0.0, "y1": 0.0, "x2": 0.9, "y2": 0.0},
        {"x1": 0.9, "y1": 0.0, "x2": 1.0, "y2": 0.1},
        {"x1": 1.0, "y1": 0.1, "x2": 1.0, "y2": 1.0},
    ]


def test_short_chamfer_is_replaced_by_adjacent_wall_ray_intersection():
    fitted, source_indices = fit_short_corner_walls(_chamfered_corner())

    assert source_indices == [0, 2]
    assert len(fitted) == 2
    assert fitted[0]["x2"] == pytest.approx(1.0)
    assert fitted[0]["y2"] == pytest.approx(0.0)
    assert fitted[1]["x1"] == pytest.approx(1.0)
    assert fitted[1]["y1"] == pytest.approx(0.0)


def test_endpoint_snapping_pipeline_applies_corner_fitting():
    fitted = snap_wall_endpoints(_chamfered_corner(), threshold=0.05)

    assert len(fitted) == 2
    assert fitted[0]["x2"] == pytest.approx(fitted[1]["x1"])
    assert fitted[0]["y2"] == pytest.approx(fitted[1]["y1"])


def test_overlapping_collinear_walls_are_unioned_with_source_aliases():
    merged, source_groups = merge_overlapping_walls([
        {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0},
        {"x1": 0.02, "y1": 0.01, "x2": 4.02, "y2": 0.01},
        {"x1": 4.0, "y1": 0.0, "x2": 4.0, "y2": 3.0},
    ])

    assert source_groups == [[0, 1], [2]]
    assert len(merged) == 2
    assert merged[0]["length"] == pytest.approx(4.02)


def test_wall_snapping_removes_overlapping_segments():
    walls = snap_wall_endpoints([
        {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0},
        {"x1": 0.02, "y1": 0.01, "x2": 4.02, "y2": 0.01},
        {"x1": 4.0, "y1": 0.0, "x2": 4.0, "y2": 3.0},
    ], threshold=0.001)

    assert len(walls) == 2
    assert max(wall["length"] for wall in walls) == pytest.approx(4.02)
