"""Geometry contracts for merged hosted openings."""
from __future__ import annotations

import numpy as np
import pytest

from bim_recon.element_merger import (
    clip_opening_to_wall,
    clip_points_to_wall_span,
)


def test_opening_extent_is_cropped_at_host_wall_endpoint():
    wall = {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0}

    center_x, center_y, width, start_t, end_t = clip_opening_to_wall(
        3.8, 0.4, 1.0, wall,
    )

    assert center_x == pytest.approx(3.65)
    assert center_y == pytest.approx(0.0)
    assert width == pytest.approx(0.7)
    assert start_t == pytest.approx(3.3)
    assert end_t == pytest.approx(4.0)


def test_radar_mask_points_are_projected_and_clipped_to_host_wall():
    wall = {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0}
    points = np.array([
        [-1.0, 0.4],
        [2.0, -0.3],
        [5.0, 0.2],
    ])

    clipped = clip_points_to_wall_span(points, wall, 0.5, 3.5)

    np.testing.assert_allclose(clipped, np.array([
        [0.5, 0.0],
        [2.0, 0.0],
        [3.5, 0.0],
    ]))
