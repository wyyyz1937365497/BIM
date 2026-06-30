"""Tests for floorplan registration and FloorPlanGuidedFitter."""
from __future__ import annotations

import numpy as np
import pytest

from bim_recon.floorplan import ManualProvider, WallSegment
from bim_recon.floorplan_registration import register_floorplan
from bim_recon.wall_fitter import FloorPlanGuidedFitter, WallFitter


class TestRegistration:
    """Auto-registration of a 2D floorplan to a 3DGS horizontal plane."""

    def test_register_rectangle_center_aligns(self) -> None:
        fp = ManualProvider.from_rectangle(5.0, 4.0).get_floorplan()
        # 3DGS wall footprint centered at (10, -2) with the same orientation.
        rng = np.random.default_rng(0)
        gs_pts = np.array([
            [8.0, -4.0], [12.0, -4.0], [12.0, 0.0], [8.0, 0.0],
        ], dtype=np.float64)
        gs_pts += rng.normal(scale=0.05, size=gs_pts.shape)
        registered = register_floorplan(
            fp, gs_pts, floor_centroid_2d=np.array([10.0, -2.0]),
        )
        center = np.array([
            sum((w.x1 + w.x2) / 2 for w in registered.walls) / len(registered.walls),
            sum((w.y1 + w.y2) / 2 for w in registered.walls) / len(registered.walls),
        ])
        assert center == pytest.approx(np.array([10.0, -2.0]), abs=0.1)

    def test_register_rotates_to_match_pca(self) -> None:
        # Floorplan is axis-aligned 4×3 rectangle.
        fp = ManualProvider.from_rectangle(4.0, 3.0).get_floorplan()
        # 3DGS footprint is the same rectangle rotated 30° around origin,
        # sampled densely along edges so the corridor score is meaningful.
        rng = np.random.default_rng(2)
        edge_pts: list[np.ndarray] = []
        for (x1, y1), (x2, y2) in [
            ((-2.0, -1.5), (2.0, -1.5)),
            ((2.0, -1.5), (2.0, 1.5)),
            ((2.0, 1.5), (-2.0, 1.5)),
            ((-2.0, 1.5), (-2.0, -1.5)),
        ]:
            t = rng.uniform(0.0, 1.0, 200)
            xs = x1 + t * (x2 - x1) + rng.normal(scale=0.02, size=200)
            ys = y1 + t * (y2 - y1) + rng.normal(scale=0.02, size=200)
            edge_pts.append(np.column_stack([xs, ys]))
        base_pts = np.concatenate(edge_pts, axis=0)
        angle = np.radians(30.0)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        gs_pts = (rot @ base_pts.T).T
        registered = register_floorplan(fp, gs_pts)
        # Registered long walls should be ~30° from axis-aligned.
        long_wall = max(registered.walls, key=lambda w: w.length())
        dx = long_wall.x2 - long_wall.x1
        dy = long_wall.y2 - long_wall.y1
        wall_angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [-90, 90] because wall direction is ambiguous by 180°.
        wall_angle = ((wall_angle + 90.0) % 180.0) - 90.0
        assert abs(wall_angle - 30.0) < 5.0

    def test_register_scale_override(self) -> None:
        # Floorplan is a 1×1 square but drawn in a different unit (e.g., dm).
        fp = ManualProvider.from_rectangle(1.0, 1.0).get_floorplan()
        # 3DGS footprint is a 6×6 square in meters; scale override = 6.0.
        rng = np.random.default_rng(1)
        gs_pts = np.array([
            [-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0],
        ], dtype=np.float64)
        gs_pts += rng.normal(scale=0.05, size=gs_pts.shape)
        registered = register_floorplan(fp, gs_pts, scale=6.0)
        widths = [w.length() for w in registered.walls]
        # All sides should be ~6m after scaling.
        assert pytest.approx(6.0, abs=0.3) in widths


class TestGuidedFitter:
    """FloorPlanGuidedFitter rejects furniture noise via corridor filtering."""

    def _make_room_with_noise(
        self,
        rng: np.random.Generator,
        n_noise: int = 2000,
    ) -> tuple[np.ndarray, ManualProvider]:
        """Build a 5×4 rectangle room (z-up) plus random noise outside walls."""
        walls_xy = [
            (0.0, 0.0, 5.0, 0.0),
            (5.0, 0.0, 5.0, 4.0),
            (5.0, 4.0, 0.0, 4.0),
            (0.0, 4.0, 0.0, 0.0),
        ]
        pts_list: list[np.ndarray] = []
        for x1, y1, x2, y2 in walls_xy:
            n = 600
            t = rng.uniform(0.0, 1.0, n)
            xs = x1 + t * (x2 - x1) + rng.normal(scale=0.03, size=n)
            ys = y1 + t * (y2 - y1) + rng.normal(scale=0.03, size=n)
            zs = rng.uniform(0.0, 2.8, n)
            pts_list.append(np.column_stack([xs, ys, zs]))

        # Noise: random points scattered in the room interior (not on walls).
        noise_x = rng.uniform(-1.0, 6.0, n_noise)
        noise_y = rng.uniform(-1.0, 5.0, n_noise)
        noise_z = rng.uniform(0.0, 2.8, n_noise)
        pts_list.append(np.column_stack([noise_x, noise_y, noise_z]))

        points = np.concatenate(pts_list, axis=0)
        provider = ManualProvider.from_rectangle(5.0, 4.0)
        return points, provider

    def test_guided_returns_exact_wall_count(self) -> None:
        rng = np.random.default_rng(42)
        points, provider = self._make_room_with_noise(rng)
        fp = provider.get_floorplan()
        fitter = FloorPlanGuidedFitter(corridor_width=0.5)
        walls = fitter.fit_guided(points, fp, up_axis=2)
        assert len(walls) == 4

    def test_guided_vs_blind_fewer_walls(self) -> None:
        rng = np.random.default_rng(43)
        points, provider = self._make_room_with_noise(rng)
        fp = provider.get_floorplan()
        blind = WallFitter().fit(points, up_axis=2)
        guided = FloorPlanGuidedFitter(corridor_width=0.5).fit_guided(points, fp, up_axis=2)
        # Guided should not invent phantom walls from interior noise.
        assert len(guided) <= len(blind)
        assert len(guided) == 4

    def test_guided_corridor_filters_outside_points(self) -> None:
        rng = np.random.default_rng(44)
        points, provider = self._make_room_with_noise(rng, n_noise=3000)
        fp = provider.get_floorplan()
        # Tight corridor: only true wall points are within 0.3m.
        fitter = FloorPlanGuidedFitter(corridor_width=0.3)
        walls = fitter.fit_guided(points, fp, up_axis=2)
        assert len(walls) == 4

    def test_guided_skips_missing_wall(self) -> None:
        rng = np.random.default_rng(45)
        points, provider = self._make_room_with_noise(rng)
        fp = provider.get_floorplan()
        # Add an extra floorplan wall that has no 3DGS points nearby.
        extra = WallSegment(x1=10.0, y1=10.0, x2=12.0, y2=10.0)
        fp_with_extra = ManualProvider.from_rectangle(5.0, 4.0).get_floorplan()
        fp_with_extra.walls.append(extra)
        fitter = FloorPlanGuidedFitter(corridor_width=0.5)
        walls = fitter.fit_guided(points, fp_with_extra, up_axis=2)
        # Should fit the 4 real walls and skip the extra phantom wall.
        assert len(walls) == 4

    def test_guided_height_override(self) -> None:
        rng = np.random.default_rng(46)
        points, provider = self._make_room_with_noise(rng)
        fp = provider.get_floorplan()
        fitter = FloorPlanGuidedFitter(corridor_width=0.5)
        walls = fitter.fit_guided(
            points, fp, up_axis=2, floor_z=0.1, ceiling_z=2.7,
        )
        assert len(walls) == 4
        for wall in walls:
            assert wall.height == pytest.approx(2.6, abs=1e-6)
            assert wall.p0[2] == pytest.approx(0.1, abs=1e-6)
            assert wall.p1[2] == pytest.approx(0.1, abs=1e-6)
