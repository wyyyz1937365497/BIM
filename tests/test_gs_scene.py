"""Unit tests for bim_recon.gs_scene — no real PLY data required.

Covers:
- CameraPose / look_at_pose / fov_to_intrinsics
- GSScene.from_synthetic + render produces correct shapes
- select_by_mask returns indices inside a box mask
- PLY round-trip via a nerfstudio-format writer

Run with: pytest tests/test_gs_scene.py -v
Must be run with MSVC in PATH (gsplat JIT-compiles CUDA on first run).
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from bim_recon.gs_scene import (
    CameraPose,
    GSScene,
    RenderResult,
    fov_to_intrinsics,
    look_at_pose,
)


# ----------------------------- camera utils ---------------------------------


def test_fov_to_intrinsics_basic():
    K = fov_to_intrinsics(fov_degrees=90.0, width=200, height=100)
    # fx = 0.5 * 200 / tan(45deg) = 100 / 1 = 100
    assert K[0, 0] == pytest.approx(100.0, abs=1e-3)
    assert K[1, 1] == pytest.approx(100.0, abs=1e-3)
    assert K[0, 2] == pytest.approx(100.0)  # cx = width/2
    assert K[1, 2] == pytest.approx(50.0)   # cy = height/2
    assert K[2, 2] == pytest.approx(1.0)


def test_look_at_pose_in_front():
    # Camera at origin looking toward +z. Resulting forward axis = +z.
    pose = look_at_pose(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0), up=(0.0, 1.0, 0.0))
    viewmat = pose.to_viewmat()
    # The camera origin in world maps to zero in camera space
    origin_cam = viewmat @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(origin_cam[:3], 0.0, atol=1e-5)
    # A point at (0,0,5) world should map to (0,0,5) camera (pure translation along +z)
    pt_cam = viewmat @ np.array([0.0, 0.0, 5.0, 1.0], dtype=np.float32)
    assert pt_cam[2] == pytest.approx(5.0, abs=1e-4)


def test_viewmat_is_world_to_camera():
    pose = look_at_pose(eye=(1.0, 2.0, 3.0), target=(1.0, 2.0, 8.0))
    viewmat = pose.to_viewmat()
    # camera position in world should map to zero in camera space
    eye_homog = np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float32)
    eye_cam = viewmat @ eye_homog
    assert np.allclose(eye_cam[:3], 0.0, atol=1e-4)


# ----------------------------- synthetic render -----------------------------


@pytest.fixture(scope="module")
def synthetic_scene():
    """Four Gaussians forming a small wall in front of the camera."""
    means = np.array(
        [[-0.5, -0.5, 3.0],
         [ 0.5, -0.5, 3.0],
         [-0.5,  0.5, 3.0],
         [ 0.5,  0.5, 3.0]],
        dtype=np.float32,
    )
    colors = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return GSScene.from_synthetic(means=means, colors_rgb=colors, scales=np.full((4, 3), 0.15, dtype=np.float32))


def test_synthetic_scene_fields(synthetic_scene):
    assert synthetic_scene.num_gaussians == 4
    mn, mx = synthetic_scene.scene_bounds()
    assert mn[2] == pytest.approx(3.0)
    assert mx[2] == pytest.approx(3.0)


def test_render_shapes_and_depth(synthetic_scene):
    pose = look_at_pose(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0))
    result = synthetic_scene.render(pose, width=200, height=200, fov_degrees=90.0)
    assert isinstance(result, RenderResult)
    assert result.colors.shape == (200, 200, 3)
    assert result.depth.shape == (200, 200)
    assert result.alpha.shape == (200, 200)
    # Some pixels should be rendered (alpha > 0)
    assert (result.alpha > 0.5).sum() > 100
    # Depth at rendered pixels should be near 3.0 (the wall is at z=3)
    rendered_mask = result.alpha > 0.5
    assert rendered_mask.sum() > 0
    rendered_depths = result.depth[rendered_mask]
    assert rendered_depths.mean() == pytest.approx(3.0, abs=0.2)


def test_render_off_axis_changes_visibility(synthetic_scene):
    # Pointing camera away from the wall should reduce rendered pixels.
    front = look_at_pose(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0))
    away = look_at_pose(eye=(0.0, 0.0, 0.0), target=(5.0, 0.0, 0.0))
    r_front = synthetic_scene.render(front, 200, 200, 90.0)
    r_away = synthetic_scene.render(away, 200, 200, 90.0)
    assert (r_front.alpha > 0.5).sum() > (r_away.alpha > 0.5).sum()


# ----------------------------- PLY round-trip -------------------------------


def _write_synthetic_ply(path: Path, n: int = 4):
    """Write a tiny nerfstudio-format PLY for the loader test."""
    # SH DC for red, green, blue, yellow; position z=3
    means = np.array(
        [[-0.5, -0.5, 3.0], [0.5, -0.5, 3.0], [-0.5, 0.5, 3.0], [0.5, 0.5, 3.0]],
        dtype=np.float32,
    )[:n]
    sh_dc = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )[:n]
    opacity_logit = np.log(np.ones(n) / (1 - np.ones(n) * 0.999))  # ~1.0 probability
    log_scales = np.log(np.full((n, 3), 0.15, dtype=np.float32))
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment test\n")
        f.write(f"element vertex {n}\n".encode())
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            f.write(f"property float {name}\n".encode())
        for i in range(3):
            f.write(f"property float f_dc_{i}\n".encode())
        f.write(b"property float opacity\n")
        for i in range(3):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(4):
            f.write(f"property float rot_{i}\n".encode())
        f.write(b"end_header\n")
        for j in range(n):
            f.write(struct.pack("<fff", *means[j]))
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            f.write(struct.pack("<fff", *sh_dc[j]))
            f.write(struct.pack("<f", float(opacity_logit[j])))
            f.write(struct.pack("<fff", *log_scales[j]))
            f.write(struct.pack("<ffff", *quats[j]))


def test_ply_round_trip(tmp_path: Path):
    ply = tmp_path / "mini.ply"
    _write_synthetic_ply(ply, n=4)
    scene = GSScene.from_ply(ply)
    assert scene.num_gaussians == 4
    # Means should match
    means_np = scene.means.cpu().numpy()
    assert means_np[0, 2] == pytest.approx(3.0, abs=1e-4)
    # Opacity after sigmoid should be close to 1.0
    opac = scene.opacities.cpu().numpy()
    assert np.all(opac > 0.9)
    # Scales after exp should be 0.15
    scales_np = scene.scales.cpu().numpy()
    assert np.allclose(scales_np, 0.15, atol=1e-4)
    # Color of first (red SH DC=1) after SH conversion: C0*1 + 0.5 = ~0.78
    colors_np = scene.colors.cpu().numpy()
    expected_r = 0.28209479177387814 * 1.0 + 0.5
    assert colors_np[0, 0] == pytest.approx(expected_r, abs=1e-3)


def test_ply_render_after_load(tmp_path: Path):
    ply = tmp_path / "mini.ply"
    _write_synthetic_ply(ply, n=4)
    scene = GSScene.from_ply(ply)
    pose = look_at_pose(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0))
    result = scene.render(pose, 200, 200, 90.0)
    assert (result.alpha > 0.5).sum() > 50


# ----------------------------- select_by_mask -------------------------------


def test_select_by_mask_returns_inliers(synthetic_scene):
    pose = look_at_pose(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0))
    W = H = 200
    # Full-frame mask should select all 4 Gaussians
    full_mask = np.ones((H, W), dtype=bool)
    ids = synthetic_scene.select_by_mask(pose=pose, mask=full_mask, width=W, height=H, fov_degrees=90.0)
    assert ids.shape[0] == 4
    # Tiny corner mask should select nothing (Gaussians are near image center)
    corner_mask = np.zeros((H, W), dtype=bool)
    corner_mask[0:10, 0:10] = True
    ids_corner = synthetic_scene.select_by_mask(pose=pose, mask=corner_mask, width=W, height=H, fov_degrees=90.0)
    assert ids_corner.shape[0] == 0
