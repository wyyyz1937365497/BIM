#!/usr/bin/env python
"""Generate panorama strips for each wall.

For each wall, sweeps the camera across the wall's angular extent at
fixed 60° FOV, renders overlapping views, and stitches them into one
seamless long strip via cylindrical projection (each panorama column
is sampled from the view closest to that angle).

Usage:
    vcvars64 + bim-recon env
    python scripts/test_wall_panorama.py --ply data/splat/splat.ply
"""
from __future__ import annotations
import argparse, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from bim_recon.gs_scene import GSScene, look_at_pose


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a-b, wrapped to [-π, π]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def render_wall_panoramas(
    scene: GSScene,
    walls: list[dict],
    center: tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    up_axis: int = 2,
    out_dir: str | None = None,
    fov: float = 60.0,
    view_width: int = 768,
    view_height: int = 512,
    step_deg: float = 20.0,
) -> dict[int, Image.Image]:
    """Generate a cylindrical-projection panorama strip for each wall.

    Returns ``{wall_idx: PIL.Image}``.
    """
    mid_z = (floor_z + ceiling_z) / 2.0
    h_axes = [i for i in range(3) if i != up_axis]
    cx, cy = float(center[0]), float(center[1])
    fx = 0.5 * view_width / math.tan(0.5 * math.radians(fov))
    step_rad = math.radians(step_deg)
    margin_rad = math.radians(5)

    panoramas: dict[int, Image.Image] = {}

    for wi, wall in enumerate(walls):
        ws = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
        we = np.array([wall["x2"], wall["y2"]], dtype=np.float64)

        # Angular range from center, relative to wall midpoint direction
        wmid = (ws + we) / 2.0
        a_mid = math.atan2(wmid[h_axes[1]] - cy, wmid[h_axes[0]] - cx)
        da_s = _angle_diff(math.atan2(ws[h_axes[1]] - cy, ws[h_axes[0]] - cx), a_mid)
        da_e = _angle_diff(math.atan2(we[h_axes[1]] - cy, we[h_axes[0]] - cx), a_mid)
        a_lo = a_mid + min(da_s, da_e) - margin_rad
        a_hi = a_mid + max(da_s, da_e) + margin_rad
        angular_span = a_hi - a_lo

        # View angles
        view_angles = np.arange(a_lo, a_hi + step_rad / 2, step_rad)

        print(f"  Wall {wi}: span={math.degrees(angular_span):.0f}° "
              f"→ {len(view_angles)} views at {step_deg}° step")

        # Render all views
        renders: list[tuple[np.ndarray, float]] = []
        for va in view_angles:
            eye = [0.0, 0.0, 0.0]
            eye[h_axes[0]] = cx
            eye[h_axes[1]] = cy
            eye[up_axis] = mid_z
            tgt = [0.0, 0.0, 0.0]
            tgt[h_axes[0]] = cx + math.cos(va)
            tgt[h_axes[1]] = cy + math.sin(va)
            tgt[up_axis] = mid_z
            up = [0.0, 0.0, 0.0]
            up[up_axis] = 1.0
            pose = look_at_pose(tuple(eye), tuple(tgt), tuple(up))
            result = scene.render(pose, width=view_width, height=view_height,
                                  fov_degrees=fov)
            renders.append((result.colors, float(va)))

        # Build panorama via cylindrical projection
        pan_width = max(1, int(fx * angular_span))
        panorama = np.zeros((view_height, pan_width, 3), dtype=np.float32)

        for u in range(pan_width):
            world_angle = a_lo + (u + 0.5) / pan_width * angular_span

            # Find the view closest to this angle
            best_vi = 0
            best_dist = float("inf")
            for vi, (_, va) in enumerate(renders):
                dist = abs(_angle_diff(world_angle, va))
                if dist < best_dist:
                    best_dist = dist
                    best_vi = vi

            # Map world_angle to pixel in the best view
            offset = _angle_diff(world_angle, renders[best_vi][1])
            u_view = int(view_width / 2.0 + fx * offset)
            if 0 <= u_view < view_width:
                panorama[:, u] = renders[best_vi][0][:, u_view]

        pan_img = Image.fromarray(
            (panorama * 255).clip(0, 255).astype(np.uint8))
        panoramas[wi] = pan_img

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            pan_img.save(os.path.join(out_dir, f"wall_panorama_{wi}.png"))

        print(f"    → {pan_img.width}×{pan_img.height}px panorama saved")

    return panoramas


def main():
    ap = argparse.ArgumentParser(description="Wall panorama generator")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--feat", default=None)
    ap.add_argument("--walls-json", default=None,
                    help="Path to wall_lines_snapped.json. If omitted, uses "
                         "the latest output/<scene>/ timestamp dir.")
    ap.add_argument("--out-dir", default="output/wall_panorama_test")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=20.0,
                    help="Angular step between views (degrees)")
    args = ap.parse_args()

    print(f"Loading: {args.ply}")
    scene = GSScene.from_ply(args.ply, feat_path=args.feat)
    means = scene.means.cpu().numpy()

    # Determine up axis + bounds (same logic as pipeline)
    p5 = np.percentile(means, 5, axis=0)
    p95 = np.percentile(means, 95, axis=0)
    up_axis = int(np.argmin(p95 - p5))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(np.percentile(means[:, up_axis], 1))
    ceiling_z = float(np.percentile(means[:, up_axis], 99))
    cx = float((p5[h_axes[0]] + p95[h_axes[0]]) / 2)
    cy = float((p5[h_axes[1]] + p95[h_axes[1]]) / 2)
    print(f"  Room: {p95[h_axes[0]]-p5[h_axes[0]]:.1f}×"
          f"{p95[h_axes[1]]-p5[h_axes[1]]:.1f}×"
          f"{ceiling_z-floor_z:.1f}m, up={up_axis}")

    # Load walls
    import json
    if args.walls_json:
        walls_path = args.walls_json
    else:
        # Find latest wall_lines_snapped.json
        scene_name = os.path.splitext(os.path.basename(args.ply))[0]
        base = os.path.join("output", scene_name)
        candidates = sorted(
            [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
            reverse=True,
        ) if os.path.isdir(base) else []
        walls_path = os.path.join(base, candidates[0], "wall_lines_snapped.json") \
            if candidates else None

    if not walls_path or not os.path.exists(walls_path):
        print("ERROR: No wall_lines_snapped.json found. Run the pipeline first.")
        return 1

    print(f"  Walls: {walls_path}")
    walls = json.loads(open(walls_path).read())
    if isinstance(walls, dict) and "walls" in walls:
        walls = walls["walls"]
    print(f"  {len(walls)} wall segments")

    # Generate panoramas
    print(f"\nGenerating panoramas (FOV={args.fov}°, step={args.step}°)...")
    panoramas = render_wall_panoramas(
        scene, walls, (cx, cy), floor_z, ceiling_z,
        up_axis=up_axis, out_dir=args.out_dir,
        fov=args.fov, step_deg=args.step,
    )

    print(f"\nDone. {len(panoramas)} panoramas in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
