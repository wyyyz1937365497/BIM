#!/usr/bin/env python
"""Generate orthographic elevation images for each wall.

For each wall, renders multiple overlapping views from room center,
then re-projects all pixels onto a flat wall plane using depth.  This
produces a true architectural elevation (no parallax, no seams).

Usage:
    python scripts/test_wall_panorama.py --ply data/splat/splat.ply
"""
from __future__ import annotations
import argparse, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from bim_recon.gs_scene import GSScene, look_at_pose


def render_wall_elevations(
    scene: GSScene,
    walls: list[dict],
    center: tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    up_axis: int = 2,
    out_dir: str | None = None,
    fov: float = 60.0,
    view_width: int = 1536,
    view_height: int = 1024,
    step_deg: float = 20.0,
    mm_per_pixel: float = 2.0,
) -> dict[int, Image.Image]:
    """Generate a true orthographic elevation image for each wall.

    Pipeline per wall:
      1. Compute angular range from wall endpoints.
      2. Sweep camera across the range at fixed FOV → overlapping views.
      3. For each view pixel with valid depth → unproject to 3D.
      4. Project the 3D point onto the wall plane → (t_along, height).
      5. Write colour to the elevation texture at that position.

    No parallax, no seams — every 3D point lands at its true wall position.
    """
    mid_z = (floor_z + ceiling_z) / 2.0
    h_axes = [i for i in range(3) if i != up_axis]
    h0, h1 = h_axes
    cx, cy = float(center[0]), float(center[1])
    fx = 0.5 * view_width / math.tan(0.5 * math.radians(fov))
    step_rad = math.radians(step_deg)
    margin_rad = math.radians(8)

    elevations: dict[int, Image.Image] = {}

    for wi, wall in enumerate(walls):
        ws = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
        we = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
        wdir = we - ws
        wlen = float(np.linalg.norm(wdir))
        if wlen < 0.1:
            continue
        wdir /= wlen
        wmid = (ws + we) / 2.0
        wall_h = ceiling_z - floor_z

        # Elevation texture dimensions
        tex_w = max(64, int(wlen * 1000 / mm_per_pixel))
        tex_h = max(64, int(wall_h * 1000 / mm_per_pixel))

        # Accumulators
        texture = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
        weight = np.zeros((tex_h, tex_w), dtype=np.float32)

        # Angular range for view sweep
        a_mid = math.atan2(wmid[h1] - cy, wmid[h0] - cx)

        def _adiff(a, b):
            d = a - b
            return (d + math.pi) % (2 * math.pi) - math.pi

        da_s = _adiff(math.atan2(ws[h1] - cy, ws[h0] - cx), a_mid)
        da_e = _adiff(math.atan2(we[h1] - cy, we[h0] - cx), a_mid)
        a_lo = a_mid + min(da_s, da_e) - margin_rad
        a_hi = a_mid + max(da_s, da_e) + margin_rad

        view_angles = np.arange(a_lo, a_hi + step_rad / 2, step_rad)

        print(f"  Wall {wi}: {wlen:.2f}m, {len(view_angles)} views "
              f"→ tex {tex_w}×{tex_h}")

        # Render views and project onto wall plane
        for va in view_angles:
            eye = [0.0, 0.0, 0.0]
            eye[h0] = cx
            eye[h1] = cy
            eye[up_axis] = mid_z
            tgt = [0.0, 0.0, 0.0]
            tgt[h0] = cx + math.cos(va)
            tgt[h1] = cy + math.sin(va)
            tgt[up_axis] = mid_z
            up = [0.0, 0.0, 0.0]
            up[up_axis] = 1.0
            pose = look_at_pose(tuple(eye), tuple(tgt), tuple(up))
            result = scene.render(pose, width=view_width, height=view_height,
                                  fov_degrees=fov)

            # --- Vectorised depth unprojection → wall plane projection ---
            depth = result.depth  # (H, W) float32
            colors = result.colors  # (H, W, 3) float32
            alpha = result.alpha  # (H, W) float32

            valid = (depth > 0.1) & (alpha > 0.1)
            if not valid.any():
                continue

            ys, xs = np.where(valid)
            ds = depth[ys, xs].astype(np.float64)
            cols = colors[ys, xs]  # (K, 3)

            # Camera-space coordinates
            x_cam = (xs - view_width / 2.0) / fx * ds
            y_cam = (ys - view_height / 2.0) / fx * ds

            # World coordinates
            viewmat = pose.to_viewmat()
            R_c2w = viewmat[:3, :3].T
            eye_np = np.array(eye, dtype=np.float64)
            P_world = R_c2w @ np.stack([x_cam, y_cam, ds], axis=0) + eye_np.reshape(3, 1)

            # Project onto wall plane
            # t_along = dot(P_horizontal - wall_start, wall_dir)
            off0 = P_world[h0, :] - ws[0]
            off1 = P_world[h1, :] - ws[1]
            t_along = off0 * wdir[0] + off1 * wdir[1]
            height = P_world[up_axis, :] - floor_z

            # Quality weight: prefer pixels from views near perpendicular
            # (lower obliquity → sharper texture)
            view_obliquity = abs(_adiff(va, a_mid))
            quality = max(0.0, 1.0 - view_obliquity / (math.pi / 2))

            # Map to texture coordinates
            tu = np.clip((t_along / wlen * tex_w).astype(int), 0, tex_w - 1)
            tv = np.clip(((1.0 - height / wall_h) * tex_h).astype(int), 0, tex_h - 1)

            # Write to texture (keep best quality per pixel)
            for k in range(len(tu)):
                r, c = tv[k], tu[k]
                if quality > weight[r, c]:
                    texture[r, c] = cols[k]
                    weight[r, c] = quality

        # Fill small holes via nearest-neighbour dilation
        holes = weight == 0
        if holes.any() and not holes.all():
            from scipy.ndimage import binary_dilation
            filled = False
            for _ in range(5):
                dilated = binary_dilation(weight > 0, iterations=1)
                new_pixels = dilated & holes
                if not new_pixels.any():
                    break
                # Copy from nearest filled pixel (simple approach)
                from scipy.ndimage import grey_dilation
                tex_r = grey_dilation(texture[:, :, 0], size=3)
                tex_g = grey_dilation(texture[:, :, 1], size=3)
                tex_b = grey_dilation(texture[:, :, 2], size=3)
                texture[:, :, 0] = np.where(new_pixels, tex_r, texture[:, :, 0])
                texture[:, :, 1] = np.where(new_pixels, tex_g, texture[:, :, 1])
                texture[:, :, 2] = np.where(new_pixels, tex_b, texture[:, :, 2])
                weight[new_pixels] = 0.01
                holes = weight == 0

        img = Image.fromarray(
            (texture * 255).clip(0, 255).astype(np.uint8))
        elevations[wi] = img

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, f"wall_elevation_{wi}.png"))

        coverage = float((weight > 0).mean())
        print(f"    → {img.width}×{img.height}px, coverage={coverage:.0%}")

    return elevations


def main():
    ap = argparse.ArgumentParser(description="Wall elevation generator")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--feat", default=None)
    ap.add_argument("--walls-json", default=None)
    ap.add_argument("--out-dir", default="output/wall_elevation_test")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--resolution", type=float, default=2.0,
                    help="mm per pixel (lower = sharper)")
    args = ap.parse_args()

    print(f"Loading: {args.ply}")
    scene = GSScene.from_ply(args.ply, feat_path=args.feat)
    means = scene.means.cpu().numpy()

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
          f"{ceiling_z-floor_z:.1f}m")

    if args.walls_json:
        walls_path = args.walls_json
    else:
        scene_name = os.path.splitext(os.path.basename(args.ply))[0]
        base = os.path.join("output", scene_name)
        candidates = sorted(
            [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
            reverse=True) if os.path.isdir(base) else []
        walls_path = os.path.join(base, candidates[0], "wall_lines_snapped.json") \
            if candidates else None

    import json
    if not walls_path or not os.path.exists(walls_path):
        print("ERROR: No wall_lines_snapped.json. Run the pipeline first.")
        return 1

    walls = json.loads(open(walls_path).read())
    if isinstance(walls, dict) and "walls" in walls:
        walls = walls["walls"]
    print(f"  {len(walls)} wall segments")

    print(f"\nGenerating orthographic elevations "
          f"(FOV={args.fov}°, step={args.step}°, {args.resolution}mm/px)...")
    elevations = render_wall_elevations(
        scene, walls, (cx, cy), floor_z, ceiling_z,
        up_axis=up_axis, out_dir=args.out_dir,
        fov=args.fov, step_deg=args.step,
        mm_per_pixel=args.resolution,
    )

    print(f"\nDone. {len(elevations)} elevations in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
