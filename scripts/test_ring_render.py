#!/usr/bin/env python
"""Render 8 views at mid-height from the scene center.

Uses 5th-95th percentile bounds to exclude outlier Gaussians, builds a
KD-Tree to find a free camera position, then renders 360° in 8 steps.

Usage:
    python scripts/test_ring_render.py --ply data/splat/splat.ply
    python scripts/test_ring_render.py --ply data/splat/splat.ply --feat output/splat/splat_feat.pt
"""
from __future__ import annotations
import argparse, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from bim_recon.gs_scene import GSScene, look_at_pose


def main():
    ap = argparse.ArgumentParser(description="Render 8 views from scene center")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--feat", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=768)
    args = ap.parse_args()

    out_dir = args.out_dir or "output/ring_test"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {args.ply}")
    scene = GSScene.from_ply(args.ply, feat_path=args.feat)
    means = scene.means.cpu().numpy()
    N = scene.num_gaussians

    # --- Determine up axis + room bounds ---
    # If feat available, use SceneSplat floor/ceiling (most reliable).
    # Otherwise, the tightest cluster dimension is the height.
    up_axis = None
    floor_z = ceiling_z = None

    if args.feat and scene.semantic_querier is not None:
        try:
            q = scene.semantic_querier
            dom = q.get_dominant_labels(label_set=["floor", "ceiling"])
            floor_mask = dom == 0
            ceil_mask = dom == 1
            if floor_mask.sum() > 100 and ceil_mask.sum() > 100:
                fc = means[floor_mask].mean(axis=0)
                cc = means[ceil_mask].mean(axis=0)
                up_axis = int(np.argmax(np.abs(cc - fc)))
                floor_z = float(fc[up_axis])
                ceiling_z = float(cc[up_axis])
                print(f"SceneSplat: up_axis={up_axis}, floor={floor_z:.3f}, ceil={ceiling_z:.3f}")
        except Exception as e:
            print(f"SceneSplat axis detection failed: {e}")

    if up_axis is None:
        # Fallback: use 5th-95th percentile, height = smallest cluster dimension
        p5 = np.percentile(means, 5, axis=0)
        p95 = np.percentile(means, 95, axis=0)
        up_axis = int(np.argmin(p95 - p5))
        print(f"Geometric fallback: up_axis={up_axis} (smallest core extent)")

    h_axes = [i for i in range(3) if i != up_axis]

    # Room bounds from 5th-95th percentile (excludes outliers)
    p5 = np.percentile(means, 5, axis=0)
    p95 = np.percentile(means, 95, axis=0)
    if floor_z is None:
        floor_z = float(p5[up_axis])
        ceiling_z = float(p95[up_axis])

    # Validate / fix height
    height = ceiling_z - floor_z
    if height < 1.5:
        floor_z = float(np.percentile(means[:, up_axis], 1))
        ceiling_z = float(np.percentile(means[:, up_axis], 99))
        height = ceiling_z - floor_z
        print(f"Height implausible (<1.5m), using 1-99 percentile: {height:.3f}m")

    eye_h = (floor_z + ceiling_z) / 2.0
    cx = float((p5[h_axes[0]] + p95[h_axes[0]]) / 2)
    cy = float((p5[h_axes[1]] + p95[h_axes[1]]) / 2)

    print(f"  Room: {p95[h_axes[0]]-p5[h_axes[0]]:.2f} × {p95[h_axes[1]]-p5[h_axes[1]]:.2f} × {height:.2f}m")
    print(f"  Floor={floor_z:.3f} Ceiling={ceiling_z:.3f} Eye={eye_h:.3f}")
    print(f"  Center=({cx:.3f}, {cy:.3f})")

    # --- KD-Tree occupancy ---
    print("Building KD-Tree...")
    scene.build_occupancy()

    center_3d = np.zeros(3)
    center_3d[h_axes[0]] = cx
    center_3d[h_axes[1]] = cy
    center_3d[up_axis] = eye_h

    free = scene.is_position_free(center_3d)
    print(f"  Center free: {free}")

    if not free:
        # Search outward in a spiral pattern for a free position
        print("  Searching for free position near center...")
        found = False
        for r in np.arange(0.1, 3.0, 0.1):
            for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                pos = center_3d.copy()
                pos[h_axes[0]] += r * math.cos(angle)
                pos[h_axes[1]] += r * math.sin(angle)
                if scene.is_position_free(pos):
                    center_3d = pos
                    found = True
                    break
            if found:
                break
        if found:
            print(f"  Found free at ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})")
        else:
            print(f"  ⚠ No free position found, using center anyway")

    # --- Render 8 views ---
    print(f"\nRendering from ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})...")
    fov = args.fov
    w = args.width

    for i in range(8):
        az = math.radians(i * 45)
        eye = [0.0] * 3
        eye[h_axes[0]] = float(center_3d[h_axes[0]])
        eye[h_axes[1]] = float(center_3d[h_axes[1]])
        eye[up_axis] = float(center_3d[up_axis])
        tgt = [0.0] * 3
        tgt[h_axes[0]] = eye[h_axes[0]] + math.cos(az)
        tgt[h_axes[1]] = eye[h_axes[1]] + math.sin(az)
        tgt[up_axis] = eye[up_axis]
        up = [0.0] * 3
        up[up_axis] = 1.0

        pose = look_at_pose(tuple(eye), tuple(tgt), tuple(up))
        result = scene.render(pose, width=w, height=w, fov_degrees=fov)

        coverage = float((result.alpha > 0.1).mean())
        vd = result.depth[result.alpha > 0.1]
        min_d = float(vd.min()) if len(vd) > 0 else 0
        mean_d = float(vd.mean()) if len(vd) > 0 else 0

        img = Image.fromarray((result.colors * 255).clip(0, 255).astype(np.uint8))
        fname = f"view_{i:02d}_{i*45:03d}deg.png"
        img.save(os.path.join(out_dir, fname))

        quality = "✓ GOOD" if (coverage > 0.3 and min_d > 0.3) else \
                  "⚠ CLOSE" if min_d < 0.15 else "~ OK"
        print(f"  [{i}] {i*45:3d}°: cov={coverage:.0%} depth[min={min_d:.2f} mean={mean_d:.2f}] "
              f"{quality} → {fname}")

    print(f"\n8 views → {out_dir}/")


if __name__ == "__main__":
    main()
