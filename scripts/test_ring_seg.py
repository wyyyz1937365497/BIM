#!/usr/bin/env python
"""Ring scan test: 360° overlapping render + Falcon polygon seg + mask backproject.

Produces a top-down radar plot showing:
  - Camera position and viewing angle fans
  - Wall lines (from wall_lines_snapped.json or auto-detected)
  - Depth rays (thin lines from center)
  - Window mask backprojections (should appear as LINE SEGMENTS on walls
    when viewed from above, since each mask is a surface on the wall plane)

Usage (needs vcvars64 for gsplat JIT + Falcon server running):

    cmd /c "\"C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat\" && python scripts/test_ring_seg.py --name splat"

    # Skip Falcon (just render + depth backproject, no segmentation):
    cmd /c "\"...\\vcvars64.bat\" && python scripts/test_ring_seg.py --name splat --no-falcon"
"""
from __future__ import annotations

import argparse
import base64 as b64
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ring scan segmentation test")
    parser.add_argument("--name", default="splat", help="Scene name in data/")
    parser.add_argument("--n-views", type=int, default=12, help="Number of ring views")
    parser.add_argument("--fov", type=float, default=45.0, help="FOV per view (degrees)")
    parser.add_argument("--img-size", type=int, default=768, help="Image size per view")
    parser.add_argument("--query", default="window", help="Falcon query text")
    parser.add_argument("--falcon-host", default="127.0.0.1")
    parser.add_argument("--falcon-port", type=int, default=8390)
    parser.add_argument("--no-falcon", action="store_true", help="Skip Falcon segmentation")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # ── Load scene ──────────────────────────────────────────────────
    from bim_recon.gs_scene import GSScene, look_at_pose
    from bim_recon.virtual_scanner import VirtualScanner
    from bim_recon.wall_line_extractor import multi_height_scan, extract_wall_lines

    data_dir = ROOT / "data" / args.name
    ply_candidates = sorted(data_dir.glob("point_cloud_*.ply")) + sorted(data_dir.glob("*.ply"))
    if not ply_candidates:
        print(f"No PLY files found in {data_dir}")
        return 1
    ply_path = ply_candidates[0]
    feat_path = sorted(data_dir.glob("*feat*.pt"))[0] if list(data_dir.glob("*feat*.pt")) else None
    print(f"PLY:  {ply_path.name}")
    print(f"Feat: {feat_path.name if feat_path else 'none'}")

    text_emb = str(ROOT / "data" / "bim_text_emb.pt")
    class_names = str(ROOT / "data" / "bim_class_names.json")
    warm = {}
    if Path(text_emb).exists() and Path(class_names).exists():
        warm = {"text_emb_path": text_emb, "class_names_path": class_names}
    scene = GSScene.from_ply(ply_path, feat_path=feat_path, **warm)
    print(f"Gaussians: {scene.num_gaussians}")

    # ── Detect coordinate system ────────────────────────────────────
    labels = ["wall", "floor", "ceiling", "door", "window"]
    scanner = VirtualScanner(scene, up_axis=2, labels=labels)

    has_semantics = scene._has_feat and scene.semantic_querier is not None
    if has_semantics:
        floor_c = np.array(scene.query_semantics("floor", mode="dominant", label_set=labels)["centroid"])
        ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant", label_set=labels)["centroid"])
        up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(floor_c[up_axis])
        ceiling_z = float(ceiling_c[up_axis])
        if ceiling_z - floor_z < 1.5:
            coords = scene.means[:, up_axis].cpu().numpy()
            floor_z = float(np.percentile(coords, 1))
            ceiling_z = float(np.percentile(coords, 99))
        center = (float(floor_c[h_axes[0]]), float(floor_c[h_axes[1]]))
    else:
        print("  No feat.pt — using Gaussian distribution for coordinate detection")
        means = scene.means.cpu().numpy()
        ranges = means.max(axis=0) - means.min(axis=0)
        up_axis = int(np.argmax(ranges))  # tallest axis = up
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(np.percentile(means[:, up_axis], 1))
        ceiling_z = float(np.percentile(means[:, up_axis], 99))
        center = (float(np.median(means[:, h_axes[0]])),
                  float(np.median(means[:, h_axes[1]])))
    mid_z = (floor_z + ceiling_z) / 2.0
    print(f"up_axis={up_axis} floor_z={floor_z:.2f} ceiling_z={ceiling_z:.2f} center={center}")

    # Re-init scanner with correct up_axis
    scanner = VirtualScanner(scene, up_axis=up_axis, labels=labels)

    # ── Quick wall extraction (for radar overlay) ───────────────────
    print("\nScanning for walls...")
    scans = multi_height_scan(scanner, center, floor_z, ceiling_z,
                              num_heights=8, num_views=8, width=512)
    wall_lines, wall_pts = extract_wall_lines(scans, labels=labels, center=np.array(center))
    print(f"  {len(wall_lines)} wall segments")

    # ── Output dir ──────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "output" / args.name / f"ring_seg_test_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ring_dir = out_dir / "ring_views"
    ring_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Ring render ─────────────────────────────────────────────────
    from bim_recon.ring_scanner import render_ring_views
    print(f"\nRendering {args.n_views} ring views ({args.fov}° FOV)...")
    views = render_ring_views(
        scene, center, mid_z, up_axis=up_axis,
        n_views=args.n_views, fov=args.fov, img_size=args.img_size,
    )
    print(f"  {len(views)} views rendered")

    # Save ring views
    from PIL import Image
    for v in views:
        Image.fromarray(v.image).save(str(ring_dir / f"view_{v.idx:02d}_{v.azimuth_deg:.0f}.png"))

    # ── Falcon segmentation ─────────────────────────────────────────
    falcon = None
    if not args.no_falcon:
        from bim_recon.falcon_client import FalconClient
        falcon = FalconClient(host=args.falcon_host, port=args.falcon_port)
        if falcon.health():
            print(f"Falcon server: connected")
        else:
            print(f"Falcon server: unreachable, skipping segmentation")
            falcon = None

    # Collect all mask backprojections
    all_mask_points = []  # List of (N, 2) arrays in world XY
    all_det_info = []     # Metadata per detection

    if falcon is not None:
        from pycocotools import mask as mask_utils
        print(f"\nSegmenting '{args.query}' in each view...")
        for vi, view in enumerate(views):
            img = Image.fromarray(view.image)
            try:
                dets = falcon.segment(img, args.query, task="segmentation")
            except Exception as e:
                print(f"  view {vi} ({view.azimuth_deg:.0f}°): error - {e}")
                continue

            for di, det in enumerate(dets):
                # Decode polygon mask
                mask_arr = None
                if det.mask_rle and det.mask_size:
                    try:
                        counts = b64.b64decode(det.mask_rle)
                        mask_arr = mask_utils.decode(
                            {"counts": counts, "size": det.mask_size})
                        mh, mw = det.mask_size
                        if mh != view.height or mw != view.width:
                            mask_arr = np.array(
                                Image.fromarray(mask_arr).resize(
                                    (view.width, view.height), Image.NEAREST))
                    except Exception:
                        mask_arr = None

                if mask_arr is None or mask_arr.sum() < 10:
                    continue

                # Backproject all mask pixels via depth
                ys, xs = np.where(mask_arr)
                depths = view.depth[ys, xs].astype(np.float64)
                valid = depths > 0.1
                if valid.sum() < 5:
                    continue

                xs_v, ys_v, ds_v = xs[valid], ys[valid], depths[valid]
                # Camera-space → world
                x_cam = (xs_v - view.cx_pix) / view.fx * ds_v
                y_cam = (ys_v - view.cy_pix) / view.fy * ds_v
                P_cam = np.stack([x_cam, y_cam, ds_v], axis=1)  # (K, 3)
                R_c2w = np.stack([view.right, view.up, view.forward], axis=1)
                P_world = P_cam @ R_c2w.T + view.eye  # (K, 3)

                # Extract world XY
                wxy = P_world[:, h_axes]  # (K, 2)
                all_mask_points.append(wxy)
                all_det_info.append({
                    "view": vi,
                    "azimuth": view.azimuth_deg,
                    "n_pixels": int(valid.sum()),
                    "depth_mean": float(np.mean(ds_v)),
                })

                # Save annotated view
                annotated = img.convert("RGBA")
                from PIL import ImageDraw
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                colored = np.zeros((*img.size[::-1], 4), dtype=np.uint8)
                colored[mask_arr > 0] = [255, 0, 0, 120]
                overlay = Image.alpha_composite(annotated, Image.fromarray(colored, "RGBA"))
                overlay.convert("RGB").save(
                    str(ring_dir / f"seg_{vi:02d}_{view.azimuth_deg:.0f}_det{di}.png"))

            n_dets = len([d for d in dets if d.mask_rle])
            print(f"  view {vi:2d} ({view.azimuth_deg:5.1f}°): {n_dets} detections")

    print(f"\nTotal detections: {len(all_mask_points)}")

    # ── Merge nearby detections in polar space ──────────────────────
    from bim_recon.element_merger import merge_detections
    merge_input = []
    for pts in all_mask_points:
        wx, wy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        extent = max(pts[:, 0].max() - pts[:, 0].min(),
                     pts[:, 1].max() - pts[:, 1].min())
        merge_input.append({
            "element_class": args.query,
            "world_x": wx, "world_y": wy,
            "sill_height": 0.0, "header_height": 2.0,
            "width_m": max(extent, 0.1),
            "confidence": 0.5,
        })
    merged = merge_detections(merge_input, center, up_axis=up_axis,
                              merge_threshold=0.5)
    print(f"Merged: {len(all_mask_points)} raw -> {len(merged)} unique")
    for me in merged:
        print(f"  theta={me.theta_center:.1f}deg r={me.r_mean:.2f}m "
              f"w={me.width_m:.2f}m (from {me.num_sources} masks)")


    # ── Generate top-down radar plot ────────────────────────────────
    print("\nGenerating radar plot...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    cx, cy = float(center[0]), float(center[1])

    # 1. Draw wall lines
    for wl in wall_lines:
        x1, y1 = wl.x1 - cx, wl.y1 - cy
        x2, y2 = wl.x2 - cx, wl.y2 - cy
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)
        # Wall endpoints
        ax.plot([x1, x2], [y1, y2], "ko", markersize=4, zorder=6)

    # 2. Draw camera viewing angle fans
    fan_color = "lightblue"
    for v in views:
        az = math.radians(v.azimuth_deg)
        half_fov = math.radians(args.fov / 2)
        # Draw fan wedge
        az_start = math.degrees(az - half_fov)
        az_end = math.degrees(az + half_fov)
        # Find max depth in this view
        max_d = float(np.percentile(v.depth[v.depth > 0.1], 90)) if (v.depth > 0.1).any() else 5.0
        wedge = Wedge((0, 0), max_d, az_start, az_end,
                       alpha=0.05, color=fan_color, zorder=1)
        ax.add_patch(wedge)
        # Draw center ray
        ax.plot([0, max_d * math.cos(az)], [0, max_d * math.sin(az)],
                "-", color="steelblue", linewidth=0.3, alpha=0.3, zorder=2)

    # 3. Draw depth samples (sparse, for visualization)
    for v in views:
        az = math.radians(v.azimuth_deg)
        # Sample every Nth pixel in middle row
        mid_row = v.depth[v.height // 2]
        for pi in range(0, len(mid_row), max(1, len(mid_row) // 50)):
            d = mid_row[pi]
            if d < 0.1:
                continue
            # Pixel angle relative to view azimuth
            pix_angle = math.atan2(pi - v.cx_pix, v.fx)
            world_angle = az + pix_angle
            ax.plot(d * math.cos(world_angle), d * math.sin(world_angle),
                    ".", color="gray", markersize=1, alpha=0.3, zorder=2)

    # 4. Draw mask backprojections (should appear as LINE SEGMENTS on walls)
    colors_mask = plt.cm.Set1(np.linspace(0, 1, max(len(all_mask_points), 1)))
    for mi, wxy in enumerate(all_mask_points):
        pts = wxy - np.array([cx, cy])
        color = colors_mask[mi % len(colors_mask)]
        # Plot as scatter
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=2, alpha=0.5, zorder=7)
        # Draw convex hull or bounding line to show the "line segment"
        if len(pts) > 3:
            # PCA to find main direction
            centered = pts - pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            main_dir = Vt[0]
            projections = centered @ main_dir
            t_min, t_max = projections.min(), projections.max()
            center_pt = pts.mean(axis=0)
            p1 = center_pt + main_dir * t_min
            p2 = center_pt + main_dir * t_max
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "-", color=color, linewidth=2.5, zorder=8,
                    label=f"mask {mi}" if mi < 10 else "")
            # Endpoints
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "o", color=color, markersize=5, zorder=9)

    # 5. Camera position
    ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")

    # Formatting
    ax.set_aspect("equal")
    max_range = max(
        abs(wl.x1 - cx).max() if wall_lines else 5,
        abs(wl.x2 - cx).max() if wall_lines else 5,
        5,
    )
    ax.set_xlim(-max_range - 1, max_range + 1)
    ax.set_ylim(-max_range - 1, max_range + 1)
    ax.set_xlabel(f"World {h_axes[0]} (m)")
    ax.set_ylabel(f"World {h_axes[1]} (m)")
    ax.set_title(f"Ring Segmentation Backproject: {args.query}\n"
                 f"{args.n_views} views × {args.fov}° FOV, "
                 f"{len(all_mask_points)} masks backprojected", fontsize=14)
    ax.grid(True, alpha=0.3)
    if all_mask_points:
        ax.legend(fontsize=8, loc="upper right")

    radar_path = str(out_dir / "radar_ring_seg.png")
    fig.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar saved: {radar_path}")

    # ── Generate MERGED radar plot ──────────────────────────────────
    if merged:
        print("Generating merged radar plot...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 14))

        # Wall lines
        for wl in wall_lines:
            x1, y1 = wl.x1 - cx, wl.y1 - cy
            x2, y2 = wl.x2 - cx, wl.y2 - cy
            ax2.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)

        # View fans (light)
        for v in views:
            az = math.radians(v.azimuth_deg)
            half_fov = math.radians(args.fov / 2)
            az_start = math.degrees(az - half_fov)
            az_end = math.degrees(az + half_fov)
            max_d = float(np.percentile(v.depth[v.depth > 0.1], 90)) if (v.depth > 0.1).any() else 5.0
            wedge = Wedge((0, 0), max_d, az_start, az_end,
                           alpha=0.04, color="lightblue", zorder=1)
            ax2.add_patch(wedge)

        # Draw merged elements — combine all source mask points
        colors_merged = plt.cm.Set1(np.linspace(0, 1, max(len(merged), 1)))
        for mi, me in enumerate(merged):
            color = colors_merged[mi % len(colors_merged)]
            # Gather all source mask points for this merged element
            src_pts = []
            for si in me.source_indices:
                if si < len(all_mask_points):
                    src_pts.append(all_mask_points[si] - np.array([cx, cy]))
            if not src_pts:
                continue
            all_pts = np.vstack(src_pts)

            # Scatter all source points in this cluster's color
            ax2.scatter(all_pts[:, 0], all_pts[:, 1], c=[color],
                        s=3, alpha=0.4, zorder=7)

            # PCA line segment for the merged cluster
            if len(all_pts) > 3:
                centered = all_pts - all_pts.mean(axis=0)
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                main_dir = Vt[0]
                projections = centered @ main_dir
                t_min, t_max = projections.min(), projections.max()
                center_pt = all_pts.mean(axis=0)
                p1 = center_pt + main_dir * t_min
                p2 = center_pt + main_dir * t_max
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         "-", color=color, linewidth=3, zorder=8,
                         label=f"window {mi} ({me.num_sources} masks)")
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         "o", color=color, markersize=6, zorder=9)

        ax2.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
        ax2.set_aspect("equal")
        ax2.set_xlim(-max_range - 1, max_range + 1)
        ax2.set_ylim(-max_range - 1, max_range + 1)
        ax2.set_xlabel(f"World {h_axes[0]} (m)")
        ax2.set_ylabel(f"World {h_axes[1]} (m)")
        ax2.set_title(f"Merged {args.query}: {len(merged)} unique elements\n"
                      f"(from {len(all_mask_points)} raw detections)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="upper right")

        merged_path = str(out_dir / "radar_merged_seg.png")
        fig2.savefig(merged_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Merged radar saved: {merged_path}")


    # ── Save JSON summary ───────────────────────────────────────────
    summary = {
        "scene": args.name,
        "n_views": len(views),
        "fov": args.fov,
        "query": args.query,
        "n_walls": len(wall_lines),
        "n_mask_detections": len(all_mask_points),
        "mask_info": all_det_info,
        "mask_xy_world": [
            {"x": float(pts[:, 0].mean()), "y": float(pts[:, 1].mean()),
             "extent_x": float(pts[:, 0].max() - pts[:, 0].min()),
             "extent_y": float(pts[:, 1].max() - pts[:, 1].min())}
            for pts in all_mask_points
        ],
    }
    (out_dir / "ring_seg_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone! Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
