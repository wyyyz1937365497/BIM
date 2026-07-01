"""Generate walls from any SceneSplat PLY scene.

通用墙线生成脚本：加载 PLY + feat.pt → 多高度虚拟扫描 → 墙线提取 → JSON + PNG

Usage:
    cmd /c "...\\vcvars64.bat && python scripts/generate_walls.py --data-dir data/playroom --name playroom"
    cmd /c "...\\vcvars64.bat && python scripts/generate_walls.py --data-dir data/4 --name scene0002"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.gs_scene import GSScene
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.wall_line_extractor import (
    extract_wall_lines,
    multi_height_scan,
    save_wall_lines_plot,
    wall_lines_to_json,
)


def find_scene_files(data_dir: Path) -> tuple[Path, Path]:
    """Auto-discover PLY and feat.pt in data_dir."""
    plys = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))
    feats = sorted(data_dir.glob("*_feat.pt"))
    if not plys:
        raise FileNotFoundError(f"No *_feat_vis_3dgs.ply in {data_dir}")
    if not feats:
        raise FileNotFoundError(f"No *_feat.pt in {data_dir}")
    return plys[0], feats[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate walls from SceneSplat PLY scene")
    parser.add_argument("--data-dir", required=True, help="Directory containing PLY + feat.pt")
    parser.add_argument("--name", required=True, help="Scene name (for output directory)")
    parser.add_argument("--ply", default=None, help="Explicit PLY path (overrides auto-discover)")
    parser.add_argument("--feat", default=None, help="Explicit feat.pt path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # Auto-discover first, then allow explicit overrides
    ply_path, feat_path = find_scene_files(data_dir)
    if args.ply:
        ply_path = Path(args.ply)
    if args.feat:
        feat_path = Path(args.feat)

    # Reuse BIM vocabulary from data/0 (universal across scenes)
    text_emb_path = ROOT / "data" / "0" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "0" / "bim_class_names.json"

    out_dir = ROOT / "output" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {args.name}")
    print(f"  PLY:  {ply_path}")
    print(f"  Feat: {feat_path}")
    scene = GSScene.from_ply(
        ply_path,
        feat_path=feat_path,
        text_emb_path=text_emb_path,
        class_names_path=class_names_path,
    )
    print(f"Loaded {scene.num_gaussians} Gaussians")
    bounds = scene.scene_bounds()
    print(f"Bounds: {bounds[0]} to {bounds[1]}")
    print(f"Dimensions: {bounds[1] - bounds[0]}")

    # Detect up_axis: the axis where floor and ceiling centroids differ most
    # (argmin(floor_centroid) is unreliable when floor X happens to be most negative)
    floor_result = scene.query_semantics("floor", mode="dominant")
    ceiling_result = scene.query_semantics("ceiling", mode="dominant")
    floor_c = np.array(floor_result["centroid"])
    ceiling_c = np.array(ceiling_result["centroid"])
    up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_c[up_axis])
    ceiling_z = float(ceiling_c[up_axis])
    center_x = float(floor_c[h_axes[0]])
    center_y = float(floor_c[h_axes[1]])
    print(f"\nCoordinate system:")
    print(f"  up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")
    print(f"  Scan center: ({center_x:.2f}, {center_y:.2f})")

    # Quick semantic overview
    print("\nSemantic overview:")
    for label in ["wall", "floor", "ceiling", "door", "window", "furniture"]:
        try:
            r = scene.query_semantics(label, mode="dominant")
            print(f"  {label}: {r['num_gaussians']} gaussians, centroid={r['centroid']}")
        except Exception as e:
            print(f"  {label}: query failed ({e})")

    # Multi-height scan
    num_heights = 8
    print(f"\nScanning at {num_heights} heights from {floor_z:.2f}m to {ceiling_z:.2f}m...")
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scans = multi_height_scan(
        scanner,
        center_2d=(center_x, center_y),
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        num_heights=num_heights,
        num_views=8,
        fov=60.0,
        width=512,
    )
    total_pts = sum(len(s.angles_deg) for s in scans)
    print(f"Total scan points across all heights: {total_pts}")

    # Per-height stats
    print("\nPer-height scan stats:")
    for i, s in enumerate(scans):
        wall_mask = s.semantic_labels == 0
        print(f"  H{i}: {len(s.angles_deg)} pts, wall_pts={wall_mask.sum()}, z={s.height:.2f}m")

    # Extract wall lines
    print("\nExtracting wall lines...")
    center_arr = np.array([center_x, center_y])
    wall_lines, wall_pts = extract_wall_lines(
        scans,
        wall_class_idx=0,
        exclude_classes=[1, 2, 8],
        center=center_arr,
    )
    print(f"Wall points (semantic=wall): {len(wall_pts)}")
    print(f"Extracted {len(wall_lines)} wall lines:")
    for i, wl in enumerate(wall_lines):
        print(
            f"  {i}: ({wl.x1:.2f},{wl.y1:.2f}) -> ({wl.x2:.2f},{wl.y2:.2f}), "
            f"length={wl.length:.2f}m, pts={wl.num_points}"
        )

    # Save JSON
    output_json = wall_lines_to_json(wall_lines, scans, center_arr)
    json_path = str(out_dir / "wall_lines.json")
    Path(json_path).write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    print(f"\nWall lines JSON saved to: {json_path}")

    # Save top-down plot
    png_path = str(out_dir / "wall_lines_topdown.png")
    save_wall_lines_plot(
        wall_lines,
        wall_pts,
        center_arr,
        png_path,
        title=f"{args.name}: {len(wall_lines)} walls, {len(wall_pts)} pts",
    )
    print(f"Top-down plot saved to: {png_path}")

    # Summary
    try:
        ply_rel = str(ply_path.resolve().relative_to(ROOT))
        feat_rel = str(feat_path.resolve().relative_to(ROOT))
    except ValueError:
        ply_rel = str(ply_path)
        feat_rel = str(feat_path)
    summary = {
        "scene": args.name,
        "ply_file": ply_rel,
        "feat_file": feat_rel,
        "num_gaussians": scene.num_gaussians,
        "up_axis": up_axis,
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "center": [center_x, center_y],
        "scan_total_points": total_pts,
        "wall_points": len(wall_pts),
        "num_walls": len(wall_lines),
        "walls": [
            {
                "id": i,
                "x1": wl.x1, "y1": wl.y1,
                "x2": wl.x2, "y2": wl.y2,
                "length": wl.length,
                "num_points": wl.num_points,
            }
            for i, wl in enumerate(wall_lines)
        ],
    }
    summary_path = str(out_dir / "summary.json")
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
