"""Generate walls from scene0002_00 (data/4).

This scene is smaller and structurally simpler than the previous dataset.
Uses PLY format + SceneSplat feat.pt for semantic-aware wall extraction.

Run with vcvars64:
    cmd /c "...\\vcvars64.bat && python scripts/generate_walls_scene0002.py"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def main() -> int:
    # scene0002_00 uses PLY format (not .npy)
    ply_path = ROOT / "data" / "4" / "scene0002_00_feat_vis_3dgs.ply"
    feat_path = ROOT / "data" / "4" / "scene0002_00_feat.pt"
    # Reuse BIM vocabulary from data/0 (universal across scenes)
    text_emb_path = ROOT / "data" / "0" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "0" / "bim_class_names.json"

    # Output directory specific to this scene
    out_dir = ROOT / "output" / "scene0002"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene from PLY: {ply_path}")
    print(f"  Gaussians: 1,500,000 (expected)")
    print(f"  Features:  {feat_path}")
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

    # Detect up_axis + floor/ceiling
    floor_result = scene.query_semantics("floor", mode="dominant")
    up_axis = int(np.argmin(floor_result["centroid"]))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_result["centroid"][up_axis])
    ceiling_result = scene.query_semantics("ceiling", mode="dominant")
    ceiling_z = float(ceiling_result["centroid"][up_axis])
    center_x = float(floor_result["centroid"][h_axes[0]])
    center_y = float(floor_result["centroid"][h_axes[1]])
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
        wall_mask = s.semantic_labels == 0  # wall class
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
        title=f"scene0002: {len(wall_lines)} walls, {len(wall_pts)} pts",
    )
    print(f"Top-down plot saved to: {png_path}")

    # Also save a quick summary
    summary = {
        "scene": "scene0002_00",
        "ply_file": str(ply_path.relative_to(ROOT)),
        "feat_file": str(feat_path.relative_to(ROOT)),
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
