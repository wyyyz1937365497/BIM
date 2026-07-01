"""Multi-height wall line extraction probe.

Scans the SceneSplat scene at multiple heights, extracts wall lines via
grid rasterization + morphological closing + contour extraction +
Douglas-Peucker simplification, and outputs:
  - output/wall_lines.json: wall endpoints
  - output/wall_lines_topdown.png: top-down wall visualization

Run with vcvars64:
    cmd /c "...\\vcvars64.bat && python scripts/wall_line_probe.py"
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
    data_dir = ROOT / "data"
    feat_path = ROOT / "output" / "data_feat.pt"
    text_emb_path = ROOT / "data" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "bim_class_names.json"

    print("Loading scene...")
    scene = GSScene.from_npy(
        data_dir,
        feat_path=feat_path,
        text_emb_path=text_emb_path,
        class_names_path=class_names_path,
    )
    print(f"Loaded {scene.num_gaussians} Gaussians")

    # Detect up_axis + floor/ceiling
    floor_result = scene.query_semantics("floor", mode="dominant")
    up_axis = int(np.argmin(floor_result["centroid"]))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_result["centroid"][up_axis])
    ceiling_result = scene.query_semantics("ceiling", mode="dominant")
    ceiling_z = float(ceiling_result["centroid"][up_axis])
    center_x = float(floor_result["centroid"][h_axes[0]])
    center_y = float(floor_result["centroid"][h_axes[1]])
    print(f"up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")
    print(f"Scan center: ({center_x:.2f}, {center_y:.2f})")

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

    # Extract wall lines
    print("\nExtracting wall lines...")
    wall_lines, wall_pts = extract_wall_lines(
        scans,
        wall_class_idx=0,
        exclude_classes=[1, 2, 8],
        center=np.array([center_x, center_y]),
    )
    print(f"Wall points (semantic=wall): {len(wall_pts)}")
    print(f"Extracted {len(wall_lines)} wall lines:")
    for i, wl in enumerate(wall_lines):
        print(
            f"  {i}: ({wl.x1:.2f},{wl.y1:.2f}) -> ({wl.x2:.2f},{wl.y2:.2f}), "
            f"length={wl.length:.2f}m, pts={wl.num_points}"
        )

    # Save JSON
    center_arr = np.array([center_x, center_y])
    output_json = wall_lines_to_json(wall_lines, scans, center_arr)
    json_path = str(ROOT / "output" / "wall_lines.json")
    Path(json_path).write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    print(f"\nWall lines JSON saved to: {json_path}")

    # Save top-down plot
    png_path = str(ROOT / "output" / "wall_lines_topdown.png")
    save_wall_lines_plot(
        wall_lines,
        wall_pts,
        center_arr,
        png_path,
        title=f"Wall Lines from Multi-Height Scan ({len(wall_lines)} walls, {len(wall_pts)} pts)",
    )
    print(f"Top-down plot saved to: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
