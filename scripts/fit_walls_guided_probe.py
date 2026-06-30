"""Real-data probe for floorplan-guided wall fitting.

Run inside vcvars64 environment so gsplat can JIT-compile CUDA on first run.

Example:
    cmd /c "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat && python scripts/fit_walls_guided_probe.py"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.floorplan import ManualProvider
from bim_recon.floorplan_registration import register_floorplan
from bim_recon.gs_scene import GSScene
from bim_recon.wall_fitter import FloorPlanGuidedFitter


def main() -> int:
    data_dir = ROOT / "data"
    feat_path = ROOT / "output" / "data_feat.pt"
    text_emb_path = ROOT / "data" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "bim_class_names.json"
    floorplan_path = ROOT / "SceneSplat" / "floorplan_manual.json"
    evidence_path = ROOT / ".omo" / "evidence" / "task-5-floorplan-guided.txt"

    print(f"Loading scene from {data_dir}...")
    scene = GSScene.from_npy(
        data_dir,
        feat_path=feat_path,
        text_emb_path=text_emb_path,
        class_names_path=class_names_path,
    )
    print(f"Loaded {scene.num_gaussians} Gaussians")

    print(f"Loading floorplan from {floorplan_path}...")
    floorplan = ManualProvider.from_json(floorplan_path).get_floorplan()
    print(f"Floorplan has {len(floorplan.walls)} wall segments")

    # Auto-detect up_axis from floor centroid
    floor_result = scene.query_semantics("floor", mode="dominant")
    up_axis = int(np.argmin(floor_result["centroid"]))
    print(f"Detected up_axis={up_axis}")

    wall_result = scene.query_semantics("wall", mode="dominant")
    wall_indices = wall_result["indices"]
    print(f"Wall Gaussians (dominant): {len(wall_indices)}")

    wall_means = scene.means[
        torch.as_tensor(wall_indices, dtype=torch.long)
    ].cpu().numpy().astype(np.float64)

    floor_z = float(floor_result["centroid"][up_axis])
    ceiling_result = scene.query_semantics("ceiling", mode="dominant")
    ceiling_z = float(ceiling_result["centroid"][up_axis])
    print(f"floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}, height={ceiling_z - floor_z:.3f}")

    # Register floorplan. The registration uses the floor footprint for
    # rotation and polygon-inclusion scoring (cleaner than noisy wall
    # Gaussians) and the wall footprint for corridor scoring.
    h_axes = [i for i in range(3) if i != up_axis]
    floor_indices = floor_result["indices"]
    floor_means = scene.means[
        torch.as_tensor(floor_indices, dtype=torch.long)
    ].cpu().numpy().astype(np.float64)
    floor_means_2d = floor_means[:, h_axes]
    wall_means_2d = wall_means[:, h_axes]
    registered_fp = register_floorplan(
        floorplan,
        wall_means_2d,
        floor_means_2d=floor_means_2d,
        corridor_width=0.5,
    )
    print("\nRegistered floorplan wall segments:")
    for i, w in enumerate(registered_fp.walls):
        print(f"  {i}: ({w.x1:.2f},{w.y1:.2f}) -> ({w.x2:.2f},{w.y2:.2f}), length={w.length():.2f}")

    # Guided fit
    # Fit walls
    fitter = FloorPlanGuidedFitter(corridor_width=0.5)
    walls = fitter.fit_guided(
        wall_means, registered_fp, up_axis=up_axis,
        floor_z=floor_z, ceiling_z=ceiling_z,
    )
    print(f"\nFitted {len(walls)} walls")

    output = {
        "up_axis": up_axis,
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "height": ceiling_z - floor_z,
        "num_walls": len(walls),
        "walls": [w.to_dict() for w in walls],
    }

    for i, w in enumerate(walls):
        print(
            f"  {i}: p0={w.p0.tolist()}, p1={w.p1.tolist()}, "
            f"length={w.length:.2f}, height={w.height:.2f}, "
            f"thickness={w.thickness:.3f}, inliers={w.num_inliers}"
        )

    # Save evidence
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nEvidence saved to {evidence_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
