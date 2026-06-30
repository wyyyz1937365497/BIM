"""Virtual 2D laser scan probe: render a radar scan from the SceneSplat 3DGS scene.

Run with vcvars64 (gsplat JIT needs MSVC):
    cmd /c "...\\vcvars64.bat && python scripts/virtual_scan_probe.py"

Outputs a PNG radar plot at output/virtual_scan_h{height}m.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.gs_scene import GSScene
from bim_recon.virtual_scanner import VirtualScanner, save_scan_plot


def main() -> int:
    data_dir = ROOT / "data"
    feat_path = ROOT / "output" / "data_feat.pt"
    text_emb_path = ROOT / "data" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "bim_class_names.json"

    print(f"Loading scene from {data_dir}...")
    scene = GSScene.from_npy(
        data_dir,
        feat_path=feat_path,
        text_emb_path=text_emb_path,
        class_names_path=class_names_path,
    )
    print(f"Loaded {scene.num_gaussians} Gaussians")

    # Detect up_axis from floor centroid
    floor_result = scene.query_semantics("floor", mode="dominant")
    up_axis = int(np.argmin(floor_result["centroid"]))
    print(f"Detected up_axis={up_axis}")

    h_axes = [i for i in range(3) if i != up_axis]
    floor_centroid = floor_result["centroid"]
    center_x = floor_centroid[h_axes[0]]
    center_y = floor_centroid[h_axes[1]]
    floor_z = floor_centroid[up_axis]
    print(f"Floor centroid: ({center_x:.2f}, {center_y:.2f}, {floor_z:.3f})")

    # Scan height: 1.5m above floor (typical wall-height scan)
    scan_height = floor_z + 1.5
    print(f"Scan height: {scan_height:.3f}m (floor + 1.5m)")

    # Run virtual scan
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scan = scanner.scan(
        center_2d=(center_x, center_y),
        height=scan_height,
        num_views=8,
        fov=60.0,
        width=1024,
    )

    print(f"\nScan result: {len(scan.angles_deg)} points")
    valid = scan.distances[scan.distances > 0.01]
    if len(valid) > 0:
        print(f"  Distance range: {valid.min():.2f}m - {valid.max():.2f}m")
        print(f"  Mean distance: {valid.mean():.2f}m")

    # Save radar plot
    output_path = str(ROOT / "output" / f"virtual_scan_h{scan_height:.1f}m.png")
    save_scan_plot(
        scan,
        output_path,
        max_distance=12.0,
        title=f"Virtual 2D Scan (h={scan_height:.2f}m, {len(scan.angles_deg)} pts)",
    )
    print(f"\nRadar plot saved to: {output_path}")

    # Also save raw data as JSON
    import json
    json_path = str(ROOT / "output" / f"virtual_scan_h{scan_height:.1f}m.json")
    Path(json_path).write_text(
        json.dumps(scan.to_dict(), indent=2), encoding="utf-8",
    )
    print(f"Raw data saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
