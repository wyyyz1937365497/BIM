"""3DGS → BIM unified pipeline.

Single entry point: provide original 3DGS scene + feat.pt, get walls + doors +
windows detected.

Pipeline stages:
  1. Load scene (point_cloud.ply + feat.pt)
  2. Multi-height radar scan
  3. Wall extraction (grid + morphology + contour + DP + PCA)
  4. Element detection per type (door, window):
     feat.pt candidates → pre-filter → VLM verify (Ollama)
  5. Output JSON files

  (Planned) Stage 6: Push to Revit via MCP tools — not yet implemented.

Usage:
    cmd /c "...\\vcvars64.bat && python scripts/run_pipeline.py --name room0"
    cmd /c "...\\vcvars64.bat && python scripts/run_pipeline.py --name room0 --skip-vlm"
    cmd /c "...\\vcvars64.bat && python scripts/run_pipeline.py --name room0 --elements door window column"

Outputs:
    output/<name>/wall_lines_snapped.json     — wall endpoints (closed polygon)
    output/<name>/doors_verified.json         — VLM-confirmed doors
    output/<name>/windows_verified.json       — VLM-confirmed windows
    output/<name>/pipeline_report.json        — full pipeline summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.candidate_extractor import (
    extract_candidates,
    prefilter_candidates,
)
from bim_recon.element_config import ElementConfig, get_element_config, list_element_types
from bim_recon.gs_scene import GSScene
from bim_recon.height_detector import detect_element_heights
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.vlm_verifier import VerificationResult, verify_candidates
from bim_recon.wall_line_extractor import (
    extract_wall_lines,
    multi_height_scan,
    save_wall_lines_plot,
    wall_lines_to_json,
)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def find_scene_files(data_dir: Path) -> tuple[Path, Path]:
    """Auto-discover PLY + feat.pt, preferring original weights."""
    original = sorted(data_dir.glob("point_cloud_*.ply"))
    feat_vis = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))
    ply = original[0] if original else feat_vis[0]
    feat = sorted(data_dir.glob("*_feat.pt"))[0]
    return ply, feat


def detect_coordinate_system(scene: GSScene) -> dict:
    """Auto-detect up_axis, floor_z, ceiling_z, scan center."""
    floor_c = np.array(scene.query_semantics("floor", mode="dominant")["centroid"])
    ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant")["centroid"])
    up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
    h_axes = [i for i in range(3) if i != up_axis]
    return {
        "up_axis": up_axis,
        "h_axes": h_axes,
        "floor_z": float(floor_c[up_axis]),
        "ceiling_z": float(ceiling_c[up_axis]),
        "center": (float(floor_c[h_axes[0]]), float(floor_c[h_axes[1]])),
    }


# ---------------------------------------------------------------------------
# Wall extraction
# ---------------------------------------------------------------------------

def extract_walls(
    scans: list,
    center: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """Extract wall lines from multi-height scan data."""
    wall_lines, wall_pts = extract_wall_lines(
        scans,
        wall_class_idx=0,
        exclude_classes=[1, 2, 8],
        center=center,
    )
    # Save raw wall lines
    output_json = wall_lines_to_json(wall_lines, scans, center)
    json_path = out_dir / "wall_lines.json"
    json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")

    # Save plot
    png_path = str(out_dir / "wall_lines_topdown.png")
    save_wall_lines_plot(
        wall_lines, wall_pts, center, png_path,
        title=f"Walls ({len(wall_lines)} segments)",
    )

    # Return as simple dicts
    return [
        {"x1": wl.x1, "y1": wl.y1, "x2": wl.x2, "y2": wl.y2, "length": wl.length}
        for wl in wall_lines
    ]


def snap_wall_endpoints(walls: list[dict], threshold: float = 0.5) -> list[dict]:
    """Snap nearby wall endpoints to ensure closed polygon."""
    eps = list([w["x1"], w["y1"], w["x2"], w["y2"], w["length"]] for w in walls)
    changed = True
    iteration = 0
    while changed and iteration < 10:
        changed = False
        iteration += 1
        points = []
        for i, ep in enumerate(eps):
            points.append(("s", i, ep[0], ep[1]))
            points.append(("e", i, ep[2], ep[3]))
        snapped = set()
        for i, (t1, idx1, x1, y1) in enumerate(points):
            if i in snapped:
                continue
            group = [(t1, idx1, x1, y1)]
            for j, (t2, idx2, x2, y2) in enumerate(points[i + 1:], i + 1):
                if j in snapped:
                    continue
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist < threshold and dist > 1e-6:
                    group.append((t2, idx2, x2, y2))
                    snapped.add(j)
            if len(group) > 1:
                changed = True
                avg_x = sum(p[2] for p in group) / len(group)
                avg_y = sum(p[3] for p in group) / len(group)
                for t, idx, _, _ in group:
                    if t == "s":
                        eps[idx][0] = avg_x
                        eps[idx][1] = avg_y
                    else:
                        eps[idx][2] = avg_x
                        eps[idx][3] = avg_y
    return [
        {"x1": ep[0], "y1": ep[1], "x2": ep[2], "y2": ep[3], "length": ep[4]}
        for ep in eps
    ]


# ---------------------------------------------------------------------------
# Element detection (doors, windows, ...)
# ---------------------------------------------------------------------------

def detect_elements(
    cfg: ElementConfig,
    scans: list,
    walls: list[dict],
    coords: dict,
    scene: GSScene,
    out_dir: Path,
    ollama_model: str,
    skip_vlm: bool = False,
) -> dict:
    """Detect elements of one type from scan data + VLM verification."""
    center = coords["center"]
    floor_z = coords["floor_z"]
    up_axis = coords["up_axis"]

    # Extract candidates
    candidates = extract_candidates(
        scans, walls, floor_z, center,
        element_class=cfg.name,
        class_idx=cfg.class_idx,
        project_to_walls=cfg.structural,
    )

    # Pre-filter
    filtered = prefilter_candidates(candidates, cfg.min_width, cfg.min_points)
    print(f"  [{cfg.name}] {len(candidates)} candidates -> "
          f"{len(filtered)} after pre-filter")

    if not filtered:
        print(f"  [{cfg.name}] no candidates passed pre-filter")
        return {
            "element": cfg.name,
            "total_candidates": len(candidates),
            "after_prefilter": 0,
            "confirmed": 0,
            "rejected": 0,
            "results": [],
        }

    # VLM verification
    verify_dir = out_dir / cfg.verify_dir_name
    results = verify_candidates(
        filtered, scene, center, floor_z, verify_dir,
        element_class=cfg.name,
        ollama_model=ollama_model,
        up_axis=up_axis,
        vlm_hint=cfg.vlm_hint,
        skip_vlm=skip_vlm,
    )

    confirmed = [r for r in results if r.confirmed is True]
    rejected = [r for r in results if r.confirmed is False]
    print(f"  [{cfg.name}] {len(confirmed)} confirmed, {len(rejected)} rejected")

    # Height detection for confirmed wall-mounted elements
    height_results: list[dict | None] = [None] * len(results)
    if cfg.height_detection and confirmed:
        ceiling_z = coords["ceiling_z"]
        for i, r in enumerate(results):
            if not r.confirmed:
                continue
            wi = r.candidate.wall_idx
            if wi is None or wi >= len(walls):
                continue
            hr = detect_element_heights(
                scene, r.candidate, walls[wi],
                floor_z, ceiling_z, center,
                class_idx=cfg.class_idx,
                up_axis=up_axis,
            )
            height_results[i] = {
                "sill_height": hr.sill_height,
                "header_height": hr.header_height,
                "element_height": hr.element_height,
                "confidence": hr.confidence,
                "method": hr.method,
            }
            print(f"    [{cfg.name}] height: sill={hr.sill_height:.3f}m "
                  f"header={hr.header_height:.3f}m ({hr.method})")

    result_dicts = []
    for i, r in enumerate(results):
        d = r.to_dict()
        if height_results[i] is not None:
            d["height_detection"] = height_results[i]
        result_dicts.append(d)

    return {
        "element": cfg.name,
        "total_candidates": len(candidates),
        "after_prefilter": len(filtered),
        "confirmed": len(confirmed),
        "rejected": len(rejected),
        "results": result_dicts,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="3DGS -> BIM unified pipeline (walls + doors + windows)"
    )
    parser.add_argument("--name", required=True,
                        help="Scene name (data/<name>)")
    parser.add_argument("--elements", nargs="+",
                        default=["door", "window"],
                        help=f"Element types to detect: {list_element_types()}")
    parser.add_argument("--num-heights", type=int, default=12,
                        help="Number of scan heights")
    parser.add_argument("--ollama-model", default="gemma4:12b")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM verification (render only)")
    parser.add_argument("--snap-threshold", type=float, default=0.5,
                        help="Wall endpoint snap threshold (m)")
    args = parser.parse_args()

    data_dir = ROOT / "data" / args.name
    out_dir = ROOT / "output" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Stage 1: Load scene ===
    ply_path, feat_path = find_scene_files(data_dir)
    print(f"{'='*60}")
    print(f"3DGS -> BIM Pipeline: {args.name}")
    print(f"{'='*60}")
    print(f"  PLY:  {ply_path.name}")
    print(f"  Feat: {feat_path.name}")

    scene = GSScene.from_ply(
        ply_path, feat_path=feat_path,
        text_emb_path=str(ROOT / "data" / "0" / "bim_text_emb.pt"),
        class_names_path=str(ROOT / "data" / "0" / "bim_class_names.json"),
    )
    print(f"  Gaussians: {scene.num_gaussians}")

    # === Stage 2: Detect coordinate system ===
    coords = detect_coordinate_system(scene)
    center = coords["center"]
    floor_z = coords["floor_z"]
    ceiling_z = coords["ceiling_z"]
    up_axis = coords["up_axis"]
    print(f"  up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")
    print(f"  Scan center: ({center[0]:.2f}, {center[1]:.2f})")

    # === Stage 3: Multi-height scan (shared) ===
    print(f"\n--- Stage 1: Radar Scan ({args.num_heights} heights) ---")
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scans = multi_height_scan(
        scanner, center, floor_z, ceiling_z,
        num_heights=args.num_heights, num_views=8, width=512,
    )
    total_pts = sum(len(s.angles_deg) for s in scans)
    print(f"  Total scan points: {total_pts}")

    # === Stage 4: Wall extraction ===
    print(f"\n--- Stage 2: Wall Extraction ---")
    walls = extract_walls(scans, np.array(center), out_dir)
    print(f"  Extracted {len(walls)} wall segments")

    # Snap endpoints
    walls_snapped = snap_wall_endpoints(walls, args.snap_threshold)
    snapped_path = out_dir / "wall_lines_snapped.json"
    snapped_path.write_text(json.dumps(walls_snapped, indent=2), encoding="utf-8")
    print(f"  Snapped walls saved: {snapped_path}")

    # === Stage 5: Element detection ===
    print(f"\n--- Stage 3: Element Detection ---")
    all_results = {}
    for elem_type in args.elements:
        try:
            cfg = get_element_config(elem_type)
        except KeyError:
            print(f"  Unknown element type '{elem_type}', skipping")
            continue

        result = detect_elements(
            cfg, scans, walls_snapped, coords, scene,
            out_dir, args.ollama_model, args.skip_vlm,
        )
        all_results[elem_type] = result

        # Save per-element JSON
        elem_json = {
            "scene": args.name,
            "element": elem_type,
            "ply_used": ply_path.name,
            "ollama_model": args.ollama_model if not args.skip_vlm else None,
            **result,
        }
        elem_path = out_dir / cfg.output_json_name
        elem_path.write_text(json.dumps(elem_json, indent=2), encoding="utf-8")

    # === Stage 6: Pipeline report ===
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Walls:    {len(walls_snapped)} segments (closed polygon)")
    for elem_type, result in all_results.items():
        print(f"  {elem_type:10s} {result['confirmed']} confirmed / "
              f"{result['after_prefilter']} filtered / "
              f"{result['total_candidates']} raw")

    report = {
        "scene": args.name,
        "ply": ply_path.name,
        "feat": feat_path.name,
        "num_gaussians": scene.num_gaussians,
        "coordinate_system": {
            "up_axis": up_axis,
            "floor_z": floor_z,
            "ceiling_z": ceiling_z,
            "center": list(center),
        },
        "scan": {
            "num_heights": args.num_heights,
            "total_points": total_pts,
        },
        "walls": {
            "count": len(walls_snapped),
            "snapped": True,
        },
        "elements": all_results,
        "vlm_model": args.ollama_model if not args.skip_vlm else None,
    }
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report: {report_path}")
    print(f"  Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
