"""End-to-end VLM-verified element extraction.

Pipeline:
  1. Load 3DGS scene (prefer original weights point_cloud_*.ply).
  2. Run multi-height radar scan (or reuse existing wall lines).
  3. Extract candidates for the requested element type.
  4. Pre-filter by physical constraints.
  5. Render targeted images + verify via Ollama VLM.
  6. Save confirmed/rejected results.

Usage:
    cmd /c "...\\vcvars64.bat && python scripts/verify_elements.py --name room0 --element door"
    cmd /c "...\\vcvars64.bat && python scripts/verify_elements.py --name room0 --element window --min-width 0.5"
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
    Candidate,
    extract_candidates,
    prefilter_candidates,
)
from bim_recon.element_config import get_element_config, list_element_types
from bim_recon.gs_scene import GSScene
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.vlm_verifier import VerificationResult, verify_candidates
from bim_recon.wall_line_extractor import multi_height_scan


def find_scene_files(data_dir: Path) -> tuple[Path, Path]:
    """Auto-discover PLY + feat.pt, preferring original weights."""
    original = sorted(data_dir.glob("point_cloud_*.ply"))
    feat_vis = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))
    ply = original[0] if original else feat_vis[0]
    feat = sorted(data_dir.glob("*_feat.pt"))[0]
    return ply, feat


def load_walls(out_dir: Path) -> list[dict] | None:
    """Load snapped wall lines if available."""
    for name in ("wall_lines_snapped.json", "wall_lines.json"):
        p = out_dir / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM-verified element extraction from 3DGS"
    )
    parser.add_argument("--name", required=True, help="Scene name (data/<name>)")
    parser.add_argument("--element", default="door",
                        help=f"Element class: {', '.join(list_element_types())}")
    parser.add_argument("--min-width", type=float, default=None,
                        help="Min width to keep candidate (m). Default: per-type from config.")
    parser.add_argument("--min-points", type=int, default=None,
                        help="Min scan points to keep candidate. Default: per-type from config.")
    parser.add_argument("--ollama-model", default="gemma4:12b")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Only render images, skip VLM verification")
    parser.add_argument("--num-heights", type=int, default=12,
                        help="Number of scan heights")
    args = parser.parse_args()

    element = args.element.lower()
    try:
        cfg = get_element_config(element)
    except KeyError:
        print(f"ERROR: Unknown element '{element}'. "
              f"Valid: {list_element_types()}")
        return 1

    # Use CLI override or config default
    min_width: float = args.min_width if args.min_width is not None else cfg.min_width
    min_points: int = args.min_points if args.min_points is not None else cfg.min_points

    data_dir = ROOT / "data" / args.name
    out_dir = ROOT / "output" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load scene ---
    ply_path, feat_path = find_scene_files(data_dir)
    is_original = "point_cloud" in ply_path.name
    print(f"Scene: {args.name}")
    print(f"  PLY:  {ply_path.name} ({'original' if is_original else 'feat_vis'})")
    print(f"  Feat: {feat_path.name}")
    scene = GSScene.from_ply(
        ply_path, feat_path=feat_path,
        text_emb_path=str(ROOT / "data" / "0" / "bim_text_emb.pt"),
        class_names_path=str(ROOT / "data" / "0" / "bim_class_names.json"),
    )

    # --- Detect coordinate system ---
    floor_c = np.array(scene.query_semantics("floor", mode="dominant")["centroid"])
    ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant")["centroid"])
    up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_c[up_axis])
    ceiling_z = float(ceiling_c[up_axis])
    cx = float(floor_c[h_axes[0]])
    cy = float(floor_c[h_axes[1]])
    print(f"  up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")

    # --- Load or scan walls ---
    walls = load_walls(out_dir)
    if walls:
        print(f"  Walls: loaded {len(walls)} from existing file")
    else:
        print("  Walls: running wall line extraction...")
        scanner = VirtualScanner(scene, up_axis=up_axis)
        scans = multi_height_scan(
            scanner, (cx, cy), floor_z, ceiling_z,
            num_heights=args.num_heights, num_views=8, width=512,
        )
        from bim_recon.wall_line_extractor import (
            extract_wall_lines, wall_lines_to_json,
        )
        wall_lines, _ = extract_wall_lines(
            scans, wall_class_idx=0, exclude_classes=[1, 2, 8],
            center=np.array([cx, cy]),
        )
        walls = [
            {"x1": wl.x1, "y1": wl.y1, "x2": wl.x2, "y2": wl.y2,
             "length": wl.length}
            for wl in wall_lines
        ]
        print(f"  Walls: extracted {len(walls)} wall lines")

    # --- Multi-height scan for element candidates ---
    print(f"\nScanning at {args.num_heights} heights for '{element}' candidates...")
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scans = multi_height_scan(
        scanner, (cx, cy), floor_z, ceiling_z,
        num_heights=args.num_heights, num_views=8, width=512,
    )
    total_pts = sum(len(s.angles_deg) for s in scans)
    print(f"  Total scan points: {total_pts}")

    class_idx = cfg.class_idx
    is_structural = cfg.structural

    # --- Extract candidates ---
    candidates = extract_candidates(
        scans, walls, floor_z, (cx, cy),
        element_class=element,
        class_idx=class_idx,
        project_to_walls=is_structural,
    )
    print(f"\nCandidates from feat.pt: {len(candidates)}")
    for c in candidates:
        print(f"  Wall {c.wall_idx}: t={c.t_min:.2f}-{c.t_max:.2f}, "
              f"width={c.width_m:.2f}m, pts={c.num_points}, "
              f"θ={c.theta_center:.1f}°")

    # --- Pre-filter ---
    filtered = prefilter_candidates(
        candidates, min_width, min_points,
    )
    print(f"\nAfter pre-filter (width>={min_width}m, pts>={min_points}): "
          f"{len(filtered)}")

    if not filtered:
        print("No candidates passed pre-filter. Exiting.")
        return 0

    # --- VLM verification ---
    verify_dir = out_dir / cfg.verify_dir_name
    print(f"\nVLM verification ({'Ollama ' + args.ollama_model if not args.skip_vlm else 'SKIP'})...")

    def progress(i: int, total: int, r: VerificationResult) -> None:
        status = "CONFIRMED" if r.confirmed else "REJECTED" if r.confirmed is False else "ERROR"
        print(f"  [{i+1}/{total}] θ={r.theta:.1f}° → {status}")

    results = verify_candidates(
        filtered, scene, (cx, cy), floor_z, verify_dir,
        element_class=element,
        ollama_model=args.ollama_model,
        up_axis=up_axis,
        skip_vlm=args.skip_vlm,
        progress_callback=progress,
    )

    # --- Summary ---
    confirmed = [r for r in results if r.confirmed is True]
    rejected = [r for r in results if r.confirmed is False]
    errors = [r for r in results if r.confirmed is None]

    output = {
        "scene": args.name,
        "element": element,
        "ply_used": ply_path.name,
        "ollama_model": args.ollama_model if not args.skip_vlm else None,
        "total_candidates": len(candidates),
        "after_prefilter": len(filtered),
        "confirmed": len(confirmed),
        "rejected": len(rejected),
        "errors": len(errors),
        "results": [r.to_dict() for r in results],
    }
    out_path = out_dir / cfg.output_json_name
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Results: {len(confirmed)} confirmed, {len(rejected)} rejected, "
          f"{len(errors)} errors")
    for r in results:
        status = "[OK]" if r.confirmed else "[X]" if r.confirmed is False else "[?]"
        preview = r.vlm_response[:80].replace("\n", " ") if r.vlm_response else ""
        print(f"  {status} Wall {r.candidate.wall_idx}, "
              f"theta={r.theta:.1f}, width={r.candidate.width_m:.2f}m  {preview}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
