"""Standalone height-detection runner.

Loads an already-verified element JSON (windows_verified.json or
doors_verified.json), reconstructs Candidate objects, and runs
:func:`detect_element_heights` against the live 3DGS scene.

Outputs:
  - ``<element>_heights.json``  — sill/header/confidence per element
  - ``<element>_heights_viz.png`` — wall-elevation visualization

Usage (needs MSVC for gsplat JIT):

    cmd /c "\"...\\vcvars64.bat\" && python scripts/run_height_detection.py \
        --name room0 --element window"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bim_recon.candidate_extractor import Candidate
from bim_recon.gs_scene import GSScene
from bim_recon.height_detector import detect_element_heights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_from_dict(d: dict) -> Candidate:
    """Reconstruct a Candidate from its serialized dict form."""
    return Candidate(
        element_class=d["element_class"],
        class_idx=d["class_idx"],
        wall_idx=d["wall_idx"],
        t_min=d["t_min"],
        t_max=d["t_max"],
        theta_center=d["theta_center"],
        theta_span=d["theta_span"],
        r_mean=d["r_mean"],
        h_min=d["h_min"],
        h_max=d["h_max"],
        width_m=d["width_m"],
        num_points=d["num_points"],
        world_x=d["world_x"],
        world_y=d["world_y"],
    )


def _wall_direction(wall: dict) -> np.ndarray:
    """Unit direction vector along the wall (used for the elevation x-axis)."""
    start = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
    end = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
    d = end - start
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _render_elevation(
    walls: list[dict],
    floor_z: float,
    ceiling_z: float,
    results: list[dict],
    out_path: Path,
) -> None:
    """Draw a per-wall elevation chart marking detected openings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    wall_height = ceiling_z - floor_z
    # Group openings by wall index.
    by_wall: dict[int, list[dict]] = {}
    for r in results:
        wi = r["wall_idx"]
        if wi is not None:
            by_wall.setdefault(wi, []).append(r)

    n_walls = len(walls)
    fig, axes = plt.subplots(1, n_walls, figsize=(4 * n_walls, 5))
    if n_walls == 1:
        axes = [axes]

    for wi, wall in enumerate(walls):
        ax = axes[wi]
        length = wall["length"]
        # Wall body.
        ax.add_patch(Rectangle((0, 0), length, wall_height,
                               facecolor="#d9d9d9", edgecolor="#404040", lw=1.5))
        # Openings on this wall.
        for r in by_wall.get(wi, []):
            cand = r["candidate"]
            # Position along wall from t-parameter.
            t_mid = (cand["t_min"] + cand["t_max"]) / 2.0
            x_mid = t_mid * length
            w = cand["width_m"]
            sill = r["sill_height"]
            hdr = r["header_height"]
            conf = r["confidence"]
            color = "#4ba3ff" if r["method"] != "fallback" else "#ff9a4b"
            ax.add_patch(Rectangle((x_mid - w / 2, sill), w, hdr - sill,
                                   facecolor=color, edgecolor="#1a3a5c", lw=1.5,
                                   alpha=0.85))
            ax.annotate(
                f"h={hdr - sill:.2f}m\nsill={sill:.2f}\nc={conf}",
                xy=(x_mid, hdr), xytext=(x_mid, hdr + 0.15),
                ha="center", fontsize=7, color="#1a3a5c",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.5),
            )
        ax.set_xlim(-0.5, length + 0.5)
        ax.set_ylim(-0.2, wall_height + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Wall {wi}  ({length:.2f}m)", fontsize=9)
        ax.set_xlabel("along wall (m)", fontsize=8)
        ax.set_ylabel("height above floor (m)", fontsize=8)

    fig.suptitle("Detected window openings (height detection)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="scene name (e.g. room0)")
    ap.add_argument("--element", default="window",
                    choices=["window", "door"],
                    help="which verified JSON to process")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    name = args.name
    element = args.element
    data_dir = Path(args.data_root) / name
    out_dir = Path(args.output_root) / name

    # --- load existing pipeline artifacts --------------------------------
    verified_path = out_dir / f"{element}s_verified.json"
    walls_path = out_dir / "wall_lines_snapped.json"
    report_path = out_dir / "pipeline_report.json"
    for p in (verified_path, walls_path, report_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    verified = json.loads(verified_path.read_text())
    walls = json.loads(walls_path.read_text())
    report = json.loads(report_path.read_text())

    coords = report["coordinate_system"]
    floor_z = coords["floor_z"]
    ceiling_z = coords["ceiling_z"]
    center = tuple(coords["center"])
    up_axis = coords["up_axis"]

    # --- load scene ------------------------------------------------------
    ply_path = data_dir / "point_cloud_30000.ply"
    feat_path = data_dir / f"{name}_feat.pt"
    text_emb_path = Path(args.data_root) / "0" / "bim_text_emb.pt"
    class_names_path = Path(args.data_root) / "0" / "bim_class_names.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading scene from {ply_path} (device={device}) ...")
    scene = GSScene.from_ply(
        ply_path,
        device=device,
        feat_path=feat_path,
        text_emb_path=text_emb_path,
        class_names_path=class_names_path,
    )
    print(f"  {scene.num_gaussians} Gaussians loaded")

    # --- run height detection on confirmed elements ----------------------
    confirmed = [
        r for r in verified["results"] if r.get("confirmed") is True
    ]
    print(f"\n{len(confirmed)} confirmed {element}(s) to process\n")

    results_out: list[dict] = []
    for i, r in enumerate(confirmed):
        cand_dict = r["candidate"]
        cand = _candidate_from_dict(cand_dict)
        wi = cand.wall_idx
        if wi is None or wi >= len(walls):
            print(f"  [{i}] wall_idx={wi} out of range, skipping")
            continue
        wall = walls[wi]

        print(f"  [{i}] Wall {wi}  t=[{cand.t_min:.3f},{cand.t_max:.3f}]  "
              f"xy=({cand.world_x:.3f},{cand.world_y:.3f})")
        hr = detect_element_heights(
            scene, cand, wall,
            floor_z, ceiling_z, center,
            class_idx=cand.class_idx,
            up_axis=up_axis,
        )
        print(f"      -> sill={hr.sill_height:.3f}m  header={hr.header_height:.3f}m  "
              f"h={hr.element_height:.3f}m  conf={hr.confidence}  ({hr.method})")

        results_out.append({
            "wall_idx": wi,
            "candidate": cand_dict,
            "sill_height": hr.sill_height,
            "header_height": hr.header_height,
            "element_height": hr.element_height,
            "confidence": hr.confidence,
            "method": hr.method,
        })

    # --- save JSON -------------------------------------------------------
    out_json = out_dir / f"{element}_heights.json"
    out_json.write_text(json.dumps({
        "scene": name,
        "element": element,
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "wall_height": ceiling_z - floor_z,
        "up_axis": up_axis,
        "count": len(results_out),
        "results": results_out,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")

    # --- visualization ---------------------------------------------------
    out_png = out_dir / f"{element}_heights_viz.png"
    _render_elevation(walls, floor_z, ceiling_z, results_out, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
