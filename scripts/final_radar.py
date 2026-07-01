"""Generate the final radar plot showing the complete VLM-verified door pipeline.

Pipeline stages visualized:
  1. Radar scan at door height (floor + 1.0m) — semantic-colored polar points
  2. Wall lines from wall_line_extractor — blue polygon
  3. feat.pt candidates — yellow markers
  4. VLM-verified doors — green arcs (confirmed) vs red arcs (rejected)

Output: output/room0/final_radar.png

Run with vcvars64:
    cmd /c "...\\vcvars64.bat && python scripts/final_radar.py --name room0"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="room0")
    args = parser.parse_args()

    out_dir = ROOT / "output" / args.name

    # Load existing results
    walls = json.load(open(out_dir / "wall_lines_snapped.json"))
    verified = json.load(open(out_dir / "doors_verified.json"))
    summary = json.load(open(out_dir / "summary.json"))

    scan_center = tuple(summary["center"])
    floor_z = summary["floor_z"]
    up_axis = summary["up_axis"]

    # Load scene and run a single-height scan at door height (floor + 1.0m)
    data_dir = ROOT / "data" / args.name
    ply_path = sorted(data_dir.glob("point_cloud_*.ply"))
    ply_path = ply_path[0] if ply_path else sorted(data_dir.glob("*_feat_vis_3dgs.ply"))[0]
    feat_path = sorted(data_dir.glob("*_feat.pt"))[0]

    from bim_recon.gs_scene import GSScene
    from bim_recon.virtual_scanner import VirtualScanner, SEMANTIC_PALETTE

    print(f"Loading scene: {ply_path.name}")
    scene = GSScene.from_ply(
        ply_path, feat_path=feat_path,
        text_emb_path=str(ROOT / "data" / "0" / "bim_text_emb.pt"),
        class_names_path=str(ROOT / "data" / "0" / "bim_class_names.json"),
    )

    door_height = floor_z + 1.0  # 1m above floor = mid-door
    print(f"Scanning at door height: z={door_height:.2f}m (floor+1.0m)")
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scan = scanner.scan(
        center_2d=scan_center, height=door_height,
        num_views=8, fov=60.0, width=512,
    )
    print(f"Scan points: {len(scan.angles_deg)}")

    # ---- Visualization ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    fig = plt.figure(figsize=(18, 16))

    # ==== Panel 1: Polar Radar ====
    ax_polar = fig.add_subplot(2, 2, 1, projection="polar")
    mask = scan.distances <= 15.0
    angles_rad = np.radians(scan.angles_deg[mask])
    dists = scan.distances[mask]
    sem = scan.semantic_labels
    assert sem is not None
    labels = sem[mask]

    # Color by semantic class
    colors = np.array([
        SEMANTIC_PALETTE[l] if 0 <= l < len(SEMANTIC_PALETTE) else (0.5, 0.5, 0.5)
        for l in labels
    ])

    # Plot non-door, non-wall points faintly
    other_mask = ~((labels == 0) | (labels == 3))
    wall_mask = labels == 0
    door_mask = labels == 3

    if other_mask.sum() > 0:
        ax_polar.scatter(angles_rad[other_mask], dists[other_mask],
                         s=0.3, c=colors[other_mask], alpha=0.2)
    if wall_mask.sum() > 0:
        ax_polar.scatter(angles_rad[wall_mask], dists[wall_mask],
                         s=0.5, c=colors[wall_mask], alpha=0.4)
    if door_mask.sum() > 0:
        ax_polar.scatter(angles_rad[door_mask], dists[door_mask],
                         s=3.0, c="red", alpha=0.8, zorder=5)

    # Mark confirmed doors with green arcs
    for r in verified["results"]:
        theta_c = np.radians(r["theta"])
        r_dist = r["r"]
        theta_span_rad = np.radians(r["candidate"]["theta_span"])
        if r["confirmed"]:
            # Green arc for confirmed
            theta_range = np.linspace(
                theta_c - theta_span_rad / 2,
                theta_c + theta_span_rad / 2, 20,
            )
            ax_polar.plot(theta_range, [r_dist] * len(theta_range),
                          "g-", linewidth=4, alpha=0.8, zorder=10)
            ax_polar.annotate("[DOOR]", (theta_c, r_dist + 0.3),
                              fontsize=8, ha="center", fontweight="bold",
                              color="green")
        else:
            # Red X for rejected
            ax_polar.scatter([theta_c], [r_dist], s=80, c="red",
                             marker="x", zorder=10, linewidths=2)

    ax_polar.set_ylim(0, max(dists.max() + 1, 6))
    ax_polar.set_title("Polar Radar Scan\n(floor+1.0m, red=door pts, green=VLM confirmed)",
                       pad=20, fontsize=11)
    ax_polar.grid(True, alpha=0.3)

    # ==== Panel 2: Top-down with walls + VLM results ====
    ax_td = fig.add_subplot(2, 2, 2)
    pts = scan.points_2d[mask]

    # Plot scan points (faint)
    if wall_mask.sum() > 0:
        wall_pts = pts[wall_mask]
        ax_td.scatter(wall_pts[:, 0], wall_pts[:, 1], s=0.3, c="gray", alpha=0.2)
    if door_mask.sum() > 0:
        door_pts = pts[door_mask]
        ax_td.scatter(door_pts[:, 0], door_pts[:, 1], s=2.0, c="red", alpha=0.5)

    # Plot wall lines
    for i, w in enumerate(walls):
        ax_td.plot([w["x1"], w["x2"]], [w["y1"], w["y2"]],
                   "b-", linewidth=2.5, alpha=0.7, zorder=5)
        mx = (w["x1"] + w["x2"]) / 2
        my = (w["y1"] + w["y2"]) / 2
        ax_td.annotate(f"W{i}", (mx, my), fontsize=7, ha="center",
                       bbox=dict(boxstyle="round,pad=0.15", fc="lightblue", alpha=0.7))

    # Plot VLM results on walls
    for r in verified["results"]:
        c = r["candidate"]
        wi = c["wall_idx"]
        w = walls[wi]
        ws = np.array([w["x1"], w["y1"]])
        we = np.array([w["x2"], w["y2"]])
        p0 = ws + c["t_min"] * (we - ws)
        p1 = ws + c["t_max"] * (we - ws)
        pc = (p0 + p1) / 2

        if r["confirmed"]:
            ax_td.plot([p0[0], p1[0]], [p0[1], p1[1]],
                       "g-", linewidth=5, alpha=0.9, zorder=10)
            ax_td.annotate(
                f"[OK] Door\n{c['width_m']:.1f}m",
                (pc[0], pc[1]), fontsize=7, ha="center", fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="green", alpha=0.85),
                zorder=11,
            )
        else:
            ax_td.plot([p0[0], p1[0]], [p0[1], p1[1]],
                       "r--", linewidth=2, alpha=0.6, zorder=8)
            # Get short reason from VLM response
            vlm = r["vlm_response"]
            reason = "window" if "window" in vlm.lower() or "blind" in vlm.lower() else "painting" if "paint" in vlm.lower() else "?"
            ax_td.annotate(
                f"[X] {reason}",
                (pc[0], pc[1]), fontsize=6, ha="center",
                color="red",
                bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow", alpha=0.8),
                zorder=9,
            )

    ax_td.plot(scan_center[0], scan_center[1], "k+",
               markersize=12, markeredgewidth=2, zorder=12)
    margin = 2.0
    all_x = [w["x1"] for w in walls] + [w["x2"] for w in walls]
    all_y = [w["y1"] for w in walls] + [w["y2"] for w in walls]
    ax_td.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax_td.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax_td.set_aspect("equal")
    h0, h1 = [i for i in range(3) if i != up_axis]
    ax_td.set_xlabel(f"World {'XYZ'[h0]} (m)")
    ax_td.set_ylabel(f"World {'XYZ'[h1]} (m)")
    ax_td.set_title("Top-Down: Walls + VLM-Verified Doors\n(green=✅confirmed, red dashed=❌rejected)",
                    fontsize=11)
    ax_td.grid(True, alpha=0.3)

    # ==== Panel 3: VLM Verification Summary Table ====
    ax_tbl = fig.add_subplot(2, 2, 3)
    ax_tbl.axis("off")
    ax_tbl.set_title("VLM Verification Results (Ollama gemma4:12b)", fontsize=11, pad=10)

    table_data = []
    for i, r in enumerate(verified["results"]):
        c = r["candidate"]
        status = "[OK] CONFIRMED" if r["confirmed"] else "[X] REJECTED"
        vlm_short = r["vlm_response"].split("\n", 1)[-1][:60] if "\n" in r["vlm_response"] else r["vlm_response"][:60]
        table_data.append([
            f"#{i}",
            f"W{c['wall_idx']}",
            f"{c['width_m']:.2f}m",
            f"{c['num_points']}",
            f"θ={r['theta']:.0f}°",
            status,
            vlm_short,
        ])

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=["Cand", "Wall", "Width", "Pts", "Azimuth", "VLM", "Description"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    # Color confirmed rows green, rejected rows red
    for i, r in enumerate(verified["results"]):
        color = "#c8e6c9" if r["confirmed"] else "#ffcdd2"
        for j in range(7):
            table[(i + 1, j)].set_facecolor(color)

    # ==== Panel 4: Pipeline Flow Diagram ====
    ax_flow = fig.add_subplot(2, 2, 4)
    ax_flow.axis("off")
    ax_flow.set_title("Pipeline Flow", fontsize=11, pad=10)

    flow_text = (
        f"Pipeline: radar scan -> feat.pt candidates -> VLM verify\n\n"
        f"  Stage 1: Multi-height scan\n"
        f"    -> {len(scan.angles_deg)} points at door height\n"
        f"    -> Semantic labels from feat.pt\n\n"
        f"  Stage 2: Candidate extraction (class=door)\n"
        f"    -> {verified['total_candidates']} raw candidates\n"
        f"    -> {verified['after_prefilter']} after pre-filter\n"
        f"       (width>=0.7m, pts>=100)\n\n"
        f"  Stage 3: 3DGS render + Ollama VLM\n"
        f"    -> {verified['confirmed']} confirmed doors\n"
        f"    -> {verified['rejected']} rejected (false positives)\n"
        f"    -> VLM: {verified['errors']} errors\n\n"
        f"  Scene: {verified['ply_used']}\n"
        f"  Model: {verified['ollama_model']}\n"
        f"  Result: 14 -> 5 -> 2 confirmed doors"
    )
    ax_flow.text(0.05, 0.95, flow_text, transform=ax_flow.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="gray", alpha=0.8))

    # Legend at bottom
    legend_elems = [
        Patch(facecolor="gray", alpha=0.4, label="Wall points"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=6, label="Door points (feat.pt)"),
        Line2D([0], [0], color="blue", linewidth=2, label="Wall lines"),
        Line2D([0], [0], color="green", linewidth=4, label="[OK] VLM-confirmed door"),
        Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="[X] VLM-rejected"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=9)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    png_path = str(out_dir / "final_radar.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFinal radar saved to: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
