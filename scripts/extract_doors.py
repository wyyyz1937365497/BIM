"""Extract doors from room0 and visualize on radar scans.

Pipeline:
  1. Multi-height virtual scan (12 heights for better vertical resolution)
  2. Collect door-classified points (semantic_label == 3)
  3. Project onto known wall lines → parameter t along each wall
  4. Cluster to identify door openings (position, width, height range)
  5. Visualize: radar plot at door height + top-down with door segments

Run with vcvars64:
    cmd /c "...\\vcvars64.bat && python scripts/extract_doors.py --name room0"
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
from bim_recon.virtual_scanner import VirtualScanner, SEMANTIC_PALETTE
from bim_recon.wall_line_extractor import multi_height_scan


def project_point_to_wall_segment(
    pt: np.ndarray, wall_start: np.ndarray, wall_end: np.ndarray
) -> tuple[float, float]:
    """Project a 2D point onto a wall segment.

    Returns (t, dist) where:
      t = parameter along wall [0, 1] (clamped)
      dist = perpendicular distance from point to wall line (meters)
    """
    seg = wall_end - wall_start
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return 0.0, float(np.linalg.norm(pt - wall_start))
    t = np.dot(pt - wall_start, seg) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = wall_start + t_clamped * seg
    dist = float(np.linalg.norm(pt - closest))
    return float(t_clamped), dist


def cluster_door_openings(
    door_ts: list[float], door_heights: list[float], wall_length: float,
    min_gap: float = 0.3, min_pts: int = 5,
) -> list[dict]:
    """Cluster door t-parameters into openings.

    Sorts t values, splits where gap > min_gap (meters).
    Each cluster with >= min_pts points is a door opening.
    """
    if len(door_ts) < min_pts:
        return []
    order = np.argsort(door_ts)
    ts = np.array(door_ts)[order]
    hs = np.array(door_heights)[order]

    clusters = []
    start = 0
    for i in range(1, len(ts)):
        gap_m = (ts[i] - ts[i - 1]) * wall_length
        if gap_m > min_gap:
            clusters.append((ts[start:i], hs[start:i]))
            start = i
    clusters.append((ts[start:], hs[start:]))

    openings = []
    for cluster_ts, cluster_hs in clusters:
        if len(cluster_ts) < min_pts:
            continue
        t_center = float(np.mean(cluster_ts))
        t_min = float(np.min(cluster_ts))
        t_max = float(np.max(cluster_ts))
        width = (t_max - t_min) * wall_length
        openings.append({
            "t_center": t_center,
            "t_min": t_min,
            "t_max": t_max,
            "width_m": width,
            "position_m": t_center * wall_length,
            "height_min": float(np.min(cluster_hs)),
            "height_max": float(np.max(cluster_hs)),
            "num_points": len(cluster_ts),
        })
    return openings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="room0", help="Scene name")
    args = parser.parse_args()

    data_dir = ROOT / "data" / args.name
    out_dir = ROOT / "output" / args.name
    snapped_path = out_dir / "wall_lines_snapped.json"

    # Load wall lines
    walls_data = json.load(open(snapped_path))
    walls = []
    for w in walls_data:
        walls.append({
            "start": np.array([w["x1"], w["y1"]]),
            "end": np.array([w["x2"], w["y2"]]),
            "length": w["length"],
        })
    print(f"Loaded {len(walls)} wall lines from {snapped_path}")

    # Load scene
    ply_path = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))[0]
    feat_path = sorted(data_dir.glob("*_feat.pt"))[0]
    text_emb_path = ROOT / "data" / "0" / "bim_text_emb.pt"
    class_names_path = ROOT / "data" / "0" / "bim_class_names.json"

    print(f"Loading scene: {ply_path.name}")
    scene = GSScene.from_ply(
        ply_path, feat_path=feat_path,
        text_emb_path=text_emb_path, class_names_path=class_names_path,
    )

    # Detect coordinate system (same logic as generate_walls.py)
    floor_c = np.array(scene.query_semantics("floor", mode="dominant")["centroid"])
    ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant")["centroid"])
    up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_c[up_axis])
    ceiling_z = float(ceiling_c[up_axis])
    center_x = float(floor_c[h_axes[0]])
    center_y = float(floor_c[h_axes[1]])
    print(f"up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")

    # Multi-height scan with MORE heights for better vertical resolution
    num_heights = 12
    print(f"\nScanning at {num_heights} heights...")
    scanner = VirtualScanner(scene, up_axis=up_axis)
    scans = multi_height_scan(
        scanner,
        center_2d=(center_x, center_y),
        floor_z=floor_z, ceiling_z=ceiling_z,
        num_heights=num_heights, num_views=8, fov=60.0, width=512,
    )

    # Per-height door point count
    print("\nPer-height door point count:")
    for i, s in enumerate(scans):
        door_mask = s.semantic_labels == 3
        wall_mask = s.semantic_labels == 0
        rel_h = s.height - floor_z
        print(f"  H{i}: z={s.height:.2f}m (floor+{rel_h:.2f}m), "
              f"door={door_mask.sum()}, wall={wall_mask.sum()}")

    # ============================================================
    # Step 1: Collect all door points, project onto walls
    # ============================================================
    print("\n--- Door point projection ---")
    wall_door_ts: list[list[float]] = [[] for _ in walls]
    wall_door_heights: list[list[float]] = [[] for _ in walls]

    total_door_pts = 0
    for s in scans:
        door_mask = s.semantic_labels == 3
        if door_mask.sum() == 0:
            continue
        door_pts = s.points_2d[door_mask]
        total_door_pts += len(door_pts)
        rel_h = s.height - floor_z  # height above floor

        for pt in door_pts:
            # Find nearest wall
            best_wall = -1
            best_dist = 1e9
            best_t = 0.0
            for wi, w in enumerate(walls):
                t, dist = project_point_to_wall_segment(pt, w["start"], w["end"])
                if dist < best_dist:
                    best_dist = dist
                    best_wall = wi
                    best_t = t

            # Only keep if within 0.5m of a wall (door is ON a wall)
            if best_dist < 0.5 and best_wall >= 0:
                wall_door_ts[best_wall].append(best_t)
                wall_door_heights[best_wall].append(rel_h)

    print(f"Total door points: {total_door_pts}")
    print(f"Projected onto walls (<0.5m):", end="")
    for wi in range(len(walls)):
        print(f"  W{wi}={len(wall_door_ts[wi])}", end="")
    print()

    # ============================================================
    # Step 2: Cluster door openings per wall
    # ============================================================
    print("\n--- Door openings ---")
    all_doors = []
    for wi, w in enumerate(walls):
        openings = cluster_door_openings(
            wall_door_ts[wi], wall_door_heights[wi], w["length"]
        )
        for opening in openings:
            opening["wall_idx"] = wi
            # World position on wall
            pos = w["start"] + opening["t_center"] * (w["end"] - w["start"])
            opening["world_x"] = float(pos[0])
            opening["world_y"] = float(pos[1])
            all_doors.append(opening)
            print(f"  Wall {wi}: door at t={opening['t_center']:.2f} "
                  f"({opening['position_m']:.2f}m along wall), "
                  f"width={opening['width_m']:.2f}m, "
                  f"height={opening['height_min']:.2f}-{opening['height_max']:.2f}m above floor, "
                  f"pts={opening['num_points']}")

    if not all_doors:
        print("  No door openings detected.")

    # ============================================================
    # Step 3: Visualization
    # ============================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Find the scan height closest to "1.0m above floor" (mid-door)
    target_rel_h = 1.0
    best_scan_idx = 0
    best_h_diff = 1e9
    for i, s in enumerate(scans):
        diff = abs((s.height - floor_z) - target_rel_h)
        if diff < best_h_diff:
            best_h_diff = diff
            best_scan_idx = i

    door_scan = scans[best_scan_idx]
    print(f"\nBest scan for door visualization: H{best_scan_idx} "
          f"(floor+{door_scan.height - floor_z:.2f}m)")

    # --- Figure: 3 panels ---
    fig = plt.figure(figsize=(20, 7))

    # Panel 1: Polar radar at door height
    ax_polar = fig.add_subplot(1, 3, 1, projection="polar")
    mask = door_scan.distances <= 15.0
    angles_rad = np.radians(door_scan.angles_deg[mask])
    dists = door_scan.distances[mask]
    sem = door_scan.semantic_labels
    assert sem is not None, "Scanner must produce semantic labels"
    labels = sem[mask]
    colors = np.array([
        SEMANTIC_PALETTE[l] if 0 <= l < len(SEMANTIC_PALETTE) else (0.5, 0.5, 0.5)
        for l in labels
    ])
    # Make door points larger and more visible
    is_door = labels == 3
    is_wall = labels == 0
    other = ~(is_door | is_wall)

    ax_polar.scatter(angles_rad[other], dists[other], s=0.3, c=colors[other], alpha=0.3)
    ax_polar.scatter(angles_rad[is_wall], dists[is_wall], s=0.5, c=colors[is_wall], alpha=0.5)
    ax_polar.scatter(angles_rad[is_door], dists[is_door], s=8.0, c="red", alpha=0.9,
                     zorder=5, edgecolors="darkred", linewidths=0.3)
    rel_h = door_scan.height - floor_z
    ax_polar.set_title(f"Polar Radar (floor+{rel_h:.2f}m)\nRed = door points", pad=15)
    ax_polar.grid(True, alpha=0.3)

    # Panel 2: Top-down with wall lines + door openings
    ax_td = fig.add_subplot(1, 3, 2)
    # Plot all scan points at door height (faint)
    pts = door_scan.points_2d[mask]
    door_pts = pts[is_door]
    wall_pts_scan = pts[is_wall]
    if len(wall_pts_scan) > 0:
        ax_td.scatter(wall_pts_scan[:, 0], wall_pts_scan[:, 1], s=0.3, c="gray", alpha=0.3)
    if len(door_pts) > 0:
        ax_td.scatter(door_pts[:, 0], door_pts[:, 1], s=5.0, c="red", alpha=0.8, zorder=5)

    # Plot wall lines
    for wi, w in enumerate(walls):
        ax_td.plot([w["start"][0], w["end"][0]],
                   [w["start"][1], w["end"][1]],
                   "b-", linewidth=2, alpha=0.7)
        mid = (w["start"] + w["end"]) / 2
        ax_td.annotate(f"W{wi}", (mid[0], mid[1]), fontsize=8, ha="center",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))

    # Plot door openings as thick red segments on walls
    for d in all_doors:
        wi = d["wall_idx"]
        w = walls[wi]
        p0 = w["start"] + d["t_min"] * (w["end"] - w["start"])
        p1 = w["start"] + d["t_max"] * (w["end"] - w["start"])
        ax_td.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", linewidth=4, alpha=0.9)
        cx = (p0[0] + p1[0]) / 2
        cy = (p0[1] + p1[1]) / 2
        ax_td.annotate(f"door\n{d['width_m']:.1f}m", (cx, cy), fontsize=6,
                       ha="center", va="center",
                       color="white", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8))

    ax_td.plot(center_x, center_y, "k+", markersize=12, markeredgewidth=2)
    ax_td.set_aspect("equal")
    margin = 2.0
    all_x = np.concatenate([w["start"][0:1] for w in walls] + [w["end"][0:1] for w in walls])
    all_y = np.concatenate([w["start"][1:2] for w in walls] + [w["end"][1:2] for w in walls])
    ax_td.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax_td.set_ylim(all_y.min() - margin, all_y.max() + margin)
    h0, h1 = h_axes
    ax_td.set_xlabel(f"World {'XYZ'[h0]} (m)")
    ax_td.set_ylabel(f"World {'XYZ'[h1]} (m)")
    ax_td.set_title(f"Top-Down: Walls + Doors\n(floor+{rel_h:.2f}m scan)")
    ax_td.grid(True, alpha=0.3)

    # Panel 3: Door height profile (which heights have door points per wall)
    ax_hp = fig.add_subplot(1, 3, 3)
    bar_width = 0.8 / max(len(walls), 1)
    scan_rel_heights = [s.height - floor_z for s in scans]
    for wi in range(len(walls)):
        counts = []
        for s in scans:
            # Count door points near this wall at this height
            door_mask_h = s.semantic_labels == 3
            if door_mask_h.sum() == 0:
                counts.append(0)
                continue
            door_pts_h = s.points_2d[door_mask_h]
            count = 0
            for pt in door_pts_h:
                _, dist = project_point_to_wall_segment(pt, walls[wi]["start"], walls[wi]["end"])
                if dist < 0.5:
                    count += 1
            counts.append(count)
        offset = (wi - len(walls) / 2 + 0.5) * bar_width
        ax_hp.bar(np.array(scan_rel_heights) + offset, counts, bar_width * 0.9,
                  label=f"Wall {wi}", alpha=0.8)

    ax_hp.set_xlabel("Height above floor (m)")
    ax_hp.set_ylabel("Door point count")
    ax_hp.set_title("Door Point Distribution by Height")
    ax_hp.legend(fontsize=7)
    ax_hp.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = str(out_dir / "doors_radar.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDoor visualization saved to: {png_path}")

    # Save door data JSON
    door_json = {
        "scene": args.name,
        "num_doors": len(all_doors),
        "scan_heights_above_floor": [round(h, 3) for h in scan_rel_heights],
        "doors": [
            {
                "wall_idx": d["wall_idx"],
                "position_along_wall_m": round(d["position_m"], 3),
                "width_m": round(d["width_m"], 3),
                "height_above_floor_min": round(d["height_min"], 3),
                "height_above_floor_max": round(d["height_max"], 3),
                "world_x": round(d["world_x"], 3),
                "world_y": round(d["world_y"], 3),
                "num_points": d["num_points"],
            }
            for d in all_doors
        ],
    }
    json_path = str(out_dir / "doors.json")
    Path(json_path).write_text(json.dumps(door_json, indent=2), encoding="utf-8")
    print(f"Door data saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
