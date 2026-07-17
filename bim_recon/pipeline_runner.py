"""BIM pipeline runner — callable generator that yields real-time progress.

Both the CLI (``scripts/run_pipeline.py``) and the Gradio UI call
``run_pipeline()`` directly.  No subprocess, no stdout parsing.

Usage (CLI)::

    from bim_recon.pipeline_runner import PipelineConfig, run_pipeline
    config = PipelineConfig(name="splat", elements=["door", "window"], ...)
    for msg, data in run_pipeline(config):
        print(msg)

Usage (Gradio)::

    for msg, data in run_pipeline(config):
        yield (msg, ...)  # update UI components
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np

from bim_recon.element_config import get_element_config
from bim_recon.element_merger import merge_detections
from bim_recon.falcon_client import FalconClient
from bim_recon.gs_scene import GSScene
from bim_recon.ring_scanner import render_ring_views, segment_ring_views, render_element_view
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.vlm_verifier import query_vlm
from bim_recon.wall_line_extractor import (
    extract_wall_lines, multi_height_scan, save_wall_lines_plot, wall_lines_to_json,
)

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """All parameters for one pipeline run."""
    name: str
    elements: list[str] = field(default_factory=lambda: ["door", "window"])
    skip_vlm: bool = False
    vlm_api_base: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""
    falcon_host: str = "127.0.0.1"
    falcon_port: int = 8390
    num_heights: int = 8
    snap_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_scene_files(data_dir: Path) -> tuple[Path, Path | None]:
    """Auto-discover PLY + feat.pt."""
    original = sorted(data_dir.glob("point_cloud_*.ply")) or sorted(data_dir.glob("*.ply"))
    if not original:
        raise FileNotFoundError(f"No PLY files in {data_dir}")
    ply = original[0]
    feat_candidates = sorted(data_dir.glob("*_feat.pt"))
    if not feat_candidates:
        out_dir = data_dir.parent.parent / "output" / data_dir.name
        feat_candidates = sorted(out_dir.glob("*_feat.pt"))
    feat = feat_candidates[0] if feat_candidates else None
    return ply, feat


def detect_coordinate_system(scene: GSScene, label_set: list[str] | None = None) -> dict:
    """Auto-detect up_axis, floor_z, ceiling_z, scan center."""
    has_sem = scene._has_feat and scene.semantic_querier is not None
    if has_sem:
        floor_c = np.array(scene.query_semantics("floor", mode="dominant", label_set=label_set)["centroid"])
        ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant", label_set=label_set)["centroid"])
        up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(floor_c[up_axis])
        ceiling_z = float(ceiling_c[up_axis])
        if ceiling_z - floor_z < 1.5:
            coords = scene.means[:, up_axis].cpu().numpy()
            floor_z = float(np.percentile(coords, 1))
            ceiling_z = float(np.percentile(coords, 99))
        center = (float(floor_c[h_axes[0]]), float(floor_c[h_axes[1]]))
    else:
        means = scene.means.cpu().numpy()
        ranges = means.max(axis=0) - means.min(axis=0)
        up_axis = int(np.argmax(ranges))
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(np.percentile(means[:, up_axis], 1))
        ceiling_z = float(np.percentile(means[:, up_axis], 99))
        center = (float(np.median(means[:, h_axes[0]])), float(np.median(means[:, h_axes[1]])))
    return {"up_axis": up_axis, "h_axes": h_axes, "floor_z": floor_z,
            "ceiling_z": ceiling_z, "center": center}


def extract_walls(scans, center, out_dir, labels=None):
    """Extract wall lines from multi-height scan data."""
    wall_lines, wall_pts = extract_wall_lines(scans, labels=labels, center=center)
    output_json = wall_lines_to_json(wall_lines, scans, center)
    (out_dir / "wall_lines.json").write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    save_wall_lines_plot(wall_lines, wall_pts, center,
                         str(out_dir / "wall_lines_topdown.png"),
                         title=f"Walls ({len(wall_lines)} segments)")
    return [{"x1": wl.x1, "y1": wl.y1, "x2": wl.x2, "y2": wl.y2, "length": wl.length}
            for wl in wall_lines]


def snap_wall_endpoints(walls, threshold=0.5):
    """Snap nearby wall endpoints to ensure closed polygon."""
    eps = [[w["x1"], w["y1"], w["x2"], w["y2"], w["length"]] for w in walls]
    changed = True
    for _ in range(10):
        if not changed:
            break
        changed = False
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
                if 1e-6 < dist < threshold:
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
    return [{"x1": ep[0], "y1": ep[1], "x2": ep[2], "y2": ep[3], "length": ep[4]}
            for ep in eps]


# ---------------------------------------------------------------------------
# Main pipeline generator
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> Generator[tuple[str, dict], None, None]:
    """Run the full BIM pipeline. Yields (message, data) for progress.

    Stages:
        1. Load scene
        2. Detect coordinate system
        3. 3D spherical scan → floor/ceiling refinement
        4. Multi-height horizontal scan
        5. Wall extraction + snap
        6. Ring scan (8 views × 60° wide-angle)
        7. Falcon segmentation per view
        8. Polar merge (wall-aware)
        9. Width recalculation from combined mask point clouds
        10. VLM verification per merged element
        11. Generate radar plots
        12. Save output JSON
    """
    data_dir = ROOT / "data" / config.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / config.name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Falcon client (mandatory) ---
    yield ("Connecting to Falcon server...", {})
    falcon = FalconClient(host=config.falcon_host, port=config.falcon_port)
    if not falcon.health():
        yield (f"ERROR: Falcon server unreachable at {config.falcon_host}:{config.falcon_port}", {})
        return

    # --- Stage 1: Load scene ---
    ply_path, feat_path = find_scene_files(data_dir)
    yield (f"Loading scene: {ply_path.name}...", {})
    text_emb = str(ROOT / "data" / "bim_text_emb.pt")
    class_names = str(ROOT / "data" / "bim_class_names.json")
    warm = {}
    if Path(text_emb).exists() and Path(class_names).exists():
        warm = {"text_emb_path": text_emb, "class_names_path": class_names}
    scene = GSScene.from_ply(ply_path, feat_path=feat_path, **warm)
    yield (f"Loaded {scene.num_gaussians} Gaussians", {})

    # Build label set
    structural_labels = ["wall", "floor", "ceiling"]
    element_labels = []
    for et in config.elements:
        try:
            element_labels.append(get_element_config(et).semantic_label)
        except KeyError:
            pass
    labels = list(dict.fromkeys(structural_labels + element_labels))

    # --- Stage 2: Detect coordinate system ---
    yield ("Detecting coordinate system...", {})
    coords = detect_coordinate_system(scene, label_set=labels)
    up_axis = coords["up_axis"]
    floor_z, ceiling_z = coords["floor_z"], coords["ceiling_z"]
    center = coords["center"]
    yield (f"up_axis={up_axis} floor={floor_z:.2f} ceiling={ceiling_z:.2f}", {})

    scanner = VirtualScanner(scene, up_axis=up_axis, labels=labels)

    # --- Stage 3: 3D spherical scan ---
    yield ("3D spherical scan...", {})
    scan_3d = scanner.scan_3d(center, floor_z, ceiling_z,
                              n_azimuth_views=12, n_elevation_bands=5,
                              width=512, fov=45.0)
    floor_ref, ceil_ref = VirtualScanner.detect_floor_ceiling(scan_3d, labels=labels)
    if abs(floor_ref - floor_z) < 0.5 and abs(ceil_ref - ceiling_z) < 0.5:
        floor_z, ceiling_z = floor_ref, ceil_ref
    yield (f"3D scan: {len(scan_3d.points_3d)} points, floor={floor_z:.2f} ceil={ceiling_z:.2f}", {})

    # --- Stage 4: Multi-height horizontal scan ---
    yield (f"Horizontal scan ({config.num_heights} heights)...", {})
    scans = multi_height_scan(scanner, center, floor_z, ceiling_z,
                              num_heights=config.num_heights, num_views=8, width=512)
    total_pts = sum(len(s.angles_deg) for s in scans)
    yield (f"Scan complete: {total_pts} points", {})

    # --- Stage 5: Wall extraction ---
    yield ("Extracting walls...", {})
    walls = extract_walls(scans, np.array(center), out_dir, labels=labels)
    walls_snapped = snap_wall_endpoints(walls, config.snap_threshold)
    (out_dir / "wall_lines_snapped.json").write_text(
        json.dumps(walls_snapped, indent=2), encoding="utf-8")
    yield (f"Extracted {len(walls_snapped)} wall segments", {})

    # --- Stage 6: Ring scan ---
    mid_z = (floor_z + ceiling_z) / 2.0
    n_ring, ring_fov = 8, 60.0
    yield (f"Ring scan: {n_ring} views × {ring_fov}°...", {})
    ring_views = render_ring_views(scene, center, mid_z, up_axis=up_axis,
                                   n_views=n_ring, fov=ring_fov, img_size=768)
    # Save ring views
    ring_dir = out_dir / "ring_views"
    ring_dir.mkdir(exist_ok=True)
    from PIL import Image as _PIL
    for v in ring_views:
        _PIL.fromarray(v.image).save(str(ring_dir / f"view_{v.idx:02d}_{v.azimuth_deg:.0f}.png"))
    yield (f"Rendered {len(ring_views)} views", {})

    # --- Stage 7: Falcon segmentation ---
    yield ("Falcon segmentation per view...", {})
    view_dets = segment_ring_views(ring_views, falcon, element_labels,
                                   center_2d=center, floor_z=floor_z,
                                   ceiling_z=ceiling_z, up_axis=up_axis)
    raw_json = [
        {"label": d.label, "view": d.view_idx, "azimuth": round(d.azimuth_deg, 1),
         "world_x": round(d.world_x, 3), "world_y": round(d.world_y, 3),
         "width_m": round(d.width_m, 3)}
        for d in view_dets
    ]
    (out_dir / "ring_raw_detections.json").write_text(
        json.dumps(raw_json, indent=2, ensure_ascii=False), encoding="utf-8")
    yield (f"Detected {len(view_dets)} raw objects across views", {})

    # --- Stage 8: Polar merge (wall-aware) ---
    yield ("Merging detections (wall-aware)...", {})
    merge_input = [
        {"element_class": d.label, "world_x": d.world_x, "world_y": d.world_y,
         "sill_height": d.sill_height, "header_height": d.header_height,
         "width_m": d.width_m, "confidence": d.centrality}
        for d in view_dets
    ]
    merged_elements = merge_detections(merge_input, center, up_axis=up_axis,
                                       merge_threshold=1.5, height_tolerance=0.5,
                                       walls=walls_snapped)
    yield (f"Merged: {len(view_dets)} raw → {len(merged_elements)} unique", {})

    # --- Stage 9: Width recalculation from mask point clouds ---
    for me in merged_elements:
        all_pts_xy = []
        for si in me.source_indices:
            if si < len(view_dets) and view_dets[si].mask_points_xy is not None:
                all_pts_xy.append(view_dets[si].mask_points_xy)
        if len(all_pts_xy) < 2:
            continue
        combined = np.vstack(all_pts_xy)
        if me.wall_idx is not None and me.wall_idx < len(walls_snapped):
            wl = walls_snapped[me.wall_idx]
            ws = np.array([wl["x1"], wl["y1"]])
            we = np.array([wl["x2"], wl["y2"]])
            wall_dir = we - ws
            wall_len = np.linalg.norm(wall_dir)
            if wall_len > 1e-6:
                wall_dir /= wall_len
                proj = (combined - ws) @ wall_dir
                true_w = float(np.percentile(proj, 97) - np.percentile(proj, 3))
                if true_w > me.width_m:
                    me.width_m = true_w
        me.world_x = float(np.median(combined[:, 0]))
        me.world_y = float(np.median(combined[:, 1]))

    # --- Stage 10: VLM verification ---
    verify_dir = out_dir / "verify_merged"
    verify_dir.mkdir(exist_ok=True)
    confirmed: list[dict] = []

    for mi, me in enumerate(merged_elements):
        yield (f"VLM {mi+1}/{len(merged_elements)}: {me.element_class} θ={me.theta_center:.1f}° rendering...", {})
        ev = render_element_view(scene, me.world_x, me.world_y,
                                 width_m=me.width_m, height_m=max(me.element_height, 0.5),
                                 mid_z=mid_z, center_2d=center, up_axis=up_axis,
                                 img_size=768, margin=0.5)
        if ev is None:
            yield (f"  render failed, skipping", {})
            continue

        img_name = f"merged_{mi}_{me.element_class}.png"
        from PIL import Image as _PIL
        _PIL.fromarray(ev.image).save(str(verify_dir / img_name))

        if config.skip_vlm:
            vlm_ok, vlm_resp = True, "skipped"
        else:
            try:
                cfg = get_element_config(me.element_class)
                hint = cfg.vlm_hint
            except KeyError:
                hint = me.element_class
            prompt = (f"Look at this image carefully. Is there {hint} in this image? "
                      f"Answer with YES or NO only.")
            yield (f"  querying VLM (timeout=30s)...", {})
            try:
                vlm_resp = query_vlm(str(verify_dir / img_name), prompt,
                                     config.vlm_api_base, config.vlm_model,
                                     config.vlm_api_key, timeout=30)
            except Exception as ex:
                yield (f"  VLM error: {ex}", {})
                vlm_resp = ""
            resp_lower = vlm_resp.lower().strip()
            vlm_ok = any(kw in resp_lower for kw in
                         ("yes", "是", "有", "确认", "confir", "correct",
                          "true", "indeed", "确实", "存在"))
            if not vlm_ok and me.element_class in resp_lower:
                vlm_ok = True

        tag = "CONFIRMED" if vlm_ok else "REJECTED"
        yield (f"  → {tag}", {})

        # Always record (confirmed + rejected) so images show in gallery
        confirmed.append({**me.to_dict(), "image_path": img_name,
                          "vlm_response": vlm_resp, "fov_deg": ev.fov_deg,
                          "vlm_confirmed": vlm_ok})

    # --- Build per-element results ---
    all_results: dict[str, dict] = {}
    for elem_type in config.elements:
        try:
            cfg = get_element_config(elem_type)
        except KeyError:
            continue
        type_entries = [c for c in confirmed if c["element_class"] == cfg.semantic_label]
        type_confirmed = [c for c in type_entries if c.get("vlm_confirmed")]
        all_results[elem_type] = {
            "total_candidates": len(view_dets),
            "after_prefilter": len(merged_elements),
            "confirmed": len(type_confirmed),
            "results": [{
                "confirmed": c.get("vlm_confirmed", False),
                "candidate": {"world_x": c["world_x"], "world_y": c["world_y"],
                              "theta_center": c["theta_center"], "r_mean": c["r_mean"],
                              "width_m": c["width_m"]},
                "height_detection": {"sill_height": c["sill_height"],
                                     "header_height": c["header_height"],
                                     "element_height": c["element_height"],
                                     "width_m": c["width_m"]},
                "image_path": c["image_path"],
                "vlm_response": c.get("vlm_response", ""),
            } for c in type_entries],
        }
        elem_json = {"scene": config.name, "element": elem_type,
                     "ply_used": ply_path.name,
                     "vlm_model": config.vlm_model if not config.skip_vlm else None,
                     **all_results[elem_type]}
        (out_dir / cfg.output_json_name).write_text(
            json.dumps(elem_json, indent=2), encoding="utf-8")

    # --- Translate all coordinates to center room at origin (floor at z=0) ---
    yield ("Centering room at origin...", {})
    cx, cy = float(center[0]), float(center[1])
    for wl in walls_snapped:
        wl["x1"] -= cx; wl["y1"] -= cy
        wl["x2"] -= cx; wl["y2"] -= cy
    for c in confirmed:
        c["world_x"] -= cx; c["world_y"] -= cy
    for me in merged_elements:
        me.world_x -= cx; me.world_y -= cy
    # Shift Z so floor = 0
    _floor_offset = floor_z
    ceiling_z -= _floor_offset
    floor_z = 0.0
    for c in confirmed:
        if "sill_height" in c:
            pass  # sill/header are already relative to floor
    # Re-save wall lines and per-element JSON with centered coordinates
    (out_dir / "wall_lines_snapped.json").write_text(
        json.dumps(walls_snapped, indent=2), encoding="utf-8")
    for elem_type in config.elements:
        try:
            cfg = get_element_config(elem_type)
        except KeyError:
            continue
        type_entries = [c for c in confirmed if c["element_class"] == cfg.semantic_label]
        type_confirmed = [c for c in type_entries if c.get("vlm_confirmed")]
        all_results[elem_type] = {
            "total_candidates": len(view_dets),
            "after_prefilter": len(merged_elements),
            "confirmed": len(type_confirmed),
            "results": [{
                "confirmed": c.get("vlm_confirmed", False),
                "candidate": {"world_x": c["world_x"], "world_y": c["world_y"],
                              "theta_center": c["theta_center"], "r_mean": c["r_mean"],
                              "width_m": c["width_m"]},
                "height_detection": {"sill_height": c["sill_height"],
                                     "header_height": c["header_height"],
                                     "element_height": c["element_height"],
                                     "width_m": c["width_m"]},
                "image_path": c["image_path"],
                "vlm_response": c.get("vlm_response", ""),
            } for c in type_entries],
        }
        elem_json = {"scene": config.name, "element": elem_type,
                     "ply_used": ply_path.name,
                     "vlm_model": config.vlm_model if not config.skip_vlm else None,
                     **all_results[elem_type]}
        (out_dir / cfg.output_json_name).write_text(
            json.dumps(elem_json, indent=2), encoding="utf-8")
    center = (0.0, 0.0)


    # --- Save merged elements ---
    merged_json = {"raw_count": len(view_dets), "merged_count": len(merged_elements),
                   "confirmed_count": sum(1 for c in confirmed if c.get("vlm_confirmed")),
                   "total_vlm_results": len(confirmed),
                   "merged": [me.to_dict() for me in merged_elements],
                   "confirmed": confirmed}
    (out_dir / "merged_elements.json").write_text(
        json.dumps(merged_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Generate radar plots ---
    yield ("Generating radar plots...", {})
    _generate_radars(out_dir, view_dets, merged_elements, walls_snapped,
                     center, ring_views, ring_fov)

    # --- Pipeline report ---
    report = {
        "scene": config.name, "ply": ply_path.name,
        "num_gaussians": scene.num_gaussians,
        "coordinate_system": {"up_axis": up_axis, "floor_z": floor_z,
                              "ceiling_z": ceiling_z, "center": list(center)},
        "scan": {"num_heights": config.num_heights, "total_points": total_pts,
                 "scan_3d_points": len(scan_3d.points_3d)},
        "walls": {"count": len(walls_snapped)},
        "elements": all_results,
        "merged_elements": {"count": len(merged_elements),
                            "confirmed": len(confirmed)},
        "vlm_model": config.vlm_model if not config.skip_vlm else None,
    }
    (out_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    confirmed_total = sum(r.get("confirmed", 0) for r in all_results.values())
    yield ("Pipeline complete!", {
        "out_dir": str(out_dir),
        "walls": walls_snapped,
        "all_results": all_results,
        "merged_elements": merged_elements,
        "confirmed_count": confirmed_total,
        "report": report,
    })


# ---------------------------------------------------------------------------
# Radar plot generation
# ---------------------------------------------------------------------------

def _generate_radars(out_dir, view_dets, merged_elements, walls_snapped,
                     center, ring_views, ring_fov):
    """Generate Cartesian top-down radar plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    cx, cy = float(center[0]), float(center[1])

    def _draw_walls(ax):
        for wl in walls_snapped:
            x1, y1 = wl["x1"] - cx, wl["y1"] - cy
            x2, y2 = wl["x2"] - cx, wl["y2"] - cy
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)

    def _draw_pca_line(ax, pts, color, label=None):
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=3, alpha=0.4, zorder=7)
        if len(pts) > 3:
            centered = pts - pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[0]
            c = pts.mean(axis=0)
            p1 = c + Vt[0] * proj.min()
            p2 = c + Vt[0] * proj.max()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color,
                    linewidth=2.5, zorder=8, label=label)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color=color, markersize=5, zorder=9)

    max_r = max(
        max((abs(wl["x1"] - cx) for wl in walls_snapped), default=5),
        max((abs(wl["x2"] - cx) for wl in walls_snapped), default=5), 5)

    # Radar 1: Raw detections
    if view_dets:
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        _draw_walls(ax)
        lc = {"door": "red", "window": "blue", "column": "gray"}
        for di, d in enumerate(view_dets):
            color = lc.get(d.label, "green")
            if d.mask_points_xy is not None:
                _draw_pca_line(ax, d.mask_points_xy - np.array([cx, cy]), color,
                               label=f"{d.label} v{d.view_idx}" if di < 12 else None)
            else:
                ax.scatter(d.world_x - cx, d.world_y - cy, c=color, s=30, zorder=7)
        ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
        ax.set_aspect("equal")
        ax.set_xlim(-max_r - 1, max_r + 1)
        ax.set_ylim(-max_r - 1, max_r + 1)
        ax.set_title(f"Ring Raw Detections ({len(view_dets)} masks)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.savefig(str(out_dir / "radar_ring_raw.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Radar 2: Merged elements
    if merged_elements:
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        _draw_walls(ax)
        for v in ring_views:
            az = math.radians(v.azimuth_deg)
            hfov = math.radians(ring_fov / 2)
            max_d = float(np.percentile(v.depth[v.depth > 0.1], 90)) if (v.depth > 0.1).any() else 5.0
            ax.add_patch(Wedge((0, 0), max_d, math.degrees(az - hfov),
                               math.degrees(az + hfov), alpha=0.04,
                               color="lightblue", zorder=1))
        palette = plt.cm.Set1(np.linspace(0, 1, max(len(merged_elements), 1)))
        for mi, me in enumerate(merged_elements):
            color = palette[mi % len(palette)]
            src_pts = []
            for si in me.source_indices:
                if si < len(view_dets) and view_dets[si].mask_points_xy is not None:
                    src_pts.append(view_dets[si].mask_points_xy - np.array([cx, cy]))
            if src_pts:
                _draw_pca_line(ax, np.vstack(src_pts), color,
                               label=f"{me.element_class} ({me.num_sources} masks)")
            else:
                ax.scatter(me.world_x - cx, me.world_y - cy, c=[color],
                           s=100, marker="*", zorder=8)
        ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
        ax.set_aspect("equal")
        ax.set_xlim(-max_r - 1, max_r + 1)
        ax.set_ylim(-max_r - 1, max_r + 1)
        ax.set_title(f"Merged Elements ({len(merged_elements)} unique)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(str(out_dir / "radar_merged.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
