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
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.candidate_extractor import (
    extract_candidates,
    prefilter_candidates,
    resolve_class_index,
    CLASSIC_BIM_VOCAB,
)
from bim_recon.config import load_config
from bim_recon.element_config import ElementConfig, get_element_config, list_element_types
from bim_recon.falcon_client import FalconClient
from bim_recon.gs_scene import GSScene
from bim_recon.height_detector import detect_element_heights
from bim_recon.spatial_extractor import extract_spatial
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest
from bim_recon.mesh_registrar import (
    MeshPlacement,
    compute_placement_transform,
    extract_object_from_render,
    register_mesh_in_revit,
)
from bim_recon.mesh_readiness import render_and_check_mesh_readiness
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.element_merger import merge_detections, merge_falcon_ring_detections
from bim_recon.vlm_verifier import verify_candidates
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
    if not original:
        original = sorted(data_dir.glob("*.ply"))
    feat_vis = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))
    ply = original[0] if original else feat_vis[0]
    # feat.pt may be in data_dir or in output/<name>/
    feat_candidates = sorted(data_dir.glob("*_feat.pt"))
    output_dir = data_dir.parent.parent / "output" / data_dir.name
    if not feat_candidates:
        feat_candidates = sorted(output_dir.glob("*_feat.pt"))
    if not feat_candidates:
        raise FileNotFoundError(
            f"No *_feat.pt found in {data_dir} or {output_dir}"
        )
    feat = feat_candidates[0]
    return ply, feat
def detect_coordinate_system(scene: GSScene, label_set: list[str] | None = None) -> dict:
    """Auto-detect up_axis, floor_z, ceiling_z, scan center.

    *label_set* is forwarded to the floor/ceiling dominant queries so the
    argmax runs over the active open-vocabulary label set.

    If the semantic floor/ceiling detection produces an implausible room
    height (< 1.5 m), falls back to the Gaussian position distribution
    (1st/99th percentile along the detected up-axis).
    """
    floor_c = np.array(scene.query_semantics("floor", mode="dominant", label_set=label_set)["centroid"])
    ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant", label_set=label_set)["centroid"])
    up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
    h_axes = [i for i in range(3) if i != up_axis]
    floor_z = float(floor_c[up_axis])
    ceiling_z = float(ceiling_c[up_axis])

    # Validate detected height; fall back to Gaussian distribution if implausible.
    if ceiling_z - floor_z < 1.5:
        coords = scene.means[:, up_axis].cpu().numpy()
        floor_z = float(np.percentile(coords, 1))
        ceiling_z = float(np.percentile(coords, 99))
        print(f"  ⚠ Semantic height detection implausible "
              f"(<1.5m); geometric fallback: floor={floor_z:.3f}, "
              f"ceiling={ceiling_z:.3f}, height={ceiling_z-floor_z:.3f}m")

    return {
        "up_axis": up_axis,
        "h_axes": h_axes,
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "center": (float(floor_c[h_axes[0]]), float(floor_c[h_axes[1]])),
    }


# ---------------------------------------------------------------------------
# Wall extraction
# ---------------------------------------------------------------------------

def extract_walls(
    scans: list,
    center: np.ndarray,
    out_dir: Path,
    labels: list[str] | None = None,
) -> list[dict]:
    """Extract wall lines from multi-height scan data."""
    wall_lines, wall_pts = extract_wall_lines(
        scans,
        labels=labels,
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
# Falcon ring scan: direct 2D detection + depth backprojection
# ---------------------------------------------------------------------------


def falcon_wall_scan(
    scene: GSScene,
    walls: list[dict],
    center: tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    falcon: "FalconClient",
    out_dir: Path,
    query: str = "door, window",
    up_axis: int = 2,
    img_size: int = 768,
) -> list[dict]:
    """Per-wall Falcon detection: one panoramic view per wall.

    For each wall, renders from room center toward the wall midpoint with
    FOV wide enough to cover the entire wall.  Eliminates duplicate
    detections (same object seen from multiple 360° views).  Each
    detection carries its wall_idx and a cropped+annotated image.
    """
    import math
    import base64 as _b64
    from PIL import Image as _PIL, ImageDraw as _ID
    from pycocotools import mask as mask_utils
    from bim_recon.gs_scene import GSScene as _GS, look_at_pose

    mid_z = (floor_z + ceiling_z) / 2.0
    h_axes = [i for i in range(3) if i != up_axis]
    cx, cy = float(center[0]), float(center[1])
    labels = [l.strip() for l in query.split(",") if l.strip()]
    found: list[dict] = []

    for wi, wall in enumerate(walls):
        ws = np.array([wall["x1"], wall["y1"]], dtype=np.float64)
        we = np.array([wall["x2"], wall["y2"]], dtype=np.float64)
        wlen = float(np.linalg.norm(we - ws))
        wmid = (ws + we) / 2.0
        wall_dist = float(np.linalg.norm(np.array([cx, cy]) - wmid))

        # FOV to cover the entire wall + 0.3m margin
        fov_deg = min(
            math.degrees(2 * math.atan((wlen / 2 + 0.3) / max(wall_dist, 0.5))),
            110.0,
        )
        fx = 0.5 * img_size / math.tan(0.5 * math.radians(fov_deg))

        eye = [0.0, 0.0, 0.0]
        eye[h_axes[0]] = cx
        eye[h_axes[1]] = cy
        eye[up_axis] = mid_z
        tgt = [0.0, 0.0, 0.0]
        tgt[h_axes[0]] = float(wmid[0])
        tgt[h_axes[1]] = float(wmid[1])
        tgt[up_axis] = mid_z
        up = [0.0, 0.0, 0.0]
        up[up_axis] = 1.0
        pose = look_at_pose(tuple(eye), tuple(tgt), tuple(up))

        result, reason, _ = _GS.render_validated(
            scene, pose, img_size, img_size, fov_deg)
        if result is None:
            print(f"    [wall_scan] wall {wi}: invalid ({reason}), skip")
            continue

        img = _PIL.fromarray((result.colors * 255).clip(0, 255).astype(np.uint8))
        img.save(str(out_dir / f"falcon_wall_{wi}.png"))

        viewmat = pose.to_viewmat()
        R_c2w = viewmat[:3, :3].T
        eye_np = np.array(eye, dtype=np.float64)
        annotated = img.convert("RGBA")
        wall_dets: list[dict] = []

        for label in labels:
            try:
                dets = falcon.segment(img, label, task="segmentation")
            except Exception as ex:
                print(f"    [wall_scan] wall {wi} '{label}': {ex}")
                continue
            for det in dets:
                mb = det.mask_bbox or det.bbox
                # Decode mask
                mask_arr = None
                if det.mask_rle and det.mask_size:
                    try:
                        counts = _b64.b64decode(det.mask_rle)
                        mask_arr = mask_utils.decode(
                            {"counts": counts, "size": det.mask_size})
                        mh, mw = det.mask_size
                        if mh != img_size or mw != img_size:
                            mask_arr = np.array(_PIL.fromarray(mask_arr).resize(
                                (img_size, img_size), _PIL.NEAREST))
                    except Exception:
                        mask_arr = None

                # Sampling pixels
                if mask_arr is not None:
                    ys, xs = np.where(mask_arr)
                    cu, cv = int(np.median(xs)), int(np.median(ys))
                else:
                    cu = max(0, min(img_size - 1, int(mb["x"] * img_size)))
                    cv = max(0, min(img_size - 1, int(mb["y"] * img_size)))

                d_center = float(result.depth[cv, cu])
                if d_center < 0.1:
                    continue

                # Full-mask backprojection
                if mask_arr is not None:
                    mys, mxs = np.where(mask_arr)
                    mds = result.depth[mys, mxs].astype(np.float64)
                    ok = mds > 0.1
                    if ok.sum() < 10:
                        continue
                    mxs, mys, mds = mxs[ok], mys[ok], mds[ok]
                    x_cam = (mxs - img_size / 2.0) / fx * mds
                    y_cam = (mys - img_size / 2.0) / fx * mds
                    P_w = R_c2w @ np.stack([x_cam, y_cam, mds], 0) + eye_np.reshape(3, 1)
                    wz = P_w[up_axis, :]
                    sill_h = float(np.percentile(wz, 5)) - floor_z
                    header_h = float(np.percentile(wz, 95)) - floor_z
                    wh0 = P_w[h_axes[0], :]
                    width_m = float(np.percentile(wh0, 95) - np.percentile(wh0, 5))
                    P_world = np.median(P_w, axis=1).tolist()
                    method = "mask_full_backproject"
                else:
                    x_c = (cu - img_size / 2.0) / fx * d_center
                    y_c = (cv - img_size / 2.0) / fx * d_center
                    P_world = (R_c2w @ np.array([x_c, y_c, d_center]) + eye_np).tolist()
                    sill_h = header_h = 0.0
                    width_m = float(mb["w"] * 2.0)
                    method = "bbox_center_depth"

                wall_dets.append({
                    "label": label, "bbox": det.bbox, "mask_bbox": mb,
                    "mask_rle": det.mask_rle, "mask_size": det.mask_size,
                    "world_pos": [round(c, 3) for c in P_world],
                    "wall_idx": wi, "depth": round(d_center, 3),
                    "sill_height": round(sill_h, 3),
                    "header_height": round(header_h, 3),
                    "element_height": round(max(0, header_h - sill_h), 3),
                    "width_m": round(width_m, 3), "method": method,
                })

        # Per-detection cropped+annotated images
        for di, vd in enumerate(wall_dets):
            ov = _PIL.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
            dr = _ID.Draw(ov)
            mb = vd["mask_bbox"]
            if vd.get("mask_rle"):
                try:
                    counts = _b64.b64decode(vd["mask_rle"])
                    marr = mask_utils.decode({"counts": counts, "size": vd["mask_size"]})
                    mh, mw = vd["mask_size"]
                    if mh != img_size or mw != img_size:
                        marr = np.array(_PIL.fromarray(marr).resize((img_size, img_size), _PIL.NEAREST))
                    colored = np.zeros((img_size, img_size, 4), dtype=np.uint8)
                    colored[marr > 0] = [255, 0, 0, 100]
                    ov = _PIL.fromarray(colored, "RGBA")
                    dr = _ID.Draw(ov)
                except Exception:
                    pass
            else:
                dr.rectangle([int((mb["x"]-mb["w"]/2)*img_size), int((mb["y"]-mb["h"]/2)*img_size),
                              int((mb["x"]+mb["w"]/2)*img_size), int((mb["y"]+mb["h"]/2)*img_size)],
                             outline=(255, 0, 0, 255), width=3)
            dr.text((int((mb["x"]-mb["w"]/2)*img_size)+4, int((mb["y"]-mb["h"]/2)*img_size)+4),
                    vd["label"], fill=(255, 0, 0, 255))
            cx_px, cy_px = int(mb["x"]*img_size), int(mb["y"]*img_size)
            hw, hh = int(mb["w"]*img_size*0.75), int(mb["h"]*img_size*0.75)
            crop = _PIL.alpha_composite(annotated, ov).crop(
                (max(0, cx_px-hw), max(0, cy_px-hh),
                 min(img_size, cx_px+hw), min(img_size, cy_px+hh))).convert("RGB")
            fname = f"falcon_det_w{wi}_{vd['label']}_{di}.png"
            crop.save(str(out_dir / fname))
            vd["image_path"] = fname

        found.extend(wall_dets)
        print(f"    [wall_scan] wall {wi} ({wlen:.1f}m, fov={fov_deg:.0f}°): "
              f"{len(wall_dets)} detections")

    return found


def _detect_from_falcon(
    falcon_dets: list[dict],
    walls: list[dict],
    cfg: ElementConfig,
    out_dir: Path,
    up_axis: int,
) -> dict:
    """Convert Falcon wall-scan detections to pipeline result format.

    Spatial measurements and per-detection images are already computed in
    falcon_wall_scan. This function just assigns each detection to its
    wall position (t-parameter) and builds the result dict.
    """
    from bim_recon.candidate_extractor import Candidate, project_point_to_wall

    h_axes = [j for j in range(3) if j != up_axis]
    result_dicts = []

    for i, det in enumerate(falcon_dets):
        wi = det["wall_idx"]
        wall = walls[wi]
        ws = np.array([wall["x1"], wall["y1"]])
        we = np.array([wall["x2"], wall["y2"]])
        world_h = np.array([det["world_pos"][h_axes[0]], det["world_pos"][h_axes[1]]])
        t_along, _ = project_point_to_wall(world_h, ws, we)

        sill = det.get("sill_height", 0.0)
        header = det.get("header_height", sill + 1.0)
        width_m = det.get("width_m", 0.8)

        cand = Candidate(
            element_class=cfg.name, class_idx=0, wall_idx=wi,
            t_min=max(0, t_along - 0.05), t_max=min(1, t_along + 0.05),
            theta_center=0.0, theta_span=10.0,
            r_mean=det["depth"], h_min=sill, h_max=header,
            width_m=width_m, num_points=100,
            world_x=float(world_h[0]), world_y=float(world_h[1]),
        )

        d = {
            "candidate": cand.to_dict(),
            "confirmed": True,
            "image_path": det.get("image_path", ""),
            "vlm_response": "falcon_wall_scan",
            "height_detection": {
                "sill_height": sill,
                "header_height": header,
                "element_height": det.get("element_height", max(0, header - sill)),
                "width_m": width_m,
                "t_min": t_along, "t_max": t_along,
                "confidence": 0.85,
                "method": det.get("method", "falcon_wall_scan"),
            },
        }
        print(f"    [{cfg.name}] #{i} (wall {wi}): sill={sill:.3f}m "
              f"header={header:.3f}m w={width_m:.3f}m")
        result_dicts.append(d)

    return {
        "element": cfg.name,
        "total_candidates": len(falcon_dets),
        "after_prefilter": len(falcon_dets),
        "confirmed": len(falcon_dets),
        "rejected": 0,
        "results": result_dicts,
    }

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
    vlm_api_base: str,
    vlm_model: str,
    vlm_api_key: str,
    skip_vlm: bool = False,
    falcon: FalconClient | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Detect elements of one type from scan data + VLM verification.

    If ``falcon`` is provided and reachable, uses Falcon-Perception
    segmentation for precise spatial extraction (sill/header/width).
    Falls back to depth-probing (:mod:`bim_recon.height_detector`)
    when Falcon is unavailable or returns no result.
    """
    center = coords["center"]
    floor_z = coords["floor_z"]
    up_axis = coords["up_axis"]

    # Resolve the element's open-vocabulary semantic label to an index in the
    # active label set (falls back to the classic 9-class vocab).
    active_labels = labels if labels is not None else list(CLASSIC_BIM_VOCAB)
    class_idx = resolve_class_index(cfg.semantic_label, active_labels)

    # Extract candidates
    candidates = extract_candidates(
        scans, walls, floor_z, center,
        element_class=cfg.name,
        class_idx=class_idx,
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
        vlm_api_base=vlm_api_base,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        up_axis=up_axis,
        vlm_hint=cfg.vlm_hint,
        skip_vlm=skip_vlm,
    )

    confirmed = [r for r in results if r.confirmed is True]
    rejected = [r for r in results if r.confirmed is False]
    print(f"  [{cfg.name}] {len(confirmed)} confirmed, {len(rejected)} rejected")

    # Spatial extraction for confirmed wall-mounted elements.
    # When Falcon is online: Falcon's verdict is authoritative — if it finds
    # nothing, the element is rejected (no depth-probe fallback).
    # Depth-probe fallback is used ONLY when the Falcon server is offline.
    height_results: list[dict | None] = [None] * len(results)
    falcon_rejected: set[int] = set()
    if cfg.height_detection and confirmed:
        ceiling_z = coords["ceiling_z"]
        falcon_tag = "Falcon" if falcon is not None else "depth-probe"
        print(f"  [{cfg.name}] spatial extraction ({falcon_tag})")
        for i, r in enumerate(results):
            if not r.confirmed:
                continue
            wi = r.candidate.wall_idx
            if wi is None or wi >= len(walls):
                continue

            spatial_dict: dict | None = None

            if falcon is not None:
                # --- Falcon online: authoritative segmentation ---
                elev_path = str(out_dir / f"{cfg.name}_{i}_elevation.png")
                try:
                    spatial = extract_spatial(
                        falcon, scene, r.candidate, walls[wi],
                        floor_z, ceiling_z, center,
                        element_name=cfg.name,
                        up_axis=up_axis,
                        save_image_path=elev_path,
                    )
                except (TimeoutError, OSError, Exception) as ex:
                    print(f"    [{cfg.name}] #{i}: Falcon 超时/错误 ({ex})，跳过")
                    continue
                if spatial is not None:
                    spatial_dict = {
                        "sill_height": spatial.sill_height,
                        "header_height": spatial.header_height,
                        "element_height": spatial.element_height,
                        "width_m": spatial.width_m,
                        "t_min": spatial.t_min,
                        "t_max": spatial.t_max,
                        "confidence": spatial.confidence,
                        "method": spatial.method,
                        "elevation_params": spatial.elevation_params,
                    }
                else:
                    # Falcon online but found nothing → element doesn't exist
                    falcon_rejected.add(i)
                    print(f"    [{cfg.name}] #{i}: Falcon 未检测到，拒绝该构件")
                    continue
            else:
                # --- Falcon offline: depth-probe fallback ---
                hr = detect_element_heights(
                    scene, r.candidate, walls[wi],
                    floor_z, ceiling_z, center,
                    class_idx=class_idx,
                    labels=labels,
                    up_axis=up_axis,
                )
                spatial_dict = {
                    "sill_height": hr.sill_height,
                    "header_height": hr.header_height,
                    "element_height": hr.element_height,
                    "confidence": hr.confidence,
                    "method": hr.method,
                }

            height_results[i] = spatial_dict
            sd = spatial_dict
            print(f"    [{cfg.name}] #{i}: sill={sd['sill_height']:.3f}m "
                  f"header={sd['header_height']:.3f}m "
                  f"h={sd['element_height']:.3f}m ({sd['method']})")

    result_dicts = []
    final_confirmed = 0
    for i, r in enumerate(results):
        d = r.to_dict()
        if i in falcon_rejected:
            d["confirmed"] = False
            d["reject_reason"] = "falcon_not_detected"
        elif height_results[i] is not None:
            d["height_detection"] = height_results[i]
            final_confirmed += 1
        elif r.confirmed:
            final_confirmed += 1
        result_dicts.append(d)

    if falcon_rejected:
        print(f"  [{cfg.name}] {len(falcon_rejected)} rejected by Falcon "
              f"(false positives removed)")

    return {
        "element": cfg.name,
        "total_candidates": len(candidates),
        "after_prefilter": len(filtered),
        "confirmed": final_confirmed,
        "rejected": len(rejected) + len(falcon_rejected),
        "results": result_dicts,
    }


# ---------------------------------------------------------------------------
# Per-element radar visualization
# ---------------------------------------------------------------------------

def _generate_element_radars(
    scans: list,
    walls: list[dict],
    all_results: dict[str, dict],
    center: tuple[float, float],
    floor_z: float,
    up_axis: int,
    out_dir: Path,
    label_set: list[str] | None = None,
) -> None:
    """Generate multi-panel radar PNG for each detected element type.

    Creates a 2-panel figure per element:
      1. Top-down scatter with semantic colors + wall lines + element positions
      2. Polar radar scan with semantic colors + element arcs

    Saves as ``radar_<element>.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bim_recon.virtual_scanner import label_palette

    h_axes = [i for i in range(3) if i != up_axis]
    h0, h1 = h_axes[0], h_axes[1]

    # Collect all scan points + labels
    all_pts = []
    all_labels = []
    all_dists = []
    all_angles = []
    for scan in scans:
        if scan.semantic_labels is None:
            continue
        all_pts.append(scan.points_2d)
        all_labels.append(scan.semantic_labels)
        all_dists.append(scan.distances)
        all_angles.append(scan.angles_deg)
    if not all_pts:
        return
    pts = np.concatenate(all_pts)
    labels = np.concatenate(all_labels)
    dists = np.concatenate(all_dists)
    angles = np.concatenate(all_angles)
    # Resolve the active open-vocabulary label set (classic 9-class fallback).
    active_vocab = list(label_set) if label_set is not None else list(CLASSIC_BIM_VOCAB)
    wall_idx = active_vocab.index("wall") if "wall" in active_vocab else 0
    palette = np.array(label_palette(max(len(active_vocab), 9)), dtype=np.float64)

    for elem_type, result in all_results.items():
        elem_class_idx = None
        try:
            cfg_elem = get_element_config(elem_type)
            elem_class_idx = resolve_class_index(cfg_elem.semantic_label, active_vocab)
        except KeyError:
            continue

        fig = plt.figure(figsize=(14, 6))

        # ==== Panel 1: Top-down scatter ====
        ax_td = fig.add_subplot(1, 2, 1)

        # Plot scan points: wall=gray, target=red, others=faint
        safe_idx = np.clip(labels, 0, len(palette) - 1)
        in_range = (labels >= 0) & (labels < len(palette))
        colors = np.where(in_range[:, None], palette[safe_idx], 0.5)

        wall_mask = labels == wall_idx
        target_mask = labels == elem_class_idx
        other_mask = ~wall_mask & ~target_mask

        if other_mask.sum() > 0:
            ax_td.scatter(pts[other_mask, 0], pts[other_mask, 1],
                          s=0.3, c=colors[other_mask], alpha=0.15)
        if wall_mask.sum() > 0:
            ax_td.scatter(pts[wall_mask, 0], pts[wall_mask, 1],
                          s=0.5, c="gray", alpha=0.3, label="wall pts")
        if target_mask.sum() > 0:
            ax_td.scatter(pts[target_mask, 0], pts[target_mask, 1],
                          s=2.0, c="red", alpha=0.6, label=f"{elem_type} pts")

        # Plot wall lines
        for w in walls:
            ax_td.plot([w["x1"], w["x2"]], [w["y1"], w["y2"]],
                       "b-", linewidth=2, alpha=0.7)

        # Plot detected elements
        for r in result.get("results", []):
            c = r.get("candidate", {})
            wx, wy = c.get("world_x", 0), c.get("world_y", 0)
            if r.get("confirmed"):
                ax_td.scatter([wx], [wy], s=80, c="lime", marker="o",
                              edgecolors="black", linewidths=0.5, zorder=5,
                              label="confirmed" if "confirmed" not in ax_td.get_legend_handles_labels()[1] else "")
            else:
                ax_td.scatter([wx], [wy], s=60, c="red", marker="x",
                              linewidths=1.5, zorder=5,
                              label="rejected" if "rejected" not in ax_td.get_legend_handles_labels()[1] else "")

        ax_td.set_aspect("equal")
        ax_td.set_title(f"Top-Down: {elem_type} ({result.get('confirmed', 0)} confirmed)")
        ax_td.legend(fontsize=8, loc="upper right")
        ax_td.grid(True, alpha=0.3)

        # ==== Panel 2: Polar radar ====
        ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

        # Filter to reasonable distance
        dist_mask = dists <= 15.0
        angles_rad = np.radians(angles[dist_mask])
        dists_filtered = dists[dist_mask]
        labels_filtered = labels[dist_mask]

        # Color by semantic class
        safe_idx_f = np.clip(labels_filtered, 0, len(palette) - 1)
        in_range_f = (labels_filtered >= 0) & (labels_filtered < len(palette))
        colors_f = np.where(in_range_f[:, None], palette[safe_idx_f], 0.5)

        # Plot non-target, non-wall points faintly
        other_mask_f = ~((labels_filtered == 0) | (labels_filtered == elem_class_idx))
        wall_mask_f = labels_filtered == 0
        target_mask_f = labels_filtered == elem_class_idx

        if other_mask_f.sum() > 0:
            ax_polar.scatter(angles_rad[other_mask_f], dists_filtered[other_mask_f],
                             s=0.3, c=colors_f[other_mask_f], alpha=0.2)
        if wall_mask_f.sum() > 0:
            ax_polar.scatter(angles_rad[wall_mask_f], dists_filtered[wall_mask_f],
                             s=0.5, c=colors_f[wall_mask_f], alpha=0.4)
        if target_mask_f.sum() > 0:
            ax_polar.scatter(angles_rad[target_mask_f], dists_filtered[target_mask_f],
                             s=3.0, c="red", alpha=0.8, zorder=5)

        # Mark confirmed elements with green arcs
        for r in result.get("results", []):
            theta_c = np.radians(r.get("theta", 0))
            r_dist = r.get("r", 0)
            theta_span_rad = np.radians(r.get("candidate", {}).get("theta_span", 5))
            if r.get("confirmed"):
                theta_range = np.linspace(
                    theta_c - theta_span_rad / 2,
                    theta_c + theta_span_rad / 2, 20,
                )
                ax_polar.plot(theta_range, [r_dist] * len(theta_range),
                              "g-", linewidth=4, alpha=0.8, zorder=10)
            else:
                ax_polar.scatter([theta_c], [r_dist], s=80, c="red",
                                 marker="x", zorder=10, linewidths=2)

        if len(dists_filtered) > 0:
            ax_polar.set_ylim(0, max(dists_filtered.max() + 1, 6))
        ax_polar.set_title(
            f"Polar Radar: {elem_type}\n(red={elem_type} pts, green=confirmed)",
            pad=20, fontsize=11,
        )
        ax_polar.grid(True, alpha=0.3)

        fig.tight_layout()
        radar_path = out_dir / f"radar_{elem_type}.png"
        fig.savefig(str(radar_path), dpi=100)
        plt.close(fig)
        print(f"  Radar saved: {radar_path.name}")


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
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM verification (render only)")
    parser.add_argument("--snap-threshold", type=float, default=0.5,
                        help="Wall endpoint snap threshold (m)")
    parser.add_argument("--vlm-model", default=None,
                        help="Override VLM model from config.json")
    parser.add_argument("--vlm-api-base", default=None,
                        help="Override VLM API base URL from config.json")
    parser.add_argument("--vlm-api-key", default=None,
                        help="Override VLM API key from config.json")
    parser.add_argument("--falcon-host", default="127.0.0.1",
                        help="Falcon inference server host")
    parser.add_argument("--falcon-port", type=int, default=8390,
                        help="Falcon inference server port")
    parser.add_argument("--no-falcon", action="store_true",
                        help="Disable Falcon spatial extraction (use depth-probing only)")
    parser.add_argument("--trellis-host", default=None,
                        help="Override TRELLIS server host from config.json")
    parser.add_argument("--trellis-port", type=int, default=None,
                        help="Override TRELLIS server port from config.json")
    parser.add_argument("--no-trellis", action="store_true",
                        help="Disable TRELLIS B-class mesh generation")
    args = parser.parse_args()

    # === Load VLM config from config.json (CLI args override) ===
    app_config = load_config()
    vlm = app_config.vlm
    vlm_api_base = args.vlm_api_base or vlm.api_base
    vlm_model = args.vlm_model or vlm.model
    vlm_api_key = args.vlm_api_key if args.vlm_api_key is not None else vlm.api_key
    if not args.skip_vlm:
        print(f"  VLM: {vlm_model} @ {vlm_api_base}")

    # === Falcon client (optional) ===
    falcon: FalconClient | None = None
    if not args.no_falcon:
        falcon = FalconClient(host=args.falcon_host, port=args.falcon_port)
        if falcon.health():
            print(f"  Falcon server: connected ({args.falcon_host}:{args.falcon_port})")
        else:
            print(f"  Falcon server: unreachable, using depth-probing fallback")
            falcon = None

    # === TRELLIS client (optional, for B-class mesh generation) ===
    trellis: TrellisClient | None = None
    if not args.no_trellis:
        t_cfg = app_config.trellis
        t_host = args.trellis_host or t_cfg.host
        t_port = args.trellis_port or t_cfg.port
        trellis = TrellisClient(host=t_host, port=t_port, timeout=t_cfg.timeout)
        if trellis.health():
            print(f"  TRELLIS server: connected ({t_host}:{t_port})")
        else:
            print(f"  TRELLIS server: unreachable, B-class mesh generation disabled")
            trellis = None

    data_dir = ROOT / "data" / args.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / args.name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {out_dir}")

    # === Stage 1: Load scene ===
    ply_path, feat_path = find_scene_files(data_dir)
    print(f"{'='*60}")
    print(f"3DGS -> BIM Pipeline: {args.name}")
    print(f"{'='*60}")
    print(f"  PLY:  {ply_path.name}")
    print(f"  Feat: {feat_path.name}")

    text_emb_path = str(ROOT / "data" / "bim_text_emb.pt")
    class_names_path = str(ROOT / "data" / "bim_class_names.json")
    warm_kwargs: dict = {}
    if Path(text_emb_path).exists() and Path(class_names_path).exists():
        warm_kwargs = {"text_emb_path": text_emb_path, "class_names_path": class_names_path}
    scene = GSScene.from_ply(ply_path, feat_path=feat_path, **warm_kwargs)
    print(f"  Gaussians: {scene.num_gaussians}")

    # Build the active open-vocabulary label set: structural defaults + the
    # semantic label of every requested element type (deduped, structural first).
    structural_labels = ["wall", "floor", "ceiling"]
    element_labels: list[str] = []
    for elem_type in args.elements:
        try:
            element_labels.append(get_element_config(elem_type).semantic_label)
        except KeyError:
            pass
    labels = list(dict.fromkeys(structural_labels + element_labels))
    print(f"  Open-vocab label set ({len(labels)}): {labels}")

    # === Stage 2: Detect coordinate system ===
    coords = detect_coordinate_system(scene, label_set=labels)
    center = coords["center"]
    floor_z = coords["floor_z"]
    ceiling_z = coords["ceiling_z"]
    up_axis = coords["up_axis"]
    print(f"  up_axis={up_axis}, floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")
    print(f"  Scan center: ({center[0]:.2f}, {center[1]:.2f})")

    # === Stage 3: Multi-height scan (shared) ===
    print(f"\n--- Stage 1: Radar Scan ({args.num_heights} heights) ---")
    scanner = VirtualScanner(scene, up_axis=up_axis, labels=labels)
    scans = multi_height_scan(
        scanner, center, floor_z, ceiling_z,
        num_heights=args.num_heights, num_views=8, width=512,
    )
    total_pts = sum(len(s.angles_deg) for s in scans)
    print(f"  Total scan points: {total_pts}")

    # === Stage 1b: 3D spherical scan (floor/ceiling + unified point cloud) ===
    print(f"\n--- Stage 1b: 3D Spherical Scan ---")
    scan_3d = scanner.scan_3d(
        center, floor_z, ceiling_z,
        n_azimuth_views=12, n_elevation_bands=5,
        width=512, fov=45.0,
    )
    print(f"  3D points: {len(scan_3d.points_3d)}")

    # Refine floor/ceiling from 3D point cloud
    floor_refined, ceil_refined = VirtualScanner.detect_floor_ceiling(
        scan_3d, labels=labels)
    if abs(floor_refined - floor_z) < 0.5 and abs(ceil_refined - ceiling_z) < 0.5:
        floor_z = floor_refined
        ceiling_z = ceil_refined
        print(f"  Refined floor={floor_z:.3f}, ceiling={ceiling_z:.3f} "
              f"(height={ceiling_z - floor_z:.3f}m)")
    else:
        print(f"  3D refinement skipped (delta too large), keeping semantic values")


    # === Stage 4: Wall extraction ===
    print(f"\n--- Stage 2: Wall Extraction ---")
    walls = extract_walls(scans, np.array(center), out_dir, labels=labels)
    print(f"  Extracted {len(walls)} wall segments")

    # Snap endpoints
    walls_snapped = snap_wall_endpoints(walls, args.snap_threshold)
    snapped_path = out_dir / "wall_lines_snapped.json"
    snapped_path.write_text(json.dumps(walls_snapped, indent=2), encoding="utf-8")
    print(f"  Snapped walls saved: {snapped_path}")

    # === Stage 3: Ring scan → per-view seg → polar merge → targeted VLM ===
    all_results: dict[str, dict] = {}
    merged_elements: list = []
    falcon_dets: list[dict] = []

    # Determine element labels for Falcon query
    elem_labels = []
    for et in args.elements:
        try:
            elem_labels.append(get_element_config(et).semantic_label)
        except KeyError:
            pass

    if falcon is not None and elem_labels:
        from bim_recon.ring_scanner import (
            render_ring_views, segment_ring_views, render_element_view,
        )

        # Stage 3a: Render overlapping ring views
        mid_z = (floor_z + ceiling_z) / 2.0
        n_ring = 12
        ring_fov = 45.0
        print(f"\n--- Stage 3a: Ring Scan ({n_ring} views x {ring_fov}deg) ---")
        ring_views = render_ring_views(
            scene, center, mid_z, up_axis=up_axis,
            n_views=n_ring, fov=ring_fov, img_size=768,
        )
        print(f"  Rendered {len(ring_views)} views")

        # Save ring views for debugging
        ring_dir = out_dir / "ring_views"
        ring_dir.mkdir(exist_ok=True)
        for v in ring_views:
            from PIL import Image as _PIL
            _PIL.fromarray(v.image).save(
                str(ring_dir / f"view_{v.idx:02d}_{v.azimuth_deg:.0f}.png"))

        # Stage 3b: Falcon segmentation per view
        print(f"\n--- Stage 3b: Per-View Falcon Segmentation ---")
        view_dets = segment_ring_views(
            ring_views, falcon, elem_labels,
            center_2d=center, floor_z=floor_z, ceiling_z=ceiling_z,
            up_axis=up_axis,
        )
        print(f"  Raw detections across all views: {len(view_dets)}")
        from collections import Counter
        for label, count in Counter(d.label for d in view_dets).most_common():
            print(f"    {label}: {count}")

        # Save raw detections
        raw_json = [
            {"label": d.label, "view": d.view_idx, "azimuth": d.azimuth_deg,
             "world_x": round(d.world_x, 3), "world_y": round(d.world_y, 3),
             "sill_h": round(d.sill_height, 3), "header_h": round(d.header_height, 3),
             "width_m": round(d.width_m, 3), "centrality": round(d.centrality, 3)}
            for d in view_dets
        ]
        (out_dir / "ring_raw_detections.json").write_text(
            json.dumps(raw_json, indent=2, ensure_ascii=False), encoding="utf-8")

        # Stage 3c: Merge in polar space
        print(f"\n--- Stage 3c: Polar Merge ---")
        merge_input = [
            {"element_class": d.label, "world_x": d.world_x, "world_y": d.world_y,
             "sill_height": d.sill_height, "header_height": d.header_height,
             "width_m": d.width_m, "confidence": d.centrality}
            for d in view_dets
        ]
        merged_elements = merge_detections(
            merge_input, center, up_axis=up_axis, merge_threshold=0.5)
        print(f"  Merged: {len(view_dets)} raw -> {len(merged_elements)} unique")
        for me in merged_elements:
            print(f"    [{me.element_class}] theta={me.theta_center:.1f}deg "
                  f"r={me.r_mean:.2f}m w={me.width_m:.2f}m "
                  f"(from {me.num_sources} views)")

        # Stage 3d: Targeted VLM verification per merged element
        print(f"\n--- Stage 3d: Targeted VLM Verification ---")
        verify_dir = out_dir / "verify_merged"
        verify_dir.mkdir(exist_ok=True)

        confirmed: list[dict] = []
        for mi, me in enumerate(merged_elements):
            # Render a fresh targeted view with auto-FOV
            ev = render_element_view(
                scene, me.world_x, me.world_y,
                width_m=me.width_m, height_m=max(me.element_height, 0.5),
                mid_z=mid_z, center_2d=center, up_axis=up_axis,
                img_size=768, margin=0.5,
            )
            if ev is None:
                print(f"    [{me.element_class}] theta={me.theta_center:.1f}deg: "
                      f"render failed, skipping")
                continue

            img_name = f"merged_{mi}_{me.element_class}.png"
            from PIL import Image as _PIL
            _PIL.fromarray(ev.image).save(str(verify_dir / img_name))

            # VLM judgment
            if args.skip_vlm:
                vlm_ok = True
                vlm_resp = "skipped"
            else:
                from bim_recon.vlm_verifier import query_vlm
                try:
                    cfg = get_element_config(me.element_class)
                    prompt = cfg.vlm_hint
                except KeyError:
                    prompt = f"Is there a {me.element_class} in this image?"
                vlm_resp = query_vlm(
                    str(verify_dir / img_name), prompt,
                    vlm_api_base, vlm_model, vlm_api_key,
                )
                vlm_ok = any(kw in vlm_resp.lower() for kw in
                             ("yes", "confir", "correct", "true", "is a"))

            tag = "CONFIRMED" if vlm_ok else "REJECTED"
            print(f"    [{me.element_class}] theta={me.theta_center:.1f}deg "
                  f"r={me.r_mean:.2f}m: {tag}")

            if vlm_ok:
                confirmed.append({
                    **me.to_dict(), "image_path": img_name,
                    "vlm_response": vlm_resp, "fov_deg": ev.fov_deg,
                })

        # Build per-element-type results (downstream-compatible format)
        for elem_type in args.elements:
            try:
                cfg = get_element_config(elem_type)
            except KeyError:
                continue
            type_conf = [c for c in confirmed if c["element_class"] == cfg.semantic_label]
            all_results[elem_type] = {
                "total_candidates": len(view_dets),
                "after_prefilter": len(merged_elements),
                "confirmed": len(type_conf),
                "results": [{
                    "confirmed": True,
                    "candidate": {
                        "world_x": c["world_x"], "world_y": c["world_y"],
                        "theta_center": c["theta_center"], "r_mean": c["r_mean"],
                        "width_m": c["width_m"],
                    },
                    "height_detection": {
                        "sill_height": c["sill_height"],
                        "header_height": c["header_height"],
                        "element_height": c["element_height"],
                        "width_m": c["width_m"],
                    },
                    "image_path": c["image_path"],
                } for c in type_conf],
            }
            elem_json = {"scene": args.name, "element": elem_type,
                         "ply_used": ply_path.name,
                         "vlm_model": vlm_model if not args.skip_vlm else None,
                         **all_results[elem_type]}
            (out_dir / cfg.output_json_name).write_text(
                json.dumps(elem_json, indent=2), encoding="utf-8")

        # Save merged results
        merged_json = {
            "raw_count": len(view_dets),
            "merged_count": len(merged_elements),
            "confirmed_count": len(confirmed),
            "merged": [me.to_dict() for me in merged_elements],
            "confirmed": confirmed,
        }
        (out_dir / "merged_elements.json").write_text(
            json.dumps(merged_json, indent=2, ensure_ascii=False), encoding="utf-8")

        # === Generate ring-scan radar plots (Cartesian top-down) ===
        print(f"\n--- Generating Ring Radar Plots ---")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge

        cx, cy = float(center[0]), float(center[1])
        h_axes = [i for i in range(3) if i != up_axis]

        def _draw_walls(ax):
            for wl in walls_snapped:
                x1, y1 = wl["x1"] - cx, wl["y1"] - cy
                x2, y2 = wl["x2"] - cx, wl["y2"] - cy
                ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)

        def _draw_pca_line(ax, pts, color, label=None, lw=2.5):
            """Draw PCA main-axis line segment through point cloud."""
            if len(pts) < 3:
                ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=3, alpha=0.5, zorder=7)
                return
            ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=3, alpha=0.4, zorder=7)
            centered = pts - pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            main_dir = Vt[0]
            proj = centered @ main_dir
            c = pts.mean(axis=0)
            p1 = c + main_dir * proj.min()
            p2 = c + main_dir * proj.max()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color,
                    linewidth=lw, zorder=8, label=label)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color=color,
                    markersize=5, zorder=9)

        # Radar 1: Raw detections (Cartesian, mask point clouds)
        if view_dets:
            fig, ax = plt.subplots(1, 1, figsize=(14, 14))
            _draw_walls(ax)
            label_colors = {"door": "red", "window": "blue", "column": "gray"}
            palette = plt.cm.Set1(np.linspace(0, 1, max(len(view_dets), 1)))
            for di, d in enumerate(view_dets):
                color = label_colors.get(d.label, "green")
                if d.mask_points_xy is not None:
                    pts = d.mask_points_xy - np.array([cx, cy])
                    _draw_pca_line(ax, pts, color,
                                   label=f"{d.label} v{d.view_idx}" if di < 12 else None)
                else:
                    dx, dy = d.world_x - cx, d.world_y - cy
                    ax.scatter(dx, dy, c=color, s=30, zorder=7)
            ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
            ax.set_aspect("equal")
            max_r = max(abs(wl["x1"]-cx).max() if walls_snapped else 5,
                        abs(wl["x2"]-cx).max() if walls_snapped else 5, 5)
            ax.set_xlim(-max_r-1, max_r+1)
            ax.set_ylim(-max_r-1, max_r+1)
            ax.set_title(f"Ring Raw Detections ({len(view_dets)} masks, pre-merge)", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.savefig(str(out_dir / "radar_ring_raw.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  radar_ring_raw.png saved")

        # Radar 2: Merged elements (Cartesian, combined point clouds)
        if merged_elements:
            fig, ax = plt.subplots(1, 1, figsize=(14, 14))
            _draw_walls(ax)
            # View fans (light)
            for v in ring_views:
                az = math.radians(v.azimuth_deg)
                hfov = math.radians(v.fov_deg / 2)
                max_d = float(np.percentile(v.depth[v.depth > 0.1], 90)) if (v.depth > 0.1).any() else 5.0
                ax.add_patch(Wedge((0, 0), max_d,
                                    math.degrees(az - hfov), math.degrees(az + hfov),
                                    alpha=0.04, color="lightblue", zorder=1))
            palette = plt.cm.Set1(np.linspace(0, 1, max(len(merged_elements), 1)))
            for mi, me in enumerate(merged_elements):
                color = palette[mi % len(palette)]
                # Combine all source mask points
                src_pts = []
                for si in me.source_indices:
                    if si < len(view_dets) and view_dets[si].mask_points_xy is not None:
                        src_pts.append(view_dets[si].mask_points_xy - np.array([cx, cy]))
                if src_pts:
                    all_pts = np.vstack(src_pts)
                    _draw_pca_line(ax, all_pts, color,
                                   label=f"{me.element_class} ({me.num_sources} masks)")
                else:
                    dx, dy = me.world_x - cx, me.world_y - cy
                    ax.scatter(dx, dy, c=[color], s=100, marker="*", zorder=8)
            ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
            ax.set_aspect("equal")
            max_r = max(abs(wl["x1"]-cx).max() if walls_snapped else 5,
                        abs(wl["x2"]-cx).max() if walls_snapped else 5, 5)
            ax.set_xlim(-max_r-1, max_r+1)
            ax.set_ylim(-max_r-1, max_r+1)
            ax.set_title(f"Merged Elements ({len(merged_elements)} unique, post-clustering)", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.savefig(str(out_dir / "radar_merged.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  radar_merged.png saved")



    else:
        # --- FALLBACK: old SceneSplat + VLM flow (no Falcon) ---
        print(f"\n--- Stage 3b: Element Detection (SceneSplat fallback) ---")
        for elem_type in args.elements:
            try:
                cfg = get_element_config(elem_type)
            except KeyError:
                continue
            result = detect_elements(
                cfg, scans, walls_snapped, coords, scene,
                out_dir, vlm_api_base, vlm_model, vlm_api_key, args.skip_vlm,
                falcon=None, labels=labels,
            )
            all_results[elem_type] = result
            elem_json = {"scene": args.name, "element": elem_type,
                         "ply_used": ply_path.name,
                         "vlm_model": vlm_model if not args.skip_vlm else None,
                         **result}
            (out_dir / cfg.output_json_name).write_text(
                json.dumps(elem_json, indent=2), encoding="utf-8")
        merged_elements = []

    # === Generate radar plots (SceneSplat semantic points as annotation) ===
    _generate_element_radars(scans, walls_snapped, all_results,
                             center, floor_z, up_axis, out_dir, label_set=labels)



    # === Stage 5c: B-class mesh generation + placement via TRELLIS (optional) ===
    trellis_results: list[dict] = []
    if trellis is not None:
        print(f"\n--- Stage 4: B-class Mesh Generation + Placement (TRELLIS) ---")
        trellis_dir = out_dir / "trellis_meshes"
        trellis_dir.mkdir(parents=True, exist_ok=True)

        # Build a lookup: image filename → candidate world coordinates + dims
        candidate_lookup: dict[str, dict] = {}
        for elem_type, result in all_results.items():
            for r in result.get("results", []):
                img_name = r.get("image_path", "")
                if not img_name:
                    continue
                c = r.get("candidate", {})
                hd = r.get("height_detection", {}) or {}
                candidate_lookup[Path(img_name).stem] = {
                    "element": elem_type,
                    "world_x": c.get("world_x", 0.0),
                    "world_y": c.get("world_y", 0.0),
                    "width_m": hd.get("width_m", c.get("width_m", 0.5)),
                    "height_m": hd.get("element_height", c.get("h_max", 1.0) - c.get("h_min", 0.0)),
                    "confirmed": r.get("confirmed"),
                }

        # Collect VLM-verified images — only for B-class elements per config
        routing = app_config.element_routing
        b_class_types = set(routing.b_class_types())
        vlm_images: list[tuple[str, str]] = []
        for elem_type, result in all_results.items():
            if elem_type not in b_class_types:
                continue
            verify_subdir = out_dir / f"verify_{elem_type}"
            for r in result.get("results", []):
                # VLM-rejected views must NOT reach Falcon/TRELLIS — only
                # confirmed candidates proceed to segmentation + mesh gen.
                if r.get("confirmed") is not True:
                    continue
                img_name = r.get("image_path", "")
                if not img_name:
                    continue
                img_path = verify_subdir / img_name
                if img_path.exists():
                    vlm_images.append((elem_type, str(img_path)))

        print(f"  B-class types ({', '.join(sorted(b_class_types)) or 'none'}): "
              f"{len(vlm_images)} images to process")

        if vlm_images:
            for elem_type, img_path in vlm_images:
                img_stem = Path(img_path).stem
                try:
                    cand = candidate_lookup.get(img_stem)

                    # Step 0: Mesh readiness check — render multi-angle, VLM judges suitability
                    readiness_dir = trellis_dir / "readiness"
                    if cand and not args.skip_vlm:
                        readiness = render_and_check_mesh_readiness(
                            scene=scene,
                            world_x=cand["world_x"],
                            world_y=cand["world_y"],
                            h_min=cand.get("h_min", 0.0),
                            h_max=cand.get("h_max", 2.0),
                            scan_center=center,
                            floor_z=floor_z,
                            element_class=elem_type,
                            vlm_api_base=vlm_api_base,
                            vlm_model=vlm_model,
                            vlm_api_key=vlm_api_key,
                            output_dir=readiness_dir,
                            name_prefix=img_stem,
                            up_axis=up_axis,
                        )
                        if not readiness.is_ready:
                            print(f"  ✗ {img_stem}: mesh readiness FAILED — {readiness.reason}")
                            trellis_results.append({
                                "element": elem_type,
                                "source_image": img_path,
                                "error": f"readiness_failed: {readiness.reason}",
                            })
                            continue
                        print(f"  👁 {img_stem}: readiness OK ({readiness.reason})")
                        assert readiness.best_image_path is not None
                        best_image: Path = readiness.best_image_path
                    else:
                        best_image = Path(img_path)  # fallback: use VLM verification image

                    # Step 1: Falcon mask → clean object image
                    clean_image_path: Path = best_image
                    if falcon is not None:
                        from PIL import Image as PILImage
                        render = PILImage.open(str(best_image)).convert("RGB")
                        detections = falcon.segment(render, elem_type, task="segmentation")
                        det_dicts = [
                            {"bbox": d.bbox, "mask_bbox": d.mask_bbox, "mask_area_ratio": d.mask_area_ratio}
                            for d in detections
                        ]
                        clean = extract_object_from_render(render, det_dicts)
                        if clean is not None:
                            clean_image_path = trellis_dir / f"{img_stem}_clean.png"
                            clean.save(str(clean_image_path))
                            print(f"  📌 {img_stem}: Falcon masked → {clean.size}")
                        else:
                            print(f"  ⚠ {img_stem}: Falcon returned no mask, using raw image")

                    # Step 2: Send clean image to TRELLIS
                    mesh_result = trellis.generate_mesh(TrellisMeshRequest(
                        image_path=Path(clean_image_path),
                        output_dir=trellis_dir,
                        name=f"{elem_type}_{img_stem}",
                    ))

                    entry: dict = {
                        "element": elem_type,
                        "source_image": img_path,
                        "glb_path": str(mesh_result.glb_path),
                        "gaussian_path": str(mesh_result.gaussian_path) if mesh_result.gaussian_path else None,
                    }

                    # Compute placement transform if we have candidate coordinates
                    cand = candidate_lookup.get(img_stem)
                    if cand and cand.get("confirmed") is not False:
                        placement = MeshPlacement(
                            glb_path=mesh_result.glb_path,
                            world_x=cand["world_x"],
                            world_y=cand["world_y"],
                            floor_z=floor_z,
                            ceiling_z=ceiling_z,
                            element_width_m=cand["width_m"],
                            element_height_m=cand["height_m"],
                            up_axis=up_axis,
                            name=f"{elem_type}_{img_stem}",
                        )
                        transform = compute_placement_transform(placement)
                        reg_result = register_mesh_in_revit(placement, transform)
                        entry["placement"] = {
                            "scale": transform.scale,
                            "vertex_count": reg_result["vertex_count"],
                            "face_count": reg_result["face_count"],
                            "status": reg_result["status"],
                        }
                        print(f"  ✓ {img_stem}: GLB={mesh_result.glb_path.name}, "
                              f"placement={reg_result['status']}")
                    else:
                        print(f"  ✓ {img_stem}: GLB={mesh_result.glb_path.name} (no placement data)")

                    trellis_results.append(entry)

                except Exception as e:
                    print(f"  ✗ {img_stem}: {e}")
                    trellis_results.append({
                        "element": elem_type,
                        "source_image": img_path,
                        "error": str(e),
                    })
        else:
            print(f"  No VLM images found, skipping mesh generation")
    elif not args.no_trellis:
        print(f"\n  (TRELLIS server unreachable, B-class mesh generation skipped)")

    # === Stage 6: Pipeline report ===
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Walls:    {len(walls_snapped)} segments (closed polygon)")
    for elem_type, result in all_results.items():
        print(f"  {elem_type:10s} {result['confirmed']} confirmed / "
              f"{result['after_prefilter']} filtered / "
              f"{result['total_candidates']} raw")
    if merged_elements:
        print(f"  Merged:   {len(merged_elements)} unique elements "
              f"(from multi-view dedup)")

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
            "scan_3d_points": len(scan_3d.points_3d),
        },
        "walls": {
            "count": len(walls_snapped),
            "snapped": True,
        },
        "elements": all_results,
        "merged_elements": {
            "count": len(merged_elements),
            "elements": [me.to_dict() for me in merged_elements],
        },
        "vlm_model": vlm_model if not args.skip_vlm else None,
        "trellis": {
            "enabled": trellis is not None,
            "meshes_generated": len([r for r in trellis_results if "glb_path" in r]),
            "results": trellis_results,
        },
    }
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report: {report_path}")
    print(f"  Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
