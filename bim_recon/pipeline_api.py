"""Programmatic API for running the 3DGS→BIM pipeline and loading results.

Wraps the stage functions from ``scripts.run_pipeline`` so the Gradio UI
(and other callers) can invoke the pipeline without argparse/CLI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bim_recon.spatial_extractor import bbox_to_wall_coords, ElevationParams

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Results data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ElementResult:
    """One detected element (door or window) with all metadata."""
    element_class: str
    confirmed: bool
    vlm_response: str
    image_path: str
    world_x: float
    world_y: float
    wall_idx: int
    result_index: int
    height_detection: dict[str, Any] | None = None
    elevation_image: str | None = None
    overlay_image: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineResults:
    """All results from a pipeline run, loaded from the output directory."""
    out_dir: Path
    walls: list[dict[str, Any]]
    doors: list[ElementResult]
    windows: list[ElementResult]
    coords: dict[str, Any]
    wall_topdown_image: str | None = None
    report: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scene_name: str,
    elements: list[str] | None = None,
    use_falcon: bool = True,
    falcon_host: str = "127.0.0.1",
    falcon_port: int = 8390,
    skip_vlm: bool = False,
) -> Path:
    """Run the pipeline and return the output directory path.

    Delegates to ``scripts.run_pipeline.main()`` via subprocess to ensure
    gsplat JIT compilation context (vcvars64) is available.
    """
    import subprocess
    import sys
    from datetime import datetime

    elements = elements or ["door", "window"]
    args = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
            "--name", scene_name, "--elements", *elements]
    if skip_vlm:
        args.append("--skip-vlm")
    if not use_falcon:
        args.append("--no-falcon")
    else:
        args.extend(["--falcon-host", falcon_host, "--falcon-port", str(falcon_port)])

    subprocess.run(args, check=True, cwd=str(ROOT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "output" / scene_name / timestamp


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(out_dir: Path) -> PipelineResults:
    """Load all pipeline results from an output directory."""
    walls = json.loads((out_dir / "wall_lines_snapped.json").read_text("utf-8"))
    report = json.loads((out_dir / "pipeline_report.json").read_text("utf-8"))
    coords = report.get("coords", {})

    topdown = str(out_dir / "wall_lines_topdown.png")

    doors = _load_elements(out_dir, "doors_verified.json")
    windows = _load_elements(out_dir, "windows_verified.json")

    return PipelineResults(
        out_dir=out_dir, walls=walls, doors=doors, windows=windows,
        coords=coords, wall_topdown_image=topdown, report=report,
    )


def _load_elements(out_dir: Path, filename: str) -> list[ElementResult]:
    """Parse a ``*_verified.json`` file into ``ElementResult`` list.

    Loads ALL candidates (confirmed, rejected, and VLM-error) so the UI
    always shows feedback. The ``confirmed`` field on each ElementResult
    distinguishes: ``True`` = VLM confirmed, ``False`` = VLM rejected,
    ``None`` = VLM error (e.g. server unreachable).
    """
    path = out_dir / filename
    if not path.exists():
        return []
    data = json.loads(path.read_text("utf-8"))
    results: list[ElementResult] = []
    for result_index, r in enumerate(data.get("results", [])):
        c = r.get("candidate", {})
        hd = r.get("height_detection")
        elem_class = c.get("element_class", "")
        # Elevation images: search by result index
        elevation_candidates = list(out_dir.glob(f"*_{result_index}_elevation.png"))
        overlay_candidates = list(out_dir.glob(f"*_{result_index}_elevation_overlay.png"))
        elevation_img = str(elevation_candidates[0]) if elevation_candidates else None
        overlay_img = str(overlay_candidates[0]) if overlay_candidates else None
        # VLM verification images: stored in verify_<class>/ subdirectory
        vlm_name = r.get("image_path", "")
        vlm_path = out_dir / f"verify_{elem_class}" / vlm_name
        if not vlm_path.exists():
            vlm_path = out_dir / vlm_name  # fallback to flat layout
        results.append(ElementResult(
            element_class=elem_class,
            confirmed=r.get("confirmed", False),
            vlm_response=r.get("vlm_response", ""),
            image_path=str(vlm_path) if vlm_path.exists() else "",
            world_x=c.get("world_x", 0.0),
            world_y=c.get("world_y", 0.0),
            wall_idx=c.get("wall_idx", -1),
            result_index=result_index,
            height_detection=hd,
            elevation_image=elevation_img if elevation_img and Path(elevation_img).exists() else None,
            overlay_image=overlay_img if overlay_img and Path(overlay_img).exists() else None,
        ))
    return results


# ---------------------------------------------------------------------------
# Seg bbox remap
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ManualBBox:
    """Normalized bbox (0-1) for manual seg adjustment."""
    cx: float
    cy: float
    w: float
    h: float


def remap_manual_bbox(
    bbox: ManualBBox,
    params: ElevationParams,
    floor_z: float,
    ceiling_z: float,
) -> dict[str, Any]:
    """Map a user-adjusted bbox back to wall-local metric coordinates.

    Uses the same math as ``bbox_to_wall_coords`` in spatial_extractor.
    """
    result = bbox_to_wall_coords(
        {"x": bbox.cx, "y": bbox.cy, "w": bbox.w, "h": bbox.h},
        params, floor_z, ceiling_z,
    )
    if result is None:
        return {"error": "Invalid bbox mapping"}
    return {
        "sill_height": result.sill_height,
        "header_height": result.header_height,
        "element_height": result.element_height,
        "width_m": result.width_m,
    }


def remap_from_json(
    cx: float,
    cy: float,
    w: float,
    h: float,
    elev_params_dict: dict[str, Any] | None,
    floor_z: float,
    ceiling_z: float,
) -> dict[str, Any]:
    """Remap a manual bbox to wall coordinates using serialized ElevationParams.

    This is the JSON-friendly version: accepts the dict saved in
    ``height_detection.elevation_params`` rather than the ElevationParams
    dataclass (which contains numpy arrays).
    """
    if not elev_params_dict:
        return {"error": "No elevation_params in saved results. "
                         "Re-run pipeline with updated spatial_extractor."}

    import numpy as np
    params = ElevationParams(
        camera_dist=2.5,
        fov_degrees=60.0,
        img_size=800,
        wall_length=elev_params_dict["wall_length"],
        wall_dir=np.array(elev_params_dict["wall_dir"], dtype=np.float64),
        wall_start=np.array(elev_params_dict["wall_start"], dtype=np.float64),
        target_along=elev_params_dict["target_along"],
        cam_h_above_floor=elev_params_dict["cam_h_above_floor"],
        extent_h=elev_params_dict["extent_h"],
        extent_v=elev_params_dict["extent_v"],
    )
    bbox = ManualBBox(cx=cx, cy=cy, w=w, h=h)
    return remap_manual_bbox(bbox, params, floor_z, ceiling_z)


def mask_to_bbox(
    mask_rgba: "np.ndarray",
    elev_params_dict: dict[str, Any] | None,
    floor_z: float,
    ceiling_z: float,
) -> dict[str, Any]:
    """Convert a painted RGBA mask to wall coordinates.

    Takes the ``layers[0]`` output from ``gr.ImageMask`` (RGBA numpy array
    where alpha > 0 marks the painted region), computes the tightest
    normalized bounding box, and maps it to wall-local metres via
    ``bbox_to_wall_coords``.

    Returns the same dict as :func:`remap_from_json`.
    """
    if not elev_params_dict:
        return {"error": "No elevation_params in saved results. "
                         "Re-run pipeline with updated spatial_extractor."}

    # Extract binary mask from alpha channel
    if mask_rgba.ndim != 3 or mask_rgba.shape[2] < 4:
        return {"error": f"Expected RGBA mask, got shape {mask_rgba.shape}"}
    alpha = mask_rgba[:, :, 3]
    binary = alpha > 10  # threshold to avoid anti-aliasing noise
    if not binary.any():
        return {"error": "Mask is empty — please paint over the element."}

    # Tightest bbox in normalized [0, 1] coordinates
    ys, xs = np.where(binary)
    h_img, w_img = alpha.shape
    x_min, x_max = xs.min() / w_img, (xs.max() + 1) / w_img
    y_min, y_max = ys.min() / h_img, (ys.max() + 1) / h_img
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    params = ElevationParams(
        camera_dist=2.5,
        fov_degrees=60.0,
        img_size=800,
        wall_length=elev_params_dict["wall_length"],
        wall_dir=np.array(elev_params_dict["wall_dir"], dtype=np.float64),
        wall_start=np.array(elev_params_dict["wall_start"], dtype=np.float64),
        target_along=elev_params_dict["target_along"],
        cam_h_above_floor=elev_params_dict["cam_h_above_floor"],
        extent_h=elev_params_dict["extent_h"],
        extent_v=elev_params_dict["extent_v"],
    )
    bbox = ManualBBox(cx=cx, cy=cy, w=w, h=h)
    return remap_manual_bbox(bbox, params, floor_z, ceiling_z)
