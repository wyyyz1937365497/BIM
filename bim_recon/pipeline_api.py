"""Programmatic API for running the 3DGS→BIM pipeline and loading results.

Wraps the stage functions from ``scripts.run_pipeline`` so the Gradio UI
(and other callers) can invoke the pipeline without argparse/CLI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Parse a ``*_verified.json`` file into ``ElementResult`` list (confirmed only)."""
    path = out_dir / filename
    if not path.exists():
        return []
    data = json.loads(path.read_text("utf-8"))
    results: list[ElementResult] = []
    for r in data.get("results", []):
        if not r.get("confirmed"):
            continue
        c = r.get("candidate", {})
        hd = r.get("height_detection")
        idx = r.get("image_path", "")
        elem_class = c.get("element_class", "")
        idx_base = elem_class + "_" + idx.split("_")[-2] if "_" in idx else idx
        elevation_img = str(out_dir / f"{idx_base}_elevation.png") if idx_base else None
        overlay_img = str(out_dir / f"{idx_base}_elevation_overlay.png") if idx_base else None
        results.append(ElementResult(
            element_class=elem_class,
            confirmed=r.get("confirmed", False),
            vlm_response=r.get("vlm_response", ""),
            image_path=str(out_dir / r.get("image_path", "")),
            world_x=c.get("world_x", 0.0),
            world_y=c.get("world_y", 0.0),
            wall_idx=c.get("wall_idx", -1),
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
