"""Element type configuration registry.

Defines per-element-type defaults for the VLM-verified extraction pipeline.
Each element type (door, window, furniture, ...) has its own physical
constraints: typical width range, minimum scan point count, whether it
attaches to walls or stands freely, and a VLM prompt hint.

Usage::

    from bim_recon.element_config import get_element_config

    cfg = get_element_config("window")
    candidates = extract_candidates(..., element_class=cfg.name, class_idx=cfg.class_idx, ...)
    filtered = prefilter_candidates(candidates, cfg.min_width, cfg.min_points)
    results = verify_candidates(..., element_class=cfg.name, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ElementConfig:
    """Configuration for a BIM element type in the extraction pipeline."""

    name: str               # "door", "window", "furniture", ...
    class_idx: int          # semantic class index in feat.pt
    structural: bool        # True = project to walls, False = free-standing DBSCAN
    min_width: float        # minimum candidate width (meters)
    min_points: int         # minimum scan points per candidate
    vlm_hint: str           # extra context for VLM prompt (appended to element name)
    height_detection: bool  # True = refine sill/header heights after VLM confirm

    @property
    def output_json_name(self) -> str:
        """Output JSON filename: e.g. 'doors_verified.json'."""
        return f"{self.name}s_verified.json"

    @property
    def verify_dir_name(self) -> str:
        """Per-candidate image directory: e.g. 'verify_door/'."""
        return f"verify_{self.name}"


# ---------------------------------------------------------------------------
# Registry: per-element defaults
# ---------------------------------------------------------------------------

ELEMENT_CONFIGS: Dict[str, ElementConfig] = {
    "door": ElementConfig(
        name="door",
        class_idx=3,
        structural=True,
        min_width=0.7,
        min_points=100,
        vlm_hint="a door (passage between rooms, usually has a frame and handle)",
        height_detection=True,
    ),
    "window": ElementConfig(
        name="window",
        class_idx=4,
        structural=True,
        min_width=0.5,
        min_points=50,
        vlm_hint="a window (glass opening in a wall, may have blinds or curtains)",
        height_detection=True,
    ),
    "column": ElementConfig(
        name="column",
        class_idx=5,
        structural=True,
        min_width=0.2,
        min_points=80,
        vlm_hint="a structural column (vertical load-bearing pillar)",
        height_detection=False,
    ),
    "furniture": ElementConfig(
        name="furniture",
        class_idx=8,
        structural=False,   # free-standing → DBSCAN clustering
        min_width=0.3,
        min_points=50,
        vlm_hint="a piece of furniture (e.g. bed, sofa, cabinet, table)",
        height_detection=False,
    ),
}


def get_element_config(element_class: str) -> ElementConfig:
    """Look up element configuration by name.

    Raises:
        KeyError: If *element_class* is not in the registry.
    """
    if element_class not in ELEMENT_CONFIGS:
        raise KeyError(
            f"Unknown element type '{element_class}'. "
            f"Available: {sorted(ELEMENT_CONFIGS.keys())}"
        )
    return ELEMENT_CONFIGS[element_class]


def list_element_types() -> list[str]:
    """Return all registered element type names."""
    return sorted(ELEMENT_CONFIGS.keys())
