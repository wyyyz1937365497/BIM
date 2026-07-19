"""Persist overlap-free wall geometry while preserving hosted element indices."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from bim_recon.wall_geometry import merge_overlapping_walls


def _write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact without leaving a partial result."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remap_wall_indices(value: Any, index_map: dict[int, int]) -> Any:
    if isinstance(value, list):
        return [_remap_wall_indices(item, index_map) for item in value]
    if not isinstance(value, dict):
        return value
    remapped: dict[str, Any] = {}
    for key, child in value.items():
        if key == "wall_idx" and isinstance(child, int):
            if child not in index_map:
                raise ValueError(f"wall_idx {child} has no retained wall")
            remapped[key] = index_map[child]
        else:
            remapped[key] = _remap_wall_indices(child, index_map)
    return remapped


def clean_saved_wall_list(out_dir: str | Path) -> dict[str, int]:
    """Remove overlapping saved walls and remap every persisted ``wall_idx``.

    Pipeline results use the numeric wall position as the host reference for
    doors and windows.  This operation therefore rewrites the wall list and
    all verified-element/report references in one atomic-per-file migration.
    """
    output_dir = Path(out_dir)
    walls_path = output_dir / "wall_lines_snapped.json"
    walls = json.loads(walls_path.read_text(encoding="utf-8"))
    if not isinstance(walls, list):
        raise ValueError(f"Expected a wall list in {walls_path}")

    cleaned_walls, source_groups = merge_overlapping_walls(walls)
    index_map = {
        source_index: merged_index
        for merged_index, source_indices in enumerate(source_groups)
        for source_index in source_indices
    }
    removed = len(walls) - len(cleaned_walls)
    if not removed:
        return {"input_walls": len(walls), "removed_walls": 0, "output_walls": len(walls)}

    _write_json(walls_path, cleaned_walls)
    rewritten_files = 0
    for path in sorted(output_dir.glob("*_verified.json")):
        _write_json(
            path,
            _remap_wall_indices(
                json.loads(path.read_text(encoding="utf-8")),
                index_map,
            ),
        )
        rewritten_files += 1

    report_path = output_dir / "pipeline_report.json"
    if report_path.exists():
        report = _remap_wall_indices(
            json.loads(report_path.read_text(encoding="utf-8")),
            index_map,
        )
        if isinstance(report.get("walls"), dict):
            report["walls"]["count"] = len(cleaned_walls)
        _write_json(report_path, report)
        rewritten_files += 1

    return {
        "input_walls": len(walls),
        "removed_walls": removed,
        "output_walls": len(cleaned_walls),
        "rewritten_files": rewritten_files,
    }
