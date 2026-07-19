"""Contracts for cleaning duplicate walls in completed pipeline outputs."""
from __future__ import annotations

import json
from pathlib import Path

from bim_recon.wall_cleanup import clean_saved_wall_list


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_clean_saved_wall_list_remaps_verified_and_report_hosts(tmp_path):
    _write_json(tmp_path / "wall_lines_snapped.json", [
        {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0},
        {"x1": 0.02, "y1": 0.01, "x2": 4.02, "y2": 0.01},
        {"x1": 4.0, "y1": 0.0, "x2": 4.0, "y2": 3.0},
    ])
    _write_json(tmp_path / "doors_verified.json", {
        "results": [{"candidate": {"wall_idx": 2}}],
    })
    _write_json(tmp_path / "windows_verified.json", {
        "results": [{"candidate": {"wall_idx": 1}}],
    })
    _write_json(tmp_path / "pipeline_report.json", {
        "walls": {"count": 3},
        "elements": {"door": {"results": [{"candidate": {"wall_idx": 2}}]}},
    })

    summary = clean_saved_wall_list(tmp_path)

    assert summary == {
        "input_walls": 3,
        "removed_walls": 1,
        "output_walls": 2,
        "rewritten_files": 3,
    }
    walls = json.loads((tmp_path / "wall_lines_snapped.json").read_text("utf-8"))
    assert len(walls) == 2
    assert json.loads((tmp_path / "doors_verified.json").read_text("utf-8"))["results"][0]["candidate"]["wall_idx"] == 1
    assert json.loads((tmp_path / "windows_verified.json").read_text("utf-8"))["results"][0]["candidate"]["wall_idx"] == 0
    report = json.loads((tmp_path / "pipeline_report.json").read_text("utf-8"))
    assert report["walls"]["count"] == 2
    assert report["elements"]["door"]["results"][0]["candidate"]["wall_idx"] == 1
