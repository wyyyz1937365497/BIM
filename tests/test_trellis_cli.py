"""Tests for the TRELLIS mesh generation CLI glue."""
from __future__ import annotations

from pathlib import Path

from scripts.generate_trellis_mesh import parse_args


class TestGenerateTrellisMeshCli:
    def test_parse_required_paths(self, tmp_path):
        image = tmp_path / "input.png"
        out = tmp_path / "out"

        args = parse_args([
            "--image", str(image),
            "--output-dir", str(out),
            "--name", "chair",
        ])

        assert args.image == image
        assert args.output_dir == out
        assert args.name == "chair"
