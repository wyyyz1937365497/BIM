"""Tests for TRELLIS config parsing."""
from __future__ import annotations

import json

from bim_recon.config import load_config


class TestTrellisConfig:
    def test_defaults_preserve_existing_config_and_add_trellis(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")

        assert cfg.vlm.model == "gemma4:12b"
        assert cfg.llm.model == "qwen2.5:32b"
        assert cfg.trellis.host == "127.0.0.1"
        assert cfg.trellis.port == 8391
        assert cfg.trellis.model == "microsoft/TRELLIS-image-large"

    def test_loads_trellis_section(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "trellis": {
                    "host": "0.0.0.0",
                    "port": 9000,
                    "model": "G:/TJ/BIM/TRELLIS/TRELLIS-image-large",
                    "timeout": 1200,
                }
            }),
            encoding="utf-8",
        )

        cfg = load_config(path)

        assert cfg.trellis.host == "0.0.0.0"
        assert cfg.trellis.port == 9000
        assert cfg.trellis.model == "G:/TJ/BIM/TRELLIS/TRELLIS-image-large"
        assert cfg.trellis.timeout == 1200
