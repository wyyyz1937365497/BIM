"""Tests for TRELLIS + element routing config parsing."""
from __future__ import annotations

import json

from bim_recon.config import load_config


class TestTrellisConfig:
    def test_defaults_preserve_existing_config_and_add_trellis(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")

        assert cfg.vlm.model == "gemma4:12b"
        assert cfg.trellis.host == "127.0.0.1"
        assert cfg.trellis.port == 18391
        assert cfg.trellis.model == "microsoft/TRELLIS-image-large"
        assert cfg.falcon.host == "127.0.0.1"
        assert cfg.falcon.port == 18390

    def test_loads_trellis_section(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "trellis": {
                    "host": "0.0.0.0",
                    "port": 19000,
                    "model": "G:/TJ/BIM/TRELLIS/TRELLIS-image-large",
                    "timeout": 1200,
                }
            }),
            encoding="utf-8",
        )

        cfg = load_config(path)

        assert cfg.trellis.host == "0.0.0.0"
        assert cfg.trellis.port == 19000
        assert cfg.trellis.model == "G:/TJ/BIM/TRELLIS/TRELLIS-image-large"
        assert cfg.trellis.timeout == 1200

    def test_loads_falcon_section(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "falcon": {"host": "0.0.0.0", "port": 19001, "timeout": 45},
            }),
            encoding="utf-8",
        )

        cfg = load_config(path)

        assert cfg.falcon.host == "0.0.0.0"
        assert cfg.falcon.port == 19001
        assert cfg.falcon.timeout == 45


class TestWorkflowConfig:
    def test_vlm_config_remains_available_for_pipeline_verification(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "vlm": {
                    "provider": "openai",
                    "api_base": "https://example.invalid/v1",
                    "model": "glm-5v-turbo",
                    "api_key": "test-key",
                },
            }),
            encoding="utf-8",
        )

        cfg = load_config(path)

        assert cfg.vlm.model == "glm-5v-turbo"
        assert not hasattr(cfg, "llm")

    def test_revit_mcp_timeout_is_typed(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "revit_mcp": {
                    "command": "node",
                    "args": ["server.js"],
                    "timeout": 300,
                },
            }),
            encoding="utf-8",
        )

        cfg = load_config(path)

        assert cfg.revit_mcp.args == ["server.js"]
        assert cfg.revit_mcp.timeout == 300


class TestElementRouting:
    def test_defaults_route_door_window_to_a_furniture_to_b(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")
        r = cfg.element_routing

        assert r.get_route("door") == "A"
        assert r.get_route("window") == "A"
        assert r.get_route("column") == "A"
        assert r.get_route("furniture") == "B"

    def test_is_b_class(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")
        r = cfg.element_routing

        assert r.is_b_class("furniture") is True
        assert r.is_b_class("door") is False

    def test_unknown_type_defaults_to_a(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")
        assert cfg.element_routing.get_route("stairs") == "A"

    def test_custom_routing_from_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({
                "element_routing": {
                    "door": "A",
                    "window": "A",
                    "column": "B",
                    "furniture": "B",
                    "stairs": "B",
                }
            }),
            encoding="utf-8",
        )
        cfg = load_config(path)
        r = cfg.element_routing

        assert r.get_route("door") == "A"
        assert r.get_route("column") == "B"
        assert r.get_route("stairs") == "B"
        assert r.is_b_class("column") is True
        assert sorted(r.b_class_types()) == ["column", "furniture", "stairs"]
        assert sorted(r.a_class_types()) == ["door", "window"]

    def test_b_class_types_returns_sorted(self, tmp_path):
        cfg = load_config(tmp_path / "missing.json")
        types = cfg.element_routing.b_class_types()
        assert types == sorted(types)
        assert "furniture" in types
