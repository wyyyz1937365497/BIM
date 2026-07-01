"""Unit tests for element_config — registry lookups and per-type defaults."""
from __future__ import annotations

import pytest

from bim_recon.element_config import (
    ELEMENT_CONFIGS,
    ElementConfig,
    get_element_config,
    list_element_types,
)


class TestGetElementConfig:
    def test_door_config(self):
        cfg = get_element_config("door")
        assert cfg.name == "door"
        assert cfg.class_idx == 3
        assert cfg.structural is True
        assert cfg.min_width == 0.7
        assert cfg.min_points == 100

    def test_window_config(self):
        cfg = get_element_config("window")
        assert cfg.name == "window"
        assert cfg.class_idx == 4
        assert cfg.structural is True
        assert cfg.min_width == 0.5

    def test_furniture_config(self):
        cfg = get_element_config("furniture")
        assert cfg.name == "furniture"
        assert cfg.class_idx == 8
        assert cfg.structural is False  # free-standing

    def test_column_config(self):
        cfg = get_element_config("column")
        assert cfg.name == "column"
        assert cfg.class_idx == 5
        assert cfg.structural is True

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_element_config("elevator")

    def test_case_sensitive(self):
        with pytest.raises(KeyError):
            get_element_config("Door")


class TestOutputNames:
    def test_door_output_json(self):
        cfg = get_element_config("door")
        assert cfg.output_json_name == "doors_verified.json"

    def test_window_verify_dir(self):
        cfg = get_element_config("window")
        assert cfg.verify_dir_name == "verify_window"

    def test_furniture_output_json(self):
        cfg = get_element_config("furniture")
        assert cfg.output_json_name == "furnitures_verified.json"


class TestListElementTypes:
    def test_includes_door_window_furniture(self):
        types = list_element_types()
        assert "door" in types
        assert "window" in types
        assert "furniture" in types

    def test_sorted(self):
        types = list_element_types()
        assert types == sorted(types)

    def test_all_have_configs(self):
        for t in list_element_types():
            cfg = get_element_config(t)
            assert cfg.name == t


class TestElementConfigFrozen:
    def test_frozen_dataclass(self):
        cfg = get_element_config("door")
        with pytest.raises(Exception):
            cfg.min_width = 999  # type: ignore
