"""Tests for FloorPlan contract + ManualProvider.

P0 Week-1 deliverable: the decoupled horizontal-base-map abstraction.
The contract is defined in PLAN.md Appendix A.
"""
import json
from pathlib import Path

import pytest

from bim_recon.floorplan import (
    FloorPlan,
    FrameMeta,
    ManualProvider,
    Opening,
    OpeningKind,
    WallSegment,
    WallType,
)


# --- contract / dataclass behaviour -----------------------------------------

def test_wall_segment_defaults_thickness_by_type():
    w = WallSegment(x1=0, y1=0, x2=5, y2=0, type=WallType.BEARING)
    assert w.resolved_thickness() == pytest.approx(0.24)

    w2 = WallSegment(x1=0, y1=0, x2=5, y2=0, type=WallType.PARTITION)
    assert w2.resolved_thickness() == pytest.approx(0.12)

    # explicit thickness overrides the type default
    w3 = WallSegment(x1=0, y1=0, x2=5, y2=0, thickness=0.30, type=WallType.BEARING)
    assert w3.resolved_thickness() == pytest.approx(0.30)


def test_opening_kind_and_sill_height():
    door = Opening(wall_index=0, offset=1.0, width=1.0, kind=OpeningKind.DOOR)
    assert door.sill_height is None  # doors sit on the floor

    window = Opening(
        wall_index=0, offset=1.5, width=1.2,
        kind=OpeningKind.WINDOW, sill_height=1.0,
    )
    assert window.sill_height == pytest.approx(1.0)


# --- ManualProvider: rectangle convenience ----------------------------------

def test_manual_provider_rectangle_builds_four_walls():
    fp = ManualProvider.from_rectangle(room_width=5.0, room_depth=4.0).get_floorplan()
    assert isinstance(fp, FloorPlan)
    assert len(fp.walls) == 4
    # closed loop: each wall's start == previous wall's end
    for i in range(4):
        cur = fp.walls[i]
        nxt = fp.walls[(i + 1) % 4]
        assert (cur.x2, cur.y2) == (nxt.x1, nxt.y1)
    assert fp.openings == []
    assert fp.meta.scale_known is True  # manual is metric by definition


# --- ManualProvider: JSON parsing -------------------------------------------

def test_manual_provider_from_dict_with_openings(tmp_path: Path):
    data = {
        "walls": [
            {"x1": 0, "y1": 0, "x2": 5, "y2": 0, "type": "bearing"},
            {"x1": 5, "y1": 0, "x2": 5, "y2": 4, "type": "partition"},
            {"x1": 5, "y1": 4, "x2": 0, "y2": 4, "type": "bearing"},
            {"x1": 0, "y1": 4, "x2": 0, "y2": 0, "type": "partition"},
        ],
        "openings": [
            {"wall_index": 0, "offset": 1.0, "width": 1.0, "kind": "door"},
            {"wall_index": 1, "offset": 1.5, "width": 1.2,
             "kind": "window", "sill_height": 1.0},
        ],
    }
    fp = ManualProvider.from_dict(data).get_floorplan()

    assert len(fp.walls) == 4
    assert fp.walls[0].type is WallType.BEARING
    assert len(fp.openings) == 2
    assert fp.openings[0].kind is OpeningKind.DOOR
    assert fp.openings[0].sill_height is None
    assert fp.openings[1].kind is OpeningKind.WINDOW
    assert fp.openings[1].sill_height == pytest.approx(1.0)


def test_manual_provider_reads_json_file(tmp_path: Path):
    payload = {
        "walls": [
            {"x1": 0, "y1": 0, "x2": 3, "y2": 0},
            {"x1": 3, "y1": 0, "x2": 3, "y2": 3},
        ],
        "openings": [],
    }
    p = tmp_path / "room.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    fp = ManualProvider.from_json(p).get_floorplan()
    assert len(fp.walls) == 2
    # default wall type when omitted
    assert fp.walls[0].type is WallType.UNKNOWN
    assert fp.walls[0].resolved_thickness() == pytest.approx(0.24)  # unknown -> bearing default


def test_manual_provider_rejects_bad_wall_index():
    data = {
        "walls": [{"x1": 0, "y1": 0, "x2": 5, "y2": 0}],
        "openings": [{"wall_index": 5, "offset": 1.0, "width": 1.0, "kind": "door"}],
    }
    with pytest.raises(ValueError):
        ManualProvider.from_dict(data)
