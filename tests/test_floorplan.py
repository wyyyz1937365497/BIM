"""FloorPlan、ManualProvider 与 P0 代码生成测试。"""
import json
from pathlib import Path

import pytest

from bim_recon.diff_report import report_wall_differences
from bim_recon.floorplan import (
    FloorPlan,
    FrameMeta,
    ManualProvider,
    Opening,
    OpeningKind,
    WallSegment,
    WallType,
    floorplan_to_dict,
)
from bim_recon.gs_mcp_scaffold import BackendNotConfiguredError, GsMcpToolFacade
from bim_recon.revit_code import generate_revit_csharp


def test_wall_segment_defaults_thickness_by_type():
    w = WallSegment(x1=0, y1=0, x2=5, y2=0, type=WallType.BEARING)
    assert w.resolved_thickness() == pytest.approx(0.24)
    assert w.length() == pytest.approx(5.0)

    w2 = WallSegment(x1=0, y1=0, x2=5, y2=0, type=WallType.PARTITION)
    assert w2.resolved_thickness() == pytest.approx(0.12)

    # 显式墙厚优先级高于类型默认值。
    w3 = WallSegment(x1=0, y1=0, x2=5, y2=0, thickness=0.30, type=WallType.BEARING)
    assert w3.resolved_thickness() == pytest.approx(0.30)


def test_wall_segment_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        WallSegment(x1=0, y1=0, x2=0, y2=0)
    with pytest.raises(ValueError):
        WallSegment(x1=0, y1=0, x2=1, y2=0, thickness=-0.1)


def test_opening_kind_and_sill_height():
    door = Opening(wall_index=0, offset=1.0, width=1.0, kind=OpeningKind.DOOR)
    assert door.sill_height is None

    window = Opening(
        wall_index=0, offset=1.5, width=1.2,
        kind=OpeningKind.WINDOW, sill_height=1.0,
    )
    assert window.sill_height == pytest.approx(1.0)


def test_opening_rejects_invalid_shape():
    with pytest.raises(ValueError):
        Opening(wall_index=0, offset=0, width=0)
    with pytest.raises(ValueError):
        Opening(wall_index=0, offset=0, width=1, kind=OpeningKind.DOOR, sill_height=0.1)
    with pytest.raises(ValueError):
        Opening(wall_index=0, offset=0, width=1, kind=OpeningKind.WINDOW)


def test_manual_provider_rectangle_builds_four_walls():
    fp = ManualProvider.from_rectangle(room_width=5.0, room_depth=4.0).get_floorplan()
    assert isinstance(fp, FloorPlan)
    assert len(fp.walls) == 4
    # 闭合矩形：每段终点等于下一段起点。
    for i in range(4):
        cur = fp.walls[i]
        nxt = fp.walls[(i + 1) % 4]
        assert (cur.x2, cur.y2) == (nxt.x1, nxt.y1)
    assert fp.openings == []
    assert fp.meta.scale_known is True


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


def test_manual_provider_from_rectangle_json():
    data = {
        "rectangle": {"width": 5, "depth": 4, "wall_type": "partition"},
        "openings": [
            {"wall_index": 0, "offset": 1.0, "width": 0.9, "kind": "door"},
        ],
        "meta": {"gravity_axis": [0, 0, 1]},
    }
    fp = ManualProvider.from_dict(data).get_floorplan()
    assert len(fp.walls) == 4
    assert fp.walls[0].type is WallType.PARTITION
    assert fp.openings[0].width == pytest.approx(0.9)
    assert fp.meta.gravity_axis == (0.0, 0.0, 1.0)


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
    assert fp.walls[0].type is WallType.UNKNOWN
    assert fp.walls[0].resolved_thickness() == pytest.approx(0.24)


def test_manual_provider_rejects_bad_wall_index():
    data = {
        "walls": [{"x1": 0, "y1": 0, "x2": 5, "y2": 0}],
        "openings": [{"wall_index": 5, "offset": 1.0, "width": 1.0, "kind": "door"}],
    }
    with pytest.raises(ValueError):
        ManualProvider.from_dict(data)


def test_manual_provider_rejects_opening_exceeding_wall_length():
    data = {
        "walls": [{"x1": 0, "y1": 0, "x2": 2, "y2": 0}],
        "openings": [{"wall_index": 0, "offset": 1.5, "width": 1.0, "kind": "door"}],
    }
    with pytest.raises(ValueError):
        ManualProvider.from_dict(data)


def test_floorplan_to_dict_is_serializable():
    fp = ManualProvider.from_rectangle(5, 4).get_floorplan()
    payload = floorplan_to_dict(fp)
    assert payload["walls"][0]["type"] == "bearing"
    assert payload["meta"]["gravity_axis"] == [0.0, 0.0, 1.0]


def test_revit_csharp_generation_contains_native_api_calls():
    fp = ManualProvider.from_dict(
        {
            "rectangle": {"width": 5, "depth": 4},
            "openings": [{"wall_index": 0, "offset": 1, "width": 0.9, "kind": "door"}],
        }
    ).get_floorplan()
    code = generate_revit_csharp(fp)
    assert "Wall.Create" in code
    assert "Floor.Create" in code
    assert "NewOpening" in code
    assert "OST_Doors" in code
    assert "MetersToFeet" in code


def test_diff_report_only_reports_extra_detected_wall():
    fp = ManualProvider.from_rectangle(5, 4).get_floorplan()
    detected = list(fp.walls) + [WallSegment(x1=2, y1=2, x2=3, y2=2)]
    report = report_wall_differences(fp, detected)
    assert report.has_diff is True
    assert report.unmatched_floorplan_walls == []
    assert len(report.unmatched_detected_walls) == 1
    assert "仅报告" in report.unmatched_detected_walls[0].reason


def test_gs_mcp_facade_requires_backend():
    facade = GsMcpToolFacade()
    assert [tool.name for tool in facade.list_tool_specs()] == [
        "render_from_pose",
        "get_depth",
        "select_cluster",
        "report_diff",
    ]
    with pytest.raises(BackendNotConfiguredError):
        facade.render_from_pose([[1, 0, 0, 0]], width=64, height=64)
