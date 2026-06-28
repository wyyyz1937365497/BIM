"""FloorPlan 契约与 ManualProvider。

这是 PLAN.md 中“水平底图 Provider”的最小稳定接口。核心重建链路只消费
FloorPlan，不关心底图来自手量、LiDAR、图纸还是未来的其他 Provider。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from pathlib import Path
from typing import Any, Optional


class WallType(Enum):
    BEARING = "bearing"      # 承重墙，默认厚 0.24 m
    PARTITION = "partition"  # 隔墙，默认厚 0.12 m
    UNKNOWN = "unknown"      # 类型未知时按承重墙厚度兜底

    @classmethod
    def parse(cls, raw: Optional[str]) -> "WallType":
        if raw is None:
            return cls.UNKNOWN
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.UNKNOWN


class OpeningKind(Enum):
    DOOR = "door"
    WINDOW = "window"

    @classmethod
    def parse(cls, raw: Optional[str]) -> "OpeningKind":
        if raw is None:
            return cls.DOOR
        return cls(raw.strip().lower())


_DEFAULT_THICKNESS = {
    WallType.BEARING: 0.24,
    WallType.PARTITION: 0.12,
    WallType.UNKNOWN: 0.24,
}


@dataclass
class WallSegment:
    """水平 XY 平面内的一段直墙，单位为米。"""
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: Optional[float] = None
    type: WallType = WallType.UNKNOWN

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)
        self.x2 = float(self.x2)
        self.y2 = float(self.y2)
        if self.thickness is not None:
            self.thickness = float(self.thickness)
        validate_wall_segment(self)

    def length(self) -> float:
        return hypot(self.x2 - self.x1, self.y2 - self.y1)

    def resolved_thickness(self) -> float:
        """返回实际墙厚，单位米。"""
        if self.thickness is not None:
            return self.thickness
        return _DEFAULT_THICKNESS[self.type]


@dataclass
class Opening:
    """墙上的门窗洞口，通过 wall_index 引用对应墙段。"""
    wall_index: int
    offset: float
    width: float
    kind: OpeningKind = OpeningKind.DOOR
    sill_height: Optional[float] = None

    def __post_init__(self) -> None:
        self.wall_index = int(self.wall_index)
        self.offset = float(self.offset)
        self.width = float(self.width)
        if self.sill_height is not None:
            self.sill_height = float(self.sill_height)
        validate_opening_shape(self)


@dataclass
class FrameMeta:
    """底图坐标系元数据。"""
    scale_known: bool = True
    orientation_known: bool = False
    gravity_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.gravity_axis) != 3:
            raise ValueError("gravity_axis must contain exactly 3 numbers")
        self.gravity_axis = tuple(float(v) for v in self.gravity_axis)


@dataclass
class FloorPlan:
    """核心流水线消费的统一水平底图。"""
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    meta: FrameMeta = field(default_factory=FrameMeta)

    def __post_init__(self) -> None:
        validate_floorplan(self)


class FloorPlanProvider:
    """所有底图来源的统一接口。"""

    def get_floorplan(self) -> FloorPlan:
        raise NotImplementedError


class ManualProvider(FloorPlanProvider):
    """手量底图 Provider。

    支持两类 JSON：
    1. 直接给 walls/openings；
    2. 给 rectangle.width/rectangle.depth，再用 openings 描述门窗。
    """

    def __init__(self, floorplan: FloorPlan) -> None:
        validate_floorplan(floorplan)
        self._floorplan = floorplan

    @classmethod
    def from_rectangle(
        cls,
        room_width: float,
        room_depth: float,
        wall_type: WallType = WallType.BEARING,
        thickness: Optional[float] = None,
    ) -> "ManualProvider":
        """构造以 (0,0) 为左下角的闭合矩形房间。"""
        width = float(room_width)
        depth = float(room_depth)
        if width <= 0 or depth <= 0:
            raise ValueError("room_width and room_depth must be positive")
        points = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
        walls = [
            WallSegment(
                x1=points[i][0], y1=points[i][1],
                x2=points[(i + 1) % 4][0], y2=points[(i + 1) % 4][1],
                thickness=thickness, type=wall_type,
            )
            for i in range(4)
        ]
        return cls(FloorPlan(walls=walls, meta=FrameMeta(scale_known=True)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManualProvider":
        """从 JSON-like dict 解析并校验 FloorPlan。"""
        meta = _parse_meta(data.get("meta", {}) or {})
        walls = _parse_walls(data)
        openings = [_parse_opening(opening) for opening in data.get("openings", [])]
        return cls(FloorPlan(walls=walls, openings=openings, meta=meta))

    @classmethod
    def from_json(cls, path: str | Path) -> "ManualProvider":
        """从磁盘 JSON 文件读取手量底图。"""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def get_floorplan(self) -> FloorPlan:
        return self._floorplan


def validate_wall_segment(wall: WallSegment) -> None:
    if hypot(wall.x2 - wall.x1, wall.y2 - wall.y1) <= 0:
        raise ValueError("wall segment length must be positive")
    if wall.thickness is not None and wall.thickness <= 0:
        raise ValueError("wall thickness must be positive")


def validate_opening_shape(opening: Opening) -> None:
    if opening.wall_index < 0:
        raise ValueError("opening wall_index must be non-negative")
    if opening.offset < 0:
        raise ValueError("opening offset must be non-negative")
    if opening.width <= 0:
        raise ValueError("opening width must be positive")
    if opening.sill_height is not None and opening.sill_height < 0:
        raise ValueError("opening sill_height must be non-negative")
    if opening.kind is OpeningKind.DOOR and opening.sill_height is not None:
        raise ValueError("door sill_height must be omitted")
    if opening.kind is OpeningKind.WINDOW and opening.sill_height is None:
        raise ValueError("window sill_height is required")


def validate_floorplan(floorplan: FloorPlan) -> None:
    for index, wall in enumerate(floorplan.walls):
        validate_wall_segment(wall)
        if wall.resolved_thickness() <= 0:
            raise ValueError(f"wall {index} resolved thickness must be positive")
    for opening in floorplan.openings:
        validate_opening_shape(opening)
        if opening.wall_index >= len(floorplan.walls):
            raise ValueError(
                f"opening references wall_index {opening.wall_index}, "
                f"but only {len(floorplan.walls)} walls exist"
            )
        wall = floorplan.walls[opening.wall_index]
        if opening.offset + opening.width > wall.length():
            raise ValueError(
                f"opening on wall {opening.wall_index} exceeds wall length "
                f"({opening.offset + opening.width:.3f}m > {wall.length():.3f}m)"
            )


def floorplan_to_dict(floorplan: FloorPlan) -> dict[str, Any]:
    """把 FloorPlan 转成可序列化 dict，供 CLI、MCP 和测试复用。"""
    return {
        "walls": [
            {
                "x1": wall.x1,
                "y1": wall.y1,
                "x2": wall.x2,
                "y2": wall.y2,
                "thickness": wall.thickness,
                "type": wall.type.value,
            }
            for wall in floorplan.walls
        ],
        "openings": [
            {
                "wall_index": opening.wall_index,
                "offset": opening.offset,
                "width": opening.width,
                "kind": opening.kind.value,
                "sill_height": opening.sill_height,
            }
            for opening in floorplan.openings
        ],
        "meta": {
            "scale_known": floorplan.meta.scale_known,
            "orientation_known": floorplan.meta.orientation_known,
            "gravity_axis": list(floorplan.meta.gravity_axis),
        },
    }


def _parse_meta(meta_data: dict[str, Any]) -> FrameMeta:
    return FrameMeta(
        scale_known=bool(meta_data.get("scale_known", True)),
        orientation_known=bool(meta_data.get("orientation_known", False)),
        gravity_axis=tuple(meta_data.get("gravity_axis", (0.0, 0.0, 1.0))),
    )


def _parse_walls(data: dict[str, Any]) -> list[WallSegment]:
    if "rectangle" in data:
        rectangle = data["rectangle"]
        provider = ManualProvider.from_rectangle(
            room_width=float(rectangle["width"]),
            room_depth=float(rectangle["depth"]),
            wall_type=WallType.parse(rectangle.get("wall_type")),
            thickness=rectangle.get("thickness"),
        )
        return provider.get_floorplan().walls

    return [
        WallSegment(
            x1=wall["x1"], y1=wall["y1"],
            x2=wall["x2"], y2=wall["y2"],
            thickness=wall.get("thickness"),
            type=WallType.parse(wall.get("type")),
        )
        for wall in data.get("walls", [])
    ]


def _parse_opening(data: dict[str, Any]) -> Opening:
    return Opening(
        wall_index=data["wall_index"],
        offset=data["offset"],
        width=data["width"],
        kind=OpeningKind.parse(data.get("kind")),
        sill_height=data.get("sill_height"),
    )
