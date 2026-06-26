"""FloorPlan contract + ManualProvider.

This is the decoupled "horizontal base map" abstraction (PLAN.md section 4.1).
The core reconstruction pipeline consumes only :class:`FloorPlan` and is
agnostic to where the plan came from.

Contract: PLAN.md Appendix A.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# --- enums ------------------------------------------------------------------

class WallType(Enum):
    BEARING = "bearing"      # load-bearing wall, default thickness 0.24 m
    PARTITION = "partition"  # partition wall, default thickness 0.12 m
    UNKNOWN = "unknown"      # resolved like BEARING when thickness is absent

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
    def parse(cls, raw: str) -> "OpeningKind":
        return cls(raw.strip().lower())


# --- default thicknesses (m) ------------------------------------------------

_DEFAULT_THICKNESS = {
    WallType.BEARING: 0.24,
    WallType.PARTITION: 0.12,
    WallType.UNKNOWN: 0.24,
}


# --- contract dataclasses ---------------------------------------------------

@dataclass
class WallSegment:
    """A straight wall segment in the horizontal (XY) plane, metric units."""
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: Optional[float] = None  # m; if None, resolved from `type`
    type: WallType = WallType.UNKNOWN

    def resolved_thickness(self) -> float:
        """Effective wall thickness in metres."""
        if self.thickness is not None:
            return self.thickness
        return _DEFAULT_THICKNESS[self.type]


@dataclass
class Opening:
    """A door or window hole in a wall, referenced by wall index."""
    wall_index: int
    offset: float       # distance along the wall segment, in metres
    width: float        # opening width, in metres
    kind: OpeningKind = OpeningKind.DOOR
    sill_height: Optional[float] = None  # m above floor; None for doors


@dataclass
class FrameMeta:
    """Metadata describing the reference frame the plan lives in."""
    scale_known: bool = True                # is the plan already metric?
    orientation_known: bool = False         # is plan north known?
    gravity_axis: tuple = (0.0, 0.0, 1.0)   # world up (from IMU)


@dataclass
class FloorPlan:
    """The unified horizontal base map consumed by the core pipeline."""
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    meta: FrameMeta = field(default_factory=FrameMeta)


# --- provider interface + ManualProvider ------------------------------------

class FloorPlanProvider:
    """Base class for all base-map sources. Core pipeline is provider-agnostic."""

    def get_floorplan(self) -> FloorPlan:
        raise NotImplementedError


class ManualProvider(FloorPlanProvider):
    """Base-map provider fed by hand measurement (rectangular room or JSON).

    Zero hardware. Enables the whole pipeline to run before the LiDAR
    provider exists.
    """

    def __init__(self, floorplan: FloorPlan) -> None:
        self._fp = floorplan

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_rectangle(
        cls,
        room_width: float,
        room_depth: float,
        wall_type: WallType = WallType.BEARING,
        thickness: Optional[float] = None,
    ) -> "ManualProvider":
        """Build a closed rectangular room (4 walls) around the origin."""
        W, D = float(room_width), float(room_depth)
        pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        walls = [
            WallSegment(
                x1=pts[i][0], y1=pts[i][1],
                x2=pts[(i + 1) % 4][0], y2=pts[(i + 1) % 4][1],
                thickness=thickness, type=wall_type,
            )
            for i in range(4)
        ]
        return cls(FloorPlan(walls=walls, meta=FrameMeta(scale_known=True)))

    @classmethod
    def from_dict(cls, data: dict) -> "ManualProvider":
        """Parse a JSON-like dict into a validated :class:`FloorPlan`."""
        walls = [
            WallSegment(
                x1=float(w["x1"]), y1=float(w["y1"]),
                x2=float(w["x2"]), y2=float(w["y2"]),
                thickness=w.get("thickness"),
                type=WallType.parse(w.get("type")),
            )
            for w in data.get("walls", [])
        ]
        openings = []
        for o in data.get("openings", []):
            idx = int(o["wall_index"])
            if not (0 <= idx < len(walls)):
                raise ValueError(
                    f"opening references wall_index {idx}, but only "
                    f"{len(walls)} walls exist"
                )
            openings.append(
                Opening(
                    wall_index=idx,
                    offset=float(o["offset"]),
                    width=float(o["width"]),
                    kind=OpeningKind.parse(o["kind"]),
                    sill_height=o.get("sill_height"),
                )
            )
        meta_data = data.get("meta", {}) or {}
        meta = FrameMeta(
            scale_known=bool(meta_data.get("scale_known", True)),
            orientation_known=bool(meta_data.get("orientation_known", False)),
        )
        return cls(FloorPlan(walls=walls, openings=openings, meta=meta))

    @classmethod
    def from_json(cls, path: str | Path) -> "ManualProvider":
        """Load a base map from a JSON file on disk."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    # -- provider contract --------------------------------------------------

    def get_floorplan(self) -> FloorPlan:
        return self._fp
