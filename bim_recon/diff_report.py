"""底图墙线与 3DGS/VLM 检出墙线的差异报告。

MVP 策略是“仅报告，不自动采纳”。这个模块只输出结构化差异，不修改
FloorPlan，也不替用户做自动补墙决策。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot

from .floorplan import FloorPlan, WallSegment


@dataclass(frozen=True)
class WallDiff:
    """一条未匹配墙线的报告项。"""

    source: str
    wall_index: int
    reason: str
    wall: WallSegment
    nearest_distance: float | None = None


@dataclass(frozen=True)
class DiffReport:
    """差异报告结果。"""

    unmatched_floorplan_walls: list[WallDiff]
    unmatched_detected_walls: list[WallDiff]

    @property
    def has_diff(self) -> bool:
        return bool(self.unmatched_floorplan_walls or self.unmatched_detected_walls)


def report_wall_differences(
    floorplan: FloorPlan,
    detected_walls: list[WallSegment],
    max_midpoint_distance: float = 0.25,
    max_angle_degrees: float = 8.0,
) -> DiffReport:
    """对比底图墙线和 3DGS/VLM 检出墙线。

    匹配标准采用保守近似：中点距离足够近，且墙线方向夹角足够小。该标准
    适合 MVP 阶段输出人工复核报告，不适合直接做自动几何合并。
    """
    matched_floorplan = set()
    matched_detected = set()

    for floorplan_index, floorplan_wall in enumerate(floorplan.walls):
        best_index, best_distance = _best_match(
            floorplan_wall,
            detected_walls,
            max_midpoint_distance,
            max_angle_degrees,
        )
        if best_index is not None:
            matched_floorplan.add(floorplan_index)
            matched_detected.add(best_index)

    unmatched_floorplan = [
        WallDiff(
            source="floorplan",
            wall_index=index,
            reason="底图中存在，但 3DGS/VLM 检出结果未匹配",
            wall=wall,
            nearest_distance=_nearest_midpoint_distance(wall, detected_walls),
        )
        for index, wall in enumerate(floorplan.walls)
        if index not in matched_floorplan
    ]
    unmatched_detected = [
        WallDiff(
            source="detected",
            wall_index=index,
            reason="3DGS/VLM 检出墙线存在，但底图未匹配；MVP 仅报告不自动采纳",
            wall=wall,
            nearest_distance=_nearest_midpoint_distance(wall, floorplan.walls),
        )
        for index, wall in enumerate(detected_walls)
        if index not in matched_detected
    ]
    return DiffReport(unmatched_floorplan, unmatched_detected)


def _best_match(
    wall: WallSegment,
    candidates: list[WallSegment],
    max_midpoint_distance: float,
    max_angle_degrees: float,
) -> tuple[int | None, float | None]:
    best_index = None
    best_distance = None
    for index, candidate in enumerate(candidates):
        angle = _undirected_angle_degrees(wall, candidate)
        distance = _midpoint_distance(wall, candidate)
        if angle <= max_angle_degrees and distance <= max_midpoint_distance:
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
    return best_index, best_distance


def _nearest_midpoint_distance(
    wall: WallSegment,
    candidates: list[WallSegment],
) -> float | None:
    if not candidates:
        return None
    return min(_midpoint_distance(wall, candidate) for candidate in candidates)


def _midpoint_distance(a: WallSegment, b: WallSegment) -> float:
    ax, ay = ((a.x1 + a.x2) / 2.0, (a.y1 + a.y2) / 2.0)
    bx, by = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
    return hypot(ax - bx, ay - by)


def _undirected_angle_degrees(a: WallSegment, b: WallSegment) -> float:
    avx, avy = (a.x2 - a.x1, a.y2 - a.y1)
    bvx, bvy = (b.x2 - b.x1, b.y2 - b.y1)
    dot = avx * bvx + avy * bvy
    norm = hypot(avx, avy) * hypot(bvx, bvy)
    cosine = max(-1.0, min(1.0, abs(dot) / norm))
    return degrees(acos(cosine))
