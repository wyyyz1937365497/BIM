"""Deterministic A-class BIM creation workflow for Revit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from bim_recon.element_merger import clip_opening_to_wall
from bim_recon.mcp_gateway import ToolGateway, response_items
from bim_recon.pipeline_api import PipelineResults
from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
    WorkflowWarning,
)


MIN_REVIT_CURVE_M = 0.001
MIN_WALL_LENGTH_M = 0.01
MIN_FLOOR_EDGE_M = 0.01
MIN_OPENING_WIDTH_M = 0.1


class _CreateLevel(Event):
    pass


class _CreateFloor(Event):
    pass


class _CreateWalls(Event):
    pass


class _CreateOpenings(Event):
    pass


class _VerifyCreated(Event):
    pass


@dataclass(frozen=True, slots=True)
class RevitBuildOptions:
    """Explicit Revit creation settings; every length is in millimetres."""

    level_name: str = "BIM-Recon Level 1"
    level_elevation: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    wall_thickness: float = 200.0
    floor_thickness: float = 200.0
    wall_type_id: int | None = None
    floor_type_id: int | None = None
    door_type_id: int = 94654
    window_type_id: int = 93304
    create_floor: bool = True


@dataclass
class _RevitRuntime:
    results: PipelineResults
    options: RevitBuildOptions
    created: dict[str, list[int]] = field(default_factory=lambda: {
        "levels": [], "floors": [], "walls": [], "doors": [], "windows": [],
    })
    wall_id_by_index: dict[int, int] = field(default_factory=dict)
    floor_boundary: list[tuple[float, float]] = field(default_factory=list)
    valid_walls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)


class RevitBuildWorkflow(Workflow):
    """Create levels, floor, walls and hosted openings using fixed MCP calls."""

    workflow_name = "revit_build"

    def __init__(
        self,
        results: PipelineResults,
        gateway: ToolGateway,
        options: RevitBuildOptions | None = None,
    ):
        super().__init__(timeout=None, verbose=False)
        self.gateway = gateway
        self.state = _RevitRuntime(
            results=results,
            options=options or RevitBuildOptions(),
        )

    def _emit(self, ctx: Context, stage: str, message: str) -> None:
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage=stage,
            message=message,
        ))

    @step
    async def prepare(
        self, ctx: Context, ev: StartEvent,
    ) -> _CreateLevel | StopEvent:
        rt = self.state
        rt.valid_walls = [
            (index, wall)
            for index, wall in enumerate(rt.results.walls)
            if _wall_length_m(wall) >= MIN_WALL_LENGTH_M
        ]
        if not rt.valid_walls:
            message = "No non-degenerate reconstructed walls are available for Revit creation"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="prepare",
                message=message,
            ))
            return StopEvent(result={"error": message})
        skipped = len(rt.results.walls) - len(rt.valid_walls)
        if skipped:
            ctx.write_event_to_stream(WorkflowWarning(
                workflow=self.workflow_name,
                stage="prepare",
                message=f"Skipped {skipped} short wall segments",
            ))
        rt.floor_boundary = _ordered_boundary(rt.results.walls)
        confirmed = [element for element in rt.results.elements if element.confirmed]
        self._emit(
            ctx,
            "prepare",
            (
                f"Plan: 1 level, {len(rt.valid_walls)} walls, "
                f"{len(confirmed)} confirmed openings"
            ),
        )
        return _CreateLevel()

    @step
    async def create_level(
        self, ctx: Context, ev: _CreateLevel,
    ) -> _CreateFloor:
        options = self.state.options
        self._emit(ctx, "level", f"Creating {options.level_name}...")
        payload = await self.gateway.call_tool("create_level", {
            "data": [{
                "name": options.level_name,
                "elevation": options.level_elevation,
                "description": "Created by BIM reconstruction workflow",
                "isMainLevel": True,
                "isBuildingStory": True,
                "createFloorPlan": True,
                "createCeilingPlan": False,
            }],
        })
        self.state.created["levels"] = _extract_element_ids(payload)
        return _CreateFloor()

    @step
    async def create_floor(
        self, ctx: Context, ev: _CreateFloor,
    ) -> _CreateWalls:
        rt = self.state
        options = rt.options
        if not options.create_floor:
            return _CreateWalls()
        if len(rt.floor_boundary) < 3:
            ctx.write_event_to_stream(WorkflowWarning(
                workflow=self.workflow_name,
                stage="floor",
                message="Wall graph has no usable floor boundary; floor skipped",
            ))
            return _CreateWalls()
        points = [
            (
                x * 1000.0 + options.offset_x,
                y * 1000.0 + options.offset_y,
            )
            for x, y in rt.floor_boundary
        ]
        segments = []
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            segments.append({
                "p0": {"x": point[0], "y": point[1], "z": options.level_elevation},
                "p1": {"x": next_point[0], "y": next_point[1], "z": options.level_elevation},
            })
        floor: dict[str, Any] = {
            "name": "Reconstructed floor",
            "category": "OST_Floors",
            "boundary": {"outerLoop": segments},
            "thickness": options.floor_thickness,
            "baseLevel": options.level_elevation,
            "baseOffset": 0.0,
        }
        if options.floor_type_id is not None:
            floor["typeId"] = options.floor_type_id
        self._emit(ctx, "floor", f"Creating floor with {len(points)} vertices...")
        payload = await self.gateway.call_tool(
            "create_surface_based_element",
            {"data": [floor]},
        )
        rt.created["floors"] = _extract_element_ids(payload)
        return _CreateWalls()

    @step
    async def create_walls(
        self, ctx: Context, ev: _CreateWalls,
    ) -> _CreateOpenings:
        rt = self.state
        options = rt.options
        floor_z = float(rt.results.coords.get("floor_z", 0.0))
        ceiling_z = float(rt.results.coords.get("ceiling_z", 3.0))
        height_mm = max((ceiling_z - floor_z) * 1000.0, 1000.0)
        walls: list[dict[str, Any]] = []
        source_indices: list[int] = []
        for source_index, wall in rt.valid_walls:
            item: dict[str, Any] = {
                "category": "OST_Walls",
                "locationLine": {
                    "p0": {
                        "x": wall["x1"] * 1000.0 + options.offset_x,
                        "y": wall["y1"] * 1000.0 + options.offset_y,
                        "z": options.level_elevation,
                    },
                    "p1": {
                        "x": wall["x2"] * 1000.0 + options.offset_x,
                        "y": wall["y2"] * 1000.0 + options.offset_y,
                        "z": options.level_elevation,
                    },
                },
                "thickness": options.wall_thickness,
                "height": height_mm,
                "baseLevel": options.level_elevation,
                "baseOffset": 0.0,
            }
            if options.wall_type_id is not None:
                item["typeId"] = options.wall_type_id
            walls.append(item)
            source_indices.append(source_index)
        self._emit(ctx, "walls", f"Creating {len(walls)} walls...")
        payload = await self.gateway.call_tool(
            "create_line_based_element",
            {"data": walls},
        )
        wall_ids = _extract_element_ids(payload)
        rt.created["walls"] = wall_ids
        rt.wall_id_by_index = dict(zip(source_indices, wall_ids))
        if len(wall_ids) != len(walls):
            ctx.write_event_to_stream(WorkflowWarning(
                workflow=self.workflow_name,
                stage="walls",
                message=(
                    f"Revit returned {len(wall_ids)} wall IDs for "
                    f"{len(walls)} requested walls"
                ),
            ))
        return _CreateOpenings()

    @step
    async def create_openings(
        self, ctx: Context, ev: _CreateOpenings,
    ) -> _VerifyCreated:
        rt = self.state
        options = rt.options
        by_category: dict[str, list[dict[str, Any]]] = {
            "doors": [], "windows": [],
        }
        for index, element in enumerate(rt.results.elements):
            if not element.confirmed or element.element_class not in {"door", "window"}:
                continue
            if element.wall_idx is None or element.wall_idx not in rt.wall_id_by_index:
                ctx.write_event_to_stream(WorkflowWarning(
                    workflow=self.workflow_name,
                    stage="openings",
                    message=(
                        f"{element.element_class} #{index + 1} has no created host wall; skipped"
                    ),
                ))
                continue
            is_door = element.element_class == "door"
            key = "doors" if is_door else "windows"
            width_default = 750.0 if is_door else 915.0
            height_default = 2000.0 if is_door else 1220.0
            dimensions = element.height_detection or {}
            detected_width = float(
                dimensions.get("width_m", width_default / 1000.0)
            )
            sill_height = float(dimensions.get("sill_height", 0.0))
            detected_height = float(
                dimensions.get(
                    "element_height",
                    dimensions.get("header_height", sill_height) - sill_height,
                )
            )
            desired_width_m = max(detected_width, width_default / 1000.0)
            (
                opening_x,
                opening_y,
                clipped_width_m,
                _opening_start,
                _opening_end,
            ) = clip_opening_to_wall(
                element.world_x,
                element.world_y,
                desired_width_m,
                rt.results.walls[element.wall_idx],
            )
            if clipped_width_m < MIN_OPENING_WIDTH_M:
                ctx.write_event_to_stream(WorkflowWarning(
                    workflow=self.workflow_name,
                    stage="openings",
                    message=(
                        f"{element.element_class} #{index + 1} has less than "
                        f"{MIN_OPENING_WIDTH_M:.2f} m on its host wall; skipped"
                    ),
                ))
                continue
            width = clipped_width_m * 1000.0
            height = max(detected_height * 1000.0, height_default)
            base_offset = 0.0 if is_door else max(sill_height * 1000.0, 0.0)
            by_category[key].append({
                "name": f"Reconstructed {element.element_class} {index + 1}",
                "typeId": options.door_type_id if is_door else options.window_type_id,
                "locationPoint": {
                    "x": opening_x * 1000.0 + options.offset_x,
                    "y": opening_y * 1000.0 + options.offset_y,
                    "z": options.level_elevation,
                },
                "width": width,
                "height": height,
                "baseLevel": options.level_elevation,
                "baseOffset": base_offset,
                "hostWallId": rt.wall_id_by_index[element.wall_idx],
            })
        for key in ("doors", "windows"):
            items = by_category[key]
            if not items:
                continue
            self._emit(ctx, "openings", f"Creating {len(items)} {key}...")
            payload = await self.gateway.call_tool(
                "create_point_based_element",
                {"data": items},
            )
            rt.created[key] = _extract_element_ids(payload)
        return _VerifyCreated()

    @step
    async def verify_created(
        self, ctx: Context, ev: _VerifyCreated,
    ) -> StopEvent:
        rt = self.state
        expected_ids = {
            element_id
            for key in ("walls", "doors", "windows")
            for element_id in rt.created[key]
        }
        self._emit(ctx, "verify", "Verifying created Revit elements...")
        payload = await self.gateway.call_tool("get_current_view_elements", {
            "modelCategoryList": ["OST_Walls", "OST_Doors", "OST_Windows"],
            "annotationCategoryList": [],
            "includeHidden": True,
            "limit": max(len(expected_ids) + 100, 200),
        })
        observed_ids = set(_extract_element_ids(payload))
        missing = sorted(expected_ids - observed_ids) if observed_ids else []
        if missing:
            ctx.write_event_to_stream(WorkflowWarning(
                workflow=self.workflow_name,
                stage="verify",
                message=f"Created IDs not visible in active view: {missing}",
                payload={"missing_ids": missing},
            ))
        result = {
            "created": rt.created,
            "wall_id_by_index": rt.wall_id_by_index,
            "verified_ids": sorted(expected_ids & observed_ids),
            "missing_ids": missing,
        }
        ctx.write_event_to_stream(WorkflowCompleted(
            workflow=self.workflow_name,
            result=result,
        ))
        return StopEvent(result=result)


def _extract_element_ids(value: Any) -> list[int]:
    """Collect element IDs from Revit response envelopes without guessing order."""
    ids: list[int] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = key.replace("_", "").lower()
                if normalized in {"elementid", "id"}:
                    try:
                        ids.append(int(child))
                    except (TypeError, ValueError):
                        pass
                elif normalized == "elementids" and isinstance(child, list):
                    for item in child:
                        try:
                            ids.append(int(item))
                        except (TypeError, ValueError):
                            pass
                else:
                    visit(child)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    for item in response_items(value):
        if isinstance(item, int) and not isinstance(item, bool):
            ids.append(item)
        else:
            visit(item)
    if not ids:
        visit(value)
    return list(dict.fromkeys(ids))


def _wall_length_m(wall: dict[str, Any]) -> float:
    return math.hypot(
        float(wall["x2"]) - float(wall["x1"]),
        float(wall["y2"]) - float(wall["y1"]),
    )


def _sanitize_boundary(
    points: list[tuple[float, float]],
    min_edge_length: float = MIN_FLOOR_EDGE_M,
) -> list[tuple[float, float]]:
    """Remove near-duplicate vertices before converting the loop to Revit curves."""
    clean: list[tuple[float, float]] = []
    for point in points:
        if not clean or math.dist(clean[-1], point) >= min_edge_length:
            clean.append(point)
    if len(clean) > 1 and math.dist(clean[-1], clean[0]) < min_edge_length:
        clean.pop()
    if len(clean) < 3 or abs(_polygon_area(clean)) < min_edge_length ** 2:
        return []
    return clean


def _ordered_boundary(walls: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Recover a Revit-safe closed wall loop, falling back to a convex hull."""
    usable_walls = [
        wall for wall in walls
        if _wall_length_m(wall) >= MIN_REVIT_CURVE_M
    ]
    points = [
        (float(wall[key_x]), float(wall[key_y]))
        for wall in usable_walls
        for key_x, key_y in (("x1", "y1"), ("x2", "y2"))
    ]
    if len(points) < 3:
        return []

    def key(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], 4), round(point[1], 4)

    canonical = {key(point): point for point in points}
    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for wall in usable_walls:
        start = key((float(wall["x1"]), float(wall["y1"])))
        end = key((float(wall["x2"]), float(wall["y2"])))
        if start == end:
            continue
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    loops: list[list[tuple[float, float]]] = []
    visited_edges: set[frozenset[tuple[float, float]]] = set()
    for start, neighbours in adjacency.items():
        for first in neighbours:
            first_edge = frozenset((start, first))
            if first_edge in visited_edges:
                continue
            path = [start]
            previous, current = start, first
            local_edges: set[frozenset[tuple[float, float]]] = set()
            while True:
                path.append(current)
                edge = frozenset((previous, current))
                local_edges.add(edge)
                candidates = [
                    node for node in adjacency.get(current, [])
                    if node != previous
                ]
                if current == start:
                    loop = _sanitize_boundary(
                        [canonical[node] for node in path[:-1]]
                    )
                    if loop and not _boundary_self_intersects(loop):
                        loops.append(loop)
                    visited_edges.update(local_edges)
                    break
                if not candidates or len(path) > len(usable_walls) + 2:
                    break
                unvisited = [
                    node for node in candidates
                    if frozenset((current, node)) not in local_edges
                ]
                if not unvisited:
                    break
                previous, current = current, unvisited[0]
    if loops:
        return max(loops, key=lambda loop: abs(_polygon_area(loop)))
    hull = _sanitize_boundary(_convex_hull(list(canonical.values())))
    return [] if _boundary_self_intersects(hull) else hull


def _boundary_self_intersects(points: list[tuple[float, float]]) -> bool:
    """Return whether non-adjacent edges touch or cross."""
    if len(points) < 3:
        return True

    def orientation(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )

    def on_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        tolerance = 1e-9
        return (
            min(start[0], end[0]) - tolerance
            <= point[0]
            <= max(start[0], end[0]) + tolerance
            and min(start[1], end[1]) - tolerance
            <= point[1]
            <= max(start[1], end[1]) + tolerance
        )

    def intersects(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> bool:
        tolerance = 1e-9
        orientations = (
            orientation(first_start, first_end, second_start),
            orientation(first_start, first_end, second_end),
            orientation(second_start, second_end, first_start),
            orientation(second_start, second_end, first_end),
        )
        if (
            orientations[0] * orientations[1] < -tolerance
            and orientations[2] * orientations[3] < -tolerance
        ):
            return True
        return any((
            abs(orientations[0]) <= tolerance
            and on_segment(second_start, first_start, first_end),
            abs(orientations[1]) <= tolerance
            and on_segment(second_end, first_start, first_end),
            abs(orientations[2]) <= tolerance
            and on_segment(first_start, second_start, second_end),
            abs(orientations[3]) <= tolerance
            and on_segment(first_end, second_start, second_end),
        ))

    edges = list(zip(points, points[1:] + points[:1]))
    edge_count = len(edges)
    for first_index, (first_start, first_end) in enumerate(edges):
        for second_index in range(first_index + 1, edge_count):
            if second_index in {
                (first_index - 1) % edge_count,
                (first_index + 1) % edge_count,
            }:
                continue
            if intersects(
                first_start,
                first_end,
                edges[second_index][0],
                edges[second_index][1],
            ):
                return True
    return False


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]
