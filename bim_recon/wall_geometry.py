"""Lightweight 2D wall topology cleanup shared by pipeline and Revit workflows."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

Point2D = tuple[float, float]


def _point(wall: Mapping[str, Any], endpoint: int) -> Point2D:
    suffix = "1" if endpoint == 0 else "2"
    return float(wall[f"x{suffix}"]), float(wall[f"y{suffix}"])


def _length(wall: Mapping[str, Any]) -> float:
    start = _point(wall, 0)
    end = _point(wall, 1)
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _set_endpoint(wall: dict[str, Any], endpoint: int, point: Point2D) -> None:
    suffix = "1" if endpoint == 0 else "2"
    wall[f"x{suffix}"] = point[0]
    wall[f"y{suffix}"] = point[1]
    wall["length"] = _length(wall)


def _ray_intersection(
    first_origin: Point2D,
    first_direction: Point2D,
    second_origin: Point2D,
    second_direction: Point2D,
) -> tuple[Point2D, float, float] | None:
    cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(cross) <= 1e-9:
        return None
    delta = (
        second_origin[0] - first_origin[0],
        second_origin[1] - first_origin[1],
    )
    first_t = (
        delta[0] * second_direction[1]
        - delta[1] * second_direction[0]
    ) / cross
    second_t = (
        delta[0] * first_direction[1]
        - delta[1] * first_direction[0]
    ) / cross
    intersection = (
        first_origin[0] + first_t * first_direction[0],
        first_origin[1] + first_t * first_direction[1],
    )
    return intersection, first_t, second_t


def fit_short_corner_walls(
    walls: Sequence[Mapping[str, Any]],
    *,
    max_short_length: float = 0.15,
    min_adjacent_ratio: float = 4.0,
    connection_tolerance: float = 0.02,
    max_ray_extension: float = 0.5,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Replace a short corner chamfer with the intersection of adjacent wall rays.

    The returned source-index list remains parallel to the returned walls so
    callers loading older result files can preserve existing ``wall_idx``
    references after the chamfer segment is removed.
    """
    adjusted = [dict(wall) for wall in walls]
    for wall in adjusted:
        wall["length"] = _length(wall)
    active = [True] * len(adjusted)

    def connected_long_wall(
        short_index: int,
        endpoint: Point2D,
        minimum_length: float,
    ) -> tuple[int, int] | None:
        matches: list[tuple[float, float, int, int]] = []
        for wall_index, wall in enumerate(adjusted):
            if wall_index == short_index or not active[wall_index]:
                continue
            wall_length = _length(wall)
            if wall_length < minimum_length:
                continue
            for endpoint_index in (0, 1):
                candidate = _point(wall, endpoint_index)
                distance = math.dist(endpoint, candidate)
                if distance <= connection_tolerance:
                    matches.append((distance, -wall_length, wall_index, endpoint_index))
        if not matches:
            return None
        _distance, _negative_length, wall_index, endpoint_index = min(matches)
        return wall_index, endpoint_index

    short_indices = sorted(range(len(adjusted)), key=lambda index: _length(adjusted[index]))
    for short_index in short_indices:
        if not active[short_index]:
            continue
        short_wall = adjusted[short_index]
        short_length = _length(short_wall)
        if short_length > max_short_length:
            continue
        minimum_adjacent = max(
            max_short_length,
            short_length * min_adjacent_ratio,
        )
        first_match = connected_long_wall(
            short_index, _point(short_wall, 0), minimum_adjacent,
        )
        second_match = connected_long_wall(
            short_index, _point(short_wall, 1), minimum_adjacent,
        )
        if first_match is None or second_match is None:
            continue
        first_index, first_endpoint = first_match
        second_index, second_endpoint = second_match
        if first_index == second_index:
            continue

        first_corner = _point(adjusted[first_index], first_endpoint)
        first_inner = _point(adjusted[first_index], 1 - first_endpoint)
        second_corner = _point(adjusted[second_index], second_endpoint)
        second_inner = _point(adjusted[second_index], 1 - second_endpoint)
        first_vector = (
            first_corner[0] - first_inner[0],
            first_corner[1] - first_inner[1],
        )
        second_vector = (
            second_corner[0] - second_inner[0],
            second_corner[1] - second_inner[1],
        )
        first_norm = math.hypot(*first_vector)
        second_norm = math.hypot(*second_vector)
        if first_norm <= 1e-9 or second_norm <= 1e-9:
            continue
        first_direction = (
            first_vector[0] / first_norm,
            first_vector[1] / first_norm,
        )
        second_direction = (
            second_vector[0] / second_norm,
            second_vector[1] / second_norm,
        )
        result = _ray_intersection(
            first_corner,
            first_direction,
            second_corner,
            second_direction,
        )
        if result is None:
            continue
        intersection, first_t, second_t = result
        if (
            first_t < -connection_tolerance
            or second_t < -connection_tolerance
            or first_t > max_ray_extension
            or second_t > max_ray_extension
        ):
            continue

        _set_endpoint(adjusted[first_index], first_endpoint, intersection)
        _set_endpoint(adjusted[second_index], second_endpoint, intersection)
        active[short_index] = False

    source_indices = [index for index, keep in enumerate(active) if keep]
    return [adjusted[index] for index in source_indices], source_indices
