"""BIM-Recon Python utilities."""

from .diff_report import DiffReport, WallDiff, report_wall_differences
from .floorplan import (
    FloorPlan,
    FloorPlanProvider,
    FrameMeta,
    ManualProvider,
    Opening,
    OpeningKind,
    WallSegment,
    WallType,
    floorplan_to_dict,
    validate_floorplan,
)
from .revit_code import RevitGenerationOptions, generate_revit_csharp

__all__ = [
    "DiffReport",
    "FloorPlan",
    "FloorPlanProvider",
    "FrameMeta",
    "ManualProvider",
    "Opening",
    "OpeningKind",
    "RevitGenerationOptions",
    "WallDiff",
    "WallSegment",
    "WallType",
    "floorplan_to_dict",
    "generate_revit_csharp",
    "report_wall_differences",
    "validate_floorplan",
]
