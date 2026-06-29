"""把 FloorPlan 转成 Revit API C# 脚本。

本模块只做确定性代码生成，不直接连接 Revit。生成结果用于
`mcp-servers-for-revit` 的 `send_code_to_revit` 工具，或复制进 Revit C#
执行环境中调试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .floorplan import FloorPlan, Opening, OpeningKind, WallSegment


@dataclass(frozen=True)
class RevitGenerationOptions:
    """Revit 原生图元生成参数，单位除特别说明外均为米。"""

    level_name: str = "BIM Recon Level 0"
    base_elevation: float = 0.0
    wall_height: float = 2.8
    door_height: float = 2.1
    window_height: float = 1.2
    floor_thickness: float = 0.12
    ceiling_height: float = 2.8
    create_floor: bool = True
    create_ceiling: bool = True
    place_hosted_families: bool = True


def generate_revit_csharp(
    floorplan: FloorPlan,
    options: RevitGenerationOptions | None = None,
) -> str:
    """生成可交给 Revit MCP 执行的 C# 代码。

    生成的代码假设执行上下文里存在 `uiapp` 变量，类型为
    `Autodesk.Revit.UI.UIApplication`。多数 Revit C# 动态执行器都会提供
    这个变量；若具体 MCP 环境变量名不同，只需要替换文件顶部第一行。
    """
    opts = options or RevitGenerationOptions()
    wall_blocks = "\n".join(
        _wall_block(index, wall, opts)
        for index, wall in enumerate(floorplan.walls)
    )
    opening_blocks = "\n".join(
        _opening_block(index, opening, floorplan.walls[opening.wall_index], opts)
        for index, opening in enumerate(floorplan.openings)
    )
    floor_block = _floor_block(floorplan.walls, opts) if opts.create_floor else ""
    ceiling_block = _ceiling_block(floorplan.walls, opts) if opts.create_ceiling else ""

    return f"""// BIM-Recon P0 Revit API script
// 用法：通过 mcp-servers-for-revit 的 send_code_to_revit 执行。
// 假设执行上下文提供 UIApplication uiapp；如果你的 MCP 变量名不同，只改下一行。
var doc = uiapp.ActiveUIDocument.Document;

const double MetersToFeet = 3.280839895013123;
double M(double value) => value * MetersToFeet;

using (var tx = new Autodesk.Revit.DB.Transaction(doc, "BIM-Recon create native elements"))
{{
    tx.Start();

    var level = GetOrCreateLevel(doc, "{_escape(opts.level_name)}", M({_fmt(opts.base_elevation)}));
    var createdWalls = new System.Collections.Generic.List<Autodesk.Revit.DB.Wall>();

{wall_blocks}
{floor_block}
{ceiling_block}
{opening_blocks}

    tx.Commit();
}}

Autodesk.Revit.DB.Level GetOrCreateLevel(Autodesk.Revit.DB.Document document, string name, double elevation)
{{
    var levelCollector = new Autodesk.Revit.DB.FilteredElementCollector(document)
        .OfClass(typeof(Autodesk.Revit.DB.Level));
    var levels = System.Linq.Enumerable.Cast<Autodesk.Revit.DB.Level>(levelCollector);
    var existing = System.Linq.Enumerable.FirstOrDefault(
        levels,
        l => System.Math.Abs(l.Elevation - elevation) < 0.001
    );
    if (existing != null) return existing;
    var level = Autodesk.Revit.DB.Level.Create(document, elevation);
    level.Name = name;
    return level;
}}

Autodesk.Revit.DB.WallType GetWallType(Autodesk.Revit.DB.Document document, double thicknessFeet)
{{
    string typeName = "BIM-Recon Wall " + System.Math.Round(thicknessFeet / MetersToFeet, 3).ToString("0.###") + "m";
    var wallTypeCollector = new Autodesk.Revit.DB.FilteredElementCollector(document)
        .OfClass(typeof(Autodesk.Revit.DB.WallType));
    var wallTypes = System.Linq.Enumerable.Cast<Autodesk.Revit.DB.WallType>(wallTypeCollector);
    var existing = System.Linq.Enumerable.FirstOrDefault(wallTypes, t => t.Name == typeName);
    if (existing != null) return existing;

    var baseType = System.Linq.Enumerable.First(
        wallTypes,
        t => t.Kind == Autodesk.Revit.DB.WallKind.Basic
    );
    var duplicated = (Autodesk.Revit.DB.WallType)baseType.Duplicate(typeName);
    var structure = duplicated.GetCompoundStructure();
    if (structure != null && structure.LayerCount > 0)
    {{
        structure.SetLayerWidth(0, thicknessFeet);
        duplicated.SetCompoundStructure(structure);
    }}
    return duplicated;
}}

Autodesk.Revit.DB.FloorType GetFloorType(Autodesk.Revit.DB.Document document)
{{
    var floorTypeCollector = new Autodesk.Revit.DB.FilteredElementCollector(document)
        .OfClass(typeof(Autodesk.Revit.DB.FloorType));
    return System.Linq.Enumerable.First(
        System.Linq.Enumerable.Cast<Autodesk.Revit.DB.FloorType>(floorTypeCollector)
    );
}}

Autodesk.Revit.DB.FamilySymbol FindHostedSymbol(Autodesk.Revit.DB.Document document, Autodesk.Revit.DB.BuiltInCategory category)
{{
    var symbolCollector = new Autodesk.Revit.DB.FilteredElementCollector(document)
        .OfClass(typeof(Autodesk.Revit.DB.FamilySymbol));
    var symbols = System.Linq.Enumerable.Cast<Autodesk.Revit.DB.FamilySymbol>(symbolCollector);
    return System.Linq.Enumerable.FirstOrDefault(
        symbols,
        s => s.Category != null && s.Category.Id.IntegerValue == (int)category
    );
}}
"""


def _wall_block(index: int, wall: WallSegment, opts: RevitGenerationOptions) -> str:
    return f"""    // wall {index}
    var wallLine{index} = Autodesk.Revit.DB.Line.CreateBound(
        new Autodesk.Revit.DB.XYZ(M({_fmt(wall.x1)}), M({_fmt(wall.y1)}), M({_fmt(opts.base_elevation)})),
        new Autodesk.Revit.DB.XYZ(M({_fmt(wall.x2)}), M({_fmt(wall.y2)}), M({_fmt(opts.base_elevation)}))
    );
    var wallType{index} = GetWallType(doc, M({_fmt(wall.resolved_thickness())}));
    var wall{index} = Autodesk.Revit.DB.Wall.Create(
        doc, wallLine{index}, wallType{index}.Id, level.Id, M({_fmt(opts.wall_height)}), 0.0, false, false
    );
    createdWalls.Add(wall{index});
"""


def _floor_block(walls: Iterable[WallSegment], opts: RevitGenerationOptions) -> str:
    loop_lines = _curve_loop_lines(walls, suffix="Floor", z_value=opts.base_elevation)
    return f"""    // floor slab
    var floorLoop = new Autodesk.Revit.DB.CurveLoop();
{loop_lines}
    var floorLoops = new System.Collections.Generic.List<Autodesk.Revit.DB.CurveLoop>() {{ floorLoop }};
    var floor = Autodesk.Revit.DB.Floor.Create(doc, floorLoops, GetFloorType(doc).Id, level.Id);
    var floorOffset = floor.get_Parameter(Autodesk.Revit.DB.BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM);
    if (floorOffset != null && !floorOffset.IsReadOnly) floorOffset.Set(M({_fmt(opts.base_elevation)}));
"""


def _ceiling_block(walls: Iterable[WallSegment], opts: RevitGenerationOptions) -> str:
    loop_lines = _curve_loop_lines(walls, suffix="Ceiling", z_value=opts.ceiling_height)
    return f"""    // ceiling slab, represented as a native Floor at ceiling height for MVP editability
    var ceilingLoop = new Autodesk.Revit.DB.CurveLoop();
{loop_lines}
    var ceilingLoops = new System.Collections.Generic.List<Autodesk.Revit.DB.CurveLoop>() {{ ceilingLoop }};
    var ceiling = Autodesk.Revit.DB.Floor.Create(doc, ceilingLoops, GetFloorType(doc).Id, level.Id);
    var ceilingOffset = ceiling.get_Parameter(Autodesk.Revit.DB.BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM);
    if (ceilingOffset != null && !ceilingOffset.IsReadOnly) ceilingOffset.Set(M({_fmt(opts.ceiling_height)}));
"""


def _opening_block(
    index: int,
    opening: Opening,
    wall: WallSegment,
    opts: RevitGenerationOptions,
) -> str:
    start_x, start_y = _point_on_wall(wall, opening.offset)
    end_x, end_y = _point_on_wall(wall, opening.offset + opening.width)
    bottom = 0.0 if opening.kind is OpeningKind.DOOR else float(opening.sill_height or 0.9)
    height = opts.door_height if opening.kind is OpeningKind.DOOR else opts.window_height
    top = bottom + height
    category = (
        "OST_Doors" if opening.kind is OpeningKind.DOOR else "OST_Windows"
    )
    family_block = ""
    if opts.place_hosted_families:
        center_x, center_y = _point_on_wall(wall, opening.offset + opening.width / 2.0)
        family_block = f"""
    var symbol{index} = FindHostedSymbol(doc, Autodesk.Revit.DB.BuiltInCategory.{category});
    if (symbol{index} != null)
    {{
        if (!symbol{index}.IsActive) symbol{index}.Activate();
        doc.Create.NewFamilyInstance(
            new Autodesk.Revit.DB.XYZ(M({_fmt(center_x)}), M({_fmt(center_y)}), M({_fmt(bottom)})),
            symbol{index}, createdWalls[{opening.wall_index}], level,
            Autodesk.Revit.DB.Structure.StructuralType.NonStructural
        );
    }}
"""
    return f"""    // opening {index}: {opening.kind.value} on wall {opening.wall_index}
    doc.Create.NewOpening(
        createdWalls[{opening.wall_index}],
        new Autodesk.Revit.DB.XYZ(M({_fmt(start_x)}), M({_fmt(start_y)}), M({_fmt(bottom)})),
        new Autodesk.Revit.DB.XYZ(M({_fmt(end_x)}), M({_fmt(end_y)}), M({_fmt(top)}))
    );{family_block}
"""


def _curve_loop_lines(
    walls: Iterable[WallSegment],
    suffix: str,
    z_value: float,
) -> str:
    lines = []
    for index, wall in enumerate(walls):
        lines.append(
            f"""    {suffix.lower()}Loop.Append(Autodesk.Revit.DB.Line.CreateBound(
        new Autodesk.Revit.DB.XYZ(M({_fmt(wall.x1)}), M({_fmt(wall.y1)}), M({_fmt(z_value)})),
        new Autodesk.Revit.DB.XYZ(M({_fmt(wall.x2)}), M({_fmt(wall.y2)}), M({_fmt(z_value)}))
    )); // edge {index}"""
        )
    return "\n".join(lines)


def _point_on_wall(wall: WallSegment, offset: float) -> tuple[float, float]:
    length = wall.length()
    ratio = offset / length
    return (
        wall.x1 + (wall.x2 - wall.x1) * ratio,
        wall.y1 + (wall.y2 - wall.y1) * ratio,
    )


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
