# -*- coding: utf-8 -*-
"""pyRevit test script: create a wall and host a door on it.

RUN INSIDE REVIT via pyRevit (e.g. pyRevit -> Run Script, or as a custom
command). It does NOT run in the conda env -- it needs the Revit API
(Autodesk.Revit.DB), which only exists inside the Revit process.

Purpose: prove the pyRevit <-> Revit interop path that replaces the IFC
exchange (see PLAN.md). Creates a 5 m x 2.8 m wall on the first Level and
hosts one door at its midpoint, using whatever Basic Wall type and door
FamilySymbol are already in the document.
"""
__title__ = "BIM-Recon Test\nWall + Door"
__doc__ = "Creates a 5m wall (2.8m high) on the first Level and hosts a door at its midpoint."

from pyrevit import revit, DB, script

doc = revit.doc
output = script.get_output()

FEET_PER_M = 1.0 / 0.3048  # Revit internal length unit is feet


def first_level():
    levels = list(DB.FilteredElementCollector(doc).OfClass(DB.Level).ToElements())
    return levels[0] if levels else None


def first_basic_wall_type():
    # Exclude curtain wall types (those don't take height the same way).
    types = [wt for wt in
             DB.FilteredElementCollector(doc).OfClass(DB.WallType).ToElements()
             if wt.Kind != DB.WallKind.Curtain]
    return types[0] if types else None


def first_door_symbol():
    syms = list(DB.FilteredElementCollector(doc).OfClass(DB.FamilySymbol)
                .OfCategory(DB.BuiltInCategory.OST_Doors).ToElements())
    return syms[0] if syms else None


def main():
    level = first_level()
    if level is None:
        output.print_md("**ERROR:** no Level in the document. "
                        "Open an architectural template / project with a level first.")
        return

    wall_type = first_basic_wall_type()
    if wall_type is None:
        output.print_md("**ERROR:** no non-curtain WallType found.")
        return

    door_sym = first_door_symbol()

    # Wall geometry (metric -> feet for the API).
    length_m, height_m = 5.0, 2.8
    start = DB.XYZ(0.0, 0.0, 0.0)
    end = DB.XYZ(length_m * FEET_PER_M, 0.0, 0.0)
    wall_line = DB.Line.CreateBound(start, end)

    with revit.Transaction("BIM-Recon: create wall + door"):
        # Wall.Create(doc, curve, wallTypeId, levelId, height, offset, flip, structural)
        wall = DB.Wall.Create(doc, wall_line, wall_type.Id, level.Id,
                              height_m * FEET_PER_M, 0.0, False, False)
        output.print_md("**Wall created** Id={} &mdash; {} m long, {} m high, type '{}', level '{}'."
                        .format(wall.Id, length_m, height_m, wall_type.Name, level.Name))

        if door_sym is None:
            output.print_md("**No door** FamilySymbol (OST_Doors) in the document &mdash; "
                            "wall only. Load a door family (e.g. Single-Flush) and re-run to test hosting.")
            return

        if not door_sym.IsActive:
            door_sym.Activate()
            doc.Regenerate()

        # Host the door at the wall midpoint. Host-based overload:
        # NewFamilyInstance(XYZ, FamilySymbol, Element host, StructuralType)
        mid = DB.XYZ((start.X + end.X) / 2.0, 0.0, 0.0)
        door = doc.Create.NewFamilyInstance(
            mid, door_sym, wall, DB.Structure.StructuralType.NonStructural)
        output.print_md("**Door hosted** Id={} on wall &mdash; family '{} | {}' at midpoint."
                        .format(door.Id, door_sym.FamilyName, door_sym.Name))


main()
