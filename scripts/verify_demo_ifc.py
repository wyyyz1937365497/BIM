"""Maximal local verification of output/demo.ifc before Revit QA.

Covers what the pip ifcopenshell build CAN check (no OCC IfcGeom kernel,
so no boolean volume, but everything else):
  1. IFC schema validation (catches rule violations Revit would reject on)
  2. every product carries a Body representation (esp. the door, since
     add_door_representation may return None)
  3. every product has an ObjectPlacement (no "floating" elements)
  4. void/fill relationships resolve to the right element types
  5. opening placement is relative to (hosted in) the wall
  6. dump the wall/opening/door world placements to eyeball alignment
"""
from __future__ import annotations
import sys
import ifcopenshell
import ifcopenshell.validate
from ifcopenshell.util.placement import get_local_placement


def main() -> int:
    model = ifcopenshell.open("output/demo.ifc")
    failures: list[str] = []
    print("=== 1. IFC schema validation ===")
    errors: list[str] = []
    try:
        ifcopenshell.validate.validate(model)
        print("  schema validation: PASS (no exceptions)")
    except Exception as e:
        # validate raises on the first error; capture it
        print("  schema validation: RAISED ->", type(e).__name__, str(e)[:200])
        failures.append(f"schema: {e}")

    print("\n=== 2. Body representation per product ===")
    for cls in ("IfcWall", "IfcSlab", "IfcDoor", "IfcOpeningElement"):
        prods = model.by_type(cls)
        for p in prods:
            shape = p.Representation
            body_reps = []
            if shape is not None:
                body_reps = [r for r in shape.Representations
                             if r.RepresentationIdentifier == "Body"]
            status = "OK" if body_reps else "MISSING BODY"
            rtype = body_reps[0].RepresentationType if body_reps else "-"
            print(f"  {cls} '{p.Name}': {status} ({rtype})")
            if not body_reps:
                failures.append(f"{cls} has no Body representation")

    print("\n=== 3. ObjectPlacement presence ===")
    for cls in ("IfcSite", "IfcBuilding", "IfcBuildingStorey",
                "IfcWall", "IfcSlab", "IfcDoor", "IfcOpeningElement"):
        for p in model.by_type(cls):
            ok = p.ObjectPlacement is not None
            print(f"  {cls} '{p.Name}': placement={'OK' if ok else 'MISSING'}")
            if not ok:
                failures.append(f"{cls} has no ObjectPlacement")

    print("\n=== 4. void/fill relationships ===")
    voids = model.by_type("IfcRelVoidsElement")
    fills = model.by_type("IfcRelFillsElement")
    if voids:
        v = voids[0]
        print(f"  IfcRelVoidsElement: wall={v.RelatingBuildingElement.is_a()}, "
              f"opening={v.RelatedOpeningElement.is_a()}")
        if not (v.RelatingBuildingElement.is_a("IfcWall")
                and v.RelatedOpeningElement.is_a("IfcOpeningElement")):
            failures.append("void relationship has wrong element types")
    else:
        failures.append("no IfcRelVoidsElement")
    if fills:
        f = fills[0]
        print(f"  IfcRelFillsElement: opening={f.RelatingOpeningElement.is_a()}, "
              f"door={f.RelatedBuildingElement.is_a()}")
        if not (f.RelatingOpeningElement.is_a("IfcOpeningElement")
                and f.RelatedBuildingElement.is_a("IfcDoor")):
            failures.append("fill relationship has wrong element types")
    else:
        failures.append("no IfcRelFillsElement")

    print("\n=== 5/6. placements (wall world frame; opening should be wall-relative) ===")
    wall = model.by_type("IfcWall")[0]
    opening = model.by_type("IfcOpeningElement")[0]
    door = model.by_type("IfcDoor")[0]
    print("  wall ObjectPlacement is_a:", wall.ObjectPlacement.is_a())
    print("  opening ObjectPlacement is_a:", opening.ObjectPlacement.is_a(),
          "(should be relative, hosted in wall)")
    print("  door   ObjectPlacement is_a:", door.ObjectPlacement.is_a())
    # world placement matrices (rounded) for eyeball alignment
    try:
        for name, prod in (("wall", wall), ("opening", opening), ("door", door)):
            m = get_local_placement(prod.ObjectPlacement)
            t = [round(float(m[i][3]), 3) for i in range(3)]
            print(f"  {name:8s} world translate = {t}")
    except Exception as e:
        print("  placement matrix dump unavailable:", e)

    print("\n=== RESULT ===")
    if failures:
        print(f"  {len(failures)} ISSUE(S):")
        for f in failures:
            print("   -", f)
        return 1
    print("  ALL LOCAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
