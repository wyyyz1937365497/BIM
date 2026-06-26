"""Diagnose the unit + stored-dimension story of output/demo.ifc.

Confirms whether geometry is stored consistently in mm (the BIM convention)
so that Revit shows a 5 m wall, not a 5 mm or 5 km wall.
"""
import ifcopenshell

m = ifcopenshell.open("output/demo.ifc")

print("== units declared in IfcUnitAssignment ==")
for ua in m.by_type("IfcUnitAssignment"):
    for u in ua.Units:
        if u.is_a("IfcSIUnit"):
            print(f"  IfcSIUnit  type={u.UnitType:12s} name={u.Name:8s} prefix={u.Prefix}")
        elif u.is_a("IfcConversionBasedUnit"):
            print(f"  IfcConversionBasedUnit type={u.UnitType} name={u.Name}")
        else:
            print(f"  {u.is_a()} type={getattr(u,'UnitType','-')}")


def dump_extrusion(label, prod):
    print(f"== {label} stored geometry ==")
    if prod.Representation is None:
        print("  (no representation)")
        return
    for r in prod.Representation.Representations:
        for it in r.Items:
            print(f"  item {it.is_a()}")
            if it.is_a("IfcExtrudedAreaSolid"):
                sa = it.SweptArea
                info = {}
                if sa.is_a("IfcRectangleProfileDef"):
                    info = {"XDim": sa.XDim, "YDim": sa.YDim}
                elif sa.is_a("IfcArbitraryClosedProfileDef"):
                    try:
                        pts = sa.OuterCurve.Points
                        info = {"n_pts": len(pts),
                                "bbox_xy_mm": _bbox(pts)}
                    except Exception as e:
                        info = {"err": str(e)[:60]}
                print(f"    Depth(mm)={it.Depth}  profile={sa.is_a()} {info}")


def _bbox(points):
    xs = [float(p.Coordinates[0]) for p in points]
    ys = [float(p.Coordinates[1]) for p in points]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


dump_extrusion("wall W1", m.by_type("IfcWall")[0])
dump_extrusion("opening D1-Opening", m.by_type("IfcOpeningElement")[0])
dump_extrusion("slab S1", m.by_type("IfcSlab")[0])
dump_extrusion("door D1", m.by_type("IfcDoor")[0])
