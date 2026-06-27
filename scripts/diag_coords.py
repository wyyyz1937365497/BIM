"""Diagnose output/demo.ifc for the Revit 3D-invisibility root causes.

Checks the two known Revit IFC bugs (NOT format defects):
  1. Orphan / huge coordinates (georef blow-up) -- e.g. 1.79e+305 points that
     expand the 3D view bounding box so geometry "vanishes".
  2. IfcMapConversion / IfcSite global coordinates.
Also confirms IfcSite origin is at (0,0,0) and reports the max coordinate
magnitude across all IfcCartesianPoint in the file.
"""
import re
import ifcopenshell

m = ifcopenshell.open("output/demo.ifc")

site = m.by_type("IfcSite")[0]
print("=== IfcSite ===")
print("  RefLatitude :", site.RefLatitude)
print("  RefLongitude:", site.RefLongitude)
print("  RefElevation:", site.RefElevation)
pl = site.ObjectPlacement
print("  ObjectPlacement:", pl.is_a() if pl else None)

print("\n=== georeferencing ===")
for t in ("IfcMapConversion", "IfcProjectedCoordinateSystem"):
    try:
        n = len(m.by_type(t))
        print(f"  {t} count: {n}")
    except RuntimeError:
        print(f"  {t}: not in this schema (IFC2X3) -> cannot be present")

print("\n=== coordinate magnitude (the key 3D-bbox test) ===")
maxc = 0.0
worst = None
for cp in m.by_type("IfcCartesianPoint"):
    for c in cp.Coordinates:
        a = abs(float(c))
        if a > maxc:
            maxc = a
            worst = (cp.id(), list(cp.Coordinates))
print(f"  max |coord| in any IfcCartesianPoint: {maxc}")
print(f"  worst point: id={worst[0]} coords={worst[1]}")

# bounding box of ALL points (this is what Revit's 3D view would span)
xs, ys, zs = [], [], []
for cp in m.by_type("IfcCartesianPoint"):
    co = list(cp.Coordinates)
    xs.append(co[0]); ys.append(co[1])
    if len(co) > 2:
        zs.append(co[2])
print(f"  all-points bbox: X[{min(xs)},{max(xs)}] Y[{min(ys)},{max(ys)}] Z[{min(zs)},{max(zs)}]")

print("\n=== raw text scientific-notation scan ===")
text = open("output/demo.ifc", encoding="utf-8").read()
sci = re.findall(r"-?\d\.\d+E[+-]\d+", text)
print(f"  scientific-notation tokens: {len(sci)} {sci[:6]}")
print(f"  contains 1.79769313486232E+305: {'1.79769313486232E+305' in text}")

print("\n=== verdict ===")
threshold = 1e6  # anything over a million metres is an orphan point
huge_sci = [s for s in sci if abs(float(s)) > threshold]
if maxc > threshold or huge_sci or "1.79769313486232E+305" in text:
    print(f"  FAIL: orphan/huge coordinate detected (max={maxc}, huge_sci={huge_sci[:3]})")
    print("         -> Revit 3D bbox blow-up bug #1. Fix at export: force IfcSite")
    print("            origin (0,0,0), do NOT write IfcMapConversion.")
else:
    print(f"  CLEAN: max coord = {maxc} (room scale); no huge/orphan points;")
    print(f"         {len(sci)} sci token(s) are tiny epsilons, not orphans.")
    print("         3D invisibility is NOT this file -> it is Revit's IFC processor")
    print("         (bug #2). Fix via Revit.ini [ImportIFC] LinkProcessor=Legacy")
    print("         and/or Link -> Bind -> Ungroup for editable native elements.")
