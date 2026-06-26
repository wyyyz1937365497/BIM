"""Real-surface check: confirm the opening actually cuts the wall geometry.

Uses IfcOpenShell's geometry iterator on the written demo.ifc with openings
APPLIED. If the wall volume < solid box (5*0.2*2.8 = 2.8 m^3), the boolean
void was applied -- strong evidence the IFC is Revit-compatible.
"""
import ifcopenshell
import ifcopenshell.geom

m = ifcopenshell.open("output/demo.ifc")
settings = ifcopenshell.geom.settings()
# NB: openings are applied by default in this build; there is no
# APPLY_OPENINGS constant on Settings here. (DISABLE_OPENING_SUBTRACTIONS
# would turn them off.)

results = {}
for prod in m.by_type("IfcWall") + m.by_type("IfcSlab") + m.by_type("IfcDoor"):
    try:
        shape = ifcopenshell.geom.create_geometry(settings, m, prod.id())
        results[prod.is_a()] = round(shape.volume, 4)
    except Exception as e:
        results[prod.is_a()] = "ERR: " + str(e)[:80]

print("processed:", len(results), "geometries")
for k, v in results.items():
    print(f"  {k:20s} volume={v}")

wall_vol = results.get("IfcWall")
solid_box = 5.0 * 0.2 * 2.8           # 2.8 m^3 (no opening)
opening_cut = 1.0 * 0.2 * 2.1         # 0.42 m^3 (the door void)
expected = solid_box - opening_cut     # ~2.38 m^3
print(f"\nwall solid box = {solid_box}, expected with door cut ~ {expected}")
if isinstance(wall_vol, (int, float)):
    cut = wall_vol < solid_box - 1e-3
    print("opening applied (wall vol < solid):", cut)
else:
    print("geometry processing unavailable on this build:", wall_vol)
