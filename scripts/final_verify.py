"""Final local verification of output/demo.ifc before Revit QA.

1. IFC schema validation (catches rule violations Revit rejects on).
2. Export every element's mesh to output/demo.obj (SI metres) so the
   wall + door hole + door + slab can be eyeballed in ANY viewer, not just
   Revit -- an independent verification path.
"""
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.validate


def run_schema_validation(model):
    print("=== schema validation ===")
    logger = ifcopenshell.validate.json_logger()
    try:
        ifcopenshell.validate.validate(model, logger)
    except Exception as e:
        print("  validate RAISED:", type(e).__name__, str(e)[:160])
        return
    # json_logger stores collected entries in .statements (errors() is a logger method, not a getter)
    stmts = getattr(logger, "statements", []) or []
    print(f"  schema issues: {len(stmts)}")
    for s in stmts[:25]:
        print("   -", s)


def export_obj(model, path):
    print("\n=== OBJ export ===")
    settings = ifcopenshell.geom.settings()
    elements = []
    for cls in ("IfcSlab", "IfcWall", "IfcOpeningElement", "IfcDoor"):
        prods = model.by_type(cls)
        if prods:
            elements.append((f"{cls}_{prods[0].Name}", prods[0]))

    lines = ["# demo.ifc geometry (SI metres)"]
    voff = 0
    for label, prod in elements:
        shape = ifcopenshell.geom.create_shape(settings, prod)
        g = shape.geometry
        verts = list(g.verts)
        faces = list(g.faces)
        nv = len(verts) // 3
        lines.append(f"o {label}")
        for i in range(0, len(verts), 3):
            lines.append(f"v {verts[i]:.4f} {verts[i + 1]:.4f} {verts[i + 2]:.4f}")
        for i in range(0, len(faces), 3):
            lines.append(f"f {faces[i] + 1 + voff} {faces[i + 1] + 1 + voff} {faces[i + 2] + 1 + voff}")
        voff += nv
        print(f"  {label}: {nv} verts, {len(faces) // 3} tris")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path} ({voff} verts total)")


def main():
    model = ifcopenshell.open("output/demo.ifc")
    run_schema_validation(model)
    export_obj(model, "output/demo.obj")


if __name__ == "__main__":
    main()
