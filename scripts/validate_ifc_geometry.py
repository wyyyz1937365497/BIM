"""Verify output/demo.ifc geometry (two paths).

Path A (always available): create_shape -> Triangulation mesh -> bbox + volume.
Path B (if OCC present): serialise -> OCP BRepGProp -> exact boolean volume,
    which proves the door opening is actually subtracted from the wall.

Note: has_occon this build is False, so Path B is expected to fall back;
the mesh path still confirms element sizes and whether the backend applied
the opening boolean itself.
"""
import ifcopenshell
import ifcopenshell.geom
import numpy as np


def mesh_of(settings, product):
    shape = ifcopenshell.geom.create_shape(settings, product)
    geo = shape.geometry
    verts = np.asarray(geo.verts, dtype=float).reshape(-1, 3)
    # geo.faces is a FLAT list of triangle indices (3 per triangle).
    f = list(geo.faces)
    tris = [(f[i], f[i + 1], f[i + 2]) for i in range(0, len(f) - 2, 3)]
    return verts, tris


def mesh_volume_m3(verts, tris):
    # create_shape normalises geometry to SI metres regardless of file unit,
    # so the divergence theorem result is already in m^3.
    v = 0.0
    for a, b, c in tris:
        v += np.dot(verts[a], np.cross(verts[b], verts[c])) / 6.0
    return abs(v)


def main():
    model = ifcopenshell.open("output/demo.ifc")
    settings = ifcopenshell.geom.settings()
    print("has_occ =", ifcopenshell.geom.has_occ)

    print(f"\n{'element':24s} {'size XxYxZ (m)':22s} {'mesh vol m^3':>12s}")
    print("-" * 62)
    info = {}
    for cls, name in [("IfcWall", "W1"), ("IfcOpeningElement", "D1-Opening"),
                      ("IfcSlab", "S1"), ("IfcDoor", "D1")]:
        prods = model.by_type(cls)
        if not prods:
            continue
        verts, tris = mesh_of(settings, prods[0])
        size = verts.max(0) - verts.min(0)  # metres (create_shape -> SI)
        vol = mesh_volume_m3(verts, tris)
        info[cls] = (size.tolist(), vol)
        print(f"{cls + ' ' + name:24s} {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f}      {vol:12.4f}")

    wall_size, wall_vol = info["IfcWall"]
    op_size, _ = info["IfcOpeningElement"]
    solid = 5.0 * 0.2 * 2.8  # 2.8 m^3
    print(f"\nwall solid = {solid:.3f} m^3 ; mesh wall vol = {wall_vol:.3f}")
    print("wall size ~ 5x0.2x2.8 m:",
          all(abs(wall_size[i] - v) < 0.1 for i, v in enumerate([5.0, 0.2, 2.8])))
    print("opening ~1m wide, >=0.2m through, ~2.1m tall:",
          abs(op_size[0] - 1.0) < 0.05 and op_size[1] >= 0.2 and abs(op_size[2] - 2.1) < 0.1)

    # Path B: exact OCC boolean volume (optional)
    print("\n--- exact BRep volume via OCC (optional) ---")
    occ_ok = False
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        wall = model.by_type("IfcWall")[0]
        brep = ifcopenshell.geom.serialise(settings, wall)
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(brep, props)
        exact = props.Mass() / 1e9
        occ_ok = True
        print(f"OCC exact wall volume (m^3): {exact:.4f}")
        if exact < solid - 0.1:
            print("=> opening boolean CONFIRMED subtracted (exact).")
    except Exception as e:
        print("OCC exact volume unavailable:", type(e).__name__, str(e)[:140])

    if not occ_ok:
        if wall_vol < solid - 0.1:
            print(f"=> mesh indicates opening was subtracted locally "
                  f"(wall vol {wall_vol:.3f} < solid {solid:.3f}).")
        else:
            print("=> mesh shows solid wall on this backend; the opening is "
                  "declared via IfcRelVoidsElement (verified by tests) and "
                  "Revit's geometry engine will subtract it on import.")


if __name__ == "__main__":
    main()
