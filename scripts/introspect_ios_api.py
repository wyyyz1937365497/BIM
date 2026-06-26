"""Introspect the locally installed IfcOpenShell 0.8.5 api to get exact
signatures for the feature (opening/void/filling) and geometry usecases.

Output is written to stdout AND saved to Docs/ifcopenshell-python/_api_signatures.txt
so it becomes a permanent local reference.
"""
import inspect
import ifcopenshell
import ifcopenshell.api.unit as unit
import ifcopenshell.api.context as context
import ifcopenshell.api.aggregate as aggregate
import ifcopenshell.api.spatial as spatial
import ifcopenshell.api.root as root
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.feature as feature
import ifcopenshell.api.profile as ios_profile

OUT = ["# IfcOpenShell Python API signatures (introspected from installed %s)" % ifcopenshell.version, ""]

def dump(mod, names):
    OUT.append("## module: %s" % mod.__name__)
    for n in names:
        fn = getattr(mod, n, None)
        if fn is None:
            OUT.append("  %s -> MISSING" % n)
            continue
        try:
            sig = str(inspect.signature(fn))
        except (ValueError, TypeError):
            sig = "(?)"
        OUT.append("  %s%s" % (n, sig))
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        if doc:
            OUT.append("      doc: " + doc[0][:200])
    OUT.append("")

dump(unit, ["add_si_unit", "assign_unit"])
dump(context, ["add_context"])
dump(aggregate, ["assign_object"])
dump(spatial, ["assign_container"])
dump(root, ["create_entity", "remove_product"])
dump(geometry, [
    "edit_object_placement", "assign_representation",
    "add_wall_representation", "add_slab_representation",
    "add_door_representation", "add_window_representation",
    "add_profile_representation", "add_mesh_representation",
    "add_box_representation", "clip_solid", "add_void",
])
dump(feature, [n for n in dir(feature) if not n.startswith("_") and callable(getattr(feature, n, None)) and not n.startswith("wrap")])
dump(ios_profile, [n for n in dir(ios_profile) if not n.startswith("_") and callable(getattr(ios_profile, n, None)) and not n.startswith("wrap")])

text = "\n".join(OUT)
print(text)

import os
os.makedirs("Docs/ifcopenshell-python", exist_ok=True)
with open("Docs/ifcopenshell-python/_api_signatures.txt", "w", encoding="utf-8") as f:
    f.write(text)
