"""IfcOpenShell seed script (P0 Week-1).

Builds a minimal but valid IFC4 model with:
  * full spatial hierarchy (Project/Site/Building/Storey),
  * metric units,
  * an IfcWall (SweptSolid body),
  * an IfcOpeningElement voided into the wall (IfcRelVoidsElement),
  * an IfcDoor filling that opening (IfcRelFillsElement),
  * an IfcSlab.

API verified against the locally installed IfcOpenShell 0.8.5
(see Docs/ifcopenshell-python/_api_signatures.txt) and the official
authoring docs cloned under Docs/ifcopenshell-python/.

This is the "IFC tail" of the pipeline: it proves IfcOpenShell can emit
Revit-editable structure. Geometry-fitting from 3DGS replaces the hardcoded
dimensions later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.api
# Submodules must be imported explicitly so that ifcopenshell.api.<module>
# resolves at call sites (they are otherwise lazily loaded and not attached
# as attributes of the parent package on attribute access alone).
import ifcopenshell.api.project  # noqa: F401
import ifcopenshell.api.unit  # noqa: F401
import ifcopenshell.api.context  # noqa: F401
import ifcopenshell.api.aggregate  # noqa: F401
import ifcopenshell.api.spatial  # noqa: F401
import ifcopenshell.api.root  # noqa: F401
import ifcopenshell.api.geometry  # noqa: F401
import ifcopenshell.api.feature  # noqa: F401
import ifcopenshell.api.profile  # noqa: F401
import ifcopenshell.api.owner  # noqa: F401


def _translate(x: float, y: float, z: float) -> np.ndarray:
    """4x4 homogeneous transform with translation (x,y,z), identity rotation."""
    m = np.eye(4, dtype=float)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def build_demo_ifc() -> ifcopenshell.file:
    # IFC2X3 is the project standard: Revit's direct-open importer supports
    # IFC2X3 natively (IFC4 can only be Linked, not opened/edited). See the
    # Revit QA finding recorded in PLAN.md.
    model = ifcopenshell.api.project.create_file(version="IFC2X3")

    # IFC2X3 does not auto-create an OwnerHistory (the IFC4 default path does),
    # so set one up explicitly before root.create_entity needs it: the api's
    # get_user requires an IfcPersonAndOrganization AND an IfcApplication.
    person = ifcopenshell.api.owner.add_person(
        model, identification="bim-recon", family_name="Recon", given_name="BIM")
    org = ifcopenshell.api.owner.add_organisation(
        model, identification="bim-recon", name="BIM-Recon")
    ifcopenshell.api.owner.add_person_and_organisation(
        model, person=person, organisation=org)
    ifcopenshell.api.owner.add_application(
        model, application_developer=org,
        application_full_name="BIM-Recon", application_identifier="bim-recon")
    ifcopenshell.api.owner.create_owner_history(model)

    # --- project must exist before units can be assigned -------------------
    project = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcProject", name="BIM-Recon Demo")

    # --- units (metric: SI metre) ------------------------------------------
    ifcopenshell.api.unit.assign_unit(model)  # defaults to a full metric unit set

    # --- geometric representation contexts ---------------------------------
    model_ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        model, context_type="Model",
        context_identifier="Body", target_view="MODEL_VIEW", parent=model_ctx,
    )

    # --- spatial hierarchy: site > building > storey -----------------------
    site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcBuildingStorey", name="Ground Floor")
    ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[building])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=building, products=[storey])

    # Explicit placements for the spatial containers (origin; storey at z=0).
    ifcopenshell.api.geometry.edit_object_placement(model, product=site)
    ifcopenshell.api.geometry.edit_object_placement(model, product=building)
    ifcopenshell.api.geometry.edit_object_placement(model, product=storey)

    # --- wall (SweptSolid: 5 m long, 2.8 m high, 0.2 m thick) --------------
    wall = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name="W1")
    ifcopenshell.api.geometry.edit_object_placement(model, product=wall)
    wall_rep = ifcopenshell.api.geometry.add_wall_representation(
        model, context=body, length=5.0, height=2.8, thickness=0.2)
    ifcopenshell.api.geometry.assign_representation(model, product=wall, representation=wall_rep)
    ifcopenshell.api.spatial.assign_container(
        model, relating_structure=storey, products=[wall])

    # --- opening voided into the wall (door: 1.0 x 2.1 m) ------------------
    # Opening body: rectangle profile (width x through-thickness), extruded up by door height.
    opening = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcOpeningElement", name="D1-Opening")
    # host the opening in the wall first (creates IfcRelVoidsElement + relative placement)
    ifcopenshell.api.feature.add_feature(model, feature=opening, element=wall)

    rect_profile = ifcopenshell.api.profile.add_parameterized_profile(
        model, ifc_class="IfcRectangleProfileDef", profile_type="AREA")
    # NB: edit_profile writes RAW IFC attributes, stored in the project unit
    # (millimetre here). The add_*_representation helpers take metres and
    # convert automatically, but profile dims do NOT -- provide mm directly.
    ifcopenshell.api.profile.edit_profile(
        model, profile=rect_profile,
        attributes={"XDim": 1000.0, "YDim": 400.0})  # 1.0 m wide, 0.4 m through (cuts the 0.2 m wall)
    opening_rep = ifcopenshell.api.geometry.add_profile_representation(
        model, context=body, profile=rect_profile, depth=2.1)  # 2.1 m tall (extrude along Z)
    ifcopenshell.api.geometry.assign_representation(
        model, product=opening, representation=opening_rep)
    # position the opening along the wall (wall-local: door centred at x=1.5)
    ifcopenshell.api.geometry.edit_object_placement(
        model, product=opening, matrix=_translate(1.5, 0.0, 0.0))

    # --- door filling the opening -----------------------------------------
    door = ifcopenshell.api.root.create_entity(
        model, ifc_class="IfcDoor", name="D1",
        predefined_type="DOOR")  # create_entity applies schema defaults (OperationType, etc.)
    door_rep = ifcopenshell.api.geometry.add_door_representation(
        # NB: overall_height/overall_width are in PROJECT units (mm here),
        # not metres -- see add_door_representation docstring.
        model, context=body, overall_height=2100.0, overall_width=1000.0)
    if door_rep is not None:
        ifcopenshell.api.geometry.assign_representation(
            model, product=door, representation=door_rep)
    ifcopenshell.api.geometry.edit_object_placement(
        model, product=door, matrix=_translate(1.5, 0.0, 0.0))
    ifcopenshell.api.spatial.assign_container(
        model, relating_structure=storey, products=[door])
    # door fills the opening (creates IfcRelFillsElement)
    ifcopenshell.api.feature.add_filling(model, opening=opening, element=door)

    # --- slab (5 x 4 m floor, 0.2 m thick) --------------------------------
    slab = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSlab", name="S1")
    ifcopenshell.api.geometry.edit_object_placement(model, product=slab)
    slab_rep = ifcopenshell.api.geometry.add_slab_representation(
        model, context=body, depth=0.2,
        polyline=[(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)])
    ifcopenshell.api.geometry.assign_representation(
        model, product=slab, representation=slab_rep)
    ifcopenshell.api.spatial.assign_container(
        model, relating_structure=storey, products=[slab])

    return model


def main() -> None:
    model = build_demo_ifc()
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "demo.ifc"
    model.write(str(out))

    # quick summary to stdout
    summary = {
        "IfcProject": len(model.by_type("IfcProject")),
        "IfcSite": len(model.by_type("IfcSite")),
        "IfcBuilding": len(model.by_type("IfcBuilding")),
        "IfcBuildingStorey": len(model.by_type("IfcBuildingStorey")),
        "IfcWall": len(model.by_type("IfcWall")),
        "IfcSlab": len(model.by_type("IfcSlab")),
        "IfcOpeningElement": len(model.by_type("IfcOpeningElement")),
        "IfcDoor": len(model.by_type("IfcDoor")),
        "IfcRelVoidsElement": len(model.by_type("IfcRelVoidsElement")),
        "IfcRelFillsElement": len(model.by_type("IfcRelFillsElement")),
    }
    print("Wrote", out.resolve())
    for k, v in summary.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
