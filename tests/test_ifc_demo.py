"""Tests for the IfcOpenShell seed script (P0 Week-1, part 2/2).

These define the contract the generated IFC must satisfy so that it is
usable (and editable) in Revit:

  * a complete spatial hierarchy (Project/Site/Building/Storey),
  * a wall with a voided opening, a door filling that opening,
  * a slab,
  * metric units.

The exact API calls come from the verified IfcOpenShell recipe; this file
only asserts the *outcome* on the real IFC object.
"""
import ifcopenshell

from bim_recon import ifc_demo


def _count(model: ifcopenshell.file, ifc_class: str) -> int:
    return len(model.by_type(ifc_class))


def test_demo_builds_valid_ifc_with_full_hierarchy():
    model = ifc_demo.build_demo_ifc()

    assert _count(model, "IfcProject") >= 1
    assert _count(model, "IfcSite") >= 1
    assert _count(model, "IfcBuilding") >= 1
    assert _count(model, "IfcBuildingStorey") >= 1


def test_demo_has_wall_slab_opening_door():
    model = ifc_demo.build_demo_ifc()

    assert _count(model, "IfcWall") >= 1
    assert _count(model, "IfcSlab") >= 1
    assert _count(model, "IfcOpeningElement") >= 1
    assert _count(model, "IfcDoor") >= 1


def test_opening_voids_wall_and_door_fills_opening():
    model = ifc_demo.build_demo_ifc()

    # IfcRelVoidsElement: the opening must relate to a building element (the wall)
    rels_voids = model.by_type("IfcRelVoidsElement")
    assert len(rels_voids) >= 1, "opening is not voiding any element"
    opening = rels_voids[0].RelatedOpeningElement
    assert opening.is_a("IfcOpeningElement")
    # the wall hosting the opening:
    hosting_wall = rels_voids[0].RelatingBuildingElement
    assert hosting_wall.is_a("IfcWall")

    # IfcRelFillsElement: the door must fill an opening
    rels_fills = model.by_type("IfcRelFillsElement")
    assert len(rels_fills) >= 1, "door is not filling any opening"
    assert rels_fills[0].RelatedBuildingElement.is_a("IfcDoor")
    assert rels_fills[0].RelatingOpeningElement.is_a("IfcOpeningElement")


def test_metric_length_unit_assigned():
    import ifcopenshell.util.unit as uutil
    model = ifc_demo.build_demo_ifc()

    unit_assignment = uutil.get_unit_assignment(model)
    assert unit_assignment is not None, "no unit assignment found"

    units = unit_assignment.Units
    has_length_metre = any(
        u.is_a("IfcSIUnit")
        and u.UnitType == "LENGTHUNIT"
        and u.Name == "METRE"
        for u in units
    )
    assert has_length_metre, "no metric length unit (METRE) assigned"


def test_wall_has_body_representation():
    """The wall must carry a Body representation (SweptSolid) so it renders in Revit."""
    model = ifc_demo.build_demo_ifc()
    wall = model.by_type("IfcWall")[0]
    shape = wall.Representation  # IfcProductDefinitionShape
    assert shape is not None, "wall has no representation"
    reps = shape.Representations
    body_reps = [r for r in reps if r.RepresentationIdentifier == "Body"]
    assert body_reps, "wall has no Body representation"
    assert body_reps[0].RepresentationType in {"SweptSolid", "Brep", "Clipping"}


def test_demo_write_to_disk_round_trip(tmp_path):
    """build_demo_ifc must be serializable and re-parseable."""
    model = ifc_demo.build_demo_ifc()
    out = tmp_path / "demo.ifc"
    model.write(str(out))

    reloaded = ifcopenshell.open(str(out))
    assert len(reloaded.by_type("IfcWall")) >= 1
    assert len(reloaded.by_type("IfcDoor")) >= 1
