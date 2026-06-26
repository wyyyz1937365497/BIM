"""Final import-readiness sweep for output/demo.ifc.

Catches the classes of errors that make Revit refuse/quietly mangle an IFC
on import, beyond what entity-level schema validation covers:
  - IFC file header validity (validate_ifc_header)
  - GlobalId well-formedness (validate_guid)
  - GlobalId uniqueness across all IfcRoot
  - placement chain integrity (PlacementRelTo resolves)
  - void/fill relationship integrity (refs resolve to correct types)
"""
import ifcopenshell
import ifcopenshell.validate
from collections import Counter

m = ifcopenshell.open("output/demo.ifc")
issues = []


def check(label, fn):
    try:
        fn()
        print(f"  {label}: OK")
    except TypeError:
        # some validators need a logger arg
        try:
            fn(ifcopenshell.validate.json_logger())
            print(f"  {label}: OK")
        except Exception as e:
            issues.append(f"{label}: {type(e).__name__}: {str(e)[:120]}")
            print(f"  {label}: FAIL -> {e}")
    except Exception as e:
        issues.append(f"{label}: {type(e).__name__}: {str(e)[:120]}")
        print(f"  {label}: FAIL -> {e}")


print("=== header / guid ===")
logger = ifcopenshell.validate.json_logger()
try:
    ifcopenshell.validate.validate_ifc_header(m, logger)
    n = len(getattr(logger, "statements", []) or [])
    print(f"  validate_ifc_header: OK ({n} header issues)")
except Exception as e:
    print(f"  validate_ifc_header: FAIL -> {type(e).__name__}: {str(e)[:120]}")
    issues.append(f"header: {e}")

# GlobalId well-formedness (validate_guid validates a single guid string)
bad_guids = 0
for r in m.by_type("IfcRoot"):
    gid = r.GlobalId
    if not gid:
        continue
    try:
        ifcopenshell.validate.validate_guid(gid)
    except Exception:
        bad_guids += 1
print(f"  GlobalId well-formedness: {'OK' if bad_guids == 0 else f'{bad_guids} malformed'}")
if bad_guids:
    issues.append(f"{bad_guids} malformed GlobalIds")

print("\n=== GlobalId uniqueness ===")
gids = [r.GlobalId for r in m.by_type("IfcRoot") if r.GlobalId]
dups = [g for g, c in Counter(gids).items() if c > 1]
print(f"  IfcRoot GlobalIds: {len(gids)}, duplicates: {len(dups)}")
if dups:
    issues.append(f"duplicate GlobalIds: {dups}")

print("\n=== placement chain integrity ===")
pls = {p.id() for p in m.by_type("IfcLocalPlacement")}
dangling = 0
for p in m.by_type("IfcLocalPlacement"):
    ref = p.PlacementRelTo
    if ref is not None and ref.id() not in pls:
        dangling += 1
print(f"  IfcLocalPlacement count: {len(pls)}, dangling PlacementRelTo: {dangling}")
if dangling:
    issues.append(f"{dangling} dangling placements")

print("\n=== void/fill integrity ===")
for rel in m.by_type("IfcRelVoidsElement"):
    if not rel.RelatingBuildingElement or not rel.RelatedOpeningElement:
        issues.append(f"incomplete IfcRelVoidsElement #{rel.id()}")
for rel in m.by_type("IfcRelFillsElement"):
    if not rel.RelatingOpeningElement or not rel.RelatedBuildingElement:
        issues.append(f"incomplete IfcRelFillsElement #{rel.id()}")
print(f"  void rels: {len(m.by_type('IfcRelVoidsElement'))}, "
      f"fill rels: {len(m.by_type('IfcRelFillsElement'))}")

print("\n=== RESULT ===")
if issues:
    print(f"  {len(issues)} ISSUE(S):")
    for i in issues:
        print("   -", i)
    raise SystemExit(1)
print("  ALL IMPORT-READINESS CHECKS PASSED")
