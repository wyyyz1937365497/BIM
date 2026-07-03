// create_custom_window.cs
// Create a window with CUSTOM width/height/sill by duplicating the base
// family type and setting type parameters.
//
// @param parameters[0] (long)   hostWallId   - Host wall ElementId
// @param parameters[1] (double) locationX    - Window center X (feet)
// @param parameters[2] (double) locationY    - Window center Y (feet)
// @param parameters[3] (double) sillHeight   - Sill height above level (feet)
// @param parameters[4] (double) width        - Window width (feet)
// @param parameters[5] (double) height       - Window height (feet)
// @param parameters[6] (bool)   facingFlipped - Flip facing (optional, default false)
// @returns { elementId, typeName, width_mm, height_mm, sill_mm }
//
// Usage:
//   revit_send_code_to_revit(code=<this file>, parameters=[337596, 5.0, 3.0, 3.3, 3.0, 4.0, false])

// --- Parse parameters ---
long hostWallId = Convert.ToInt64(parameters[0]);
double locX = Convert.ToDouble(parameters[1]);
double locY = Convert.ToDouble(parameters[2]);
double sillHeight = Convert.ToDouble(parameters[3]);
double widthFt = Convert.ToDouble(parameters[4]);
double heightFt = Convert.ToDouble(parameters[5]);
bool facingFlipped = parameters.Length > 6 && Convert.ToBoolean(parameters[6]);

// --- Find host wall ---
Element hostElem = document.GetElement(new ElementId(hostWallId));
if (!(hostElem is Wall hostWall))
    return new { error = $"hostWallId {hostWallId} is not a Wall" };

// --- Find base window FamilySymbol ---
FamilySymbol baseSymbol = new FilteredElementCollector(document)
    .OfClass(typeof(FamilySymbol))
    .OfCategory(BuiltInCategory.OST_Windows)
    .Cast<FamilySymbol>()
    .FirstOrDefault(s => s.IsActive) ?? new FilteredElementCollector(document)
    .OfClass(typeof(FamilySymbol))
    .OfCategory(BuiltInCategory.OST_Windows)
    .Cast<FamilySymbol>()
    .FirstOrDefault();

if (baseSymbol == null)
    return new { error = "No window family types found in project" };

// --- Duplicate type with custom dimensions ---
double widthMm = Math.Round(widthFt * 304.8);
double heightMm = Math.Round(heightFt * 304.8);
string typeName = $"BIM-Recon Window {widthMm:0}x{heightMm:0}";

FamilySymbol customSymbol = new FilteredElementCollector(document)
    .OfClass(typeof(FamilySymbol))
    .OfCategory(BuiltInCategory.OST_Windows)
    .Cast<FamilySymbol>()
    .FirstOrDefault(s => s.Name == typeName && s.FamilyName == baseSymbol.FamilyName);

if (customSymbol == null)
{
    customSymbol = (FamilySymbol)baseSymbol.Duplicate(typeName);

    var wParam = customSymbol.LookupParameter("Width")
              ?? customSymbol.get_Parameter(BuiltInParameter.FAMILY_WIDTH_PARAM);
    if (wParam != null && !wParam.IsReadOnly)
        wParam.Set(widthFt);

    var hParam = customSymbol.LookupParameter("Height")
              ?? customSymbol.get_Parameter(BuiltInParameter.FAMILY_HEIGHT_PARAM);
    if (hParam != null && !hParam.IsReadOnly)
        hParam.Set(heightFt);
}

if (!customSymbol.IsActive)
    customSymbol.Activate();

// --- Find level ---
Level level = document.GetElement(hostWall.LevelId) as Level;
if (level == null)
{
    level = new FilteredElementCollector(document)
        .OfClass(typeof(Level))
        .Cast<Level>()
        .OrderBy(l => l.Elevation)
        .FirstOrDefault();
}

// --- Create family instance ---
var location = new XYZ(locX, locY, level.Elevation + sillHeight);

var instance = document.Create.NewFamilyInstance(
    location,
    customSymbol,
    hostWall,
    Autodesk.Revit.DB.Structure.StructuralType.NonStructural
);

if (instance == null)
    return new { error = "Failed to create window instance" };

// --- Set sill height ---
var sillParam = instance.get_Parameter(BuiltInParameter.WINDOW_SILL_HEIGHT_PARAM)
             ?? instance.LookupParameter("Sill Height");
if (sillParam != null && !sillParam.IsReadOnly)
    sillParam.Set(sillHeight);

// --- Handle facing ---
if (facingFlipped)
    instance.flipFacing();

return new
{
    elementId = instance(long)instance.Id.Value,
    typeId = customSymbol(long)instance.Id.Value,
    typeName = typeName,
    width_mm = widthMm,
    height_mm = heightMm,
    sill_mm = Math.Round(sillHeight * 304.8),
    familyName = customSymbol.FamilyName,
    hostWallId = hostWallId,
};
