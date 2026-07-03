// create_walls_from_json.cs
// Batch create walls from pipeline JSON. Accepts a JSON string describing
// wall segments with metric coordinates, creates them all in one transaction.
//
// @param parameters[0] (string) jsonStr - JSON array of wall objects:
//   [{
//     "x1": -3.0, "y1": -2.0,    // start point (meters)
//     "x2":  3.0, "y2": -2.0,    // end point (meters)
//     "thickness": 0.2,           // wall thickness (meters)
//     "height": 2.8               // wall height (meters)
//   }, ...]
// @returns { created: N, wallIds: [...], errors: [...] }
//
// Usage:
//   revit_send_code_to_revit(code=<this file>, parameters=['[{"x1":0,...}]'])

const double MetersToFeet = 3.280839895013123;
double M(double m) => m * MetersToFeet;

string jsonStr = Convert.ToString(parameters[0]);
var wallsData = JsonConvert.DeserializeObject<List<Dictionary<string, double>>>(jsonStr);

if (wallsData == null || wallsData.Count == 0)
    return new { error = "No wall data in JSON" };

// --- Get or create level ---
Level level = new FilteredElementCollector(document)
    .OfClass(typeof(Level))
    .Cast<Level>()
    .OrderBy(l => l.Elevation)
    .FirstOrDefault();

if (level == null)
{
    level = Level.Create(document, 0);
    level.Name = "BIM-Recon Level 0";
}

// --- Get or create wall type by thickness ---
WallType GetWallType(double thicknessM)
{
    string typeName = "BIM-Recon Wall " + Math.Round(thicknessM, 3).ToString("0.###") + "m";
    var existing = new FilteredElementCollector(document)
        .OfClass(typeof(WallType))
        .Cast<WallType>()
        .FirstOrDefault(t => t.Name == typeName);
    if (existing != null) return existing;

    var baseType = new FilteredElementCollector(document)
        .OfClass(typeof(WallType))
        .Cast<WallType>()
        .FirstOrDefault(t => t.Kind == WallKind.Basic);
    if (baseType == null) return null;

    var dup = (WallType)baseType.Duplicate(typeName);
    var structure = dup.GetCompoundStructure();
    if (structure != null && structure.LayerCount > 0)
    {
        structure.SetLayerWidth(0, M(thicknessM));
        dup.SetCompoundStructure(structure);
    }
    return dup;
}

// --- Create walls ---
var wallIds = new List<int>();
var errors = new List<string>();

for (int i = 0; i < wallsData.Count; i++)
{
    try
    {
        var w = wallsData[i];
        var line = Line.CreateBound(
            new XYZ(M(w["x1"]), M(w["y1"]), 0),
            new XYZ(M(w["x2"]), M(w["y2"]), 0)
        );

        double thickness = w.ContainsKey("thickness") ? w["thickness"] : 0.2;
        double height = w.ContainsKey("height") ? w["height"] : 2.8;

        var wallType = GetWallType(thickness);
        if (wallType == null)
        {
            errors.Add($"Wall {i}: no base wall type found");
            continue;
        }

        var wall = Wall.Create(document, line, wallType.Id, level.Id, M(height), 0.0, false, false);
        wallIds.Add(wall.Id.GetValue());
    }
    catch (Exception ex)
    {
        errors.Add($"Wall {i}: {ex.Message}");
    }
}

return new
{
    created = wallIds.Count,
    wallIds = wallIds,
    errors = errors,
    levelName = level.Name,
};
