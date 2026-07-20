// create_directshape_from_mesh.cs
// Create a Revit DirectShape from raw mesh data (vertices + faces).
//
// @param parameters[0] (string) payload - JSON string OR file path to JSON.
//   File path recommended for large meshes (>1MB JSON stalls MCP stdio).
//   JSON: {"name","category","vertices":[x,y,z,...],"faces":[i,j,k,...]}
//   Vertices in Revit internal units (feet).
// @returns { status, elementId, name, vertexCount, faceCount }

string param0 = Convert.ToString(parameters[0]);
string jsonStr;
if ((param0.Length > 2 && param0[1] == ':' && (param0[2] == '\\' || param0[2] == '/'))
    || param0.StartsWith("/"))
{
    jsonStr = System.IO.File.ReadAllText(param0);
}
else
{
    jsonStr = param0;
}

var doc = System.Text.Json.JsonDocument.Parse(jsonStr);
var root = doc.RootElement;

string dsName = root.TryGetProperty("name", out var nameEl) ? nameEl.GetString() : "B-class Mesh";
string categoryStr = root.TryGetProperty("category", out var catEl) ? catEl.GetString() : "OST_GenericModel";

var vertsEl = root.GetProperty("vertices");
int vertexCount = vertsEl.GetArrayLength() / 3;
if (vertexCount < 3)
    return new { error = "Need at least 3 vertices" };

var verts = new double[vertexCount * 3];
int vi = 0;
foreach (var v in vertsEl.EnumerateArray())
    verts[vi++] = v.GetDouble();

var facesEl = root.GetProperty("faces");
int faceCount = facesEl.GetArrayLength() / 3;
if (faceCount < 1)
    return new { error = "Need at least 1 face" };

var faces = new int[faceCount * 3];
int fi = 0;
foreach (var f in facesEl.EnumerateArray())
    faces[fi++] = f.GetInt32();

BuiltInCategory bic;
try { bic = (BuiltInCategory)System.Enum.Parse(typeof(BuiltInCategory), categoryStr); }
catch { bic = BuiltInCategory.OST_GenericModel; }
ElementId categoryId = new ElementId(bic);

Level level = new FilteredElementCollector(document)
    .OfClass(typeof(Level))
    .Cast<Level>()
    .OrderBy(l => l.Elevation)
    .FirstOrDefault();
if (level == null) {
    level = Level.Create(document, 0);
}

// --- Build TessellatedShape ---
var builder = new TessellatedShapeBuilder();
builder.OpenConnectedFaceSet(false);

for (int i = 0; i < faceCount; i++) {
    int i0 = faces[i * 3];
    int i1 = faces[i * 3 + 1];
    int i2 = faces[i * 3 + 2];
    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
        continue;

    XYZ v0 = new XYZ(verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]);
    XYZ v1 = new XYZ(verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]);
    XYZ v2 = new XYZ(verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);

    double area = v1.Subtract(v0).CrossProduct(v2.Subtract(v0)).GetLength();
    if (area < 1e-9)
        continue;

    builder.AddFace(new TessellatedFace(new List<XYZ> { v0, v1, v2 }, ElementId.InvalidElementId));
}

builder.CloseConnectedFaceSet();
builder.Target = TessellatedShapeBuilderTarget.Mesh;
builder.Fallback = TessellatedShapeBuilderFallback.Salvage;
builder.Build();

var result = builder.GetBuildResult();
if (result == null || result.GetGeometricalObjects().Count == 0)
    return new { error = "Failed to build tessellated shape" };

// --- Create DirectShape ---
var shapeList = result.GetGeometricalObjects().ToList();
DirectShape ds = DirectShape.CreateElement(document, categoryId);
ds.Name = dsName;
ds.SetShape(shapeList);

return new {
    status = "ok",
    elementId = ds.Id.Value,
    name = dsName,
    vertexCount = vertexCount,
    faceCount = faceCount,
};
