// query_family_types.cs
// List all loaded FamilySymbol types for a given built-in category.
//
// @param parameters[0] (string) category - Built-in category name, e.g. "OST_Doors", "OST_Windows", "OST_Walls"
// @returns List of { familyName, symbolName, typeId, width, height } for each type
//
// Usage:
//   revit_send_code_to_revit(code=<this file>, parameters=["OST_Doors"])

string categoryName = parameters.Length > 0 ? Convert.ToString(parameters[0]) : "OST_Doors";

// Parse category
BuiltInCategory bic;
if (!Enum.TryParse(categoryName.Replace(".", ""), true, out bic))
{
    return new { error = $"Unknown category: {categoryName}" };
}

var collector = new FilteredElementCollector(document)
    .OfClass(typeof(FamilySymbol))
    .OfCategory(bic);

var results = new List<object>();
foreach (FamilySymbol sym in collector)
{
    double w = 0, h = 0;
    // Try to read Width and Height type parameters
    var wParam = sym.get_Parameter(BuiltInParameter.FAMILY_WIDTH_PARAM);
    var hParam = sym.get_Parameter(BuiltInParameter.FAMILY_HEIGHT_PARAM);
    if (wParam != null) w = wParam.AsDouble();
    if (hParam != null) h = hParam.AsDouble();

    // Also check for type parameters by name (common in door/window families)
    if (w == 0)
    {
        var wp = sym.LookupParameter("Width");
        if (wp != null && wp.StorageType == StorageType.Double) w = wp.AsDouble();
    }
    if (h == 0)
    {
        var hp = sym.LookupParameter("Height");
        if (hp != null && hp.StorageType == StorageType.Double) h = hp.AsDouble();
    }

    results.Add(new
    {
        familyName = sym.FamilyName,
        symbolName = sym.Name,
        typeId = sym(long)sym.Id.Value,
        width_mm = Math.Round(w * 304.8, 1),
        height_mm = Math.Round(h * 304.8, 1),
        active = sym.IsActive,
    });
}

return new
{
    category = categoryName,
    count = results.Count,
    types = results,
};
