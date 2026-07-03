// delete_elements_by_category.cs
// Delete all elements of a given category. Useful for cleanup before
// re-running pipeline. Optionally filter by a name prefix (e.g. "BIM-Recon").
//
// @param parameters[0] (string) category     - e.g. "OST_Doors", "OST_Windows", "OST_Walls"
// @param parameters[1] (string) namePrefix   - Only delete elements whose type name starts with this (optional, "" = all)
// @returns { deleted: N, elementIds: [...], skipped: N }
//
// Usage:
//   revit_send_code_to_revit(code=<this file>, parameters=["OST_Doors", "BIM-Recon"])

string categoryName = parameters.Length > 0 ? Convert.ToString(parameters[0]) : "OST_Doors";
string namePrefix = parameters.Length > 1 ? Convert.ToString(parameters[1]) : "";

BuiltInCategory bic;
if (!Enum.TryParse(categoryName.Replace(".", ""), true, out bic))
    return new { error = $"Unknown category: {categoryName}" };

var collector = new FilteredElementCollector(document)
    .OfCategory(bic)
    .WhereElementIsNotElementType();

var toDelete = new List<ElementId>();
int skipped = 0;

foreach (Element elem in collector)
{
    if (!string.IsNullOrEmpty(namePrefix))
    {
        // Check type name prefix
        var typeElem = document.GetElement(elem.TypeId) as FamilySymbol;
        string typeName = typeElem != null ? typeElem.Name : "";
        if (!typeName.StartsWith(namePrefix))
        {
            skipped++;
            continue;
        }
    }
    toDelete.Add(elem.Id);
}

foreach (var id in toDelete)
    document.Delete(id);

return new
{
    category = categoryName,
    namePrefix = namePrefix,
    deleted = toDelete.Count,
    skipped = skipped,
};
