# Revit C# Script Library

Reusable C# scripts for `send_code_to_revit` MCP tool. Each `.cs` file is a
self-contained method body that gets injected into:

```csharp
public static object Execute(Document document, object[] parameters)
{
    // <script content here>
}
```

## Quick Usage (AI Agent)

1. **Read** the desired `.cs` file from this folder.
2. **Call** `revit_send_code_to_revit` with:
   - `code` = file content
   - `parameters` = array of values matching the script's `@param` comments
3. Transaction is auto-wrapped by the handler.

```
# Example: query available door family types
code = read("revit_scripts/query_family_types.cs")
result = revit_send_code_to_revit(code=code, parameters=["OST_Doors"])
```

## Quick Usage (Pipeline)

```python
from bim_recon.revit_runner import RevitScriptRunner

runner = RevitScriptRunner()
result = runner.run("create_custom_door", parameters=[
    337595,       # hostWallId
    5.0,          # locationX (feet)
    3.0,          # locationY (feet)
    0.0,          # sillHeight (feet)
    3.0,          # width (feet)
    7.0,          # height (feet)
    false,        # facingFlipped
])
```

## File Naming

| Pattern | Purpose |
|---|---|
| `create_*.cs` | Create elements (doors, windows, walls, floors) |
| `query_*.cs` | Query/read data from the model (family types, elements) |
| `modify_*.cs` | Modify existing elements (parameters, colors) |
| `delete_*.cs` | Delete elements |
| `test_*.cs` | Quick test/debug scripts |

## Script Conventions

Each script file MUST have:

1. **Header block** with `@param` comments documenting parameters:
   ```csharp
   // @param parameters[0] (long)   hostWallId - Wall element ID
   // @param parameters[1] (double) locationX  - X position in feet
   // @returns { "elementId": 12345, "typeName": "..." }
   ```

2. **Self-contained** logic — no external dependencies beyond the
   auto-imported namespaces (System, System.Linq, Autodesk.Revit.DB,
   Autodesk.Revit.UI, System.Collections.Generic).

3. **Return value** — always return a serializable object (anonymous
   types, dictionaries, or JSON strings).

## Units

All geometric values are in **feet** (Revit internal units) unless noted.
Conversion: `1 m = 3.28084 ft`, `1 mm = 1/304.8 ft`.

## Available Scripts

| Script | Purpose |
|---|---|
| `query_family_types.cs` | List all family types for a category |
| `create_custom_door.cs` | Create a door with custom width/height/sill |
| `create_custom_window.cs` | Create a window with custom width/height/sill |
| `create_walls_from_json.cs` | Batch create walls from pipeline JSON |
| `delete_elements_by_category.cs` | Delete all elements of a category |
