# AGENTS.md — Project Knowledge Base

## Environment Paths

| Item | Path |
|---|---|
| **Revit 2026** | `F:\Software\AutoDesk\Revit 2026\Revit.exe` |
| **Revit Addins (2026)** | `%APPDATA%\Autodesk\Revit\Addins\2026\` |
| **mcp-servers-for-revit** | `G:\TJ\BIM\mcp-servers-for-revit\` |
| **.NET SDK** | 10.0.203 |
| **Conda: bim-recon** | `G:\Miniconda3\envs\bim-recon\python.exe` (gsplat, torch 2.7) |
| **Conda: transformerv** | `G:\Miniconda3\envs\transformerv\python.exe` (Falcon-Perception, torch 2.11) |

## Revit MCP Build

- 唯一安装版本: **Revit 2026** (R26, .NET 8.0, `net8.0-windows10.0.19041.0`)
- 构建命令: `.\scripts\build_deploy_revit.ps1` (一键构建 C# + TypeScript + 部署)
- 构建流程 (5 步): Kill Revit → Build C# → Build TS Server → Deploy → Verify
- Debug 构建自动部署到 Addins 文件夹
- 输出暂存: `plugin\bin\AddIn 2026 Debug R26\`
- CommandSet DLL: `commandset\bin\Debug R26\2026\`
- TS Server: `server\build\index.js`
- opencode MCP 配置 (`~/.config/opencode/opencode.json`): revit server 指向本地 `node server\build\index.js`

## Revit C# 脚本库

- `revit_scripts/` — 通过 `send_code_to_revit` MCP 工具执行的 C# 模板
- `bim_recon/revit_runner.py` — Python 加载器/运行器
- 注意：动态执行的代码中 `ElementId.Value`（返回 long）可用，但 `.IntegerValue` / `.GetValue()` / `.GetIntValue()` 不可用（Revit 2026 移除了 IntegerValue，GetValue/GetIntValue 是 CommandSet 扩展方法）
- 注意：动态执行中 `Element.TypeId` / `Element.Host` 不可用（是扩展方法），需用 `element.GetTypeId()` 和 `(element as FamilyInstance).Host`
- 注意：中文 Revit 项目中 `LookupParameter("Width")` 找不到参数，参数名是中文（"宽度"/"高度"/"窗台高度"）。`BuiltInParameter.FAMILY_WIDTH_PARAM` 等枚举是语言无关的，始终可用

## Revit MCP 工具使用要点

- `create_point_based_element` 的 `typeId=-1` 可能找不到族类型（因为没有任何类型标记为 Active），应始终指定具体 `typeId`
- 门类型（OST_Doors）： typeId 94654 = "单扇 - 与墙齐 750 x 2000mm"
- 窗类型（OST_Windows）： typeId 93304 = "固定 0915 x 1220mm"
- 自定义尺寸已支持：设置 `width` + `height`（mm）会自动复制族类型并设置尺寸参数，类型名格式 "Custom {W}x{H}mm"
- `parameters: {"sillHeight": N}` 设置窗台高度（mm），仅门窗有效

## 项目约定

- 永远不要在意字符/字体警告
- VLM 固定使用 Ollama gemma4:12b
- 两环境不可合并：bim-recon (gsplat) vs transformerv (falcon_perception)
