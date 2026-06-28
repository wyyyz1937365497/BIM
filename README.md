# 3DGS -> BIM 自动重建系统

本仓库是 `PLAN.md` 的工程落地目录。目前完成的是 P0 阶段的一部分基础设施：`FloorPlan` 契约、手量底图 Provider、Revit C# 代码生成、差异报告与 3DGS MCP 工具边界脚手架。

注意：这不是完整项目交付，也还没有在 Windows + Revit 环境中验证原生 BIM 图元创建。

## 当前文件说明

| 路径 | 作用 |
| --- | --- |
| `PLAN.md` | 项目总计划与架构事实来源。包含 24 周路线、技术栈、风险和历史决策记录。 |
| `README.md` | 当前仓库状态说明，记录已经完成和未完成的内容。 |
| `bim_recon/__init__.py` | Python 包导出入口。 |
| `bim_recon/floorplan.py` | `FloorPlan` 契约、`WallSegment`、`Opening`、`FrameMeta`、`ManualProvider` 和合法性校验。 |
| `bim_recon/revit_code.py` | 将 `FloorPlan` 转成 Revit API C# 代码，用于 `mcp-servers-for-revit` 的 `send_code_to_revit` 工具。 |
| `bim_recon/diff_report.py` | 底图墙线与 3DGS/VLM 检出墙线的差异报告模块。MVP 策略是只报告，不自动改底图。 |
| `bim_recon/gs_mcp_scaffold.py` | 3DGS 侧 MCP 工具边界脚手架，定义 `render_from_pose`、`get_depth`、`select_cluster`、`report_diff` 等工具规格。 |
| `scripts/manual_to_revit_code.py` | CLI：读取手量 JSON，输出可交给 Revit MCP 执行的 C# 脚本。 |
| `examples/manual-room.json` | 手量矩形房间示例，包含 4 面墙、1 个门洞、1 个窗洞。 |
| `tests/test_floorplan.py` | Python 单元测试，覆盖 FloorPlan、ManualProvider、Revit C# 生成、差异报告和 3DGS MCP facade。 |
| `a.py` | 早期 IfcOpenShell 几何内核检查脚本。当前主线已转向 Revit API，不属于新主链路。 |
| `mcp-servers-for-revit/` | 计划使用的 C#/TypeScript Revit MCP 子模块目录。当前本地目录为空或未初始化，尚未接入。 |
| `mcp-server-for-revit-python/` | 旧 pyRevit/IronPython 路线遗留目录。根据 `PLAN.md` 后续决策，该路线已被 C# MCP 方案替代。 |

## 已完成内容

### 1. FloorPlan 契约

对应 `PLAN.md`：

- `§4.1 去耦：FloorPlan Provider`
- `§8 第 1 周可执行任务清单` 的第 3 项
- `附录 A：FloorPlan 契约`

完成情况：

- 定义了 `WallType`、`OpeningKind`、`WallSegment`、`Opening`、`FrameMeta`、`FloorPlan`。
- 实现了 `FloorPlanProvider` 抽象接口。
- 实现了 `ManualProvider`。
- 支持直接给 `walls/openings`。
- 支持通过 `rectangle.width/depth` 快速生成矩形房间。
- 支持 JSON 文件读取。
- 增加了墙长、墙厚、洞口宽度、洞口位置、窗台高度、门窗类型等合法性校验。
- 墙厚默认规则：承重墙 `0.24m`，隔墙 `0.12m`，未知墙按 `0.24m` 兜底。

### 2. 手量 JSON 示例

对应 `PLAN.md`：

- `§8 第 1 周可执行任务清单` 的第 3 项

完成情况：

- 新增 `examples/manual-room.json`。
- 示例描述一个 `5m x 4m` 的矩形房间。
- 包含 1 个门洞和 1 个窗洞。

### 3. Revit API C# 代码生成

对应 `PLAN.md`：

- 开头架构变更：从 IFC 改为 Revit API 原生图元。
- `§3 系统架构` 的 Revit MCP 服务器部分。
- `§12.3` 和 `§12.4` 中关于 `mcp-servers-for-revit` 的 C# MCP 路线。

完成情况：

- 新增 `bim_recon/revit_code.py`。
- 能把 `FloorPlan` 转成 Revit API C# 脚本。
- 生成逻辑包含：
- 创建或复用 Level。
- 创建 Wall。
- 根据墙厚尝试复制 WallType。
- 创建 Floor 作为地板。
- 创建 Floor 作为天花板占位表达。
- 通过 `doc.Create.NewOpening` 创建门窗洞口。
- 可选尝试放置宿主门窗族。

限制：

- 还没有在 Windows + Revit 中实机执行。
- 生成代码默认假设 MCP 执行上下文提供 `uiapp` 变量。
- 门窗族放置依赖当前 Revit 项目是否已加载对应族。

### 4. CLI：手量 JSON -> Revit C# 脚本

对应 `PLAN.md`：

- `§8 第 1 周可执行任务清单` 的第 2、3 项之间的桥接工具。

完成情况：

可以运行：

```bash
python scripts/manual_to_revit_code.py examples/manual-room.json
```

常用输出到文件：

```bash
python scripts/manual_to_revit_code.py examples/manual-room.json -o output/manual-room.cs
```

只开洞、不尝试放置门窗族：

```bash
python scripts/manual_to_revit_code.py examples/manual-room.json --no-hosted-families
```

### 5. 差异报告模块

对应 `PLAN.md`：

- `§2.3 已锁定的关键决策` 中“3DGS 有墙 / 底图没有：仅报告，不自动采纳”
- `§4.5 差异仅报告`
- `§13-14 P2 差异报告模块`

完成情况：

- 新增 `bim_recon/diff_report.py`。
- 输入：底图 `FloorPlan` 和检测出的墙线列表。
- 输出：底图中未被检测匹配的墙、检测中多出来但底图没有的墙。
- 匹配依据：中点距离 + 墙线方向夹角。
- 不自动修改 `FloorPlan`。

### 6. 3DGS MCP 工具边界脚手架

对应 `PLAN.md`：

- `§3 VLM 决策循环`
- `§5 技术栈与依赖清单`
- `附录 B：MCP 工具集`

完成情况：

- 新增 `bim_recon/gs_mcp_scaffold.py`。
- 定义了 3DGS 后端需要实现的最小协议：
- `render_from_pose`
- `get_depth`
- `select_cluster`
- 定义了工具规格：
- `render_from_pose`
- `get_depth`
- `select_cluster`
- `report_diff`

限制：

- 目前只是接口边界，不包含真实 `gsplat`、SAGA 或 Gaussian Grouping 实现。
- 没有加载 3DGS 模型。
- 没有启动 MCP server。

### 7. 本地测试

对应 `PLAN.md`：

- 属于 P0 工程质量保障，不等价于 Revit 实机验收。

当前测试命令：

```bash
pytest -q
```

当前结果：

```text
14 passed
```

## 尚未完成内容

以下是 `PLAN.md` 中仍未完成的主要部分。

### P0 未完成

- 未安装或验证 COLMAP。
- 未安装或验证 nerfstudio / splatfacto。
- 未安装或验证 gsplat。
- 未安装或验证 open3d。
- 未从手机视频跑通 COLMAP。
- 未训练 3DGS。
- 未用 gsplat 渲染 RGB+ED。
- 未做 Open3D RANSAC 墙面拟合。
- 未在 Windows + Revit 中通过 MCP 执行生成的 C#。
- 未验证 Revit 中墙、板、门窗洞口是否真实可编辑。

### P1 未完成

- 未接入 SAGA 或 Gaussian Grouping。
- 未实现 2D mask 到 3D Gaussian cluster 的真实选择。
- 未实现真正的 3DGS MCP server。
- 未接入 Claude / GPT-4o / Gemini 的 VLM 决策循环。
- 未实现 VLM 找墙、确认墙、回看 validate 的闭环。

### P2 未完成

- 未完成确定性墙拟合器。
- 未做重力对齐与拉伸。
- 未完善地板、天花板、门窗全要素的 Revit 实机验证。
- 未做 2-3 个房间复测。
- 未完成算法 MVP。

### P3 未完成

- 未实现 LiDARProvider。
- 未接入 ROS2 `/scan`。
- 未实现 split-and-merge 墙线提取。
- 未做 2D LiDAR 到 3DGS/FloorPlan 的配准。
- 未比较有无 LiDAR 的质量差异。

### P4 未完成

- 未做难例测试。
- 未做精度报告。
- 未做多房间接口占位之外的真实拼接。
- 未做演示视频、论文、报告或 Slides。

## 当前进度判断

当前完成度应理解为：

```text
P0 基础工程骨架完成一部分
```

不是：

```text
PLAN.md 全部完成
```

也不是：

```text
算法 MVP 达成
```

当前最接近完成的计划项是：

- `FloorPlan` 契约
- `ManualProvider`
- 手量输入到 Revit C# 代码生成
- 差异报告的纯 Python 版本
- 3DGS MCP 工具边界定义

## 下一步建议

推荐下一步优先级：

1. 在 Windows + Revit 环境中初始化 `mcp-servers-for-revit`，确认 `send_code_to_revit` 的真实执行上下文变量名。
2. 用 `examples/manual-room.json` 生成 C#，在 Revit 中实测墙、板、洞口能否创建。
3. 根据实测结果修正 `bim_recon/revit_code.py`。
4. 再推进 COLMAP / nerfstudio / gsplat 的 3DGS 链路。

