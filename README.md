# 3DGS → BIM 自动重建系统

从消费级手机视频/RGB-D 数据，通过 3D Gaussian Splatting + SceneSplat 语义特征 + 虚拟激光扫描，自动提取墙体几何并导入 Revit。

## 系统架构

```
手机视频 → COLMAP → nerfstudio 3DGS训练 → gsplat场景
                                                  │
                                    SceneSplat PT-v3 推理 → feat.pt
                                                  │
                           ┌──────────────────────┼──────────────────────┐
                           ▼                      ▼                      ▼
                    虚拟激光扫描            语义高斯查询            底图引导拟合
                 (virtual_scanner)       (query_semantics)      (fit_walls_guided)
                           │                      │                      │
                           ▼                      ▼                      ▼
                    墙线提取管线            WallFitter RANSAC       FloorPlanGuidedFitter
                 (wall_line_extractor)     (盲拟合，无底图)         (走廊+直方图峰值)
                           │                      │                      │
                           └──────────┬───────────┴──────────┬───────────┘
                                      ▼                      ▼
                              wall_lines.json          Revit MCP 工具
                              wall_lines_topdown.png   create_line_based_element
```

## 已实现功能

### P0：基础设施
- **FloorPlan 契约**（`bim_recon/floorplan.py`）：WallSegment、Opening、ManualProvider，支持 JSON 输入和矩形房间快速生成。
- **Revit C# 代码生成**（`bim_recon/revit_code.py`）：FloorPlan → Revit API C#（墙、板、门窗洞口）。
- **差异报告**（`bim_recon/diff_report.py`）：底图墙线 vs 检测墙线的匹配与差异输出。

### P1：3DGS + 语义
- **GSScene**（`bim_recon/gs_scene.py`）：加载 PLY / SceneSplat .npy 场景，gsplat 渲染（RGB+ED），语义查询。支持 `from_npy()` 加载 post-activation 格式数据。
- **SemanticQuerier**（`bim_recon/semantics.py`）：加载 SceneSplat `feat.pt` + SigLIP2 文本嵌入，三种查询模式（dominant/threshold/top_percent）。9 类 BIM 词表（wall/floor/ceiling/door/window/column/beam/stairs/furniture）。
- **MCP Server**（`bim_recon/mcp_gs.py`）：**9 个工具**暴露 3DGS 场景给 VLM/Agent——get_scene_info、list_cameras、render_from_pose、get_depth_grid、select_cluster、query_semantics、render_semantic_overlay、fit_walls、fit_walls_guided。
- **COLMAP + 训练包装**（`bim_recon/colmap_runner.py`、`scripts/train_gs.py`）：包装 nerfstudio 命令。

### P2：墙体重建
- **WallFitter**（`bim_recon/wall_fitter.py`）：迭代 RANSAC + 遮挡补全合并 + 重力对齐 + 端点精修 + 高度提取。无底图盲拟合。
- **FloorPlanGuidedFitter**（`bim_recon/wall_fitter.py`）：走廊筛选 + 固定法向直方图峰值拟合。需用户提供底图 JSON。
- **FloorPlan 自动配准**（`bim_recon/floorplan_registration.py`）：PCA 旋转 + 90° 候选搜索 + 平移网格搜索 + 地板多边形评分。
- **虚拟激光扫描器**（`bim_recon/virtual_scanner.py`）：从 3DGS 深度渲染模拟 2D 激光扫描，多视角拼接 360° 极坐标扫描，每个扫描点携带 feat.pt 语义标签。
- **栅格化墙线提取**（`bim_recon/wall_line_extractor.py`）：多高度扫描 → DBSCAN 去噪 → 栅格化 + 形态学闭运算 → 轮廓提取 → Douglas-Peucker → RANSAC/PCA 精修 → 闭合墙线多边形。

### P2.5：VLM 验证元素提取
- **候选提取器**（`bim_recon/candidate_extractor.py`）：从多高度扫描 + feat.pt 语义标签提取门/窗/家具候选位置。投影到墙线 + 间隙聚类 + 极坐标 (θ, r) 计算。支持结构构件（投影到墙）和自由构件（DBSCAN 聚类）。
- **VLM 验证器**（`bim_recon/vlm_verifier.py`）：极坐标→相机位姿映射 → 3DGS 渲染针对性图像 → Ollama gemma4:12b VLM 确认/排除。两阶段检测：feat.pt 高召回找候选，VLM 高精度做判定。
- **端到端 CLI**（`scripts/verify_elements.py`）：加载场景 → 扫描 → 候选提取 → 预过滤 → VLM 验证 → 结果 JSON。

### Revit 集成
- 通过 `mcp-servers-for-revit` 的 MCP 工具（26 个），直接在 Revit 中创建墙、板、门窗等原生图元。
- 已验证：`revit_create_line_based_element` 成功创建墙体（6 面底图引导墙 + 4 面正方形测试房间）。

## 快速开始

### 环境要求
- Python 3.11+（conda 环境 `bim-recon`）
- PyTorch 2.7+ with CUDA 12.8
- gsplat 1.4.0（首次运行需 MSVC JIT 编译）
- OpenCV、scikit-learn、shapely、open3d、matplotlib
- **Windows** + Visual Studio 2022（gsplat JIT 编译需要 vcvars64）
- Revit + `mcp-servers-for-revit`（可选，用于 Revit 图元创建）

### 1. 启动 3DGS MCP Server

```powershell
# 方法 A：使用启动脚本（自动配置 vcvars64 + conda）
scripts\run_mcp_gs.bat

# 方法 B：手动启动（需要先初始化 vcvars64）
python -m bim_recon.mcp_gs ^
    --data-dir data ^
    --feat output/data_feat.pt ^
    --text-emb data/bim_text_emb.pt ^
    --class-names data/bim_class_names.json
```

MCP Server 暴露 9 个工具，可在 opencode/Claude Desktop 中配置使用。

### 2. 虚拟激光扫描（从 3DGS 提取 2D 雷达图）

```powershell
# 需要先初始化 vcvars64（gsplat JIT）
cmd /c "\"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/virtual_scan_probe.py"
```

**输出**：
- `output/virtual_scan_h1.5m.png` — 雷达极坐标图 + 俯视散点图（语义颜色编码）
- `output/virtual_scan_h1.5m.json` — 原始扫描数据（含 semantic_labels）

### 3. 多高度墙线提取（从扫描点自动提取闭合墙线）

```powershell
cmd /c "\"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/wall_line_probe.py"
```

**输出**：
- `output/wall_lines.json` — 墙线端点坐标（闭合多边形）
- `output/wall_lines_topdown.png` — 俯视墙线图（红色墙线 + 蓝色扫描点）

**管线**：8 高度扫描 → 23K 语义过滤点 → DBSCAN 去噪 → 0.05m/px 栅格化 → 7×7 闭运算 → findContours → Douglas-Peucker → RANSAC/PCA 精修 → 11 面闭合墙线

### 4. 底图引导墙拟合（用户提供底图 JSON）

将底图保存为 JSON（ManualProvider 格式）：

```json
{"walls": [
    {"x1": 0, "y1": 0, "x2": 10.0, "y2": 0},
    {"x1": 10.0, "y1": 0, "x2": 10.0, "y2": 8.0},
    {"x1": 10.0, "y1": 8.0, "x2": 0, "y2": 8.0},
    {"x1": 0, "y1": 8.0, "x2": 0, "y2": 0}
]}
```

通过 MCP 工具 `fit_walls_guided(floorplan_json=...)` 调用，或运行探针：

```powershell
cmd /c "\"...\vcvars64.bat\" && python scripts/fit_walls_guided_probe.py"
```

### 5. VLM 验证元素提取（门/窗检测）

两阶段检测：feat.pt 语义标签找候选 → 3DGS 渲染 → Ollama VLM 确认。

```powershell
# 需要 vcvars64 + Ollama gemma4:12b 运行中
cmd /c "\"...\vcvars64.bat\" && python scripts/verify_elements.py --name room0 --element door"
```

**输出**：
- `output/room0/doors_verified.json` — 确认/排除结果 + VLM 描述
- `output/room0/verify_door/*.png` — 每个候选的针对性渲染图

**room0 实测**：14 个 feat.pt 门候选 → 预过滤 5 个 → VLM 确认 2 个真门，排除 3 个（2 个实为窗，1 个实为画）

### 6. 在 Revit 中创建墙体

通过 `mcp-servers-for-revit` MCP 工具，将提取的墙线创建为 Revit 原生墙体：

```python
# 通过 MCP 工具调用（每面墙逐个创建避免超时）
revit_create_line_based_element(data=[{
    "category": "OST_Walls",
    "locationLine": {"p0": {"x": 1569, "y": 8226, "z": 35}, "p1": {"x": 1597, "y": -1922, "z": 35}},
    "thickness": 200,
    "height": 2540,
    "baseLevel": 0,
    "baseOffset": 0
}])
# 坐标单位：毫米（米 × 1000）
```

### 7. 手量底图 → Revit C# 脚本

```bash
python scripts/manual_to_revit_code.py examples/manual-room.json -o output/manual-room.cs
```

生成的 C# 脚本可通过 `send_code_to_revit` 工具在 Revit 中执行。

## 运行测试

```bash
pytest -q
```

当前 107 个测试（1 个需 MSVC 环境跳过）：

| 测试文件 | 覆盖 | 状态 |
|---|---|---|
| `tests/test_floorplan.py` | FloorPlan 契约、ManualProvider、Revit C# 生成、差异报告 | 14/14 通过 |
| `tests/test_gs_scene.py` | GSScene 相机工具、合成渲染、PLY 往返 | 8/9 通过（1 需 MSVC） |
| `tests/test_semantics.py` | SemanticQuerier init/query/dominant/top_percent | 18/18 通过 |
| `tests/test_wall_fitter.py` | WallFitter RANSAC/merge/align/refine/height | 16/16 通过 |
| `tests/test_floorplan_guided.py` | FloorPlanGuidedFitter + register_floorplan | 8/8 通过 |
| `tests/test_candidate_extractor.py` | 候选提取（投影/聚类/多墙/DBSCAN自由构件/过滤）| 17/17 通过 |
| `tests/test_vlm_verifier.py` | 极坐标/视角映射(X/Y/Z-up)/VLM响应解析/prompt/Mock端到端 | 25/25 通过 |

MCP 工具集成测试（需 MSVC）：

```powershell
cmd /c "\"...\vcvars64.bat\" && python scripts/test_mcp_gs.py"
```

## 项目结构

```
bim_recon/
├── gs_scene.py              # 3DGS 场景加载 + gsplat 渲染 + 语义查询
├── semantics.py             # SemanticQuerier (feat.pt + SigLIP2 文本嵌入)
├── mcp_gs.py                # MCP Server (9 工具)
├── wall_fitter.py           # WallFitter + FloorPlanGuidedFitter
├── floorplan_registration.py # 底图→3DGS 自动配准
├── virtual_scanner.py       # 虚拟 2D 激光扫描器
├── wall_line_extractor.py   # 栅格化+形态学+轮廓+DP+RANSAC 墙线提取
├── candidate_extractor.py   # 元素候选提取（门/窗/家具，feat.pt 语义+墙线投影）
├── vlm_verifier.py          # VLM 验证（极坐标→渲染→Ollama gemma4:12b 确认/排除）
├── floorplan.py             # FloorPlan 契约 + ManualProvider
├── revit_code.py            # FloorPlan → Revit C# 代码生成
├── diff_report.py           # 底图 vs 检测差异报告
└── colmap_runner.py         # COLMAP 包装

scripts/
├── verify_elements.py       # 端到端 VLM 验证 CLI（扫描→候选→VLM 判定）
├── generate_walls.py        # 通用墙线生成（扫描→栅格化→墙线 JSON+PNG）
├── virtual_scan_probe.py    # 虚拟扫描探针 → 雷达 PNG + JSON
├── wall_line_probe.py       # 墙线提取探针 → 墙线 JSON + 俯视图
├── fit_walls_guided_probe.py # 底图引导拟合探针
├── encode_bim_labels.py     # SigLIP2 文本嵌入生成器
├── train_gs.py              # nerfstudio 训练包装
├── test_mcp_gs.py           # MCP 工具集成测试
└── run_mcp_gs.bat           # MCP Server 启动器

data/                        # SceneSplat .npy 数据 + BIM 词表
output/                      # feat.pt + 生成的扫描图/墙线
```

## 关键技术栈

| 组件 | 技术 | 用途 |
|---|---|---|
| 3DGS 训练 | nerfstudio / splatfacto | 从图像训练 3D 高斯场景 |
| 渲染引擎 | gsplat 1.4.0 | CUDA 加速光栅化（RGB+Depth） |
| 语义特征 | SceneSplat (ICCV'25 Oral) | PT-v3 预训练编码器 → per-Gaussian 768 维语言特征 |
| 文本对齐 | SigLIP2 | BIM 词表文本嵌入，与 feat.pt 零样本对齐 |
| 虚拟扫描 | gsplat depth rendering | 从任意位姿渲染深度 → 模拟 LiDAR |
| 墙线提取 | OpenCV + scikit-learn | 栅格化 + 形态学 + 轮廓 + Douglas-Peucker + PCA |
| Revit 桥接 | mcp-servers-for-revit | C# MCP Server，26 个工具直接操作 Revit API |
| VLM 决策 | Claude / GPT-4o | 通过 MCP 工具巡视场景、提取墙体 |

## 当前限制

- **单房间 MVP**：仅支持单房间墙体提取，不支持多房间拼接。
- **COLMAP + nerfstudio 训练**：需用户手动运行（agent 不代跑）。
- **gsplat JIT**：首次运行需 MSVC（vcvars64）环境。
- **精度**：厘米级（依赖 SfM 度量对齐质量），非施工级。
- **LiDAR Provider**：P3 规划中，尚未实现。

## 下一步

- 实现门窗洞口检测（在闭合墙线上分析扫描点的语义间隙）
- 多房间拼接
- LiDARProvider（ROS2 `/scan` → split-and-merge 墙线）
- 精度评估报告

---

详见 `PLAN.md` 获取完整架构设计、24 周路线图和技术决策记录。
