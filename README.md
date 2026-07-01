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
- Ollama + gemma4:12b（VLM 验证，本地部署）
- **Windows** + Visual Studio 2022（gsplat JIT 编译需要 vcvars64）
- Revit + `mcp-servers-for-revit`（可选，用于 Revit 图元创建）

### 1. 一键运行完整管线（推荐）

```powershell
cmd /c "\"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/run_pipeline.py --name room0"
```

**输入**：`data/room0/point_cloud_30000.ply` + `data/room0/room0_feat.pt`

**输出**（`output/room0/`）：
- `wall_lines_snapped.json` — 墙线端点（闭合多边形）
- `doors_verified.json` — VLM 确认的门
- `windows_verified.json` — VLM 确认的窗
- `pipeline_report.json` — 完整管线报告
- `wall_lines_topdown.png` — 墙线俯视图

**管线流程**：加载场景 → 12 高度雷达扫描 → 墙线提取（栅格+形态学+轮廓+PCA）→ 门检测（feat.pt 候选 → 预过滤 → Ollama VLM 验证）→ 窗检测 → 结果 JSON

**跳过 VLM（仅渲染）**：
```powershell
cmd /c "\"...\vcvars64.bat\" && python scripts/run_pipeline.py --name room0 --skip-vlm"
```

**指定检测的构件类型**：
```powershell
python scripts/run_pipeline.py --name room0 --elements door window column
```

## 运行测试

```bash
pytest -q
```

当前 120 个测试（1 个需 MSVC 环境跳过）：

| 测试文件 | 覆盖 | 状态 |
|---|---|---|
| `tests/test_floorplan.py` | FloorPlan 契约、ManualProvider、Revit C# 生成、差异报告 | 14/14 通过 |
| `tests/test_gs_scene.py` | GSScene 相机工具、合成渲染、PLY 往返 | 8/9 通过（1 需 MSVC） |
| `tests/test_semantics.py` | SemanticQuerier init/query/dominant/top_percent | 18/18 通过 |
| `tests/test_wall_fitter.py` | WallFitter RANSAC/merge/align/refine/height | 16/16 通过 |
| `tests/test_floorplan_guided.py` | FloorPlanGuidedFitter + register_floorplan | 8/8 通过 |
| `tests/test_candidate_extractor.py` | 候选提取（投影/聚类/多墙/DBSCAN自由构件/过滤）| 17/17 通过 |
| `tests/test_vlm_verifier.py` | 极坐标/视角映射(X/Y/Z-up)/VLM响应解析/prompt/Mock端到端 | 25/25 通过 |
| `tests/test_element_config.py` | 元素类型配置注册表（查找/属性/输出名/frozen）| 14/14 通过 |

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
├── element_config.py        # 元素类型配置注册表（door/window/column/furniture）
├── floorplan.py             # FloorPlan 契约 + ManualProvider
├── revit_code.py            # FloorPlan → Revit C# 代码生成
├── diff_report.py           # 底图 vs 检测差异报告
└── colmap_runner.py         # COLMAP 包装

scripts/
├── run_pipeline.py          # 主流程：scene → walls → doors → windows → JSON（唯一入口）
├── generate_walls.py        # 单独提取墙线
├── final_radar.py           # 可视化：4 面板管线结果图
├── encode_bim_labels.py     # SigLIP2 文本嵌入生成器（工具）
├── manual_to_revit_code.py  # 手量底图 → Revit C# 脚本（工具）
├── test_mcp_gs.py           # MCP 工具集成测试
└── train_gs.py              # nerfstudio 训练包装

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
