# 3DGS → BIM 自动重建系统 · 项目计划

> **状态**：计划已定稿，待启动
> **⚠️ 架构变更（2026-06-27）**：**放弃 IFC 交换生态**，改用 **pyRevit + Revit API** 直接在 Revit 中生成/编辑原生图元（原生可编辑、无 IFC 导入兼容性问题）。下文所有 IfcOpenShell/IFC 相关内容（§5、§8、§10.1、§12 等）视为**历史/已弃用**，以本条为准。FloorPlan 契约、3DGS、VLM/MCP 等其余架构不变。
> **日期**：2026-06-26
> **类型**：大学生创新训练计划（SITP）/ 大创
> **MVP 周期**：约 24 周（1–2 人）
> **本文件用途**：团队同步用的唯一事实来源（single source of truth）。实施时以本文件为准。

---

## 项目约定

- **永远不要在意字符/字体警告**（emoji 缺失、CJK glyph 警告等）。这些是 matplotlib 字体限制，不影响程序运行。不要为此浪费时间去修改。程序能正常运行即可。

---

## 1. 项目概述

用**消费级设备**（手机 + 一个 50 元 2D 旋转 LiDAR + 3D 打印支架）采集房间，以 **3D Gaussian Splatting (3DGS)** 作为"信息丰富的中间表示"，让**多模态视觉大模型 (VLM) 借助 MCP** 在 3DGS 场景中自由巡视、分割出符合建筑语义的结构构件，最终由 **pyRevit + Revit API** 直接在 Revit 中生成**可编辑的原生 BIM 图元**（不再走 IFC 交换）。

**目标用户**：设计师（快速现状测绘 + 后期可改图），**不是**施工级 BIM。

**精度定位**：厘米级（现状测绘 / 体量级），不追求毫米级施工精度。

---

## 2. MVP 范围与成功标准

### 2.1 MVP 必须达成
- **输入**：单房间的手机视频 + 一张水平底图（手量矩形 / 2D LiDAR 扫描，二选一）。
- **输出**：可在 **Revit 中打开并编辑**的 IFC，包含：
  - `IfcWall`（墙，SweptSolid 表示）
  - `IfcSlab`（地板 / 天花板）
  - `IfcColumn`（柱，若存在）
  - `IfcOpeningElement` + `IfcDoor` / `IfcWindow`（门窗洞口）
- **可编辑性**：在 Revit 中能拖动墙厚、移动门窗、删除构件而不报错。

### 2.2 MVP 不包含（写入未来工作）
- B 类复杂构件（管道、楼梯、异形件）的 mesh 回灌。
- 多房间 / 多层拼接（仅留接口占位）。
- 施工级精度。

### 2.3 已锁定的关键决策
| 决策点 | 结论 |
|---|---|
| 底图来源（去耦） | **FloorPlan Provider 接口**；MVP 实现 **Manual + LiDAR** 两个 Provider |
| 差异处理（3DGS 有墙 / 底图没有） | **仅报告**，不自动采纳 |
| 采集平台 | **无小车**；手机手持录像 + LiDAR 静止扫描 |
| 位姿来源 | **COLMAP** 起步（单房间够用），失败再上 ARCore / VINS-Fusion |
| 手机 IMU 用法 | MVP 仅作**重力对齐**，位姿交给 COLMAP |
| 是否自研 Android App | **不做**；手机自带录像即可（位姿不依赖图像匹配） |
| GPU | 两张 2080Ti 22G（单房间 3DGS 训练充裕） |
| **IFC 版本标准** | **IFC4**。原生 `IfcTriangulatedFaceSet` 便于未来 B 类 mesh 回灌（见 §10.1）。3D 不可见的真正根因是 Revit 导入后"Phase Created（创建阶段）"默认为不存在的阶段（改为"现有"即恢复），**与 IFC 版本无关**（见 §12.1） |

---

## 3. 系统架构

```
┌────────────────────────── 采集（一人即可） ──────────────────────────┐
│  ① 手机手持绕房间录像 ──────────────┐                                 │
│  ② 手机 IMU ────────────────────────┤ (离线处理，无需实时同步)        │
│  ③ 2D LiDAR 架三脚架静止扫描 ───────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────── 前端重建 ──────────────────────────────────┐
│  COLMAP (位姿) + Metric3D V2 (度量深度先验)                          │
│    → nerfstudio / splatfacto (深度正则 3DGS)                          │
│    → 语义高斯 (SAGA / Gaussian Grouping：把 SAM/CLIP 钉到每个高斯)    │
└──────────────────────────────────────────────────────────────────────┘
        │                                              ▲
        ▼                                              │ 底图权威(水平XY)
┌────────────────── 去耦：FloorPlan Provider ──────────────────────────┐
│  ManualProvider (手量矩形/多边形)  ── 零硬件, 第一天即可用             │
│  LiDARProvider  (ROS2 /scan → 墙线 + 门口缺口) ── 可并行/可砍         │
│  (未来) DrawingProvider (消防图/CAD/PDF 读比例)                       │
│        ↓ 统一契约 FloorPlan{wall_segments, openings, frame_meta}     │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────── VLM 决策循环 (via 双 MCP 服务器) ────────────────────┐
│  3DGS 侧 MCP: render_from_pose[gsplat] · get_depth · select_cluster  │
│               [SAGA] · project_mask · validate · report               │
│  Revit 侧 MCP: place_family · execute_revit_code · get_revit_view    │
│               · list_families · list_levels · get_current_view_elements│
│  VLM(Claude/GPT-4o): 看 3DGS 渲染图→判定"这是墙/门洞"→调 Revit MCP  │
│  原则: VLM 只下判定; 几何数值由求解器产出; Revit MCP 执行原生图元创建  │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────── 几何规范化 + 配准 ─────────────────────────────────┐
│  配准: 3DGS(相对尺度) ↔ FloorPlan(米制) → 解 s,θ,t                   │
│  拟合: Open3D RANSAC 平面 + 重力对齐(IMU) + 拉伸 → 墙参数化          │
│  差异: 3DGS 墙 vs 底图墙线 → 仅报告                                   │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────── Revit MCP 服务器 → Revit 原生图元 ─────────────────┐
│  mcp-servers-for-revit (git 子模块, C# + TypeScript MCP + WebSocket)         │
│    → Wall.Create / NewFamilyInstance / execute_revit_code             │
│    → Revit 中原生可编辑图元（墙/板/门/窗/柱）                         │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据流一句话总结
**底图锁水平 XY（绝对尺度）+ 3DGS+VLM 锁三维与语义（高度/窗户/类型）→ 参数化 BIM。**

---

## 4. 核心设计决策（含理由）

### 4.1 去耦：FloorPlan Provider
LiDAR 的本质作用只是"提供房间水平底图（绝对尺寸 + 墙位）"。把它抽象成 `FloorPlan` 接口后，底图可来自：手量、LiDAR 扫描、消防图/CAD 导入。
- **最大收益**：`ManualProvider` 让主链路第一天就脱离任何硬件依赖，LiDAR 降为"可并行、可砍"的支线。项目最大硬件风险被移出关键路径。
- **扩展性**：适配几乎所有既有建筑（无 LiDAR 也能用消防图），大创影响力论点更硬。

### 4.2 VLM 只判定，不算坐标
VLM 的能力边界是语义判断与推理，**不是**生成精确几何数值。让 VLM 输出"墙平面：原点(1.234,2.567,0)、法向(…)"——数值不可靠、单位会乱、多次不一致。
- **分工**：VLM 产出意图/约束（"这片是墙 X""到此角落为止"），确定性求解器产出几何数值。这是系统能稳定的前提。

### 4.3 语义通道：SceneSplat 3DGS-native 语言特征
- **离线编码**：SceneSplat（ICCV 2025 Oral）的 PT-v3 预训练编码器直接在 3DGS 参数上输出 per-Gaussian 768 维语言特征（`feat.pt`），零样本对齐 SigLIP2 文本嵌入——**取代 SAGA/Gaussian Grouping 的逐帧 SAM/CLIP 蒸馏**。
- **文本查询**：VLM 直接用自然语言查询高斯（"哪些是墙？"→ `query_semantics("wall")`），无需 2D mask 反投影。
- **语义渲染**：`render_semantic_overlay` 把匹配高斯染红/其余染青，或全局按 argmax 类别着色，供 VLM 视觉确认。
- **阈值校准**：实测 SceneSplat 余弦相似度聚集在 ~0.1±0.015，sigmoid 后概率全在 ~0.52——**绝对阈值不可靠，argmax(dominant) 是可靠信号**（详见 §12.5）。

### 4.4 几何由确定性求解器产出
3DGS→Mesh 现有方法（Poisson/marching cubes on opacity）只用高斯位置、丢视觉语义，平面坑洼。本项目的对策：
- **一旦 VLM 告诉系统"这块是墙"，就不 mesh 它——直接拟合平面**，平滑问题消失。
- 语义信息的真正杠杆 = "这区域是墙→拟合平面" vs "这是家具→才 mesh"。
- 拟合工具：Open3D RANSAC + 重力对齐 + 拉伸。

### 4.5 差异仅报告
发现"3DGS 有墙但底图没有"时，MVP 只标记、不自动改图。保守、可解释，后续版本再支持自动采纳。

---

## 5. 技术栈与依赖清单

### 5.1 复用 vs 自建
| 流水线环节 | 库（仓库） | 作用 | 复用/自建 |
|---|---|---|---|
| 相机位姿 (SfM) | **COLMAP** `colmap/colmap` | 特征匹配/稀疏点/位姿 | 复用 |
| 度量深度先验 | **Metric3D V2** `YvanYin/Metric3D` | 单目度量深度，喂深度正则损失 | 复用 |
| 3DGS 训练（深度正则+已知位姿） | **nerfstudio / splatfacto** `nerfstudio-project/nerfstudio` | 端到端训练 | 复用 |
| **核心光栅器（MCP 渲染引擎）** | **gsplat** `nerfstudio-project/gsplat` | `rasterization(...,render_mode="RGB+ED")` 从任意位姿出 RGB+深度；**2026-03 新增旋转 LiDAR 相机模型** `pip install "gsplat[lidar]"` | 复用 ⭐ |
| **语义高斯：3DGS-native 语言特征** | **SceneSplat** `Say6n/SceneSplat` (ICCV'25 Oral) | PT-v3 预训练编码器直出 per-Gaussian 768 维语言特征，零样本对齐 SigLIP2；feat.pt 文件交接 | 复用 ⭐（已替代 SAGA） |
| ~~开放词表（逐帧蒸馏）~~ | ~~OpenGaussian / LangSplat V2 / SAGA / Gaussian Grouping~~ | ~~逐帧 SAM/CLIP 蒸馏~~ | **已弃用**（被 SceneSplat 替代） |
| 文本嵌入 | **SigLIP2** `google/siglip2-base-patch16-512` | 生成 BIM 词表文本嵌入，与 feat.pt 对齐 | 复用 |
| 曲面/Mesh（楼板/复杂面备用） | **PGSR** `zju3dv/PGSR`（平面基，适合墙）/ **2DGS** `hbb1/2d-gaussian-splatting`（TSDF）/ **SuGaR** `Anttwo/SuGaR` | 需要网格时 | 选择性 |
| 几何处理 | **Open3D** `isl-org/Open3D` | RANSAC 平面、2D ICP 配准 | 复用 |
| **VLM 工具服务器（3DGS 侧）** | **MCP Python SDK** `modelcontextprotocol/python-sdk` | 暴露 render_from_pose/pick/segment 等 3DGS 工具给 VLM | **自建（薄壳）** |
| VLM | Claude / GPT-4o / Gemini | 推理循环 | API |
| **Revit MCP 服务器** | **mcp-servers-for-revit** `mcp-servers-for-revit/mcp-servers-for-revit`（git 子模块） | C# + TypeScript MCP + WebSocket 桥接 Revit API；已实现 26 个工具（`create_line_based_element`/`create_surface_based_element`/`send_code_to_revit` 等）；可添加自定义 BIM-Recon 扩展 | 复用 |
| **水平底图 Provider** | Manual（自写）+ LiDAR（ROS2 `/scan`→矢量化） | 去耦底图接口 | **自建** |
| 2D LiDAR 取墙线 | 现有 **ROS2 driver** + split-and-merge | 扫描→墙线段 | 复用 driver + 小算法 |

### 5.2 按阶段安装清单（提前准备依赖用）
- **第 1 周必备**：`COLMAP`(二进制)、`nerfstudio`、`gsplat`、`open3d`、`mcp[cli]`、`httpx`、`uvicorn`（mcp-servers-for-revit 依赖）；`pyRevit`（Revit 侧，需安装到 Revit + 激活 Routes Server）
- **第 5 周前**：`SceneSplat`（独立 conda 环境 Python 3.10 / PyTorch 2.5.1）+ `transformers`（bim-recon 环境，SigLIP2 文本嵌入）
- **第 11 周前**：`Metric3D V2` 权重（深度正则）
- **第 16 周前**：ROS2 LiDAR driver + `pip install "gsplat[lidar]"`
- **弹性**：`PGSR` / `2DGS`（墙平面化加成）、`LangSplat`（开放词表）

---

## 6. 关键复用发现（避免造轮子）
1. **SceneSplat（ICCV'25 Oral）直出 per-Gaussian 语言特征**——PT-v3 预训练编码器在 3DGS 参数上推理一次即得 `feat.pt`(N,768)，零样本对齐 SigLIP2 文本嵌入，**完全取代 SAGA/Gaussian Grouping 的逐帧 SAM/CLIP 蒸馏**。feat.pt 文件交接实现两个 conda 环境零耦合（scene_splat env 推理，bim-recon env 加载）。
2. **SceneSplat 余弦相似度聚集紧致（~0.1±0.015）**：sigmoid 后概率全在 ~0.52，绝对阈值不可靠；argmax(dominant label) 是可靠分类信号。`SemanticQuerier` 提供三种查询模式：`dominant`（默认，最可靠）/ `threshold` / `top_percent`。
3. **gsplat 的 `rasterization()` 就是 MCP `render_from_pose` / `get_depth` 工具本体**，几行代码即可。
4. **gsplat 2026-03 旋转 LiDAR 光栅化**可"从 3DGS 仿真一次 LiDAR 扫描"，与真实扫描对齐解配准 (s,θ,t)——比通用 ICP 更原理性。
5. **没有现成的"3DGS 场景探索 MCP server"**——MCP server 自建，但它是薄壳，底下全是成熟库。
6. **PGSR 是平面基 3DGS**，与"墙=平面"天然契合，可作为墙拟合的加成选项。

---

## 7. 24 周里程碑计划

> 2 人配置：A 主攻 P1→P2（VLM/MCP/几何/IFC），B 主攻 P3（LiDAR/ROS2/配准，W4 后启动）。P0 两人同做。

| 周 | 阶段 | 交付 | 闸门 |
|---|---|---|---|
| 1 | P0 | 环境；IfcOpenShell 出墙+门洞+板在 Revit 可编辑；定义 FloorPlan 契约 + ManualProvider | IFC 尾巴通 |
| 2–3 | P0 | 手机视频→COLMAP→splatfacto 训 3DGS；gsplat 渲 RGB+ED；Open3D 拟合墙→IfcWall；Manual 矩形配准 3DGS (s,θ,t) | 已知管线全跑通 |
| 4 | P0 | 鲁棒性首测 + 文档 | P0 收尾 |
| 5–6 | P1 | 接 SceneSplat 推理语义高斯（feat.pt）；验证文本→3D 查询 | 语义高斯可用 |
| 7–8 | P1 | 写 MCP server（薄壳 over gsplat+SAGA）；Claude 循环定位墙 | VLM 能找墙 |
| 9–10 | P1 | 确定性墙拟合器 + 重力对齐 + 拉伸；VLM vs 手工吻合度 | **核心创新验证** |
| 11–12 | P2 | 地板/天花板 IfcSlab；门/窗→IfcOpeningElement 布尔 | A类全要素 |
| 13–14 | P2 | 差异报告模块；2–3 房间复测 | **算法 MVP 达成** |
| 15 | P2 | 编辑性验证（Revit 拖动墙厚/门窗） | — |
| 16–17 | P3 | ROS2 `/scan`→墙线(split-and-merge)；gsplat 旋转LiDAR光栅仿真扫描 | 平面图能出 |
| 18–19 | P3 | LiDARProvider；Open3D 2D 配准到 3DGS；有/无 LiDAR 质量对比 | LiDAR 支线通 |
| 20–21 | P4 | 难例 + 精度报告；多房间占位；B类(PGSR/TRELLIS)写文档 | — |
| 22–23 | P4 | 系统集成 demo；Revit 编辑视频；（加分）DrawingProvider | — |
| 24 | P4 | 论文/报告/Slides | 大创交付 |

---

## 8. P0 已完成（2026-06-27）

**P0 成果**：
- ✅ FloorPlan 契约 + ManualProvider（6/6 测试通过）
- ✅ Revit MCP 服务器集成（C#/TypeScript，26 工具全实现）
- ✅ Revit MCP 验证：say_hello 成功 + 创建 3m×200mm×3m 墙 + 750×2000mm 门（原生可编辑图元）

**P0 证明**：VLM→MCP→Revit 链路已打通，可直接在 Revit 中创建/修改原生 BIM 图元。

---

## 9. P1 实施计划（3DGS 重建 + VLM 集成）

### 目标
构建 3DGS 场景 → VLM 巡视分割 → Revit 原生图元的完整闭环。

### 任务清单
1. **安装依赖**：COLMAP、nerfstudio、gsplat、open3d、mcp[cli]
2. **3DGS 训练流水线**：手机视频 → COLMAP SfM → nerfstudio splatfacto（深度正则）
3. **语义高斯**：SAGA 或 Gaussian Grouping（SAM/CLIP 特征钉到高斯）
4. **MCP 服务器（3DGS 侧）**：
   - 
ender_from_pose：gsplat 渲染 RGB+深度
   - get_depth：深度图查询
   - select_cluster：语义拾取（支持 text_query 文本过滤）
   - query_semantics：文本→高斯查询
   - render_semantic_overlay：语义着色渲染
5. **VLM 集成**：Claude/GPT-4o 通过 MCP 巡视 3DGS → 判定墙/门洞 → 调用 Revit MCP 创建图元
6. **端到端测试**：单房间视频 → 3DGS → VLM 分割 → Revit 墙体

### 技术栈
- COLMAP（SfM 位姿）
- nerfstudio / gsplat（3DGS 训练 + 渲染）
- SceneSplat + SigLIP2（3DGS-native 语义特征）
- MCP Python SDK（3DGS 侧 MCP 服务器）
- Claude / GPT-4o（VLM 推理）

### 验收标准
- 单房间视频 → 3DGS 场景（PSNR > 25dB）
- VLM 通过 MCP 巡视场景，识别墙体/门洞（准确率 > 80%）
- Revit 中生成对应原生图元（墙/门），可编辑

---

## 9.1 P1 已完成模块（2026-06-29）

### 模块清单
| 文件 | 职责 | 验证状态 |
|---|---|---|
| `bim_recon/gs_scene.py` | 加载 3DGS 场景（PLY / SceneSplat .npy），gsplat 渲染（RGB+ED），语义查询 | 5/9 pytest 通过（4 个需 MSVC JIT） |
| `bim_recon/semantics.py` | SemanticQuerier：加载 feat.pt + text_emb.pt，文本→高斯查询（dominant/threshold/top_percent 三模式）| 18/18 pytest 通过 |
| `bim_recon/mcp_gs.py` | 3DGS MCP server（**9 工具**：get_scene_info / list_cameras / render_from_pose / get_depth_grid / select_cluster / query_semantics / render_semantic_overlay / fit_walls / **fit_walls_guided**）| demo 场景全工具 + 语义守卫通过 |
| `bim_recon/colmap_runner.py` | 包装 `ns-process-data images`，输出 transforms.json + images/ | dry-run 命令构造正确 |
| `scripts/train_gs.py` | 包装 `ns-train splatfacto`，含室内深度正则 | dry-run 命令构造正确 |
| `scripts/encode_bim_labels.py` | SigLIP2 文本嵌入生成器（9 类 BIM 词表 → bim_text_emb.pt）| 已生成 (9, 768) 嵌入 |
| `scripts/run_mcp_gs.bat` | MCP server 启动器（vcvars64 + conda + 支持 --feat/--data-dir）| 文档完整 |
| `data/bim_class_names.txt` | BIM 语义词表（wall/floor/ceiling/door/window/column/beam/stairs/furniture）| 9 类 |
| `data/bim_text_emb.pt` | SigLIP2 文本嵌入 (9, 768)，L2 归一化 | 已生成 |
| `data/bim_class_names.json` | 类名→索引映射 | 已生成 |
| `tests/test_gs_scene.py` | GSScene 单元测试（相机工具、合成渲染、PLY 往返、mask 选择）| 9/9 通过（渲染类需 MSVC） |
| `tests/test_semantics.py` | SemanticQuerier 单元测试（init/query/dominant/top_percent/label_at）| 18/18 通过 |
| `scripts/test_mcp_gs.py` | MCP 工具集成测试（8 工具 + 语义守卫）| demo 场景通过 |
| `bim_recon/wall_fitter.py` | WallFit + WallFitter（迭代 RANSAC + 去重合并 + 重力对齐 + 端点精修 + 高度提取）+ **FloorPlanGuidedFitter**（走廊筛选 + 固定法向直方图峰值）+ Revit 转换函数 | 16/16 pytest 通过 + 真实数据 6/8 墙底图引导验证 |
| `bim_recon/floorplan_registration.py` | FloorPlan→3DGS 自动配准（PCA 旋转 + 90° 候选搜索 + 平移网格搜索 + 地板多边形评分）| 3/3 pytest 通过 |
| `tests/test_wall_fitter.py` | WallFitter 单元测试（basic/merge/align/refine/height/revit 转换）| 16/16 通过 |
| `tests/test_floorplan_guided.py` | FloorPlanGuidedFitter + register_floorplan 单元测试（registration/corridor/noise/height）| 8/8 通过 |
| `bim_recon/virtual_scanner.py` | VirtualScanner：从 3DGS 深度渲染模拟 2D 激光扫描（多视角拼接 360° 极坐标扫描 + feat.pt 语义标签）| 真实数据 7820 点 + 7 类语义标签验证 |
| `bim_recon/wall_line_extractor.py` | 多高度扫描 → 栅格化 + 形态学闭运算 + 轮廓提取 + Douglas-Peucker + RANSAC/PCA 精修 → 闭合墙线多边形 | 真实数据 11 墙闭合多边形 + 点云中心拟合 |
| `scripts/virtual_scan_probe.py` | 虚拟扫描探针（加载 feat.pt → 多高度扫描 → 雷达 PNG + JSON）| 已验证 |
| `scripts/wall_line_probe.py` | 墙线提取探针（多高度扫描 → 墙线 JSON + 俯视图 PNG）| 已验证 |
| `bim_recon/candidate_extractor.py` | 元素候选提取：从多高度扫描 + feat.pt 语义标签提取候选构件位置（门/窗/家具），投影到墙线 + 间隙聚类 + 极坐标计算 | 17/17 pytest 通过 |
| `bim_recon/vlm_verifier.py` | VLM 验证模块：极坐标→相机位姿映射（支持 X/Y/Z-up）→ 3DGS 渲染 → Ollama VLM（gemma4:12b）确认/排除 | 25/25 pytest 通过 |
| `tests/test_candidate_extractor.py` | 候选提取单元测试（投影/聚类/提取/DBSCAN自由构件/过滤）| 17/17 通过 |
| `tests/test_vlm_verifier.py` | VLM 验证单元测试（极坐标/视角映射 X-Y-Z-up/响应解析/prompt/Mock端到端）| 25/25 通过 |

### 关键技术决策

1. **nerfstudio 训练 + gsplat 渲染分层**：训练用 nerfstudio splatfacto（工程成熟、自带 COLMAP pipeline），渲染用 gsplat 直调（轻量、MCP server 不依赖 nerfstudio 运行时）。

2. **gsplat 1.4.0 render_mode**：用 `RGB+ED`（expected depth = sum(w_i*z_i) / sum(w_i)）得到真实度量深度；4 通道输出 = RGB(3) + depth(1)，第二返回值是 alpha 而非 depth。

3. **相机约定**：OpenCV / COLMAP（+x 右、+y 下、+z 前），与 nerfstudio / COLMAP 一致。`CameraPose.to_viewmat()` 返回 world-to-camera 4x4 矩阵。

4. **PLY 格式**：nerfstudio 导出的 SH DC 模式。加载时需 sigmoid(opacity_logit)、exp(log_scale)、C0*SH_DC+0.5（C0 = 0.28209479177387814）。

5. **MSVC JIT 编译**：gsplat 1.4.0 在 Windows 上首次使用需 JIT 编译 CUDA 后端，要求 `cl.exe` 在 PATH。通过 vcvars64.bat 解决（已内置到 `scripts/run_mcp_gs.bat`）。

6. **SceneSplat 集成（feat.pt 文件交接）**：scene_splat conda 环境（Python 3.10 / PyTorch 2.5.1）推理产出 `feat.pt`(N,768) + `.npy` 文件；bim-recon 环境用 `GSScene.from_npy()` 加载——两环境零耦合。SceneSplat 输出的 PLY（`data_feat_vis_3dgs.ply`）含 PCA 颜色非真实 RGB，故从 `color.npy`（uint8）加载真实颜色。

7. **SceneSplat 阈值校准**：实测余弦相似度聚集在 ~0.1±0.015（logits），sigmoid 后概率全在 ~0.52——绝对阈值不可靠。`SemanticQuerier` 默认用 `query_dominant`（argmax），MCP `query_semantics` 默认 `mode="dominant"`。真实数据（1.5M Gaussians）上 floor=21.6%、door=24.8%、wall=10.8% 分布合理。

8. **SigLIP2 API 变化**：transformers 5.x 中 `get_text_features()` 返回 `BaseModelOutputWithPooling`，需取 `.pooler_output`。prompt 格式 `"this is a {label}"`，max_length=64（与 SceneSplat 对齐）。

### MCP 工具语义

| 工具 | 入参 | 返回 | VLM 用途 |
|---|---|---|---|
| `get_scene_info` | 无 | JSON: 高斯数、AABB、默认相机 | 了解场景规模 |
| `list_cameras` | 无 | JSON: 训练相机列表 | 选择已有视角 |
| `render_from_pose` | eye, target, up, fov, W, H | PNG (HxWx3) | 看一眼场景 |
| `get_depth_grid` | eye, target, stride, W, H | JSON: 下采样深度网格 + 统计 | 推断墙距、房间尺寸 |
| `select_cluster` | eye, target, bbox_xyxy, [text_query], W, H | JSON: 选中高斯数 + centroid + AABB | 2D box 到 3D 高斯桥接；text_query 可选语义过滤 |
| `query_semantics` | text_query, [mode], [threshold], [percent] | JSON: 类别 + 高斯数 + centroid + AABB + 置信度 | 文本→3D 高斯查询（"哪些是墙？"）|
| `render_semantic_overlay` | eye, target, [text_query], W, H | PNG (语义着色) | VLM 视觉确认语义分类 |
| `fit_walls` | [text_query], [mode], [up_axis] | JSON: 墙线段列表 (p0/p1/height/thickness/length) | 从语义高斯自动提取墙线段 → Revit |

### 数据流

    手机视频
      -> ffmpeg 抽帧
      -> colmap_runner.py (ns-process-data images)
      -> transforms.json
      -> train_gs.py (ns-train splatfacto)
      -> nerfstudio checkpoint
      -> ns-export gaussian-splat
      -> splat.ply + .npy（coord/color/opacity/scale/quat）

    语义通道（SceneSplat，独立 conda env）:
      .npy + splat.ply
      -> SceneSplat preprocess_gs.py + lang_inference.py
      -> feat.pt (N, 768) per-Gaussian 语言特征

    bim-recon env:
      -> encode_bim_labels.py (SigLIP2)
      -> bim_text_emb.pt (C, 768)
      -> mcp_gs.py --data-dir data/ --feat output/data_feat.pt \
           --text-emb data/bim_text_emb.pt --class-names data/bim_class_names.json
      -> VLM 巡视 (render_from_pose + get_depth_grid + render_semantic_overlay)
      -> query_semantics("wall") → 墙高斯子集（argmax dominant）
      -> select_cluster (2D bbox + text_query → 3D 高斯子集)
      -> 墙拟合器 (待实现)
      -> revit MCP create_line_based_element
      -> Revit 原生墙

### 用户下一步（需要真实数据）

**几何训练（bim-recon env）：**
1. 拍摄测试房间视频（手持手机，缓慢绕场一周，约 1-2 分钟）
2. `ffmpeg -i video.mp4 -q:v 2 images/%04d.jpg` 抽帧
3. `python -m bim_recon.colmap_runner --images images/ --output data/room1`
4. `python scripts/train_gs.py --data data/room1 --output output/room1`
5. `ns-export gaussian-splat --load-config output/room1/config.yml --output-dir output/room1/splat`

**语义推理（scene_splat env，用户管理）：**
6. `python preprocess_gs.py --output-dir output/room1/scenesplat data/room1/`（产出 .npy + inverse_map）
7. `python lang_inference.py --output-dir output/room1/scenesplat ...`（产出 `data_feat.pt`）
8. `python scripts/encode_bim_labels.py --class-names data/bim_class_names.txt --output-dir data/`（bim-recon env，生成 `bim_text_emb.pt`）

**启动 MCP server：**
9. 把 opencode.json 里 `bim-recon-gs` 的 `--demo` 换成：
   `--data-dir data/room1 --feat output/room1/data_feat.pt --text-emb data/bim_text_emb.pt --class-names data/bim_class_names.json --cameras data/room1/transforms.json`
10. 重启 opencode，VLM 即可用 `query_semantics("wall")` + `render_semantic_overlay` 巡视真实房间语义

---

## 10. 第 1 周可执行任务清单（P0 启动）
1. 建 conda 环境，装：`gsplat`、`nerfstudio`、`open3d`、`mcp[cli]`、`httpx`、`uvicorn`、COLMAP(二进制)；装 `pyRevit` 到 Revit 并激活 Routes Server。
2. mcp-servers-for-revit 验证：启动 Revit + pyRevit Routes，运行 `main.py --combined`，通过 MCP 工具 `execute_revit_code` 发送 `Wall.Create` 代码生成 5×2.8m 墙 + host 门；或使用 `scripts/revit_wall_door.py` 作为备选（直接在 Revit 内运行）。
3. 定义 `FloorPlan` 契约（见附录）+ `ManualProvider`（JSON 输入长宽 + 门位）。
4. 选一间测试房间，手机录 2 分钟环绕视频。
5. （第 2 周）`ns-process-data images` → COLMAP → `ns-train splatfacto` 训 3DGS；gsplat 渲一张 RGB+ED 验证。

---

## 9. 风险与缓解
| 风险 | 等级 | 缓解 |
|---|---|---|
| COLMAP 在弱纹理墙上崩 | 中 | 单房间+家具纹理通常 OK；失败上 ARCore/VINS-Fusion |
| VLM 找不全/重复墙 | 中 | "剩余未建模场景"渲染 + 已建模状态机 |
| 单目尺度漂移 | 中高 | FloorPlan 锚定绝对尺度（LiDAR/手量都给米制） |
| 门窗洞口布尔失败 | 中 | 用 IfcOpenShell openingRel 严格建模，Revit 逐个验证 |
| LiDAR↔3DGS 配准 | 中 | gsplat 旋转LiDAR光栅仿真 + Open3D 2D 配准 |
| 工期（1–2 人/6 月） | 中 | LiDAR/App/多摄全部可砍，P0→P1→P2 是必保关键路径 |

---

## 10. 未来工作（MVP 之外）

### 10.1 B 类复杂构件 mesh 回灌

> ⚠️ **已弃用 IFC 路线**：下方原 IfcBuildingElementProxy / IfcShellBasedSurfaceModel 描述为历史记录。pyRevit 架构下，B 类 mesh 用 `doc.Create.NewDirectShape(...)`（Revit DirectShape 承载任意 mesh，原生可见但不可参数化编辑）；A 类用 `Wall.Create`/`Floor.Create` 等原生 API。

**流程**：区域人工框选 → 多视角(≥3)渲染 RGB+深度+mask → 单物体 mesh 生成器（TRELLIS / InstantMesh / TripoSR）→ **可微渲染配准**(SE(3)+scale) 回灌 → Revit **DirectShape**（via pyRevit）。

**IFC 版本与 mesh 表达策略（关键）**：项目标准为 **IFC4**（见 §2.3），B 类 mesh 直接用原生 `IfcTriangulatedFaceSet` + `IfcCartesianPointList3D`，代码简洁、文件紧凑。

| 维度 | IFC4（项目采用） | IFC2X3（备选/历史） |
|---|---|---|
| 三角网格实体 | ✅ 原生 IfcTriangulatedFaceSet（顶点索引/法线/颜色） | ❌ 无专用实体 |
| 替代表达 | 一行 `IfcTriangulatedFaceSet`（配合 `IfcCartesianPointList3D` 批量存顶点） | `IfcShellBasedSurfaceModel`+`IfcFaceSurface`+`IfcPolyLoop`+`IfcCartesianPoint`（三层嵌套） |
| 顶点存储 | `IfcCartesianPointList3D` 批量，紧凑 | 每点单独 IfcCartesianPoint，文件臃肿 |
| Revit 行为 | 几何保真度高；导入后改"Phase Created=现有"即可正常显示（见 §12.1） | 直接打开可见、可整体移动；不可参数化编辑（哑代理） |
| 代码复杂度 | 低（一行） | 高（手动三层嵌套） |

**落地建议（分阶段）**：
- **MVP（P0–P2）**：不涉及 mesh；A 类用 SweptSolid（墙/板/柱/门窗可编辑 ✅）。
- **P4 难例探索**：IFC4 + `IfcBuildingElementProxy` + `IfcTriangulatedFaceSet`。
- **若 Revit 直接打开遇阻**：先把导入图元的"Phase Created"改为"现有"；或用"链接→绑定→解组"工作流；必要时 mesh 单独导出做链接补充。

**关键技巧**（无论版本）：
- 回灌前用 `trimesh.simplify_quadric_decimation` 把面数压到 **5000 以下**，避免 IFC 文件膨胀/解析卡死。
- 容器用 `IfcBuildingElementProxy`（Revit 归类"常规模型"），**不要**用 `IfcFurnishingElement`（可能被过滤）。
- 即便是哑代理，仍通过 `IfcRelDefinesByProperties` 挂 Pset（材质/来源/置信度）供查询。
- 导出用 `.ifczip`（mesh 类 IFC 可压缩 60%+）。
- **BlenderBIM/Bonsai 预览验证** mesh 表达是否正确，比反复开 Revit 快。

> 说明：IFC4 选型正是为 B 类 mesh 兼容——原生 `IfcTriangulatedFaceSet` 让回灌代码极简；若将来必须用 IFC2X3 交换，退而用 `IfcShellBasedSurfaceModel`（三层嵌套、哑代理）也完全可行，符合 §2.2"不追求毫米级施工精度"的定位。

### 10.2 多房间 / 整层拼接
通过多点底图 + 3DGS 分区重建，软件层拼接（MVP 仅留接口占位）。

### 10.3 更多 FloorPlan Provider
`DrawingProvider`（DXF/PDF 读比例，老房子消防图场景价值高）、`ImageProvider`（拍消防图 + VLM 读图估比例）。

### 10.4 差异自动采纳
"3DGS 有墙/底图没有"从"仅报告"升级为可选自动补墙。

---

## 11. 叙事与卖点（答辩用）
"**一部手机 + 一个 50 元 2D LiDAR + 一个 3D 打印支架**：底图锁定房间绝对尺寸与墙位，3DGS 提供丰富可探索的中间表示，VLM 借 MCP 在其中自由巡视并分割结构构件，pyRevit 直接在 Revit 中生成可编辑的原生 BIM 图元。"

- **四领域交叉**：机器人/SLAM 思想 + 神经渲染(3DGS) + 多模态大模型(VLM/MCP) + BIM(pyRevit/Revit API) + 几何处理。
- **可复现、低成本**：无专业扫描仪，无施工级设备。
- **架构优雅**：去耦的 FloorPlan Provider 让系统适配几乎所有既有建筑。
- **务实分工**：VLM 做它擅长的（语义判定），几何交给确定性求解器，避免"让 LLM 算坐标"的陷阱。

---

## 12. 失败探索记录（Lessons / Failed Explorations）

### 12.1 [2026-06-27] IFC4→IFC2X3 切换 ≠ 3D 不可见的修复（误诊）

**现象**：Revit 2026 打开 `demo.ifc`，三维视图不可见，平面视图可见可拖动；"链接 IFC"可见但不可编辑。

**当时的（错误）判断**：归因为"Revit 不支持 IFC4 直接打开"，遂把种子脚本切到 IFC2X3（commit `709482e`）。

**排查假设（后证伪）**：3D 不可见初判为 Revit 的两类已知 bug，**与 IFC 版本无关**：
- **Bug ①** 导出端写入远离原点的孤立点（地理参考/IfcSite 全局坐标），如 `IFCCARTESIANPOINT((0.,0.24,1.79769313486232E+305))`，把三维视图包围盒撑爆，几何体看似"消失"；平面视图受视图范围约束仍可见。
- **Bug ②** 新版 IFC 处理器三维渲染缺陷。

**诊断证据**（`scripts/diag_coords.py`，对 IfcOpenShell 生成的 IFC2X3 `demo.ifc`）：
- 最大坐标 = 5000 mm（= 5 m 墙长，房间尺度），**无** `1.79e+305` 孤立点；
- IfcSite `RefLatitude/Longitude/Elevation` 均 None（无地理参考）；
- IFC2X3 schema 不含 `IfcMapConversion` / `IfcProjectedCoordinateSystem`（结构上不可能有）；
- 仅 1 个科学计数法 token `1.0000000000000001E-05`（无害 epsilon）。
→ **文件本身干净，3D 不可见不是本文件造成的**。

**✅ 确认的真正根因**（用户后续定位）：Revit 导入 IFC 后，图元的 **"Phase Created（创建阶段）"默认被设为项目中不存在的阶段（阶段三）**，导致三维视图不显示；**把 Phase Created 改为"现有（Existing）"即恢复正常显示**。前述孤立点/处理器 bug 仅为排查假设（文件已证清白），非本例病因。

**Phase 能否在 IFC 创建时指定？** 不能可靠指定——Revit 的 Phase 是 Revit 侧概念，导入时按视图/默认阶段分配，不读 IFC 内容（Revit 导出会写 `Pset_Revit_Phasing`，但导入不读）。手动在 Revit 改 Phase 是可靠路径。

**结论**：
- 标准**回退 IFC4**（本次 commit）：原生 `IfcTriangulatedFaceSet` 利于未来 B 类 mesh；commit `709482e` 的 IFC4→IFC2X3 误诊切换已撤销。
- IFC 文件本身经多轮验证（schema/header/guid/placement/boolean/round-trip 全绿）是正确的；3D 可见性属 Revit 导入侧 Phase 设置问题，非文件缺陷。
- 本条记录避免重复踩坑。

### 12.2 [2026-06-27] Revit 默认 IFC 映射表缺 IfcOpeningElement → 门洞不显示

**现象**：IFC 文件本身正确（IfcOpeningElement + IfcRelVoidsElement 完整、布尔已验证），但导入 Revit 后门洞不显示。

**根因**：Revit 默认 IFC 类映射表（`File → Import/Export Settings → IFC Options`）缺 `IfcOpeningElement` 条目 → 导入时该实体被丢弃 → `IfcRelVoidsElement` 无法定位 opening → 墙体未被挖空。

**修复**（Revit 侧，一次性配置）：导入自定义映射表 `Docs/importIFCClassMapping.txt`（含 `IfcOpeningElement → 常规模型` 及全部类别）。

**代码侧对应改动**（依据该规则表）：`IfcSlab` 加 `predefined_type="FLOOR"`，使其映射为 Revit "楼板"而非"常规模型"（无 type 的 IfcSlab 默认→常规模型）。`IfcWall→墙`、`IfcDoor→门` 无需 type 即正确映射。

### 12.3 [2026-06-27] mcp-servers-for-revit (C#/TypeScript) —— VLM↔Revit 互操作的基础设施

**发现**：`mcp-servers-for-revit/mcp-servers-for-revit`（已作为 git 子模块引入）是一个 C#/TypeScript 的 MCP 服务器，架构恰好是我们 PLAN.md §1 描述的 VLM↔MCP↔Revit 桥梁：

```
VLM (Claude/GPT-4o/Gemini)
  |  MCP Protocol (stdio)
  v
MCP Server (TypeScript, server/)
  |  WebSocket
  v
Revit Plugin (C#, plugin/)
  |  loads
  v
CommandSet (C#, commandset/)
  |  executes
  v
Revit API
```

**26 个工具全部实现（关键工具）**：
- `create_line_based_element`：创建墙体/梁/管道（**pyRevit 版本中此工具为 pending 状态**）
- `create_surface_based_element`：创建楼板/天花板/屋顶（**pyRevit 版本中此工具为 pending 状态**）
- `create_point_based_element`：创建门/窗/家具
- `send_code_to_revit`：在 Revit 内执行 C# 代码（比 IronPython 更强大）
- `delete_element` / `operate_element`：构件管理（**pyRevit 版本中为 pending**）
- `create_level` / `create_room` / `create_grid`：建筑工具
- `create_structural_framing_system`：结构梁系统
- `get_available_family_types` / `get_current_view_elements`：查询工具
- `get_material_quantities` / `analyze_model_statistics`：分析工具
- `tag_all_walls` / `tag_all_rooms` / `create_dimensions`：标注工具
- `export_room_data` / `store_project_data` / `query_stored_data`：数据管理

**对 BIM-Recon 的具体益处**：
1. **墙体/楼板创建已实现**：无需等待或自行实现，MVP 立即可用。
2. **C# 代码执行**：`send_code_to_revit` 比 IronPython 更强大，可访问完整 Revit API。
3. **预编译 Release**：下载 ZIP 安装到 Revit addins 文件夹，无需编译。
4. **WebSocket 桥接**：比 pyRevit Routes HTTP 更可靠。
5. **Revit 2020-2026 支持**：覆盖所有主流 Revit 版本。
6. **集成测试**：TUnit 框架对 live Revit 实例运行测试，质量有保障。

**结论**：该项目是 BIM-Recon 的核心基础设施——应将其作为 Revit 互操作层（取代 pyRevit 方案），直接使用其 26 个已实现工具。

### 12.4 [2026-06-27] 从 pyRevit MCP 切换到 C# MCP 的决策

**背景**：最初引入 `mcp-server-for-revit-python`（pyRevit/IronPython 版本），发现其 `create_line_based_element`（墙体）和 `create_surface_based_element`（楼板）为 pending 状态，需要自行实现。

**发现更优项目**：`mcp-servers-for-revit`（C#/TypeScript 版本）26 个工具全部实现，包括关键的墙体/楼板创建工具。

**切换理由**：
1. **功能完整性**：26/26 工具实现 vs 19/26（7 个 pending，包括最关键的墙体/楼板）。
2. **技术栈**：C# 原生 Revit 插件 + TypeScript MCP 服务器 + WebSocket，比 pyRevit (IronPython 2.7) + HTTP 更现代、更可靠。
3. **代码执行能力**：`send_code_to_revit` 执行 C# 代码（完整 Revit API 访问），比 `execute_revit_code`（IronPython）更强大。
4. **部署简便**：预编译 Release ZIP，复制到 addins 文件夹即可；pyRevit 版本需要安装 pyRevit + 激活 Routes Server。
5. **维护状态**：C# 项目有 CI/CD、集成测试、npm 发布流程；pyRevit 版本相对简单。

**决策**：移除 `mcp-server-for-revit-python` 子模块及其文档（`Docs/pyrevit/`、`scripts/revit_wall_door.py`），改用 `mcp-servers-for-revit`。

### 12.5 [2026-06-30] SceneSplat 集成：3DGS-native 语义特征取代 SAGA/Gaussian Grouping

**背景**：原 §4.3 计划用 SAGA / Gaussian Grouping 做逐帧 SAM/CLIP 蒸馏得到语义高斯。调研发现 **SceneSplat（ICCV 2025 Oral）**——PT-v3 预训练编码器直接在 3DGS 参数上输出 per-Gaussian 768 维语言特征，零样本对齐 SigLIP2 文本嵌入，无需逐帧处理。

**切换理由**：
1. **更原生**：直接在高斯参数上推理一次，不依赖渲染图 + SAM mask 蒸馏。
2. **开放词表**：768 维语言特征 + SigLIP2 文本嵌入 → 任意文本查询（"墙""门""家具"），无需固定类别训练。
3. **更简单**：feat.pt 文件交接，两个 conda 环境零耦合（scene_splat env 推理，bim-recon env 加载）。
4. **已验证**：真实数据（1.5M Gaussians, ARKitScenes）上 argmax 分布合理（floor 21.6%、door 24.8%、wall 10.8%）。

**关键发现：余弦相似度阈值不可靠**：
- 实测 logits（`feat @ text_emb.T`）聚集在 **~0.1 ± 0.015**，sigmoid 后概率全在 **~0.52**。
- 绝对阈值（如 0.5）会把几乎所有高斯判为所有类——无区分力。
- **argmax（dominant label）是可靠信号**：虽然各类绝对概率接近，但 argmax 跨类有区分力。
- 对策：`SemanticQuerier` 默认 `query_dominant()`（argmax），MCP `query_semantics` 默认 `mode="dominant"`；另提供 `threshold` / `top_percent` 模式备用。

**架构**：
```
scene_splat env (用户管理)          bim-recon env (agent 管理)
  .npy + splat.ply                   feat.pt (torch.load)
  -> preprocess_gs.py                text_emb.pt (SigLIP2)
  -> lang_inference.py               -> SemanticQuerier
  -> feat.pt (N,768) ──────────────> -> GSScene.from_npy()
                                     -> MCP 7 工具
```

**数据格式关键点**：
- SceneSplat `feat.pt` 是 **post-normalization**（L2 归一化）float16；加载时转 float32。
- SceneSplat `.npy` 是 **post-activation**（opacity 已 sigmoid、scale 已 exp、quat 已归一化）——`from_npy()` 不再应用激活函数（与 PLY raw 格式不同）。
- SceneSplat 导出的 PLY（`data_feat_vis_3dgs.ply`）含 **PCA 颜色非真实 RGB**；真实颜色在 `color.npy`（uint8 [0,255]），故用 `from_npy()` 而非 `from_ply()`。
- 高斯顺序保证：`preprocess_gs.py` 按 PLY vertex 顺序读取，`lang_inference.py` 用 inverse_map 恢复原始顺序。

**SigLIP2 API 变化**：transformers 5.x 中 `get_text_features()` 返回 `BaseModelOutputWithPooling`，需取 `.pooler_output` 获取嵌入张量（旧版直接返回张量）。

**新增模块**：`bim_recon/semantics.py`（SemanticQuerier）、`scripts/encode_bim_labels.py`、`data/bim_class_names.txt`（9 类）、GSScene 扩展（`from_npy`/`query_semantics(mode=)`）、MCP 新增 2 工具 + select_cluster 增强。

### 12.6 [2026-06-30] 墙拟合器：RANSAC + 遮挡补全 + 端点精修 + Revit 接入

**背景**：SceneSplat 集成（§12.5）后，query_semantics 能给出"哪些高斯是墙"，但 Revit create_line_based_element 需要的是墙的起止点坐标（几何数值）。墙拟合器补全这一跃。

**管线**：`query_semantics("wall") → WallFitter.fit() → WallFit 列表 → wallfit_to_line_based_element() → Revit`

WallFitter.fit() 内部：
1. **迭代 RANSAC**：Open3D segment_plane，每次提取最大平面，移除 inliers，重复。distance_threshold=0.08m, min_inliers=500, max_thickness=1.0m（过滤非墙散布）。
2. **重力对齐**：法向投影到水平面（去掉 up 分量），确保墙法向水平。
3. **去重/合并（遮挡补全）**：法向夹角 <10° + 共面 <0.15m 的墙段合并。**关键**：3DGS 只重建可见表面，柜子/门洞后的墙无高斯 → 共面碎片合并为一条完整墙线段（端点取投影 min/max 横跨空洞）。合成测试验证：5m 墙中间 1.5m 空洞 → 1 条 ≈5m 完整墙。
4. **端点精修**：相邻墙水平面投影的交点 = 角点。L 形墙角点坐标一致（合成测试验证）。
5. **高度提取**：floor centroid.z → ceiling centroid.z（比 inlier 范围更稳定）。

**真实数据验证**（1.5M Gaussians, ARKitScenes）：
- 161252 wall 高斯 → 9 面墙
- 长度 1.6-8.5m, 高度 2.54m（floor→ceiling）, 厚度 0.17-0.68m
- wallfit_to_line_based_element 输出正确的毫米制 Revit 参数

**关键设计决策**：
- 新建 WallFit(3D) 而非改 WallSegment(2D)：WallSegment 是 FloorPlan 契约，保持稳定。
- max_thickness=1.0m 过滤：厚度 >1m 的"平面"不是墙（是散布或地面/天花板残留）。
- 遮挡补全测试用合成点云验证（mid-gap + door-gap），确保连续性。

### 12.7 [2026-06-30] 底图引导墙拟合器：FloorPlanGuidedFitter + register_floorplan

**背景**：盲拟合（§12.6）把 161252 个 wall 高斯（含大量柜子/家具误分类点）全部丢给迭代 RANSAC → 9 面散落假墙，方向/位置各不同，用户反馈"非常糟糕"。

**根因**：query_semantics("wall") 的 161252 高斯中混入大量非墙垂直表面（柜子背面、门框等），RANSAC 给噪声也拟合平面。

**解决方案**：底图引导。用户传入 2D 底图（JSON 墙线段列表），系统自动配准到 3DGS 坐标系，然后对每条底图墙线段只在 ±0.5m 走廊范围内筛选 wall 高斯，用**固定法向直方图峰值**拟合——彻底消除走廊外噪声。

**管线**：`用户底图 JSON → register_floorplan() → FloorPlanGuidedFitter.fit_guided() → WallFit 列表 → Revit`

**register_floorplan**（自动配准）：
1. **平移**：地板 footprint AABB 中心对齐。
2. **旋转**：地板 footprint PCA 主轴 vs 底图墙线段 PCA 主轴，±90°×4 候选搜索，用地板多边形内点数评分（比 wall-only 走廊评分更鲁棒）。
3. **缩放**：默认 1.0（底图和 3DGS 均为米制），可选覆盖。
4. **平移网格精修**：3m 半径 7×7 网格搜索，最大化地板点在底图多边形内的数量。

**FloorPlanGuidedFitter.fit_guided**（单墙拟合）：
1. **自适应走廊**：0.3m → 0.5m → 0.75m → 1.0m → 1.5m，依次放宽直到找到足够 inliers。
2. **固定法向**：直接使用底图墙线段的法向作为硬约束（1 DOF），避免 RANSAC 在噪声中随机选错平面。
3. **直方图峰值**：走廊内 wall 高斯沿法向投影 → 取最密集 bin 的中位数作为平面偏移。
4. **端点约束**：墙端点 = 底图线段端点投影到拟合平面（保证墙长度/方向与底图一致）。
5. **后处理**：复用 `_gravity_align` + `_refine_endpoints` + `_compute_heights`。

**真实数据验证**（同 §12.6 数据）：
- 用户底图：8 墙段（10m×8m 矩形 + 2.5m×2.5m 凹室），手量、准确度差
- 自动配准后：底图中心/旋转/尺度与 3DGS 房间匹配
- **拟合结果：6/8 墙段成功**（左/右/上 + 凹室三边），全部 axis-aligned，长度与底图一致
- 厚度 0.08m（单面石膏板级），高度 2.54m（floor→ceiling）
- **对比**：盲拟合 9 面散落墙 → 底图引导 6 面整齐墙

**MCP 工具**：`fit_walls_guided(floorplan_json, ...)` 接收 ManualProvider 格式 JSON（矩形或显式墙线段），返回 Revit-ready 墙参数。

### 12.8 [2026-07-01] 虚拟激光扫描 + 栅格化墙线提取

**背景**：底图引导（§12.7）依赖用户提供准确底图。用户希望从 3DGS 场景**自动提取墙线**，无需底图输入。核心洞察：gsplat 深度渲染可模拟激光扫描——从房间中心在特定高度渲染深度图，取水平切片即为 2D 极坐标扫描线，等效于真实 LiDAR。

**虚拟扫描器**（`bim_recon/virtual_scanner.py`）：
1. 在房间中心放置 N 个虚拟相机（8 视角 × 60° FOV = 480° 覆盖），每个在特定高度水平渲染。
2. 提取深度图中间行（水平面切片），反投影到世界 XY 坐标 → 极坐标 (θ, r) 序列。
3. 拼接 N 视角为完整 360° 扫描。
4. **语义标签**：第二渲染通道将 feat.pt dominant class 编码到高斯 R 通道，渲染后解码每个扫描点的语义类别。

**多高度墙线提取**（`bim_recon/wall_line_extractor.py`）：
1. **多高度扫描**：从地板到天花板 8 个高度，每个高度独立扫描 → 融合（高处的扫描能看到矮家具后面的墙面）。
2. **语义过滤**：排除 floor/ceiling/furniture 类别，保留所有结构垂直表面（wall+door+window+column+beam）。
3. **DBSCAN 去噪**：eps=0.15m，min_samples=10，保留大簇。
4. **栅格化 + 形态学闭运算**：0.05m/px 占据栅格 → 7×7 闭运算核桥接 0.35m 间隙（解决遮挡断裂）。
5. **轮廓提取**：`cv2.findContours` 提取最大外轮廓（天然闭合多边形）。
6. **Douglas-Peucker 简化**：`cv2.approxPolyDP`（周长 1.2%）→ 墙角顶点。
7. **RANSAC/PCA 精修**：每面 DP 墙段筛选 ±0.3m 带宽内的原始扫描点，PCA 主成分拟合直线穿过点云几何中心，投影原始端点到拟合线（解决"线在边缘"问题）。

**真实数据验证**：
- 8 高度扫描 × 8 视角 × 2 通道 = 128 次渲染 → 31271 扫描点
- 23608 墙表面点 → 11 面墙形成闭合多边形
- 最长墙 5.49m（左）、5.10m（顶）、4.51m（底）
- 每面墙携带 PCA 拟合点数（3357–3575 pts）

**关键设计决策**：
- 栅格轮廓仅用于**拓扑**（确定"哪段是哪面墙"），不作为最终几何——RANSAC/PCA 拟合才决定墙线位置。
- 形态学闭运算桥接遮挡间隙——这是极坐标 split-and-merge 无法做到的。

### 12.9 [2026-07-01] VLM 验证元素提取（雷达扫描 + Ollama VLM 两阶段检测）

**背景**：§12.8 的栅格化墙线提取能找到墙，但 feat.pt 语义标签对门/窗等构件存在大量误报（room0: 14 个门候选中仅 2 个是真的）。硬编码过滤换房间就失效。

**核心方案**：两阶段检测，每阶段做它最擅长的事：
1. **Stage 1 — feat.pt 候选生成**（高召回）：雷达扫描的语义标签找到所有"像门"的位置
2. **Stage 2 — 3DGS 渲染 + VLM 验证**（高精度）：对每个候选位置渲染一张针对性图像，让 VLM 判定是否真的是门

**极坐标 → 渲染视角映射**（关键数学）：
雷达扫描的每个点本身就是从房间中心渲染出来的——极角 θ = 从中心看向该点的方位角，距离 r = 到表面的物理距离。给定候选的 (θ_c, r_c, h_range)：

```python
eye    = (cx, cy, floor_z + 1.5)    # 房间中心，人眼高度
target = (cx + r_c·cos(θ_c), cy + r_c·sin(θ_c), floor_z + h_mid)  # 构件中心
fov    = 60°
```

**实现模块**：
- `candidate_extractor.py`：从 ScanResult + 墙线提取 Candidate[]（投影到墙 + 间隙聚类 + 极坐标计算）
- `vlm_verifier.py`：Candidate → 相机位姿 → 3DGS 渲染 → Ollama gemma4:12b VLM 确认/排除

**真实数据验证**（room0, 1,373,014 高斯, point_cloud_30000.ply 原始权重）：
- feat.pt 产生 14 个门候选 → 预过滤（width≥0.7m, pts≥100）剩 5 个 → VLM 确认 2 个
- 排除的 3 个误报：2 个实为窗（百叶窗），1 个实为墙面装饰画
- VLM 不仅排除误报，还自动说出实际是什么（可顺带修正分类）

**关键决策**：
- 固定使用 Ollama gemma4:12b 作为 VLM 后端（本地部署，无 API 成本）
- 偏好 `point_cloud_*.ply` 原始权重渲染（颜色准确），而非 `_feat_vis_3dgs.ply`（PCA 染色）
- VLM prompt 要求首行输出 CONFIRMED/REJECTED，便于自动解析
- **up_axis 全局支持**：`candidate_to_viewpoint` 和 `verify_candidates` 接受 `up_axis` 参数，通过轴索引构建 eye/target/up 向量，与 VirtualScanner 一致。支持 X-up/Y-up/Z-up 场景（42/42 测试覆盖含 Y-up 和 X-up 用例）。
- 输出路径：`output/<scene>/verify_<element>/` 目录存放验证图，`output/<scene>/<element>s_verified.json` 存放结果。

### 12.10 [2026-07-01] 统一管线 + 元素类型注册表 + 脚本清理

**背景**：项目积累了 23 个脚本（探针、原型、工具），用户要求最终只保留一个主流程脚本：输入 3DGS 场景，自动输出墙+门+窗。

**元素类型注册表**（`bim_recon/element_config.py`）：
- `ElementConfig` frozen dataclass：name, class_idx, structural, min_width, min_points, typical_height, vlm_hint
- 注册了 4 种构件类型：door（宽≥0.7m，结构构件）、window（宽≥0.5m，结构构件）、column（宽≥0.2m，结构构件）、furniture（宽≥0.3m，自由构件 DBSCAN）
- 添加新构件类型 = 字典加一行
- `get_element_config("window")` 查找配置，`list_element_types()` 列出所有可用类型

**统一管线**（`scripts/run_pipeline.py`）：
- 唯一入口：`python scripts/run_pipeline.py --name room0`
- 流程：加载场景 → 12 高度雷达扫描 → 墙线提取 → 门检测 → 窗检测 → JSON 输出
- 可选参数：`--elements door window column`（指定检测类型）、`--skip-vlm`（跳过 VLM）
- 输出：`pipeline_report.json` + `wall_lines_snapped.json` + `doors_verified.json` + `windows_verified.json`
- （计划中）Revit MCP 推送：待 A 阶段实现，当前管线输出 JSON 供手动导入

**脚本清理**：
- 删除 15 个冗余探针/探索脚本（analyze_*, check_*, probe_*, *_probe.py 等）
- 保留 7 个脚本：run_pipeline（主流程）、generate_walls（单独墙线）、final_radar（可视化）、encode_bim_labels（工具）、manual_to_revit_code（工具）、test_mcp_gs（测试）、train_gs（训练包装）

**项目约定**：
- 永远不要在意字符/字体警告（emoji 缺失、CJK glyph 等），程序能正常运行即可

---

## 附录 A：FloorPlan 契约（草案）

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class WallType(Enum):
    BEARING = "bearing"      # 承重墙（默认厚 240mm）
    PARTITION = "partition"  # 隔墙（默认厚 120mm）
    UNKNOWN = "unknown"

class OpeningKind(Enum):
    DOOR = "door"
    WINDOW = "window"

@dataclass
class WallSegment:
    x1: float; y1: float       # 起点平面坐标（米）
    x2: float; y2: float       # 终点平面坐标（米）
    thickness: Optional[float] = None   # 米；None 则按 type 取默认
    type: WallType = WallType.UNKNOWN

@dataclass
class Opening:
    wall_index: int            # 所在墙段在 wall_segments 的下标
    offset: float              # 沿墙段的距离（米）
    width: float               # 洞口宽（米）
    kind: OpeningKind = OpeningKind.DOOR
    sill_height: Optional[float] = None  # 窗台高（米），门=None

@dataclass
class FrameMeta:
    scale_known: bool = True       # 底图是否已是米制
    orientation_known: bool = False # 是否已知朝向（北）
    gravity_axis: tuple = (0.0, 0.0, 1.0)  # 上方向（来自 IMU）

@dataclass
class FloorPlan:
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    meta: FrameMeta = field(default_factory=FrameMeta)

class FloorPlanProvider:
    """所有底图来源实现此接口。核心流水线对来源无感。"""
    def get_floorplan(self) -> FloorPlan: ...
```

**Provider 列表（MVP）**
- `ManualProvider`：JSON 输入（矩形长宽 + 门位）→ `FloorPlan`。零硬件。
- `LiDARProvider`：ROS2 `/scan` → split-and-merge 提墙线 → `FloorPlan`（门口天然缺口=门位）。

---

## 附录 B：MCP 工具集（已实现）

### 3DGS 侧（`bim-recon-gs`，9 工具）

| 工具 | 底层 | 用途 |
|---|---|---|
| `get_scene_info()` | GSScene.scene_bounds | 高斯数、AABB、默认相机 |
| `list_cameras()` | transforms.json | 训练视角列表 |
| `render_from_pose(eye, target, up, fov, W, H)` | gsplat `rasterization` | VLM 看场景（RGB PNG） |
| `get_depth_grid(eye, target, stride, W, H)` | gsplat `render_mode="RGB+ED"` | 下采样深度网格（JSON） |
| `select_cluster(eye, target, bbox, [text_query], W, H)` | gsplat 投影 + SemanticQuerier | 2D box→3D 高斯；可选语义过滤 |
| `query_semantics(text_query, [mode], [threshold], [percent])` | SemanticQuerier | 文本→高斯查询（dominant/threshold/top_percent） |
| `render_semantic_overlay(eye, target, [text_query], W, H)` | gsplat + 颜色替换 | 语义着色渲染（匹配红/其余青，或全局调色板） |
| `fit_walls([text_query], [mode], [up_axis])` | WallFitter (RANSAC+merge+refine) | 语义高斯→墙线段（p0/p1/height/thickness）→ Revit |
| `fit_walls_guided(floorplan_json, [text_query], [mode], [up_axis], [corridor_width])` | FloorPlanGuidedFitter + register_floorplan | 底图 JSON → 自动配准 → 走廊筛选直方图峰值 → 墙线段（比盲拟合更稳定）→ Revit |

### Revit 侧（`mcp-servers-for-revit`，26 工具，复用）

关键工具：`create_line_based_element`（墙）、`create_surface_based_element`（板）、`create_point_based_element`（门/窗）、`send_code_to_revit`（C# 代码执行）、`operate_element`（高亮/隔离/删除）等。

### VLM 工作流（目标）
1. `get_scene_info` → 了解场景规模
2. `render_from_pose` / `render_semantic_overlay` → 巡视
3. `query_semantics("wall")` → 拿到墙高斯子集 + AABB
4. `get_depth_grid` → 推断墙距/房间尺寸
5. `select_cluster(text_query="wall")` → 精确区域拾取
6. **`fit_walls_guided(floorplan_json)`** → 底图引导墙拟合（自动配准 + 走廊筛选 + 固定法向直方图峰值）→ Revit `create_line_based_element`
7. `render_from_pose` 重渲染叠合 → VLM 回看确认

> **底图引导优先于盲拟合**：`fit_walls_guided` 使用用户手量底图作为空间先验，彻底消除柜子/家具噪声导致的散落假墙；`fit_walls` 保留为无底图时的备用方案。

---

*本计划由需求讨论逐步收敛而成。实施过程中如遇架构变更，请同步更新本文件。*
