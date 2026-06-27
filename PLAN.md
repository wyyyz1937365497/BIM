# 3DGS → BIM 自动重建系统 · 项目计划

> **状态**：计划已定稿，待启动
> **日期**：2026-06-26
> **类型**：大学生创新训练计划（SITP）/ 大创
> **MVP 周期**：约 24 周（1–2 人）
> **本文件用途**：团队同步用的唯一事实来源（single source of truth）。实施时以本文件为准。

---

## 1. 项目概述

用**消费级设备**（手机 + 一个 50 元 2D 旋转 LiDAR + 3D 打印支架）采集房间，以 **3D Gaussian Splatting (3DGS)** 作为"信息丰富的中间表示"，让**多模态视觉大模型 (VLM) 借助 MCP** 在 3DGS 场景中自由巡视、分割出符合建筑语义的结构构件，最终由 **IfcOpenShell** 产出可在 **Revit 等 BIM 软件中打开并编辑**的 IFC 实体。

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
| **IFC 版本标准** | **IFC2X3**。Revit 2026 直接打开（可编辑）只认 IFC2X3；IFC4 只能"链接"且不可编辑（P0 Revit QA 已验证）。A 类用 SweptSolid；B 类 mesh 用 `IfcBuildingElementProxy` + `IfcShellBasedSurfaceModel`（详见 §10.1） |

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
┌────────────────── VLM 决策循环 (via MCP) ────────────────────────────┐
│  工具: render_from_pose[gsplat] · get_depth · select_cluster[SAGA]   │
│        project_mask · add_wall/slab/door/window · validate · report  │
│  VLM(Claude/GPT-4o): 看渲染图 → 判定"这是墙/门洞/到此为止" → 选区    │
│  原则: VLM 只下判定, 不算坐标; 几何数值一律由求解器产出               │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────── 几何规范化 + 配准 ─────────────────────────────────┐
│  配准: 3DGS(相对尺度) ↔ FloorPlan(米制) → 解 s,θ,t                   │
│        (LiDAR 情形用 gsplat 旋转LiDAR光栅仿真扫描做原理性配准)        │
│  拟合: Open3D RANSAC 平面 + 重力对齐(IMU) + 拉伸 → SweptSolid         │
│        (PGSR 平面化可作为墙拟合加成)                                  │
│  差异: 3DGS 墙 vs 底图墙线 → 仅报告                                   │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────── IfcOpenShell → Revit 可编辑 IFC ───────────────────┐
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

### 4.3 语义双通道
- **离线蒸馏**：SAGA / Gaussian Grouping 把 SAM/CLIP 特征钉到每个高斯 → 系统自动可聚类，VLM 不必从零看整场。
- **在线反投影**：VLM 看渲染图圈选 mask → 系统选中背后高斯（SAGA 已实现 2D-mask→3D-高斯关联，**无需自写**）。

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
| **语义高斯：2D mask→3D 拾取** | **SAGA** `Jumpat/SegAnyGAussians` (AAAI'25) 或 **Gaussian Grouping** `lkeab/gaussian-grouping` (ECCV'24) | VLM 圈选→选中背后高斯；实例分组 | 复用 ⭐ |
| （加分）开放词表查询 | **OpenGaussian** `yanmin-wu/OpenGaussian` (NeurIPS'24) / **LangSplat V2** `minghanqin/LangSplat` | 文本→高斯 | 可选 |
| 曲面/Mesh（楼板/复杂面备用） | **PGSR** `zju3dv/PGSR`（平面基，适合墙）/ **2DGS** `hbb1/2d-gaussian-splatting`（TSDF）/ **SuGaR** `Anttwo/SuGaR` | 需要网格时 | 选择性 |
| 几何处理 | **Open3D** `isl-org/Open3D` | RANSAC 平面、2D ICP 配准 | 复用 |
| **VLM 工具服务器** | **MCP Python SDK** `modelcontextprotocol/python-sdk` | 暴露 render/pick/segment/add_wall 给 VLM | **自建（薄壳）** |
| VLM | Claude / GPT-4o / Gemini | 推理循环 | API |
| **水平底图 Provider** | Manual（自写）+ LiDAR（ROS2 `/scan`→矢量化） | 去耦底图接口 | **自建** |
| 2D LiDAR 取墙线 | 现有 **ROS2 driver** + split-and-merge | 扫描→墙线段 | 复用 driver + 小算法 |
| IFC 生成 | **IfcOpenShell** `IfcOpenShell/IfcOpenShell` | 写 IfcWall/Slab/Door/Window | 复用 |

### 5.2 按阶段安装清单（提前准备依赖用）
- **第 1 周必备**：`COLMAP`(二进制)、`nerfstudio`、`gsplat`、`open3d`、`ifcopenshell`、`mcp`(Python SDK)
- **第 5 周前**：`SAGA` 或 `Gaussian Grouping`（建议先 SAGA，交互调试方便）+ segment-anything 权重
- **第 11 周前**：`Metric3D V2` 权重（深度正则）
- **第 16 周前**：ROS2 LiDAR driver + `pip install "gsplat[lidar]"`
- **弹性**：`PGSR` / `2DGS`（墙平面化加成）、`LangSplat`（开放词表）

---

## 6. 关键复用发现（避免造轮子）
1. **"VLM 在渲染图圈选→选中 3D 高斯"已由 SAGA / Gaussian Grouping 实现**——不必自写 2D-mask→3D 投影关联。
2. **gsplat 的 `rasterization()` 就是 MCP `render_from_pose` / `get_depth` 工具本体**，几行代码即可。
3. **gsplat 2026-03 旋转 LiDAR 光栅化**可"从 3DGS 仿真一次 LiDAR 扫描"，与真实扫描对齐解配准 (s,θ,t)——比通用 ICP 更原理性。
4. **没有现成的"3DGS 场景探索 MCP server"**——MCP server 自建，但它是薄壳，底下全是成熟库。
5. **PGSR 是平面基 3DGS**，与"墙=平面"天然契合，可作为墙拟合的加成选项。

---

## 7. 24 周里程碑计划

> 2 人配置：A 主攻 P1→P2（VLM/MCP/几何/IFC），B 主攻 P3（LiDAR/ROS2/配准，W4 后启动）。P0 两人同做。

| 周 | 阶段 | 交付 | 闸门 |
|---|---|---|---|
| 1 | P0 | 环境；IfcOpenShell 出墙+门洞+板在 Revit 可编辑；定义 FloorPlan 契约 + ManualProvider | IFC 尾巴通 |
| 2–3 | P0 | 手机视频→COLMAP→splatfacto 训 3DGS；gsplat 渲 RGB+ED；Open3D 拟合墙→IfcWall；Manual 矩形配准 3DGS (s,θ,t) | 已知管线全跑通 |
| 4 | P0 | 鲁棒性首测 + 文档 | P0 收尾 |
| 5–6 | P1 | 接 SAGA/Gaussian Grouping 训语义高斯；验证 2D-prompt→3D 拾取 | 语义高斯可用 |
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

## 8. 第 1 周可执行任务清单（P0 启动）
1. 建 conda 环境，装：`gsplat`、`nerfstudio`、`open3d`、`ifcopenshell`、`mcp`(Python SDK)、COLMAP(二进制)。
2. IfcOpenShell 最小脚本：生成带 `IfcOpeningElement` + `IfcDoor` 的 `IfcWall` 和一块 `IfcSlab`，存 `.ifc`，在 Revit 打开并拖动编辑验证。
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

**流程**：区域人工框选 → 多视角(≥3)渲染 RGB+深度+mask → 单物体 mesh 生成器（TRELLIS / InstantMesh / TripoSR）→ **可微渲染配准**(SE(3)+scale) 回灌 → `IfcBuildingElementProxy`。

**IFC 版本与 mesh 表达策略（关键）**：项目标准为 IFC2X3（见 §2.3），B 类 mesh 在 IFC2X3 下用"哑代理"路线，符合 MVP 定位。

| 维度 | IFC2X3（MVP 采用） | IFC4（未来扩展） |
|---|---|---|
| 三角网格实体 | ❌ 无 IfcTriangulatedFaceSet | ✅ 原生（含顶点索引/法线/颜色） |
| 替代表达 | `IfcShellBasedSurfaceModel`+`IfcFaceSurface`+`IfcPolyLoop`+`IfcCartesianPoint`（三层嵌套） | 一行 `IfcTriangulatedFaceSet`（配合 `IfcCartesianPointList3D` 批量存顶点） |
| 顶点存储 | 每点单独 IfcCartesianPoint，文件臃肿 | `IfcCartesianPointList3D` 批量，紧凑 |
| Revit 行为 | 直接打开可见、可整体移动；**不可参数化编辑**（哑代理） | 只能链接，不能打开转换；几何保真度更高但失去 A 类可编辑性 |
| 代码复杂度 | 高（手动三层嵌套） | 低（一行） |

**落地建议（分阶段）**：
- **MVP（P0–P2）**：不涉及 mesh；A 类用 SweptSolid（墙/板/柱/门窗可编辑 ✅）。
- **P4 难例探索**：IFC2X3 + `IfcBuildingElementProxy` + `IfcShellBasedSurfaceModel`，Revit 中为哑代理（可见不可编辑）。
- **未来扩展（mesh 密度爆炸时）**：混合导出 —— A 类留 IFC2X3 主文件（可编辑），B 类 mesh **单独导 IFC4 文件做链接补充**（鱼与熊掌兼得）。

**关键技巧**（无论版本）：
- 回灌前用 `trimesh.simplify_quadric_decimation` 把面数压到 **5000 以下**，否则 IFC2X3 文件膨胀/解析卡死。
- 容器用 `IfcBuildingElementProxy`（Revit 归类"常规模型"），**不要**用 `IfcFurnishingElement`（可能被过滤）。
- 即便是哑代理，仍通过 `IfcRelDefinesByProperties` 挂 Pset（材质/来源/置信度）供查询。
- 导出用 `.ifczip`（mesh 类 IFC 可压缩 60%+）。
- **BlenderBIM/Bonsai 预览验证** IFC2X3 mesh 表达是否正确，比反复开 Revit 快。

> 澄清：IFC2X3 并非"不支持 mesh"，而是"支持得不好看"——几何体系（CSG/SweptSolid/Brep/ShellBasedSurfaceModel）能承载任意三角网格，只是无 IFC4 的专用 `IfcTriangulatedFaceSet`，导致代码冗长、文件臃肿。对大创级应用（B 类本就是加分项/未来工作），哑代理路线完全符合 §2.2"不追求毫米级施工精度"的定位。

### 10.2 多房间 / 整层拼接
通过多点底图 + 3DGS 分区重建，软件层拼接（MVP 仅留接口占位）。

### 10.3 更多 FloorPlan Provider
`DrawingProvider`（DXF/PDF 读比例，老房子消防图场景价值高）、`ImageProvider`（拍消防图 + VLM 读图估比例）。

### 10.4 差异自动采纳
"3DGS 有墙/底图没有"从"仅报告"升级为可选自动补墙。

---

## 11. 叙事与卖点（答辩用）
"**一部手机 + 一个 50 元 2D LiDAR + 一个 3D 打印支架**：底图锁定房间绝对尺寸与墙位，3DGS 提供丰富可探索的中间表示，VLM 借 MCP 在其中自由巡视并分割结构构件，IfcOpenShell 产出可在 Revit 编辑的 BIM。"

- **四领域交叉**：机器人/SLAM 思想 + 神经渲染(3DGS) + 多模态大模型(VLM/MCP) + BIM(IfcOpenShell) + 几何处理。
- **可复现、低成本**：无专业扫描仪，无施工级设备。
- **架构优雅**：去耦的 FloorPlan Provider 让系统适配几乎所有既有建筑。
- **务实分工**：VLM 做它擅长的（语义判定），几何交给确定性求解器，避免"让 LLM 算坐标"的陷阱。

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

## 附录 B：MCP 工具集（草案）
| 工具 | 底层 | 用途 |
|---|---|---|
| `render_from_pose(pose)` | gsplat `rasterization` | VLM 看场景 |
| `get_depth(pose)` | gsplat `render_mode="ED"` | 几何查询 |
| `select_cluster(mask_2d)` | SAGA/Gaussian Grouping | 2D 圈选→3D 高斯 |
| `list_elements()` | 内部状态机 | 已建模元素 |
| `add_wall / add_slab / add_door / add_window` | Open3D 拟合 + IfcOpenShell | 写入 BIM |
| `validate(element_id)` | gsplat 重渲染叠合 | VLM 回看确认 |
| `report_diff()` | 3DGS 墙 vs FloorPlan | 差异报告 |

---

*本计划由需求讨论逐步收敛而成。实施过程中如遇架构变更，请同步更新本文件。*
