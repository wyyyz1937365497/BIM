"""3DGS → BIM Pipeline — Gradio Web UI (modular).

UI layout + event wiring only. All callback logic lives in
``bim_recon.gradio_helpers``.

Launch:
    python scripts/gradio_app.py
The gsplat renderer initializes the Visual Studio compiler environment on demand.
"""
from __future__ import annotations

# --- Localhost proxy bypass (must run before httpx/gradio import) ---
# 系统代理（如 Clash 127.0.0.1:7892）会拦截 gradio 对 localhost 的内部请求，
# 导致回调创建 event 后永不执行、前端按钮卡死。这里强制 localhost 绕过代理。
import os as _os
_LOCALHOST = "127.0.0.1,localhost,::1"
_existing = _os.environ.get("NO_PROXY", "")
_os.environ["NO_PROXY"] = _LOCALHOST if not _existing else f"{_existing},{_LOCALHOST}"

import asyncio
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
from PIL import Image

# Import all helpers (callbacks, business logic, constants)
from bim_recon.gradio_helpers import (
    logger, ROOT, MAX_CONSOLE_LINES,
    SCENESPLAT, _SCENE_CACHE, _FALCON_CACHE,
    validate_ply, check_preprocess_status, list_available_scenes,
    find_latest_output, list_available_results, _result_dir_from_label,
    _prepare_results, run_pipeline_direct, load_results_cb, clean_wall_overlaps_cb,
    _find_element, update_interactive_radar, draw_bbox_on_image,
    on_element_select, on_mask_apply, fetch_camera_state, launch_scene_viewer,
    _get_scene, _get_falcon, _mask_bbox_to_wall_coords,
    resegment_from_viewpoint, apply_vlm_review,
    detect_revit_version, install_revit_mcp,
)
from bim_recon.config import load_config
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest
from bim_recon.pipeline_api import PipelineResults
from bim_recon.mcp_gateway import StdioMCPGateway
from bim_recon.revit_workflow import RevitBuildOptions, RevitBuildWorkflow
from bim_recon.workflow_runtime import stream_workflow_gradio
from bim_recon.pipeline_runner import find_scene_files
from bim_recon.radar_viz import (
    draw_spatial_context as _draw_spatial_context,
    observation_radar as _observation_radar,
    registration_radar as _registration_radar,
)
from bim_recon.trellis_workflow import (
    ApprovedMeshObject,
    TrellisRevitWorkflow,
    TrellisWorkflowConfig,
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="3DGS → BIM 管线") as app:
        gr.Markdown("# 3DGS → BIM 自动重建管线")
        scene_state = gr.State("")
        results_state = gr.State(None)
        viewer_state = gr.State({})

        # ====== ① 场景与数据准备 ======
        gr.Markdown("## ① 场景与数据准备")
        with gr.Row():
            with gr.Column(scale=1):
                scene_dropdown = gr.Dropdown(
                    label="选择已有场景", choices=list_available_scenes(),
                    value="", allow_custom_value=True,
                    info="data/ 文件夹中的场景",
                )
                refresh_scenes_btn = gr.Button("🔄 刷新场景列表")
            with gr.Column(scale=1):
                ply_file = gr.File(label="或上传新 PLY", file_types=[".ply"])
                scene_name = gr.Textbox(
                    label="场景名称（留空自动提取）",
                    placeholder="如 room0",
                )
                with gr.Row():
                    validate_btn = gr.Button("验证 PLY", variant="secondary")
                    preprocess_btn = gr.Button("预处理", variant="primary")
        val_status = gr.JSON(label="验证结果")
        prep_status = gr.Textbox(label="预处理状态", interactive=False)

        # ====== ② 运行管线 ======
        gr.Markdown("---\n## ② 运行管线")
        with gr.Row():
            cb_doors = gr.Checkbox(True, label="门")
            cb_windows = gr.Checkbox(True, label="窗")
            cb_falcon = gr.Checkbox(True, label="Falcon 分割")
            cb_skipvlm = gr.Checkbox(False, label="跳过 VLM")
            run_btn = gr.Button("🚀 运行管线", variant="primary")

        console_out = gr.Textbox(
            label="控制台输出（实时）", lines=12, max_lines=25,
            interactive=False,
            placeholder="点击「运行管线」后，输出将在此实时显示...",
        )
        current_scene_md = gr.Markdown("**当前场景:** (无)")
        out_dir_box = gr.Textbox(visible=False)  # hidden state for pipeline output path
        with gr.Row():
            results_dropdown = gr.Dropdown(
                label="选择已有结果", choices=[], value="",
                allow_custom_value=True, scale=3,
                info="output/ 目录中的管线运行结果",
            )
            refresh_results_btn = gr.Button("🔄", scale=1)
            load_btn = gr.Button("📥 加载结果", variant="secondary", scale=1)
            clean_walls_btn = gr.Button("🧹 清理重叠墙", scale=1)

        # ====== ③ 检测结果（按管线流程顺序展示） ======
        gr.Markdown("---\n## ③ 检测结果")
        summary_md = gr.Markdown("运行管线后，结果将在此显示。")

        gr.Markdown("### 墙线提取 + 构件雷达图")
        radar_gallery = gr.Gallery(
            columns=3, height=None, label="", show_label=False,
            object_fit="contain", preview=True,
        )

        gr.Markdown("### VLM 确认图")
        vlm_gallery = gr.Gallery(
            columns=6, height=None, label="", show_label=False,
            object_fit="contain", preview=True,
        )

        with gr.Row():
            with gr.Column(scale=3):
                vlm_review_cbs = gr.CheckboxGroup(
                    label="已确认构件（取消勾选则拒绝）", choices=[], value=[],
                )
            with gr.Column(scale=1):
                review_btn = gr.Button("应用审核")
                review_status = gr.Markdown("")
        report_json = gr.JSON(label="完整管线报告")

        gr.Markdown("### 交互式雷达图（勾选/取消构件实时更新）")
        interactive_radar = gr.Image(
            label="交互式雷达图", height=600, show_label=False,
        )



        # ====== ④ 微调（Mask 绘制 + 视角重分割） ======
        gr.Markdown("---\n## ④ 微调")
        gr.Markdown(
            "**方式一** — 手动 Mask：选择构件 → 在立面图上用红色画笔涂出门窗区域 → 重算尺寸\n\n"
            "**方式二** — 在独立 3D 查看器中漫游到覆盖构件的视角 → 获取视角 → 以此视角重新分割"
        )
        camera_data = gr.State({})  # 存储捕获的相机参数
        with gr.Row():
            with gr.Column(scale=2):
                elem_sel = gr.Dropdown(
                    label="选择构件", choices=[], allow_custom_value=True,
                )
                mask_editor = gr.ImageMask(
                    label="立面渲染图（用画笔涂出门/窗区域）",
                    type="numpy",
                    height=500,
                    layers=False,
                    brush=gr.Brush(
                        colors=["#FF0000"], default_size=30,
                        color_mode="fixed",
                    ),
                    transforms=[],
                    sources=[],
                )
            with gr.Column(scale=1):
                gr.Markdown("#### 📐 方式一：手动 Mask")
                remap_btn = gr.Button("📐 从 Mask 重算墙坐标", variant="primary")
                remap_out = gr.JSON(label="手动 Mask 结果")
                gr.Markdown("---\n#### 🔲 方式二：视角重分割")
                cam_btn = gr.Button("📸 从独立查看器获取视角", variant="secondary")
                cam_status = gr.Markdown(
                    "运行管线后，Viewer Manager 会异步启动该场景的独立查看器。"
                )
                reseg_btn = gr.Button("🔲 以此视角重新分割", variant="primary")
                reseg_preview = gr.Image(
                    label="新视角渲染 + Falcon 分割", height=300,
                )
                reseg_out = gr.JSON(label="视角重分割结果")

        # ====== ⑤ Revit 确定性工作流 ======
        gr.Markdown("---\n## ⑤ Revit 确定性工作流")
        gr.Markdown(
            "按固定步骤创建标高、楼板、墙、门窗并核验 ElementId。"
            "墙固定使用 Rewrite 项目中的“常规 - 200mm”实心基本墙"
            "（类型 ID 398）。需要 Revit 已运行且 MCP 插件已加载。"
        )
        with gr.Row():
            revit_wall_thickness = gr.Number(
                label="墙厚 (mm，实心墙类型固定)", value=200.0,
                minimum=200.0, maximum=200.0, interactive=False,
            )
            revit_floor_thickness = gr.Number(
                label="楼板厚度 (mm)", value=200.0, minimum=1.0,
            )
            revit_create_floor = gr.Checkbox(label="创建楼板", value=True)
        with gr.Row():
            revit_offset_x = gr.Number(label="X 偏移 (mm)", value=0.0)
            revit_offset_y = gr.Number(label="Y 偏移 (mm)", value=0.0)
            revit_level_name = gr.Textbox(
                label="标高名称", value="BIM-Recon Level 1",
            )
        with gr.Row():
            revit_check_btn = gr.Button("检测 Revit MCP", variant="secondary")
            revit_build_btn = gr.Button("执行 Revit 工作流", variant="primary")
        revit_connection_status = gr.Markdown(
            "尚未检测。连接检测不会调用 Revit API，也不会打开 Revit 弹窗。"
        )
        revit_build_status = gr.Markdown("等待执行")
        revit_build_result = gr.JSON(label="创建与核验结果")

        gr.Markdown(
            "### ⑤b B 类物体确定性导入（Falcon 分割 → TRELLIS Mesh → DirectShape）\n"
            "对管线已确认的家具等 B 类检测结果，自动渲染视角、Falcon 分割抠图、"
            "TRELLIS 生成 mesh、并通过编译版 `create_directshape_from_mesh` 注册到 Revit。"
            "需要 Falcon 服务（端口 18390）、TRELLIS 服务（端口 18391）和 Revit MCP 同时在线。"
        )
        bclass_run_btn = gr.Button(
            "执行 B 类物体确定性导入", variant="primary",
        )
        bclass_status = gr.Markdown("等待执行")
        bclass_result = gr.JSON(label="B 类物体导入结果")

        # ====== ⑤b B类构件手动提取 ======
        # (自动化扫描与审批已暂时移除，仅保留手动选取流程)
        with gr.Tab("手动选取（视角 + 框选 + 识别）"):
            gr.Markdown(
                "**步骤：**\n"
                "1. 在独立查看器中漫游到目标物体\n"
                "2. 点击「捕获视角并渲染」\n"
                "3. 用画笔在渲染图上**粗略框选**目标物体（不必精确）\n"
                "4. 点击「🔍 VLM识别 + Falcon分割」— VLM 自动识别物体，Falcon 生成精确遮罩\n"
                "5. 确认抠图后点击「生成 Mesh + 注册 Revit」"
            )
            bmesh_cam_btn = gr.Button("捕获视角并渲染", variant="secondary")
            bmesh_cam_status = gr.Markdown("需先从独立查看器捕获视角")
            bmesh_mask_editor = gr.ImageMask(
                label="渲染图（用画笔粗略框选目标物体）",
                type="numpy", height=400, layers=False,
                brush=gr.Brush(
                    colors=["#FF0000"], default_size=15, color_mode="fixed",
                ),
                transforms=[], sources=[],
            )
            with gr.Row():
                bmesh_seed_manual = gr.Number(
                    label="种子", value=1, precision=0,
                )
                bmesh_identify_btn = gr.Button(
                    "🔍 VLM识别 + Falcon分割", variant="secondary",
                )
                bmesh_gen_btn = gr.Button(
                    "① 生成 Mesh + 配准", variant="primary",
                )
                bmesh_import_btn = gr.Button(
                    "② 导入 Revit", variant="secondary",
                )
            bmesh_identified_label = gr.Textbox(
                label="VLM 识别结果", interactive=False,
            )
            bmesh_segmentation_preview = gr.Image(
                label="Falcon 分割结果", height=300,
            )
            bmesh_manual_preview = gr.Image(label="抠图预览", height=300)
            bmesh_manual_output = gr.JSON(label="生成结果")
            bmesh_import_status = gr.Markdown("")
            gr.Markdown("### B 类空间雷达（分割反投影 / GLB 配准）")
            with gr.Row():
                bmesh_observation_radar = gr.Image(
                    label="分割照片反投影雷达图", height=520, interactive=False,
                )
                bmesh_glb_radar = gr.Image(
                    label="GLB 配准后场景雷达图", height=520, interactive=False,
                )
            bmesh_cutout_state = gr.State(None)
            bmesh_render_state = gr.State(None)
            bmesh_cam_state = gr.State(None)
            bmesh_detection_state = gr.State(None)
            bmesh_scene_state = gr.State("")
            bmesh_ready_state = gr.State(None)
        # ====== ⑥ 独立 3D 查看器 ======
        gr.Markdown(
            "---\n## ⑥ 独立 3D 查看器\n\n"
            "完成第 ① 步 SceneSplat 预处理后，可为当前场景异步启动独立查看器。"
        )
        viewer_start_btn = gr.Button(
            "启动当前场景的 3D 查看器",
            variant="primary",
            interactive=False,
        )
        viewer_panel_html = gr.HTML(
            "<div>请选择已完成第 ① 步处理的场景。</div>",
        )

        # ====== 事件绑定 ======

        # --- 数据准备 ---
        def on_validate(ply_file_path) -> dict:
            if ply_file_path is None:
                return {"错误": "未选择文件"}
            return validate_ply(ply_file_path)

        def on_preprocess(ply_file_path, name: str) -> tuple[str, str]:
            if ply_file_path is None:
                return "错误：请先选择 PLY 文件", ""
            src = Path(ply_file_path)
            if not src.exists():
                return f"错误：文件不存在: {ply_file_path}", ""
            if not name:
                name = src.stem
            name = name.strip()
            if not name:
                return "错误：无法从文件名提取场景名", ""
            scene_dir = ROOT / "data" / name
            scene_dir.mkdir(parents=True, exist_ok=True)
            dst_ply = scene_dir / src.name
            shutil.copy2(str(src), str(dst_ply))
            preprocessed_dir = scene_dir / "preprocessed"
            output_dir = ROOT / "output" / name
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            if check_preprocess_status(name)["feat_pt_exists"]:
                return f"'{name}' 已预处理，跳过", name
            try:
                cmd1 = [sys.executable, "-m", "scripts.preprocess_gs",
                        "--input", str(dst_ply), "--output", str(preprocessed_dir)]
                subprocess.run(cmd1, cwd=str(SCENESPLAT), check=True)
                ckpt = SCENESPLAT / "ckpt" / "lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"
                if not ckpt.exists():
                    raise FileNotFoundError(f"SceneSplat 权重不存在: {ckpt}")
                cmd2 = [sys.executable, "-m", "tools.lang_inference",
                        "--config", "configs/inference/lang-pretrain-pt-v3m1-3dgs.py",
                        "--checkpoint", str(ckpt),
                        "--input-root", str(preprocessed_dir),
                        "--output-dir", str(output_dir),
                        "--scene-name", name]
                subprocess.run(cmd2, cwd=str(SCENESPLAT), check=True)
                return f"✅ 预处理完成，场景: {name}", name
            except FileNotFoundError as e:
                return f"错误: {e}", ""
            except subprocess.CalledProcessError as e:
                return f"SceneSplat 失败 (code {e.returncode}): {e}", ""
            except Exception as e:
                return f"未知错误: {e}", ""

        validate_btn.click(fn=on_validate, inputs=ply_file, outputs=val_status)
        preprocess_btn.click(fn=on_preprocess, inputs=[ply_file, scene_name],
                             outputs=[prep_status, scene_state])

        # --- 管线与结果 ---
        scene_state.change(
            fn=lambda s: f"**当前场景:** `{s or '(无)'}`",
            inputs=scene_state, outputs=current_scene_md,
        )
        # 场景变化时刷新结果下拉列表
        scene_state.change(
            fn=lambda s: gr.update(choices=list_available_results(s), value=""),
            inputs=scene_state, outputs=results_dropdown,
        )

        def _viewer_launch_availability(scene_name: str):
            ready = bool(scene_name) and all(
                check_preprocess_status(scene_name).values()
            )
            if ready:
                return (
                    gr.update(interactive=True),
                    {},
                    "<div>场景已完成第 ① 步处理，可启动独立查看器。</div>",
                )
            return (
                gr.update(interactive=False),
                {},
                "<div>请先选择并完成第 ① 步 SceneSplat 预处理。</div>",
            )

        scene_state.change(
            fn=_viewer_launch_availability,
            inputs=scene_state,
            outputs=[viewer_start_btn, viewer_state, viewer_panel_html],
        )
        viewer_start_btn.click(
            fn=launch_scene_viewer,
            inputs=scene_state,
            outputs=[viewer_state, viewer_panel_html],
        )
        scene_dropdown.change(fn=lambda s: s, inputs=scene_dropdown, outputs=scene_state)
        refresh_scenes_btn.click(
            fn=lambda: gr.update(choices=list_available_scenes()),
            outputs=scene_dropdown,
        )
        refresh_results_btn.click(
            fn=lambda s: gr.update(choices=list_available_results(s)),
            inputs=scene_state, outputs=results_dropdown,
        )
        run_btn.click(
            fn=run_pipeline_direct,
            inputs=[
                scene_state, cb_doors, cb_windows, cb_falcon, cb_skipvlm,
                viewer_state,
            ],
            outputs=[console_out, out_dir_box, results_state, summary_md,
                     radar_gallery, vlm_gallery, report_json,
                     vlm_review_cbs, elem_sel, viewer_state, viewer_panel_html],
        )
        # 管线运行完成后，刷新结果下拉列表并选中最新结果
        out_dir_box.change(
            fn=lambda s: gr.update(
                choices=list_available_results(s),
                value=list_available_results(s)[0] if list_available_results(s) else "",
            ) if s else gr.update(),
            inputs=scene_state, outputs=results_dropdown,
        )

        def _on_load(scene_name: str, result_label: str):
            out_dir = _result_dir_from_label(scene_name, result_label)
            return load_results_cb(out_dir)

        load_btn.click(
            fn=_on_load,
            inputs=[scene_state, results_dropdown],
            outputs=[results_state, summary_md, vlm_gallery,
                      radar_gallery, report_json, vlm_review_cbs, elem_sel],
        )
        clean_walls_btn.click(
            fn=lambda scene_name, result_label: clean_wall_overlaps_cb(
                _result_dir_from_label(scene_name, result_label)
            ),
            inputs=[scene_state, results_dropdown],
            outputs=[results_state, summary_md, vlm_gallery,
                      radar_gallery, report_json, vlm_review_cbs, elem_sel],
        )
        review_btn.click(fn=apply_vlm_review,
                         inputs=[results_state, vlm_review_cbs],
                         outputs=[results_state, review_status])
        vlm_review_cbs.change(
            fn=update_interactive_radar,
            inputs=[vlm_review_cbs, results_state],
            outputs=interactive_radar,
        )

        # --- 微调 ---
        elem_sel.change(
            fn=on_element_select,
            inputs=[elem_sel, results_state],
            outputs=mask_editor,
        )
        cam_btn.click(
            fn=fetch_camera_state,
            inputs=[viewer_state],
            outputs=[cam_status, camera_data],
        )
        remap_btn.click(
            fn=on_mask_apply,
            inputs=[mask_editor, elem_sel, results_state],
            outputs=remap_out,
        )
        reseg_btn.click(
            fn=resegment_from_viewpoint,
            inputs=[camera_data, elem_sel, results_state, scene_state],
            outputs=[reseg_preview, reseg_out, cam_status,
                     results_state, mask_editor],
        )

        # --- Revit deterministic workflow ---
        async def _check_revit_connection():
            config = load_config().revit_mcp
            gateway = StdioMCPGateway(
                command=config.command,
                args=tuple(config.args),
                cwd=str(ROOT),
                timeout_seconds=5.0,
            )
            try:
                result = await gateway.call_tool("check_revit_connection", {})
            except Exception as exc:
                return f"❌ Revit MCP 检测失败：{exc}"
            if isinstance(result, dict) and result.get("connected"):
                return (
                    "✅ Revit MCP 插件端口可达 "
                    f"({result.get('host')}:{result.get('port')}, "
                    f"{result.get('latencyMs')} ms)。未调用 Revit API。"
                )
            detail = result.get("error", "连接不可达") if isinstance(result, dict) else result
            return f"❌ Revit MCP 插件不可达：{detail}"

        async def _run_revit_workflow(
            results: PipelineResults | None,
            confirmed_labels: list[str],
            wall_thickness: float,
            floor_thickness: float,
            create_floor: bool,
            offset_x: float,
            offset_y: float,
            level_name: str,
        ):
            if results is None:
                yield "请先加载一次管线结果。", {"error": "missing_pipeline_results"}
                return
            results, _review_status = apply_vlm_review(
                results, confirmed_labels,
            )
            assert results is not None
            config = load_config().revit_mcp
            gateway = StdioMCPGateway(
                command=config.command,
                args=tuple(config.args),
                cwd=str(ROOT),
                timeout_seconds=float(config.timeout),
            )
            workflow = RevitBuildWorkflow(
                results,
                gateway,
                RevitBuildOptions(
                    level_name=level_name.strip() or "BIM-Recon Level 1",
                    wall_thickness=float(wall_thickness),
                    floor_thickness=float(floor_thickness),
                    create_floor=bool(create_floor),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                ),
            )
            lines: list[str] = []
            latest: dict = {}
            async for update in stream_workflow_gradio(workflow):
                lines.append(update.message)
                latest = update.payload or latest
                yield "\n\n".join(lines[-30:]), latest

        revit_check_btn.click(
            fn=_check_revit_connection,
            outputs=revit_connection_status,
        )

        revit_build_btn.click(
            fn=_run_revit_workflow,
            inputs=[
                results_state,
                vlm_review_cbs,
                revit_wall_thickness,
                revit_floor_thickness,
                revit_create_floor,
                revit_offset_x,
                revit_offset_y,
                revit_level_name,
            ],
            outputs=[revit_build_status, revit_build_result],
        )
        # --- B-class deterministic extraction → TRELLIS → DirectShape ---
        async def _run_bclass_workflow(
            results: PipelineResults | None,
            scene_name: str,
        ):
            if results is None:
                yield "请先加载一次管线结果。", {"error": "missing_pipeline_results"}
                return
            if not scene_name:
                yield "请先选择场景。", {"error": "missing_scene"}
                return
            b_elements = [
                e for e in results.elements
                if e.confirmed and e.element_class not in {"door", "window"}
            ]
            if not b_elements:
                yield "未发现已确认的 B 类物体（运行管线时勾选 furniture）。", {
                    "error": "no_b_class_elements",
                }
                return

            from bim_recon.bmesh_pipeline import extract_bclass_element
            from bim_recon.trellis_workflow import (
                TrellisRevitWorkflow,
                TrellisWorkflowConfig,
                approved_object_from_extraction,
            )

            scene = _get_scene(scene_name)
            if scene is None:
                yield f"❌ 无法加载场景 {scene_name}", {"error": "scene_load_failed"}
                return
            falcon = _get_falcon()
            if falcon is None:
                yield "❌ Falcon 服务不可达，请先启动 Falcon-Perception。", {
                    "error": "falcon_unreachable",
                }
                return

            coords = results.coords or {}
            up_axis = int(coords.get("up_axis", 2))
            floor_z = float(coords.get("floor_z", 0.0))
            ceiling_z = float(coords.get("ceiling_z", 3.0))
            center = coords.get("center", [0.0, 0.0])
            scan_center = (float(center[0]), float(center[1]))

            out_dir = ROOT / "output" / scene_name / "_bclass_deterministic"
            out_dir.mkdir(parents=True, exist_ok=True)

            extractions = []
            lines: list[str] = []
            for idx, element in enumerate(b_elements, start=1):
                lines.append(
                    f"[{idx}/{len(b_elements)}] 提取 {element.element_class} "
                    f"#{element.result_index}..."
                )
                yield "\n\n".join(lines[-30:]), {"phase": "extract", "current": idx}
                try:
                    extraction = await asyncio.to_thread(
                        extract_bclass_element,
                        element, scene, falcon,
                        output_dir=out_dir,
                        scan_center=scan_center,
                        floor_z=floor_z, ceiling_z=ceiling_z,
                        up_axis=up_axis,
                        debug=False,
                    )
                except Exception as exc:
                    lines.append(f"  ⚠️ 提取失败: {exc}")
                    yield "\n\n".join(lines[-30:]), {"phase": "extract", "current": idx}
                    continue
                if extraction is None:
                    lines.append("  ⚠️ Falcon 未分割到目标物体，跳过")
                    yield "\n\n".join(lines[-30:]), {"phase": "extract", "current": idx}
                    continue
                extractions.append(extraction)
                lines.append(
                    f"  ✅ 抠图就绪: {extraction.cutout_path.name} "
                    f"@ ({extraction.position_3d[0]:.2f},"
                    f"{extraction.position_3d[1]:.2f},"
                    f"{extraction.position_3d[2]:.2f})"
                )
                yield "\n\n".join(lines[-30:]), {"phase": "extract", "current": idx}

            if not extractions:
                lines.append("❌ 没有可用的抠图，终止")
                yield "\n\n".join(lines[-30:]), {"error": "no_cutouts"}
                return

            cfg = load_config()
            trellis_cfg = cfg.trellis
            client_factory = lambda: TrellisClient(
                host=trellis_cfg.host, port=trellis_cfg.port,
                timeout=trellis_cfg.timeout,
            )
            revit_cfg = cfg.revit_mcp
            gateway = StdioMCPGateway(
                command=revit_cfg.command,
                args=tuple(revit_cfg.args),
                cwd=str(ROOT),
                timeout_seconds=float(revit_cfg.timeout),
            )
            objects = tuple(
                approved_object_from_extraction(ex)
                for ex in extractions
            )
            workflow = TrellisRevitWorkflow(
                TrellisWorkflowConfig(
                    objects=objects,
                    output_dir=out_dir,
                    register_in_revit=True,
                ),
                client_factory=client_factory,
                gateway=gateway,
            )
            latest: dict = {}
            async for update in stream_workflow_gradio(workflow):
                lines.append(update.message)
                latest = update.payload or latest
                yield "\n\n".join(lines[-30:]), latest
            summary = {
                "objects_total": len(objects),
                "completed": latest.get("completed", 0) if isinstance(latest, dict) else 0,
                "failed": latest.get("failed", 0) if isinstance(latest, dict) else 0,
                "output_dir": str(out_dir),
                "manifest": latest,
            }
            yield "✅ B 类物体确定性导入完成\n\n" + "\n".join(lines[-30:]), summary

        bclass_run_btn.click(
            fn=_run_bclass_workflow,
            inputs=[results_state, scene_state],
            outputs=[bclass_status, bclass_result],
        )
        # --- TRELLIS mesh generation ---
        def _on_trellis_generate(image_path: str, name: str, seed: int) -> dict:
            if not image_path:
                return {"错误": "请先上传或选择输入图像"}
            cfg = load_config().trellis
            client = TrellisClient(host=cfg.host, port=cfg.port, timeout=cfg.timeout)
            if not client.health():
                return {"错误": f"TRELLIS 服务不可达 ({cfg.host}:{cfg.port})，请先启动服务"}
            out_dir = ROOT / "output" / "_trellis_meshes"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = client.generate_mesh(TrellisMeshRequest(
                    image_path=Path(image_path),
                    output_dir=out_dir,
                    name=name or "b_class_mesh",
                    seed=int(seed),
                ))
                return {
                    "状态": "✅ 生成成功",
                    "GLB": str(result.glb_path),
                    "PLY": str(result.gaussian_path) if result.gaussian_path else None,
                    "种子": result.seed,
                }
            except Exception as e:
                return {"错误": f"生成失败: {e}"}

        # --- B类 Mesh: 手动选取 Tab ---

        def _on_bmesh_capture(scene_name: str, viewer_session: dict):
            """Capture viewpoint, render at 2048p, and store depth + camera for placement."""
            if not scene_name:
                return None, None, "❌ 请先选择场景"
            status, cam_data = fetch_camera_state(viewer_session)
            if not cam_data:
                return None, None, status

            scene = _get_scene(scene_name)
            if scene is None:
                return None, None, f"❌ 无法加载场景 {scene_name}"

            eye = cam_data.get("position", [0, 0, 0])
            target = cam_data.get("look_at", [0, 0, 1])
            fov = cam_data.get("fov_degrees", 60)
            up = cam_data.get("up", [0, 0, 1])
            up_axis = cam_data.get("up_axis", 2)

            from bim_recon.gs_scene import look_at_pose
            pose = look_at_pose(
                (eye[0], eye[1], eye[2]),
                (target[0], target[1], target[2]),
                up=(up[0], up[1], up[2]),
            )
            render_result = scene.render(pose, width=2048, height=1536, fov_degrees=fov)
            render_arr = (render_result.colors * 255).clip(0, 255).astype(np.uint8)

            render_state = {
                "rgb": render_arr,
                "depth": render_result.depth,
                "cam": {"eye": eye, "target": target, "fov": fov, "up": up, "up_axis": up_axis},
                "scene": scene_name,
                "width": 2048,
                "height": 1536,
            }
            return render_arr, render_state, f"✅ 渲染完成 ({render_arr.shape[1]}×{render_arr.shape[0]})"

        bmesh_cam_btn.click(
            fn=_on_bmesh_capture,
            inputs=[scene_state, viewer_state],
            outputs=[bmesh_mask_editor, bmesh_render_state, bmesh_cam_status],
        )

        def _on_bmesh_identify(mask_editor_val, render_state):
            """Brush-bbox → VLM classify → Falcon segment → clean RGBA cutout."""
            from bim_recon.bmesh_extractor import classify_and_segment_from_mask_editor
            from bim_recon.vlm_verifier import query_vlm

            scene_name = (render_state or {}).get("scene", "default")
            cfg = load_config()
            falcon = _get_falcon()

            def vlm_caller(image_path: str, prompt: str) -> str:
                return query_vlm(
                    image_path, prompt,
                    cfg.vlm.api_base, cfg.vlm.model, cfg.vlm.api_key,
                    timeout=30,
                )

            out_dir = ROOT / "output" / scene_name / "_trellis_meshes"
            out_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = out_dir / f"debug_{int(time.time())}"
            result = classify_and_segment_from_mask_editor(
                mask_editor_val, vlm_caller, falcon, debug_dir=debug_dir,
            )

            cutout_path = None
            cutout_preview = None
            detection_info = None
            if result.cutout is not None:
                cutout_path = str(debug_dir / "04_cutout.png")
                cutout_preview = np.array(result.cutout)
                # Extract Falcon detection metadata for placement
                from bim_recon.bmesh_extractor import _extract_user_bbox, _select_detection
                extracted = _extract_user_bbox(mask_editor_val)
                if extracted:
                    base_rgb, user_bbox = extracted
                    h_img, w_img = base_rgb.shape[:2]
                    if falcon is not None:
                        from PIL import Image as PILImg
                        try:
                            detections = falcon.segment(
                                PILImg.fromarray(base_rgb), result.label, task="segmentation",
                            )
                            selected = _select_detection(detections, user_bbox, w_img, h_img)
                            if selected:
                                detection_info = {
                                    "norm_bbox": selected.mask_bbox or selected.bbox,
                                    "mask_area_ratio": selected.mask_area_ratio,
                                }
                                # Persist the full-frame Falcon mask for the
                                # render-compare optimizer (M_obs).
                                try:
                                    from bim_recon.bmesh_pipeline import _full_frame_mask
                                    full_mask = _full_frame_mask(selected, h_img, w_img)
                                    if full_mask is not None:
                                        mask_path = debug_dir / "03_falcon_mask.png"
                                        PILImg.fromarray(full_mask, mode="L").save(mask_path)
                                        detection_info["mask_path"] = str(mask_path)
                                except Exception:
                                    pass
                        except Exception:
                            pass

            return (
                result.label,
                result.overlay,
                cutout_preview,
                cutout_path,
                detection_info,
                result.detail,
            )

        def _bmesh_context_from_results(results):
            """Build the A-class room context (walls + scan center) for the radars.

            A-class walls are saved center-relative, so the scan center (read
            from wall_lines.json scan_info) must be subtracted from the B-class
            absolute world positions to align them in the same radar frame."""
            if not results or not getattr(results, "walls", None):
                return None
            center = [0.0, 0.0]
            try:
                raw = json.loads((results.out_dir / "wall_lines.json").read_text("utf-8"))
                c = raw.get("scan_info", {}).get("center")
                if c and len(c) >= 2:
                    center = [float(c[0]), float(c[1])]
            except Exception:
                pass
            elements = [
                {"element_class": e.element_class,
                 "world_x": e.world_x, "world_y": e.world_y}
                for e in getattr(results, "elements", [])
                if getattr(e, "confirmed", False) and e.element_class in ("door", "window")
            ]
            return {"walls": list(results.walls), "elements": elements, "center_offset": center}

        def _bmesh_observation_radar(render_state, detection_info, results):
            return _observation_radar(
                render_state, detection_info, _bmesh_context_from_results(results),
            )

        def _bmesh_registration_radar(output, results):
            context = _bmesh_context_from_results(results)
            if not context:
                return None
            output = output or {}
            placement = output.get("placement") or {}
            diagnostics = placement.get("diagnostics")
            if not diagnostics:
                return _draw_spatial_context(context)
            manifest = {
                "assets": {"glb": output.get("GLB")},
                "registration": {"placement": diagnostics},
            }
            return _registration_radar(manifest, context)

        identify_event = bmesh_identify_btn.click(
            fn=_on_bmesh_identify,
            inputs=[bmesh_mask_editor, bmesh_render_state],
            outputs=[
                bmesh_identified_label, bmesh_segmentation_preview,
                bmesh_manual_preview, bmesh_cutout_state,
                bmesh_detection_state, bmesh_cam_status,
            ],
        )
        identify_event.then(
            _bmesh_observation_radar,
            inputs=[bmesh_render_state, bmesh_detection_state, results_state],
            outputs=bmesh_observation_radar,
        )


        def _on_bmesh_generate(cutout_path, label, render_state, detection_info,
                               results, seed):
            """TRELLIS mesh → render-compare alignment → formatted Revit payload (no import)."""
            if not cutout_path:
                return None, {"错误": "请先点击「🔍 VLM识别 + Falcon分割」生成抠图"}, None
            from PIL import Image as PILImage

            preview = np.array(PILImage.open(cutout_path).convert("RGBA"))
            cfg = load_config()
            trellis = TrellisClient(host=cfg.trellis.host, port=cfg.trellis.port, timeout=cfg.trellis.timeout)
            if not trellis.health():
                return preview, {"错误": "TRELLIS 服务不可达"}, None

            scene_name = (render_state or {}).get("scene", "default")
            out_dir = ROOT / "output" / scene_name / "_trellis_meshes"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{label or 'object'}_{int(time.time())}"
            clean_path = out_dir / f"{name}_clean.png"
            PILImage.open(cutout_path).save(str(clean_path))

            try:
                mesh_result = trellis.generate_mesh(TrellisMeshRequest(
                    image_path=clean_path,
                    output_dir=out_dir,
                    name=name,
                    seed=int(seed),
                ))
            except Exception as e:
                return preview, {"错误": f"TRELLIS 生成失败: {e}"}, None

            output = {
                "状态": "✅ Mesh 生成成功",
                "GLB": str(mesh_result.glb_path),
                "PLY": str(mesh_result.gaussian_path) if mesh_result.gaussian_path else None,
            }

            # --- Compute world placement from depth + Falcon bbox ---
            cam = (render_state or {}).get("cam")
            depth_map = (render_state or {}).get("depth")
            img_w = (render_state or {}).get("width", 2048)
            img_h = (render_state or {}).get("height", 1536)

            placement_info = {}
            if cam and depth_map is not None and detection_info:
                norm_bbox = detection_info.get("norm_bbox") or {}
                mask_path = detection_info.get("mask_path")
                up_axis = int(cam.get("up_axis", 2))
                coords = (results.coords if results else {}) or {}
                floor_z = float(coords.get("floor_z", 0.0))
                ceiling_z = float(coords.get("ceiling_z", 3.0))
                depth_arr = np.asarray(depth_map, dtype=np.float32)

                # Rough physical-size init from bbox + depth + FOV; the optimizer
                # refines scale around it.
                cx_norm = float(norm_bbox.get("x", 0.5))
                cy_norm = float(norm_bbox.get("y", 0.5))
                w_norm = float(norm_bbox.get("w", 0.2))
                h_norm = float(norm_bbox.get("h", 0.2))
                px_c = int(min(max(cx_norm * img_w, 0), img_w - 1))
                py_c = int(min(max(cy_norm * img_h, 0), img_h - 1))
                d_init = float(depth_arr[py_c, px_c]) if np.isfinite(depth_arr[py_c, px_c]) else 1.5
                vfov_rad = math.radians(cam.get("fov", 60.0))
                aspect = img_w / img_h
                hfov_rad = 2.0 * math.atan(math.tan(vfov_rad / 2.0) * aspect)
                element_width_m = max(float(w_norm * 2.0 * d_init * math.tan(hfov_rad / 2.0)), 0.1)
                element_height_m = max(float(h_norm * 2.0 * d_init * math.tan(vfov_rad / 2.0)), 0.1)

                # --- Constrained render-compare alignment (attachment-1 §四) ---
                # Matches the backprojected observation (Falcon mask + 3DGS depth)
                # against the TRELLIS mesh via triangle rasterization, optimizing
                # yaw + scale + planar translation with floor contact. Replaces the
                # old bbox-center anchor + point-projection yaw search.
                rc_result = None
                obs_mask = None
                if mask_path and Path(mask_path).is_file():
                    try:
                        from PIL import Image as _MaskImg
                        obs_mask = np.asarray(_MaskImg.open(mask_path).convert("L")) > 0
                    except Exception:
                        obs_mask = None
                if obs_mask is not None and obs_mask.shape == depth_arr.shape:
                    try:
                        from bim_recon.render_compare import optimize_placement
                        rc_result = optimize_placement(
                            mesh_result.glb_path,
                            {"camera": cam, "depth": depth_arr,
                             "mask": obs_mask, "norm_bbox": norm_bbox},
                            floor_z=floor_z, ceiling_z=ceiling_z, up_axis=up_axis,
                            element_width_m=element_width_m,
                            element_height_m=element_height_m,
                            debug_dir=out_dir / f"{name}_render_compare",
                        )
                    except Exception as exc:
                        placement_info["render_compare_error"] = str(exc)

                ready = None
                if rc_result is not None:
                    from bim_recon.mesh_registrar import (
                        MeshPlacement, compute_placement_transform,
                        register_mesh_in_revit, serialize_placement_diagnostics,
                    )
                    wx, wy = rc_result.world_xy
                    placement_kwargs = dict(
                        glb_path=Path(mesh_result.glb_path),
                        world_x=float(wx), world_y=float(wy),
                        floor_z=floor_z, ceiling_z=ceiling_z,
                        element_width_m=element_width_m,
                        element_height_m=element_height_m,
                        up_axis=up_axis,
                        scale_multiplier=rc_result.scale_multiplier,
                        preserve_floor_contact=True,
                        category="OST_GenericModel", name=label or name,
                    )
                    if rc_result.rotation_override is not None:
                        placement_kwargs["rotation_override"] = rc_result.rotation_override
                    else:
                        placement_kwargs["yaw_degrees"] = rc_result.yaw_degrees
                    placement = MeshPlacement(**placement_kwargs)
                    placement_info.update({
                        "world_x": round(float(wx), 3),
                        "world_y": round(float(wy), 3),
                        "yaw_degrees": round(rc_result.yaw_degrees, 2),
                        "scale_multiplier": round(rc_result.scale_multiplier, 4),
                        "render_compare": rc_result.diagnostics(),
                    })
                    try:
                        transform = compute_placement_transform(placement)
                        revit_result = register_mesh_in_revit(placement, transform)
                        placement_info["diagnostics"] = serialize_placement_diagnostics(placement, transform)
                        placement_info["revit_payload"] = revit_result
                        if revit_result.get("status") == "formatted":
                            output["状态"] = "✅ 生成 + 配准完成（待导入 Revit）"
                            ready = {
                                "payload_path": revit_result.get("payload_path"),
                                "name": label or name,
                                "iou": rc_result.iou,
                                "depth_mae_m": rc_result.depth_mae_m,
                            }
                        else:
                            output["状态"] = "⚠️ Revit 载荷格式化失败，请检查 mesh"
                    except Exception as exc:
                        output["配准错误"] = str(exc)

            output["placement"] = placement_info
            return preview, output, ready

        gen_event = bmesh_gen_btn.click(
            fn=_on_bmesh_generate,
            inputs=[
                bmesh_cutout_state, bmesh_identified_label,
                bmesh_render_state, bmesh_detection_state,
                results_state, bmesh_seed_manual,
            ],
            outputs=[bmesh_manual_preview, bmesh_manual_output, bmesh_ready_state],
        )
        gen_event.then(
            _bmesh_registration_radar,
            inputs=[bmesh_manual_output, results_state],
            outputs=bmesh_glb_radar,
        )

        def _on_bmesh_import(ready):
            """Import the generated + aligned mesh into Revit as a DirectShape."""
            if not ready or not ready.get("payload_path"):
                return "⚠️ 请先点击「① 生成 Mesh + 配准」生成可用网格"
            try:
                from bim_recon.mcp_gateway import StdioMCPGateway
                cfg = load_config()
                revit_cfg = cfg.revit_mcp
                gateway = StdioMCPGateway(
                    command=revit_cfg.command,
                    args=tuple(revit_cfg.args),
                    cwd=str(ROOT),
                    timeout_seconds=float(revit_cfg.timeout),
                )
                asyncio.run(gateway.call_tool(
                    "create_directshape_from_mesh",
                    {"meshFile": ready["payload_path"]},
                ))
                iou = ready.get("iou", 0.0)
                dmae = ready.get("depth_mae_m", 0.0)
                return (f"✅ 已导入 Revit：{ready.get('name', '')} "
                        f"(IoU={iou:.2f}, depth_mae={dmae*100:.1f}cm)")
            except Exception as exc:
                return f"❌ 导入失败：{exc}"

        bmesh_import_btn.click(
            fn=_on_bmesh_import,
            inputs=[bmesh_ready_state],
            outputs=[bmesh_import_status],
        )

        # ====== ⑧ Revit MCP 安装 ======
        gr.Markdown("---\n## ⑧ Revit MCP 安装")
        gr.Markdown(
            "将 `mcp-servers-for-revit` 插件构建并部署到本地 Revit Addins 目录。"
            "从 Revit.exe 版本号自动检测年份（支持 2020-2026）。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                revit_exe_box = gr.Textbox(
                    label="Revit.exe 路径",
                    value=r"F:\Software\AutoDesk\Revit 2026\Revit.exe",
                    placeholder="粘贴或输入 Revit.exe 的完整路径",
                    info="程序将从 exe 版本号自动检测 Revit 年份",
                )
            with gr.Column(scale=1):
                revit_ver_btn = gr.Button("🔍 检测版本", variant="secondary")
        revit_ver_info = gr.Textbox(label="版本检测结果", interactive=False)

        with gr.Row():
            revit_config = gr.Radio(
                ["Debug", "Release"], value="Debug", label="构建配置",
                info="Debug 含调试符号；Release 为优化版本",
            )
            revit_clean = gr.Checkbox(False, label="全量重新编译")
            revit_skip_kill = gr.Checkbox(False, label="跳过关闭 Revit")
            revit_skip_launch = gr.Checkbox(False, label="安装后不重启 Revit")

        revit_install_btn = gr.Button("🔧 开始安装", variant="primary")
        revit_install_status = gr.Textbox(label="安装状态", interactive=False)
        revit_install_log = gr.Textbox(
            label="安装日志（实时）", lines=15, max_lines=30,
            interactive=False,
            placeholder="点击「开始安装」后，构建日志将在此实时显示...",
        )

        async def _detect_revit_for_ui(revit_exe: str):
            import time as _t
            t0 = _t.time()
            try:
                res = detect_revit_version(revit_exe)
                logger.info("[REVIT-DETECT] %.0fms: %s", (_t.time()-t0)*1000, res.get("message"))
            except Exception as exc:
                logger.exception("[REVIT-DETECT] failed")
                res = {"status": "error", "message": f"内部错误: {exc}"}
            prefix = "✅" if res.get("status") == "ok" else "❌"
            return f"{prefix} {res.get('message', '未知错误')}"

        revit_ver_btn.click(
            fn=_detect_revit_for_ui,
            inputs=[revit_exe_box],
            outputs=[revit_ver_info],
        )
        revit_install_btn.click(
            fn=install_revit_mcp,
            inputs=[revit_exe_box, revit_config, revit_skip_kill,
                    revit_skip_launch, revit_clean],
            outputs=[revit_install_log, revit_install_status],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=19255, server_name="127.0.0.1", max_threads=8)
