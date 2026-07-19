"""3DGS → BIM Pipeline — Gradio Web UI (modular).

UI layout + event wiring only. All callback logic lives in
``bim_recon.gradio_helpers``.

Launch:
    python scripts/gradio_app.py
The gsplat renderer initializes the Visual Studio compiler environment on demand.
"""
from __future__ import annotations

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
)
from bim_recon.config import load_config
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest
from bim_recon.pipeline_api import PipelineResults
from bim_recon.mcp_gateway import StdioMCPGateway
from bim_recon.revit_workflow import RevitBuildOptions, RevitBuildWorkflow
from bim_recon.workflow_runtime import stream_workflow_gradio
from bim_recon.explorer_controller import ExplorerCamera
from bim_recon.explorer_workflow import ExplorerScanConfig, ExplorerScanWorkflow
from bim_recon.pipeline_runner import find_scene_files
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

        # ====== ⑤b B类构件受控工作流 ======
        gr.Markdown("---\n## ⑤b B类构件受控工作流")
        gr.Markdown(
            "自动扫描采用固定视角序列：每个视角只渲染一次，"
            "按指定标签调用 Falcon，完成三维定位和去重。"
            "TRELLIS 与 Revit 创建只处理人工勾选的物体。"
        )
        bmesh_scene_state = gr.State("")
        bmesh_cam_data = gr.State({})
        explorer_results_state = gr.State({})

        with gr.Tab("自动扫描与审批"):
            with gr.Accordion("扫描参数", open=True):
                with gr.Row():
                    explore_cam_btn = gr.Button("从独立查看器获取初始视角")
                    explore_render_btn = gr.Button(
                        "📸 渲染初始视角", variant="secondary",
                    )
                explore_cam_status = gr.Markdown(
                    "在独立查看器中漫游到扫描起点后捕获相机。"
                )
                explore_initial_view = gr.Image(
                    label="初始视角渲染图", height=360,
                )
                explore_cam_data = gr.State({})
                explore_labels = gr.Textbox(
                    label="开放词汇标签（英文逗号分隔）",
                    value="chair, table, sofa, cabinet, bed, lamp, vase, plant, shelf, desk",
                )
                with gr.Row():
                    explore_num_views = gr.Slider(
                        label="扫描视角数", minimum=1, maximum=12,
                        value=8, step=1,
                    )
                    explore_turn_degrees = gr.Number(
                        label="每步旋转角度", value=45.0,
                    )
                explore_run_btn = gr.Button("执行自动扫描", variant="primary")
            explore_progress = gr.Markdown("等待扫描")
            explore_current_view = gr.Image(label="当前扫描视图", height=360)
            explore_gallery = gr.Gallery(
                label="已发现物体", columns=4, height=250,
                show_label=True, object_fit="contain", preview=True,
            )
            explore_select = gr.CheckboxGroup(
                label="人工确认：选择需要生成 Mesh 的物体",
                choices=[],
            )
            explore_output = gr.JSON(label="扫描结果")
            with gr.Accordion("TRELLIS 与 Revit 参数", open=True):
                with gr.Row():
                    bmesh_width = gr.Number(
                        label="物体宽度 (m)", value=0.8, minimum=0.05,
                    )
                    bmesh_height = gr.Number(
                        label="物体高度 (m)", value=1.0, minimum=0.05,
                    )
                    bmesh_seed = gr.Number(label="种子", value=1, precision=0)
                bmesh_register_revit = gr.Checkbox(
                    label="生成后注册到 Revit", value=True,
                )
                bmesh_generate_selected_btn = gr.Button(
                    "生成已确认物体", variant="primary",
                )
            bmesh_workflow_status = gr.Markdown("等待人工确认")
            bmesh_workflow_output = gr.JSON(label="Mesh 与 Revit 结果")

        with gr.Tab("手动选取（视角 + 框选 + 识别）"):
            gr.Markdown(
                "**步骤：**\n"
                "1. 在独立查看器中漫游到目标物体\n"
                "2. 点击「捕获视角并渲染」\n"
                "3. 用画笔在渲染图上**粗略框选**目标物体（不必精确）\n"
                "4. 点击「🔍 VLM识别 + Falcon分割」— VLM 自动识别物体，Falcon 生成精确遮罩\n"
                "5. 确认抠图后点击「生成 Mesh」"
            )
            with gr.Row():
                bmesh_cam_btn = gr.Button("捕获视角并渲染", variant="secondary")
            bmesh_cam_status = gr.Markdown("需先从独立查看器捕获视角")
            bmesh_mask_editor = gr.ImageMask(
                label="渲染图（用画笔粗略框选目标物体）",
                type="numpy", height=400, layers=False,
                brush=gr.Brush(
                    colors=["#FF0000"], default_size=50, color_mode="fixed",
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
                bmesh_gen_manual_btn = gr.Button(
                    "生成 Mesh", variant="primary",
                )
            bmesh_identified_label = gr.Textbox(
                label="VLM 识别结果", interactive=False,
            )
            bmesh_segmentation_preview = gr.Image(
                label="Falcon 分割结果", height=300,
            )
            bmesh_manual_preview = gr.Image(label="抠图预览", height=300)
            bmesh_manual_output = gr.JSON(label="生成结果")
            bmesh_cutout_state = gr.State(None)  # stores the RGBA cutout path

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

        # --- B类 Mesh: deterministic scan and approval ---
        explore_cam_btn.click(
            fn=fetch_camera_state,
            inputs=[viewer_state],
            outputs=[explore_cam_status, explore_cam_data],
        )

        def _on_explore_render(scene_name: str, cam_data: dict):
            """Render the captured initial viewpoint via scene.render (MCP render_from_pose equivalent)."""
            if not scene_name:
                return None, "❌ 请先选择场景"
            if not cam_data or "position" not in cam_data:
                return None, "⚠️ 请先点击「从独立查看器获取初始视角」捕获相机"
            scene = _get_scene(scene_name)
            if scene is None:
                return None, f"❌ 无法加载场景 {scene_name}"

            eye = cam_data.get("position", [0, 0, 0])
            target = cam_data.get("look_at", [0, 0, 1])
            fov = cam_data.get("fov_degrees", 60)
            up = cam_data.get("up", [0, 0, 1])

            from bim_recon.gs_scene import look_at_pose
            pose = look_at_pose(
                (eye[0], eye[1], eye[2]),
                (target[0], target[1], target[2]),
                up=(up[0], up[1], up[2]),
            )
            render_result = scene.render(pose, width=800, height=600, fov_degrees=fov)
            render_arr = (render_result.colors * 255).clip(0, 255).astype(np.uint8)
            return render_arr, f"✅ 初始视角渲染完成 ({render_arr.shape[1]}×{render_arr.shape[0]})"

        explore_render_btn.click(
            fn=_on_explore_render,
            inputs=[scene_state, explore_cam_data],
            outputs=[explore_initial_view, explore_cam_status],
        )

        async def _run_explorer_scan(
            scene_name: str,
            camera_data: dict,
            labels_text: str,
            num_views: int,
            turn_degrees: float,
        ):
            if not scene_name:
                yield (
                    "请先选择场景", None, [], gr.update(choices=[]),
                    {"error": "missing_scene"}, {},
                )
                return
            if not camera_data or "position" not in camera_data:
                yield (
                    "请先从查看器获取初始视角", None, [],
                    gr.update(choices=[]),
                    {"error": "missing_camera"}, {},
                )
                return
            labels = tuple(
                label.strip()
                for label in labels_text.split(",")
                if label.strip()
            )
            try:
                ply_path, feat_path = find_scene_files(ROOT / "data" / scene_name)
                falcon = load_config().falcon
                position = tuple(float(value) for value in camera_data["position"])
                look_at = tuple(float(value) for value in camera_data["look_at"])
                camera = ExplorerCamera(
                    eye=position,
                    look_at=look_at,
                    fov=float(camera_data.get("fov_degrees", 60.0)),
                )
                workflow = ExplorerScanWorkflow(ExplorerScanConfig(
                    ply_path=ply_path,
                    feat_path=feat_path,
                    output_root=ROOT / "output" / scene_name / "explore",
                    camera=camera,
                    labels=labels,
                    falcon_host=falcon.host,
                    falcon_port=falcon.port,
                    num_views=int(num_views),
                    turn_degrees=float(turn_degrees),
                ))
                lines: list[str] = []
                current_image = None
                latest: dict = {}
                found: list[dict] = []
                async for update in stream_workflow_gradio(workflow):
                    lines.append(update.message)
                    current_image = update.image_path or current_image
                    latest = update.payload or latest
                    found = latest.get("found_objects", found)
                    gallery = [
                        (obj["best_view"], f'{obj["label"]} ({obj["id"]})')
                        for obj in found
                        if obj.get("best_view") and Path(obj["best_view"]).exists()
                    ]
                    choices = [
                        (f'{obj["label"]} ({obj["id"]})', obj["id"])
                        for obj in found
                    ]
                    state = latest if "found_objects" in latest else {
                        **latest,
                        "found_objects": found,
                    }
                    yield (
                        "\n\n".join(lines[-30:]),
                        current_image,
                        gallery,
                        gr.update(choices=choices),
                        latest,
                        state,
                    )
            except Exception as exc:
                yield (
                    f"扫描失败: {exc}", None, [], gr.update(choices=[]),
                    {"error": str(exc)}, {},
                )

        explore_run_btn.click(
            fn=_run_explorer_scan,
            inputs=[
                scene_state,
                explore_cam_data,
                explore_labels,
                explore_num_views,
                explore_turn_degrees,
            ],
            outputs=[
                explore_progress,
                explore_current_view,
                explore_gallery,
                explore_select,
                explore_output,
                explorer_results_state,
            ],
        )

        async def _run_approved_meshes(
            selected_ids: list[str],
            explorer_state: dict,
            width_m: float,
            height_m: float,
            seed: int,
            register_in_revit: bool,
        ):
            found = explorer_state.get("found_objects", []) if explorer_state else []
            selected = [
                obj for obj in found if obj.get("id") in set(selected_ids or [])
            ]
            if not selected:
                yield "请先勾选至少一个已发现物体。", {"error": "no_approval"}
                return
            status = explorer_state.get("status", {})
            up_axis = status.get("camera", {}).get("up_axis", 2)
            approved = tuple(
                ApprovedMeshObject(
                    object_id=obj["id"],
                    label=obj["label"],
                    image_path=Path(obj["best_view"]),
                    position_3d=tuple(float(value) for value in obj["position_3d"]),
                    up_axis=int(up_axis),
                    width_m=float(width_m),
                    height_m=float(height_m),
                    seed=int(seed),
                )
                for obj in selected
            )
            app_config = load_config()
            trellis_config = app_config.trellis
            gateway = None
            if register_in_revit:
                revit = app_config.revit_mcp
                gateway = StdioMCPGateway(
                    command=revit.command,
                    args=tuple(revit.args),
                    cwd=str(ROOT),
                    timeout_seconds=float(revit.timeout),
                )
            output_root = Path(
                explorer_state.get(
                    "output_dir",
                    ROOT / "output" / "_trellis_meshes",
                )
            ) / "meshes"
            workflow = TrellisRevitWorkflow(
                TrellisWorkflowConfig(
                    objects=approved,
                    output_dir=output_root,
                    register_in_revit=bool(register_in_revit),
                ),
                client_factory=lambda: TrellisClient(
                    host=trellis_config.host,
                    port=trellis_config.port,
                    timeout=trellis_config.timeout,
                ),
                gateway=gateway,
            )
            lines: list[str] = []
            latest: dict = {}
            async for update in stream_workflow_gradio(workflow):
                lines.append(update.message)
                latest = update.payload or latest
                yield "\n\n".join(lines[-30:]), latest

        bmesh_generate_selected_btn.click(
            fn=_run_approved_meshes,
            inputs=[
                explore_select,
                explorer_results_state,
                bmesh_width,
                bmesh_height,
                bmesh_seed,
                bmesh_register_revit,
            ],
            outputs=[bmesh_workflow_status, bmesh_workflow_output],
        )

        # --- B类 Mesh: 手动选取 Tab ---

        def _on_bmesh_capture(scene_name: str, viewer_session: dict):
            """Capture a camera from the manager-assigned viewer and render it."""
            if not scene_name:
                return None, "❌ 请先选择场景"
            status, cam_data = fetch_camera_state(viewer_session)
            if not cam_data:
                return None, status

            scene = _get_scene(scene_name)
            if scene is None:
                return None, f"❌ 无法加载场景 {scene_name}"

            cam = cam_data
            eye = cam.get("position", [0, 0, 0])
            target = cam.get("look_at", [0, 0, 1])
            fov = cam.get("fov_degrees", 60)
            up = cam.get("up", [0, 0, 1])

            from bim_recon.gs_scene import look_at_pose
            pose = look_at_pose(
                (eye[0], eye[1], eye[2]),
                (target[0], target[1], target[2]),
                up=(up[0], up[1], up[2]),
            )
            render_result = scene.render(pose, width=800, height=600, fov_degrees=fov)
            render_arr = (render_result.colors * 255).clip(0, 255).astype(np.uint8)

            return render_arr, f"✅ 渲染完成 ({render_arr.shape[1]}×{render_arr.shape[0]})"

        bmesh_cam_btn.click(
            fn=_on_bmesh_capture,
            inputs=[scene_state, viewer_state],
            outputs=[bmesh_mask_editor, bmesh_cam_status],
        )

        def _on_bmesh_identify(mask_editor_val, scene_name: str):
            """Rough-bbox → VLM classify → Falcon segment → clean RGBA cutout."""
            from bim_recon.bmesh_extractor import classify_and_segment
            from bim_recon.vlm_verifier import query_vlm

            cfg = load_config()
            falcon = _get_falcon()

            def vlm_caller(image_path: str, prompt: str) -> str:
                return query_vlm(
                    image_path, prompt,
                    cfg.vlm.api_base, cfg.vlm.model, cfg.vlm.api_key,
                    timeout=30, max_tokens=20,
                )

            result = classify_and_segment(mask_editor_val, vlm_caller, falcon)

            cutout_path = None
            cutout_preview = None
            if result.cutout is not None:
                out_dir = ROOT / "output" / (scene_name or "default") / "_trellis_meshes"
                out_dir.mkdir(parents=True, exist_ok=True)
                cutout_path = str(out_dir / f"{result.label}_{int(time.time())}_cutout.png")
                result.cutout.save(cutout_path)
                cutout_preview = np.array(result.cutout)

            return (
                result.label,                          # bmesh_identified_label
                result.overlay,                        # bmesh_segmentation_preview
                cutout_preview,                        # bmesh_manual_preview
                cutout_path,                           # bmesh_cutout_state
                result.detail,                         # bmesh_cam_status
            )

        bmesh_identify_btn.click(
            fn=_on_bmesh_identify,
            inputs=[bmesh_mask_editor, scene_state],
            outputs=[
                bmesh_identified_label, bmesh_segmentation_preview,
                bmesh_manual_preview, bmesh_cutout_state, bmesh_cam_status,
            ],
        )

        def _on_bmesh_generate(cutout_path: str | None, label: str, scene_name: str, seed: int):
            """Send the Falcon-segmented cutout to TRELLIS for mesh generation."""
            if not cutout_path:
                return None, {"错误": "请先点击「🔍 VLM识别 + Falcon分割」生成抠图"}
            from PIL import Image as PILImage

            preview = np.array(PILImage.open(cutout_path).convert("RGBA"))
            cfg = load_config()
            trellis = TrellisClient(host=cfg.trellis.host, port=cfg.trellis.port, timeout=cfg.trellis.timeout)
            if not trellis.health():
                return preview, {"错误": "TRELLIS 服务不可达"}

            out_dir = ROOT / "output" / (scene_name or "default") / "_trellis_meshes"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{label or 'object'}_{int(time.time())}"
            clean_path = out_dir / f"{name}_clean.png"
            PILImage.open(cutout_path).save(str(clean_path))

            try:
                result = trellis.generate_mesh(TrellisMeshRequest(
                    image_path=clean_path,
                    output_dir=out_dir,
                    name=name,
                    seed=int(seed),
                ))
                return preview, {
                    "状态": "✅ Mesh 生成成功",
                    "GLB": str(result.glb_path),
                    "PLY": str(result.gaussian_path) if result.gaussian_path else None,
                }
            except Exception as e:
                return preview, {"错误": f"TRELLIS 生成失败: {e}"}

        bmesh_gen_manual_btn.click(
            fn=_on_bmesh_generate,
            inputs=[bmesh_cutout_state, bmesh_identified_label, scene_state, bmesh_seed_manual],
            outputs=[bmesh_manual_preview, bmesh_manual_output],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=19255, server_name="127.0.0.1")
