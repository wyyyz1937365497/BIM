"""3DGS → BIM Pipeline — Gradio Web UI (modular).

UI layout + event wiring only. All callback logic lives in
``bim_recon.gradio_helpers``.

Launch:
    python scripts/gradio_app.py
The gsplat renderer initializes the Visual Studio compiler environment on demand.
"""
from __future__ import annotations

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
                bmesh_gen_manual_btn = gr.Button(
                    "生成 Mesh + 注册 Revit", variant="primary",
                )
            bmesh_identified_label = gr.Textbox(
                label="VLM 识别结果", interactive=False,
            )
            bmesh_segmentation_preview = gr.Image(
                label="Falcon 分割结果", height=300,
            )
            bmesh_manual_preview = gr.Image(label="抠图预览", height=300)
            bmesh_manual_output = gr.JSON(label="生成结果")
            bmesh_cutout_state = gr.State(None)
            bmesh_render_state = gr.State(None)
            bmesh_cam_state = gr.State(None)
            bmesh_detection_state = gr.State(None)
            bmesh_scene_state = gr.State("")
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

        bmesh_identify_btn.click(
            fn=_on_bmesh_identify,
            inputs=[bmesh_mask_editor, bmesh_render_state],
            outputs=[
                bmesh_identified_label, bmesh_segmentation_preview,
                bmesh_manual_preview, bmesh_cutout_state,
                bmesh_detection_state, bmesh_cam_status,
            ],
        )

        def _unproject_pixel_to_world(cam, depth_map, px, py, img_w, img_h):
            """Unproject a pixel + depth to 3DGS world coordinates.

            Uses the camera look-at convention: forward toward target, up
            roughly aligned with world up.  Returns (x, y, z) in meters.
            """
            eye = np.array(cam["eye"], dtype=np.float64)
            target = np.array(cam["target"], dtype=np.float64)
            up_world = np.array(cam.get("up", [0, 0, 1]), dtype=np.float64)
            vfov_rad = math.radians(cam.get("fov", 60.0))

            iy = max(0, min(img_h - 1, int(py)))
            ix = max(0, min(img_w - 1, int(px)))
            d = float(depth_map[iy, ix])
            if d <= 0.1:
                # Fallback: median of non-zero depths near the pixel
                y0 = max(0, iy - 20)
                y1 = min(img_h, iy + 20)
                x0 = max(0, ix - 20)
                x1 = min(img_w, ix + 20)
                patch = depth_map[y0:y1, x0:x1]
                valid = patch[patch > 0.1]
                if len(valid) == 0:
                    return None
                d = float(np.median(valid))

            focal_y = 0.5 * img_h / math.tan(vfov_rad / 2.0)
            focal_x = focal_y  # square pixels

            x_cam = (px - img_w / 2.0) / focal_x * d
            y_cam = (py - img_h / 2.0) / focal_y * d
            z_cam = d

            forward = target - eye
            forward /= np.linalg.norm(forward) + 1e-12
            right = np.cross(forward, up_world)
            right /= np.linalg.norm(right) + 1e-12
            down = np.cross(forward, right)

            world = eye + right * x_cam + down * y_cam + forward * z_cam
            return world, d

        def _on_bmesh_generate(cutout_path, label, render_state, detection_info,
                               results, seed):
            """TRELLIS mesh → world placement → Revit DirectShape."""
            if not cutout_path:
                return None, {"错误": "请先点击「🔍 VLM识别 + Falcon分割」生成抠图"}
            from PIL import Image as PILImage

            preview = np.array(PILImage.open(cutout_path).convert("RGBA"))
            cfg = load_config()
            trellis = TrellisClient(host=cfg.trellis.host, port=cfg.trellis.port, timeout=cfg.trellis.timeout)
            if not trellis.health():
                return preview, {"错误": "TRELLIS 服务不可达"}

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
                return preview, {"错误": f"TRELLIS 生成失败: {e}"}

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
                cx_norm = norm_bbox.get("x", 0.5)
                cy_norm = norm_bbox.get("y", 0.5)
                w_norm = norm_bbox.get("w", 0.2)
                h_norm = norm_bbox.get("h", 0.2)

                px_center = cx_norm * img_w
                py_center = cy_norm * img_h
                result_unproj = _unproject_pixel_to_world(
                    cam, depth_map, px_center, py_center, img_w, img_h,
                )
                if result_unproj is not None:
                    world_pos, depth_val = result_unproj
                    up_axis = cam.get("up_axis", 2)
                    h_axes = [i for i in range(3) if i != up_axis]
                    world_x = float(world_pos[h_axes[0]])
                    world_y = float(world_pos[h_axes[1]])
                    world_z = float(world_pos[up_axis])

                    # Estimate physical dimensions from bbox + depth + FOV
                    vfov_rad = math.radians(cam.get("fov", 60.0))
                    aspect = img_w / img_h
                    hfov_rad = 2.0 * math.atan(math.tan(vfov_rad / 2.0) * aspect)
                    element_width_m = float(w_norm * 2.0 * depth_val * math.tan(hfov_rad / 2.0))
                    element_height_m = float(h_norm * 2.0 * depth_val * math.tan(vfov_rad / 2.0))

                    # Floor/ceiling from pipeline results or defaults
                    coords = (results.coords if results else {}) or {}
                    floor_z = float(coords.get("floor_z", 0.0))
                    ceiling_z = float(coords.get("ceiling_z", 3.0))

                    placement_info = {
                        "world_x": round(world_x, 3),
                        "world_y": round(world_y, 3),
                        "world_z": round(world_z, 3),
                        "element_width_m": round(element_width_m, 3),
                        "element_height_m": round(element_height_m, 3),
                        "depth": round(depth_val, 3),
                    }

                    # --- Register mesh in Revit ---
                    from bim_recon.mesh_registrar import (
                        MeshPlacement, compute_placement_transform,
                        register_mesh_in_revit,
                    )
                    from bim_recon.revit_runner import RevitScriptRunner
                    placement = MeshPlacement(
                        glb_path=Path(mesh_result.glb_path),
                        world_x=world_x,
                        world_y=world_y,
                        floor_z=floor_z,
                        ceiling_z=ceiling_z,
                        element_width_m=max(element_width_m, 0.1),
                        element_height_m=max(element_height_m, 0.1),
                        up_axis=up_axis,
                        category="OST_GenericModel",
                        name=label or name,
                    )
                    try:
                        transform = compute_placement_transform(placement)
                        from bim_recon.mcp_gateway import StdioMCPGateway
                        revit_cfg = cfg.revit_mcp
                        runner = RevitScriptRunner(mcp_sender=None)
                        revit_result = register_mesh_in_revit(
                            placement, transform, runner=runner,
                        )
                        placement_info["revit"] = revit_result
                        if revit_result.get("status") == "formatted":
                            # Call the compiled create_directshape_from_mesh MCP tool
                            gateway = StdioMCPGateway(
                                command=revit_cfg.command,
                                args=tuple(revit_cfg.args),
                                cwd=str(ROOT),
                                timeout_seconds=float(revit_cfg.timeout),
                            )
                            import asyncio as _aio
                            resp = _aio.run(gateway.call_tool(
                                "create_directshape_from_mesh",
                                {"meshFile": revit_result["payload_path"]},
                            ))
                            placement_info["revit_response"] = resp
                            output["Revit"] = "✅ DirectShape 已创建"
                    except Exception as exc:
                        output["Revit 错误"] = str(exc)

            output["placement"] = placement_info
            return preview, output

        bmesh_gen_manual_btn.click(
            fn=_on_bmesh_generate,
            inputs=[
                bmesh_cutout_state, bmesh_identified_label,
                bmesh_render_state, bmesh_detection_state,
                results_state, bmesh_seed_manual,
            ],
            outputs=[bmesh_manual_preview, bmesh_manual_output],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=19255, server_name="127.0.0.1")
