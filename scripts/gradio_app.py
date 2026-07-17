"""3DGS → BIM Pipeline — Gradio Web UI (modular).

UI layout + event wiring only. All callback logic lives in
``bim_recon.gradio_helpers``.

Launch:
    python scripts/gradio_app.py
The gsplat renderer initializes the Visual Studio compiler environment on demand.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
from PIL import Image

# Import all helpers (callbacks, business logic, constants)
from bim_recon.gradio_helpers import (
    logger, ROOT, VIEWER_PORT, CAMERA_PORT, MAX_CONSOLE_LINES,
    SCENESPLAT, MAX_PORT_WAIT_S, _SCENE_CACHE, _FALCON_CACHE,
    validate_ply, check_preprocess_status, list_available_scenes,
    _wait_for_port, _kill_child_procs, start_viewer,
    find_latest_output, list_available_results, _result_dir_from_label,
    _prepare_results, run_pipeline_direct, load_results_cb,
    _find_element, update_interactive_radar, draw_bbox_on_image,
    on_element_select, on_mask_apply, fetch_camera_state,
    _get_scene, _get_falcon, _mask_bbox_to_wall_coords,
    resegment_from_viewpoint, apply_vlm_review,
    _check_revit_port, _format_agent_error, _get_agent,
    _build_agent_context, agent_chat,
    _reset_explorer_agent, _get_explorer_agent,
)
from bim_recon.config import load_config, save_config, test_llm_connection
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest
from bim_recon.pipeline_api import PipelineResults


def build_app() -> gr.Blocks:
    with gr.Blocks(title="3DGS → BIM 管线") as app:
        gr.Markdown("# 3DGS → BIM 自动重建管线")
        scene_state = gr.State("")
        results_state = gr.State(None)

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
            "**方式二** — 视角重分割：在下方查看器漫游到覆盖构件的视角 → 获取视角 → 以此视角重新分割"
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
                cam_btn = gr.Button("📸 从查看器获取视角", variant="secondary")
                cam_status = gr.Markdown("（需先启动下方查看器）")
                reseg_btn = gr.Button("🔲 以此视角重新分割", variant="primary")
                reseg_preview = gr.Image(
                    label="新视角渲染 + Falcon 分割", height=300,
                )
                reseg_out = gr.JSON(label="视角重分割结果")

        # ====== ⑤ AI Agent（Revit MCP） ======
        gr.Markdown("---\n## ⑤ AI Agent（Revit MCP）")
        _init_cfg = load_config()
        with gr.Accordion("⚙️ LLM API 配置", open=False):
            with gr.Row():
                llm_api_base = gr.Textbox(
                    label="API Base", value=_init_cfg.llm.api_base,
                    scale=3, info="OpenAI 兼容端点，如 http://127.0.0.1:11434/v1",
                )
                llm_api_key = gr.Textbox(
                    label="API Key", value=_init_cfg.llm.api_key,
                    scale=2, type="password",
                )
            with gr.Row():
                llm_model = gr.Textbox(
                    label="模型名称", value=_init_cfg.llm.model,
                    scale=3, info="如 qwen2.5:32b / gpt-4o / deepseek-chat",
                )
                test_btn = gr.Button("🔌 测试连接", scale=1)
                save_cfg_btn = gr.Button("💾 保存配置", variant="primary", scale=1)
            test_status = gr.Markdown("")
        gr.Markdown(
            "Agent 可使用 Revit MCP 工具自动创建构件。需要 Revit 已运行 + MCP 插件已加载。"
        )
        agent_chatbot = gr.Chatbot(
            label="对话", height=400,
            placeholder="告诉 Agent 要做什么，如「把检测到的墙全部导入 Revit」",
        )
        with gr.Row():
            agent_input = gr.Textbox(
                label="消息", scale=4, placeholder="如：创建所有墙体，厚度200mm，高度3000mm",
            )
            agent_send = gr.Button("发送", variant="primary", scale=1)

        # ====== ⑤b B类构件 Mesh 生成（TRELLIS） ======
        gr.Markdown("---\n## ⑤b B类构件 Mesh 生成（TRELLIS）")
        gr.Markdown(
            "B 类构件 = 家具/管道/楼梯等复杂异形件，通过 TRELLIS 生成 3D mesh。\n\n"
            "**VLM 自动探索**：VLM 在场景中 360° 自主探索，Falcon 分割 + 3D 定位\n"
            "**手动选取**：在 ⑥ 查看器中漫游到目标 → 捕获渲染 → 手动 mask + 输入物体名 → 生成"
        )

        # B-class mesh state
        bmesh_scene_state = gr.State("")  # for rendering
        bmesh_cam_data = gr.State({})     # captured camera

        with gr.Tab("VLM 自动探索"):
            gr.Markdown(
                "VLM Agent 在 3DGS 场景中自主导航，通过 Falcon 检测细粒度物体。\n"
                "需要 Falcon server (端口 8390) 运行中。\n\n"
                "**流程**：设置初始视角 → 初始化 → 在对话中引导 Agent 搜索物体"
            )
            with gr.Accordion("📍 初始视角设置", open=True):
                with gr.Row():
                    explore_cam_btn = gr.Button("📸 从查看器获取视角", variant="secondary", scale=1)
                    explore_cam_status = gr.Markdown(
                        "先在 **⑥ 查看器** 中漫游到目标位置，再点击上方按钮获取视角。", scale=2,
                    )
                explore_cam_data = gr.State({})
                explore_init_btn = gr.Button("🚀 初始化探索 Agent", variant="primary")
                explore_init_status = gr.Markdown("")
            explore_chatbot = gr.Chatbot(
                label="VLM 探索过程", height=400,
                placeholder="初始化后，输入指令如「搜索房间内所有椅子和桌子」",
            )
            with gr.Row():
                explore_input = gr.Textbox(
                    label="指令", scale=4,
                    placeholder="如：搜索房间内所有椅子、桌子、柜子、花瓶",
                )
                explore_send = gr.Button("发送", variant="primary", scale=1)
            explore_gallery = gr.Gallery(
                label="已发现物体", columns=4, height=250,
                show_label=False, object_fit="contain", preview=True,
            )
            explore_results = gr.State([])
            with gr.Row():
                explore_refresh_btn = gr.Button("🔄 刷新已发现物体", variant="secondary")
                explore_queue_btn = gr.Button("📦 全部加入 TRELLIS 队列", variant="secondary")
            explore_output = gr.JSON(label="队列状态")

        with gr.Tab("手动选取（视角 + Mask）"):
            gr.Markdown(
                "**步骤**：① 启动 ⑥ 查看器 → ② 漫游到目标物体 → ③ 捕获视角 → ④ 在渲染图上用画笔涂出物体 → ⑤ 输入物体名称 → ⑥ 生成"
            )
            with gr.Row():
                bmesh_cam_btn = gr.Button("📸 捕获视角并渲染", variant="secondary", scale=1)
                bmesh_prompt = gr.Textbox(
                    label="物体名称（英文，如: chair / lamp / sofa）",
                    placeholder="输入物体名称...",
                    scale=2,
                )
            bmesh_cam_status = gr.Markdown("（需先启动查看器）")
            bmesh_mask_editor = gr.ImageMask(
                label="渲染图（用红色画笔涂出要提取的物体）",
                type="numpy", height=400,
                layers=False,
                brush=gr.Brush(colors=["#FF0000"], default_size=30, color_mode="fixed"),
                transforms=[], sources=[],
            )
            with gr.Row():
                bmesh_seed_manual = gr.Number(label="种子", value=1, precision=0, scale=1)
                bmesh_gen_manual_btn = gr.Button("🎯 Mask + 生成 Mesh", variant="primary", scale=2)
            bmesh_manual_preview = gr.Image(label="抠图预览", height=300)
            bmesh_manual_output = gr.JSON(label="生成结果")

        # ====== ⑥ 3D 查看器（底部） ======
        gr.Markdown("---\n## ⑥ 3D 查看器")
        viewer_btn = gr.Button("▶ 启动查看器", variant="secondary")
        viewer_html = gr.HTML("<p>点击上方按钮启动 3DGS 查看器（nerfview + 相机捕获端口 8082）</p>")

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
        viewer_btn.click(fn=start_viewer, inputs=scene_state, outputs=viewer_html)

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
            inputs=[scene_state, cb_doors, cb_windows, cb_falcon, cb_skipvlm],
            outputs=[console_out, out_dir_box, results_state, summary_md,
                      radar_gallery, vlm_gallery, report_json,
                     vlm_review_cbs, elem_sel],
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
        review_btn.click(fn=apply_vlm_review,
                         inputs=[results_state, vlm_review_cbs],
                         outputs=review_status)
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

        # --- AI Agent ---
        def _on_test_llm(api_base: str, api_key: str, model: str) -> str:
            return test_llm_connection(api_base, api_key, model)

        def _on_save_config(api_base: str, api_key: str, model: str) -> str:
            global _AGENT_CACHE, _MCP_CM_CACHE
            cfg = load_config()
            from dataclasses import replace as dc_replace
            new_cfg = dc_replace(cfg, llm=dc_replace(cfg.llm,
                api_base=api_base, api_key=api_key, model=model))
            save_config(new_cfg)
            _AGENT_CACHE = None      # force re-create agent with new model
            _MCP_CM_CACHE = None     # force re-connect MCP server
            return "✅ 配置已保存到 config.json，Agent 将使用新模型"

        test_btn.click(
            fn=_on_test_llm,
            inputs=[llm_api_base, llm_api_key, llm_model],
            outputs=test_status,
        )
        save_cfg_btn.click(
            fn=_on_save_config,
            inputs=[llm_api_base, llm_api_key, llm_model],
            outputs=test_status,
        )

        def _agent_respond(message: str, history: list, results, scene_name: str):
            if not message.strip():
                yield history, ""
                return

            from smolagents import (
                ActionStep, PlanningStep, FinalAnswerStep, ChatMessageStreamDelta,
            )

            # 1. 显示用户消息
            history = history + [{"role": "user", "content": message}]
            yield history, ""

            # 2. 初始化一个不断更新的 assistant 消息
            assistant_msg = {"role": "assistant", "content": "🤔 **Agent 思考中...**\n\n"}
            history = history + [assistant_msg]
            yield history, ""

            # 累积每一步的文本
            steps_lines: list[str] = []
            current_step_lines: list[str] = []
            final_answer = ""
            step_count = 0
            in_tool_call = False

            try:
                agent = _get_agent(results, scene_name)

                for step in agent.run(message, stream=True):
                    updated = False

                    # --- PlanningStep: 规划 ---
                    if isinstance(step, PlanningStep):
                        plan = step.plan if hasattr(step, "plan") else ""
                        if plan:
                            steps_lines.append(f"📋 **规划**: {plan}")
                            updated = True

                    # --- ChatMessageStreamDelta: LLM token 流 ---
                    elif isinstance(step, ChatMessageStreamDelta):
                        delta = step.content if hasattr(step, "content") else ""
                        if delta and not in_tool_call:
                            # token 流附加到当前步骤
                            if current_step_lines:
                                current_step_lines[-1] += delta
                            else:
                                current_step_lines.append(delta)
                            updated = True

                    # --- ActionStep: 工具调用 / 观察 ---
                    elif isinstance(step, ActionStep):
                        step_count += 1

                        # 工具调用
                        if step.tool_calls:
                            for tc in step.tool_calls:
                                args = tc.arguments
                                if isinstance(args, dict):
                                    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                                else:
                                    arg_str = str(args)
                                current_step_lines.append(
                                    f"**步骤 {step_count}**: 🔧 调用 `{tc.name}({arg_str})`"
                                )
                                in_tool_call = True
                            updated = True

                        # 观察结果
                        if step.observations:
                            obs = str(step.observations).strip()
                            if obs:
                                current_step_lines.append(f"→ **结果**: {obs}")
                                in_tool_call = False
                                # 完成一个完整步骤，保存到历史
                                steps_lines.extend(current_step_lines)
                                current_step_lines = []
                                updated = True

                        # 错误
                        if step.error:
                            err = str(step.error)
                            current_step_lines.append(f"→ **错误**: {err}")
                            steps_lines.extend(current_step_lines)
                            current_step_lines = []
                            updated = True

                        # 最终答案标记
                        if step.is_final_answer:
                            if current_step_lines:
                                steps_lines.extend(current_step_lines)
                                current_step_lines = []

                    # --- FinalAnswerStep: 最终答案 ---
                    elif isinstance(step, FinalAnswerStep):
                        final_answer = str(step.output) if hasattr(step, "output") else str(step)
                        if current_step_lines:
                            steps_lines.extend(current_step_lines)
                            current_step_lines = []
                        updated = True

                    # 更新 chatbot 显示
                    if updated:
                        content_parts = ["🤔 **Agent 思考中...**\n"]
                        content_parts.extend(steps_lines)
                        if current_step_lines:
                            content_parts.extend(current_step_lines)

                        # 如果有最终答案，替换为最终答案
                        if final_answer:
                            content_parts = [final_answer]

                        history[-1] = {
                            "role": "assistant",
                            "content": "\n".join(content_parts),
                        }
                        yield history, ""

                # 流结束后的最终更新
                if not final_answer and steps_lines:
                    history[-1] = {
                        "role": "assistant",
                        "content": "\n".join(steps_lines),
                    }
                    yield history, ""

            except Exception as e:
                error_msg = _format_agent_error(e)
                history[-1] = {"role": "assistant", "content": error_msg}
                yield history, ""

        agent_send.click(
            fn=_agent_respond,
            inputs=[agent_input, agent_chatbot, results_state, scene_state],
            outputs=[agent_chatbot, agent_input],
        )
        agent_input.submit(
            fn=_agent_respond,
            inputs=[agent_input, agent_chatbot, results_state, scene_state],
            outputs=[agent_chatbot, agent_input],
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

        # --- B类 Mesh: VLM Agent 探索 Tab ---
        explore_cam_btn.click(
            fn=fetch_camera_state,
            outputs=[explore_cam_status, explore_cam_data],
        )

        def _explorer_init(scene_name: str, camera_data: dict):
            """初始化探索 Agent：用查看器视角 + 启动 MCP 子进程 + 首条消息。"""
            if not scene_name:
                return "❌ 未选择场景", []
            if not camera_data or "position" not in camera_data:
                return "⚠️ 请先点击「📸 从查看器获取视角」", []
            import math
            pos = camera_data["position"]
            look = camera_data["look_at"]
            eye_x, eye_y, eye_z = float(pos[0]), float(pos[1]), float(pos[2])
            dx, dz = float(look[0]) - eye_x, float(look[2]) - eye_z
            yaw = round(math.degrees(math.atan2(dz, dx)), 1)
            try:
                _reset_explorer_agent()
                agent = _get_explorer_agent(scene_name)
                msg = (
                    f"请在位置 ({eye_x:.2f}, {eye_y:.2f}, {eye_z:.2f}) "
                    f"以朝向 {yaw}° 初始化探索。"
                    f"调用 explore_init(eye_x={eye_x:.2f}, eye_y={eye_y:.2f}, "
                    f"eye_z={eye_z:.2f}, initial_yaw={yaw})，"
                    f"然后简要描述你看到的场景（2-3 句话）。"
                )
                response = agent.run(msg)
                chat = [{"role": "assistant", "content": str(response)}]
                return f"✅ Agent 已初始化 (视角来自查看器: {eye_x:.1f}, {eye_y:.1f}, {eye_z:.1f}, yaw={yaw}°)", chat
            except Exception as e:
                return _format_agent_error(e), []

        explore_init_btn.click(
            fn=_explorer_init,
            inputs=[scene_state, explore_cam_data],
            outputs=[explore_init_status, explore_chatbot],
        )

        def _explorer_respond(message: str, history: list, scene_name: str):
            """流式显示 Agent 探索过程（思考 + 工具调用）。"""
            if not message.strip():
                yield history, ""
                return
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            try:
                agent = _get_explorer_agent(scene_name)
                from smolagents import PlanningStep, FinalAnswerStep

                steps_lines: list[str] = []
                current_step_lines: list[str] = []
                final_answer = ""

                for step in agent.run(message, stream=True):
                    updated = False
                    if isinstance(step, PlanningStep):
                        steps_lines.append(f"📋 **计划**: {step.plan}")
                        updated = True
                    elif hasattr(step, "tool_calls") and step.tool_calls:
                        current_step_lines = []
                        for tc in step.tool_calls:
                            name = tc.name if hasattr(tc, "name") else str(tc)
                            args_str = str(tc.arguments) if hasattr(tc, "arguments") else ""
                            if len(args_str) > 200:
                                args_str = args_str[:200] + "…"
                            current_step_lines.append(f"🔧 **调用**: `{name}`({args_str})")
                        if hasattr(step, "observations") and step.observations:
                            obs = str(step.observations)[:300]
                            current_step_lines.append(f"   ↳ {obs}")
                        updated = True
                    elif isinstance(step, FinalAnswerStep):
                        final_answer = str(step.output) if hasattr(step, "output") else str(step)
                        updated = True

                    if updated:
                        parts = ["🤔 **Agent 探索中...**\n"]
                        parts.extend(steps_lines)
                        parts.extend(current_step_lines)
                        if final_answer:
                            parts = [final_answer]
                        history[-1] = {"role": "assistant", "content": "\n".join(parts)}
                        yield history, ""

                if not final_answer and steps_lines:
                    history[-1] = {"role": "assistant", "content": "\n".join(steps_lines)}
                    yield history, ""
            except Exception as e:
                history[-1] = {"role": "assistant", "content": _format_agent_error(e)}
                yield history, ""

        explore_send.click(
            fn=_explorer_respond,
            inputs=[explore_input, explore_chatbot, scene_state],
            outputs=[explore_chatbot, explore_input],
        )
        explore_input.submit(
            fn=_explorer_respond,
            inputs=[explore_input, explore_chatbot, scene_state],
            outputs=[explore_chatbot, explore_input],
        )

        def _refresh_explore_gallery(scene_name: str):
            """从 found_objects.json 读取已发现物体。"""
            if not scene_name:
                return [], []
            found_path = ROOT / "output" / scene_name / "explore" / "found_objects.json"
            if not found_path.exists():
                return [], []
            found = json.loads(found_path.read_text(encoding="utf-8"))
            gallery = [
                (o["best_view"], f'{o["label"]} #{o["id"]}')
                for o in found if o.get("best_view") and Path(o["best_view"]).exists()
            ]
            return gallery, found

        explore_refresh_btn.click(
            fn=_refresh_explore_gallery,
            inputs=[scene_state],
            outputs=[explore_gallery, explore_results],
        )

        def _queue_all_for_trellis(scene_name: str):
            """从 found_objects.json 读取，写入 TRELLIS 队列。"""
            if not scene_name:
                return {"错误": "未选择场景"}
            found_path = ROOT / "output" / scene_name / "explore" / "found_objects.json"
            if not found_path.exists():
                return {"错误": "没有已发现的物体"}
            found = json.loads(found_path.read_text(encoding="utf-8"))
            if not found:
                return {"错误": "没有已发现的物体"}
            queue_dir = ROOT / "output" / scene_name / "trellis_queue"
            queue_dir.mkdir(parents=True, exist_ok=True)
            queued = 0
            for obj in found:
                entry = {
                    "object_id": obj["id"], "label": obj["label"],
                    "image_path": obj.get("best_view", ""),
                    "position_3d": obj.get("position_3d", [0, 0, 0]),
                }
                (queue_dir / f"{obj['id']}.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8",
                )
                queued += 1
            return {"状态": f"已加入 {queued} 个物体到 TRELLIS 队列",
                    "队列目录": str(queue_dir)}

        explore_queue_btn.click(
            fn=_queue_all_for_trellis,
            inputs=[scene_state],
            outputs=[explore_output],
        )

        # --- B类 Mesh: 手动选取 Tab ---

        def _on_bmesh_capture(scene_name: str):
            """捕获视角 → 渲染 → 加载到 ImageMask。"""
            if not scene_name:
                return None, "❌ 请先选择场景"
            status, cam_data = fetch_camera_state()
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
            inputs=[scene_state],
            outputs=[bmesh_mask_editor, bmesh_cam_status],
        )

        def _on_bmesh_manual_generate(mask_editor_val, prompt: str, scene_name: str, seed: int):
            """从 ImageMask 提取 mask → 清理背景 → TRELLIS 生成。"""
            if not prompt.strip():
                return None, {"错误": "请输入物体名称"}
            if mask_editor_val is None:
                return None, {"错误": "请先捕获视角并渲染"}

            layers = (mask_editor_val or {}).get("layers", [])
            if not layers:
                return None, {"错误": "请用红色画笔涂出要提取的物体"}

            mask_arr = layers[0]
            if not isinstance(mask_arr, np.ndarray) or mask_arr.ndim < 3:
                return None, {"错误": f"Mask 格式异常: {type(mask_arr)}"}

            from PIL import Image as PILImage

            # Extract background image from the editor (composite layer)
            bg = mask_editor_val.get("background")
            if bg is not None and isinstance(bg, np.ndarray):
                base_img = PILImage.fromarray(bg.astype(np.uint8)).convert("RGB")
            else:
                # Use the mask array itself (RGBA, background is the composite)
                base_img = PILImage.fromarray(mask_arr[:, :, :3].astype(np.uint8)).convert("RGB")

            # Alpha channel from mask layer = where user drew
            alpha = mask_arr[:, :, 3] if mask_arr.shape[2] == 4 else np.zeros(mask_arr.shape[:2])
            has_mask = alpha > 10

            if not has_mask.any():
                return None, {"错误": "未检测到绘制内容 — 请在渲染图上涂出物体"}

            # Find tight bbox of mask
            rows = np.any(has_mask, axis=1)
            cols = np.any(has_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop with padding
            pad = 20
            h_img, w_img = base_img.size[1], base_img.size[0]
            rmin = max(0, rmin - pad)
            rmax = min(h_img, rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(w_img, cmax + pad)

            cropped = base_img.crop((cmin, rmin, cmax, rmax))
            alpha_crop = alpha[rmin:rmax, cmin:cmax]
            rgba = cropped.convert("RGBA")
            rgba.putalpha(Image.fromarray(alpha_crop.astype(np.uint8), mode="L"))

            cfg = load_config()
            trellis = TrellisClient(host=cfg.trellis.host, port=cfg.trellis.port, timeout=cfg.trellis.timeout)
            if not trellis.health():
                return rgba, {"错误": "TRELLIS 服务不可达"}

            out_dir = ROOT / "output" / (scene_name or "default") / "_trellis_meshes"
            out_dir.mkdir(parents=True, exist_ok=True)
            clean_path = out_dir / f"{prompt}_{int(time.time())}_clean.png"
            rgba.save(str(clean_path))

            try:
                result = trellis.generate_mesh(TrellisMeshRequest(
                    image_path=clean_path,
                    output_dir=out_dir,
                    name=f"{prompt}_{int(time.time())}",
                    seed=int(seed),
                ))
                return rgba, {
                    "状态": "✅ Mask + 生成成功",
                    "GLB": str(result.glb_path),
                    "PLY": str(result.gaussian_path) if result.gaussian_path else None,
                }
            except Exception as e:
                return rgba, {"错误": f"TRELLIS 生成失败: {e}"}

        bmesh_gen_manual_btn.click(
            fn=_on_bmesh_manual_generate,
            inputs=[bmesh_mask_editor, bmesh_prompt, scene_state, bmesh_seed_manual],
            outputs=[bmesh_manual_preview, bmesh_manual_output],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=19255, server_name="127.0.0.1")
