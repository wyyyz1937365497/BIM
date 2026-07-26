"""Focused Gradio workspace for TRELLIS generation and automatic registration.

Launch with ``scripts/launch_trellis_registration.bat``. This deliberately
keeps the main BIM pipeline untouched: it generates GLB assets and solves the
first registration pass through the project's existing silhouette-search mesh
registrar. Use the main Gradio page for full render-compare + ICP refinement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from bim_recon.config import load_config
from bim_recon.gradio_helpers import (
    launch_scene_viewer,
    list_available_results,
    list_available_scenes,
)
from bim_recon.mesh_registrar import parse_glb_vertices_faces
from bim_recon.radar_viz import (
    draw_spatial_context as _draw_spatial_context,
    observation_radar as _observation_radar,
    registration_radar as _registration_radar,
)
from bim_recon.trellis_client import TrellisClient
from bim_recon.trellis_registration import (
    RegistrationInputs,
    backproject_observation,
    capture_viewer_observation,
    extract_observation_from_editor,
    generate_mesh,
    register_mesh,
    register_observation,
    safe_stem,
)


EXAMPLE_REGISTRATION = json.dumps(
    {
        "world_position": [0.0, 0.0, 0.8],
        "floor_z": 0.0,
        "ceiling_z": 3.0,
        "element_width_m": 0.8,
        "element_height_m": 1.0,
        "camera_eye": [2.5, -3.0, 1.8],
        "camera_target": [0.0, 0.0, 0.8],
        "camera_fov_deg": 45.0,
        "image_size": [800, 800],
        "up_axis": 2,
        "bbox": [0.5, 0.5, 0.35, 0.55],
    },
    indent=2,
    ensure_ascii=False,
)


def _client() -> TrellisClient:
    config = load_config().trellis
    return TrellisClient(host=config.host, port=config.port, timeout=config.timeout)


def _health() -> dict[str, Any]:
    client = _client()
    config = load_config().trellis
    ok = client.health()
    return {
        "status": "ok" if ok else "unavailable",
        "service": f"{config.host}:{config.port}",
        "message": "TRELLIS model ready" if ok else "请先启动 scripts\\launch_trellis_server.bat",
    }


def _capture(scene_name: str, viewer_session: dict[str, Any] | None):
    if not scene_name:
        return None, None, "请先填写场景名称。"
    if not viewer_session:
        return None, None, "请先启动主 3DGS 查看器并在其中选定视角。"
    out_dir = ROOT / "output" / scene_name / "_trellis_registration" / f"capture_{safe_stem(scene_name)}"
    try:
        return capture_viewer_observation(scene_name, viewer_session, out_dir)
    except Exception as exc:
        return None, None, f"捕获失败：{exc}"


def _segment(mask_editor_value, render_state):
    cfg = load_config()
    try:
        from bim_recon.gradio_helpers import _get_falcon
        from bim_recon.vlm_verifier import query_vlm
        falcon = _get_falcon()
        def vlm_caller(image_path: str, prompt: str) -> str:
            return query_vlm(image_path, prompt, cfg.vlm.api_base, cfg.vlm.model, cfg.vlm.api_key, timeout=60)
        out_dir = ROOT / "output" / (render_state or {}).get("scene", "_unknown") / "_trellis_registration"
        return extract_observation_from_editor(mask_editor_value, render_state, output_dir=out_dir, vlm_caller=vlm_caller, falcon_client=falcon)
    except Exception as exc:
        return "", None, None, None, None, f"分割失败：{exc}"


def _generate_observed(cutout_path, label, render_state, seed, simplify, texture_size):
    if not cutout_path or not render_state:
        return "请先完成视角捕获和 Falcon 分割。", None, None, None, None
    try:
        scene_name = render_state.get("scene", "_unknown")
        out_dir = ROOT / "output" / scene_name / "_trellis_registration" / safe_stem(label or "object")
        result = generate_mesh(_client(), cutout_path, out_dir, name=label or "object", seed=int(seed), simplify=float(simplify), texture_size=int(texture_size))
        glb_path = str(result.glb_path)
        preview_path = str(result.preview_path) if result.preview_path else None
        return "TRELLIS GLB 已生成，可继续自动配准。", glb_path, glb_path, preview_path, {"glb": glb_path, "preview": preview_path}
    except Exception as exc:
        return f"TRELLIS 生成失败：{exc}", None, None, None, {"error": str(exc)}


def _register_observed(glb_path, cutout_path, label, render_state, detection_info, registration_name, floor_z, ceiling_z, yaw_override):
    if not glb_path or not cutout_path or not render_state or not detection_info:
        return "请先完成捕获、分割和 GLB 生成。", None, None, {"error": "incomplete_observation"}
    try:
        out_dir = ROOT / "output" / render_state.get("scene", "_unknown") / "_trellis_registration" / safe_stem(registration_name or label or "object")
        manifest = register_observation(glb_path, cutout_path, render_state, detection_info, out_dir, name=registration_name or label or "object", label=label or "object", floor_z=float(floor_z), ceiling_z=float(ceiling_z), yaw_override=None if yaw_override is None else float(yaw_override))
        score = manifest["registration"]["yaw_search"]["best_iou"]
        debug_path = out_dir / "yaw_debug" / "overlay_best.png"
        return f"配准完成：自动 yaw {manifest['registration']['resolved_yaw_degrees']:.1f}°，silhouette IoU {score:.3f}。使用主 Gradio 页面进行渲染-比较 + ICP 精修。", manifest["manifest_path"], str(debug_path) if debug_path.is_file() else None, manifest
    except Exception as exc:
        return f"配准失败：{exc}", None, None, {"error": str(exc)}


def _generate(
    image_path: str | None,
    name: str,
    seed: float,
    simplify: float,
    texture_size: float,
):
    if not image_path:
        return "请上传一个对象图片或 RGBA cutout。", None, None, None
    try:
        result = generate_mesh(
            _client(),
            image_path,
            ROOT / "output" / "_trellis_registration" / safe_stem(name),
            name=name or "trellis_object",
            seed=int(seed),
            simplify=float(simplify),
            texture_size=int(texture_size),
        )
        payload = {
            "status": "generated",
            "glb": str(result.glb_path),
            "gaussian": str(result.gaussian_path) if result.gaussian_path else None,
            "preview": str(result.preview_path) if result.preview_path else None,
            "seed": result.seed,
        }
        return "生成成功。请在下方预览 GLB，并进入自动配准。", str(result.glb_path), str(result.preview_path) if result.preview_path else None, payload
    except Exception as exc:
        return f"生成失败：{exc}", None, None, {"status": "error", "error": str(exc)}


def _parse_inputs(raw: str, cutout_path: str) -> RegistrationInputs:
    data = json.loads(raw or "{}")
    if "registration" in data and isinstance(data["registration"], dict):
        data = data["registration"].get("inputs", data["registration"])
    if not data:
        raise ValueError("请填写配准参数 JSON")

    def vector(name: str, length: int) -> tuple[float, ...]:
        values = data.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != length:
            raise ValueError(f"{name} 必须是长度为 {length} 的数组")
        return tuple(float(value) for value in values)

    image_size_raw = data.get("image_size")
    if image_size_raw is None:
        with Image.open(cutout_path) as image:
            image_size = (image.width, image.height)
    else:
        if len(image_size_raw) != 2:
            raise ValueError("image_size 必须是 [width, height]")
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

    bbox_raw = data.get("bbox", [0.5, 0.5, 1.0, 1.0])
    if len(bbox_raw) != 4:
        raise ValueError("bbox 必须是 [center_x, center_y, width, height]")
    return RegistrationInputs(
        world_position=vector("world_position", 3),
        floor_z=float(data.get("floor_z", 0.0)),
        ceiling_z=float(data.get("ceiling_z", 3.0)),
        element_width_m=float(data["element_width_m"]),
        element_height_m=float(data["element_height_m"]),
        camera_eye=vector("camera_eye", 3),
        camera_target=vector("camera_target", 3),
        camera_fov_deg=float(data.get("camera_fov_deg", data.get("camera_fov", 45.0))),
        image_size=image_size,
        up_axis=int(data.get("up_axis", 2)),
        bbox=tuple(float(value) for value in bbox_raw),
    )


def _register(glb_path: str | None, cutout_path: str | None, raw_inputs: str, name: str):
    if not glb_path:
        return "请先生成或上传 GLB。", None, None, None
    if not cutout_path:
        return "请上传与 GLB 对应的对象 cutout。", None, None, None
    try:
        inputs = _parse_inputs(raw_inputs, cutout_path)
        output_dir = ROOT / "output" / "_trellis_registration" / safe_stem(name or Path(glb_path).stem)
        manifest = register_mesh(
            glb_path,
            cutout_path,
            output_dir,
            inputs,
            name=name or Path(glb_path).stem,
        )
        diagnostics = manifest["registration"]
        score = diagnostics["yaw_search"]["best_iou"]
        status = f"配准完成。自动 yaw = {diagnostics['yaw_search']['best_yaw']:.1f}°，silhouette IoU = {score:.3f}。"
        debug_image = output_dir / "yaw_debug" / "overlay_best.png"
        return status, manifest["manifest_path"], str(debug_image) if debug_image.is_file() else None, manifest
    except Exception as exc:
        return f"配准失败：{exc}", None, None, {"status": "error", "error": str(exc)}


def _result_dir(scene_name: str, batch_label: str) -> Path:
    """Resolve one A-class batch label inside its scene output directory."""
    timestamp = (batch_label or "").split("(", 1)[0].strip()
    if not scene_name or not timestamp:
        raise ValueError("请先选择 A 类结果批次")
    scene_dir = (ROOT / "output" / scene_name).resolve()
    result_dir = (scene_dir / timestamp).resolve()
    if result_dir.parent != scene_dir or not result_dir.is_dir():
        raise ValueError("A 类结果批次不存在")
    return result_dir


def _load_a_class_context(scene_name: str, batch_label: str):
    """Load one prior A-class result and reconstruct its original scene frame."""
    try:
        result_dir = _result_dir(scene_name, batch_label)
        walls = json.loads(
            (result_dir / "wall_lines_snapped.json").read_text(encoding="utf-8")
        )
        raw_walls = json.loads(
            (result_dir / "wall_lines.json").read_text(encoding="utf-8")
        )
        scan_info = raw_walls.get("scan_info", {})
        center = scan_info.get("center", [0.0, 0.0])
        heights = [float(value) for value in scan_info.get("heights", [])]
        elements: list[dict[str, Any]] = []
        for kind in ("door", "window"):
            path = result_dir / f"{kind}s_verified.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for result in data.get("results", []):
                if result.get("confirmed") is True:
                    elements.append(dict(result.get("candidate", {})))
        context = {
            "result_dir": str(result_dir),
            "center_offset": [float(center[0]), float(center[1])],
            "walls": walls,
            "elements": elements,
        }
        floor_value = heights[0] - 0.15 if heights else None
        ceiling_value = heights[-1] + 0.15 if heights else None
        status = (
            f"已加载 `{result_dir.name}`：{len(walls)} 段墙，"
            f"{sum(e.get('element_class') == 'door' for e in elements)} 扇门，"
            f"{sum(e.get('element_class') == 'window' for e in elements)} 扇窗。"
        )
        base_radar = _draw_spatial_context(context)
        return (
            base_radar,
            base_radar.copy(),
            context,
            status,
            gr.update(value=floor_value) if floor_value is not None else gr.update(),
            gr.update(value=ceiling_value) if ceiling_value is not None else gr.update(),
        )
    except Exception as exc:
        return None, None, None, f"A 类结果加载失败：{exc}", gr.update(), gr.update()


def _refresh_a_class_results(scene_name: str):
    choices = list_available_results(scene_name)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def _preview_existing_glb(glb_path: str | None):
    """Validate an uploaded GLB before handing it to the model viewer."""
    if not glb_path:
        return None
    path = Path(glb_path)
    if not path.is_file() or path.suffix.lower() not in {".glb", ".gltf"}:
        raise gr.Error("请选择有效的 GLB 或 glTF 文件")
    return str(path.resolve())


def build_app() -> gr.Blocks:
    css = """
    .gradio-container { max-width: 1480px !important; background: #101417; }
    .hero { border: 1px solid #40504b; background: linear-gradient(135deg,#172723,#101417); padding: 30px; border-radius: 18px; }
    .hero h1 { font-family: Georgia, serif; letter-spacing: -.04em; color: #e5f3e8; }
    .eyebrow { color: #86d7ac; text-transform: uppercase; letter-spacing: .18em; font-size: 11px; }
    .hint { color: #a7b5ad; }
    """
    with gr.Blocks(title="B-class Registration Lab", css=css) as app:
        with gr.Row():
            scene_name = gr.Dropdown(
                label="场景名称",
                choices=list_available_scenes(),
                value=None,
                allow_custom_value=False,
                filterable=True,
                scale=4,
            )
            refresh_scenes_btn = gr.Button("刷新场景列表", variant="secondary", scale=1)
        with gr.Row():
            a_class_batch = gr.Dropdown(
                label="A 类墙门窗结果批次",
                choices=[],
                value=None,
                allow_custom_value=False,
                filterable=True,
                scale=4,
            )
            refresh_a_class_btn = gr.Button("刷新 A 类结果", variant="secondary", scale=1)
        a_class_status = gr.Markdown("请选择包含 A 类提取结果的批次")
        a_class_context = gr.State(None)
        viewer_session = gr.JSON(label="Viewer session（点击启动后自动写入）", value={})
        viewer_start_btn = gr.Button("启动当前场景的 3DGS 在线查看器", variant="secondary")
        viewer_panel = gr.HTML("请选择场景，然后启动查看器")
        refresh_scenes_btn.click(
            lambda: gr.update(choices=list_available_scenes()),
            outputs=scene_name,
        )
        viewer_start_btn.click(
            launch_scene_viewer,
            inputs=scene_name, outputs=[viewer_session, viewer_panel],
        )
        capture_btn = gr.Button("📸 捕获当前视角并反投影准备", variant="primary")
        capture_status = gr.Markdown("等待捕获")
        render_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 02 · VLM 指代 + Falcon 精确分割")
                mask_editor = gr.ImageMask(
                    label="3DGS 观测图（直接用画笔粗略涂抹目标）",
                    type="numpy",
                    height=520,
                    layers=False,
                    brush=gr.Brush(
                        colors=["#ff5533"],
                        default_size=18,
                        color_mode="fixed",
                    ),
                    transforms=[],
                    sources=[],
                )
                segment_btn = gr.Button("🔍 VLM 识别 + Falcon 分割", variant="secondary")
            with gr.Column(scale=1):
                segmentation_overlay = gr.Image(label="Falcon overlay", height=300)
                cutout_preview = gr.Image(label="RGBA cutout", height=300)
                label_state = gr.Textbox(label="VLM referring expression", interactive=False)
                cutout_state = gr.State(None)
                detection_state = gr.State(None)
                segmentation_status = gr.Markdown("等待分割")
        gr.Markdown("### 03 · TRELLIS mesh 生成")
        with gr.Row():
            mesh_seed = gr.Number(label="Seed", value=1, precision=0)
            mesh_simplify = gr.Slider(label="Simplify", minimum=0.5, maximum=1.0, value=0.95, step=0.05)
            mesh_texture = gr.Number(label="Texture size", value=1024, precision=0)
            mesh_btn = gr.Button("生成 TRELLIS GLB", variant="primary")
        mesh_status = gr.Markdown("等待生成")
        with gr.Row():
            generated_model = gr.Model3D(
                label="交互式 GLB 预览",
                display_mode="solid",
                clear_color=(0.055, 0.071, 0.078, 1.0),
                height=520,
                interactive=False,
                scale=2,
            )
            generated_preview = gr.Image(label="TRELLIS 静态预览", height=520, scale=1)
        generated_glb = gr.File(label="GLB 文件", interactive=True)
        mesh_json = gr.JSON(label="Mesh result")
        gr.Markdown("### 04 · 深度反投影 + 自动姿态注册")
        with gr.Row():
            registration_name = gr.Textbox(label="数据集对象 ID", value="object_001")
            floor_z = gr.Number(label="Floor Z", value=0.0)
            ceiling_z = gr.Number(label="Ceiling Z", value=3.0)
            yaw_override = gr.Number(label="手动 yaw 覆盖（留空使用 silhouette search）", value=None)
            register_btn = gr.Button("自动配准并导出 manifest", variant="primary")
        registration_status = gr.Markdown("等待配准")
        registration_manifest = gr.File(label="Registration manifest")
        registration_debug = gr.Image(label="Yaw overlay：红观测 / 蓝投影 / 绿重叠")
        registration_json = gr.JSON(label="Placement manifest")
        gr.Markdown("### 05 · 两阶段空间雷达对照")
        with gr.Row():
            observation_radar = gr.Image(
                label="分割照片反投影雷达图",
                height=620,
                interactive=False,
            )
            glb_radar = gr.Image(
                label="GLB 配准后场景雷达图",
                height=620,
                interactive=False,
            )
        scene_result_event = scene_name.change(
            _refresh_a_class_results,
            inputs=scene_name,
            outputs=a_class_batch,
        )
        scene_result_event.then(
            _load_a_class_context,
            inputs=[scene_name, a_class_batch],
            outputs=[observation_radar, glb_radar, a_class_context, a_class_status, floor_z, ceiling_z],
        )
        refresh_result_event = refresh_a_class_btn.click(
            _refresh_a_class_results,
            inputs=scene_name,
            outputs=a_class_batch,
        )
        refresh_result_event.then(
            _load_a_class_context,
            inputs=[scene_name, a_class_batch],
            outputs=[observation_radar, glb_radar, a_class_context, a_class_status, floor_z, ceiling_z],
        )
        a_class_batch.input(
            _load_a_class_context,
            inputs=[scene_name, a_class_batch],
            outputs=[observation_radar, glb_radar, a_class_context, a_class_status, floor_z, ceiling_z],
        )
        capture_btn.click(
            _capture,
            inputs=[scene_name, viewer_session],
            outputs=[mask_editor, render_state, capture_status],
        )
        segment_event = segment_btn.click(_segment, inputs=[mask_editor, render_state], outputs=[label_state, segmentation_overlay, cutout_preview, cutout_state, detection_state, segmentation_status])
        segment_event.then(
            _observation_radar,
            inputs=[render_state, detection_state, a_class_context],
            outputs=observation_radar,
        )
        mesh_btn.click(
            _generate_observed,
            inputs=[cutout_state, label_state, render_state, mesh_seed, mesh_simplify, mesh_texture],
            outputs=[mesh_status, generated_model, generated_glb, generated_preview, mesh_json],
        )
        generated_glb.upload(
            _preview_existing_glb,
            inputs=generated_glb,
            outputs=generated_model,
        )
        registration_event = register_btn.click(
            _register_observed,
            inputs=[generated_glb, cutout_state, label_state, render_state, detection_state, registration_name, floor_z, ceiling_z, yaw_override],
            outputs=[registration_status, registration_manifest, registration_debug, registration_json],
        )
        registration_event.then(
            _registration_radar,
            inputs=[registration_json, a_class_context],
            outputs=glb_radar,
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=19256)
