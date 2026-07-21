"""Focused Gradio workspace for TRELLIS generation and automatic registration.

Launch with ``scripts/launch_trellis_registration.bat``. This deliberately
keeps the main BIM pipeline untouched: it generates GLB assets and solves the
first registration pass through the project's existing silhouette-search mesh
registrar. Blender annotation export remains a later dataset step.
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
from PIL import Image

from bim_recon.config import load_config
from bim_recon.trellis_client import TrellisClient
from bim_recon.trellis_registration import (
    RegistrationInputs,
    generate_mesh,
    register_mesh,
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


def build_app() -> gr.Blocks:
    css = """
    .gradio-container { max-width: 1420px !important; background: #111417; }
    .hero { border: 1px solid #39434a; background: linear-gradient(135deg,#192328,#101315); padding: 28px; border-radius: 16px; }
    .hero h1 { font-family: Georgia, serif; letter-spacing: -.03em; color: #e6f1ea; }
    .eyebrow { color: #86d7ac; text-transform: uppercase; letter-spacing: .16em; font-size: 12px; }
    .hint { color: #9ca9a3; }
    """
    with gr.Blocks(title="TRELLIS Registration Lab", css=css) as app:
        gr.HTML(
            '<div class="hero"><div class="eyebrow">BIM / POSE LAB</div>'
            '<h1>TRELLIS Registration Lab</h1>'
            '<p class="hint">独立的 GLB 生成与自动配准工作台。这里不运行主 BIM 管线，也不直接修改 Revit。</p></div>'
        )
        with gr.Row():
            health_btn = gr.Button("检查 TRELLIS 服务", variant="secondary")
            health_json = gr.JSON(label="服务状态", value=_health)
        with gr.Tabs():
            with gr.Tab("01 · 生成 GLB"):
                gr.Markdown(
                    "上传对象图或透明背景 cutout。TRELLIS 只负责 image → GLB/PLY；"
                    "生成结果会保存到 `output/_trellis_registration/`。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        source_image = gr.Image(label="输入图像", type="filepath", sources=["upload"])
                        object_name = gr.Textbox(label="对象名称", value="chair_view_001")
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=1, precision=0)
                            simplify = gr.Slider(label="Simplify", minimum=0.5, maximum=1.0, value=0.95, step=0.05)
                        texture = gr.Number(label="Texture size", value=1024, precision=0)
                        generate_btn = gr.Button("生成 TRELLIS GLB", variant="primary")
                    with gr.Column(scale=1):
                        generation_status = gr.Markdown("等待生成")
                        generated_glb = gr.File(label="生成的 GLB")
                        generated_preview = gr.Image(label="TRELLIS Preview")
                        generation_json = gr.JSON(label="生成结果")
            with gr.Tab("02 · 自动配准"):
                gr.Markdown(
                    "上传 GLB 和同一观测图得到的 cutout，填写一次相机/尺寸 JSON。"
                    "系统会复用现有 `mesh_registrar.find_best_yaw_silhouette()`，"
                    "通过投影分析搜索最匹配的 yaw，并输出可审计 manifest。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        registration_glb = gr.File(label="TRELLIS GLB", file_types=[".glb"], type="filepath")
                        registration_cutout = gr.Image(label="对象 cutout（PNG）", type="filepath", sources=["upload"])
                        registration_name = gr.Textbox(label="配准名称", value="chair_view_001")
                        registration_json = gr.Code(
                            label="配准参数 JSON",
                            language="json",
                            value=EXAMPLE_REGISTRATION,
                            lines=24,
                        )
                        register_btn = gr.Button("自动配准（Silhouette Search）", variant="primary")
                    with gr.Column(scale=1):
                        registration_status = gr.Markdown("等待配准")
                        registration_manifest = gr.File(label="Registration manifest")
                        registration_debug = gr.Image(label="Yaw overlay（红=观测，蓝=投影，绿=重叠）")
                        registration_result = gr.JSON(label="配准诊断")
        health_btn.click(_health, outputs=health_json)
        generate_btn.click(
            _generate,
            inputs=[source_image, object_name, seed, simplify, texture],
            outputs=[generation_status, generated_glb, generated_preview, generation_json],
        )
        register_btn.click(
            _register,
            inputs=[registration_glb, registration_cutout, registration_json, registration_name],
            outputs=[registration_status, registration_manifest, registration_debug, registration_result],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=19256)
