"""Gradio callback functions and helpers — separated from UI layout.

All business logic for the Gradio UI lives here. The UI layout
(component definitions, layout, event wiring) stays in
``scripts/gradio_app.py``.
"""
from __future__ import annotations

import json
import atexit
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings('ignore', message='.*HTTP_422_UNPROCESSABLE_ENTITY.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gradio.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='starlette.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    force=True,
)
logger = logging.getLogger('gradio_app')

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.pipeline_api import (
    PipelineResults, load_results, remap_from_json, mask_to_bbox,
)
from bim_recon.config import load_config, get_llm_model, save_config, test_llm_connection
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest

VIEWER_PORT = 8081
CAMERA_PORT = 8082
MAX_PORT_WAIT_S = 30
SCENESPLAT = ROOT / "SceneSplat"
MAX_CONSOLE_LINES = 200

_SCENE_CACHE: dict[str, Any] = {}
_FALCON_CACHE: Any = None


# ---------------------------------------------------------------------------
# PLY 验证
# ---------------------------------------------------------------------------

def validate_ply(ply_path: str) -> dict:
    """检查 PLY 文件是否为标准 3D Gaussian Splatting 格式。"""
    if ply_path is None:
        return {"错误": "未选择文件"}
    path = Path(ply_path)
    if not path.exists():
        return {"错误": f"文件不存在: {ply_path}"}
    if path.suffix.lower() != ".ply":
        return {"错误": "文件必须为 .ply 格式"}
    try:
        with open(path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                line = f.readline()
                if not line:
                    break
                header += line
        header_str = header.decode("ascii", errors="ignore")
        properties: list[str] = []
        num_points = 0
        for line in header_str.split("\n"):
            line = line.strip()
            if line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif line.startswith("element vertex "):
                num_points = int(line.split()[-1])
        required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                    "rot_0", "rot_1", "rot_2", "rot_3"]
        missing = [r for r in required if r not in properties]
        if missing:
            return {"有效": False, "错误": f"缺少高斯属性: {missing}",
                    "已有属性": properties, "高斯数": num_points}
        has_sh = any(p.startswith("f_dc_") for p in properties)
        return {"有效": True, "高斯数": num_points,
                "属性": properties, "含SH系数": has_sh}
    except Exception as e:
        return {"错误": f"读取PLY失败: {e}"}


# ---------------------------------------------------------------------------
# 场景管理
# ---------------------------------------------------------------------------

def check_preprocess_status(scene_name: str) -> dict:
    preprocessed = ROOT / "data" / scene_name / "preprocessed"
    feat_pt = ROOT / "output" / scene_name / f"{scene_name}_feat.pt"
    return {
        "preprocessed_exists": preprocessed.exists() and any(preprocessed.iterdir()),
        "feat_pt_exists": feat_pt.exists(),
    }


def list_available_scenes() -> list[str]:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return []
    return [d.name for d in sorted(data_dir.iterdir())
            if d.is_dir() and any(d.glob("*.ply"))]


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def _wait_for_port(port: int, timeout: float = MAX_PORT_WAIT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


_CHILD_PROCS: list = []

def _kill_child_procs():
    """Kill all tracked child processes (viewer, pipeline, etc.)."""
    for p in _CHILD_PROCS:
        if p.poll() is None:  # still running
            try:
                p.kill()
            except Exception:
                pass
    _CHILD_PROCS.clear()

atexit.register(_kill_child_procs)


def start_viewer(scene_name: str) -> str:
    if not scene_name:
        return '<p style="color:red">请先选择场景</p>'
    input_root = str(ROOT / "data" / scene_name / "preprocessed")
    feat_path = str(ROOT / "output" / scene_name / f"{scene_name}_feat.pt")
    if not Path(input_root).exists():
        return f'<p style="color:red">预处理数据不存在: {input_root}</p>'
    if not Path(feat_path).exists():
        return f'<p style="color:red">feat.pt 不存在: {feat_path}</p>'
    subprocess.run(
        f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{VIEWER_PORT} ^| findstr LISTENING\') do taskkill /f /pid %a',
        shell=True, capture_output=True,
    )
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_viewer.py"),
         "--input-root", input_root, "--feature-path", feat_path,
         "--port", str(VIEWER_PORT)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    _CHILD_PROCS.append(proc)
    if not _wait_for_port(VIEWER_PORT):
        return f'<p style="color:red">查看器启动失败 (端口 {VIEWER_PORT})</p>'
    return (f'<iframe src="http://127.0.0.1:{VIEWER_PORT}" '
            f'style="width:100%;height:600px;border:none;"></iframe>')


# ---------------------------------------------------------------------------
# 管线运行（流式输出）
# ---------------------------------------------------------------------------

def find_latest_output(scene_name: str) -> str:
    base = ROOT / "output" / scene_name
    if not base.exists():
        return ""
    dirs = [d for d in base.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(dirs[0]) if dirs else ""


def list_available_results(scene_name: str) -> list[str]:
    """列出 output/<scene>/ 下所有含 pipeline_report.json 的时间戳结果目录。"""
    if not scene_name:
        return []
    base = ROOT / "output" / scene_name
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir()
            if d.is_dir() and (d / "pipeline_report.json").exists()]
    dirs.sort(key=lambda d: d.name, reverse=True)  # 最新在前
    return [f"{d.name}  ({d.stat().st_size // 1024} KB)" for d in dirs]


def _result_dir_from_label(scene_name: str, label: str) -> str:
    """从 Dropdown 标签解析出完整路径。标签格式: '20260703_155545  (123 KB)'"""
    if not label or not scene_name:
        return ""
    timestamp = label.split("(")[0].strip()
    return str(ROOT / "output" / scene_name / timestamp)


def _prepare_results(res: PipelineResults):
    """从 PipelineResults 准备 UI 输出元组。

    返回顺序: (results, summary, vlm_imgs, radar_imgs, report,
               vlm_cb_update, elem_dd_update)
    """
    logger.info("=" * 60)
    logger.info("开始准备结果展示")
    
    # Radar gallery: wall top-down + per-element radar plots
    radar_imgs = []
    if res.wall_topdown_image and Path(res.wall_topdown_image).exists():
        radar_imgs.append((res.wall_topdown_image, "墙线俯视图"))
        logger.info(f"添加墙线俯视图: {res.wall_topdown_image}")
    else:
        logger.warning(f"墙线俯视图不存在: {res.wall_topdown_image}")
    
    # Scan output dir for radar_*.png
    if res.wall_topdown_image:
        out_dir = Path(res.wall_topdown_image).parent
        radar_files = sorted(out_dir.glob("radar_*.png"))
        logger.info(f"扫描雷达图库目录: {out_dir}, 找到 {len(radar_files)} 个文件")
        for r in radar_files:
            elem_name = r.stem.replace("radar_", "")
            label_map = {
                "ring_raw": "环形检测 (合并前)",
                "merged": "合并构件 (合并后)",
                "door": "门 雷达图",
                "window": "窗 雷达图",
            }
            label = label_map.get(elem_name, f"{elem_name} 雷达图")
            radar_imgs.append((str(r), label))
            logger.info(f"添加雷达图: {r.name}")

    # VLM gallery
    logger.info(f"处理 VLM 图库: 共 {len(res.doors)} 扇门, {len(res.windows)} 扇窗")
    vlm_imgs = []
    for e in (res.doors + res.windows):
        logger.info(f"检查构件: {e.element_class} #{e.result_index}, image_path={e.image_path}")
        if e.image_path and Path(e.image_path).exists():
            # 三态: ✓ confirmed / ✗ rejected / ⚠️ VLM error
            status = '✓' if e.confirmed is True else ('✗' if e.confirmed is False else '⚠️')
            vlm_imgs.append((e.image_path, f"{e.element_class} #{e.result_index} ({status})"))
            logger.info(f"  ✓ 添加 VLM 图: {e.image_path}")
        else:
            logger.warning(f"  ✗ VLM 图不存在: {e.image_path}")

    
    summary = (f"### 结果\n- 墙体: {len(res.walls)}\n"
               f"- 门: {sum(1 for d in res.doors if d.confirmed)}/{len(res.doors)} 已确认\n"
               f"- 窗: {sum(1 for w in res.windows if w.confirmed)}/{len(res.windows)} 已确认")
    all_elems = res.doors + res.windows
    choices = [f"{e.element_class} #{e.result_index}" for e in all_elems]
    defaults = [f"{e.element_class} #{e.result_index}" for e in all_elems if e.confirmed]
    
    logger.info(f"最终统计: radar_imgs={len(radar_imgs)}, vlm_imgs={len(vlm_imgs)}")
    logger.info("=" * 60)

    print(f"[DEBUG] gallery counts: radar={len(radar_imgs)} vlm={len(vlm_imgs)}")

    return (res, summary, vlm_imgs, radar_imgs, res.report,
            gr.update(choices=choices, value=defaults),
            gr.update(choices=choices))


def run_pipeline_direct(scene: str, doors: bool, windows: bool,
                        falcon: bool, skip_vlm: bool):
    """Run pipeline directly (no subprocess). Yields 9 values for Gradio.

    yield: (console, out_dir, results, summary, radar_gallery,
            vlm_gallery, report, vlm_cb, elem_dd)
    """
    if not scene:
        yield ("Error: no scene selected", "", None, "", None, [], None,
               gr.update(), gr.update())
        return
    elems = []
    if doors:
        elems.append("door")
    if windows:
        elems.append("window")
    if not elems:
        yield ("Error: select at least one element type", "", None, "",
               None, [], None, gr.update(), gr.update())
        return

    from bim_recon.pipeline_runner import PipelineConfig, run_pipeline
    app_config = load_config()
    vlm = app_config.vlm

    pipe_config = PipelineConfig(
        name=scene,
        elements=elems,
        skip_vlm=skip_vlm,
        vlm_api_base=vlm.api_base,
        vlm_model=vlm.model,
        vlm_api_key=vlm.api_key,
    )

    console_lines: list[str] = []
    final_data = {}

    for msg, data in run_pipeline(pipe_config):
        console_lines.append(msg)
        console = "\n".join(console_lines[-MAX_CONSOLE_LINES:])
        if msg.startswith("ERROR"):
            yield (f"{console}\n\nPipeline failed.", "", None, "Failed",
                   None, [], None, gr.update(), gr.update())
            return
        yield (console, "", None, "Running...", None, [], None,
               gr.update(), gr.update())
        final_data = data

    # Pipeline complete — load results
    out_dir = final_data.get("out_dir", "")
    if not out_dir:
        yield (f"Pipeline finished but no output directory.\n" +
               "\n".join(console_lines[-10:]), "", None, "Failed",
               None, [], None, gr.update(), gr.update())
        return

    console = "\n".join(console_lines[-MAX_CONSOLE_LINES:]) + f"\nDone: {out_dir}"
    logger.info(f"Pipeline complete: {out_dir}")

    try:
        res = load_results(Path(out_dir))
        logger.info(f"Results loaded: {len(res.doors)} doors, {len(res.windows)} windows")
    except Exception as e:
        logger.error(f"Failed to load results: {e}", exc_info=True)
        yield (f"{console}\nError loading results: {e}", out_dir, None,
               "Error", None, [], None, gr.update(), gr.update())
        return

    r = _prepare_results(res)
    # r = (res, summary, vlm_imgs, radar_imgs, report, vlm_cb, elem_dd)
    yield (console, out_dir, r[0], r[1], r[3], r[2], r[4], r[5], r[6])


def load_results_cb(out_dir: str):
    """从输出目录加载已有结果。返回 7 个值。"""
    logger.info(f"加载结果回调: out_dir={out_dir}")
    if not out_dir or not Path(out_dir).exists():
        logger.warning(f"输出目录不存在: {out_dir}")
        return None, "无结果", [], [], None, gr.update(choices=[]), gr.update(choices=[])
    try:
        logger.info(f"开始加载结果: {out_dir}")
        res = load_results(Path(out_dir))
        logger.info(f"结果加载成功: {len(res.doors)} 扇门, {len(res.windows)} 扇窗, {len(res.walls)} 面墙")
    except Exception as e:
        logger.error(f"加载结果失败: {e}", exc_info=True)
        return None, f"❌ 加载失败: {e}", [], [], None, gr.update(choices=[]), gr.update(choices=[])
    logger.info("开始准备结果展示")
    result = _prepare_results(res)
    logger.info(f"结果准备完成，返回给 UI")
    return result


# ---------------------------------------------------------------------------
# 微调：bbox 可视化
# ---------------------------------------------------------------------------

def _find_element(results: PipelineResults, elem_label: str):
    if not elem_label or "#" not in elem_label:
        return None
    parts = elem_label.split("#")
    elem_class = parts[0].strip()
    try:
        target_idx = int(parts[-1].split()[0])
    except ValueError:
        return None
    for e in results.elements:
        if e.result_index == target_idx and e.element_class == elem_class:
            return e
    return None

# ---------------------------------------------------------------------------
# Interactive radar: checkbox → live radar update
# ---------------------------------------------------------------------------

def update_interactive_radar(checked_items, results):
    """Regenerate radar plot showing only checked elements.

    Triggered when the CheckboxGroup selection changes.  Draws walls +
    camera + markers for each checked element, hides unchecked ones.
    """
    if not results or not results.walls:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math as _m

    # Get center from report
    report = results.report or {}
    coord_sys = report.get("coordinate_system", {})
    center = coord_sys.get("center", [0, 0])
    cx, cy = float(center[0]), float(center[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw walls
    for wl in results.walls:
        x1, y1 = wl["x1"] - cx, wl["y1"] - cy
        x2, y2 = wl["x2"] - cx, wl["y2"] - cy
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)

    # Camera position
    ax.plot(0, 0, "k^", markersize=14, zorder=10, label="Camera")

    # Draw checked elements
    label_colors = {"door": "#e74c3c", "window": "#3498db", "column": "#95a5a6"}
    for e in results.elements:
        label = f"{e.element_class} #{e.result_index}"
        if label not in checked_items:
            continue
        dx = e.world_x - cx
        dy = e.world_y - cy
        color = label_colors.get(e.element_class, "#2ecc71")
        marker = "*" if e.element_class == "door" else "D"
        ax.scatter(dx, dy, c=color, s=250, marker=marker, zorder=8,
                   edgecolors="black", linewidths=1.5)
        # Draw a circle to show approximate extent
        hd = e.height_detection or {}
        width = hd.get("width_m", 0.5)
        circle = plt.Circle((dx, dy), width / 2, fill=False,
                             color=color, linewidth=2, linestyle="--", zorder=7)
        ax.add_patch(circle)
        ax.annotate(f"{e.element_class}#{e.result_index}\n"
                    f"r={_m.hypot(dx, dy):.1f}m w={width:.2f}m",
                    (dx, dy), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 12), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor=color))

    # Draw unchecked elements as faint outlines
    for e in results.elements:
        label = f"{e.element_class} #{e.result_index}"
        if label in checked_items:
            continue
        dx = e.world_x - cx
        dy = e.world_y - cy
        ax.scatter(dx, dy, c="lightgray", s=80, marker="x", zorder=6, alpha=0.4)

    ax.set_aspect("equal")
    max_r = max(
        max(abs(wl["x1"] - cx) for wl in results.walls) if results.walls else 5,
        max(abs(wl["x2"] - cx) for wl in results.walls) if results.walls else 5,
        5,
    )
    ax.set_xlim(-max_r - 1, max_r + 1)
    ax.set_ylim(-max_r - 1, max_r + 1)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    checked_count = len(checked_items)
    total_count = len(results.elements)
    ax.set_title(f"Interactive Radar - {checked_count}/{total_count} visible",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    fig.canvas.draw()
    # Modern matplotlib: use buffer_rgba instead of deprecated tostring_rgb
    buf = fig.canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # drop alpha
    plt.close(fig)
    return img_array



def draw_bbox_on_image(image_path: str, cx: float, cy: float,
                       w: float, h: float) -> np.ndarray | None:
    """在立面图上绘制归一化 bbox 叠加，返回 numpy 数组。"""
    if not image_path or not Path(image_path).exists():
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size
        x1 = int((cx - w / 2) * iw)
        y1 = int((cy - h / 2) * ih)
        x2 = int((cx + w / 2) * iw)
        y2 = int((cy + h / 2) * ih)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        cmx, cmy = int(cx * iw), int(cy * ih)
        draw.line([(cmx - 10, cmy), (cmx + 10, cmy)], fill="red", width=2)
        draw.line([(cmx, cmy - 10), (cmx, cmy + 10)], fill="red", width=2)
        return np.array(img)
    except Exception:
        return None


def on_element_select(elem_label: str, results: PipelineResults | None):
    """选择构件时，加载其立面图到 ImageMask 编辑器。"""
    if results is None or not elem_label:
        return None
    elem = _find_element(results, elem_label)
    if elem is None or not elem.elevation_image:
        return None
    return elem.elevation_image


def on_mask_apply(editor_value: dict | None, elem_label: str,
                  results: PipelineResults | None) -> dict:
    """用户画完 mask 后，提取 alpha 通道 → 紧致 bbox → 墙局部坐标。"""
    if results is None or not elem_label:
        return {"提示": "请先加载结果并选择构件"}
    elem = _find_element(results, elem_label)
    if elem is None:
        return {"错误": f"未找到构件: {elem_label}"}
    layers = (editor_value or {}).get("layers", [])
    if not layers:
        return {"错误": "未检测到绘制内容 — 请在立面图上画出门窗区域"}
    mask_rgba = layers[0]
    if not isinstance(mask_rgba, np.ndarray) or mask_rgba.ndim < 3:
        return {"错误": f"Mask 格式异常: {type(mask_rgba)}"}
    hd = elem.height_detection or {}
    elev_params = hd.get("elevation_params")
    if not elev_params:
        return {"错误": "结果中无 elevation_params，请用更新后的管线重新运行"}
    coords = results.report.get("coords", {}) if results.report else {}
    floor_z = coords.get("floor_z")
    ceiling_z = coords.get("ceiling_z")
    if floor_z is None or ceiling_z is None:
        return {"错误": "报告中缺少 floor_z/ceiling_z"}
    result = mask_to_bbox(mask_rgba, elev_params, floor_z, ceiling_z)
    if "error" in result:
        return {"错误": result["error"]}
    return {
        "原方法": hd.get("method", "N/A"),
        "原窗台(m)": round(hd.get("sill_height", 0), 3),
        "原窗顶(m)": round(hd.get("header_height", 0), 3),
        "原宽度(m)": round(hd.get("width_m", 0), 3),
        "重算窗台(m)": round(result["sill_height"], 3),
        "重算窗顶(m)": round(result["header_height"], 3),
        "重算宽度(m)": round(result["width_m"], 3),
        "重算高度(m)": round(result["element_height"], 3),
    }


def fetch_camera_state() -> tuple[str, dict]:
    """从 nerfview HTTP 端点 (端口 8082) 获取当前相机参数。返回 (显示文本, 原始数据)。"""
    import urllib.request
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8082/camera-state", timeout=2,
        ) as resp:
            data = json.loads(resp.read())
        if "error" in data:
            return f"⚠️ {data['error']}", {}
        pos = data["position"]
        look = data["look_at"]
        return (
            f"✅ **相机位置**: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n\n"
            f"**看向**: ({look[0]:.3f}, {look[1]:.3f}, {look[2]:.3f})\n\n"
            f"**FOV**: {data['fov_degrees']:.1f}°\n\n"
            f"**Aspect**: {data['aspect']:.3f}",
            data,
        )
    except Exception as e:
        return f"⚠️ 无法连接查看器 (端口 8082)\n\n请先启动下方查看器。\n\n错误: {e}", {}


# ---------------------------------------------------------------------------
# 以自定义视角重新分割 (render → Falcon → 射线平面交点 → 墙坐标)
# ---------------------------------------------------------------------------

def _get_scene(scene_name: str):
    """Lazy-load and cache a GSScene for rendering. Evicts old scene to free GPU memory."""
    if scene_name in _SCENE_CACHE:
        return _SCENE_CACHE[scene_name]
    # Free previous scene's GPU memory before loading a new one
    _SCENE_CACHE.clear()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    from bim_recon.gs_scene import GSScene
    data_dir = ROOT / "data" / scene_name
    ply_candidates = list(data_dir.glob("point_cloud_*.ply"))
    if not ply_candidates:
        ply_candidates = list(data_dir.glob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"找不到 PLY: {data_dir}")
    scene = GSScene.from_ply(str(ply_candidates[0]))
    _SCENE_CACHE[scene_name] = scene
    return scene


def _get_falcon():
    """Lazy-connect to Falcon server (port 8390)."""
    global _FALCON_CACHE
    if _FALCON_CACHE is not None:
        return _FALCON_CACHE
    from bim_recon.falcon_client import FalconClient
    falcon = FalconClient("127.0.0.1", 8390)
    if not falcon.health():
        return None
    _FALCON_CACHE = falcon
    return falcon


def _mask_bbox_to_wall_coords(
    mask_bbox: dict,
    eye: np.ndarray,
    target: np.ndarray,
    up_world: np.ndarray,
    fov_horizontal_deg: float,
    img_w: int,
    img_h: int,
    wall_start: np.ndarray,
    wall_dir: np.ndarray,
    up_axis: int,
) -> dict | None:
    """Map Falcon mask_bbox (normalized) to wall-local metres via ray-plane intersection.

    Casts rays from camera through the 4 corners + edge midpoints of the
    mask_bbox rectangle, intersects each with the wall plane, and returns
    the tightest wall-local bounding box.
    """
    # Camera basis vectors (OpenCV: +Z forward, +X right, +Y down)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(forward, right)

    focal = 0.5 * img_w / np.tan(0.5 * np.radians(fov_horizontal_deg))

    # Wall plane normal (perpendicular to wall, pointing toward camera)
    wall_normal = np.cross(wall_dir, up_world)
    wall_normal = wall_normal / (np.linalg.norm(wall_normal) + 1e-12)
    if np.dot(wall_normal, eye - wall_start) < 0:
        wall_normal = -wall_normal

    # Sample 8 points along mask_bbox perimeter
    cx, cy = mask_bbox["x"], mask_bbox["y"]
    bw, bh = mask_bbox["w"], mask_bbox["h"]
    half_l = cx - bw / 2
    half_r = cx + bw / 2
    half_t = cy - bh / 2
    half_b = cy + bh / 2
    sample_norm = [
        (half_l, half_t), (cx, half_t), (half_r, half_t),
        (half_l, cy),                     (half_r, cy),
        (half_l, half_b), (cx, half_b), (half_r, half_b),
    ]

    alongs, heights = [], []
    for nx, ny in sample_norm:
        px, py = nx * img_w, ny * img_h
        dx = (px - img_w / 2) / focal
        dy = (py - img_h / 2) / focal
        ray = forward + dx * right + dy * down
        ray = ray / (np.linalg.norm(ray) + 1e-12)

        denom = float(np.dot(ray, wall_normal))
        if abs(denom) < 1e-6:
            continue
        t = float(np.dot(wall_start - eye, wall_normal)) / denom
        if t < 0:
            continue

        point = eye + t * ray
        rel = point - wall_start
        along = float(np.dot(rel, wall_dir))
        height = float(point[up_axis])
        alongs.append(along)
        heights.append(height)

    if not alongs:
        return None
    return {
        "along_min": min(alongs),
        "along_max": max(alongs),
        "height_min": min(heights),
        "height_max": max(heights),
    }


def resegment_from_viewpoint(
    camera_data: dict,
    elem_label: str,
    results: PipelineResults | None,
    scene_name: str,
) -> tuple:
    """从捕获的视角渲染 → Falcon 分割 → 射线交点 → 墙坐标.

    Returns: (rendered_image_np, overlay_image_np, dimensions_json, status_str)
    """
    if not camera_data or "position" not in camera_data:
        return None, None, {}, "⚠️ 请先点击「📸 获取视角」捕获相机参数"
    if results is None or not elem_label:
        return None, None, {}, "⚠️ 请先加载结果并选择构件"

    elem = _find_element(results, elem_label)
    if elem is None:
        return None, None, {}, f"⚠️ 未找到构件: {elem_label}"

    hd = elem.height_detection or {}
    elev_params = hd.get("elevation_params")
    if not elev_params:
        return None, None, {}, "⚠️ 结果中无 elevation_params"

    coords = results.report.get("coords", {}) if results.report else {}
    floor_z = coords.get("floor_z", 0.0)
    ceiling_z = coords.get("ceiling_z", 3.0)
    up_axis = coords.get("up_axis", 2)

    # Parse camera params
    eye = np.array(camera_data["position"], dtype=np.float32)
    target = np.array(camera_data["look_at"], dtype=np.float32)
    up_world = np.zeros(3, dtype=np.float32)
    up_world[up_axis] = 1.0
    # viser fov is vertical (radians) → convert to horizontal degrees
    vfov_rad = camera_data["fov"]
    aspect = camera_data.get("aspect", 1.5)
    hfov_rad = 2.0 * np.arctan(np.tan(vfov_rad / 2.0) * aspect)
    hfov_deg = float(np.degrees(hfov_rad))

    img_w, img_h = 800, 600

    # 1. Load scene & render
    try:
        scene = _get_scene(scene_name)
    except Exception as e:
        return None, None, {}, f"⚠️ 加载场景失败: {e}"

    from bim_recon.gs_scene import look_at_pose
    pose = look_at_pose(
        (float(eye[0]), float(eye[1]), float(eye[2])),
        (float(target[0]), float(target[1]), float(target[2])),
        (float(up_world[0]), float(up_world[1]), float(up_world[2])),
    )
    render = scene.render(pose, width=img_w, height=img_h, fov_degrees=hfov_deg)
    rgb_uint8 = (np.clip(render.colors, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    rendered_pil = Image.fromarray(rgb_uint8)

    # 2. Falcon segmentation
    falcon = _get_falcon()
    if falcon is None:
        return rgb_uint8, None, {}, "⚠️ Falcon 服务器未运行 (端口 8390)，仅返回渲染图"

    detections = falcon.segment(rendered_pil, elem.element_class, task="segmentation")
    if not detections:
        return rgb_uint8, None, {}, f"⚠️ Falcon 未检测到 {elem.element_class}"

    # Pick best detection (largest mask area)
    best = max(
        detections,
        key=lambda d: d.mask_area_ratio if d.mask_area_ratio else d.bbox.get("w", 0) * d.bbox.get("h", 0),
    )
    norm_bbox = best.mask_bbox if best.mask_bbox else best.bbox

    # 3. Ray-plane intersection → wall coords
    # elevation_params stores wall_start/wall_dir as 2D (floor plane XY);
    # extend to 3D by inserting floor_z at the up-axis position.
    ws_raw = elev_params["wall_start"]
    wd_raw = elev_params["wall_dir"]
    wall_start = np.zeros(3, dtype=np.float32)
    wall_dir = np.zeros(3, dtype=np.float32)
    axes_2d = [i for i in range(3) if i != up_axis]
    for idx, val in enumerate(ws_raw):
        wall_start[axes_2d[idx]] = val
    for idx, val in enumerate(wd_raw):
        wall_dir[axes_2d[idx]] = val
    wall_start[up_axis] = floor_z
    wall_dir = wall_dir / (np.linalg.norm(wall_dir) + 1e-12)

    wall_coords = _mask_bbox_to_wall_coords(
        norm_bbox, eye, target, up_world, hfov_deg,
        img_w, img_h, wall_start, wall_dir, up_axis,
    )
    if wall_coords is None:
        return rgb_uint8, None, {}, "⚠️ 射线无法与墙面相交（视角可能偏离墙面太远）"

    # 4. Draw overlay on rendered image
    overlay = rgb_uint8.copy()
    from PIL import ImageDraw
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)
    bx, by, bw, bh = norm_bbox["x"], norm_bbox["y"], norm_bbox["w"], norm_bbox["h"]
    x1 = int((bx - bw / 2) * img_w)
    y1 = int((by - bh / 2) * img_h)
    x2 = int((bx + bw / 2) * img_w)
    y2 = int((by + bh / 2) * img_h)
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    overlay = np.array(overlay_pil)

    sill = wall_coords["height_min"] - floor_z
    header = wall_coords["height_max"] - floor_z
    width_m = wall_coords["along_max"] - wall_coords["along_min"]
    elem_h = header - sill

    # 5. Save rendered + overlay to disk (overwrite old elevation files)
    new_elev_path = None
    if elem.elevation_image:
        out_dir = Path(elem.elevation_image).parent
        new_elev_path = str(out_dir / f"{elem.element_class}_{elem.result_index}_elevation.png")
        new_overlay_path = str(out_dir / f"{elem.element_class}_{elem.result_index}_elevation_overlay.png")
        Image.fromarray(rgb_uint8).save(new_elev_path)
        Image.fromarray(overlay).save(new_overlay_path)
    else:
        new_overlay_path = None

    # 6. Update ElementResult — replace dataclass with updated copy
    from dataclasses import replace as dc_replace
    updated_hd = dict(hd) if hd else {}
    updated_hd["sill_height"] = round(sill, 3)
    updated_hd["header_height"] = round(header, 3)
    updated_hd["width_m"] = round(width_m, 3)
    updated_hd["element_height"] = round(elem_h, 3)
    updated_hd["method"] = "falcon_resegment"
    new_elem = dc_replace(
        elem,
        height_detection=updated_hd,
        elevation_image=new_elev_path or elem.elevation_image,
        overlay_image=new_overlay_path or elem.overlay_image,
    )
    # Replace the matched element in the canonical elements list.
    # (doors/windows are read-only filtered views of results.elements.)
    for idx, e in enumerate(results.elements):
        if e.result_index == elem.result_index and e.element_class == elem.element_class:
            results.elements[idx] = new_elem
            break

    dims = {
        "视角位置": [round(float(v), 3) for v in eye],
        "看向": [round(float(v), 3) for v in target],
        "水平FOV": round(hfov_deg, 1),
        "重算窗台(m)": round(sill, 3),
        "重算窗顶(m)": round(header, 3),
        "重算宽度(m)": round(width_m, 3),
        "重算高度(m)": round(elem_h, 3),
        "方法": "falcon_resegment",
    }


    status = f"✅ Falcon 检测到 {elem.element_class}，已从新视角重算尺寸并更新"

    return (
        overlay,           # reseg_preview
        dims,              # reseg_out
        status,            # cam_status
        results,           # results_state (updated in-place)
        new_elev_path,     # mask_editor (new elevation image)
    )


def apply_vlm_review(results: PipelineResults | None, confirmed_labels: list[str]) -> str:
    if results is None:
        return "未加载结果"
    all_elems = results.doors + results.windows
    confirmed_set = set(confirmed_labels)
    total = len(all_elems)
    kept = sum(1 for e in all_elems if f"{e.element_class} #{e.result_index}" in confirmed_set)
    return f"VLM审核：{kept}/{total} 个构件已确认，可导出到Revit"


# ---------------------------------------------------------------------------
# AI Agent (smolagents + Revit MCP)
# ---------------------------------------------------------------------------

_AGENT_CACHE: Any = None    # ToolCallingAgent instance, lazily created
_MCP_CM_CACHE: Any = None   # ToolCollection context manager — MUST stay alive to keep MCP subprocess + event loop running

REVIT_WS_PORT = 8080  # Revit plugin WebSocket port


def _check_revit_port() -> bool:
    """Lightweight check: is the Revit plugin WebSocket port open?

    This only checks TCP reachability — no Revit API round-trip,
    so it's fast and doesn't require Revit to respond to commands.
    """
    import socket
    try:
        with socket.create_connection(("127.0.0.1", REVIT_WS_PORT), timeout=3):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _format_agent_error(e: Exception) -> str:
    """Format agent errors into user-friendly Chinese messages.

    Detects common LLM API failure modes (rate limit, auth, network, timeout)
    and provides actionable guidance instead of raw stack traces.
    """
    err_str = str(e)
    err_type = type(e).__name__

    # OpenAI API errors
    if "429" in err_str or "RateLimitError" in err_type or "余额不足" in err_str:
        return (
            "❌ **API 额度不足 / 请求频率超限**\n\n"
            "LLM API 返回 429 错误，可能原因：\n"
            "- 账户余额不足，请充值\n"
            "- 请求过于频繁，请稍后重试\n"
            "- 免费额度已用完\n\n"
            f"原始错误：{err_str[:200]}"
        )

    if "401" in err_str or "AuthenticationError" in err_type or "无效" in err_str and "key" in err_str.lower():
        return (
            "❌ **API 认证失败**\n\n"
            "API Key 无效或已过期。请在 ⚙️ LLM API 配置 面板中检查：\n"
            "- API Key 是否正确\n"
            "- API Base URL 是否正确\n\n"
            f"原始错误：{err_str[:200]}"
        )

    if "Connection" in err_type or "ConnectionError" in err_type or "Connect" in err_str:
        return (
            "❌ **无法连接到 API 服务**\n\n"
            "可能原因：\n"
            "- 网络问题（防火墙 / 代理）\n"
            "- API Base URL 错误\n"
            "- 服务暂时不可用\n\n"
            f"原始错误：{err_str[:200]}"
        )

    if "Timeout" in err_type or "timeout" in err_str.lower():
        return (
            "❌ **API 请求超时**\n\n"
            "LLM 服务响应时间过长，可能原因：\n"
            "- 服务端负载高\n"
            "- 请求过于复杂（对话太长）\n"
            "- 网络延迟\n\n"
            f"原始错误：{err_str[:200]}"
        )

    if "Event loop is closed" in err_str:
        return (
            "❌ **MCP 连接断开**\n\n"
            "Revit MCP 子进程的 asyncio 事件循环已关闭。\n"
            "请点击「💾 保存配置」重置 Agent 连接后重试。"
        )

    # Generic fallback
    return f"❌ **Agent 错误** ({err_type})\n\n{err_str[:500]}"


def _get_agent(results: PipelineResults | None, scene_name: str):
    """Lazily create a smolagents ToolCallingAgent with Revit MCP tools."""
    global _AGENT_CACHE, _MCP_CM_CACHE
    if _AGENT_CACHE is not None:
        return _AGENT_CACHE

    from smolagents import ToolCollection, ToolCallingAgent
    from mcp import StdioServerParameters

    cfg = load_config()

    # Connect to Revit MCP server (stdio)
    # Store context manager at module scope so it's NOT garbage-collected
    # (GC would close the MCP subprocess and its asyncio event loop)
    server_params = StdioServerParameters(
        command=cfg.revit_mcp.command,
        args=cfg.revit_mcp.args,
    )
    _MCP_CM_CACHE = ToolCollection.from_mcp(server_params, trust_remote_code=True)
    tool_collection = _MCP_CM_CACHE.__enter__()  # type: ignore[attr-defined]
    tool_list = tool_collection.tools  # type: ignore[attr-defined]

    # Create LLM model from config
    model = get_llm_model(cfg)

    # Build context from pipeline results
    context = _build_agent_context(results, scene_name)

    # ToolCallingAgent calls tools directly from main thread (not sandboxed),
    # which is required for MCP tools that use asyncio internally.
    agent = ToolCallingAgent(
        tools=tool_list,
        model=model,
        instructions=context,
        max_steps=20,
    )
    _AGENT_CACHE = agent
    return agent


def _build_agent_context(results: PipelineResults | None, scene_name: str) -> str:
    """Build a system prompt with pipeline detection results and Revit workflow."""
    parts: list[str] = [
        "# 角色",
        "你是一个 BIM 自动化助手。你可以通过 Revit MCP 工具在 Revit 中创建建筑构件。",
        "你收到的检测数据来自 3DGS 场景分析（单位为米），向 Revit 工具传递坐标时**必须乘以 1000 转换为毫米(mm)**。",
        "",
        "# ⚠️ 最关键规则：门窗必须设置 hostWallId",
        "",
        "Revit 中门窗只有在**正确设置 hostWallId** 时才会在墙上自动开洞。",
        "不设 hostWallId 的门窗是自由放置的，**不会在墙上开洞**。",
        "",
        "正确流程：",
        "1. 用 `create_line_based_element` 批量创建所有墙 → **记录每面墙返回的 ElementId**",
        "2. 用 `create_point_based_element` 创建门窗时，**必须传入 `hostWallId`** 指定宿主墙的 ElementId",
        "",
        "# 完整创建流程（必须按此顺序）",
        "",
        "## 第 0 步：连接检查",
        "```",
        "say_hello()  // 确认 Revit MCP 连接正常",
        "```",
        "",
        "## 第 1 步：创建标高",
        "```",
        "create_level(data=[{name: 'BIM-Recon Floor', elevation: 0}])",
        "```",
        "",
        "## 第 2 步：批量创建墙体",
        "",
        "用 `create_line_based_element` 一次创建所有墙（data 是数组）：",
        "```",
        "create_line_based_element(data=[",
        "  {category:'OST_Walls', locationLine:{p0:{x,y,z:0}, p1:{x,y,z:0}}, thickness:200, height:2000, baseLevel:0, baseOffset:0},",
        "  // ... 更多墙",
        "])",
        "→ 返回 ElementId 数组，例如 [337703, 337706, 337709, 337712]",
        "  墙0 = ElementId 337703",
        "  墙1 = ElementId 337706",
        "  墙2 = ElementId 337709",
        "  墙3 = ElementId 337712",
        "```",
        "",
        "**记录每个 ElementId 与检测数据中墙索引(墙#N)的对应关系。**",
        "",
        "## 第 3 步：查询族类型",
        "```",
        "get_available_family_types(categoryList=['OST_Doors', 'OST_Windows'])",
        "→ 返回 typeId 列表，从中选择合适的基础类型",
        "```",
        "门推荐 typeId: 查到的「单扇 - 与墙齐」系列的任意一个",
        "窗推荐 typeId: 查到的「固定」系列的任意一个",
        "（设置 width+height 会自动复制类型并设自定义尺寸）",
        "",
        "## 第 4 步：批量创建门（hostWallId = 对应墙的 ElementId）",
        "```",
        "create_point_based_element(data=[",
        "  {",
        "    name: 'door #1',",
        "    typeId: <门的 typeId>,",
        "    locationPoint: {x, y, z: 0},     // 位置坐标 (mm)",
        "    width: 952,                        // 宽 (mm)",
        "    height: 1984,                      // 高 (mm)",
        "    baseLevel: 0,",
        "    baseOffset: 0,                     // 门: sill=0",
        "    hostWallId: 337706,               // ← 关键！墙1 的 ElementId",
        "  },",
        "])",
        "```",
        "",
        "## 第 5 步：批量创建窗（hostWallId + baseOffset = sill高度）",
        "```",
        "create_point_based_element(data=[",
        "  {",
        "    name: 'window #2',",
        "    typeId: <窗的 typeId>,",
        "    locationPoint: {x, y, z: 0},",
        "    width: 1426,",
        "    height: 1290,",
        "    baseLevel: 0,",
        "    baseOffset: 701,                  // ← 窗台高度 (mm)，来自检测数据 sill_height",
        "    hostWallId: 337709,               // ← 关键！墙2 的 ElementId",
        "  },",
        "])",
        "```",
        "",
        "# 坐标转换规则",
        "",
        "1. 检测数据单位为**米(m)**，Revit 工具单位为**毫米(mm)**",
        "2. 转换: `mm = round(m * 1000)`",
        "3. 3DGS 原点可能在负坐标，建议加偏移量（如 +10000mm X, +10000mm Y）避免与原点重叠",
        "4. Z 坐标统一用 0（baseLevel=0 即地面，baseOffset 控制高度）",
        "",
        "# 墙索引映射规则",
        "",
        "检测数据中 `墙#N` 对应 `wall_lines_snapped.json` 中的第 N 面墙。",
        "创建墙时按数组顺序创建，返回的 ElementId 数组索引 = 墙索引。",
        "门窗的 `wall_idx` 字段告诉你在哪面墙上开洞。",
        "",
        "# 其他规则",
        "",
        "1. **批量优先**: `create_line_based_element` 和 `create_point_based_element` 的 data 是数组，尽量一次性创建多个",
        "2. **typeId**: 门窗的 typeId 用 `get_available_family_types` 查询，设 width+height 会自动复制类型",
        "3. **逐步确认**: 每步完成后简要报告结果（创建了几个、ElementId 是什么）再继续",
        "4. **错误重试**: 工具返回错误时检查参数（特别是 hostWallId、typeId、单位）后重试",
        "5. **不创建重复**: 如用户要求「将所有构件导入」，按上述流程一次性完成，不要反复创建",
        "",
        "# 场景检测数据",
        f"场景名称: `{scene_name}`",
    ]

    if results and results.walls:
        parts.append(f"\n## 墙体（{len(results.walls)} 面）")
        for i, w in enumerate(results.walls):
            parts.append(
                f"  墙{i}: ({w['x1']:.3f}, {w['y1']:.3f}) → "
                f"({w['x2']:.3f}, {w['y2']:.3f}), "
                f"长 {w['length']:.3f}m"
            )

    if results and results.doors:
        confirmed = [d for d in results.doors if d.confirmed]
        parts.append(f"\n## 门（{len(confirmed)} 个已确认）")
        for d in confirmed:
            hd = d.height_detection or {}
            parts.append(
                f"  门#{d.result_index} (在墙#{d.wall_idx}上): "
                f"位置({d.world_x:.3f}, {d.world_y:.3f})m, "
                f"宽 {hd.get('width_m', 0):.3f}m, "
                f"高 {hd.get('element_height', 0):.3f}m, "
                f"窗台 {hd.get('sill_height', 0):.3f}m"
            )

    if results and results.windows:
        confirmed = [w for w in results.windows if w.confirmed]
        parts.append(f"\n## 窗（{len(confirmed)} 个已确认）")
        for w in confirmed:
            hd = w.height_detection or {}
            parts.append(
                f"  窗#{w.result_index} (在墙#{w.wall_idx}上): "
                f"位置({w.world_x:.3f}, {w.world_y:.3f})m, "
                f"宽 {hd.get('width_m', 0):.3f}m, "
                f"高 {hd.get('element_height', 0):.3f}m, "
                f"窗台 {hd.get('sill_height', 0):.3f}m"
            )

    if not results or (not results.walls and not results.doors and not results.windows):
        parts.append("\n（尚未加载检测结果，请先运行管线或在上方加载结果）")

    parts.append("")
    return "\n".join(parts)


def agent_chat(
    message: str,
    history: list,
    results: PipelineResults | None,
    scene_name: str,
) -> str:
    """Handle a chat message from the user."""
    if not message.strip():
        return ""
    # Lightweight Revit connectivity check (TCP port only, no API round-trip)
    if not _check_revit_port():
        return "❌ Revit 未连接：端口 8080 不可达。请确认 Revit 已启动且 MCP 插件已加载。"
    try:
        agent = _get_agent(results, scene_name)
        response = agent.run(message)
        return str(response)
    except Exception as e:
        return f"❌ Agent 错误: {e}"



# ---------------------------------------------------------------------------
# Explorer Agent (smolagents + Explorer MCP) — mirrors Revit agent
# ---------------------------------------------------------------------------

_EXP_AGENT_CACHE: Any = None
_EXP_MCP_CM: Any = None


def _reset_explorer_agent():
    """Kill old explorer subprocess + clear caches."""
    global _EXP_AGENT_CACHE, _EXP_MCP_CM
    if _EXP_MCP_CM is not None:
        try:
            _EXP_MCP_CM.__exit__(None, None, None)
        except Exception:
            pass
    _EXP_AGENT_CACHE = None
    _EXP_MCP_CM = None


def _get_explorer_agent(scene_name: str):
    """Create a ToolCallingAgent connected to the explorer MCP server."""
    global _EXP_AGENT_CACHE, _EXP_MCP_CM
    if _EXP_AGENT_CACHE is not None:
        return _EXP_AGENT_CACHE

    from smolagents import ToolCollection, ToolCallingAgent
    from mcp import StdioServerParameters

    data_dir = ROOT / "data" / scene_name
    ply_candidates = sorted(data_dir.glob("point_cloud_*.ply")) or sorted(data_dir.glob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"PLY not found in {data_dir}")
    ply_path = ply_candidates[0]
    feat_candidates = sorted(data_dir.glob("*_feat.pt"))
    if not feat_candidates:
        out_dir = ROOT / "output" / scene_name
        feat_candidates = sorted(out_dir.glob("*_feat.pt"))
    feat_path = feat_candidates[0] if feat_candidates else None

    mcp_args = ["-m", "bim_recon.mcp_explorer", "--ply", str(ply_path)]
    if feat_path and Path(feat_path).exists():
        mcp_args.extend(["--feat", str(feat_path)])
    mcp_args.extend(["--explore-dir", str(ROOT / "output" / scene_name / "explore")])

    server_params = StdioServerParameters(
        command=sys.executable,
        args=mcp_args,
    )
    _EXP_MCP_CM = ToolCollection.from_mcp(server_params, trust_remote_code=True)
    tool_collection = _EXP_MCP_CM.__enter__()
    tool_list = tool_collection.tools

    cfg = load_config()
    model = get_llm_model(cfg)

    instructions = (
        "你是一个室内场景探索者。你在 3D Gaussian Splatting 渲染的房间中，"
        "任务是找到所有 B 类物体（家具、装饰品、电器等）并记录 3D 位置。\n\n"
        "## 探索策略\n"
        "1. 首先调用 explore_init 初始化（如果尚未初始化）\n"
        "2. 系统性地旋转 360°：每次 turn(45) 后 detect_objects\n"
        "3. 搜索词要多样：chair, table, sofa, cabinet, bed, lamp, vase, plant, shelf, desk\n"
        "4. 每个检测到的物体调用 tag_object 记录\n"
        "5. 如果视角不好（看不到物体），step('forward', 1.0) 移动位置\n"
        "6. 完成后调用 list_found 查看汇总\n\n"
        "## 重要\n"
        "- 每次只处理一个视角，不要一次旋转太多\n"
        "- tag_object 的 bbox 是归一化坐标 [0,1]，从 detect_objects 的返回值中获取\n"
        "- 用 get_status 确认当前位置和已发现物体数量\n"
    )

    agent = ToolCallingAgent(
        tools=tool_list,
        model=model,
        instructions=instructions,
        max_steps=50,
    )
    _EXP_AGENT_CACHE = agent
    return agent

# ---------------------------------------------------------------------------
# UI — 先定义所有组件，最后统一绑定事件
# ---------------------------------------------------------------------------

