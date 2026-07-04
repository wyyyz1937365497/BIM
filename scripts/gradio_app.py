"""3DGS → BIM Pipeline — Gradio Web UI (v3)

三个标签页：
  1. 数据准备 — PLY上传 → 验证 → SceneSplat预处理 → 3DGS查看器
  2. 管线与结果 — 运行检测管线（实时输出）→ 墙线/VLM/Seg结果
  3. 微调 — 选择构件 → 立面图+bbox可视化调整 → 重计算尺寸

启动：
    cmd /c "call \"...\\vcvars64.bat\" && python scripts/gradio_app.py"
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.pipeline_api import (
    PipelineResults, load_results, remap_from_json, mask_to_bbox,
)
from bim_recon.config import load_config, get_llm_model, save_config, test_llm_connection

VIEWER_PORT = 8081
CAMERA_PORT = 8082
MAX_PORT_WAIT_S = 30
SCENESPLAT = ROOT / "SceneSplat"
MAX_CONSOLE_LINES = 200

# Lazy-loaded singletons (avoid importing heavy deps at module load)
_SCENE_CACHE: dict[str, Any] = {}   # scene_name -> GSScene
_FALCON_CACHE: Any = None           # FalconClient or None


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
    subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_viewer.py"),
         "--input-root", input_root, "--feature-path", feat_path,
         "--port", str(VIEWER_PORT)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
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

    返回顺序: (results, summary, vlm_imgs, seg_imgs, topdown, report,
               vlm_cb_update, elem_dd_update)
    """
    topdown = res.wall_topdown_image or None
    vlm_imgs = [(e.image_path, f"{e.element_class} #{e.result_index} ({'✓' if e.confirmed else '✗'})")
                for e in (res.doors + res.windows) if Path(e.image_path).exists()]
    # Seg gallery: show overlay if available, else elevation image
    seg_imgs = []
    for e in (res.doors + res.windows):
        if e.overlay_image and Path(e.overlay_image).exists():
            seg_imgs.append((e.overlay_image, f"{e.element_class} #{e.result_index} (Falcon)"))
        elif e.elevation_image and Path(e.elevation_image).exists():
            seg_imgs.append((e.elevation_image, f"{e.element_class} #{e.result_index} (elevation)"))
    summary = (f"### 结果\n- 墙体: {len(res.walls)}\n"
               f"- 门: {sum(1 for d in res.doors if d.confirmed)}/{len(res.doors)} 已确认\n"
               f"- 窗: {sum(1 for w in res.windows if w.confirmed)}/{len(res.windows)} 已确认")
    all_elems = res.doors + res.windows
    choices = [f"{e.element_class} #{e.result_index}" for e in all_elems]
    defaults = [f"{e.element_class} #{e.result_index}" for e in all_elems if e.confirmed]
    return (res, summary, vlm_imgs, seg_imgs, topdown, res.report,
            gr.update(choices=choices, value=defaults),
            gr.update(choices=choices))


def run_pipeline_streaming(scene: str, doors: bool, windows: bool,
                           falcon: bool, skip_vlm: bool):
    """生成器：实时流式输出管线子进程日志，完成后自动加载结果。

    yield 10 个值: (console, out_dir, results, summary, wall_img,
                    vlm_gallery, seg_gallery, report, vlm_cb, elem_dd)
    """
    if not scene:
        yield ("❌ 错误：未选择场景", "", None, "", None, [], [], None,
               gr.update(), gr.update())
        return
    elems = []
    if doors:
        elems.append("door")
    if windows:
        elems.append("window")
    if not elems:
        yield ("❌ 错误：至少选择一种构件类型", "", None, "", None, [], [], None,
               gr.update(), gr.update())
        return

    args = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
            "--name", scene, "--elements", *elems]
    if skip_vlm:
        args.append("--skip-vlm")
    if not falcon:
        args.append("--no-falcon")

    yield (f"启动管线...\n{' '.join(args)}\n", "", None, "运行中...",
           None, [], [], None, gr.update(), gr.update())

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT), bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line.rstrip())
        yield ("\n".join(lines[-MAX_CONSOLE_LINES:]), "", None, "运行中...",
               None, [], [], None, gr.update(), gr.update())

    proc.wait()
    if proc.returncode != 0:
        console = "\n".join(lines[-MAX_CONSOLE_LINES:])
        yield (f"{console}\n\n❌ 管线失败 (退出码 {proc.returncode})",
               "", None, "❌ 失败", None, [], [], None, gr.update(), gr.update())
        return

    out = find_latest_output(scene)
    console = "\n".join(lines[-MAX_CONSOLE_LINES:]) + f"\n✅ 完成: {out}"

    try:
        res = load_results(Path(out))
    except Exception as e:
        yield (f"{console}\n❌ 加载结果失败: {e}", out, None, "❌ 加载失败",
               None, [], [], None, gr.update(), gr.update())
        return

    r = _prepare_results(res)
    # r = (res, summary, vlm_imgs, seg_imgs, topdown, report, vlm_cb, elem_dd)
    yield (console, out, r[0], r[1], r[4], r[2], r[3], r[5], r[6], r[7])


def load_results_cb(out_dir: str):
    """从输出目录加载已有结果。返回 8 个值。"""
    if not out_dir or not Path(out_dir).exists():
        return None, "无结果", [], [], None, None, gr.update(choices=[]), gr.update(choices=[])
    try:
        res = load_results(Path(out_dir))
    except Exception as e:
        return None, f"❌ 加载失败: {e}", [], [], None, None, gr.update(choices=[]), gr.update(choices=[])
    return _prepare_results(res)


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
    for e in results.doors + results.windows:
        if e.result_index == target_idx and e.element_class == elem_class:
            return e
    return None


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
    """Lazy-load and cache a GSScene for rendering."""
    if scene_name in _SCENE_CACHE:
        return _SCENE_CACHE[scene_name]
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

    # 6. Update ElementResult — replace frozen dataclass with updated copy
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
    # Replace in the results lists
    elem_list = results.doors if elem.element_class == "door" else results.windows
    for idx, e in enumerate(elem_list):
        if e.result_index == elem.result_index:
            elem_list[idx] = new_elem
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

    # 7. Prepare refreshed Seg gallery + mask editor image
    seg_imgs = []
    for e in (results.doors + results.windows):
        if e.overlay_image and Path(e.overlay_image).exists():
            seg_imgs.append((e.overlay_image, f"{e.element_class} #{e.result_index} (Falcon)"))
        elif e.elevation_image and Path(e.elevation_image).exists():
            seg_imgs.append((e.elevation_image, f"{e.element_class} #{e.result_index} (elevation)"))

    status = f"✅ Falcon 检测到 {elem.element_class}，已从新视角重算尺寸并更新"

    return (
        overlay,           # reseg_preview
        dims,              # reseg_out
        status,            # cam_status
        results,           # results_state (updated in-place)
        seg_imgs,          # seg_gallery
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
    """Build a system prompt with pipeline detection results."""
    parts = [
        "你是一个 BIM 自动化助手。你可以通过 Revit MCP 工具在 Revit 中创建建筑构件。",
        "所有坐标单位为毫米(mm)。以下是管线检测结果：",
        f"\n场景: {scene_name}",
    ]
    if results and results.walls:
        parts.append(f"\n## 墙体 ({len(results.walls)} 面)")
        for i, w in enumerate(results.walls):
            parts.append(
                f"  墙{i}: ({w['x1']:.0f}, {w['y1']:.0f}) → "
                f"({w['x2']:.0f}, {w['y2']:.0f}), 长{w['length']:.0f}mm"
            )
    if results and results.doors:
        confirmed = [d for d in results.doors if d.confirmed]
        parts.append(f"\n## 门 ({len(confirmed)} 个已确认)")
        for d in confirmed:
            hd = d.height_detection or {}
            parts.append(
                f"  {d.element_class}#{d.result_index}: "
                f"位置({d.world_x:.0f}, {d.world_y:.0f}), "
                f"墙#{d.wall_idx}, "
                f"宽{hd.get('width_m', 0)*1000:.0f}mm, "
                f"高{hd.get('element_height', 0)*1000:.0f}mm, "
                f"窗台{hd.get('sill_height', 0)*1000:.0f}mm"
            )
    if results and results.windows:
        confirmed = [w for w in results.windows if w.confirmed]
        parts.append(f"\n## 窗 ({len(confirmed)} 个已确认)")
        for w in confirmed:
            hd = w.height_detection or {}
            parts.append(
                f"  {w.element_class}#{w.result_index}: "
                f"位置({w.world_x:.0f}, {w.world_y:.0f}), "
                f"墙#{w.wall_idx}, "
                f"宽{hd.get('width_m', 0)*1000:.0f}mm, "
                f"高{hd.get('element_height', 0)*1000:.0f}mm, "
                f"窗台{hd.get('sill_height', 0)*1000:.0f}mm"
            )
    parts.append(
        "\n你可以使用 Revit MCP 工具来创建这些构件。"
        "例如：先创建墙体，再在墙上放置门窗。"
        "请逐步执行，每步确认结果。"
    )
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
# UI — 先定义所有组件，最后统一绑定事件
# ---------------------------------------------------------------------------

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

        # ====== ③ 检测结果 ======
        gr.Markdown("---\n## ③ 检测结果")
        summary_md = gr.Markdown("运行管线后，结果将在此显示。")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 墙线俯视图")
                wall_img = gr.Image(height=400, label="", show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("#### VLM 验证图库")
                vlm_gallery = gr.Gallery(
                    columns=2, height=400, label="", show_label=False,
                    object_fit="contain", preview=True,
                )
        gr.Markdown("#### Seg 叠加图库（Falcon 分割 / 立面渲染）")
        seg_gallery = gr.Gallery(
            columns=4, height=450, label="", show_label=False,
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
            fn=run_pipeline_streaming,
            inputs=[scene_state, cb_doors, cb_windows, cb_falcon, cb_skipvlm],
            outputs=[console_out, out_dir_box, results_state, summary_md,
                     wall_img, vlm_gallery, seg_gallery, report_json,
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
            outputs=[results_state, summary_md, vlm_gallery, seg_gallery,
                     wall_img, report_json, vlm_review_cbs, elem_sel],
        )
        review_btn.click(fn=apply_vlm_review,
                         inputs=[results_state, vlm_review_cbs],
                         outputs=review_status)

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
                     results_state, seg_gallery, mask_editor],
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
            history = history + [{"role": "user", "content": message}]
            yield history, ""  # show user message immediately
            try:
                agent = _get_agent(results, scene_name)
                response = agent.run(message)
                history.append({"role": "assistant", "content": str(response)})
                yield history, ""
            except Exception as e:
                history.append({"role": "assistant", "content": f"❌ Agent 错误: {e}"})
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

    return app


if __name__ == "__main__":
    build_app().launch(server_port=7860, server_name="127.0.0.1")
