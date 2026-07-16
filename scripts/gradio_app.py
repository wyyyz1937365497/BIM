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

# 抑制 Starlette 的弃用警告（StarletteDeprecationWarning 是自定义类，非标准 DeprecationWarning）
warnings.filterwarnings('ignore', message='.*HTTP_422_UNPROCESSABLE_ENTITY.*')
warnings.filterwarnings('ignore', message='.*deprecated.*', module='starlette.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gradio.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='starlette.*')

# 配置日志（force=True 覆盖 Gradio/uvicorn 的默认配置，确保我们的日志可见）
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
    
    logger.info(f"最终统计: radar_imgs={len(radar_imgs)}, vlm_imgs={len(vlm_imgs)}, seg_imgs={len(seg_imgs)}")
    logger.info("=" * 60)

    print(f"[DEBUG] gallery counts: radar={len(radar_imgs)} vlm={len(vlm_imgs)}")

    return (res, summary, vlm_imgs, radar_imgs, res.report,
            gr.update(choices=choices, value=defaults),
            gr.update(choices=choices))


def run_pipeline_streaming(scene: str, doors: bool, windows: bool,
                           falcon: bool, skip_vlm: bool):
    """生成器：实时流式输出管线子进程日志，完成后自动加载结果。

    yield 9 个值: (console, out_dir, results, summary, radar_gallery,
                    vlm_gallery, report, vlm_cb, elem_dd)
    """
    if not scene:
        yield ("❌ 错误：未选择场景", "", None, "", None, [], None, gr.update(), gr.update())
        return
    elems = []
    if doors:
        elems.append("door")
    if windows:
        elems.append("window")
    if not elems:
        yield ("❌ 错误：至少选择一种构件类型", "", None, "", None, [], None, gr.update(), gr.update())
        return

    args = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
            "--name", scene, "--elements", *elems]
    if skip_vlm:
        args.append("--skip-vlm")

    yield (f"启动管线...\n{' '.join(args)}\n", "", None, "运行中...",
           None, [], None, gr.update(), gr.update())

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT), bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line.rstrip())
        yield ("\n".join(lines[-MAX_CONSOLE_LINES:]), "", None, "运行中...",
               None, [], None, gr.update(), gr.update())

    proc.wait()
    if proc.returncode != 0:
        console = "\n".join(lines[-MAX_CONSOLE_LINES:])
        yield (f"{console}\n\n❌ 管线失败 (退出码 {proc.returncode})",
               "", None, "❌ 失败", None, [], None, gr.update(), gr.update())
        return

    out = find_latest_output(scene)
    console = "\n".join(lines[-MAX_CONSOLE_LINES:]) + f"\n✅ 完成: {out}"
    logger.info(f"管线完成，输出目录: {out}")
    print(f"\n[DEBUG run_pipeline_streaming] 输出目录: {out}", flush=True)

    try:
        logger.info(f"开始加载结果: {out}")
        res = load_results(Path(out))
        logger.info(f"结果加载成功: {len(res.doors)} 扇门, {len(res.windows)} 扇窗, {len(res.walls)} 面墙")
        print(f"[DEBUG] load_results: 门={len(res.doors)} 窗={len(res.windows)} 墙={len(res.walls)}", flush=True)
    except Exception as e:
        logger.error(f"加载结果失败: {e}", exc_info=True)
        print(f"[DEBUG] load_results 失败: {e}", flush=True)
        yield (f"{console}\n❌ 加载结果失败: {e}", out, None, "❌ 加载失败",
               None, [], None, gr.update(), gr.update())
        return

    logger.info("开始准备结果展示")
    r = _prepare_results(res)
    logger.info(f"结果准备完成，准备 yield 到 UI")
    # r = (res, summary, vlm_imgs, radar_imgs, report, vlm_cb, elem_dd)
    yield (console, out, r[0], r[1], r[3], r[2], r[4], r[5], r[6])


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

    ply_path = ROOT / "data" / f"{scene_name}.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY 不存在: {ply_path}")
    feat_path = ROOT / "output" / scene_name / f"{scene_name}_feat.pt"

    mcp_args = ["-m", "bim_recon.mcp_explorer", "--ply", str(ply_path)]
    if feat_path.exists():
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
            fn=run_pipeline_streaming,
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
                    f"调用 explore_init(center_x={eye_x:.2f}, center_z={eye_z:.2f}, "
                    f"eye_height={eye_y:.2f}, initial_yaw={yaw})，"
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
    build_app().launch(server_port=7860, server_name="127.0.0.1")
