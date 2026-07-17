"""B-class Object Explorer MCP Server.

A standalone MCP server for VLM-driven autonomous object discovery in 3DGS
scenes.  Separate from the Revit / structural MCP server (``mcp_gs.py``).

The VLM "walks" through the rendered scene, detecting and tagging
fine-grained objects (chair, vase, cabinet, ...) via Falcon-Perception,
then queues them for TRELLIS mesh generation.

Tools:
  Navigation : explore_init, turn, step, look_at, get_status
  Detection  : detect_objects, tag_object, find_best_angle
  Queue      : list_found, queue_for_trellis

Run with::

    python -m bim_recon.mcp_explorer \\
        --ply data/point_cloud_30000.ply \\
        --feat output/point_cloud_30000/point_cloud_30000_feat.pt \\
        --falcon-port 8390 \\
        --explore-dir output/explore

Or set env vars ``GS_PLY_PATH``, ``GS_FEAT_PATH``, ``FALCON_PORT``.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage

from bim_recon.gs_scene import CameraPose, GSScene, look_at_pose

# Lazy import — Falcon lives in the same env but the server may start first.
_falcon_module = None


def _get_falcon_client(host: str, port: int, timeout: int = 300):
    global _falcon_module
    if _falcon_module is None:
        from bim_recon.falcon_client import FalconClient
        _falcon_module = FalconClient
    return _falcon_module(host=host, port=port, timeout=timeout)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class ExplorerState:
    """Holds the loaded scene, camera pose, found objects, and Falcon client."""

    scene: GSScene
    # Camera pose (Y-up convention, matching mcp_gs.py).
    cam_eye: List[float] = field(default_factory=lambda: [0.0, 1.5, 0.0])
    cam_yaw: float = 0.0           # radians; 0 = +x, π/2 = +z
    cam_fov: float = 60.0
    # Output.
    explore_dir: Path = field(default_factory=lambda: Path("output/explore"))
    view_counter: int = 0
    # Found objects.
    found: List[Dict[str, Any]] = field(default_factory=list)
    obj_counter: int = 0
    # Render resolution.
    width: int = 1024
    height: int = 768
    # Falcon connection params.
    falcon_host: str = "127.0.0.1"
    falcon_port: int = 8390
    # Cached scene bounds.
    bounds_min: Tuple[float, ...] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, ...] = (10.0, 3.0, 10.0)
    up_axis: int = 2               # 0=x, 1=y, 2=z (auto-detected)
    initialized: bool = False


_STATE: Optional[ExplorerState] = None


def _req() -> ExplorerState:
    if _STATE is None:
        raise RuntimeError("Explorer not initialised. Call explore_init first.")
    return _STATE


# ---------------------------------------------------------------------------
# Camera math (adaptive up-axis; horizontal plane = the two non-up axes)
# ---------------------------------------------------------------------------

# Axis indices for horizontal movement given up_axis
# up_axis=2 (Z-up): h_axes = [0, 1] (XY plane)
# up_axis=1 (Y-up): h_axes = [0, 2] (XZ plane)
# up_axis=0 (X-up): h_axes = [1, 2] (YZ plane)
def _h_axes(up_axis: int) -> Tuple[int, int]:
    if up_axis == 2:
        return 0, 1
    elif up_axis == 1:
        return 0, 2
    else:
        return 1, 2

def _up_vec(up_axis: int) -> Tuple[float, float, float]:
    v = [0.0, 0.0, 0.0]
    v[up_axis] = 1.0
    return tuple(v)

def _target_from_eye_yaw(eye: List[float], yaw: float, up_axis: int) -> List[float]:
    """Compute a look-at target from eye position + yaw (horizontal only)."""
    h0, h1 = _h_axes(up_axis)
    target = list(eye)
    target[h0] = eye[h0] + math.cos(yaw)
    target[h1] = eye[h1] + math.sin(yaw)
    return target

def _yaw_from_eye_target(eye: List[float], target: List[float], up_axis: int) -> float:
    """Compute yaw from eye → target direction (projected to horizontal plane)."""
    h0, h1 = _h_axes(up_axis)
    dh0 = target[h0] - eye[h0]
    dh1 = target[h1] - eye[h1]
    return math.atan2(dh1, dh0)

def _forward(yaw: float, up_axis: int) -> Tuple[float, ...]:
    """Horizontal forward direction for *yaw* as a full 3D vector."""
    h0, h1 = _h_axes(up_axis)
    v = [0.0, 0.0, 0.0]
    v[h0] = math.cos(yaw)
    v[h1] = math.sin(yaw)
    return tuple(v)

def _right(yaw: float, up_axis: int) -> Tuple[float, ...]:
    """Horizontal right direction for *yaw* as a full 3D vector."""
    h0, h1 = _h_axes(up_axis)
    v = [0.0, 0.0, 0.0]
    v[h0] = -math.sin(yaw)
    v[h1] = math.cos(yaw)
    return tuple(v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_current(s: ExplorerState) -> Tuple[bytes, "RenderResult", CameraPose]:
    """Render from the current camera pose. Returns (png_bytes, result, pose)."""
    target = _target_from_eye_yaw(s.cam_eye, s.cam_yaw, s.up_axis)
    pose = look_at_pose(
        eye=tuple(s.cam_eye),
        target=tuple(target),
        up=_up_vec(s.up_axis),
    )
    result = s.scene.render(pose, s.width, s.height, s.cam_fov)
    arr = np.clip(result.colors * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue(), result, pose


def _save_view(s: ExplorerState, png: bytes, suffix: str = "") -> str:
    """Save a rendered view to explore_dir and return the path."""
    s.view_counter += 1
    name = f"view_{s.view_counter:03d}{suffix}.png"
    s.explore_dir.mkdir(parents=True, exist_ok=True)
    path = s.explore_dir / name
    path.write_bytes(png)
    return str(path)


def _bbox_to_mask(bbox: Dict[str, float], w: int, h: int) -> np.ndarray:
    """Convert normalised bbox {x, y, w, h} to a boolean HxW mask."""
    x0 = max(0, int(bbox["x"] * w))
    y0 = max(0, int(bbox["y"] * h))
    x1 = min(w, int((bbox["x"] + bbox["w"]) * w))
    y1 = min(h, int((bbox["y"] + bbox["h"]) * h))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _estimate_3d(
    scene: GSScene, pose: CameraPose, bbox: Dict[str, float],
    width: int, height: int, fov: float,
) -> Optional[List[float]]:
    """Estimate the 3D centroid of an object from its 2D bbox via Gaussian IDs."""
    mask = _bbox_to_mask(bbox, width, height)
    try:
        ids = scene.select_by_mask(pose, mask, width, height, fov)
    except Exception:
        return None
    if len(ids) == 0:
        # Fallback: sample depth at bbox centre.
        return None
    means = scene.means[torch.as_tensor(ids, device=scene.device)]
    return means.mean(dim=0).cpu().tolist()


def _check_falcon(s: ExplorerState):
    """Return a FalconClient or raise with a helpful message."""
    client = _get_falcon_client(s.falcon_host, s.falcon_port)
    if not client.health():
        raise RuntimeError(
            f"Falcon server not reachable at {s.falcon_host}:{s.falcon_port}. "
            "Start it first:  python falcon_inference_server.py --port 8390 ..."
        )
    return client


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def build_server(state: ExplorerState) -> FastMCP:
    global _STATE
    _STATE = state

    mcp = FastMCP("bim-explorer")

    # ── Navigation ────────────────────────────────────────────────────

    @mcp.tool()
    def explore_init(
        center_x: float = 0.0,
        center_z: float = 0.0,
        eye_height: float = 1.5,
        initial_yaw: float = 0.0,
    ) -> Any:
        """Initialise the explorer at the room centre (or a custom position).

        Call this **once** before any other tool.  By default the camera is
        placed at the scene's horizontal centre at ``eye_height`` metres above
        the floor, looking along +x.

        Args:
            center_x: Override X coordinate.  Default: scene centre X.
            center_z: Override Z coordinate.  Default: scene centre Z.
            eye_height: Camera height above floor (default 1.5 m).
            initial_yaw: Initial look direction in degrees (0 = +x, 90 = +z).

        Returns:
            A rendered PNG of the initial viewpoint.
        """
        s = _req()
        # Lazy-load scene on first call
        if s.scene is None:
            print(f"[explorer] Loading scene: {s._ply_path}", file=sys.stderr)
            s.scene = GSScene.from_ply(s._ply_path, feat_path=s._feat_path)
            mn = s.scene.means.min(dim=0).values.cpu().numpy()
            mx = s.scene.means.max(dim=0).values.cpu().numpy()
            s.bounds_min = tuple(mn.tolist())
            s.bounds_max = tuple(mx.tolist())
            means_np = s.scene.means.cpu().numpy()
            extents = np.percentile(means_np, 95, axis=0) - np.percentile(means_np, 5, axis=0)
            s.up_axis = int(np.argmin(extents))
            print(f"[explorer] {s.scene.num_gaussians:,} Gaussians, up_axis={s.up_axis}", file=sys.stderr)

        mn, mx = s.bounds_min, s.bounds_max
        h0, h1 = _h_axes(s.up_axis)
        up = s.up_axis
        cam_eye = [(mn[i] + mx[i]) / 2 for i in range(3)]
        cam_eye[up] = float(mn[up]) + float(eye_height if eye_height is not None else 1.5)
        if center_x is not None:
            cam_eye[h0] = float(center_x)
        if center_z is not None:
            cam_eye[h1] = float(center_z)
        s.cam_eye = cam_eye
        s.cam_yaw = math.radians(initial_yaw)
        s.initialized = True
        s.explore_dir.mkdir(parents=True, exist_ok=True)

        png, _result, _pose = _render_current(s)
        saved = _save_view(s, png)
        print(f"[explorer] init at ({cam_eye[0]:.2f}, {cam_eye[1]:.2f}, {cam_eye[2]:.2f}) "
              f"yaw={initial_yaw:.0f}deg up_axis={s.up_axis} -> {saved}", file=sys.stderr)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def turn(yaw_degrees: float = 45.0) -> Any:
        """Rotate the camera in place by *yaw_degrees* and render the new view.

        Positive = clockwise when viewed from above.  Use this to systematically
        scan 360° (e.g. call ``turn(45)`` eight times).

        Returns:
            A rendered PNG of the new viewpoint.
        """
        s = _req()
        s.cam_yaw += math.radians(yaw_degrees)
        png, _r, _p = _render_current(s)
        _save_view(s, png)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def step(
        direction: str = "forward",
        distance: float = 1.0,
    ) -> Any:
        """Move horizontally in the current view direction and render.

        Args:
            direction: One of ``"forward"``, ``"back"``, ``"left"``, ``"right"``.
            distance: Movement in metres.

        Returns:
            A rendered PNG of the new viewpoint.
        """
        s = _req()
        yaw = s.cam_yaw
        if direction == "forward":
            move = _forward(yaw, s.up_axis)
        elif direction == "back":
            fwd = _forward(yaw, s.up_axis)
            move = tuple(-f for f in fwd)
        elif direction == "right":
            move = _right(yaw, s.up_axis)
        elif direction == "left":
            rt = _right(yaw, s.up_axis)
            move = tuple(-r for r in rt)
        else:
            return json.dumps({"error": f"Unknown direction '{direction}'. "
                                         "Use forward/back/left/right."})
        for i in range(3):
            s.cam_eye[i] += move[i] * distance
        png, _r, _p = _render_current(s)
        _save_view(s, png)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def look_at(
        eye_x: float, eye_y: float, eye_z: float,
        target_x: float, target_y: float, target_z: float,
        fov: float = 60.0,
    ) -> Any:
        """Move to an absolute position and look at a target.  Renders the view.

        Use this when you know exact coordinates (e.g. to revisit a found object).
        """
        s = _req()
        s.cam_eye = [eye_x, eye_y, eye_z]
        s.cam_yaw = _yaw_from_eye_target(
            [eye_x, eye_y, eye_z], [target_x, target_y, target_z],
            s.up_axis,
        )
        s.cam_fov = fov
        png, _r, _p = _render_current(s)
        _save_view(s, png)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def get_status() -> str:
        """Return current camera pose, scene bounds, and a summary of found objects.

        Call this anytime to orient yourself in the scene.
        """
        target = _target_from_eye_yaw(s.cam_eye, s.cam_yaw, s.up_axis)
        yaw_deg = math.degrees(s.cam_yaw) % 360
        summary = {
            "camera": {
                "eye": [round(v, 2) for v in s.cam_eye],
                "looking_at": [round(v, 2) for v in target],
                "yaw_degrees": round(yaw_deg, 1),
                "fov": s.cam_fov,
            },
            "scene_bounds": {
                "min": [round(v, 2) for v in s.bounds_min],
                "max": [round(v, 2) for v in s.bounds_max],
            },
            "found_objects": len(s.found),
            "objects": [
                {"id": o["id"], "label": o["label"],
                 "position_3d": [round(v, 2) for v in o["position_3d"]],
                 "trellis_status": o["trellis_status"]}
                for o in s.found
            ],
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)

    # ── Detection ─────────────────────────────────────────────────────

    @mcp.tool()
    def detect_objects(query: str) -> str:
        """Run Falcon-Perception detection on the **current** view.

        Args:
            query: Object(s) to find, comma-separated.
                   e.g. ``"chair, table, vase, cabinet, sofa, lamp, plant"``.

        Returns:
            JSON list of detections: ``[{label, bbox, confidence}, ...]``.
            ``bbox`` is normalised ``{x, y, w, h}`` with origin top-left.
        """
        s = _req()
        client = _check_falcon(s)
        png, _result, _pose = _render_current(s)
        saved_path = _save_view(s, png, suffix=f"_detect_{query.replace(',', '_').strip()}")
        print(f"[explorer] detect '{query}' → {saved_path}", file=sys.stderr)
        img = PILImage.open(io.BytesIO(png))
        dets = client.segment(img, query, task="detection")
        out = []
        for d in dets:
            out.append({
                "label": query.split(",")[0].strip(),  # Falcon returns one label per call
                "bbox": d.bbox,
                "mask_area_ratio": d.mask_area_ratio,
            })
        return json.dumps({
            "query": query,
            "count": len(out),
            "detections": out,
        }, indent=2, ensure_ascii=False)

    @mcp.tool()
    def tag_object(
        label: str,
        bbox_x: float,
        bbox_y: float,
        bbox_w: float,
        bbox_h: float,
    ) -> str:
        """Save a detected object at the given 2D bbox (normalised [0,1]).

        Estimates the 3D position by backprojecting the bbox through the depth
        map.  If the new object is within 0.3 m of an existing one with the
        same label, it is treated as a duplicate and skipped.

        Args:
            label: Fine-grained object type (e.g. ``"chair"``, ``"vase"``).
            bbox_x, bbox_y: Top-left corner of the bbox (normalised).
            bbox_w, bbox_h: Width / height of the bbox (normalised).

        Returns:
            JSON with the object ID, estimated 3D position, and status.
        """
        s = _req()
        bbox = {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h}

        # Re-render to get pose + depth for 3D estimation.
        png, result, pose = _render_current(s)
        pos3d = _estimate_3d(s.scene, pose, bbox, s.width, s.height, s.cam_fov)

        if pos3d is None:
            # Fallback: depth at bbox centre.
            cx_px = int((bbox_x + bbox_w / 2) * s.width)
            cy_px = int((bbox_y + bbox_h / 2) * s.height)
            d = float(result.depth[cy_px, cx_px])
            if d > 0:
                fx = 0.5 * s.width / math.tan(0.5 * math.radians(s.cam_fov))
                x_cam = (cx_px - s.width / 2) / fx * d
                y_cam = (cy_px - s.height / 2) / fx * d
                # Approximate world position (ignores rotation — good enough for dedup).
                pos3d = [s.cam_eye[0] + x_cam, s.cam_eye[1] - y_cam, s.cam_eye[2] + d]
            else:
                pos3d = list(s.cam_eye)

        # Duplicate check.
        for existing in s.found:
            if existing["label"] != label:
                continue
            dx = existing["position_3d"][0] - pos3d[0]
            dy = existing["position_3d"][1] - pos3d[1]
            dz = existing["position_3d"][2] - pos3d[2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < 0.3:
                return json.dumps({
                    "status": "duplicate",
                    "duplicate_of": existing["id"],
                    "label": label,
                    "position_3d": [round(v, 2) for v in pos3d],
                }, ensure_ascii=False)

        # Save the view as the object's best view.
        s.obj_counter += 1
        obj_id = f"obj_{s.obj_counter:03d}"
        view_path = _save_view(s, png, suffix=f"_{label}_{obj_id}")

        obj = {
            "id": obj_id,
            "label": label,
            "position_3d": [round(v, 3) for v in pos3d],
            "best_view": view_path,
            "best_pose": {
                "eye": [round(v, 3) for v in s.cam_eye],
                "yaw_degrees": round(math.degrees(s.cam_yaw) % 360, 1),
            },
            "bbox": bbox,
            "trellis_status": "pending",
        }
        s.found.append(obj)
        # Persist to disk so external UIs (Gradio) can sync the gallery.
        (s.explore_dir / "found_objects.json").write_text(
            json.dumps(s.found, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"[explorer] tagged {obj_id} '{label}' at "
              f"({pos3d[0]:.2f}, {pos3d[1]:.2f}, {pos3d[2]:.2f})", file=sys.stderr)
        return json.dumps({"status": "tagged", **obj}, indent=2, ensure_ascii=False)

    @mcp.tool()
    def find_best_angle(
        object_id: str,
        num_angles: int = 6,
        radius: float = 2.0,
    ) -> str:
        """Orbit around a found object to find the least-occluded view.

        Renders *num_angles* views around the object's 3D position, runs Falcon
        on each, and reports the mask area ratio for each angle.  The angle
        with the largest mask is the best view.

        Args:
            object_id: ID returned by ``tag_object``.
            num_angles: Number of orbital positions (default 6 = every 60°).
            radius: Orbit radius in metres (default 2.0).

        Returns:
            JSON with per-angle mask areas and the recommended best angle.
        """
        s = _req()
        obj = next((o for o in s.found if o["id"] == object_id), None)
        if obj is None:
            return json.dumps({"error": f"Object '{object_id}' not found."})

        client = _check_falcon(s)
        ox, oy, oz = obj["position_3d"]
        label = obj["label"]
        eye_y = s.cam_eye[1]  # keep current eye height

        results = []
        best_angle = None
        best_ratio = -1.0

        for i in range(num_angles):
            angle = 2 * math.pi * i / num_angles
            eye = [ox + radius * math.cos(angle), eye_y, oz + radius * math.sin(angle)]
            target = [ox, oy, oz]
            pose = look_at_pose(
                eye=tuple(eye), target=tuple(target), up=(0.0, 1.0, 0.0),
            )
            result = s.scene.render(pose, s.width, s.height, s.cam_fov)
            arr = np.clip(result.colors * 255, 0, 255).astype(np.uint8)
            img = PILImage.fromarray(arr, mode="RGB")
            png_path = _save_view(s,
                                  _encode_png(arr),
                                  suffix=f"_{label}_orbit{i}")
            dets = client.segment(img, label, task="segmentation")
            ratio = max((d.mask_area_ratio or 0) for d in dets) if dets else 0.0
            results.append({
                "angle_degrees": round(math.degrees(angle), 1),
                "eye": [round(v, 2) for v in eye],
                "mask_area_ratio": round(ratio, 4),
                "view": png_path,
            })
            if ratio > best_ratio:
                best_ratio = ratio
                best_angle = results[-1]

        # Update object's best view.
        if best_angle and best_ratio > 0:
            obj["best_view"] = best_angle["view"]
            obj["best_pose"] = {"eye": best_angle["eye"], "target": [ox, oy, oz]}

        return json.dumps({
            "object_id": object_id,
            "label": label,
            "angles_tested": num_angles,
            "results": results,
            "best_angle": best_angle,
        }, indent=2, ensure_ascii=False)

    # ── Queue ─────────────────────────────────────────────────────────

    @mcp.tool()
    def list_found() -> str:
        """List all found objects with their details and TRELLIS status."""
        s = _req()
        return json.dumps({
            "total": len(s.found),
            "objects": s.found,
        }, indent=2, ensure_ascii=False)

    @mcp.tool()
    def queue_for_trellis(object_id: str) -> str:
        """Queue an object's best view for TRELLIS mesh generation.

        Marks the object as ``queued``.  The pipeline's TRELLIS processor
        picks up queued objects and generates GLB meshes from their best views.

        Args:
            object_id: ID returned by ``tag_object``.

        Returns:
            JSON with the queue status.
        """
        s = _req()
        obj = next((o for o in s.found if o["id"] == object_id), None)
        if obj is None:
            return json.dumps({"error": f"Object '{object_id}' not found."})
        obj["trellis_status"] = "queued"
        # Write a queue entry for the pipeline to pick up.
        queue_dir = s.explore_dir / "trellis_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "object_id": obj["id"],
            "label": obj["label"],
            "image_path": obj["best_view"],
            "position_3d": obj["position_3d"],
        }
        (queue_dir / f"{obj['id']}.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        n_queued = sum(1 for o in s.found if o["trellis_status"] == "queued")
        print(f"[explorer] queued {object_id} for TRELLIS "
              f"({n_queued} total queued)", file=sys.stderr)
        return json.dumps({
            "status": "queued",
            "object_id": object_id,
            "label": obj["label"],
            "image_path": obj["best_view"],
            "total_queued": n_queued,
        }, indent=2, ensure_ascii=False)

    return mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="B-class Object Explorer MCP server")
    p.add_argument("--ply", default=os.environ.get("GS_PLY_PATH"),
                   help="Path to trained splat.ply.")
    p.add_argument("--feat", default=os.environ.get("GS_FEAT_PATH"),
                   help="Path to SceneSplat feat.pt (optional, for semantic queries).")
    p.add_argument("--falcon-host", default=os.environ.get("FALCON_HOST", "127.0.0.1"))
    p.add_argument("--falcon-port", type=int,
                   default=int(os.environ.get("FALCON_PORT", "8390")))
    p.add_argument("--explore-dir", default="output/explore",
                   help="Directory to save rendered views.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--fov", type=float, default=60.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not args.ply:
        print("ERROR: --ply is required (or set GS_PLY_PATH).", file=sys.stderr)
        return 2

    ply_path = Path(args.ply)
    if not ply_path.exists():
        print(f"ERROR: PLY not found: {ply_path}", file=sys.stderr)
        return 2

    # Don't load scene here — start MCP server immediately so the client
    # can connect. Scene loads lazily on first explore_init call.
    state = ExplorerState(
        scene=None,
        explore_dir=Path(args.explore_dir),
        width=args.width,
        height=args.height,
        cam_fov=args.fov,
        falcon_host=args.falcon_host,
        falcon_port=args.falcon_port,
    )
    state._ply_path = ply_path
    state._feat_path = args.feat or None
    print(f"MCP server ready. Scene loads on explore_init: {ply_path.name}", file=sys.stderr)

    mcp = build_server(state)
    mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
