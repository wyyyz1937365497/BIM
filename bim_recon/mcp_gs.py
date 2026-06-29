"""MCP server exposing a trained 3DGS scene to VLM agents.

Provides five tools so an LLM can "look around" a 3DGS scene and select
Gaussians for downstream BIM reconstruction:

  get_scene_info   — bounds, point count, configured default camera.
  list_cameras     — list the training camera poses (from transforms.json).
  render_from_pose — render RGB (PNG) + depth summary from any viewpoint.
  get_depth_grid   — render and return the depth map as a compact JSON grid.
  select_cluster   — pick Gaussians inside a 2D bounding box from a viewpoint.

Run with:
    python -m bim_recon.mcp_gs --ply path/to/splat.ply [--cameras transforms.json]

The server speaks MCP over stdio. Add it to opencode.json / Claude Desktop
config under ``mcp`` with the command above.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage

from bim_recon.gs_scene import CameraPose, GSScene, look_at_pose


# ---------------------------------------------------------------------------
# Global state — populated in main() before the server starts.
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Holds the loaded scene and optional camera list.

    MCP tool functions close over this singleton via the ``_STATE`` module
    global so they stay stateless from the protocol's perspective.
    """

    scene: GSScene
    cameras: List[Dict[str, Any]]  # each: {"name", "eye", "target", "up", "fov_degrees"}
    default_width: int = 800
    default_height: int = 600
    default_fov: float = 60.0


_STATE: Optional[ServerState] = None


def _require_state() -> ServerState:
    if _STATE is None:
        raise RuntimeError(
            "3DGS MCP server state not initialized. "
            "Start the server with --ply <path>."
        )
    return _STATE


# ---------------------------------------------------------------------------
# transforms.json parsing (nerfstudio / COLMAP convention)
# ---------------------------------------------------------------------------


def _load_transforms_json(path: Path) -> List[Dict[str, Any]]:
    """Parse a nerfstudio transforms.json into a list of camera descriptors.

    Each frame's ``transform_matrix`` is a 4x4 camera-to-world matrix in the
    nerfstudio / OpenCV convention (+x right, +y down, +z forward). We extract
    eye/target/up as a look-at descriptor for tool-friendliness.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    # Default FOV: prefer the top-level camera_angle_x if present.
    angle_x = data.get("camera_angle_x")
    default_fov = math.degrees(angle_x) if angle_x else 60.0
    out: List[Dict[str, Any]] = []
    for i, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"], dtype=np.float32).reshape(4, 4)
        eye = c2w[:3, 3].tolist()
        # Forward axis (+z in OpenCV c2w)
        forward = c2w[:3, 2]
        target = (c2w[:3, 3] + forward).tolist()
        # World up — fixed to +y unless camera appears upside-down (rare).
        up = [0.0, 1.0, 0.0]
        out.append({
            "name": frame.get("file_path", f"frame_{i:04d}"),
            "eye": [float(v) for v in eye],
            "target": [float(v) for v in target],
            "up": up,
            "fov_degrees": float(frame.get("camera_angle_x", math.radians(default_fov))
                                  if "camera_angle_x" in frame
                                  else default_fov),
        })
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_from_rgb(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 float32 [0,1] array as PNG bytes."""
    arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _parse_camera(
    eye: List[float],
    target: List[float],
    up: Optional[List[float]] = None,
) -> CameraPose:
    """Build a CameraPose from look-at parameters."""
    return look_at_pose(
        eye=(float(eye[0]), float(eye[1]), float(eye[2])),
        target=(float(target[0]), float(target[1]), float(target[2])),
        up=(0.0, 1.0, 0.0) if up is None else (float(up[0]), float(up[1]), float(up[2])),
    )


# ---------------------------------------------------------------------------
# MCP server construction
# ---------------------------------------------------------------------------


def build_server(state: ServerState) -> FastMCP:
    """Construct the FastMCP server with all tools registered."""
    global _STATE
    _STATE = state

    mcp = FastMCP("bim-recon-gs")

    @mcp.tool()
    def get_scene_info() -> str:
        """Return scene metadata: Gaussian count, world bounds, and default camera.

        Call this first to understand the scale of the scene before rendering.
        Coordinates are in the SfM / nerfstudio world frame (metres if the SfM
        model was metrically aligned).
        """
        s = _require_state()
        mn, mx = s.scene.scene_bounds()
        center = ((mn + mx) / 2.0).tolist()
        # Default camera: step back from the bounding-box center along -z.
        extent = float(np.max(mx - mn))
        eye_back = (center[0], center[1], center[2] - extent * 1.2)
        info = {
            "num_gaussians": s.scene.num_gaussians,
            "bounds_min": mn.tolist(),
            "bounds_max": mx.tolist(),
            "bounds_center": center,
            "extent": extent,
            "default_camera": {
                "eye": eye_back,
                "target": center,
                "up": [0.0, 1.0, 0.0],
                "fov_degrees": s.default_fov,
                "width": s.default_width,
                "height": s.default_height,
            },
            "num_training_cameras": len(s.cameras),
        }
        return json.dumps(info, indent=2)

    @mcp.tool()
    def list_cameras() -> str:
        """List the original training camera poses (viewpoints).

        Use these to render the scene from the same viewpoints used during 3DGS
        training — guaranteed good coverage and accurate geometry.
        """
        s = _require_state()
        return json.dumps({"cameras": s.cameras}, indent=2)

    @mcp.tool()
    def render_from_pose(
        eye: List[float],
        target: List[float],
        up: Optional[List[float]] = None,
        fov_degrees: float = 60.0,
        width: int = 800,
        height: int = 600,
    ) -> MCPImage:
        """Render the 3DGS scene from a camera pose and return an RGB PNG.

        The camera looks from ``eye`` toward ``target``. ``up`` defaults to
        world +y (vertical). Depth at the image center and rendered-pixel
        coverage is logged; for the full depth map use ``get_depth_grid``.

        Returns: PNG image (HxWx3, 8-bit).
        """
        s = _require_state()
        pose = _parse_camera(eye, target, up)
        result = s.scene.render(pose, width=width, height=height, fov_degrees=fov_degrees)
        png = _png_from_rgb(result.colors)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def get_depth_grid(
        eye: List[float],
        target: List[float],
        up: Optional[List[float]] = None,
        fov_degrees: float = 60.0,
        width: int = 200,
        height: int = 150,
        stride: int = 10,
    ) -> str:
        """Render depth from a viewpoint and return a downsampled depth grid.

        Returns a JSON object with:
          - ``stride``: pixel step between grid samples.
          - ``grid``: 2D list of metric depths (0 = no geometry / background).
          - ``stats``: min/max/mean depth over rendered pixels.
        Use this to reason about wall distances and room dimensions without
        pulling a full-resolution depth map through the context window.
        """
        s = _require_state()
        pose = _parse_camera(eye, target, up)
        result = s.scene.render(pose, width=width, height=height, fov_degrees=fov_degrees)
        depth = result.depth
        alpha = result.alpha
        rendered = depth[alpha > 1e-3]
        if rendered.size > 0:
            stats = {
                "min": float(rendered.min()),
                "max": float(rendered.max()),
                "mean": float(rendered.mean()),
                "rendered_fraction": float((alpha > 1e-3).mean()),
            }
        else:
            stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "rendered_fraction": 0.0}
        # Downsample by stride
        grid = depth[::stride, ::stride].tolist()
        return json.dumps({"stride": stride, "grid": grid, "stats": stats}, indent=2)

    @mcp.tool()
    def select_cluster(
        eye: List[float],
        target: List[float],
        bbox_xyxy: List[int],
        up: Optional[List[float]] = None,
        fov_degrees: float = 60.0,
        width: int = 800,
        height: int = 600,
    ) -> str:
        """Select Gaussians whose projection falls inside a 2D bounding box.

        Args:
            eye, target, up, fov_degrees, width, height: camera parameters.
            bbox_xyxy: [x_min, y_min, x_max, y_max] in image pixels.

        Returns JSON with:
          - ``num_gaussians``: count of selected Gaussians.
          - ``centroid``: mean world position of selected Gaussians.
          - ``bounds_min``, ``bounds_max``: AABB of selected Gaussians.
          - ``indices``: first 50 original Gaussian indices (for debugging).

        This bridges a 2D VLM region selection to the 3D Gaussian cloud,
        enabling downstream wall fitting / plane segmentation.
        """
        s = _require_state()
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must have 4 ints [x_min, y_min, x_max, y_max]")
        x0, y0, x1, y1 = bbox_xyxy
        pose = _parse_camera(eye, target, up)
        mask = np.zeros((height, width), dtype=bool)
        x0c, x1c = max(0, min(x0, x1)), min(width - 1, max(x0, x1))
        y0c, y1c = max(0, min(y0, y1)), min(height - 1, max(y0, y1))
        mask[y0c : y1c + 1, x0c : x1c + 1] = True
        ids = s.scene.select_by_mask(pose=pose, mask=mask, width=width, height=height, fov_degrees=fov_degrees)
        if ids.shape[0] == 0:
            return json.dumps({"num_gaussians": 0, "centroid": None, "bounds_min": None, "bounds_max": None, "indices": []})
        selected_means = s.scene.means[ids].cpu().numpy()
        centroid = selected_means.mean(axis=0).tolist()
        bounds_min = selected_means.min(axis=0).tolist()
        bounds_max = selected_means.max(axis=0).tolist()
        return json.dumps({
            "num_gaussians": int(ids.shape[0]),
            "centroid": centroid,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "indices": ids[:50].tolist(),
        }, indent=2)

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3DGS MCP server (stdio)")
    p.add_argument("--ply", required=False, default=os.environ.get("GS_PLY_PATH"),
                   help="Path to the trained splat.ply (nerfstudio export).")
    p.add_argument("--cameras", required=False, default=os.environ.get("GS_CAMERAS_JSON"),
                   help="Optional path to transforms.json for camera list.")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--demo", action="store_true",
                   help="Load a synthetic demo scene (no PLY needed). Useful for testing the MCP wiring.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.demo:
        scene = _build_demo_scene()
    else:
        if not args.ply:
            print("ERROR: --ply is required (or set GS_PLY_PATH, or use --demo)", file=sys.stderr)
            return 2
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"ERROR: PLY file not found: {ply_path}", file=sys.stderr)
            return 2
        print(f"Loading PLY: {ply_path}", file=sys.stderr)
        scene = GSScene.from_ply(ply_path)
        print(f"Loaded {scene.num_gaussians} Gaussians", file=sys.stderr)

    cameras: List[Dict[str, Any]] = []
    if args.cameras:
        cam_path = Path(args.cameras)
        if cam_path.exists():
            cameras = _load_transforms_json(cam_path)
            print(f"Loaded {len(cameras)} training cameras", file=sys.stderr)

    state = ServerState(
        scene=scene,
        cameras=cameras,
        default_width=args.width,
        default_height=args.height,
        default_fov=args.fov,
    )
    mcp = build_server(state)
    mcp.run()
    return 0


def _build_demo_scene() -> GSScene:
    """A small synthetic room-like scene for MCP wiring tests."""
    # 4 walls + floor + ceiling as flat Gaussian patches
    rng = np.random.default_rng(42)
    patches: List[np.ndarray] = []
    colors: List[Tuple[float, float, float]] = []
    # Floor (y=0) and ceiling (y=3), each 6x6 in x,z
    for y_plane, c in [(0.0, (0.6, 0.4, 0.3)), (3.0, (0.8, 0.8, 0.8))]:
        xs = rng.uniform(-3, 3, 200)
        zs = rng.uniform(-3, 3, 200)
        patches.append(np.stack([xs, np.full(200, y_plane), zs], axis=-1))
        colors.extend([c] * 200)
    # 4 walls
    wall_specs = [
        (-3.0, 0, (0.9, 0.9, 0.9)),  # x=-3 wall, varying y,z
        ( 3.0, 0, (0.9, 0.9, 0.9)),  # x=3 wall
    ]
    for fixed_x, _axis, c in wall_specs:
        ys = rng.uniform(0, 3, 200)
        zs = rng.uniform(-3, 3, 200)
        patches.append(np.stack([np.full(200, fixed_x), ys, zs], axis=-1))
        colors.extend([c] * 200)
    # z=-3 wall and z=3 wall
    for fixed_z, c in [(-3.0, (0.7, 0.7, 0.85)), (3.0, (0.7, 0.85, 0.7))]:
        xs = rng.uniform(-3, 3, 200)
        ys = rng.uniform(0, 3, 200)
        patches.append(np.stack([xs, ys, np.full(200, fixed_z)], axis=-1))
        colors.extend([c] * 200)
    means = np.concatenate(patches, axis=0).astype(np.float32)
    colors_arr = np.array(colors, dtype=np.float32)
    return GSScene.from_synthetic(means=means, colors_rgb=colors_arr,
                                   scales=np.full((means.shape[0], 3), 0.05, dtype=np.float32))


if __name__ == "__main__":
    sys.exit(main())
