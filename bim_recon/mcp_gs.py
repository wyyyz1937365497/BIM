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

from bim_recon.floorplan import ManualProvider
from bim_recon.floorplan_registration import register_floorplan
from bim_recon.gs_scene import CameraPose, GSScene, look_at_pose
from bim_recon.wall_fitter import FloorPlanGuidedFitter, WallFitter


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
    has_semantics: bool = False


_STATE: Optional[ServerState] = None


# Distinct color per BIM class for render_semantic_overlay (global mode).
# Order matches data/bim_class_names.txt.
_BIM_PALETTE: List[Tuple[float, float, float]] = [
    (0.80, 0.80, 0.80),  # wall      — gray
    (0.60, 0.40, 0.30),  # floor     — brown
    (0.95, 0.95, 0.90),  # ceiling   — off-white
    (0.85, 0.25, 0.25),  # door      — red
    (0.25, 0.60, 0.85),  # window    — blue
    (0.50, 0.50, 0.55),  # column    — dark gray
    (0.30, 0.30, 0.35),  # beam      — darker gray
    (0.80, 0.75, 0.25),  # stairs    — yellow-ish
    (0.25, 0.75, 0.35),  # furniture — green
]


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
        text_query: Optional[str] = None,
        up: Optional[List[float]] = None,
        fov_degrees: float = 60.0,
        width: int = 800,
        height: int = 600,
    ) -> str:
        """Select Gaussians whose projection falls inside a 2D bounding box.

        Args:
            eye, target, up, fov_degrees, width, height: camera parameters.
            bbox_xyxy: [x_min, y_min, x_max, y_max] in image pixels.
            text_query: optional semantic filter (e.g. "wall") — only Gaussians
              whose dominant class matches are kept. Requires semantic features.

        Returns JSON with:
          - ``num_gaussians``: count of selected Gaussians.
          - ``centroid``: mean world position of selected Gaussians.
          - ``bounds_min``, ``bounds_max``: AABB of selected Gaussians.
          - ``indices``: first 50 original Gaussian indices (for debugging).
          - ``semantic_filter``: the text_query used, or null.

        This bridges a 2D VLM region selection to the 3D Gaussian cloud,
        enabling downstream wall fitting / plane segmentation.
        """
        s = _require_state()
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must have 4 ints [x_min, y_min, x_max, y_max]")
        if text_query is not None and not s.has_semantics:
            raise RuntimeError(
                "text_query requires semantic features. "
                "Start the server with --feat --text-emb --class-names."
            )
        x0, y0, x1, y1 = bbox_xyxy
        pose = _parse_camera(eye, target, up)
        mask = np.zeros((height, width), dtype=bool)
        x0c, x1c = max(0, min(x0, x1)), min(width - 1, max(x0, x1))
        y0c, y1c = max(0, min(y0, y1)), min(height - 1, max(y0, y1))
        mask[y0c : y1c + 1, x0c : x1c + 1] = True
        ids = s.scene.select_by_mask(
            pose=pose, mask=mask, width=width, height=height,
            fov_degrees=fov_degrees,
            text_filter=text_query,
        )
        if ids.shape[0] == 0:
            return json.dumps({
                "num_gaussians": 0, "centroid": None,
                "bounds_min": None, "bounds_max": None,
                "indices": [], "semantic_filter": text_query,
            })
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
            "semantic_filter": text_query,
        }, indent=2)

    @mcp.tool()
    def query_semantics(
        text_query: str,
        mode: str = "dominant",
        threshold: float = 0.52,
        percent: float = 10.0,
    ) -> str:
        """Find Gaussians matching a semantic text label (e.g., 'wall', 'door').

        ``mode`` selects the query strategy:

        - ``"dominant"`` *(default)*: Gaussians whose argmax class is *text_query*.
          Most reliable for SceneSplat features because cosine similarities
          cluster tightly (~0.1 ± 0.015), making absolute thresholds fragile.
        - ``"threshold"``: sigmoid probability for *text_query* exceeds *threshold*.
        - ``"top_percent"``: top *percent* % Gaussians by probability.

        Returns JSON: class, num_gaussians, mean_confidence, centroid,
        bounds_min, bounds_max, indices (first 100).

        Requires semantic features (--feat --text-emb --class-names).
        """
        s = _require_state()
        if not s.has_semantics:
            raise RuntimeError(
                "query_semantics requires semantic features. "
                "Start the server with --feat --text-emb --class-names."
            )
        result = s.scene.query_semantics(
            text_query, mode=mode, threshold=threshold, percent=percent,
        )
        idx_arr = result["indices"]
        conf_arr = result["confidence"]
        result["indices"] = idx_arr[:100].tolist()
        result["confidence"] = conf_arr[:100].tolist()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def render_semantic_overlay(
        eye: List[float],
        target: List[float],
        text_query: Optional[str] = None,
        up: Optional[List[float]] = None,
        fov_degrees: float = 60.0,
        width: int = 800,
        height: int = 600,
    ) -> MCPImage:
        """Render the 3DGS scene from a viewpoint with semantic coloring.

        If *text_query* given: matching Gaussians → red, others → cyan.
        If *text_query* is None: each Gaussian colored by its dominant class
        using a fixed BIM palette (see _BIM_PALETTE).

        Requires semantic features (--feat --text-emb --class-names).

        Returns: PNG image (HxWx3, 8-bit).
        """
        s = _require_state()
        if not s.has_semantics:
            raise RuntimeError(
                "render_semantic_overlay requires semantic features. "
                "Start the server with --feat --text-emb --class-names."
            )
        scene = s.scene
        querier = scene.semantic_querier
        if querier is None:  # has_semantics guarantees non-None; guard for type safety
            raise RuntimeError("semantic_querier is None despite has_semantics=True")
        N = scene.num_gaussians
        dev = scene.device
        pose = _parse_camera(eye, target, up)

        orig_colors = scene.colors
        try:
            if text_query is not None:
                # Highlight mode: matching → red, others → cyan
                match = querier.query_dominant(text_query)
                overlay = torch.zeros((N, 3), dtype=torch.float32, device=dev)
                overlay[:, 1] = 1.0  # cyan baseline = (0,1,1)
                overlay[:, 2] = 1.0
                if match["num_gaussians"] > 0:
                    match_idx = torch.as_tensor(
                        match["indices"], device=dev, dtype=torch.long,
                    )
                    overlay[match_idx] = torch.tensor(
                        [1.0, 0.0, 0.0], device=dev,
                    )
            else:
                # Global mode: color by dominant class via palette
                dominant = querier.get_dominant_labels()  # (N,) int32 numpy
                overlay = torch.zeros((N, 3), dtype=torch.float32, device=dev)
                for c_idx, color in enumerate(_BIM_PALETTE):
                    if c_idx >= querier.num_classes:
                        break
                    mask_c = torch.as_tensor(dominant == c_idx, device=dev)
                    overlay[mask_c] = torch.tensor(color, device=dev)
            scene.colors = overlay
            result = scene.render(pose, width=width, height=height, fov_degrees=fov_degrees)
        finally:
            scene.colors = orig_colors

        png = _png_from_rgb(result.colors)
        return MCPImage(data=png, format="png")

    @mcp.tool()
    def fit_walls(
        text_query: str = "wall",
        mode: str = "dominant",
        up_axis: Optional[int] = None,
    ) -> str:
        """Fit wall segments from semantic Gaussian clusters.

        Queries Gaussians by ``text_query``, runs iterative RANSAC +
        occlusion-gap bridging + gravity alignment + endpoint refinement,
        returns wall segments as JSON.

        Each wall has: p0, p1 (3D endpoints at floor level, meters),
        height, normal, thickness, length, num_inliers, confidence.

        ``up_axis`` auto-detected from floor centroid if None (0=x,1=y,2=z).
        Wall height computed from floor→ceiling centroid distance.

        Requires semantic features (--feat --text-emb --class-names).
        """
        s = _require_state()
        if not s.has_semantics:
            raise RuntimeError(
                "fit_walls requires semantic features. "
                "Start the server with --feat --text-emb --class-names."
            )
        scene = s.scene

        # Auto-detect up_axis from floor centroid (lowest axis = up)
        floor_result = scene.query_semantics("floor", mode="dominant")
        if up_axis is None:
            if floor_result["centroid"] is not None:
                up_axis = int(np.argmin(floor_result["centroid"]))
            else:
                up_axis = 2

        # Get wall Gaussian means
        wall_result = scene.query_semantics(text_query, mode=mode)
        wall_indices = wall_result["indices"]
        if len(wall_indices) == 0:
            return json.dumps({"walls": [], "up_axis": up_axis, "num_walls": 0})

        wall_means = scene.means[
            torch.as_tensor(wall_indices, dtype=torch.long)
        ].cpu().numpy().astype(np.float64)

        # Floor/ceiling z for height extraction
        floor_z = float(floor_result["centroid"][up_axis]) if floor_result["centroid"] else None
        ceiling_result = scene.query_semantics("ceiling", mode="dominant")
        ceiling_z = float(ceiling_result["centroid"][up_axis]) if ceiling_result["centroid"] else None

        # Fit walls
        fitter = WallFitter()
        walls = fitter.fit(
            wall_means, up_axis=up_axis, floor_z=floor_z, ceiling_z=ceiling_z,
        )

        return json.dumps({
            "walls": [w.to_dict() for w in walls],
            "up_axis": up_axis,
            "num_walls": len(walls),
        }, indent=2)

    @mcp.tool()
    def fit_walls_guided(
        floorplan_json: str,
        text_query: str = "wall",
        mode: str = "dominant",
        up_axis: Optional[int] = None,
        corridor_width: float = 0.5,
    ) -> str:
        """Fit walls constrained by a 2D floorplan.

        Takes a floorplan JSON (ManualProvider format) and auto-registers it
        to the 3DGS scene. For each floorplan wall segment, only wall Gaussians
        within ``corridor_width`` meters of that line are kept, then a single
        RANSAC plane is fit and checked against the expected wall normal.

        The floorplan JSON can be a rectangle::

            {"rectangle": {"width": 5.0, "depth": 4.0}}

        or an explicit wall list::

            {"walls": [{"x1": 0, "y1": 0, "x2": 5, "y2": 0}, ...]}

        Requires semantic features (--feat --text-emb --class-names).
        """
        s = _require_state()
        if not s.has_semantics:
            raise RuntimeError(
                "fit_walls_guided requires semantic features. "
                "Start the server with --feat --text-emb --class-names."
            )
        scene = s.scene

        # Parse floorplan
        try:
            data = json.loads(floorplan_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid floorplan_json: {e}")
        floorplan = ManualProvider.from_dict(data).get_floorplan()

        # Auto-detect up_axis from floor centroid
        floor_result = scene.query_semantics("floor", mode="dominant")
        if up_axis is None:
            if floor_result["centroid"] is not None:
                up_axis = int(np.argmin(floor_result["centroid"]))
            else:
                up_axis = 2

        # Get wall/floor/ceiling Gaussian means
        wall_result = scene.query_semantics(text_query, mode=mode)
        wall_indices = wall_result["indices"]
        if len(wall_indices) == 0:
            return json.dumps({"walls": [], "up_axis": up_axis, "num_walls": 0})

        wall_means = scene.means[
            torch.as_tensor(wall_indices, dtype=torch.long)
        ].cpu().numpy().astype(np.float64)

        floor_z = float(floor_result["centroid"][up_axis]) if floor_result["centroid"] else None
        ceiling_result = scene.query_semantics("ceiling", mode="dominant")
        ceiling_z = float(ceiling_result["centroid"][up_axis]) if ceiling_result["centroid"] else None

        # Auto-register floorplan to the 3DGS horizontal plane
        h_axes = [i for i in range(3) if i != up_axis]
        wall_means_2d = wall_means[:, h_axes]

        floor_indices = floor_result["indices"]
        floor_means = scene.means[
            torch.as_tensor(floor_indices, dtype=torch.long)
        ].cpu().numpy().astype(np.float64)
        floor_means_2d = floor_means[:, h_axes]

        registered_floorplan = register_floorplan(
            floorplan,
            wall_means_2d,
            floor_means_2d=floor_means_2d,
            corridor_width=corridor_width,
        )

        # Guided fit
        fitter = FloorPlanGuidedFitter(corridor_width=corridor_width)
        walls = fitter.fit_guided(
            wall_means,
            registered_floorplan,
            up_axis=up_axis,
            floor_z=floor_z,
            ceiling_z=ceiling_z,
        )

        return json.dumps({
            "walls": [w.to_dict() for w in walls],
            "up_axis": up_axis,
            "num_walls": len(walls),
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
    p.add_argument("--feat", required=False, default=os.environ.get("GS_FEAT_PATH"),
                   help="Path to SceneSplat feat.pt (N,768) per-Gaussian language features.")
    p.add_argument("--text-emb", required=False, default=os.environ.get("GS_TEXT_EMB_PATH"),
                   help="Path to bim_text_emb.pt (C,768) SigLIP2 text embeddings.")
    p.add_argument("--class-names", required=False, default=os.environ.get("GS_CLASS_NAMES_PATH"),
                   help="Path to bim_class_names.json {class_name: index}.")
    p.add_argument("--data-dir", required=False, default=os.environ.get("GS_DATA_DIR"),
                   help="Path to SceneSplat .npy directory (coord/color/opacity/scale/quat.npy). "
                        "Use this instead of --ply for SceneSplat preprocessed data.")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--demo", action="store_true",
                   help="Load a synthetic demo scene (no PLY needed). Useful for testing the MCP wiring.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Determine whether semantic features are fully specified.
    sem_args = [args.feat, args.text_emb, args.class_names]
    has_semantics = all(sem_args)
    if any(sem_args) and not has_semantics:
        print("ERROR: --feat, --text-emb, and --class-names must be provided together "
              "(got only some).", file=sys.stderr)
        return 2

    if args.demo:
        scene = _build_demo_scene()
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"ERROR: data-dir not found: {data_dir}", file=sys.stderr)
            return 2
        print(f"Loading SceneSplat .npy data: {data_dir}", file=sys.stderr)
        scene = GSScene.from_npy(
            data_dir,
            feat_path=args.feat,
            text_emb_path=args.text_emb,
            class_names_path=args.class_names,
        )
        print(f"Loaded {scene.num_gaussians} Gaussians", file=sys.stderr)
    else:
        if not args.ply:
            print("ERROR: --ply (or --data-dir, or --demo) is required "
                  "(or set GS_PLY_PATH).", file=sys.stderr)
            return 2
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"ERROR: PLY file not found: {ply_path}", file=sys.stderr)
            return 2
        print(f"Loading PLY: {ply_path}", file=sys.stderr)
        scene = GSScene.from_ply(
            ply_path,
            feat_path=args.feat,
            text_emb_path=args.text_emb,
            class_names_path=args.class_names,
        )
        print(f"Loaded {scene.num_gaussians} Gaussians", file=sys.stderr)

    cameras: List[Dict[str, Any]] = []
    if args.cameras:
        cam_path = Path(args.cameras)
        if cam_path.exists():
            cameras = _load_transforms_json(cam_path)
            print(f"Loaded {len(cameras)} training cameras", file=sys.stderr)

    if has_semantics and scene.semantic_querier is not None and scene.feat is not None:
        print(f"Semantic features enabled: {scene.semantic_querier.num_classes} classes, "
              f"{scene.feat.shape[1]}-dim features", file=sys.stderr)

    state = ServerState(
        scene=scene,
        cameras=cameras,
        default_width=args.width,
        default_height=args.height,
        default_fov=args.fov,
        has_semantics=has_semantics,
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
