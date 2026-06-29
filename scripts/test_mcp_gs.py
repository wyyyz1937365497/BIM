"""Integration test for the 3DGS MCP server tools.

Calls each of the 5 tool functions directly (bypassing stdio) against a demo
synthetic scene. This verifies the MCP wiring without needing a real PLY.

Run with:
    python scripts/test_mcp_gs.py
Must be run with MSVC in PATH (gsplat JIT-compiles CUDA on first run).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure the project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.mcp_gs import ServerState, _build_demo_scene, build_server
from mcp.server.fastmcp.utilities.types import Image as MCPImage


async def main_async() -> int:
    print("[1/5] Loading demo scene...")
    scene = _build_demo_scene()
    print(f"      {scene.num_gaussians} Gaussians loaded")

    state = ServerState(scene=scene, cameras=[], default_width=400, default_height=300)
    mcp = build_server(state)

    tool_mgr = mcp._tool_manager
    tool_names = {t.name for t in tool_mgr.list_tools()}
    print(f"      registered tools: {sorted(tool_names)}")
    expected = {"get_scene_info", "list_cameras", "render_from_pose", "get_depth_grid", "select_cluster"}
    missing = expected - tool_names
    if missing:
        print(f"ERROR: missing tools: {missing}")
        return 1

    # --- 2. get_scene_info ---
    print("\n[2/5] get_scene_info")
    info = await tool_mgr.call_tool("get_scene_info", {})
    info_dict = json.loads(_extract_text(info))
    print(f"      gaussians: {info_dict['num_gaussians']}")
    print(f"      bounds:    {info_dict['bounds_min']} -> {info_dict['bounds_max']}")
    print(f"      extent:    {info_dict['extent']:.2f}")
    print(f"      default eye: {info_dict['default_camera']['eye']}")

    # --- 3. list_cameras ---
    print("\n[3/5] list_cameras")
    cams = await tool_mgr.call_tool("list_cameras", {})
    cams_dict = json.loads(_extract_text(cams))
    print(f"      camera count: {len(cams_dict['cameras'])}  (expected 0, demo has no training cameras)")

    # Camera inside the room looking toward +z wall
    eye = [0.0, 1.5, 0.0]
    target = [0.0, 1.5, 3.0]

    # --- 4. render_from_pose ---
    print("\n[4/5] render_from_pose")
    render_result = await tool_mgr.call_tool("render_from_pose", {
        "eye": eye, "target": target, "width": 400, "height": 300, "fov_degrees": 60.0,
    })
    img_bytes = _extract_image(render_result)
    print(f"      rendered PNG: {len(img_bytes)} bytes")
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n", "not a valid PNG"
    print("      PNG header OK")

    # --- 5. get_depth_grid ---
    print("\n[5/5] get_depth_grid")
    depth_result = await tool_mgr.call_tool("get_depth_grid", {
        "eye": eye, "target": target, "width": 200, "height": 150, "stride": 15, "fov_degrees": 60.0,
    })
    depth_data = json.loads(_extract_text(depth_result))
    stats = depth_data["stats"]
    print(f"      depth stats: min={stats['min']:.2f} max={stats['max']:.2f} mean={stats['mean']:.2f}")
    print(f"      rendered fraction: {stats['rendered_fraction']:.2%}")
    grid_rows = len(depth_data["grid"])
    grid_cols = len(depth_data["grid"][0]) if grid_rows else 0
    print(f"      grid shape: {grid_rows}x{grid_cols}")
    assert stats["rendered_fraction"] > 0.1, "expected the wall to be visible"

    # --- 6. select_cluster ---
    print("\n[BONUS] select_cluster (center bbox)")
    cluster_result = await tool_mgr.call_tool("select_cluster", {
        "eye": eye, "target": target,
        "bbox_xyxy": [150, 100, 250, 200],
        "width": 400, "height": 300, "fov_degrees": 60.0,
    })
    cluster = json.loads(_extract_text(cluster_result))
    print(f"      selected Gaussians: {cluster['num_gaussians']}")
    if cluster["centroid"]:
        print(f"      centroid: {[f'{v:.2f}' for v in cluster['centroid']]}")
        print(f"      bounds:   {cluster['bounds_min']} -> {cluster['bounds_max']}")
    assert cluster["num_gaussians"] > 0, "expected to hit the +z wall"

    print("\nALL TOOLS OK")
    return 0


def _extract_text(result) -> str:
    """Unwrap FastMCP tool return (convert_result=False default).

    Our tools return either str (text tools) or MCPImage (render tool).
    """
    if isinstance(result, str):
        return result
    return str(result)


def _extract_image(result) -> bytes:
    """Unwrap an MCPImage into raw PNG bytes."""
    if isinstance(result, MCPImage):
        if result.data is not None:
            return result.data
        if result.path is not None:
            return Path(result.path).read_bytes()
    raise ValueError(f"expected MCPImage, got {type(result)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
