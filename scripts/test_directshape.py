"""Test DirectShape insertion with an existing GLB file.

Loads a GLB, applies a simple placement transform, writes the payload to
a temp file, and sends it to Revit via the compiled create_directshape_from_mesh
MCP tool.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.mesh_registrar import MeshPlacement, compute_placement_transform
from bim_recon.mcp_gateway import StdioMCPGateway
from bim_recon.config import load_config


async def main():
    glb = ROOT / "output/splat/_trellis_meshes/the light-colored armchair with a blue cushion on the right_1784522514.glb"
    if not glb.exists():
        print(f"GLB not found: {glb}")
        return 1

    placement = MeshPlacement(
        glb_path=glb,
        world_x=0.0, world_y=0.0,
        floor_z=0.0, ceiling_z=3.0,
        element_width_m=0.6, element_height_m=0.8,
        up_axis=2,
        category="OST_GenericModel",
        name="Test Chair",
    )
    transform = compute_placement_transform(placement)
    print(f"Vertices: {transform.vertices_world.shape}, Faces: {transform.faces.shape}")

    meters_to_feet = 3.280839895013123
    vertices_feet = (transform.vertices_world * meters_to_feet).round(6)
    payload = {
        "name": placement.name,
        "category": placement.category,
        "vertices": vertices_feet.flatten().tolist(),
        "faces": transform.faces.flatten().tolist(),
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(payload, tmp)
        payload_path = tmp.name
    print(f"Payload file: {payload_path}")

    cfg = load_config()
    gateway = StdioMCPGateway(
        command=cfg.revit_mcp.command,
        args=tuple(cfg.revit_mcp.args),
        cwd=str(ROOT),
        timeout_seconds=300.0,
    )

    print("Calling create_directshape_from_mesh MCP tool...")
    try:
        resp = await gateway.call_tool(
            "create_directshape_from_mesh",
            {"meshFile": payload_path},
        )
        print("Response:", json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
