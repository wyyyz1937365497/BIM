"""Smoke test the focused TRELLIS registration workflow without TRELLIS inference."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bim_recon.trellis_registration import RegistrationInputs, register_mesh


def write_cube_glb(path: Path) -> None:
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5], [2, 6, 7], [2, 7, 3],
        [1, 5, 6], [1, 6, 2], [0, 3, 7], [0, 7, 4],
    ], dtype=np.uint16)
    vertex_bytes = vertices.tobytes()
    face_bytes = faces.tobytes()
    binary = vertex_bytes + face_bytes
    while len(binary) % 4:
        binary += b"\0"
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 0, "componentType": 5123, "count": 36, "type": "SCALAR", "byteOffset": len(vertex_bytes)},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
    }
    json_bytes = json.dumps(gltf).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)


def main() -> int:
    root = Path("output/trellis_registration_smoke")
    root.mkdir(parents=True, exist_ok=True)
    glb = root / "cube.glb"
    cutout = root / "cube_cutout.png"
    write_cube_glb(glb)
    image = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((280, 200, 520, 600), fill=(180, 120, 70, 255))
    image.save(cutout)
    manifest = register_mesh(
        glb,
        cutout,
        root / "result",
        RegistrationInputs(
            world_position=(0.0, 0.0, 0.8),
            floor_z=0.0,
            ceiling_z=3.0,
            element_width_m=0.8,
            element_height_m=1.0,
            camera_eye=(0.0, -3.0, 0.8),
            camera_target=(0.0, 0.0, 0.8),
            camera_fov_deg=45.0,
            image_size=(800, 800),
            bbox=(0.5, 0.5, 0.3, 0.5),
        ),
        name="cube_smoke",
    )
    assert manifest["schema_version"] == 1
    assert manifest["registration"]["yaw_search"]["method"] == "silhouette_iou"
    assert Path(manifest["manifest_path"]).is_file()
    assert (root / "result" / "yaw_debug" / "overlay_best.png").is_file()
    print("TRELLIS_REGISTRATION_SMOKE_OK")
    print(manifest["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
