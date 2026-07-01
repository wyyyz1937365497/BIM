"""Render clean verification images for door candidates + VLM verify via Ollama.

Uses original 3DGS weights (point_cloud_*.ply) for correct colors,
renders targeted viewpoints from polar coordinates, then queries
Ollama VLM to confirm/reject each candidate.

Run with vcvars64:
    cmd /c "...\\vcvars64.bat && python scripts/verify_doors.py --name room0"
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.gs_scene import GSScene, CameraPose, look_at_pose


def candidate_to_viewpoint(
    world_x: float, world_y: float,
    h_min: float, h_max: float,
    scan_center: tuple[float, float],
    floor_z: float,
    eye_height: float = 1.5,
) -> tuple[list[float], list[float], float, float]:
    """Map a candidate's polar position to a camera pose.

    Returns (eye, target, theta_deg, distance_m).
    """
    cx, cy = scan_center
    dx = world_x - cx
    dy = world_y - cy
    r = math.sqrt(dx * dx + dy * dy)

    eye = [cx, cy, floor_z + eye_height]
    h_mid = (h_min + h_max) / 2.0
    target = [world_x, world_y, floor_z + h_mid]

    theta = math.degrees(math.atan2(dy, dx)) % 360
    return eye, target, theta, r


def save_clean_image(colors: np.ndarray, path: str) -> None:
    """Save RGB array as a clean PNG (no overlay)."""
    from PIL import Image
    img = Image.fromarray((colors * 255).clip(0, 255).astype(np.uint8))
    img.save(path)


def query_ollama(image_path: str, prompt: str, model: str) -> str:
    """Send image to Ollama VLM and get response."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    return result.get("response", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="room0")
    parser.add_argument("--min-width", type=float, default=0.7)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--ollama-model", default="gemma4:12b",
                        help="Ollama model for VLM verification")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM verification, just render images")
    args = parser.parse_args()

    out_dir = ROOT / "output" / args.name
    doors_data = json.load(open(out_dir / "doors.json"))
    summary_data = json.load(open(out_dir / "summary.json"))

    scan_center = tuple(summary_data["center"])
    floor_z = summary_data["floor_z"]

    # Pre-filter
    candidates = [
        d for d in doors_data["doors"]
        if d["width_m"] >= args.min_width and d["num_points"] >= args.min_points
    ]
    print(f"Pre-filtered: {len(candidates)} candidates")
    print(f"Scan center: {scan_center}, floor_z: {floor_z:.3f}")

    # Load scene — prefer original weights (point_cloud_*.ply) over feat_vis
    data_dir = ROOT / "data" / args.name
    original_plys = sorted(data_dir.glob("point_cloud_*.ply"))
    feat_vis_plys = sorted(data_dir.glob("*_feat_vis_3dgs.ply"))
    ply_path = original_plys[0] if original_plys else feat_vis_plys[0]
    feat_path = sorted(data_dir.glob("*_feat.pt"))[0]
    print(f"Loading scene: {ply_path.name} ({'original weights' if original_plys else 'feat_vis'})")
    scene = GSScene.from_ply(
        ply_path, feat_path=feat_path,
        text_emb_path=str(ROOT / "data" / "0" / "bim_text_emb.pt"),
        class_names_path=str(ROOT / "data" / "0" / "bim_class_names.json"),
    )

    verify_dir = out_dir / "verify"
    verify_dir.mkdir(exist_ok=True)

    # Render each candidate
    results = []
    for i, c in enumerate(candidates):
        eye, target, theta, r = candidate_to_viewpoint(
            c["world_x"], c["world_y"],
            c["height_above_floor_min"], c["height_above_floor_max"],
            scan_center, floor_z,
        )
        print(f"\n[{i}] Wall {c['wall_idx']}, θ={theta:.1f}°, r={r:.2f}m, "
              f"width={c['width_m']:.2f}m, pts={c['num_points']}")

        pose = look_at_pose(
            (eye[0], eye[1], eye[2]),
            (target[0], target[1], target[2]),
            up=(0.0, 0.0, 1.0),
        )
        result = scene.render(pose, width=800, height=600, fov_degrees=60.0)

        png_path = str(verify_dir / f"candidate_{i}_wall{c['wall_idx']}.png")
        save_clean_image(result.colors, png_path)
        print(f"  Rendered: {png_path}")

        entry = {
            "candidate_idx": i,
            "wall_idx": c["wall_idx"],
            "theta": round(theta, 1),
            "r": round(r, 2),
            "width_m": c["width_m"],
            "num_points": c["num_points"],
            "world_x": c["world_x"],
            "world_y": c["world_y"],
            "image": f"verify/candidate_{i}_wall{c['wall_idx']}.png",
            "vlm_response": None,
            "vlm_confirmed": None,
        }

        # VLM verification via Ollama
        if not args.no_vlm:
            prompt = (
                "This image is rendered from inside a room. "
                "Is there a DOOR visible in this image? "
                "Answer with CONFIRMED or REJECTED on the first line, "
                "then briefly describe what you see."
            )
            print(f"  Querying Ollama {args.ollama_model}...")
            try:
                vlm_response = query_ollama(png_path, prompt, args.ollama_model)
                entry["vlm_response"] = vlm_response
                first_line = vlm_response.strip().split("\n")[0].upper()
                entry["vlm_confirmed"] = "CONFIRMED" in first_line
                status = "CONFIRMED" if entry["vlm_confirmed"] else "REJECTED"
                print(f"  VLM: {status}")
                # Print first 200 chars of response
                preview = vlm_response[:200].replace("\n", " ")
                print(f"  Response: {preview}...")
            except Exception as e:
                entry["vlm_response"] = f"ERROR: {e}"
                entry["vlm_confirmed"] = None
                print(f"  VLM ERROR: {e}")

        results.append(entry)

    # Save manifest with VLM results
    confirmed = [r for r in results if r["vlm_confirmed"] is True]
    rejected = [r for r in results if r["vlm_confirmed"] is False]
    errors = [r for r in results if r["vlm_confirmed"] is None]

    manifest = {
        "scene": args.name,
        "ply_used": ply_path.name,
        "ollama_model": args.ollama_model if not args.no_vlm else None,
        "num_candidates": len(results),
        "num_confirmed": len(confirmed),
        "num_rejected": len(rejected),
        "num_errors": len(errors),
        "candidates": results,
    }
    manifest_path = verify_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Results: {len(confirmed)} confirmed, {len(rejected)} rejected, "
          f"{len(errors)} errors")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
