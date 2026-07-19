"""Launch the standalone SceneSplat Mini Viewer on port 18081.

The viewer serves its own UI and ``/camera-state`` from that single port via
``viewer_camera_patch.py``.  Gradio does not own, start, or embed this process.
"""
from __future__ import annotations

import socket

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENESPLAT = ROOT / "SceneSplat"
PATCH_SCRIPT = ROOT / "scripts" / "viewer_camera_patch.py"

VIEWER_PORT = 18081


def _build_env() -> dict[str, str]:
    """Build subprocess env with CUDA paths (mirrors tools.mini_viewer)."""
    env = os.environ.copy()
    prefix = Path(sys.executable).resolve().parent.parent
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.exists():
        env.setdefault("CUDA_HOME", str(prefix))
        env.setdefault("CONDA_PREFIX", str(prefix))
    return env


def _ensure_port_available(host: str, port: int) -> None:
    """Fail instead of letting Viser silently move the public viewer port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Viewer port {host}:{port} is already in use. "
                "Stop the existing viewer before launching this scene."
            ) from exc


def launch_viewer(
    input_root: str,
    feature_path: str,
    port: int = VIEWER_PORT,
    host: str = "127.0.0.1",
) -> subprocess.Popen[bytes]:
    """Start the standalone nerfview process and return its handle."""
    input_path = Path(input_root)
    cmd: list[str] = [
        sys.executable, str(PATCH_SCRIPT),
    ]
    # Mini Viewer data source
    if input_path.is_dir():
        cmd.extend(["--folder-npy", str(input_path)])
    elif input_path.suffix.lower() == ".ply":
        cmd.extend(["--ply", str(input_path)])
    else:
        raise ValueError(f"Unsupported input: {input_path}")
    # Feature file
    if feature_path:
        cmd.extend(["--feature-file", str(feature_path)])
    # Standard Mini Viewer defaults
    cmd.extend([
        "--feature-type", "siglip2",
        "--device", "auto",
        "--backend", "auto",
        "--pca-device", "auto",
        "--pca-method", "torch",
        "--pca-brightness", "1.25",
        "--pca-seed", "42",
        "--host", host,
        "--port", str(port),
    ])
    _ensure_port_available(host, port)
    env = _build_env()
    return subprocess.Popen(cmd, env=env)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the standalone 3DGS Mini Viewer",
    )
    parser.add_argument(
        "--scene",
        help="Scene name; resolves data/<scene>/preprocessed and "
        "output/<scene>/<scene>_feat.pt.",
    )
    parser.add_argument("--input-root")
    parser.add_argument("--feature-path")
    parser.add_argument("--port", type=int, default=VIEWER_PORT)
    args = parser.parse_args()
    if args.scene:
        input_root = ROOT / "data" / args.scene / "preprocessed"
        feature_path = ROOT / "output" / args.scene / f"{args.scene}_feat.pt"
    else:
        if not args.input_root or not args.feature_path:
            parser.error("pass --scene, or both --input-root and --feature-path")
        input_root = Path(args.input_root)
        feature_path = Path(args.feature_path)
    if not input_root.exists():
        parser.error(f"Viewer input does not exist: {input_root}")
    if not feature_path.is_file():
        parser.error(f"Viewer feature file does not exist: {feature_path}")
    print(f"Starting standalone nerfview on http://127.0.0.1:{args.port}")
    print(f"Camera state: http://127.0.0.1:{args.port}/camera-state")
    proc = launch_viewer(str(input_root), str(feature_path), args.port)
    proc.wait()
