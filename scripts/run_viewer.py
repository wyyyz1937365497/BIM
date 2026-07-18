"""Launch the SceneSplat Mini Viewer (nerfview) on port 18081.

Also starts a camera-state HTTP endpoint on port 18082 via
``scripts/viewer_camera_patch.py`` (monkey-patches viser.ViserServer).

This wrapper bypasses ``tools/mini_viewer.py`` and launches the patched
entry point directly, forwarding all Mini Viewer args.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENESPLAT = ROOT / "SceneSplat"
PATCH_SCRIPT = ROOT / "scripts" / "viewer_camera_patch.py"

VIEWER_PORT = 18081
CAMERA_PORT = 18082


def _build_env() -> dict[str, str]:
    """Build subprocess env with CUDA paths (mirrors tools.mini_viewer)."""
    env = os.environ.copy()
    prefix = Path(sys.executable).resolve().parent.parent
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.exists():
        env.setdefault("CUDA_HOME", str(prefix))
        env.setdefault("CONDA_PREFIX", str(prefix))
    return env


def launch_viewer(
    input_root: str,
    feature_path: str,
    port: int = VIEWER_PORT,
    host: str = "127.0.0.1",
) -> subprocess.Popen[bytes]:
    """Start nerfview + camera HTTP endpoint. Returns the Popen handle.

    The camera endpoint will be available at
    ``http://127.0.0.1:{CAMERA_PORT}/camera-state``.
    """
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
    env = _build_env()
    return subprocess.Popen(cmd, env=env)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch 3DGS Mini Viewer")
    parser.add_argument("--input-root", default=str(ROOT / "data" / "room0" / "preprocessed"))
    parser.add_argument("--feature-path", default=str(ROOT / "output" / "room0" / "room0_feat.pt"))
    parser.add_argument("--port", type=int, default=VIEWER_PORT)
    args = parser.parse_args()

    print(f"Starting nerfview on http://127.0.0.1:{args.port}")
    print(f"Camera endpoint on http://127.0.0.1:{CAMERA_PORT}/camera-state")
    proc = launch_viewer(args.input_root, args.feature_path, args.port)
    proc.wait()
