"""Launch the SceneSplat Mini Viewer (nerfview) on a specified port.

This is a thin wrapper around ``SceneSplat/tools/mini_viewer.py`` that
defaults to port 8081 (to avoid conflict with Revit MCP on 8080).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENESPLAT = ROOT / "SceneSplat"


def launch_viewer(
    input_root: str,
    feature_path: str,
    port: int = 8081,
    host: str = "127.0.0.1",
) -> subprocess.Popen[bytes]:
    """Start nerfview as a subprocess. Returns the Popen handle."""
    cmd = [
        sys.executable, "-m", "tools.mini_viewer",
        "--input-root", input_root,
        "--feature-path", feature_path,
        "--host", host,
        "--port", str(port),
    ]
    return subprocess.Popen(cmd, cwd=str(SCENESPLAT))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch 3DGS Mini Viewer")
    parser.add_argument("--input-root", default=str(ROOT / "data" / "room0" / "preprocessed"))
    parser.add_argument("--feature-path", default=str(ROOT / "output" / "room0" / "room0_feat.pt"))
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    print(f"Starting nerfview on http://127.0.0.1:{args.port}")
    proc = launch_viewer(args.input_root, args.feature_path, args.port)
    proc.wait()
