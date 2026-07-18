"""Launch Mini Viewer with camera state HTTP endpoint on port 18082.

Monkey-patches ``viser.ViserServer.__init__`` to also start a tiny HTTP
server that exposes the current camera state as JSON at
``GET http://127.0.0.1:18082/camera-state``.

Usage (same args as ``run_viewer``)::

    python scripts/viewer_camera_patch.py --folder-npy ... --feature-file ... --port 18081

Response format::

    {
      "position": [x, y, z],
      "look_at":  [x, y, z],
      "up":       [x, y, z],
      "fov": 0.873,           // radians
      "fov_degrees": 50.0,
      "aspect": 1.778,
      "c2w": [[...4x4...]]
    }
"""
from __future__ import annotations

import http.server
import json
import os
import threading
from typing import Any

import numpy as np
import viser
import viser.transforms as vt

CAMERA_PORT = 18082

_original_viserserver_init = viser.ViserServer.__init__


def _start_camera_http(server: viser.ViserServer, port: int = CAMERA_PORT) -> None:
    """Start a daemon HTTP server that returns the first client's camera state."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            if self.path.rstrip("/") != "/camera-state":
                self.send_response(404)
                self._cors()
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            clients = server.get_clients()
            if not clients:
                self.wfile.write(
                    json.dumps({"error": "no client connected"}).encode()
                )
                return
            cam = list(clients.values())[0].camera
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = vt.SO3(cam.wxyz).as_matrix()
            c2w[:3, 3] = cam.position
            data: dict[str, Any] = {
                "position": np.asarray(cam.position, dtype=np.float64).tolist(),
                "look_at": np.asarray(cam.look_at, dtype=np.float64).tolist(),
                "up": np.asarray(cam.up_direction, dtype=np.float64).tolist(),
                "fov": float(cam.fov),
                "fov_degrees": float(np.degrees(cam.fov)),
                "aspect": float(cam.aspect),
                "c2w": c2w.tolist(),
            }
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # silence

    try:
        httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(
            f"[camera] HTTP endpoint ready: "
            f"http://127.0.0.1:{port}/camera-state"
        )
    except OSError as exc:
        print(f"[camera] Could not bind port {port}: {exc}")


def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _original_viserserver_init(self, *args, **kwargs)
    _start_camera_http(self, port=CAMERA_PORT)


# Apply monkey-patch BEFORE run_viewer imports viser.
viser.ViserServer.__init__ = _patched_init  # type: ignore[method-assign]

# Remove this script's own directory from sys.path so ``import run_viewer``
# resolves to the installed Mini Viewer package in site-packages, NOT our
# wrapper ``scripts/run_viewer.py`` (name collision when cwd or scripts/ is
# on sys.path).
import sys  # noqa: E402
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _script_dir]

# Delegate to the installed Mini Viewer entry point.
import run_viewer  # noqa: E402

if __name__ == "__main__":
    run_viewer.main()
