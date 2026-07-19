"""Expose the Mini Viewer and its camera state through one HTTP port.

This module patches Viser's HTTP request hook before launching the installed
``run_viewer`` entry point.  The regular viewer UI and
``GET /camera-state`` are both served from the supplied viewer port; no
second camera-bridge port is opened.
"""
from __future__ import annotations

import http
import json
import os
import sys
from typing import Any, Awaitable, Callable

import numpy as np
import viser
import viser.transforms as vt
import websockets.server

_original_serve = websockets.server.serve
_viewer_server: viser.ViserServer | None = None


def _camera_response(
    server: viser.ViserServer,
) -> tuple[http.HTTPStatus, dict[str, str], bytes]:
    """Return camera state using Viser's existing public HTTP listener."""
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
    }
    clients = server.get_clients()
    if not clients:
        return http.HTTPStatus.OK, headers, b'{"error":"no client connected"}'
    camera = next(iter(clients.values())).camera
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = vt.SO3(camera.wxyz).as_matrix()
    c2w[:3, 3] = camera.position
    data: dict[str, Any] = {
        "position": np.asarray(camera.position, dtype=np.float64).tolist(),
        "look_at": np.asarray(camera.look_at, dtype=np.float64).tolist(),
        "up": np.asarray(camera.up_direction, dtype=np.float64).tolist(),
        "fov": float(camera.fov),
        "fov_degrees": float(np.degrees(camera.fov)),
        "aspect": float(camera.aspect),
        "c2w": c2w.tolist(),
    }
    return http.HTTPStatus.OK, headers, json.dumps(data).encode()


def _patched_serve(
    handler: Callable[..., Awaitable[None]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Add /camera-state while delegating all viewer UI routes to Viser."""
    original_request = kwargs.get("process_request")

    async def process_request(path: str, request_headers: Any) -> Any:
        if path.partition("?")[0].rstrip("/") == "/camera-state":
            if _viewer_server is None:
                return (
                    http.HTTPStatus.SERVICE_UNAVAILABLE,
                    {"Content-Type": "application/json"},
                    b'{"error":"viewer initializing"}',
                )
            return _camera_response(_viewer_server)
        if original_request is None:
            return None
        return await original_request(path, request_headers)

    kwargs["process_request"] = process_request
    return _original_serve(handler, *args, **kwargs)


def _patched_init(self: viser.ViserServer, *args: Any, **kwargs: Any) -> None:
    global _viewer_server
    _viewer_server = self
    _original_viserserver_init(self, *args, **kwargs)


_original_viserserver_init = viser.ViserServer.__init__
websockets.server.serve = _patched_serve
viser.ViserServer.__init__ = _patched_init  # type: ignore[method-assign]

# Remove this script's own directory so ``import run_viewer`` resolves the
# installed Mini Viewer package rather than scripts/run_viewer.py.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _script_dir]

import run_viewer  # noqa: E402

if __name__ == "__main__":
    run_viewer.main()
