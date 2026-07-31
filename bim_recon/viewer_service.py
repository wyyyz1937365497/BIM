"""FastAPI manager for one asynchronously launched SceneSplat Mini Viewer."""
from __future__ import annotations

import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bim_recon.config import ViewerServiceConfig, load_config
from scripts.run_viewer import launch_viewer

ROOT = Path(__file__).resolve().parent.parent


class ViewerStartRequest(BaseModel):
    """Paths are project-relative so the manager owns filesystem access."""

    scene: str
    input_root: str
    feature_path: str


@dataclass
class ManagedViewer:
    scene: str
    input_root: Path
    feature_path: Path
    port: int
    process: subprocess.Popen[bytes]
    started_at: float


class ViewerManager:
    """Own a single viewer child process without blocking API callers."""

    def __init__(self, config: ViewerServiceConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._viewer: ManagedViewer | None = None

    def _project_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("paths must be relative to the project root")
        resolved = (ROOT / path).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("paths must remain inside the project root") from exc
        return resolved

    @staticmethod
    def _port_is_available(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return False
        return True

    def _next_viewer_port(self) -> int:
        # 18082 remains retired: it must never be revived as a camera bridge.
        candidates = [self._config.viewer_port, *range(18084, 18092)]
        for port in candidates:
            if port != self._config.port and self._port_is_available(self._config.host, port):
                return port
        raise RuntimeError("no available viewer port in 18081 or 18084–18091")

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _snapshot(self, viewer: ManagedViewer | None = None) -> dict[str, object]:
        viewer = viewer or self._viewer
        if viewer is None:
            return {"status": "idle"}
        exit_code = viewer.process.poll()
        return {
            "status": "exited" if exit_code is not None else "starting",
            "scene": viewer.scene,
            "port": viewer.port,
            "url": f"http://{self._config.host}:{viewer.port}",
            "pid": viewer.process.pid,
            "exit_code": exit_code,
            "started_at": viewer.started_at,
        }

    def start(self, request: ViewerStartRequest) -> dict[str, object]:
        scene = request.scene.strip()
        if not scene or Path(scene).name != scene:
            raise ValueError("scene must be a single directory name")
        input_root = self._project_path(request.input_root)
        feature_path = self._project_path(request.feature_path)
        if not input_root.is_dir():
            raise ValueError(f"viewer input directory does not exist: {request.input_root}")
        if not feature_path.is_file():
            raise ValueError(f"viewer feature file does not exist: {request.feature_path}")

        with self._lock:
            active = self._viewer
            if (
                active is not None
                and active.process.poll() is None
                and active.scene == scene
                and active.input_root == input_root
                and active.feature_path == feature_path
            ):
                return self._snapshot(active)
            if active is not None:
                # Do not wait for shutdown here: POST /viewer must stay fast for
                # the Gradio pipeline callback. A new free port is selected below.
                self._terminate_process(active.process)
            port = self._next_viewer_port()
            process = launch_viewer(
                str(input_root),
                str(feature_path),
                port=port,
                host=self._config.host,
            )
            self._viewer = ManagedViewer(
                scene=scene,
                input_root=input_root,
                feature_path=feature_path,
                port=port,
                process=process,
                started_at=time.time(),
            )
            return self._snapshot()

    def status(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot()

    def stop(self) -> dict[str, object]:
        with self._lock:
            viewer = self._viewer
            if viewer is None:
                return {"status": "idle"}
            self._stop_process(viewer.process)
            return self._snapshot(viewer)

    def close(self) -> None:
        with self._lock:
            if self._viewer is not None:
                self._stop_process(self._viewer.process)


def create_app(config: ViewerServiceConfig | None = None) -> FastAPI:
    manager = ViewerManager(config or load_config().viewer_service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(title="SceneSplat Viewer Manager", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Literal["ok"]]:
        return {"status": "ok"}

    @app.get("/viewer")
    async def viewer_status() -> dict[str, object]:
        return manager.status()

    @app.post("/viewer")
    async def start_viewer(request: ViewerStartRequest) -> dict[str, object]:
        try:
            return manager.start(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/viewer")
    async def stop_viewer() -> dict[str, object]:
        return manager.stop()

    return app


if __name__ == "__main__":
    cfg = load_config().viewer_service
    app = create_app(cfg)
    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port)
else:
    app = create_app()
