"""Typed workflow for deterministic B-class scene exploration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from bim_recon.explorer_controller import ExplorerCamera, ExplorerController
from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
)


class _InitializeExplorer(Event):
    pass


class _ScanView(Event):
    index: int


class _CompleteScan(Event):
    pass


@dataclass(frozen=True, slots=True)
class ExplorerScanConfig:
    ply_path: Path
    feat_path: Path | None
    output_root: Path
    camera: ExplorerCamera
    labels: tuple[str, ...]
    falcon_host: str = "127.0.0.1"
    falcon_port: int = 18390
    num_views: int = 8
    turn_degrees: float = 45.0
    width: int = 1024
    height: int = 768


ExplorerFactory = Callable[[ExplorerScanConfig, Path], ExplorerController]


class ExplorerScanWorkflow(Workflow):
    """Scan a bounded set of camera headings and deduplicate Falcon detections."""

    workflow_name = "explorer_scan"

    def __init__(
        self,
        config: ExplorerScanConfig,
        controller_factory: ExplorerFactory | None = None,
    ):
        super().__init__(timeout=None, verbose=False)
        self.config = config
        self.controller_factory = controller_factory or _default_factory
        self.controller: ExplorerController | None = None
        self.output_dir: Path | None = None
        self.view_results: list[dict] = []

    @step
    async def prepare(
        self, ctx: Context, ev: StartEvent,
    ) -> _InitializeExplorer | StopEvent:
        labels = tuple(dict.fromkeys(
            label.strip() for label in self.config.labels if label.strip()
        ))
        if not labels:
            message = "At least one object label is required"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="prepare",
                message=message,
            ))
            return StopEvent(result={"error": message})
        if self.config.num_views < 1:
            message = "num_views must be at least 1"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="prepare",
                message=message,
            ))
            return StopEvent(result={"error": message})
        self.config = replace(self.config, labels=labels)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.output_root / timestamp
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="prepare",
            message=(
                f"Prepared {self.config.num_views} views for labels: "
                f"{', '.join(labels)}"
            ),
        ))
        return _InitializeExplorer()

    @step
    async def initialize(
        self, ctx: Context, ev: _InitializeExplorer,
    ) -> _ScanView:
        assert self.output_dir is not None
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="initialize",
            message="Loading 3DGS scene and Falcon client...",
        ))
        self.controller = await asyncio.to_thread(
            self.controller_factory,
            self.config,
            self.output_dir,
        )
        initial_view = await asyncio.to_thread(
            self.controller.initialize,
            self.config.camera,
        )
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="initialize",
            message="Explorer initialized",
            image_path=initial_view,
            payload={"image_path": initial_view},
        ))
        return _ScanView(index=0)

    @step
    async def scan_view(
        self, ctx: Context, ev: _ScanView,
    ) -> _ScanView | _CompleteScan:
        assert self.controller is not None
        current = ev.index + 1
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="scan",
            message=f"Scanning heading {current}/{self.config.num_views}...",
            current=current,
            total=self.config.num_views,
        ))
        result = await asyncio.to_thread(
            self.controller.scan_current,
            self.config.labels,
        )
        self.view_results.append(result)
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="scan",
            message=(
                f"{len(result['detections'])} detections, "
                f"{len(result['tagged'])} new objects"
            ),
            current=current,
            total=self.config.num_views,
            image_path=result["view_path"],
            payload={
                "image_path": result["view_path"],
                "detections": result["detections"],
                "found_objects": list(self.controller.found),
            },
        ))
        if current >= self.config.num_views:
            return _CompleteScan()
        self.controller.turn(self.config.turn_degrees)
        return _ScanView(index=current)

    @step
    async def complete(
        self, ctx: Context, ev: _CompleteScan,
    ) -> StopEvent:
        assert self.controller is not None and self.output_dir is not None
        found_path = await asyncio.to_thread(self.controller.persist)
        result = {
            "output_dir": str(self.output_dir),
            "found_objects_path": str(found_path),
            "found_objects": list(self.controller.found),
            "view_results": self.view_results,
            "status": self.controller.status(),
        }
        ctx.write_event_to_stream(WorkflowCompleted(
            workflow=self.workflow_name,
            result=result,
        ))
        return StopEvent(result=result)


def _default_factory(
    config: ExplorerScanConfig,
    output_dir: Path,
) -> ExplorerController:
    return ExplorerController.load(
        config.ply_path,
        config.feat_path,
        config.falcon_host,
        config.falcon_port,
        output_dir,
        width=config.width,
        height=config.height,
    )
