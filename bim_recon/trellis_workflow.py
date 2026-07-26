"""Deterministic, approval-gated TRELLIS generation and Revit registration."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from bim_recon.mcp_gateway import ToolGateway
from bim_recon.mesh_registrar import (
    MeshPlacement,
    compute_placement_transform,
    register_mesh_in_revit,
    serialize_placement_diagnostics,
)
from bim_recon.trellis_client import (
    TrellisClient,
    TrellisMeshRequest,
    TrellisMeshResult,
)
from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
    WorkflowWarning,
)


class _GenerateObject(Event):
    index: int


class _FinishMeshes(Event):
    pass


@dataclass(frozen=True, slots=True)
class ApprovedMeshObject:
    """One user-approved explorer object and its physical placement size.

    Attributes:
        yaw_degrees: Per-object yaw around the world up axis in degrees,
            positive = clockwise from above. Lets the user correct the
            constant-offset assumption when an object is placed diagonally
            in the scene (detection reports position only, not orientation).
            Default 90.0 matches the wall-aligned convention; set to 45.0
            for objects placed at 45° to the walls, etc.
    """

    object_id: str
    label: str
    image_path: Path
    position_3d: tuple[float, float, float]
    up_axis: int = 2
    width_m: float = 0.8
    height_m: float = 1.0
    seed: int = 1
    yaw_degrees: float = 90.0
    depth_path: Path | None = None
    observation_rgb_path: Path | None = None
    mask_path: Path | None = None
    norm_bbox: tuple[float, float, float, float] | None = None
    camera_eye: tuple[float, float, float] | None = None
    camera_target: tuple[float, float, float] | None = None
    camera_up: tuple[float, float, float] | None = None
    camera_fov: float = 45.0
    camera_image_size: tuple[int, int] = (800, 800)


@dataclass(frozen=True, slots=True)
class TrellisWorkflowConfig:
    objects: tuple[ApprovedMeshObject, ...]
    output_dir: Path
    register_in_revit: bool = True
    category: str = "OST_GenericModel"
    generation_retries: int = 1


TrellisFactory = Callable[[], TrellisClient]


def approved_object_from_extraction(
    extraction,
    *,
    seed: int = 1,
    yaw_degrees: float = 90.0,
) -> ApprovedMeshObject:
    """Convert a :class:`~bim_recon.bmesh_pipeline.BClassExtraction` into an
    :class:`ApprovedMeshObject` ready for the TRELLIS workflow.
    """
    return ApprovedMeshObject(
        object_id=f"{extraction.element.element_class}_{extraction.element.result_index}",
        label=extraction.element.element_class,
        image_path=extraction.cutout_path,
        position_3d=extraction.position_3d,
        up_axis=extraction.up_axis,
        width_m=extraction.width_m,
        height_m=extraction.height_m,
        seed=seed,
        yaw_degrees=yaw_degrees,
        observation_rgb_path=extraction.render_path,
        mask_path=extraction.mask_path,
        depth_path=extraction.depth_path,
        norm_bbox=(
            float(extraction.norm_bbox["x"]),
            float(extraction.norm_bbox["y"]),
            float(extraction.norm_bbox["w"]),
            float(extraction.norm_bbox["h"]),
        ) if extraction.norm_bbox else None,
        camera_eye=extraction.camera_eye,
        camera_target=extraction.camera_target,
        camera_up=extraction.camera_up,
        camera_fov=extraction.camera_fov,
        camera_image_size=extraction.camera_image_size,
    )



class TrellisRevitWorkflow(Workflow):
    """Generate approved meshes one-by-one and optionally create DirectShapes."""

    workflow_name = "trellis_revit"

    def __init__(
        self,
        config: TrellisWorkflowConfig,
        client_factory: TrellisFactory,
        gateway: ToolGateway | None = None,
    ):
        super().__init__(timeout=None, verbose=False)
        self.config = config
        self.client_factory = client_factory
        self.gateway = gateway
        self.client: TrellisClient | None = None
        self.results: list[dict[str, Any]] = []

    @step
    async def prepare(
        self, ctx: Context, ev: StartEvent,
    ) -> _GenerateObject | StopEvent:
        if not self.config.objects:
            message = "Select at least one approved explorer object"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="approval",
                message=message,
            ))
            return StopEvent(result={"error": message})
        if self.config.register_in_revit and self.gateway is None:
            message = "Revit registration requested without an MCP gateway"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="approval",
                message=message,
            ))
            return StopEvent(result={"error": message})
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = self.client_factory()
        healthy = await asyncio.to_thread(self.client.health)
        if not healthy:
            message = "TRELLIS server is unreachable or its model is not loaded"
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="preflight",
                message=message,
            ))
            return StopEvent(result={"error": message})
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="approval",
            message=(
                f"Approved {len(self.config.objects)} objects; "
                "starting expensive mesh generation"
            ),
        ))
        return _GenerateObject(index=0)

    @step
    async def generate_object(
        self, ctx: Context, ev: _GenerateObject,
    ) -> _GenerateObject | _FinishMeshes:
        assert self.client is not None
        obj = self.config.objects[ev.index]
        current = ev.index + 1
        total = len(self.config.objects)
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage="trellis",
            message=f"Generating {obj.label} ({obj.object_id})...",
            current=current,
            total=total,
            image_path=str(obj.image_path),
        ))
        try:
            mesh = await self._generate_with_retry(obj)
            result = await self._register_mesh(obj, mesh)
            self.results.append(result)
            ctx.write_event_to_stream(WorkflowProgress(
                workflow=self.workflow_name,
                stage="revit" if self.config.register_in_revit else "trellis",
                message=f"Completed {obj.label} ({obj.object_id})",
                current=current,
                total=total,
                payload=result,
            ))
        except Exception as exc:
            failure = {
                "object_id": obj.object_id,
                "label": obj.label,
                "status": "failed",
                "error": str(exc),
            }
            self.results.append(failure)
            ctx.write_event_to_stream(WorkflowWarning(
                workflow=self.workflow_name,
                stage="object",
                message=f"{obj.object_id} failed: {exc}",
                payload=failure,
            ))
        self._write_manifest()
        if current >= total:
            return _FinishMeshes()
        return _GenerateObject(index=current)

    @step
    async def finish(
        self, ctx: Context, ev: _FinishMeshes,
    ) -> StopEvent:
        manifest = self._write_manifest()
        result = {
            "output_dir": str(self.config.output_dir),
            "manifest": str(manifest),
            "completed": sum(
                item.get("status") == "completed" for item in self.results
            ),
            "failed": sum(
                item.get("status") == "failed" for item in self.results
            ),
            "objects": self.results,
        }
        ctx.write_event_to_stream(WorkflowCompleted(
            workflow=self.workflow_name,
            result=result,
        ))
        return StopEvent(result=result)

    async def _generate_with_retry(
        self,
        obj: ApprovedMeshObject,
    ) -> TrellisMeshResult:
        assert self.client is not None
        request = TrellisMeshRequest(
            image_path=obj.image_path,
            output_dir=self.config.output_dir / obj.object_id,
            name=f"{obj.label}_{obj.object_id}",
            seed=obj.seed,
        )
        last_error: Exception | None = None
        for attempt in range(self.config.generation_retries + 1):
            try:
                return await asyncio.to_thread(self.client.generate_mesh, request)
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.config.generation_retries:
                    break
                await asyncio.sleep(min(2 ** attempt, 5))
        assert last_error is not None
        raise last_error

    async def _register_mesh(
        self,
        obj: ApprovedMeshObject,
        mesh: TrellisMeshResult,
    ) -> dict[str, Any]:
        horizontal_axes = [axis for axis in range(3) if axis != obj.up_axis]
        vertical_center = obj.position_3d[obj.up_axis]
        placement = MeshPlacement(
            glb_path=mesh.glb_path,
            world_x=obj.position_3d[horizontal_axes[0]],
            world_y=obj.position_3d[horizontal_axes[1]],
            floor_z=vertical_center - obj.height_m / 2.0,
            ceiling_z=vertical_center + max(obj.height_m / 2.0, 3.0),
            element_width_m=obj.width_m,
            element_height_m=obj.height_m,
            up_axis=obj.up_axis,
            yaw_degrees=obj.yaw_degrees,
            category=self.config.category,
            name=f"{obj.label} {obj.object_id}",
        )
        transform = await asyncio.to_thread(compute_placement_transform, placement)
        formatted = register_mesh_in_revit(placement, transform)
        diagnostics = serialize_placement_diagnostics(placement, transform)
        base = {
            "object_id": obj.object_id,
            "label": obj.label,
            "status": "completed",
            "glb_path": str(mesh.glb_path),
            "gaussian_path": str(mesh.gaussian_path) if mesh.gaussian_path else None,
            "preview_path": str(mesh.preview_path) if mesh.preview_path else None,
            "vertex_count": formatted["vertex_count"],
            "face_count": formatted["face_count"],
            "diagnostics": diagnostics,
        }
        if not self.config.register_in_revit:
            return base
        assert self.gateway is not None
        response = await self.gateway.call_tool(
            "create_directshape_from_mesh",
            {"meshFile": formatted["payload_path"]},
        )
        return {**base, "revit_response": response}

    def _write_manifest(self) -> Path:
        path = self.config.output_dir / "trellis_workflow.json"
        path.write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
