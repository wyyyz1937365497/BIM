"""Behavioral tests for deterministic workflow orchestration."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from bim_recon.explorer_controller import ExplorerCamera
from bim_recon.explorer_workflow import ExplorerScanConfig, ExplorerScanWorkflow
from bim_recon.pipeline_api import ElementResult, PipelineResults, load_results
from bim_recon.pipeline_runner import (
    PipelineConfig,
    ReconstructionWorkflow,
    snap_wall_endpoints,
)
from bim_recon.revit_workflow import (
    REWRITE_SOLID_200MM_WALL_TYPE_ID,
    RevitBuildWorkflow,
)
from bim_recon.revit_runner import RevitScriptRunner
from bim_recon.trellis_client import TrellisMeshResult
from bim_recon.trellis_workflow import (
    ApprovedMeshObject,
    TrellisRevitWorkflow,
    TrellisWorkflowConfig,
)
from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
)
from bim_recon.workflow_runtime import (
    stream_workflow_gradio,
    stream_workflow_sync,
)


class _TinyWorkflow(Workflow):
    def __init__(self):
        super().__init__(timeout=None, verbose=False)

    @step
    async def run_once(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(WorkflowProgress(
            workflow="tiny",
            stage="one",
            message="working",
            payload={"value": 1},
        ))
        ctx.write_event_to_stream(WorkflowCompleted(
            workflow="tiny",
            result={"ok": True},
        ))
        return StopEvent(result={"ok": True})


def test_runtime_streams_typed_events_to_sync_and_gradio_adapters():
    sync_events = list(stream_workflow_sync(_TinyWorkflow()))
    assert [type(event) for event in sync_events] == [
        WorkflowProgress,
        WorkflowCompleted,
    ]

    async def collect():
        return [update async for update in stream_workflow_gradio(_TinyWorkflow())]

    updates = asyncio.run(collect())
    assert [update.kind for update in updates] == ["progress", "completed"]
    assert updates[-1].payload == {"ok": True}


class _FakeGateway:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.created_ids: list[int] = []
        self._next_script_id = 401

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if name == "create_level":
            ids = [1]
        elif name == "create_surface_based_element":
            ids = [2]
        elif name == "create_line_based_element":
            ids = list(range(101, 101 + len(arguments["data"])))
        elif name == "create_point_based_element":
            type_id = arguments["data"][0]["typeId"]
            ids = [201] if type_id == 94654 else [301]
        elif name == "get_current_view_elements":
            ids = self.created_ids
        elif name == "send_code_to_revit":
            ids = [self._next_script_id]
            self._next_script_id += 1
            self.created_ids.extend(ids)
            return {"success": True, "result": json.dumps({"elementId": ids[0]})}
        elif name == "create_directshape_from_mesh":
            ids = [self._next_script_id]
            self._next_script_id += 1
            self.created_ids.extend(ids)
            return {"Success": True, "Response": ids[0]}
        else:
            raise AssertionError(name)
        if name != "get_current_view_elements":
            self.created_ids.extend(ids)
        if name in {"create_surface_based_element", "create_line_based_element"}:
            return {"Response": ids}
        return {"Response": [{"elementId": element_id} for element_id in ids]}


def _sample_results(tmp_path: Path) -> PipelineResults:
    walls = [
        {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0, "length": 4.0},
        {"x1": 4.0, "y1": 0.0, "x2": 4.0, "y2": 3.0, "length": 3.0},
        {"x1": 4.0, "y1": 3.0, "x2": 0.0, "y2": 3.0, "length": 4.0},
        {"x1": 0.0, "y1": 3.0, "x2": 0.0, "y2": 0.0, "length": 3.0},
    ]
    elements = [
        ElementResult(
            element_class="door",
            confirmed=True,
            vlm_response="yes",
            image_path="door.png",
            world_x=1.0,
            world_y=0.0,
            wall_idx=0,
            result_index=0,
            height_detection={
                "width_m": 0.9,
                "sill_height": 0.0,
                "header_height": 2.1,
                "element_height": 2.1,
            },
        ),
        ElementResult(
            element_class="window",
            confirmed=True,
            vlm_response="yes",
            image_path="window.png",
            world_x=4.0,
            world_y=1.5,
            wall_idx=1,
            result_index=1,
            height_detection={
                "width_m": 1.2,
                "sill_height": 0.8,
                "header_height": 2.0,
                "element_height": 1.2,
            },
        ),
    ]
    return PipelineResults(
        out_dir=tmp_path,
        walls=walls,
        elements=elements,
        coords={"floor_z": 0.0, "ceiling_z": 3.0},
    )

def _write_review_json_fixture(results: PipelineResults) -> None:
    """Write the same per-element JSON shape consumed by load_results."""
    sections: dict[str, dict] = {}
    for element_class in sorted({e.element_class for e in results.elements}):
        class_elements = [
            element
            for element in results.elements
            if element.element_class == element_class
        ]
        entries = [
            {
                "confirmed": element.confirmed,
                "candidate": {
                    "element_class": element.element_class,
                    "world_x": element.world_x,
                    "world_y": element.world_y,
                    "wall_idx": element.wall_idx,
                },
                "height_detection": element.height_detection,
                "image_path": "",
                "vlm_response": element.vlm_response,
            }
            for element in class_elements
        ]
        section = {
            "element": element_class,
            "confirmed": sum(bool(entry["confirmed"]) for entry in entries),
            "results": entries,
        }
        (results.out_dir / f"{element_class}s_verified.json").write_text(
            json.dumps(section),
            encoding="utf-8",
        )
        sections[element_class] = section
    (results.out_dir / "wall_lines_snapped.json").write_text(
        json.dumps(results.walls),
        encoding="utf-8",
    )
    (results.out_dir / "pipeline_report.json").write_text(
        json.dumps({
            "coordinate_system": results.coords,
            "elements": sections,
            "merged_elements": {
                "confirmed": sum(
                    section["confirmed"] for section in sections.values()
                ),
            },
        }),
        encoding="utf-8",
    )


def test_revit_workflow_creates_hosts_before_openings_and_verifies_ids(tmp_path):
    gateway = _FakeGateway()
    workflow = RevitBuildWorkflow(_sample_results(tmp_path), gateway)

    events = list(stream_workflow_sync(workflow))

    completed = next(event for event in events if isinstance(event, WorkflowCompleted))
    names = [name for name, _arguments in gateway.calls]
    assert names == [
        "create_level",
        "create_surface_based_element",
        "create_line_based_element",
        "create_point_based_element",
        "create_point_based_element",
        "get_current_view_elements",
    ]
    wall_call = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_line_based_element"
    )
    assert {
        item["typeId"] for item in wall_call["data"]
    } == {REWRITE_SOLID_200MM_WALL_TYPE_ID}
    opening_calls = [
        arguments for name, arguments in gateway.calls
        if name == "create_point_based_element"
    ]
    assert opening_calls[0]["data"][0]["hostWallId"] == 101
    assert opening_calls[1]["data"][0]["hostWallId"] == 102
    assert completed.result["missing_ids"] == []
    assert completed.result["created"]["doors"] == [201]
    assert completed.result["created"]["windows"] == [301]

    import orjson

    orjson.dumps(completed.result)
    assert completed.result["wall_id_by_index"] == {
        "0": 101,
        "1": 102,
        "2": 103,
        "3": 104,
    }


def test_revit_workflow_deduplicates_overlapping_walls_and_keeps_hosts(tmp_path):
    results = _sample_results(tmp_path)
    results.walls.append({
        "x1": 0.02, "y1": 0.01, "x2": 4.02, "y2": 0.01, "length": 4.0,
    })
    gateway = _FakeGateway()

    list(stream_workflow_sync(RevitBuildWorkflow(results, gateway)))

    wall_call = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_line_based_element"
    )
    opening_calls = [
        arguments
        for name, arguments in gateway.calls
        if name == "create_point_based_element"
    ]
    assert len(wall_call["data"]) == 4
    assert opening_calls[0]["data"][0]["hostWallId"] == 101
    assert opening_calls[1]["data"][0]["hostWallId"] == 102


def test_review_selection_limits_revit_creation_to_three_of_six(tmp_path):
    from bim_recon.gradio_helpers import apply_vlm_review

    results = _sample_results(tmp_path)
    door = results.elements[0]
    window = results.elements[1]
    window.result_index = 0
    results.elements.extend([
        replace(door, result_index=1, confirmed=True),
        replace(window, result_index=1, confirmed=True),
        replace(door, result_index=2, confirmed=True),
        replace(window, result_index=2, confirmed=True),
    ])
    _write_review_json_fixture(results)

    selected = ["door #0", "window #0", "door #1"]
    updated_results, status = apply_vlm_review(results, selected)

    assert updated_results is results
    assert [element.confirmed for element in results.elements] == [
        True, True, True, False, False, False,
    ]
    assert "3/6" in status
    assert "3 个 JSON 文件" in status

    reloaded = load_results(tmp_path)
    persisted_states = {
        f"{element.element_class} #{element.result_index}": element.confirmed
        for element in reloaded.elements
    }
    assert persisted_states == {
        "door #0": True,
        "door #1": True,
        "door #2": False,
        "window #0": True,
        "window #1": False,
        "window #2": False,
    }
    report = json.loads((tmp_path / "pipeline_report.json").read_text("utf-8"))
    assert report["elements"]["door"]["confirmed"] == 2
    assert report["elements"]["window"]["confirmed"] == 1
    assert report["merged_elements"]["confirmed"] == 3

    gateway = _FakeGateway()
    list(stream_workflow_sync(RevitBuildWorkflow(results, gateway)))
    opening_calls = [
        arguments
        for name, arguments in gateway.calls
        if name == "create_point_based_element"
    ]
    assert sum(len(arguments["data"]) for arguments in opening_calls) == 3

def test_revit_workflow_clips_opening_to_host_wall_endpoint(tmp_path):
    results = _sample_results(tmp_path)
    door = results.elements[0]
    door.world_x = 3.8
    door.height_detection["width_m"] = 0.9
    gateway = _FakeGateway()

    list(stream_workflow_sync(RevitBuildWorkflow(results, gateway)))

    door_call = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_point_based_element"
        and arguments["data"][0]["name"].startswith("Reconstructed door")
    )
    payload = door_call["data"][0]
    assert payload["locationPoint"]["x"] == pytest.approx(3675.0)
    assert payload["locationPoint"]["y"] == pytest.approx(0.0)
    assert payload["width"] == pytest.approx(650.0)

def test_revit_script_runner_passes_optional_base_type_id():
    calls: list[dict] = []
    runner = RevitScriptRunner(mcp_sender=lambda **kwargs: calls.append(kwargs) or {})

    runner.create_door(
        host_wall_id=101,
        x_m=1.0,
        y_m=2.0,
        sill_m=0.0,
        width_m=0.9,
        height_m=2.0,
        base_type_id=94654,
    )

    assert calls[0]["parameters"][0] == 101
    assert calls[0]["parameters"][7] == 94654


def test_wall_snapping_never_collapses_endpoints_of_the_same_wall():
    snapped = snap_wall_endpoints([{
        "x1": 0.0,
        "y1": 0.0,
        "x2": 0.3,
        "y2": 0.0,
        "length": 99.0,
    }], threshold=0.5)

    assert len(snapped) == 1
    assert snapped[0]["x1"] == 0.0
    assert snapped[0]["x2"] == 0.3
    assert snapped[0]["length"] == pytest.approx(0.3)


def test_revit_workflow_removes_short_curves_from_floor_and_walls(tmp_path):
    a = (0.4903733028, -1.4737401816)
    b = (-1.5815451137, 0.4419133492)
    c = (-1.0266988698, 1.2945968332)
    d = (-0.6011297413, 1.8727998858)
    e = (1.6970896625, -0.3509214441)

    def wall(start, end, reported_length):
        return {
            "x1": start[0],
            "y1": start[1],
            "x2": end[0],
            "y2": end[1],
            "length": reported_length,
        }

    walls = [
        wall(a, b, 2.6),
        wall(b, b, 0.6),
        wall(b, c, 0.9),
        wall(c, d, 0.7),
        wall(d, e, 3.3),
        wall(e, e, 0.35),
        wall(e, a, 1.4),
    ]
    results = PipelineResults(
        out_dir=tmp_path,
        walls=walls,
        elements=[],
        coords={"floor_z": 0.0, "ceiling_z": 3.0},
    )
    gateway = _FakeGateway()

    events = list(stream_workflow_sync(RevitBuildWorkflow(results, gateway)))

    completed = next(event for event in events if isinstance(event, WorkflowCompleted))
    floor_args = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_surface_based_element"
    )
    segments = floor_args["data"][0]["boundary"]["outerLoop"]
    wall_args = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_line_based_element"
    )
    lengths_mm = [
        (
            (segment["p1"]["x"] - segment["p0"]["x"]) ** 2
            + (segment["p1"]["y"] - segment["p0"]["y"]) ** 2
        ) ** 0.5
        for segment in segments
    ]

    assert len(segments) == 5
    assert min(lengths_mm) >= 1.0
    assert len(wall_args["data"]) == 5
    assert completed.result["created"]["walls"] == [101, 102, 103, 104, 105]


def test_revit_floor_profile_removes_near_duplicate_crossing_vertices(tmp_path):
    walls = [
        {
            "x1": 0.4857241228747955,
            "y1": -1.47395480709757,
            "x2": -1.583869192213017,
            "y2": 0.4399838356799301,
        },
        {
            "x1": -1.5842704518394495,
            "y1": 0.4375548773454621,
            "x2": -1.583869192213017,
            "y2": 0.4399838356799301,
        },
        {
            "x1": -1.5842704518394495,
            "y1": 0.4375548773454621,
            "x2": -1.0764082449331873,
            "y2": 1.2438549540794162,
        },
        {
            "x1": -1.0764082449331873,
            "y1": 1.2438549540794162,
            "x2": -0.6035275494976078,
            "y2": 1.8745399604270179,
        },
        {
            "x1": -0.6035275494976078,
            "y1": 1.8745399604270179,
            "x2": 1.692579322744133,
            "y2": -0.35206587366710557,
        },
        {
            "x1": 1.6931479183931366,
            "y1": -0.3506829748057044,
            "x2": 1.692579322744133,
            "y2": -0.35206587366710557,
        },
        {
            "x1": 1.6931479183931366,
            "y1": -0.3506829748057044,
            "x2": 0.4857241228747955,
            "y2": -1.47395480709757,
        },
    ]
    for wall in walls:
        wall["length"] = (
            (wall["x2"] - wall["x1"]) ** 2
            + (wall["y2"] - wall["y1"]) ** 2
        ) ** 0.5
    results = PipelineResults(
        out_dir=tmp_path,
        walls=walls,
        elements=[],
        coords={"floor_z": 0.0, "ceiling_z": 3.0},
    )
    gateway = _FakeGateway()

    list(stream_workflow_sync(RevitBuildWorkflow(results, gateway)))

    floor_args = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_surface_based_element"
    )
    segments = floor_args["data"][0]["boundary"]["outerLoop"]
    wall_args = next(
        arguments
        for name, arguments in gateway.calls
        if name == "create_line_based_element"
    )
    lengths_mm = [
        (
            (segment["p1"]["x"] - segment["p0"]["x"]) ** 2
            + (segment["p1"]["y"] - segment["p0"]["y"]) ** 2
        ) ** 0.5
        for segment in segments
    ]

    assert len(segments) == 5
    assert min(lengths_mm) >= 10.0
    assert len(wall_args["data"]) == 5


class _FakeExplorerController:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.found: list[dict] = []
        self.turns: list[float] = []
        self.scan_count = 0

    def initialize(self, camera):
        path = self.output_dir / "initial.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return str(path)

    def scan_current(self, labels):
        self.scan_count += 1
        path = self.output_dir / f"scan-{self.scan_count}.png"
        path.write_bytes(b"png")
        tagged = []
        if self.scan_count == 1:
            obj = {
                "id": "obj_001",
                "label": labels[0],
                "position_3d": [1.0, 2.0, 0.5],
                "best_view": str(path),
                "trellis_status": "pending_approval",
            }
            self.found.append(obj)
            tagged.append(obj)
        return {
            "view_path": str(path),
            "detections": [{"label": labels[0]}],
            "tagged": tagged,
            "duplicates": [],
        }

    def turn(self, degrees):
        self.turns.append(degrees)

    def persist(self):
        path = self.output_dir / "found_objects.json"
        path.write_text("[]", encoding="utf-8")
        return path

    def status(self):
        return {"camera": {"up_axis": 2}, "found_count": len(self.found)}


def test_explorer_workflow_scans_bounded_views_without_agent_loop(tmp_path):
    controllers: list[_FakeExplorerController] = []

    def factory(_config, output_dir):
        controller = _FakeExplorerController(output_dir)
        controllers.append(controller)
        return controller

    workflow = ExplorerScanWorkflow(
        ExplorerScanConfig(
            ply_path=tmp_path / "scene.ply",
            feat_path=None,
            output_root=tmp_path / "explore",
            camera=ExplorerCamera((0.0, 0.0, 1.0)),
            labels=("chair", "table"),
            num_views=3,
            turn_degrees=120.0,
        ),
        controller_factory=factory,
    )

    events = list(stream_workflow_sync(workflow))

    completed = next(event for event in events if isinstance(event, WorkflowCompleted))
    assert controllers[0].scan_count == 3
    assert controllers[0].turns == [120.0, 120.0]
    assert completed.result["found_objects"][0]["label"] == "chair"


class _FakeTrellisClient:
    def __init__(self, result: TrellisMeshResult):
        self.result = result
        self.requests = []

    def health(self):
        return True

    def generate_mesh(self, request):
        self.requests.append(request)
        return self.result


def test_trellis_workflow_requires_approval_and_registers_selected_mesh(
    tmp_path, monkeypatch,
):
    image = tmp_path / "chair.png"
    image.write_bytes(b"png")
    glb = tmp_path / "chair.glb"
    glb.write_bytes(b"glb")
    fake_client = _FakeTrellisClient(TrellisMeshResult(glb, None, None, 1))
    gateway = _FakeGateway()
    monkeypatch.setattr(
        "bim_recon.trellis_workflow.compute_placement_transform",
        lambda _placement: object(),
    )
    monkeypatch.setattr(
        "bim_recon.trellis_workflow.register_mesh_in_revit",
        lambda _placement, _transform: {
            "vertex_count": 8,
            "face_count": 12,
            "payload_path": str(tmp_path / "payload.json"),
        },
    )
    workflow = TrellisRevitWorkflow(
        TrellisWorkflowConfig(
            objects=(ApprovedMeshObject(
                object_id="obj_001",
                label="chair",
                image_path=image,
                position_3d=(1.0, 2.0, 0.5),
            ),),
            output_dir=tmp_path / "meshes",
            register_in_revit=True,
        ),
        client_factory=lambda: fake_client,
        gateway=gateway,
    )

    events = list(stream_workflow_sync(workflow))

    completed = next(event for event in events if isinstance(event, WorkflowCompleted))
    assert completed.result["completed"] == 1
    assert fake_client.requests[0].name == "chair_obj_001"
    assert gateway.calls[-1][0] == "create_directshape_from_mesh"


def test_reconstruction_workflow_surfaces_failed_preflight_without_heavy_steps(
    monkeypatch,
):
    class OfflineFalcon:
        def __init__(self, **_kwargs):
            pass

        def health(self):
            return False

    monkeypatch.setattr("bim_recon.pipeline_runner.FalconClient", OfflineFalcon)
    workflow = ReconstructionWorkflow(PipelineConfig(name="missing"))

    events = list(stream_workflow_sync(workflow))

    failed = next(event for event in events if isinstance(event, WorkflowFailed))
    assert failed.stage == "prepare"
    assert "unreachable" in failed.message


def test_gradio_preprocess_callback_runs_uncached_pipeline(
    tmp_path, monkeypatch,
):
    """The uncached callback must copy the upload and launch both subprocesses."""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    from scripts import gradio_app

    source = tmp_path / "uploaded.ply"
    source.write_bytes(b"ply\n")
    scenesplat = tmp_path / "SceneSplat"
    checkpoint = (
        scenesplat
        / "ckpt"
        / "lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(gradio_app, "ROOT", tmp_path)
    monkeypatch.setattr(gradio_app, "SCENESPLAT", scenesplat)
    monkeypatch.setattr(
        gradio_app,
        "check_preprocess_status",
        lambda _name: {"feat_pt_exists": False},
    )
    monkeypatch.setattr(
        gradio_app.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd, check)),
    )

    app = gradio_app.build_app()
    callback = next(
        block_fn.fn
        for block_fn in app.fns.values()
        if getattr(block_fn.fn, "__name__", "") == "on_preprocess"
    )

    status, scene_name = callback(str(source), "new-scene")

    copied = tmp_path / "data" / "new-scene" / source.name
    assert copied.read_bytes() == source.read_bytes()
    assert scene_name == "new-scene"
    assert "预处理完成" in status
    assert len(calls) == 2
    assert calls[0][0][2] == "scripts.preprocess_gs"
    assert calls[1][0][2] == "tools.lang_inference"
    assert all(cwd == str(scenesplat) and check for _cmd, cwd, check in calls)
