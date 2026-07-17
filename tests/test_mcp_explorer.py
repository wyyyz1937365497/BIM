from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import torch
from mcp.server.fastmcp.utilities.types import Image as MCPImage

import bim_recon.mcp_explorer as explorer


class _FakeScene:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.means = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        self.num_gaussians = 1

    def render(self, pose, width: int, height: int, fov_degrees: float):
        return SimpleNamespace(
            colors=np.full((height, width, 3), 0.5, dtype=np.float32),
            depth=np.ones((height, width), dtype=np.float32),
            alpha=np.ones((height, width), dtype=np.float32),
        )

    def select_by_mask(self, pose, mask, width: int, height: int, fov: float):
        return np.array([0], dtype=np.int64)


class _FakeFalcon:
    def health(self) -> bool:
        return True

    def segment(self, image, query: str, task: str):
        return [SimpleNamespace(
            bbox={"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
            mask_area_ratio=0.1,
        )]


def _call(manager, name: str, arguments: dict):
    return asyncio.run(manager.call_tool(name, arguments))


def test_all_explorer_tools_and_queue_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(explorer, "_get_falcon_client", lambda *args, **kwargs: _FakeFalcon())
    state = explorer.ExplorerState(
        scene=_FakeScene(),
        explore_dir=tmp_path / "explore",
        width=64,
        height=48,
        up_axis=2,
        bounds_min=(-2.0, -2.0, -1.0),
        bounds_max=(2.0, 2.0, 1.0),
    )
    manager = explorer.build_server(state)._tool_manager

    expected = {
        "explore_init", "turn", "step", "look_at", "get_status",
        "detect_objects", "tag_object", "find_best_angle", "list_found",
        "queue_for_trellis",
    }
    assert {tool.name for tool in manager.list_tools()} == expected

    init_tool = next(tool for tool in manager.list_tools() if tool.name == "explore_init")
    assert {"eye_x", "eye_y", "eye_z"} <= set(init_tool.parameters["required"])
    for coordinate in ("eye_x", "eye_y", "eye_z"):
        assert init_tool.parameters["properties"][coordinate]["type"] == "number"

    initial = _call(manager, "explore_init", {
        "eye_x": 0.11,
        "eye_y": -0.09,
        "eye_z": 0.0,
        "initial_yaw": -157.4,
    })
    assert isinstance(initial, MCPImage)
    assert state.cam_eye == [0.11, -0.09, 0.0]

    assert isinstance(_call(manager, "turn", {"yaw_degrees": 10.0}), MCPImage)
    assert isinstance(_call(manager, "step", {"direction": "forward", "distance": 0.05}), MCPImage)
    assert isinstance(_call(manager, "look_at", {
        "eye_x": 0.0, "eye_y": 0.0, "eye_z": 0.0,
        "target_x": 1.0, "target_y": 0.0, "target_z": 0.0,
        "fov": 60.0,
    }), MCPImage)

    status = json.loads(_call(manager, "get_status", {}))
    assert status["camera"]["eye"] == [0.0, 0.0, 0.0]

    detected = json.loads(_call(manager, "detect_objects", {"query": "chair"}))
    assert detected["count"] == 1

    tagged = json.loads(_call(manager, "tag_object", {
        "label": "chair", "bbox_x": 0.4, "bbox_y": 0.4,
        "bbox_w": 0.2, "bbox_h": 0.2,
    }))
    assert tagged["status"] == "tagged"
    object_id = tagged["id"]

    angles = json.loads(_call(manager, "find_best_angle", {
        "object_id": object_id, "num_angles": 2, "radius": 0.5,
    }))
    assert len(angles["results"]) == 2

    found = json.loads(_call(manager, "list_found", {}))
    assert found["total"] == 1

    queued = json.loads(_call(manager, "queue_for_trellis", {"object_id": object_id}))
    assert queued["status"] == "queued"
    assert (tmp_path / "trellis_queue" / f"{object_id}.json").is_file()
    persisted = json.loads((tmp_path / "explore" / "found_objects.json").read_text("utf-8"))
    assert persisted[0]["trellis_status"] == "queued"
