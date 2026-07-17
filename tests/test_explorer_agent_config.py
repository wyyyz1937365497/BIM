"""Explorer agent model-routing regression tests."""
from __future__ import annotations

from types import SimpleNamespace

import smolagents

import bim_recon.gradio_helpers as helpers


class _FakeToolCollectionContext:
    def __enter__(self):
        return SimpleNamespace(tools=[object()])

    def __exit__(self, *_args):
        return None


class _FakeToolCollection:
    @staticmethod
    def from_mcp(*_args, **_kwargs):
        return _FakeToolCollectionContext()


class _FakeToolCallingAgent:
    def __init__(self, *, tools, model, instructions, max_steps):
        self.tools = tools
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps


def test_explorer_agent_uses_vlm_model(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "scene"
    data_dir.mkdir(parents=True)
    (data_dir / "point_cloud.ply").write_bytes(b"ply\n")
    vision_model = object()

    monkeypatch.setattr(helpers, "ROOT", tmp_path)
    monkeypatch.setattr(helpers, "get_vlm_model", lambda _cfg: vision_model)
    monkeypatch.setattr(
        helpers,
        "get_llm_model",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("text LLM must not be used")),
    )
    monkeypatch.setattr(smolagents, "ToolCollection", _FakeToolCollection)
    monkeypatch.setattr(smolagents, "ToolCallingAgent", _FakeToolCallingAgent)
    helpers._reset_explorer_agent()

    try:
        agent = helpers._get_explorer_agent("scene")
    finally:
        helpers._reset_explorer_agent()

    assert agent.model is vision_model
    assert agent.max_steps == 50
