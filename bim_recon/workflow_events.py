"""Shared event contracts for deterministic BIM workflows.

Only small, serializable values belong in these events. GPU models, loaded scenes,
MCP sessions, and HTTP clients stay on workflow instances as runtime resources.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field
from workflows.events import Event


class WorkflowProgress(Event):
    """A user-visible progress update emitted by any BIM workflow."""

    workflow: str
    stage: str
    message: str
    current: int | None = None
    total: int | None = None
    image_path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowWarning(Event):
    """A recoverable problem that should be visible without stopping the run."""

    workflow: str
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowFailed(Event):
    """A terminal workflow failure with a stable machine-readable stage."""

    workflow: str
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowCompleted(Event):
    """The final serializable result surfaced to UI and CLI consumers."""

    workflow: str
    result: dict[str, Any] = Field(default_factory=dict)


def progress_message(
    event: WorkflowProgress | WorkflowWarning | WorkflowFailed,
) -> str:
    """Format a stable console line for Gradio and CLI adapters."""
    if isinstance(event, WorkflowFailed):
        return f"ERROR [{event.stage}] {event.message}"
    if isinstance(event, WorkflowWarning):
        return f"WARN [{event.stage}] {event.message}"
    if event.current is not None and event.total:
        return f"[{event.stage}] {event.current}/{event.total} {event.message}"
    return f"[{event.stage}] {event.message}"
