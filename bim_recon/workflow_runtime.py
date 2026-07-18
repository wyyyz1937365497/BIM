"""Runtime adapters shared by CLI and Gradio workflow consumers."""
from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass, field
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from workflows import Workflow
from workflows.events import Event, StopEvent

from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
    WorkflowWarning,
    progress_message,
)


async def stream_workflow(workflow: Workflow, **run_kwargs: Any) -> AsyncIterator[Event]:
    """Yield all public events from one workflow run.

    The workflow itself emits a :class:`WorkflowCompleted` event before its
    terminal ``StopEvent`` so callers do not need an async-generator return value.
    Awaiting the handler after the stream ensures execution errors propagate.
    """
    handler = workflow.run(**run_kwargs)
    async for event in handler.stream_events():
        if not isinstance(event, StopEvent):
            yield event
    await handler


def stream_workflow_sync(workflow: Workflow, **run_kwargs: Any) -> Iterator[Event]:
    """Bridge an async workflow event stream into a synchronous generator.

    Existing Gradio callbacks and CLI entry points are synchronous generators.
    A dedicated thread owns the event loop, while a bounded queue provides
    backpressure and preserves event order.
    """
    events: queue.Queue[Event | BaseException | object] = queue.Queue(maxsize=32)
    sentinel = object()

    async def pump() -> None:
        try:
            async for event in stream_workflow(workflow, **run_kwargs):
                events.put(event)
        except BaseException as exc:
            events.put(exc)
        finally:
            events.put(sentinel)

    def runner() -> None:
        asyncio.run(pump())

    thread = threading.Thread(target=runner, name=f"{type(workflow).__name__}-runner", daemon=True)
    thread.start()
    while True:
        item = events.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    thread.join()


@dataclass(frozen=True, slots=True)
class WorkflowUIUpdate:
    """Framework-neutral update consumed by Gradio callbacks."""

    kind: str
    message: str
    stage: str
    payload: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None


async def stream_workflow_gradio(
    workflow: Workflow,
    **run_kwargs: Any,
) -> AsyncIterator[WorkflowUIUpdate]:
    """Map typed workflow events to one stable Gradio streaming contract."""
    try:
        async for event in stream_workflow(workflow, **run_kwargs):
            if isinstance(event, WorkflowProgress):
                yield WorkflowUIUpdate(
                    kind="progress",
                    message=progress_message(event),
                    stage=event.stage,
                    payload=event.payload,
                    image_path=event.image_path,
                )
            elif isinstance(event, WorkflowWarning):
                yield WorkflowUIUpdate(
                    kind="warning",
                    message=progress_message(event),
                    stage=event.stage,
                    payload=event.payload,
                )
            elif isinstance(event, WorkflowFailed):
                yield WorkflowUIUpdate(
                    kind="failed",
                    message=progress_message(event),
                    stage=event.stage,
                    payload=event.payload,
                )
            elif isinstance(event, WorkflowCompleted):
                yield WorkflowUIUpdate(
                    kind="completed",
                    message="Workflow complete",
                    stage="complete",
                    payload=event.result,
                )
    except Exception as exc:
        yield WorkflowUIUpdate(
            kind="failed",
            message=f"ERROR [workflow] {exc}",
            stage="workflow",
            payload={"error": str(exc)},
        )
