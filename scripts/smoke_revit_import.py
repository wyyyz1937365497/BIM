"""End-to-end smoke test for the Revit MCP import flow.

Loads a cleaned reconstruction output, runs ``RevitBuildWorkflow`` against the
live Revit MCP plugin (with a configurable spatial offset so the smoke test
geometry never collides with the user's existing model), verifies that the
created IDs are visible in the active view, and deletes every element it
created.  Exits non-zero on any failure so it can be wired into CI or run
standalone::

    G:/Miniconda3/envs/bim-recon/python.exe scripts/smoke_revit_import.py \
        --output output/splat/20260719_151424 --offset-x 100000 --offset-y 100000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.pipeline_api import load_results
from bim_recon.revit_workflow import RevitBuildOptions, RevitBuildWorkflow
from bim_recon.workflow_events import WorkflowCompleted, WorkflowFailed
from bim_recon.workflow_runtime import stream_workflow_sync


async def _run(gateway, workflow):
    return list(stream_workflow_sync(workflow))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-x", type=float, default=100000.0)
    parser.add_argument("--offset-y", type=float, default=100000.0)
    parser.add_argument("--level-elevation", type=float, default=10000.0)
    parser.add_argument("--keep", action="store_true",
                        help="Keep created elements instead of deleting them")
    args = parser.parse_args()

    from bim_recon.mcp_gateway import StdioMCPGateway

    results = load_results(args.output)
    confirmed = [el for el in results.elements if el.confirmed]
    print(f"Loaded {len(results.walls)} walls, "
          f"{len(confirmed)} confirmed elements from {args.output}")

    gateway = StdioMCPGateway(
        command="node",
        args=("G:\\TJ\\BIM\\mcp-servers-for-revit\\server\\build\\index.js",),
        cwd="G:\\TJ\\BIM",
        timeout_seconds=180.0,
    )
    options = RevitBuildOptions(
        level_name="BIM-Recon Smoke Import",
        level_elevation=args.level_elevation,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
    )
    workflow = RevitBuildWorkflow(results, gateway, options)

    events = list(stream_workflow_sync(workflow))
    failed = next((e for e in events if isinstance(e, WorkflowFailed)), None)
    completed = next((e for e in events if isinstance(e, WorkflowCompleted)), None)
    if failed or completed is None:
        print("WORKFLOW FAILED:", failed.message if failed else "no completion event")
        return 2

    result = completed.result
    created = result["created"]
    summary = {key: len(value) for key, value in created.items()}
    print("Created:", summary)
    print("Missing IDs:", result["missing_ids"])
    if result["missing_ids"]:
        return 3

    if args.keep:
        print("Keeping created elements (pass without --keep to clean up)")
        return 0

    to_delete = [eid for ids in created.values() for eid in ids]
    if to_delete:
        try:
            asyncio.run(gateway.call_tool(
                "delete_element",
                {"elementIds": [str(eid) for eid in to_delete]},
            ))
            print(f"Deleted {len(to_delete)} smoke-test elements")
        except Exception as exc:
            print(f"Cleanup failed: {exc}")
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
