#!/usr/bin/env python
"""3DGS → BIM pipeline CLI entry point.

Thin wrapper around ``bim_recon.pipeline_runner.run_pipeline()``.
The actual pipeline logic lives in the importable module so both CLI
and Gradio UI share the same code path.

Usage:
    cmd /c "\"...\\vcvars64.bat\" && python scripts/run_pipeline.py --name splat"
    cmd /c "\"...\\vcvars64.bat\" && python scripts/run_pipeline.py --name splat --skip-vlm"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.config import load_config
from bim_recon.pipeline_runner import PipelineConfig, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="3DGS → BIM Pipeline")
    parser.add_argument("--name", required=True, help="Scene name in data/")
    parser.add_argument("--elements", nargs="+", default=["door", "window"],
                        help="Element types to detect")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM verification")
    parser.add_argument("--vlm-api-base", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-api-key", default=None)
    parser.add_argument("--falcon-host", default="127.0.0.1")
    parser.add_argument("--falcon-port", type=int, default=8390)
    parser.add_argument("--num-heights", type=int, default=8)
    parser.add_argument("--snap-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load VLM config (CLI args override config.json)
    app_config = load_config()
    vlm = app_config.vlm
    config = PipelineConfig(
        name=args.name,
        elements=args.elements,
        skip_vlm=args.skip_vlm,
        vlm_api_base=args.vlm_api_base or vlm.api_base,
        vlm_model=args.vlm_model or vlm.model,
        vlm_api_key=args.vlm_api_key if args.vlm_api_key is not None else vlm.api_key,
        falcon_host=args.falcon_host,
        falcon_port=args.falcon_port,
        num_heights=args.num_heights,
        snap_threshold=args.snap_threshold,
    )

    print(f"{'='*60}")
    print(f"3DGS → BIM Pipeline: {args.name}")
    print(f"{'='*60}")

    for msg, data in run_pipeline(config):
        print(msg, flush=True)
        if msg.startswith("ERROR"):
            return 1

    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    if "out_dir" in data:
        print(f"Output: {data['out_dir']}")
    if "confirmed_count" in data:
        print(f"Confirmed elements: {data['confirmed_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
