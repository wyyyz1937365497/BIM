#!/usr/bin/env python
"""Benchmark a pose-refiner checkpoint on deterministic procedural examples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from bim_recon.pose_refiner import PoseRefinerNet
from bim_recon.pose_refiner_synthetic import (
    SyntheticPoseConfig,
    SyntheticPoseDataset,
    evaluate_pose_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.smoke:
        args.samples = 4
        args.batch_size = 2
    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model = PoseRefinerNet().to(device)
    model.load_state_dict(state_dict)
    config = SyntheticPoseConfig()
    dataset = SyntheticPoseDataset(args.samples, seed=args.seed, config=config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_pose_model(
        model,
        loader,
        device,
        config,
        confidence_threshold=args.confidence_threshold,
    )
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "confidence_threshold": args.confidence_threshold,
    })
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    print(payload)
    output_path = args.output or args.checkpoint.with_suffix(".benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
