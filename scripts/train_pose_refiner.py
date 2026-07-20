#!/usr/bin/env python
"""Train the B-class residual pose refiner on procedural RGB-D-mask pairs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from bim_recon.pose_refiner import PoseRefinerNet, pose_refiner_loss
from bim_recon.pose_refiner_synthetic import (
    SyntheticPoseConfig,
    SyntheticPoseDataset,
    evaluate_pose_model,
    move_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "pose_refiner.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.smoke:
        args.epochs = 1
        args.train_samples = 8
        args.val_samples = 4
        args.batch_size = 2

    synthetic_config = SyntheticPoseConfig()
    train_dataset = SyntheticPoseDataset(
        args.train_samples, seed=args.seed, config=synthetic_config,
    )
    validation_dataset = SyntheticPoseDataset(
        args.val_samples, seed=args.seed + 10_000_019, config=synthetic_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = PoseRefinerNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1),
    )

    best_rotation_error = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["observed"],
                batch["candidate"],
                batch["mesh_features"],
                batch["metadata"],
            )
            loss, _ = pose_refiner_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
        scheduler.step()

        metrics = evaluate_pose_model(
            model, validation_loader, device, synthetic_config,
        )
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = loss_sum / max(batch_count, 1)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

        if metrics["rotation_error_deg_mean"] <= best_rotation_error:
            best_rotation_error = metrics["rotation_error_deg_mean"]
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "synthetic_config": asdict(synthetic_config),
                "training_args": vars(args),
            }, args.output)

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"checkpoint={args.output}")
    print(f"history={history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
