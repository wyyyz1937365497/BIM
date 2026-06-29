"""Train a 3DGS model with nerfstudio's splatfacto.

Thin wrapper around ``ns-train splatfacto`` with sensible indoor-scene
defaults (depth regularization, moderate resolution). Run after COLMAP has
produced transforms.json (see :mod:`bim_recon.colmap_runner`).

Usage:
    python scripts/train_gs.py --data path/to/processed --output path/to/output

After training, export the .ply for the MCP server:
    ns-export gaussian-splat --load-config path/to/config.yml \\
        --output-dir path/to/output
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def train_splatfacto(
    data: Path,
    output: Path,
    max_iters: int = 30000,
    max_resolution: int = 1600,
    depth_loss_mult: float = 0.1,
    eval_every: int = 5000,
    dry_run: bool = False,
) -> Path:
    """Launch ns-train splatfacto.

    Args:
        data: Directory containing transforms.json + images/.
        output: Where to store checkpoints and logs.
        max_iters: Training iterations. 30k is a good default; 15k for fast
            preview, 50k for production.
        max_resolution: Cap on longest image edge. 1600 for 2080Ti (22GB);
            lower if OOM occurs.
        depth_loss_mult: Depth supervision weight. 0.1 adds mild depth
            regularization without needing depth sensors. Set 0.0 to disable.
        eval_every: How often to log metrics / save checkpoints.
        dry_run: If True, print the command without executing.

    Returns:
        Path to the experiment output directory.
    """
    data = Path(data).resolve()
    output = Path(output).resolve()

    transforms = data / "transforms.json"
    if not transforms.exists():
        raise FileNotFoundError(f"transforms.json not found in data dir: {data}")

    output.mkdir(parents=True, exist_ok=True)

    cmd = _build_ns_train_cmd(
        data=data,
        output=output,
        max_iters=max_iters,
        max_resolution=max_resolution,
        depth_loss_mult=depth_loss_mult,
        eval_every=eval_every,
    )
    print("Training 3DGS (splatfacto):")
    print("  " + " ".join(str(c) for c in cmd))

    if dry_run:
        print("(dry-run; not executing)")
        return output

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"ns-train failed with exit code {result.returncode}")

    print(f"\nTraining complete. Output: {output}")
    print(f"Next: export the .ply with:")
    print(f"  ns-export gaussian-splat --load-config {output}/config.yml --output-dir {output}/splat")
    return output


def _build_ns_train_cmd(
    data: Path,
    output: Path,
    max_iters: int,
    max_resolution: int,
    depth_loss_mult: float,
    eval_every: int,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "nerfstudio.scripts.train",
        "splatfacto",
        "--data", str(data),
        "--output-dir", str(output),
        "--max-num-iterations", str(max_iters),
        "--max-resolution", str(max_resolution),
        "--pipeline.datamanager.dataparser.data", str(data),
        "--vis", "wandb",  # set to "viewer" to launch the live viewer
        "--relative-model-dir", "splatfacto",
    ]
    # Depth regularization: splatfacto-n-big uses depth loss.
    # The flag name varies by nerfstudio version; we set it via the model config.
    if depth_loss_mult > 0:
        cmd.extend(["--pipeline.model.depth-loss-mult", str(depth_loss_mult)])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train 3DGS via nerfstudio splatfacto")
    p.add_argument("--data", required=True,
                   help="Data dir with transforms.json + images/")
    p.add_argument("--output", required=True,
                   help="Output directory for checkpoints + logs")
    p.add_argument("--max-iters", type=int, default=30000,
                   help="Training iterations (default: 30000)")
    p.add_argument("--max-resolution", type=int, default=1600,
                   help="Max image edge pixels (default: 1600)")
    p.add_argument("--depth-loss-mult", type=float, default=0.1,
                   help="Depth regularization weight (default: 0.1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without executing")
    args = p.parse_args(argv)

    train_splatfacto(
        data=Path(args.data),
        output=Path(args.output),
        max_iters=args.max_iters,
        max_resolution=args.max_resolution,
        depth_loss_mult=args.depth_loss_mult,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
