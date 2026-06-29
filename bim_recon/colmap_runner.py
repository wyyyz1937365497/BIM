"""COLMAP data pipeline wrapper for 3DGS training.

Wraps nerfstudio's ``ns-process-data images`` command, which runs COLMAP
SfM (feature extraction, matching, sparse mapping) and produces a
``transforms.json`` with camera poses plus an ``images/`` folder.

Pipeline:
    raw images (or video frames)
        -> COLMAP SfM (features + matches + mapper)
        -> transforms.json (nerfstudio camera poses)
        -> ready for ns-train splatfacto

Run with:
    python -m bim_recon.colmap_runner --images path/to/images --output path/to/output

For video input, extract frames first (ffmpeg) or use --video flag.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ColmapResult:
    """Result of a COLMAP / ns-process-data run."""

    output_dir: Path
    transforms_json: Path
    images_dir: Path
    num_frames: int
    num_cameras: int


def run_colmap(
    images: Path,
    output: Path,
    matching_method: str = "vocab_tree",
    num_downscales: int = 2,
    max_size: Optional[int] = 1600,
    skip_colmap: bool = False,
    dry_run: bool = False,
) -> ColmapResult:
    """Run the nerfstudio image processing pipeline.

    Args:
        images: Directory of input images (JPG/PNG).
        output: Output directory (created if missing).
        matching_method: COLMAP matching method — "exhaustive", "sequential",
            "vocab_tree", or "spatial". Indoor rooms with handheld capture
            typically work well with "vocab_tree" or "sequential".
        num_downscales: How many times to downscale images for training.
            2 means the largest training images are 1/4 resolution.
        max_size: Cap the longest image edge at this pixel count. 1600 is a
            good default for 2080Ti (22GB) with indoor scenes.
        skip_colmap: If True, reuse an existing transforms.json and skip SfM.
        dry_run: If True, print the command without executing.

    Returns:
        ColmapResult with paths to the produced artifacts.
    """
    images = Path(images).resolve()
    output = Path(output).resolve()

    if not images.exists():
        raise FileNotFoundError(f"Input images dir not found: {images}")

    output.mkdir(parents=True, exist_ok=True)

    cmd = _build_ns_process_cmd(
        images=images,
        output=output,
        matching_method=matching_method,
        num_downscales=num_downscales,
        max_size=max_size,
        skip_colmap=skip_colmap,
    )
    print("Running COLMAP pipeline:")
    print("  " + " ".join(str(c) for c in cmd))

    if dry_run:
        print("(dry-run; not executing)")
        return ColmapResult(
            output_dir=output,
            transforms_json=output / "transforms.json",
            images_dir=output / "images",
            num_frames=0,
            num_cameras=0,
        )

    # ns-process-data writes progress to stderr; let it stream through.
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ns-process-data failed with exit code {result.returncode}. "
            "Check that COLMAP is installed and on PATH."
        )

    transforms = output / "transforms.json"
    if not transforms.exists():
        raise RuntimeError(f"Expected output not found: {transforms}")

    num_frames = _count_frames(transforms)
    images_dir = output / "images"
    if not images_dir.exists():
        # nerfstudio sometimes uses "images_2" etc. for downscaled versions
        candidates = sorted(output.glob("images*"))
        images_dir = candidates[0] if candidates else output

    print(f"\nCOLMAP pipeline done:")
    print(f"  transforms: {transforms}")
    print(f"  cameras:    {num_frames}")
    print(f"  images dir: {images_dir}")
    return ColmapResult(
        output_dir=output,
        transforms_json=transforms,
        images_dir=images_dir,
        num_frames=num_frames,
        num_cameras=num_frames,
    )


def _build_ns_process_cmd(
    images: Path,
    output: Path,
    matching_method: str,
    num_downscales: int,
    max_size: Optional[int],
    skip_colmap: bool,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "nerfstudio.scripts.process_data",
        "images",
        "--data", str(images),
        "--output-dir", str(output),
        "--matching-method", matching_method,
        "--num-downscales", str(num_downscales),
    ]
    if max_size is not None:
        cmd.extend(["--max-size", str(max_size)])
    if skip_colmap:
        cmd.append("--skip-colmap")
    return cmd


def _count_frames(transforms_json: Path) -> int:
    with open(transforms_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("frames", []))


def _check_colmap_installed() -> bool:
    return shutil.which("colmap") is not None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run COLMAP SfM via nerfstudio")
    p.add_argument("--images", required=True, help="Input images directory")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--matching-method", default="vocab_tree",
                   choices=["exhaustive", "sequential", "vocab_tree", "spatial"],
                   help="COLMAP matcher (default: vocab_tree)")
    p.add_argument("--num-downscales", type=int, default=2,
                   help="Downscale levels for training (default: 2)")
    p.add_argument("--max-size", type=int, default=1600,
                   help="Max image edge in pixels (default: 1600; 0 to skip)")
    p.add_argument("--skip-colmap", action="store_true",
                   help="Reuse existing transforms.json; skip SfM step")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without executing")
    args = p.parse_args(argv)

    if not _check_colmap_installed():
        print("ERROR: COLMAP not found on PATH. Install COLMAP first.", file=sys.stderr)
        return 2

    max_size = args.max_size if args.max_size > 0 else None
    run_colmap(
        images=Path(args.images),
        output=Path(args.output),
        matching_method=args.matching_method,
        num_downscales=args.num_downscales,
        max_size=max_size,
        skip_colmap=args.skip_colmap,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
