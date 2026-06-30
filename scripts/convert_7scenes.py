"""Convert Microsoft 7-Scenes RGB-D dataset to nerfstudio transforms.json format.

Key conversions:
  1. Pose: 7-Scenes uses OpenGL camera convention (+x right, +y up, -z forward).
           nerfstudio uses OpenCV convention (+x right, +y down, +z forward).
           Conversion: c2w_opencv = c2w_opengl @ diag(1, -1, -1, 1)
  2. Depth: 7-Scenes stores uint16 millimetres; nerfstudio reads depth and
            multiplies by depth_unit_scale_factor. We set it to 0.001 (mm -> m).
            Invalid depth (65535) is left as-is; nerfstudio masks it.
  3. Intrinsics: dataset ships uncalibrated; use the published defaults
                 (fx=fy=585, cx=320, cy=240 for 640x480).

Usage:
    python scripts/convert_7scenes.py \\
        --input  data/office \\
        --output data/office_ns \\
        --stride 5

The output directory contains:
    transforms.json       (training sequences)
    transforms_test.json  (test sequences, for evaluation)
No image copies are made; transforms.json references the originals via relative paths.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------- constants ------------------------------------

# Kinect default intrinsics per the 7-Scenes documentation.
FX = 585.0
FY = 585.0
CX = 320.0
CY = 240.0
IMG_W = 640
IMG_H = 480

# mm-to-metres scale factor for nerfstudio's depth loader.
DEPTH_UNIT_SCALE = 0.001

# OpenGL -> OpenCV axis flip (negate y and z columns of the c2w rotation).
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

INVALID_DEPTH = 65535


# ----------------------------- pose parsing ---------------------------------


def parse_pose(pose_path: Path) -> np.ndarray:
    """Read a 4x4 camera-to-world matrix from a 7-Scenes .pose.txt file.

    The file contains 4 rows of 4 whitespace-separated floats in scientific
    notation. Returns the matrix in OpenGL convention (as stored).
    """
    text = pose_path.read_text().strip()
    rows = re.split(r"[\n\r]+", text)
    values: List[float] = []
    for row in rows:
        toks = row.split()
        if len(toks) != 4:
            continue
        values.extend(float(t) for t in toks)
    if len(values) != 16:
        raise ValueError(f"Pose file {pose_path} has {len(values)} values, expected 16")
    return np.array(values, dtype=np.float64).reshape(4, 4)


def opengl_to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    """Convert a camera-to-world matrix from OpenGL to OpenCV convention."""
    return c2w_gl @ _GL2CV


# ----------------------------- sequence scanning ----------------------------


def scan_sequence(
    seq_dir: Path,
    stride: int = 1,
) -> List[Tuple[int, Path, Path, np.ndarray]]:
    """Walk a seq-XX directory and return (frame_idx, color_path, depth_path, c2w_opencv).

    pose is converted to OpenCV convention here so downstream code doesn't
    need to worry about the axis flip.
    """
    color_files = sorted(seq_dir.glob("frame-*.color.png"))
    out: List[Tuple[int, Path, Path, np.ndarray]] = []
    for i, color_path in enumerate(color_files):
        if i % stride != 0:
            continue
        stem = color_path.name.removesuffix(".color.png")  # e.g. "frame-000123"
        depth_path = seq_dir / f"{stem}.depth.png"
        pose_path = seq_dir / f"{stem}.pose.txt"
        if not depth_path.exists() or not pose_path.exists():
            print(f"  WARN: missing depth/pose for {stem}, skipping", file=sys.stderr)
            continue
        c2w_gl = parse_pose(pose_path)
        c2w_cv = opengl_to_opencv(c2w_gl)
        out.append((i, color_path, depth_path, c2w_cv))
    return out


def parse_split_file(split_path: Path) -> List[str]:
    """Read TrainSplit.txt / TestSplit.txt and return seq directory names.

    The file contains lines like 'sequence1', 'sequence3', etc. We map
    'sequenceN' -> 'seq-NN' (zero-padded).
    """
    names: List[str] = []
    for line in split_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"sequence(\d+)", line)
        if m:
            names.append(f"seq-{int(m.group(1)):02d}")
        else:
            names.append(line)
    return names


# ----------------------------- transforms.json builder ----------------------


def build_transforms(
    frames: List[Dict],
    output_dir: Path,
    input_dir: Path,
) -> Dict:
    """Assemble the nerfstudio transforms.json dict.

    ``frames`` is a list of per-frame dicts with keys:
        color_path, depth_path, c2w (4x4 np.ndarray, OpenCV convention)
    Paths are stored relative to output_dir (where transforms.json lives).
    """
    frame_entries: List[Dict] = []
    for fr in frames:
        color_rel = _relative_to(fr["color_path"], output_dir)
        depth_rel = _relative_to(fr["depth_path"], output_dir)
        # nerfstudio expects row-major list for transform_matrix
        c2w_list = fr["c2w"].tolist()
        entry = {
            "file_path": color_rel,
            "depth_file_path": depth_rel,
            "transform_matrix": c2w_list,
        }
        frame_entries.append(entry)

    return {
        "camera_model": "PINHOLE",
        "fl_x": FX,
        "fl_y": FY,
        "cx": CX,
        "cy": CY,
        "w": IMG_W,
        "h": IMG_H,
        "depth_unit_scale_factor": DEPTH_UNIT_SCALE,
        "frames": frame_entries,
    }


def _relative_to(target: Path, base: Path) -> str:
    """Return a forward-slash relative path from base to target.

    Unlike Path.relative_to, this handles paths that require going up with '..'
    (e.g. output dir is a sibling of the input dir).
    """
    import os
    rel = os.path.relpath(str(target.resolve()), str(base.resolve()))
    return rel.replace("\\", "/")


# ----------------------------- depth stats (sanity) ------------------------


def quick_depth_check(depth_path: Path) -> Dict:
    """Load a depth PNG and report basic stats (for sanity logging)."""
    try:
        from PIL import Image
        import numpy as np
        d = np.array(Image.open(depth_path), dtype=np.uint16)
        valid = d < INVALID_DEPTH
        if valid.any():
            return {
                "min_mm": int(d[valid].min()),
                "max_mm": int(d[valid].max()),
                "mean_mm": float(d[valid].mean()),
                "invalid_frac": float((~valid).mean()),
            }
        return {"min_mm": 0, "max_mm": 0, "mean_mm": 0, "invalid_frac": 1.0}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------- main -----------------------------------------


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert 7-Scenes to nerfstudio transforms.json")
    p.add_argument("--input", required=True, type=Path,
                   help="7-Scenes scene directory (e.g. data/office)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory for transforms.json files")
    p.add_argument("--stride", type=int, default=5,
                   help="Frame subsampling stride (default: 5, i.e. every 5th frame). "
                        "Use 1 for all frames. 7-Scenes sequences are 500-1000 frames; "
                        "stride=5 gives ~100-200 frames per sequence, suitable for 3DGS.")
    p.add_argument("--train-only", action="store_true",
                   help="Skip test split (only write transforms.json)")
    args = p.parse_args(argv)

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 2

    # Parse split files
    train_split_path = input_dir / "TrainSplit.txt"
    test_split_path = input_dir / "TestSplit.txt"
    if not train_split_path.exists():
        print(f"ERROR: TrainSplit.txt not found in {input_dir}", file=sys.stderr)
        return 2

    train_seqs = parse_split_file(train_split_path)
    test_seqs = parse_split_file(test_split_path) if test_split_path.exists() else []
    print(f"Train sequences: {train_seqs}")
    print(f"Test  sequences: {test_seqs}")

    # --- Build training transforms ---
    train_frames: List[Dict] = []
    for seq_name in train_seqs:
        seq_dir = input_dir / seq_name
        if not seq_dir.exists():
            print(f"  WARN: {seq_dir} not found, skipping", file=sys.stderr)
            continue
        scanned = scan_sequence(seq_dir, stride=args.stride)
        print(f"  {seq_name}: {len(scanned)} frames (stride={args.stride})")
        for idx, color_p, depth_p, c2w in scanned:
            train_frames.append({"color_path": color_p, "depth_path": depth_p, "c2w": c2w})

    if not train_frames:
        print("ERROR: no training frames found", file=sys.stderr)
        return 1

    # Sanity-check depth on the first frame
    stats = quick_depth_check(train_frames[0]["depth_path"])
    print(f"Depth sanity (first train frame): {stats}")

    train_tf = build_transforms(train_frames, output_dir, input_dir)
    train_path = output_dir / "transforms.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_tf, f, indent=2)
    print(f"\nWrote {len(train_frames)} train frames -> {train_path}")

    # --- Build test transforms (optional) ---
    if not args.train_only and test_seqs:
        test_frames: List[Dict] = []
        for seq_name in test_seqs:
            seq_dir = input_dir / seq_name
            if not seq_dir.exists():
                continue
            scanned = scan_sequence(seq_dir, stride=args.stride)
            print(f"  {seq_name}: {len(scanned)} frames (stride={args.stride})")
            for idx, color_p, depth_p, c2w in scanned:
                test_frames.append({"color_path": color_p, "depth_path": depth_p, "c2w": c2w})
        if test_frames:
            test_tf = build_transforms(test_frames, output_dir, input_dir)
            test_path = output_dir / "transforms_test.json"
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_tf, f, indent=2)
            print(f"Wrote {len(test_frames)} test frames -> {test_path}")

    print("\nDone. Next steps:")
    print(f"  Train: ns-train splatfacto --data {output_dir} --output-dir <your-output>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
