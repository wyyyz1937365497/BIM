#!/usr/bin/env python
"""Export a PLY containing only the furniture-class gaussians.

Loads the full scene PLY + feat.pt, classifies each gaussian via the 9-class
warm-cache SigLIP2 vocabulary, and writes a new PLY with only the vertices
whose dominant class is "furniture".  The output preserves the exact binary
layout of the input (all SH coefficients, log-scale, quaternion, etc.) so it
can be opened in any 3DGS viewer (nerfview, super-splat, etc.).

Usage::

    python scripts/export_furniture_ply.py
    python scripts/export_furniture_ply.py --ply data/point_cloud_30000.ply \\
        --feat output/point_cloud_30000/point_cloud_30000_feat.pt \\
        --output output/furniture_only.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bim_recon.gs_scene import _read_ply_header, _parse_ply_header_text
from bim_recon.semantics import SemanticQuerier


def read_ply_raw(ply_path: Path):
    """Read a binary PLY as (header_bytes, props, vertex_count, raw_structured_array)."""
    with open(ply_path, "rb") as f:
        header_bytes = _read_ply_header(f)
        props, count, fmt = _parse_ply_header_text(header_bytes)
        if not fmt.startswith("binary_little_endian"):
            raise NotImplementedError(f"Unsupported PLY format: {fmt}")

        # Build structured dtype matching the on-disk layout
        fields = []
        for prop in props:
            kind, name = prop.split(":", 1)
            if kind == "float":
                fields.append((name, "<f4"))
            elif kind in ("uchar", "uint8"):
                fields.append((name, "u1"))
            elif kind == "double":
                fields.append((name, "<f8"))
            elif kind in ("int", "int32"):
                fields.append((name, "<i4"))
            else:
                raise NotImplementedError(f"Property kind not supported: {kind}")
        dtype = np.dtype(fields)
        raw = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    return header_bytes, props, count, raw


def write_ply_raw(output_path: Path, props, raw: np.ndarray):
    """Write a binary little-endian PLY from a structured array."""
    lines = ["ply", "format binary_little_endian 1.0"]
    lines.append(f"element vertex {len(raw)}")
    for prop in props:
        kind, name = prop.split(":", 1)
        lines.append(f"property {kind} {name}")
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(raw.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Export furniture-only PLY")
    ap.add_argument("--ply", default="data/point_cloud_30000.ply")
    ap.add_argument("--feat", default="output/point_cloud_30000/point_cloud_30000_feat.pt")
    ap.add_argument("--text-emb", default="data/bim_text_emb.pt")
    ap.add_argument("--class-names", default="data/bim_class_names.json")
    ap.add_argument("--output", default="output/furniture_only.ply")
    ap.add_argument(
        "--mode", choices=["furniture", "non-structural"], default="furniture",
        help="'furniture' = 9-class 'furniture' label only; "
             "'non-structural' = everything except wall/floor/ceiling/door/window/column/beam/stairs",
    )
    args = ap.parse_args()

    ply_path = Path(args.ply)
    feat_path = Path(args.feat)
    output_path = Path(args.output)

    print(f"Reading PLY: {ply_path} ...")
    header_bytes, props, count, raw = read_ply_raw(ply_path)
    print(f"  {count:,} vertices, {len(props)} properties")

    print(f"\nLoading feat.pt + warm cache ...")
    q = SemanticQuerier(
        str(feat_path),
        text_emb_path=args.text_emb,
        class_names_path=args.class_names,
    )
    dom = q.get_dominant_labels()
    labels = q.registered_labels
    print(f"  9-class distribution:")
    for i, name in enumerate(labels):
        c = int((dom == i).sum())
        print(f"    {name:<14s} {c:>8,d}  ({c/count*100:.1f}%)")

    # Select mask
    if args.mode == "furniture":
        if "furniture" not in labels:
            raise ValueError("'furniture' not in class names")
        furniture_idx = labels.index("furniture")
        mask = dom == furniture_idx
        label_desc = f"'furniture' (idx={furniture_idx})"
    else:
        structural = {"wall", "floor", "ceiling", "door", "window",
                      "column", "beam", "stairs"}
        struc_idx = {i for i, n in enumerate(labels) if n in structural}
        mask = ~np.isin(dom, list(struc_idx))
        label_desc = "non-structural (all except wall/floor/ceiling/door/window/column/beam/stairs)"

    n_sel = int(mask.sum())
    print(f"\n  Mode: {label_desc}")
    print(f"  Selected: {n_sel:,} / {count:,} ({n_sel/count*100:.1f}%)")

    # Filter and write
    raw_sel = raw[mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting: {output_path} ...")
    write_ply_raw(output_path, props, raw_sel)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Done: {n_sel:,} vertices, {size_mb:.1f} MB")
    print(f"  Open in nerfview / super-splat to inspect.")


if __name__ == "__main__":
    main()
