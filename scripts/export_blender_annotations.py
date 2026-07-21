#!/usr/bin/env python
"""Export B-class pose annotations from a .blend file without opening the UI.

Usage:
    blender --background annotation.blend \
      --python scripts/export_blender_annotations.py -- \
      --output output/annotations/room_0.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON_ROOT = ROOT / "blender_addons"
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

import bpy

from bim_pose_annotation.annotation_io import AnnotationError, export_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write a diagnostic manifest even when blocking validation errors exist",
    )
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def main() -> int:
    args = parse_args()
    try:
        path = export_scene_manifest(
            bpy.context.scene,
            args.output,
            strict=not args.allow_invalid,
        )
    except AnnotationError as exc:
        print(f"BIM_ANNOTATION_ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"BIM_ANNOTATION_MANIFEST: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
