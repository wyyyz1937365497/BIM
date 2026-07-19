#!/usr/bin/env python
"""Remove duplicate wall segments from a completed reconstruction output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.wall_cleanup import clean_saved_wall_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = clean_saved_wall_list(args.output_dir)
    print(
        "Cleaned wall list: "
        f"{summary['input_walls']} -> {summary['output_walls']} "
        f"(removed {summary['removed_walls']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
