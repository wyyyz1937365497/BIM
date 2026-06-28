#!/usr/bin/env python3
"""从手量 JSON 生成 Revit C# 脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bim_recon.floorplan import ManualProvider
from bim_recon.revit_code import RevitGenerationOptions, generate_revit_csharp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="读取 ManualProvider JSON，输出可交给 Revit MCP send_code_to_revit 的 C#。"
    )
    parser.add_argument("input", type=Path, help="手量底图 JSON 文件")
    parser.add_argument("-o", "--output", type=Path, help="输出 .cs 文件；不传则打印到 stdout")
    parser.add_argument("--wall-height", type=float, default=2.8, help="墙高，单位米")
    parser.add_argument("--door-height", type=float, default=2.1, help="门洞高，单位米")
    parser.add_argument("--window-height", type=float, default=1.2, help="窗洞高，单位米")
    parser.add_argument("--ceiling-height", type=float, default=2.8, help="天花板高度，单位米")
    parser.add_argument("--no-floor", action="store_true", help="不生成地板")
    parser.add_argument("--no-ceiling", action="store_true", help="不生成天花板")
    parser.add_argument(
        "--no-hosted-families",
        action="store_true",
        help="只开洞，不尝试放置门窗宿主族",
    )
    args = parser.parse_args()

    floorplan = ManualProvider.from_json(args.input).get_floorplan()
    options = RevitGenerationOptions(
        wall_height=args.wall_height,
        door_height=args.door_height,
        window_height=args.window_height,
        ceiling_height=args.ceiling_height,
        create_floor=not args.no_floor,
        create_ceiling=not args.no_ceiling,
        place_hosted_families=not args.no_hosted_families,
    )
    code = generate_revit_csharp(floorplan, options)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(code, encoding="utf-8")
    else:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
