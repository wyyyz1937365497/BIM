"""Snap wall line endpoints and optionally create walls in Revit.

Usage:
    python scripts/snap_and_report.py --name playroom
    python scripts/snap_and_report.py --name playroom --min-length 1.0 --snap 0.5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def snap_endpoints(endpoints: list, threshold: float) -> list:
    """Greedy endpoint snapping: merge endpoints within threshold."""
    changed = True
    iteration = 0
    while changed and iteration < 10:
        changed = False
        iteration += 1
        points = []
        for i, ep in enumerate(endpoints):
            points.append(("s", i, ep[0], ep[1]))
            points.append(("e", i, ep[2], ep[3]))

        snapped = set()
        for i, (t1, idx1, x1, y1) in enumerate(points):
            if i in snapped:
                continue
            group = [(t1, idx1, x1, y1)]
            for j, (t2, idx2, x2, y2) in enumerate(points[i + 1:], i + 1):
                if j in snapped:
                    continue
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist < threshold and dist > 1e-6:
                    group.append((t2, idx2, x2, y2))
                    snapped.add(j)
            if len(group) > 1:
                changed = True
                avg_x = sum(p[2] for p in group) / len(group)
                avg_y = sum(p[3] for p in group) / len(group)
                for t, idx, _, _ in group:
                    if t == "s":
                        endpoints[idx][0] = avg_x
                        endpoints[idx][1] = avg_y
                    else:
                        endpoints[idx][2] = avg_x
                        endpoints[idx][3] = avg_y
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Scene name (output/<name>)")
    parser.add_argument("--min-length", type=float, default=1.0, help="Min wall length to keep (m)")
    parser.add_argument("--snap", type=float, default=0.5, help="Snap threshold (m)")
    args = parser.parse_args()

    out_dir = ROOT / "output" / args.name
    data = json.load(open(out_dir / "wall_lines.json"))

    # Filter by min length
    walls = [w for w in data["walls"] if w["length"] >= args.min_length]
    print(f"Filtered {len(walls)} walls (>={args.min_length}m) from {len(data['walls'])} total")

    endpoints = [[w["x1"], w["y1"], w["x2"], w["y2"], w["length"]] for w in walls]
    endpoints = snap_endpoints(endpoints, args.snap)

    print(f"\nSnapped walls (snap={args.snap}m):")
    for i, ep in enumerate(endpoints):
        x1, y1, x2, y2, l = ep
        print(f"  {i}: ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f}), len={l:.2f}m")

    # Check closures
    print("\nEndpoint connections:")
    for i, ep_i in enumerate(endpoints):
        for j, ep_j in enumerate(endpoints):
            if i >= j:
                continue
            for (x1, y1) in [(ep_i[0], ep_i[1]), (ep_i[2], ep_i[3])]:
                for (x2, y2) in [(ep_j[0], ep_j[1]), (ep_j[2], ep_j[3])]:
                    dist = np.hypot(x1 - x2, y1 - y2)
                    if dist < 0.01:
                        print(f"  Wall {i} -- Wall {j} (gap={dist:.4f}m)")

    # Save snapped JSON
    output = [
        {
            "x1": round(ep[0], 4), "y1": round(ep[1], 4),
            "x2": round(ep[2], 4), "y2": round(ep[3], 4),
            "length": round(ep[4], 4),
        }
        for ep in endpoints
    ]
    out_path = out_dir / "wall_lines_snapped.json"
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
