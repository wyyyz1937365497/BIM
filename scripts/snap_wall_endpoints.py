"""Snap wall line endpoints for Revit wall creation."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
data = json.load(open(root / "output" / "scene0002" / "wall_lines.json"))

# Filter walls >= 1.0m
walls = [w for w in data["walls"] if w["length"] >= 1.0]
print(f"Filtered {len(walls)} walls (>= 1.0m) from {len(data['walls'])} total")

# Collect all endpoints as mutable objects
endpoints = []
for w in walls:
    endpoints.append([w["x1"], w["y1"], w["x2"], w["y2"], w["length"]])

# Snap endpoints: if any endpoint is within threshold of another, merge them
snap_threshold = 0.5
changed = True
iteration = 0
while changed and iteration < 10:
    changed = False
    iteration += 1
    # Collect all current endpoints
    points = []
    for i, ep in enumerate(endpoints):
        points.append(("s", i, ep[0], ep[1]))
        points.append(("e", i, ep[2], ep[3]))

    snapped = set()
    for i, (t1, idx1, x1, y1) in enumerate(points):
        if i in snapped:
            continue
        group = [(t1, idx1, x1, y1)]
        for j, (t2, idx2, x2, y2) in enumerate(points[i + 1 :], i + 1):
            if j in snapped:
                continue
            dist = np.hypot(x1 - x2, y1 - y2)
            if dist < snap_threshold and dist > 1e-6:
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

print(f"\nSnap completed in {iteration} iterations")
print("\nSnapped walls:")
for i, ep in enumerate(endpoints):
    x1, y1, x2, y2, l = ep
    print(f"  {i}: ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f}), len={l:.2f}m")

# Verify closures
print("\nEndpoint gaps after snap:")
for i, ep_i in enumerate(endpoints):
    for j, ep_j in enumerate(endpoints):
        if i >= j:
            continue
        # Check if any endpoint of i matches any endpoint of j
        for (x1, y1) in [(ep_i[0], ep_i[1]), (ep_i[2], ep_i[3])]:
            for (x2, y2) in [(ep_j[0], ep_j[1]), (ep_j[2], ep_j[3])]:
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist < 0.01:
                    print(f"  Wall {i} connects to Wall {j} (gap={dist:.4f}m)")

# Save
output = []
for ep in endpoints:
    output.append(
        {
            "x1": round(ep[0], 4),
            "y1": round(ep[1], 4),
            "x2": round(ep[2], 4),
            "y2": round(ep[3], 4),
            "length": round(ep[4], 4),
        }
    )

out_path = root / "output" / "scene0002" / "wall_lines_snapped.json"
json.dump(output, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
