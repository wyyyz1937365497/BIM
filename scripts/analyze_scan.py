"""Quick stats analysis of virtual scan output."""
import json, numpy as np, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "output" / "virtual_scan_h1.5m.json"))
angles = np.array(data["angles_deg"])
dists = np.array(data["distances"])
pts = np.array(data["points_2d"])

print(f"Points: {len(angles)}")
print(f"Angle range: {angles.min():.1f} - {angles.max():.1f} deg")
print()

# Per-sector median distance
print("Per-sector (30 deg bins):")
for lo in range(0, 360, 30):
    mask = (angles >= lo) & (angles < lo + 30)
    n = int(mask.sum())
    if n > 0:
        d = dists[mask]
        print(f"  Az {lo:3d}-{lo+30:3d}: {n:4d} pts, dist {d.min():.2f}-{d.max():.2f}m, median={np.median(d):.2f}m")

print()
print("Distance histogram (0.5m bins):")
hist, edges = np.histogram(dists, bins=np.arange(0, 12, 0.5))
for i, c in enumerate(hist):
    if c > 0:
        bar = "#" * min(c, 60)
        print(f"  {edges[i]:4.1f}-{edges[i+1]:4.1f}m: {bar} ({c})")

# Check for wall-like features: consistent distances within angular sectors
print()
print("Wall candidate detection (sectors with tight distance clusters):")
for lo in range(0, 360, 10):
    mask = (angles >= lo) & (angles < lo + 10)
    n = int(mask.sum())
    if n < 5:
        continue
    d = dists[mask]
    iqr = np.percentile(d, 75) - np.percentile(d, 25)
    if iqr < 0.3:  # tight cluster = likely wall
        print(f"  Az {lo:3d}-{lo+10:3d}: {n} pts, median={np.median(d):.2f}m, IQR={iqr:.3f}m [WALL]")
