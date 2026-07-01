"""Analyze wall line extraction quality."""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Check JSON output
data = json.load(open(ROOT / "output" / "wall_lines.json"))
print(f"Walls: {data['num_walls']}")
print(f"Scan heights: {data['scan_info']['heights']}")
print()

for i, w in enumerate(data["walls"]):
    print(f"  Wall {i}: ({w['x1']:.2f},{w['y1']:.2f}) -> ({w['x2']:.2f},{w['y2']:.2f}), len={w['length']:.2f}m")

# Check PNG size
png = ROOT / "output" / "wall_lines_topdown.png"
print(f"\nPNG: {png.stat().st_size} bytes")

# Analyze wall point distribution
print("\n--- Wall point analysis ---")
# Load the scan data to check wall points
import sys
sys.path.insert(0, str(ROOT))
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.gs_scene import GSScene
from bim_recon.wall_line_extractor import extract_wall_points, multi_height_scan
import torch

scene = GSScene.from_npy(
    ROOT / "data",
    feat_path=str(ROOT / "output" / "data_feat.pt"),
    text_emb_path=str(ROOT / "data" / "bim_text_emb.pt"),
    class_names_path=str(ROOT / "data" / "bim_class_names.json"),
)
floor = scene.query_semantics("floor", mode="dominant")
up_axis = int(np.argmin(floor["centroid"]))
h_axes = [i for i in range(3) if i != up_axis]
cx, cy = floor["centroid"][h_axes[0]], floor["centroid"][h_axes[1]]
fz, cz = floor["centroid"][up_axis], scene.query_semantics("ceiling", mode="dominant")["centroid"][up_axis]

scanner = VirtualScanner(scene, up_axis=up_axis)
scans = multi_height_scan(scanner, (cx, cy), fz, cz, num_heights=8, num_views=8, width=512)
pts, heights = extract_wall_points(scans, 0)
print(f"Wall points: {len(pts)}")
print(f"X range: {pts[:,0].min():.2f} - {pts[:,0].max():.2f}")
print(f"Y range: {pts[:,1].min():.2f} - {pts[:,1].max():.2f}")

# Per-angle analysis
center = np.array([cx, cy])
dx = pts[:,0] - center[0]
dy = pts[:,1] - center[1]
angles = np.degrees(np.arctan2(dy, dx)) % 360
dists = np.sqrt(dx**2 + dy**2)

print("\nPer 30° sector:")
for lo in range(0, 360, 30):
    mask = (angles >= lo) & (angles < lo + 30)
    n = mask.sum()
    if n > 0:
        d = dists[mask]
        print(f"  Az {lo:3d}-{lo+30:3d}: {n:4d} pts, dist {d.min():.2f}-{d.max():.2f}m, med={np.median(d):.2f}m")
