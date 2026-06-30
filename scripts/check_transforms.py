"""Quick sanity check on the converted 7-Scenes transforms.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "data/office_ns/transforms.json", "r") as f:
    t = json.load(f)

print("=== top-level keys ===")
for k, v in t.items():
    if k == "frames":
        print(f"  {k}: [{len(v)} frames]")
    else:
        print(f"  {k}: {v}")

print("\n=== first frame ===")
fr = t["frames"][0]
fp = fr["file_path"]
dp = fr["depth_file_path"]
m = np.array(fr["transform_matrix"])
z_axis = m[:3, 2]
y_axis = m[:3, 1]
print(f"  file_path:       {fp}")
print(f"  depth_file_path: {dp}")
print(f"  translation:     [{m[0,3]:.4f}, {m[1,3]:.4f}, {m[2,3]:.4f}]")
print(f"  z_axis (forward, should be ~horizontal): [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
print(f"  y_axis (should be ~down, y[1]<0):       [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")

# Check that y-axis points downward (negative y in world = OpenCV convention)
print(f"\n  y_axis[1] sign: {'DOWN (OpenCV OK)' if y_axis[1] < 0 else 'UP (WRONG!)'}")

# Verify file paths resolve
base = ROOT / "data/office_ns"
depth = base / fr["depth_file_path"]
color = base / fr["file_path"]
print(f"\n  color file exists: {color.exists()}")
print(f"  depth file exists: {depth.exists()}")

# Sample a few more poses to check variety
print("\n=== pose translation stats (all frames) ===")
trans = np.array([np.array(f["transform_matrix"])[:3, 3] for f in t["frames"]])
print(f"  x range: [{trans[:,0].min():.3f}, {trans[:,0].max():.3f}]")
print(f"  y range: [{trans[:,1].min():.3f}, {trans[:,1].max():.3f}]")
print(f"  z range: [{trans[:,2].min():.3f}, {trans[:,2].max():.3f}]")
print(f"  scene extent: {np.max(trans.max(0) - trans.min(0)):.3f} m")
print("\nSanity check done.")
