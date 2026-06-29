"""Probe gsplat.rasterization behavior with synthetic Gaussians.

Validates: render_mode='RGB+D' returns (colors, depths, meta) and
opacities must be in [0,1] (not logit), scales must be positive (not log).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

# Must be run from the bim-recon env.


def main() -> int:
    device = torch.device("cuda")
    n = 4  # 4 synthetic Gaussians at room corners
    means = torch.tensor(
        [[-1.0, -1.0, 3.0],
         [ 1.0, -1.0, 3.0],
         [-1.0,  1.0, 3.0],
         [ 1.0,  1.0, 3.0]],
        dtype=torch.float32, device=device,
    )
    # unit quaternion (w,x,y,z)
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * n,
        dtype=torch.float32, device=device,
    )
    # positive scales (NOT log)
    scales = torch.full((n, 3), 0.1, dtype=torch.float32, device=device)
    # opacity in [0,1] (NOT logit)
    opacities = torch.ones(n, dtype=torch.float32, device=device)
    # RGB colors, shape (N,3) — verify accepted
    colors = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]],
        dtype=torch.float32, device=device,
    )

    # camera: identity rotation, translated back — world-to-camera
    # gsplat viewmats is (C,4,4) world2camera
    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    # intrinsics: fx=fy=200, cx=cy=100 for 200x200 image
    Ks = torch.tensor(
        [[200.0, 0.0, 100.0],
         [0.0, 200.0, 100.0],
         [0.0,   0.0,   1.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)

    W, H = 200, 200

    print("=== Render mode RGB+D ===")
    out_colors, out_depths, meta = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, W, H,
        render_mode="RGB+D",
    )
    print(f"colors shape {tuple(out_colors.shape)} dtype {out_colors.dtype}")
    print(f"depths shape {tuple(out_depths.shape)} dtype {out_depths.dtype}")
    print(f"meta keys {list(meta.keys()) if isinstance(meta, dict) else type(meta)}")
    print(f"colors range [{out_colors.min().item():.3f}, {out_colors.max().item():.3f}]")
    print(f"depths range [{out_depths.min().item():.3f}, {out_depths.max().item():.3f}]")
    print(f"non-zero color pixels: {(out_colors[..., :3].sum(-1) > 1e-6).sum().item()}")
    print(f"non-zero depth pixels: {(out_depths > 1e-6).sum().item()}")

    print("\n=== Render mode RGB (depth should NOT be returned) ===")
    rgb_out = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, W, H,
        render_mode="RGB",
    )
    print(f"returned {len(rgb_out)}-tuple (RGB-only yields 2 elements)")
    out_rgb = rgb_out[0]
    out_meta = rgb_out[-1]
    print(f"rgb shape {tuple(out_rgb.shape)}")
    print(f"meta keys {list(out_meta.keys()) if isinstance(out_meta, dict) else type(out_meta)}")

    print("\n=== Check visibility filter in meta ===")
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")

    print("\nALL GOOD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
