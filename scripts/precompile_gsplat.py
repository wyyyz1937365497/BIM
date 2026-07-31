"""Trigger gsplat CUDA kernel JIT compilation (first run takes 1-3 min).

Matches the gsplat 1.4.0 API used by bim_recon/gs_scene.py.
Run inside a vcvars64 shell (MSVC compiler required).

Usage:
    python scripts/precompile_gsplat.py
"""

import torch
from gsplat import rasterization

print("Compiling gsplat CUDA kernels (first run takes 1-3 min)...")

# 100 dummy Gaussians — enough to trigger all kernel variants
N = 100
means = torch.randn(N, 3, device="cuda")
quats = torch.randn(N, 4, device="cuda")
quats = quats / quats.norm(dim=-1, keepdim=True)  # normalize quaternions
scales = torch.rand(N, 3, device="cuda").exp()
opacities = torch.sigmoid(torch.randn(N, device="cuda"))
colors = torch.sigmoid(torch.randn(N, 3, device="cuda"))

# gsplat 1.4.0: viewmats is world-to-camera (column-major), Ks is intrinsics
viewmat = torch.eye(4, device="cuda").unsqueeze(0)
K = torch.tensor([[600, 0, 400], [0, 600, 300], [0, 0, 1]],
                 device="cuda").float().unsqueeze(0)

rasterization(
    means, quats, scales, opacities, colors,
    viewmats=viewmat, Ks=K, width=800, height=600,
    render_mode="RGB+ED",
)

print("gsplat JIT precompile done.")
