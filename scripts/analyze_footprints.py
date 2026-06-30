"""Quick analysis of wall/floor footprint extents for registration debugging."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.gs_scene import GSScene


def main() -> int:
    scene = GSScene.from_npy(
        ROOT / "data",
        feat_path=ROOT / "output" / "data_feat.pt",
        text_emb_path=ROOT / "data" / "bim_text_emb.pt",
        class_names_path=ROOT / "data" / "bim_class_names.json",
    )
    for label in ["wall", "floor", "ceiling"]:
        result = scene.query_semantics(label, mode="dominant")
        indices = result["indices"]
        if len(indices) == 0:
            print(f"{label}: no Gaussians")
            continue
        means = scene.means[torch.as_tensor(indices, dtype=torch.long)].cpu().numpy()
        xy = means[:, :2]
        print(f"{label}: count={len(means)}")
        print(f"  min: {xy.min(axis=0)}")
        print(f"  max: {xy.max(axis=0)}")
        print(f"  mean: {xy.mean(axis=0)}")
        print(f"  std: {xy.std(axis=0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
