"""Semantic querier for per-Gaussian language features from SceneSplat.

Loads ``feat.pt`` (N, 768) produced by SceneSplat's ``lang_inference.py``
and ``bim_text_emb.pt`` (C, 768) produced by ``encode_bim_labels.py``,
then provides text→Gaussian semantic queries via cosine similarity.

This module has ZERO dependency on the scene_splat conda environment —
it only needs ``torch`` and ``numpy`` which are already in ``bim-recon``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class SemanticQuerier:
    """Loads SceneSplat feat.pt + SigLIP2 text_emb.pt, provides text→Gaussian queries.

    The probability of Gaussian *i* belonging to class *c* is computed as
    ``sigmoid(feat[i] @ text_emb[c]^T)``, following SceneSplat's
    ``ZeroShotSemSegTester`` classification logic.
    """

    def __init__(
        self,
        feat_path: str | Path,
        text_emb_path: str | Path,
        class_names_path: str | Path,
        device: str | torch.device = "cuda",
    ):
        # feat.pt may be float16 (SceneSplat saves as .half()) —
        # convert to float32 for matmul precision.
        raw = torch.load(feat_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(
                f"feat.pt must contain a torch.Tensor, got {type(raw).__name__}"
            )
        self.feat: torch.Tensor = raw.float().to(device)  # (N, 768)

        text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=False)
        if not isinstance(text_emb, torch.Tensor):
            raise TypeError(
                f"text_emb must contain a torch.Tensor, got {type(text_emb).__name__}"
            )
        self.text_emb: torch.Tensor = text_emb.float().to(device)  # (C, 768)

        with open(class_names_path, "r") as f:
            self.class_names: Dict[str, int] = json.load(f)

        self.num_gaussians: int = self.feat.shape[0]
        self.num_classes: int = self.text_emb.shape[0]
        feat_dim = self.feat.shape[1]
        assert feat_dim == self.text_emb.shape[1], (
            f"Feature dimension mismatch: feat={feat_dim}, text_emb={self.text_emb.shape[1]}"
        )
        assert feat_dim == 768, f"Expected 768-dim features, got {feat_dim}"
        assert len(self.class_names) == self.num_classes, (
            f"class_names has {len(self.class_names)} entries but text_emb has {self.num_classes} rows"
        )

        # Pre-compute probability matrix (N, C) — done once, cheap relative to rendering.
        self.logits: torch.Tensor = self.feat @ self.text_emb.T  # (N, C)
        self.probs: torch.Tensor = torch.sigmoid(self.logits)  # (N, C)
        # Dominant (argmax) label per Gaussian — the most reliable classification
        # because logits cluster tightly (cosine sims ~0.1 with small std), making
        # absolute thresholds unreliable but argmax discriminative.
        self._dominant: torch.Tensor = self.probs.argmax(dim=1)  # (N,)

        self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _class_index(self, text: str) -> int:
        if text not in self.class_names:
            available = ", ".join(self.class_names.keys())
            raise ValueError(f"Unknown class '{text}'. Available: {available}")
        return self.class_names[text]

    def query(self, text: str, threshold: float = 0.52) -> Dict:
        """Return Gaussians whose probability for *text* exceeds *threshold*.

        .. warning::
            Absolute sigmoid thresholds are fragile with SceneSplat features
            because cosine similarities cluster tightly (~0.1 ± 0.015),
            mapping most probabilities to ~0.52.  For robust per-class
            selection prefer :meth:`query_dominant` (argmax-based) or
            :meth:`query_top_percent` (percentile-based).  Use this method
            only when you have a calibrated threshold for your data.

        Returns a dict with keys:
          - ``class``: the queried class name
          - ``class_index``: integer index
          - ``num_gaussians``: count of matching Gaussians
          - ``indices``: (K,) int64 numpy array of Gaussian indices
          - ``confidence``: (K,) float32 numpy array of probabilities
          - ``mean_confidence``: float
        """
        idx = self._class_index(text)
        class_probs = self.probs[:, idx]  # (N,)
        mask = class_probs > threshold
        indices = torch.where(mask)[0]
        confidence = class_probs[mask]
        return {
            "class": text,
            "class_index": idx,
            "num_gaussians": int(indices.shape[0]),
            "indices": indices.cpu().numpy().astype(np.int64),
            "confidence": confidence.cpu().numpy().astype(np.float32),
            "mean_confidence": float(confidence.mean()) if indices.shape[0] > 0 else 0.0,
        }

    def query_dominant(self, text: str) -> Dict:
        """Return Gaussians whose **dominant** (argmax) class is *text*.

        This is the most reliable selection method for SceneSplat features.
        Cosine similarities cluster tightly so absolute thresholds are
        fragile, but the argmax across classes is discriminative.

        Returns the same dict shape as :meth:`query`, with
        ``mean_confidence`` reflecting how strongly the dominant class
        wins on average.
        """
        idx = self._class_index(text)
        mask = self._dominant == idx  # (N,) bool
        indices = torch.where(mask)[0]
        confidence = self.probs[indices, idx]
        return {
            "class": text,
            "class_index": idx,
            "num_gaussians": int(indices.shape[0]),
            "indices": indices.cpu().numpy().astype(np.int64),
            "confidence": confidence.cpu().numpy().astype(np.float32),
            "mean_confidence": float(confidence.mean()) if indices.shape[0] > 0 else 0.0,
        }

    def query_top_percent(self, text: str, percent: float = 10.0) -> Dict:
        """Return the top-*percent* % Gaussians for class *text* by probability.

        Selects the highest-probability Gaussians for the given class,
        regardless of absolute confidence.  Useful when the logit
        distribution is compressed and absolute thresholds don't
        discriminate.

        ``percent`` is a fraction in [1, 100]; e.g. ``10.0`` returns
        the top 10 %.
        """
        if not (1.0 <= percent <= 100.0):
            raise ValueError(f"percent must be in [1, 100], got {percent}")
        idx = self._class_index(text)
        class_probs = self.probs[:, idx]  # (N,)
        k = max(1, int(self.num_gaussians * percent / 100.0))
        topk_vals, topk_indices = torch.topk(class_probs, k)
        return {
            "class": text,
            "class_index": idx,
            "num_gaussians": k,
            "indices": topk_indices.cpu().numpy().astype(np.int64),
            "confidence": topk_vals.cpu().numpy().astype(np.float32),
            "mean_confidence": float(topk_vals.mean()),
        }

    def get_dominant_labels(self) -> np.ndarray:
        """Return the argmax class per Gaussian — (N,) int32 array."""
        return self._dominant.cpu().numpy().astype(np.int32)

    def get_label_at(self, gaussian_indices: np.ndarray) -> Dict:
        """Return label distribution for specific Gaussians.

        Returns:
          - ``dominant``: (K,) int32 — argmax class per queried Gaussian
          - ``probs``: (K, C) float32 — full probability distribution
        """
        idx_tensor = torch.as_tensor(gaussian_indices, dtype=torch.long, device=self._device)
        sub_probs = self.probs[idx_tensor]  # (K, C)
        return {
            "dominant": sub_probs.argmax(dim=1).cpu().numpy().astype(np.int32),
            "probs": sub_probs.cpu().numpy().astype(np.float32),
        }
