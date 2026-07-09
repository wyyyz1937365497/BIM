"""Open-vocabulary semantic querier for per-Gaussian language features.

Loads ``feat.pt`` (N, 768) — per-Gaussian SigLIP2 features produced by
SceneSplat's ``lang_inference.py`` — and answers text→Gaussian queries
against an **arbitrary** label set. Labels are encoded on demand with a
SigLIP2 text encoder (``google/siglip2-base-patch16-512``), so the querier
is no longer locked to a fixed vocabulary.

A pre-encoded warm-cache vocabulary (``bim_text_emb.pt`` +
``bim_class_names.json`` from ``encode_bim_labels.py``) may optionally be
registered at construction for fast backward-compatible queries over the
classic 9-class BIM set. It is a performance cache only — every method
works without it.

Design:
  - ``feat`` is kept resident (fp32) so any label can be queried at runtime.
    This costs ~4 GB on 1.3 M Gaussians — the price of open vocabulary.
  - :meth:`encode_labels` L2-normalises SigLIP2 text embeddings and caches
    them per label.
  - :meth:`query` / :meth:`query_dominant` / :meth:`query_top_percent` work
    for any label; registered-vocab labels use a cached probability matrix,
    all others are encoded on demand.
  - :meth:`get_dominant_labels` returns the argmax class per Gaussian over
    an arbitrary label set (used by the virtual scanner and height detector,
    which encode the dominant label as a colour ramp).

This module has ZERO dependency on the scene_splat conda environment — it
only needs ``torch``, ``numpy`` and (lazily) ``transformers``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

#: Default SigLIP2 text encoder (open-vocabulary, zero-shot classification).
DEFAULT_SIGLIP2 = "google/siglip2-base-patch16-512"

#: A callable mapping a list of text labels to an (L, 768) L2-normalised
#: float32 tensor. Used for dependency injection in tests.
TextEncoder = Callable[[Sequence[str]], torch.Tensor]


def default_text_encoder_factory(model_name: str) -> TextEncoder:
    """Build a SigLIP2 text encoder callable (lazy import of transformers).

    Mirrors ``encode_bim_labels.py``: prompt ``"this is a {label}"``,
    ``tokenizer(max_length=64)``, ``model.get_text_features``, take the
    ``pooler_output``, L2-normalise.
    """
    from transformers import AutoModel, AutoTokenizer  # lazy import

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    @torch.no_grad()
    def encode(labels: Sequence[str]) -> torch.Tensor:
        prompts = [f"this is a {label}" for label in labels]
        batch = tokenizer(
            prompts, padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        )
        out = model(**batch)
        emb = out.pooler_output  # (L, 768) — transformers 5.x
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().cpu().float()

    return encode


class SemanticQuerier:
    """Open-vocabulary querier over per-Gaussian SigLIP2 features.

    Parameters
    ----------
    feat_path
        Path to SceneSplat ``feat.pt`` — an (N, 768) tensor of L2-normalised
        per-Gaussian language features.
    text_emb_path, class_names_path
        Optional warm-cache vocabulary. When both are given, the classic
        BIM label set is pre-encoded and cached for fast queries; every
        method also works without them (open vocabulary).
    device
        Torch device for the resident feature matrix.
    siglip_model
        HuggingFace id of the SigLIP2 text encoder used to encode labels
        on demand.
    text_encoder
        Optional injected encoder callable (overrides ``siglip_model``).
        Useful for unit tests that cannot download the real model.
    """

    def __init__(
        self,
        feat_path: Union[str, Path],
        text_emb_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        siglip_model: str = DEFAULT_SIGLIP2,
        text_encoder: Optional[TextEncoder] = None,
    ):
        # --- resident feature matrix (open-vocabulary backbone) -------------
        raw = torch.load(feat_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(
                f"feat.pt must contain a torch.Tensor, got {type(raw).__name__}"
            )
        self.feat: torch.Tensor = raw.float()  # (N, 768), kept resident
        del raw
        self.num_gaussians: int = self.feat.shape[0]
        feat_dim = self.feat.shape[1]
        assert feat_dim == 768, f"Expected 768-dim features, got {feat_dim}"

        self._device = torch.device(device)
        self.feat = self.feat.to(self._device)

        # --- optional warm-cache vocabulary ---------------------------------
        self.class_names: Dict[str, int] = {}
        self.registered_labels: List[str] = []
        self.probs: Optional[torch.Tensor] = None  # (N, C) for registered vocab
        self._dominant: Optional[torch.Tensor] = None  # (N,) argmax over registered
        self._registered_emb: Optional[torch.Tensor] = None  # (C, 768) warm vocab

        if text_emb_path and class_names_path:
            self._register_warm_vocab(text_emb_path, class_names_path, feat_dim)

        # --- lazy text encoder ----------------------------------------------
        self._siglip_model_name = siglip_model
        self._encoder: Optional[TextEncoder] = text_encoder
        self._emb_cache: Dict[str, torch.Tensor] = {}
        # Seed the cache from the warm vocabulary so registered labels never
        # need re-encoding.
        if self.probs is not None and self._registered_emb is not None:
            for name, idx in self.class_names.items():
                self._emb_cache[name] = self._registered_emb[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_warm_vocab(
        self,
        text_emb_path: Union[str, Path],
        class_names_path: Union[str, Path],
        feat_dim: int,
    ) -> None:
        text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=False)
        if not isinstance(text_emb, torch.Tensor):
            raise TypeError(
                f"text_emb must contain a torch.Tensor, got {type(text_emb).__name__}"
            )
        text_emb = text_emb.float()
        assert feat_dim == text_emb.shape[1], (
            f"Feature dimension mismatch: feat={feat_dim}, text_emb={text_emb.shape[1]}"
        )
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)
        assert len(self.class_names) == text_emb.shape[0], (
            f"class_names has {len(self.class_names)} entries but text_emb has "
            f"{text_emb.shape[0]} rows"
        )
        # Order labels by their stored index for stable argmax mapping.
        self.registered_labels = [
            name for name, _ in sorted(self.class_names.items(), key=lambda kv: kv[1])
        ]
        self._registered_emb = text_emb.to(self._device)  # (C, 768), L2-normed
        logits = self.feat @ self._registered_emb.T  # (N, C)
        self.probs = torch.sigmoid(logits)
        self._dominant = self.probs.argmax(dim=1)
        del logits

    def _get_encoder(self) -> TextEncoder:
        if self._encoder is None:
            self._encoder = default_text_encoder_factory(self._siglip_model_name)
        return self._encoder

    def encode_labels(self, labels: Sequence[str]) -> torch.Tensor:
        """Return an (L, 768) L2-normalised embedding matrix for *labels*.

        Uses the cached warm-vocabulary rows when available and encodes the
        rest on demand via the SigLIP2 text encoder. Cached per label.
        """
        missing = [lab for lab in labels if lab not in self._emb_cache]
        if missing:
            new = self._get_encoder()(missing)  # (M, 768) cpu float
            for lab, row in zip(missing, new):
                self._emb_cache[lab] = row
        rows = [self._emb_cache[lab] for lab in labels]
        emb = torch.stack(rows, dim=0).to(self._device).float()  # (L, 768)
        return torch.nn.functional.normalize(emb, dim=-1)

    def _probs_for_labels(self, labels: Sequence[str]) -> torch.Tensor:
        """Sigmoid probability matrix (N, L) for an arbitrary label set."""
        emb = self.encode_labels(labels)  # (L, 768) on device
        logits = self.feat @ emb.T  # (N, L)
        return torch.sigmoid(logits)

    def _resolve_dominant(
        self, text: str, label_set: Optional[Sequence[str]]
    ) -> "tuple[torch.Tensor, List[str], int]":
        """Resolve the argmax tensor, ordered label list, and text's index.

        Fast path: ``text`` is registered and no ``label_set`` given → reuse
        the cached ``_dominant`` over the warm vocabulary.
        """
        if label_set is None and text in self.class_names:
            assert self._dominant is not None
            return self._dominant, self.registered_labels, self.class_names[text]
        # Open-vocabulary / explicit label set.
        if label_set is not None:
            labels = list(label_set)
            if text not in labels:
                raise ValueError(
                    f"Unknown label '{text}'. Not in label_set {labels}."
                )
            text_idx = labels.index(text)
        else:
            # Open-vocab single query: let it compete against the structural
            # defaults so argmax is meaningful.
            labels = [text] + [
                n for n in self.registered_labels if n != text
            ]
            text_idx = 0
        probs = self._probs_for_labels(labels)  # (N, L)
        dominant = probs.argmax(dim=1)
        return dominant, labels, text_idx

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _class_index(self, text: str) -> int:
        """Index of *text* in the registered vocabulary (closed-set helper)."""
        if text not in self.class_names:
            available = ", ".join(self.class_names.keys())
            raise ValueError(f"Unknown class '{text}'. Available: {available}")
        return self.class_names[text]

    def query(
        self,
        text: str,
        threshold: float = 0.52,
        label_set: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Return Gaussians whose probability for *text* exceeds *threshold*.

        Open-vocabulary: *text* need not belong to the registered vocabulary;
        it is encoded on demand via SigLIP2. Registered labels use a cached
        probability column.

        .. warning::
            Absolute sigmoid thresholds are fragile with SceneSplat features
            because cosine similarities cluster tightly (~0.1 ± 0.015),
            mapping most probabilities to ~0.52. Prefer :meth:`query_dominant`
            (argmax-based) or :meth:`query_top_percent` (percentile-based).

        ``label_set`` is accepted for API symmetry but ignored (threshold
        queries are single-label).
        """
        if text in self.class_names and self.probs is not None:
            class_probs = self.probs[:, self.class_names[text]]
            idx = self.class_names[text]
        else:
            emb = self.encode_labels([text])  # (1, 768)
            class_probs = torch.sigmoid((self.feat @ emb.T).squeeze(-1))  # (N,)
            idx = -1  # open-vocabulary label has no fixed index
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

    def query_dominant(
        self,
        text: str,
        label_set: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Return Gaussians whose **dominant** (argmax) class is *text*.

        This is the most reliable selection method for SceneSplat features.

        ``label_set`` selects the comparison vocabulary for the argmax:

        - ``None`` and *text* registered → argmax over the warm vocabulary.
        - ``None`` and *text* open-vocab → argmax over ``{text} ∪ registered``.
        - given → argmax over exactly ``label_set`` (*text* must be in it).
        """
        dominant, labels, idx = self._resolve_dominant(text, label_set)
        mask = dominant == idx  # (N,) bool
        indices = torch.where(mask)[0]
        if label_set is None and text in self.class_names and self.probs is not None:
            confidence = self.probs[indices, idx]
        else:
            probs = self._probs_for_labels(labels)
            confidence = probs[indices, idx]
        return {
            "class": text,
            "class_index": idx,
            "num_gaussians": int(indices.shape[0]),
            "indices": indices.cpu().numpy().astype(np.int64),
            "confidence": confidence.cpu().numpy().astype(np.float32),
            "mean_confidence": float(confidence.mean()) if indices.shape[0] > 0 else 0.0,
        }

    def query_top_percent(
        self,
        text: str,
        percent: float = 10.0,
        label_set: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Return the top-*percent* % Gaussians for *text* by probability.

        ``percent`` is a fraction in [1, 100]. Open-vocabulary: *text* is
        encoded on demand when not in the registered vocabulary.
        """
        if not (1.0 <= percent <= 100.0):
            raise ValueError(f"percent must be in [1, 100], got {percent}")
        if text in self.class_names and self.probs is not None:
            class_probs = self.probs[:, self.class_names[text]]
            idx = self.class_names[text]
        else:
            emb = self.encode_labels([text])
            class_probs = torch.sigmoid((self.feat @ emb.T).squeeze(-1))
            idx = -1
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

    def get_dominant_labels(
        self, label_set: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Return the argmax class per Gaussian — (N,) int32 array.

        With ``label_set=None`` → argmax over the registered vocabulary
        (cached; requires a warm vocabulary). With an explicit ``label_set``
        → argmax over that arbitrary label set (encoded on demand); indices
        are positions into ``label_set``.
        """
        if label_set is None:
            if self._dominant is None:
                raise RuntimeError(
                    "get_dominant_labels() without a label_set requires a "
                    "registered warm vocabulary; pass an explicit label_set "
                    "for open-vocabulary queries."
                )
            return self._dominant.cpu().numpy().astype(np.int32)
        probs = self._probs_for_labels(label_set)  # (N, L)
        return probs.argmax(dim=1).cpu().numpy().astype(np.int32)

    def get_label_at(
        self,
        gaussian_indices: np.ndarray,
        label_set: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Return label distribution for specific Gaussians.

        Returns:
          - ``dominant``: (K,) int32 — argmax class per queried Gaussian
          - ``probs``: (K, C) float32 — full probability distribution
        """
        idx_tensor = torch.as_tensor(
            gaussian_indices, dtype=torch.long, device=self._device
        )
        if label_set is not None:
            probs = self._probs_for_labels(label_set)
            sub_probs = probs[idx_tensor]
        elif self.probs is not None:
            sub_probs = self.probs[idx_tensor]
        else:
            raise RuntimeError(
                "get_label_at() without a label_set requires a registered "
                "warm vocabulary; pass an explicit label_set for open-vocab."
            )
        return {
            "dominant": sub_probs.argmax(dim=1).cpu().numpy().astype(np.int32),
            "probs": sub_probs.cpu().numpy().astype(np.float32),
        }

    @property
    def num_classes(self) -> int:
        """Number of registered (warm-cache) classes — 0 if pure open-vocab.

        Note: this reflects only the warm vocabulary, NOT an arbitrary
        ``label_set``. Callers using :meth:`get_dominant_labels` with a
        custom label set should use ``len(label_set)`` instead.
        """
        return len(self.registered_labels)

