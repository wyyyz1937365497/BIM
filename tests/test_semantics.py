"""Unit tests for SemanticQuerier — uses synthetic features, no real data needed."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from bim_recon.semantics import SemanticQuerier


# ---------------------------------------------------------------------------
# Fake text encoder (avoids downloading the real SigLIP2 model)
# ---------------------------------------------------------------------------

def _make_fake_text_encoder(bases_np):
    """Build a deterministic fake SigLIP2-style text encoder for tests.

    Given the (3, 768) base vectors the synthetic fixture builds, returns a
    callable ``(list[str]) -> torch.Tensor(L, 768)`` that is L2-normalised:

      - ``'wall'`` / ``'floor'`` / ``'door'`` → the matching base vector, so
        they align with their gaussians exactly as the warm-vocab embeddings
        would (cosine sim ~0.96, sigmoid ~0.72).
      - any other label (e.g. ``'roof'``) → a deterministic random unit
        vector, which has ~0 cosine sim with every gaussian → ~0.5 sigmoid
        prob and never wins an argmax against the real bases.

    The random vectors are seeded from a hash of the label name so test runs
    are reproducible (the per-label cache in ``SemanticQuerier`` means each
    label is encoded at most once per querier anyway).
    """
    base_t = torch.from_numpy(bases_np).float()  # (3, 768)
    known = {"wall": 0, "floor": 1, "door": 2}

    def encode(labels):
        rows = []
        for lab in labels:
            if lab in known:
                rows.append(base_t[known[lab]])
            else:
                seed = int.from_bytes(
                    hashlib.sha256(str(lab).encode()).digest()[:4], "little"
                )
                v = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
                v = v / np.linalg.norm(v)
                rows.append(torch.from_numpy(v))
        emb = torch.stack(rows, dim=0).float()
        return torch.nn.functional.normalize(emb, dim=-1)

    return encode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scene(tmp_path):
    """Create synthetic feat.pt + text_emb.pt + class_names.json.

    Builds 300 Gaussians in 3 groups (100 each). Each group's feature vector
    is strongly aligned (cosine sim > 0.9) with one class's text embedding
    and weakly aligned (< 0.3) with others.

    Also exposes a ``text_encoder`` fake so open-vocabulary queries (for
    labels inside or outside the warm vocabulary) work without the real
    SigLIP2 model, plus the raw ``bases`` the encoder is built from.
    """
    rng = np.random.default_rng(42)
    dim = 768
    num_per_class = 100
    class_names_list = ["wall", "floor", "door"]
    num_classes = len(class_names_list)

    # Create orthogonal-ish base vectors for each class
    bases = rng.standard_normal((num_classes, dim)).astype(np.float32)
    bases = bases / np.linalg.norm(bases, axis=1, keepdims=True)

    # Build features: each group strongly aligned to its class base.
    # noise std=0.01 keeps cosine_sim ~0.96 for right class, ~0.0 for wrong.
    # sigmoid(0.96)≈0.72 vs sigmoid(0.0)=0.5 → threshold 0.6 cleanly separates.
    feats = np.zeros((num_per_class * num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        start = c * num_per_class
        end = start + num_per_class
        noise = rng.standard_normal((num_per_class, dim)).astype(np.float32) * 0.01
        feats[start:end] = bases[c:c+1] + noise

    # Normalize features
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    # Text embeddings = the base vectors (guarantees high cosine sim within class)
    text_emb = torch.from_numpy(bases)  # (3, 768), already normalized

    # Save files
    feat_path = tmp_path / "feat.pt"
    text_emb_path = tmp_path / "text_emb.pt"
    class_names_path = tmp_path / "class_names.json"

    torch.save(torch.from_numpy(feats), feat_path)
    torch.save(text_emb, text_emb_path)
    with open(class_names_path, "w") as f:
        json.dump({name: i for i, name in enumerate(class_names_list)}, f)

    return {
        "feat_path": str(feat_path),
        "text_emb_path": str(text_emb_path),
        "class_names_path": str(class_names_path),
        "num_gaussians": num_per_class * num_classes,
        "num_per_class": num_per_class,
        "class_names": class_names_list,
        "bases": bases,
        "text_encoder": _make_fake_text_encoder(bases),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSemanticQuerierInit:
    def test_load_success(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        assert q.num_gaussians == 300
        assert q.num_classes == 3
        assert q.probs.shape == (300, 3)

    def test_float16_feat(self, synthetic_scene, tmp_path):
        """feat.pt saved as float16 should load and convert correctly."""
        feat16_path = tmp_path / "feat16.pt"
        raw = torch.load(synthetic_scene["feat_path"])
        torch.save(raw.half(), feat16_path)

        q = SemanticQuerier(
            str(feat16_path),
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        # feat is now KEPT RESIDENT (fp32) so any label can be queried at
        # runtime — open-vocabulary backbone. Verify the float16→float32
        # conversion happened and the resident matrix is usable.
        assert q.feat is not None
        assert q.feat.dtype == torch.float32
        assert q.probs.dtype == torch.float32
        assert q.num_gaussians == 300

    def test_dimension_mismatch(self, synthetic_scene, tmp_path):
        """Wrong feature dimension should raise AssertionError."""
        bad_feat_path = tmp_path / "bad_feat.pt"
        torch.save(torch.randn(10, 256), bad_feat_path)
        with pytest.raises(AssertionError, match="768"):
            SemanticQuerier(
                str(bad_feat_path),
                synthetic_scene["text_emb_path"],
                synthetic_scene["class_names_path"],
                device="cpu",
                text_encoder=synthetic_scene["text_encoder"],
            )

    def test_non_tensor_feat(self, synthetic_scene, tmp_path):
        """Non-tensor feat.pt should raise TypeError."""
        bad_path = tmp_path / "dict.pt"
        torch.save({"data": torch.randn(10, 768)}, bad_path)
        with pytest.raises(TypeError, match="Tensor"):
            SemanticQuerier(
                str(bad_path),
                synthetic_scene["text_emb_path"],
                synthetic_scene["class_names_path"],
                device="cpu",
                text_encoder=synthetic_scene["text_encoder"],
            )


class TestQuery:
    def test_query_wall(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("wall", threshold=0.6)
        assert result["class"] == "wall"
        assert result["class_index"] == 0
        # Group 0 (indices 0-99) should be selected
        assert result["num_gaussians"] == 100
        assert all(0 <= idx < 100 for idx in result["indices"])
        assert result["mean_confidence"] > 0.65

    def test_query_floor(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("floor", threshold=0.6)
        assert result["num_gaussians"] == 100
        assert all(100 <= idx < 200 for idx in result["indices"])

    def test_query_unknown_class(self, synthetic_scene):
        """Open-vocabulary: an unregistered label ('roof') is queryable.

        It is encoded on demand via the injected text encoder. The fake
        encoder gives 'roof' a random unit vector (~0 cosine sim with every
        gaussian → ~0.5 sigmoid prob), so at threshold 0.52 it selects 0 or
        only a few gaussians. We assert the dict shape is valid and that the
        label is reported as open-vocab (no fixed class index).
        """
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("roof", threshold=0.52)
        assert result["class"] == "roof"
        assert result["class_index"] == -1  # open-vocab: no fixed index
        assert result["num_gaussians"] >= 0
        assert len(result["indices"]) == result["num_gaussians"]
        assert len(result["confidence"]) == result["num_gaussians"]

    def test_query_high_threshold_empty(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("wall", threshold=0.999)
        assert result["num_gaussians"] == 0
        assert result["mean_confidence"] == 0.0
        assert len(result["indices"]) == 0

    def test_confidence_values(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("door", threshold=0.6)
        assert all(c > 0.5 for c in result["confidence"])
        assert all(c <= 1.0 for c in result["confidence"])


class TestDominantLabels:
    def test_dominant_labels(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        labels = q.get_dominant_labels()
        assert labels.shape == (300,)
        assert labels.dtype == np.int32
        # Group 0 → class 0, Group 1 → class 1, Group 2 → class 2
        npc = synthetic_scene["num_per_class"]
        for c in range(3):
            group_labels = labels[c * npc:(c + 1) * npc]
            # At least 90% should be the correct class
            correct_frac = (group_labels == c).mean()
            assert correct_frac > 0.9, f"Group {c}: only {correct_frac:.0%} correct"


class TestQueryDominant:
    def test_dominant_wall(self, synthetic_scene):
        """query_dominant('wall') should return group 0 Gaussians."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query_dominant("wall")
        assert result["class"] == "wall"
        assert result["class_index"] == 0
        assert result["num_gaussians"] == 100
        assert all(0 <= idx < 100 for idx in result["indices"])
        assert result["mean_confidence"] > 0.6

    def test_dominant_floor(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query_dominant("floor")
        assert result["num_gaussians"] == 100
        assert all(100 <= idx < 200 for idx in result["indices"])

    def test_dominant_unknown_class(self, synthetic_scene):
        """Open-vocabulary: query_dominant('roof') does not raise.

        With the fake encoder 'roof' is unrelated to the 3 bases, so against
        {roof, wall, floor, door} it wins ~0 gaussians (its ~0.5 sigmoid
        prob never beats the matched base's ~0.72). It is still reported as
        class 'roof'.
        """
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query_dominant("roof")
        assert result["class"] == "roof"
        assert result["num_gaussians"] < 10  # ~0: never dominates

    def test_dominant_explicit_label_set(self, synthetic_scene):
        """get_dominant_labels(['wall','floor','door']) → argmax over the set.

        Indices are positions into the label set, so group 0→0, group 1→1,
        group 2→2.
        """
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        dominant = q.get_dominant_labels(["wall", "floor", "door"])
        assert dominant.shape == (300,)
        assert dominant.dtype == np.int32
        assert dominant[5] == 0    # wall group
        assert dominant[105] == 1  # floor group
        assert dominant[205] == 2  # door group

    def test_dominant_indices_disjoint(self, synthetic_scene):
        """Different classes should return disjoint Gaussian sets."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        wall = set(q.query_dominant("wall")["indices"].tolist())
        floor = set(q.query_dominant("floor")["indices"].tolist())
        door = set(q.query_dominant("door")["indices"].tolist())
        assert wall.isdisjoint(floor)
        assert wall.isdisjoint(door)
        assert floor.isdisjoint(door)
        assert len(wall | floor | door) == 300  # all Gaussians covered


class TestQueryTopPercent:
    def test_top10_percent(self, synthetic_scene):
        """Top 10% of 300 = 30 Gaussians."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query_top_percent("wall", percent=10.0)
        assert result["num_gaussians"] == 30
        # All selected should be from group 0 (wall-aligned)
        assert all(0 <= idx < 100 for idx in result["indices"])
        # Confidences should be sorted descending
        confs = result["confidence"]
        assert all(confs[i] >= confs[i + 1] for i in range(len(confs) - 1))

    def test_top100_percent(self, synthetic_scene):
        """Top 100% = all Gaussians."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query_top_percent("wall", percent=100.0)
        assert result["num_gaussians"] == 300

    def test_invalid_percent(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        with pytest.raises(ValueError, match="percent"):
            q.query_top_percent("wall", percent=0.5)
        with pytest.raises(ValueError, match="percent"):
            q.query_top_percent("wall", percent=150.0)


class TestLabelAt:
    def test_label_at_specific_indices(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        # Pick one Gaussian from each group
        test_indices = np.array([5, 105, 205], dtype=np.int64)
        result = q.get_label_at(test_indices)
        assert result["dominant"].shape == (3,)
        assert result["probs"].shape == (3, 3)
        # Each should be dominant in its own class
        assert result["dominant"][0] == 0  # wall
        assert result["dominant"][1] == 1  # floor
        assert result["dominant"][2] == 2  # door


class TestOpenVocabulary:
    """Open-vocabulary semantics: queries without a warm vocabulary."""

    def test_pure_open_vocab_construction(self, synthetic_scene):
        """Construct with ONLY feat_path — no warm vocab at all."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        assert q.num_classes == 0
        assert q.probs is None
        assert q.registered_labels == []

        # get_dominant_labels() with no label_set REQUIRES a warm vocab.
        with pytest.raises(RuntimeError, match="warm vocabulary"):
            q.get_dominant_labels()

        # ...but an explicit label_set works open-vocab.
        dominant = q.get_dominant_labels(["wall", "floor", "door"])
        assert dominant.shape == (300,)
        assert dominant.dtype == np.int32
        assert dominant[5] == 0    # wall group
        assert dominant[105] == 1  # floor group
        assert dominant[205] == 2  # door group

        # query_top_percent encodes 'wall' on demand → top 10% = 30 gaussians.
        result = q.query_top_percent("wall", 10.0)
        assert result["num_gaussians"] == 30

    def test_query_open_vocab_label(self, synthetic_scene):
        """With a warm vocab present, an unregistered label is still queryable.

        query('roof', threshold=0.0) encodes 'roof' on demand and reports it
        as open-vocabulary (class_index == -1).
        """
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        result = q.query("roof", threshold=0.0)
        assert result["class"] == "roof"
        assert result["class_index"] == -1
        # threshold 0.0 keeps every gaussian (sigmoid > 0 everywhere)
        assert result["num_gaussians"] == 300

    def test_encode_labels_caches(self, synthetic_scene):
        """encode_labels caches per label → identical output across calls.

        'wall' resolves to the registered base (high sim to group 0); 'roof'
        is a random unit vector unrelated to any group.
        """
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
            text_encoder=synthetic_scene["text_encoder"],
        )
        out1 = q.encode_labels(["wall", "roof"])
        out2 = q.encode_labels(["wall", "roof"])
        assert out1.shape == (2, 768)
        assert torch.equal(out1, out2)

        # 'wall' uses the registered base → near-perfect sim with group 0.
        wall_emb = out1[0]
        group0_mean = q.feat[:100].mean(dim=0)
        wall_sim = torch.nn.functional.cosine_similarity(
            wall_emb.unsqueeze(0), group0_mean.unsqueeze(0)
        ).item()
        assert wall_sim > 0.9

        # 'roof' is a random unit vector → low |sim| with every gaussian.
        roof_emb = out1[1]
        sims = torch.nn.functional.cosine_similarity(
            q.feat, roof_emb.unsqueeze(0)
        )
        assert sims.abs().max().item() < 0.3
