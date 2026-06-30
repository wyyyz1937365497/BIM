"""Unit tests for SemanticQuerier — uses synthetic features, no real data needed."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from bim_recon.semantics import SemanticQuerier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scene(tmp_path):
    """Create synthetic feat.pt + text_emb.pt + class_names.json.

    Builds 300 Gaussians in 3 groups (100 each). Each group's feature vector
    is strongly aligned (cosine sim > 0.9) with one class's text embedding
    and weakly aligned (< 0.3) with others.
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
        )
        assert q.feat.dtype == torch.float32
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
            )


class TestQuery:
    def test_query_wall(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
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
        )
        result = q.query("floor", threshold=0.6)
        assert result["num_gaussians"] == 100
        assert all(100 <= idx < 200 for idx in result["indices"])

    def test_query_unknown_class(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
        )
        with pytest.raises(ValueError, match="Unknown class"):
            q.query("roof")

    def test_query_high_threshold_empty(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
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
        )
        result = q.query_dominant("floor")
        assert result["num_gaussians"] == 100
        assert all(100 <= idx < 200 for idx in result["indices"])

    def test_dominant_unknown_class(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
        )
        with pytest.raises(ValueError, match="Unknown class"):
            q.query_dominant("roof")

    def test_dominant_indices_disjoint(self, synthetic_scene):
        """Different classes should return disjoint Gaussian sets."""
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
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
        )
        result = q.query_top_percent("wall", percent=100.0)
        assert result["num_gaussians"] == 300

    def test_invalid_percent(self, synthetic_scene):
        q = SemanticQuerier(
            synthetic_scene["feat_path"],
            synthetic_scene["text_emb_path"],
            synthetic_scene["class_names_path"],
            device="cpu",
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
