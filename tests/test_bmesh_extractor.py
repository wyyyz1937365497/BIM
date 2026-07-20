"""Contracts for the B-class bbox → VLM → Falcon → cutout pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image

from bim_recon.bmesh_extractor import (
    ExtractionResult,
    _build_cutout,
    _extract_user_bbox,
    _parse_vlm_label,
    _rle_to_alpha,
    _select_detection,
    classify_and_segment,
)


def _editor_dict(paint_bbox: tuple[int, int, int, int], img_size: tuple[int, int] = (200, 150)):
    """Build a Gradio ImageMask dict with a solid paint rectangle."""
    w, h = img_size
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 1] = 128  # green-ish base image
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    x0, y0, x1, y1 = paint_bbox
    layer[y0:y1, x0:x1, 3] = 255  # opaque paint
    return {"background": background, "layers": [layer]}


def _fake_detection(cx: float, cy: float, w: float = 0.2, h: float = 0.2, area: float = 0.1):
    class _Det:
        pass
    det = _Det()
    det.bbox = {"x": cx, "y": cy, "w": w, "h": h}
    det.mask_bbox = {"x": cx, "y": cy, "w": w, "h": h}
    det.mask_area_ratio = area
    det.mask_rle = None
    det.mask_size = None
    return det


def test_extract_user_bbox_returns_tight_rectangle_of_paint():
    editor = _editor_dict((20, 30, 80, 70))
    result = _extract_user_bbox(editor)
    assert result is not None
    _base, bbox = result
    # numpy paint slice [30:70, 20:80] covers inclusive indices 20..79 / 30..69
    assert bbox == (20, 30, 79, 69)


def test_extract_user_bbox_returns_none_without_paint():
    assert _extract_user_bbox({"background": np.zeros((10, 10, 3), np.uint8), "layers": []}) is None
    assert _extract_user_bbox(None) is None


def test_parse_vlm_label_preserves_referring_expression():
    assert _parse_vlm_label("the blue office chair on the left") == "the blue office chair on the left"
    assert _parse_vlm_label("a small white vase") == "a small white vase"
    assert _parse_vlm_label('"the wooden cabinet"') == "the wooden cabinet"
def test_parse_vlm_label_falls_back_for_full_sentence():
    assert _parse_vlm_label("It is a chair.") == "chair"
    assert _parse_vlm_label("This object appears to be a lamp.") == "lamp"


def test_select_detection_picks_centroid_nearest_user_bbox():
    detections = [
        _fake_detection(0.1, 0.1, area=0.2),
        _fake_detection(0.5, 0.5, area=0.3),
        _fake_detection(0.9, 0.9, area=0.5),
    ]
    selected = _select_detection(detections, (90, 90, 110, 110), img_w=200, img_h=200)
    assert selected is detections[1]


def test_classify_and_segment_runs_full_pipeline_with_fakes():
    base_rgb = np.zeros((160, 200, 3), dtype=np.uint8)
    bbox = (50, 40, 120, 100)
    falcon_calls: list[str] = []

    def fake_vlm(image_path: str, prompt: str) -> str:
        return "the brown wooden chair on the left"

    class FakeFalcon:
        def segment(self, image, query, task="segmentation"):
            falcon_calls.append(query)
            return [
                _fake_detection(0.4, 0.35, area=0.12),
                _fake_detection(0.8, 0.8, area=0.05),
            ]

    result = classify_and_segment(base_rgb, bbox, fake_vlm, FakeFalcon())

    assert isinstance(result, ExtractionResult)
    assert result.label == "the brown wooden chair on the left"
    assert falcon_calls == ["the brown wooden chair on the left"]
    assert result.cutout.mode == "RGBA"
    assert result.overlay is not None
    assert result.overlay.shape == (160, 200, 3)
    assert "chair" in result.detail


def test_classify_and_segment_reports_when_user_drew_nothing():
    result = classify_and_segment(None, None, lambda *_: "chair", None)
    assert result.label == ""
    assert result.cutout is None
    assert "框选" in result.detail


def test_classify_and_segment_reports_when_falcon_finds_nothing():
    base_rgb = np.zeros((160, 200, 3), dtype=np.uint8)
    result = classify_and_segment(
        base_rgb, (50, 40, 120, 100),
        lambda *_: "lamp",
        type("F", (), {"segment": lambda self, *a, **k: []})(),
    )
    assert result.label == "lamp"
    assert result.cutout is None
    assert "未检测到" in result.detail



def test_rle_mask_roundtrip_produces_pixel_accurate_alpha():
    """Verify the cutout uses Falcon's pixel-level RLE mask, not just a bbox crop."""
    from pycocotools import mask as mask_utils
    import base64 as b64

    # Create a 100×100 image with a known L-shaped mask region
    h, w = 100, 100
    base_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 20:60] = 1   # main block
    mask[60:80, 20:40] = 1   # appendage — non-rectangular

    # Encode as COCO RLE, then base64 like the Falcon server does
    rle = mask_utils.encode(np.asfortranarray(mask))
    counts_b64 = b64.b64encode(rle["counts"]).decode("ascii")

    class _FakeDet:
        pass
    det = _FakeDet()
    det.mask_rle = counts_b64
    det.mask_size = [h, w]
    det.mask_bbox = {"x": 0.3, "y": 0.4, "w": 0.4, "h": 0.5}
    det.bbox = det.mask_bbox

    cutout = _build_cutout(base_rgb, det, padding=0.0)
    assert cutout is not None
    assert cutout.mode == "RGBA"

    # The alpha channel must match the mask shape (non-rectangular)
    alpha = np.array(cutout)[:, :, 3]
    # Some pixels inside the bbox should be transparent (background)
    assert (alpha == 0).any(), "Expected transparent pixels outside the mask"
    # Some pixels inside the bbox should be opaque (object)
    assert (alpha == 255).any(), "Expected opaque pixels inside the mask"
    # The mask region should be non-rectangular (L-shape)
    alpha_bin = (alpha > 0).astype(np.uint8)
    # Top-right corner of the bbox should be empty (not part of L-shape)
    assert alpha_bin[:20, 40:].sum() == 0 or alpha_bin[60:, 40:].sum() == 0