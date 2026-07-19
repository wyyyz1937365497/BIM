"""Contracts for the B-class bbox → VLM → Falcon → cutout pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image

from bim_recon.bmesh_extractor import (
    ExtractionResult,
    _extract_user_bbox,
    _parse_vlm_label,
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
    editor = _editor_dict((50, 40, 120, 100), img_size=(200, 160))
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

    result = classify_and_segment(editor, fake_vlm, FakeFalcon())

    assert isinstance(result, ExtractionResult)
    assert result.label == "the brown wooden chair on the left"
    assert falcon_calls == ["the brown wooden chair on the left"]
    assert result.cutout.mode == "RGBA"
    assert result.overlay is not None
    assert result.overlay.shape == (160, 200, 3)
    assert "chair" in result.detail


def test_classify_and_segment_reports_when_user_drew_nothing():
    result = classify_and_segment(
        {"background": np.zeros((10, 10, 3), np.uint8), "layers": []},
        lambda *_: "chair",
        None,
    )
    assert result.label == ""
    assert result.cutout is None
    assert "框选" in result.detail


def test_classify_and_segment_reports_when_falcon_finds_nothing():
    editor = _editor_dict((50, 40, 120, 100))
    result = classify_and_segment(
        editor,
        lambda *_: "lamp",
        type("F", (), {"segment": lambda self, *a, **k: []})(),
    )
    assert result.label == "lamp"
    assert result.cutout is None
    assert "未检测到" in result.detail
