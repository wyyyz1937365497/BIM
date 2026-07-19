"""B-class object extraction: rough-bbox → VLM referring expression → Falcon segment → cutout.

This module implements the interactive manual-extraction pipeline:

1. The user captures a viewpoint from the 3D viewer and renders it.
2. The user roughly paints over the target object with a brush (only the
   tight bounding box of the paint is used — pixel precision is irrelevant).
3. :func:`classify_and_segment` sends the bbox-annotated render to a VLM
   and asks for a **referring expression** — a concise natural-language
   description that uniquely identifies the object (Falcon's prompt template
   is ``"Segment these expressions in the image: {query}"``, so rich
   expressions like ``"the blue armchair on the left near the window"``
   resolve to exactly one detection instead of every instance of a class).
4. Falcon segments using that expression; its RLE mask becomes the alpha
   channel of a clean RGBA cutout, ready for TRELLIS mesh generation.

The helper keeps the Gradio callback thin and is independently testable.
"""
from __future__ import annotations

import io
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "an", "the", "this", "that", "it", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "object", "item",
    "thing", "piece", "yes", "no", "image", "photo", "picture", "shown",
    "visible", "see", "looks", "like", "appears",
}

@dataclass
class ExtractionResult:
    """Outcome of :func:`classify_and_segment`."""

    label: str
    cutout: Image.Image | None
    overlay: np.ndarray | None
    detail: str


def _extract_user_bbox(mask_editor_value: dict[str, Any] | None) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return ``(base_rgb, (x0, y0, x1, y1))`` from a Gradio ``ImageMask`` dict.

    The bbox is the tight pixel rectangle enclosing every painted pixel.
    Returns ``None`` when the editor has no paint or no background image.
    """
    if not mask_editor_value:
        return None
    layers = mask_editor_value.get("layers") or []
    if not layers:
        return None
    mask_arr = layers[0]
    if not isinstance(mask_arr, np.ndarray) or mask_arr.ndim < 3:
        return None

    background = mask_editor_value.get("background")
    if isinstance(background, np.ndarray):
        base_img = Image.fromarray(background.astype(np.uint8)).convert("RGB")
    else:
        base_img = Image.fromarray(mask_arr[:, :, :3].astype(np.uint8)).convert("RGB")
    base_rgb = np.array(base_img)

    alpha = mask_arr[:, :, 3] if mask_arr.shape[2] == 4 else np.zeros(mask_arr.shape[:2])
    has_paint = alpha > 10
    if not has_paint.any():
        return None
    rows = np.any(has_paint, axis=1)
    cols = np.any(has_paint, axis=0)
    y_sel = np.where(rows)[0]
    x_sel = np.where(cols)[0]
    bbox = (int(x_sel[0]), int(y_sel[0]), int(x_sel[-1]), int(y_sel[-1]))
    return base_rgb, bbox


def _parse_vlm_label(response: str) -> str:
    """Reduce a VLM reply to a clean referring expression for Falcon.

    Referring expressions naturally start with ``the``/``a`` (e.g.
    ``"the blue chair on the left"``), so only fall back to single-word
    extraction when the VLM ignores the instruction and returns a full
    sentence like ``"It is a chair."``.
    """
    if not response:
        return ""
    cleaned = response.strip().strip("\"'`.,;:!?()[]{}").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    words = re.split(r"\s+", lowered)

    # Detect full-sentence patterns — pronoun/subject + copula — that
    # indicate the VLM ignored the "expression only" instruction.
    first_two = " ".join(words[:2])
    sentence_starts = {
        "it is", "it's", "this is", "that is", "that's",
        "there is", "there are", "i see", "the object",
        "the image", "the photo", "this object",
    }
    looks_like_sentence = (
        first_two in sentence_starts
        or (len(words) > 2 and " ".join(words[:3]) in {"i can see", "this appears"})
    )

    if not looks_like_sentence and len(words) <= 12:
        return lowered

    # Fallback: extract the first content word.
    candidates = [w for w in words if w and w not in _STOPWORDS]
    return candidates[0] if candidates else words[0]


def _annotate_bbox(base_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Draw a thick red rectangle on a copy of ``base_rgb`` for VLM input."""
    annotated = Image.fromarray(base_rgb.copy())
    draw = ImageDraw.Draw(annotated)
    x0, y0, x1, y1 = bbox
    pad = 4
    draw.rectangle(
        [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
        outline=(255, 0, 0), width=6,
    )
    return annotated


def _select_detection(
    detections: list, bbox: tuple[int, int, int, int], img_w: int, img_h: int,
):
    """Pick the Falcon detection whose centre is closest to the user's bbox centre."""
    cx_user = (bbox[0] + bbox[2]) / 2.0 / img_w
    cy_user = (bbox[1] + bbox[3]) / 2.0 / img_h
    best = None
    best_dist = float("inf")
    for det in detections:
        norm_bbox = det.mask_bbox or det.bbox
        if not norm_bbox:
            continue
        dx = norm_bbox["x"] - cx_user
        dy = norm_bbox["y"] - cy_user
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best = det
    return best


def _build_cutout(
    base_rgb: np.ndarray, detection, padding: float = 0.06,
) -> Image.Image | None:
    """Crop ``base_rgb`` to the detection bbox and apply its RLE mask as alpha."""
    norm_bbox = detection.mask_bbox or detection.bbox
    if not norm_bbox:
        return None
    h_img, w_img = base_rgb.shape[:2]
    x0 = max(0, int((norm_bbox["x"] - norm_bbox["w"] / 2 - padding) * w_img))
    y0 = max(0, int((norm_bbox["y"] - norm_bbox["h"] / 2 - padding) * h_img))
    x1 = min(w_img, int((norm_bbox["x"] + norm_bbox["w"] / 2 + padding) * w_img))
    y1 = min(h_img, int((norm_bbox["y"] + norm_bbox["h"] / 2 + padding) * h_img))
    if x1 <= x0 or y1 <= y0:
        return None

    cropped = Image.fromarray(base_rgb).crop((x0, y0, x1, y1)).convert("RGBA")
    alpha = _rle_to_alpha(detection, x0, y0, x1, y1, cropped.size)
    if alpha is not None:
        cropped.putalpha(alpha)
    return cropped


def _rle_to_alpha(detection, x0: int, y0: int, x1: int, y1: int, crop_size):
    """Decode a Falcon RLE mask to a cropped PIL ``L`` mode alpha channel."""
    rle = getattr(detection, "mask_rle", None)
    size = getattr(detection, "mask_size", None)
    if not rle or not size:
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        logger.warning("pycocotools not available; falling back to opaque alpha")
        return None
    counts = rle.encode("utf-8") if isinstance(rle, str) else rle
    try:
        full_mask = mask_utils.decode({"counts": counts, "size": list(size)})
    except Exception as exc:
        logger.warning("RLE decode failed (%s); using opaque alpha", exc)
        return None
    mask_crop = full_mask[y0:y1, x0:x1]
    if mask_crop.shape != (crop_size[1], crop_size[0]):
        return None
    return Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")


def _draw_segmentation_overlay(
    base_rgb: np.ndarray, detection,
) -> np.ndarray:
    """Overlay the selected detection's mask edge on the render for user feedback."""
    overlay = base_rgb.copy()
    h_img, w_img = overlay.shape[:2]
    norm_bbox = detection.mask_bbox or detection.bbox
    if norm_bbox:
        x0 = int((norm_bbox["x"] - norm_bbox["w"] / 2) * w_img)
        y0 = int((norm_bbox["y"] - norm_bbox["h"] / 2) * h_img)
        x1 = int((norm_bbox["x"] + norm_bbox["w"] / 2) * w_img)
        y1 = int((norm_bbox["y"] + norm_bbox["h"] / 2) * h_img)
        pil = Image.fromarray(overlay)
        ImageDraw.Draw(pil).rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        overlay = np.array(pil)
    return overlay


def classify_and_segment(
    mask_editor_value: dict[str, Any] | None,
    vlm_caller,
    falcon_client,
    *,
    vlm_prompt: str = (
        "Look at the object inside the red rectangle. "
        "Write a concise referring expression that uniquely identifies "
        "this exact object for a segmentation model. Include the object "
        "type plus 1-2 distinguishing features (color, position relative "
        "to other objects, material). "
        "Examples: 'the blue office chair on the left', "
        "'the white ceramic vase on the wooden shelf'. "
        "Reply with ONLY the expression, no other words."
    ),
) -> ExtractionResult:
    """Run the full classify → segment → cutout pipeline.

    Parameters
    ----------
    mask_editor_value
        The dict emitted by Gradio's ``ImageMask`` component. Must contain a
        painted layer and a background image.
    vlm_caller
        Callable ``(image_path: str, prompt: str) -> str`` that sends the
        annotated image to a VLM and returns its text reply. The Gradio
        callback injects :func:`bim_recon.vlm_verifier.query_vlm` with the
        configured endpoint.
    falcon_client
        ``FalconClient`` (or compatible) with a ``segment(image, query)``
        method returning a list of detections.

    Returns
    -------
    ExtractionResult
        ``label`` is always set (may be empty on VLM failure). ``cutout`` and
        ``overlay`` are ``None`` when the pipeline cannot complete. ``detail``
        is a user-facing status string.
    """
    extracted = _extract_user_bbox(mask_editor_value)
    if extracted is None:
        return ExtractionResult("", None, None, "⚠️ 请先在渲染图上用画笔粗略框选目标物体")
    base_rgb, bbox = extracted
    h_img, w_img = base_rgb.shape[:2]

    annotated = _annotate_bbox(base_rgb, bbox)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        annotated_path = tmp.name
    annotated.save(annotated_path)

    try:
        vlm_response = vlm_caller(annotated_path, vlm_prompt)
    except Exception as exc:
        Path(annotated_path).unlink(missing_ok=True)
        return ExtractionResult("", None, None, f"❌ VLM 调用失败: {exc}")
    finally:
        Path(annotated_path).unlink(missing_ok=True)

    label = _parse_vlm_label(vlm_response)
    if not label:
        return ExtractionResult("", None, None, f"⚠️ VLM 未能识别物体（原始回复: {vlm_response!r}）")

    if falcon_client is None:
        return ExtractionResult(label, None, None, f"⚠️ 识别为「{label}」，但 Falcon 服务不可用")

    try:
        detections = falcon_client.segment(
            Image.fromarray(base_rgb), label, task="segmentation",
        )
    except Exception as exc:
        return ExtractionResult(label, None, None, f"❌ Falcon 分割失败: {exc}")

    if not detections:
        return ExtractionResult(label, None, None, f"⚠️ Falcon 未检测到「{label}」")

    selected = _select_detection(detections, bbox, w_img, h_img)
    if selected is None:
        return ExtractionResult(label, None, None, f"⚠️ Falcon 检测结果与框选区域不匹配")

    cutout = _build_cutout(base_rgb, selected)
    overlay = _draw_segmentation_overlay(base_rgb, selected)
    area_pct = round((selected.mask_area_ratio or 0) * 100, 1)
    return ExtractionResult(
        label, cutout, overlay,
        f"✅ 识别为「{label}」，Falcon 分割覆盖 {area_pct}% 画面",
    )
