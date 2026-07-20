#!/usr/bin/env python
"""Diagnose VLM empty-response issue.

Sends the saved debug artifacts to the VLM API and prints the FULL response
object so we can see finish_reason, token usage, and any error details that
the normal query_vlm wrapper silently discards.

Usage::

    python scripts/test_vlm_debug.py                          # latest debug dir
    python scripts/test_vlm_debug.py path/to/debug_xxxxx/     # specific dir
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from openai import OpenAI
from bim_recon.config import load_config


def _encode_image(path: str, max_size: int = 768) -> str:
    """Encode an image as a JPEG data URL, resizing if needed (aspect-preserving)."""
    img = Image.open(path)
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _call_vlm(client: OpenAI, model: str, image_path: str, prompt: str):
    """Call VLM and print every field of the response."""
    data_url = _encode_image(image_path)
    print(f"\n{'='*70}")
    print(f"Image: {image_path}")
    img = Image.open(image_path)
    print(f"  Original size: {img.size}")
    print(f"  Encoded max: 768px, JPEG q85")
    print(f"Model: {model}")
    print(f"Prompt: {prompt!r}")

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )

    choice = resp.choices[0]
    print(f"\n--- Response ---")
    print(f"  finish_reason: {choice.finish_reason!r}")
    print(f"  message.role:  {choice.message.role!r}")
    print(f"  message.content: {choice.message.content!r}")
    if hasattr(resp, "usage") and resp.usage:
        print(f"  usage: prompt_tokens={resp.usage.prompt_tokens}, "
              f"completion_tokens={resp.usage.completion_tokens}, "
              f"total={resp.usage.total_tokens}")
    if hasattr(resp, "model"):
        print(f"  model: {resp.model}")
    return choice.message.content


def main() -> int:
    cfg = load_config()
    print(f"VLM endpoint: {cfg.vlm.api_base}")
    print(f"VLM model:    {cfg.vlm.model}")
    print(f"API key set:  {bool(cfg.vlm.api_key)}")

    # Find debug dir
    if len(sys.argv) > 1:
        debug_dir = Path(sys.argv[1])
    else:
        mesh_dir = ROOT / "output" / "splat" / "_trellis_meshes"
        dirs = sorted(mesh_dir.glob("debug_*"))
        if not dirs:
            print("No debug_* directories found")
            return 1
        debug_dir = dirs[-1]
    print(f"Debug dir:   {debug_dir}")

    annotated = str(debug_dir / "01_vlm_input_annotated.png")
    cropped = str(debug_dir / "01b_vlm_input_cropped.png")

    client = OpenAI(base_url=cfg.vlm.api_base, api_key=cfg.vlm.api_key or "empty", timeout=60)

    prompts = [
        ("annotated + Chinese referring-expr prompt", annotated,
         "请看图片中红色方框内的物体，用英文写一个简短的指代短语来描述这个物体。"
         "需要包含物体种类和1-2个区分特征（颜色、位置、材质等）。"
         "例如：the blue chair on the left, the white vase on the table。"
         "只回答这个短语即可。"),
        ("annotated + simple English prompt", annotated,
         "What object is inside the red rectangle? Answer with one English word."),
        ("cropped + simple English prompt", cropped,
         "What is this object? Answer with one English word."),
        ("annotated + generic describe prompt", annotated,
         "请描述这张图片中的内容。"),
    ]

    for label, img_path, prompt in prompts:
        if not Path(img_path).exists():
            print(f"\n[SKIP] {img_path} not found")
            continue
        try:
            _call_vlm(client, cfg.vlm.model, img_path, prompt)
        except Exception as exc:
            print(f"\n[ERROR] {label}: {type(exc).__name__}: {exc}")

    # Also try alternate models if the primary fails
    alternate_models = ["glm-4v", "glm-4v-plus", "glm-4v-flash", "glm-4.3v"]
    for alt_model in alternate_models:
        print(f"\n{'='*70}")
        print(f"Trying alternate model: {alt_model}")
        try:
            _call_vlm(client, alt_model, annotated, "What object is inside the red rectangle? Answer with one English word.")
        except Exception as exc:
            print(f"[ERROR] {alt_model}: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
