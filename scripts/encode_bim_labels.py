#!/usr/bin/env python
"""Generate SigLIP2 text embeddings for BIM class vocabulary.

Produces two files in the output directory:
  - bim_text_emb.pt      (C, 768) float32, L2-normalized
  - bim_class_names.json  {"class_name": index, ...}

The embeddings are compatible with SceneSplat's feat.pt: cosine similarity
between feat (N,768) and text_emb (C,768) gives per-Gaussian class probabilities.

Usage:
    python scripts/encode_bim_labels.py
    python scripts/encode_bim_labels.py --class-names data/bim_class_names.txt --output-dir data/
    python scripts/encode_bim_labels.py --device cpu  # if GPU memory is tight
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Must match the model used during SceneSplat language label collection.
MODEL_NAME = "google/siglip2-base-patch16-512"


def encode_labels(labels: list[str], device: torch.device) -> torch.Tensor:
    """Encode class labels into L2-normalized SigLIP2 embeddings."""
    print(f"Loading {MODEL_NAME} on {device}...")
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # SceneSplat uses "this is a {label}" prefix — must match for compatibility.
    prompts = [f"this is a {label}" for label in labels]
    print(f"Encoding {len(labels)} labels: {prompts}")

    inputs = tokenizer(
        prompts, padding="max_length", max_length=64, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.get_text_features(**inputs)
        # transformers 5.x: SigLIP2 get_text_features returns BaseModelOutputWithPooling.
        if isinstance(output, torch.Tensor):
            embeddings = output
        elif hasattr(output, "pooler_output"):
            embeddings = output.pooler_output
        else:
            raise TypeError(f"Unexpected output type: {type(output)}")
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SigLIP2 text embeddings for BIM vocabulary"
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("data/bim_class_names.txt"),
        help="Text file with one class name per line",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. 'cuda' or 'cpu'). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    # Load class names
    if not args.class_names.exists():
        print(f"ERROR: {args.class_names} not found")
        return 1
    with open(args.class_names, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        print("ERROR: No class names found")
        return 1
    print(f"Loaded {len(labels)} class names from {args.class_names}")

    # Determine device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate embeddings
    embeddings = encode_labels(labels, device)
    print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Norm check (should be ~1.0): {embeddings.norm(dim=-1).mean():.6f}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.output_dir / "bim_text_emb.pt"
    json_path = args.output_dir / "bim_class_names.json"

    torch.save(embeddings, emb_path)
    class_map = {name: i for i, name in enumerate(labels)}
    with open(json_path, "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"Saved: {emb_path}")
    print(f"Saved: {json_path}")
    print(f"Classes: {class_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
