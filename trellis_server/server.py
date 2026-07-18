"""TRELLIS image-to-mesh inference server.

Runs in the ``trellis`` conda environment. Loads TRELLIS once at startup and
serves image-to-GLB generation requests via HTTP so the ``bim-recon``
environment can call it without importing TRELLIS dependencies.

This file lives in the MAIN repo (not the TRELLIS submodule) so it survives
``git submodule update --init`` on a fresh clone.

Usage::

    conda activate trellis
    cd G:\\TJ\\BIM
    python trellis_server/server.py --port 18391

Endpoints:
    GET  /health    — liveness check
    POST /generate  — generate GLB + PLY from one input image
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ── TRELLIS repo must be on sys.path so `import trellis` works ──────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRELLIS_DIR = _REPO_ROOT / "TRELLIS"
if str(_TRELLIS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRELLIS_DIR))

# ── env vars: xformers backend (flash-attn unavailable on Windows) ─────
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trellis-server")

_pipeline = None


# ── request / response schemas ──────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request payload for image-to-GLB generation."""

    image_b64: str
    output_dir: str
    name: str = "trellis_mesh"
    seed: int = 1
    simplify: float = 0.95
    texture_size: int = 1024


class GenerateResponse(BaseModel):
    """Generated artifact paths."""

    status: str
    glb_path: str
    gaussian_path: str | None = None
    preview_path: str | None = None
    seed: int


# ── lifespan (startup) ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load TRELLIS image-to-3D model once at startup."""
    global _pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline

    model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
    logger.info("Loading TRELLIS model: %s", model)
    logger.info("ATTN_BACKEND=%s  SPCONV_ALGO=%s",
                os.environ.get("ATTN_BACKEND"), os.environ.get("SPCONV_ALGO"))
    _pipeline = TrellisImageTo3DPipeline.from_pretrained(model)
    _pipeline.cuda()
    logger.info("TRELLIS model ready")
    yield


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="TRELLIS Inference Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if _pipeline is not None else "loading"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate a GLB mesh from one input image."""
    if _pipeline is None:
        raise HTTPException(503, "TRELLIS model not loaded yet")

    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = base64.b64decode(req.image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGBA")

    try:
        outputs = _pipeline.run(
            image,
            seed=req.seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )
    except Exception as e:
        logger.error("TRELLIS pipeline.run() failed: %s", e)
        logger.error(traceback.format_exc())
        torch.cuda.empty_cache()
        raise HTTPException(500, f"TRELLIS generation failed: {e}") from e

    glb_path = output_dir / f"{req.name}.glb"
    gaussian_path = output_dir / f"{req.name}.ply"

    try:
        from trellis.utils import postprocessing_utils

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=req.simplify,
            texture_size=req.texture_size,
            verbose=False,
        )
        # Export via trimesh.Scene to ensure textures are properly embedded
        # in the GLB binary (direct Trimesh.export() can lose PBR textures).
        import trimesh
        scene = trimesh.Scene([glb])
        scene.export(str(glb_path))
        outputs["gaussian"][0].save_ply(str(gaussian_path))
    except Exception as e:
        logger.error("GLB/PLY export failed: %s", e)
        logger.error(traceback.format_exc())
        torch.cuda.empty_cache()
        raise HTTPException(500, f"GLB export failed: {e}") from e

    torch.cuda.empty_cache()

    return GenerateResponse(
        status="ok",
        glb_path=str(glb_path),
        gaussian_path=str(gaussian_path),
        preview_path=None,
        seed=req.seed,
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="TRELLIS inference server")
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large",
                        help="Model name or local path (e.g. G:/TJ/BIM/TRELLIS/TRELLIS-image-large)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18391)
    args = parser.parse_args()

    os.environ["TRELLIS_MODEL"] = args.model
    uvicorn.run(app, host=args.host, port=args.port)
