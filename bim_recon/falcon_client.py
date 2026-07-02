"""HTTP client for the Falcon-Perception inference server.

Lives in the ``bim-recon`` environment and communicates with the
``falcon_inference_server.py`` running in ``transformerv`` via HTTP.

This bridge exists because gsplat and falcon-perception require
incompatible PyTorch/CUDA versions and cannot share a conda env.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FalconDetection:
    """One detection returned by the Falcon server."""

    bbox: dict                    # {"x","y","w","h"} normalized [0,1]
    mask_bbox: Optional[dict]     # tight bbox from mask, or None
    mask_area_ratio: Optional[float]


class FalconClient:
    """Thin HTTP client wrapping the Falcon inference server.

    Args:
        host: Server hostname (default ``127.0.0.1``).
        port: Server port (default ``8390``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8390,
        timeout: int = 300,
    ):
        self._base = f"http://{host}:{port}"
        self._timeout = timeout

    # ── public API ──────────────────────────────────────────────────

    def health(self) -> bool:
        """Return True if the server is reachable and model is loaded."""
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def segment(
        self,
        image: Image.Image,
        query: str,
        task: str = "segmentation",
    ) -> List[FalconDetection]:
        """Run segmentation/detection on *image* with natural-language *query*.

        Args:
            image: PIL image (will be PNG-encoded and base64'd).
            query: Object to find, e.g. ``"window"`` or ``"door"``.
            task: ``"segmentation"`` (default) or ``"detection"``.

        Returns:
            List of :class:`FalconDetection`. Empty if nothing found.
        """
        image_b64 = self._encode_image(image)

        payload = json.dumps({
            "image_b64": image_b64,
            "query": query,
            "task": task,
        }).encode("utf-8")

        resp = self._post_json("/segment", payload)

        detections: List[FalconDetection] = []
        for d in resp.get("detections", []):
            detections.append(FalconDetection(
                bbox=d["bbox"],
                mask_bbox=d.get("mask_bbox"),
                mask_area_ratio=d.get("mask_area_ratio"),
            ))
        return detections

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _get(self, path: str) -> dict:
        req = Request(f"{self._base}{path}")
        with urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, body: bytes) -> dict:
        req = Request(
            f"{self._base}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
