"""HTTP client for the TRELLIS mesh generation server.

Lives in the ``bim-recon`` environment and communicates with
``trellis_server/server.py`` running in the separate ``trellis``
conda environment via HTTP.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


@dataclass(frozen=True, slots=True)
class TrellisMeshRequest:
    """Input for one TRELLIS image-to-mesh generation request."""

    image_path: Path
    output_dir: Path
    name: str = "trellis_mesh"
    seed: int = 1
    simplify: float = 0.95
    texture_size: int = 1024


@dataclass(frozen=True, slots=True)
class TrellisMeshResult:
    """Generated TRELLIS mesh artifact paths."""

    glb_path: Path
    gaussian_path: Path | None
    preview_path: Path | None
    seed: int


class TrellisClient:
    """Thin HTTP client wrapping the TRELLIS inference server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18391, timeout: int = 1800):
        self._base = f"http://{host}:{port}"
        self._timeout = timeout

    def health(self) -> bool:
        """Return True if the server is reachable and model is loaded."""
        try:
            req = Request(f"{self._base}/health")
            with urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "ok"
        except OSError:
            return False

    def generate_mesh(self, request: TrellisMeshRequest) -> TrellisMeshResult:
        """Generate a GLB mesh from an input image through the TRELLIS server."""
        if not request.image_path.exists():
            raise FileNotFoundError(request.image_path)

        payload = json.dumps({
            "image_b64": self._encode_image(request.image_path),
            "output_dir": str(request.output_dir),
            "name": request.name,
            "seed": request.seed,
            "simplify": request.simplify,
            "texture_size": request.texture_size,
        }).encode("utf-8")

        req = Request(
            f"{self._base}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return TrellisMeshResult(
            glb_path=Path(data["glb_path"]),
            gaussian_path=Path(data["gaussian_path"]) if data.get("gaussian_path") else None,
            preview_path=Path(data["preview_path"]) if data.get("preview_path") else None,
            seed=int(data["seed"]),
        )

    @staticmethod
    def _encode_image(path: Path) -> str:
        with Image.open(path) as image:
            buf = io.BytesIO()
            image.convert("RGBA").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
