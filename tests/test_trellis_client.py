"""Tests for TRELLIS HTTP integration boundary."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from bim_recon.trellis_client import (
    TrellisClient,
    TrellisMeshRequest,
)


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._payload


class TestTrellisClient:
    def test_generate_mesh_posts_image_and_returns_paths(self, tmp_path):
        image_path = tmp_path / "chair.png"
        output_dir = tmp_path / "out"
        _write_image(image_path)

        payload = {
            "status": "ok",
            "glb_path": str(output_dir / "chair.glb"),
            "gaussian_path": str(output_dir / "chair.ply"),
            "preview_path": str(output_dir / "chair_preview.mp4"),
            "seed": 7,
        }

        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse(payload)

        client = TrellisClient(host="127.0.0.1", port=8391, timeout=123)
        request = TrellisMeshRequest(
            image_path=image_path,
            output_dir=output_dir,
            name="chair",
            seed=7,
        )

        with patch("bim_recon.trellis_client.urlopen", side_effect=fake_urlopen):
            result = client.generate_mesh(request)

        assert captured["url"] == "http://127.0.0.1:8391/generate"
        assert captured["timeout"] == 123
        assert captured["body"]["name"] == "chair"
        assert captured["body"]["seed"] == 7
        assert captured["body"]["image_b64"]
        assert result.glb_path == output_dir / "chair.glb"
        assert result.gaussian_path == output_dir / "chair.ply"
        assert result.preview_path == output_dir / "chair_preview.mp4"

    def test_generate_mesh_rejects_missing_image_before_http(self, tmp_path):
        client = TrellisClient()
        request = TrellisMeshRequest(
            image_path=tmp_path / "missing.png",
            output_dir=tmp_path / "out",
        )

        with patch("bim_recon.trellis_client.urlopen") as mocked_urlopen:
            with pytest.raises(FileNotFoundError):
                client.generate_mesh(request)

        mocked_urlopen.assert_not_called()
