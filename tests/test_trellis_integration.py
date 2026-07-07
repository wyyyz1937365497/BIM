"""Integration test for TrellisClient against a real mock HTTP server.

Starts a lightweight HTTP server on a random port, serves /health and /generate
endpoints, and verifies TrellisClient.health() + generate_mesh() work end-to-end
over real HTTP (no urllib mock).
"""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest


class _MockTrellisHandler(BaseHTTPRequestHandler):
    """Mock TRELLIS server handler."""

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self._json_response({"error": "not found"}, code=404)

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body)
            # Verify the request has expected fields
            assert "image_b64" in req
            assert "output_dir" in req
            assert "name" in req
            self._json_response({
                "status": "ok",
                "glb_path": str(Path(req["output_dir"]) / f"{req['name']}.glb"),
                "gaussian_path": str(Path(req["output_dir"]) / f"{req['name']}.ply"),
                "preview_path": None,
                "seed": req.get("seed", 1),
            })
        else:
            self._json_response({"error": "not found"}, code=404)

    def _json_response(self, data: dict, code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        pass  # suppress logging


@pytest.fixture
def mock_trellis_server():
    """Start a mock TRELLIS HTTP server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), _MockTrellisHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    thread.join(timeout=5)


class TestTrellisClientIntegration:
    """Integration tests hitting a real mock HTTP server."""

    def test_health_returns_true_when_server_up(self, mock_trellis_server, tmp_path):
        client = TrellisClient(host="127.0.0.1", port=mock_trellis_server, timeout=10)
        assert client.health() is True

    def test_health_returns_false_when_server_down(self):
        client = TrellisClient(host="127.0.0.1", port=19999, timeout=2)
        assert client.health() is False

    def test_generate_mesh_round_trip(self, mock_trellis_server, tmp_path):
        image_path = tmp_path / "input.png"
        Image.new("RGB", (16, 16), "blue").save(image_path, format="PNG")
        output_dir = tmp_path / "meshes"
        output_dir.mkdir()

        client = TrellisClient(host="127.0.0.1", port=mock_trellis_server, timeout=10)
        result = client.generate_mesh(TrellisMeshRequest(
            image_path=image_path,
            output_dir=output_dir,
            name="test_chair",
            seed=42,
        ))

        assert result.glb_path == output_dir / "test_chair.glb"
        assert result.gaussian_path == output_dir / "test_chair.ply"
        assert result.seed == 42

    def test_generate_mesh_rejects_missing_image(self, mock_trellis_server, tmp_path):
        client = TrellisClient(host="127.0.0.1", port=mock_trellis_server, timeout=10)
        request = TrellisMeshRequest(
            image_path=tmp_path / "nonexistent.png",
            output_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            client.generate_mesh(request)
