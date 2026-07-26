"""Centralized configuration loader for deterministic 3DGS→BIM workflows.

Reads ``config.json`` from the project root. Pipeline verification, Revit MCP,
TRELLIS, and element routing share this typed configuration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = ROOT / "config.json"


@dataclass(frozen=True)
class ModelConfig:
    """Vision-model endpoint used for deterministic image verification."""
    provider: str = "ollama"
    api_base: str = "http://127.0.0.1:11434"
    model: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class RevitMCPConfig:
    """Revit MCP server launch parameters."""
    command: str = "node"
    args: list[str] = field(default_factory=list)

    timeout: int = 120

@dataclass(frozen=True)
class FalconConfig:
    """Falcon-Perception HTTP server config."""

    host: str = "127.0.0.1"
    port: int = 18390
    timeout: int = 300


@dataclass(frozen=True)
class TrellisConfig:
    """TRELLIS mesh generation HTTP server config."""

    host: str = "127.0.0.1"
    port: int = 18391
    model: str = "microsoft/TRELLIS-image-large"
    timeout: int = 1800


@dataclass(frozen=True)
class ViewerServiceConfig:
    """FastAPI process manager for the standalone Mini Viewer."""

    host: str = "127.0.0.1"
    port: int = 18083
    viewer_port: int = 18081


@dataclass(frozen=True)
class ElementRoutingConfig:
    """A/B class routing per element type.

    "A" → parametric Revit elements (Wall/Door/Window via MCP tools)
    "B" → TRELLIS mesh generation + DirectShape insertion
    """

    routing: dict[str, str] = field(default_factory=lambda: {
        "door": "A",
        "window": "A",
        "column": "A",
        "furniture": "B",
    })

    def get_route(self, element_type: str) -> str:
        """Return 'A' or 'B' for the given element type. Defaults to 'A'."""
        return self.routing.get(element_type, "A")

    def is_b_class(self, element_type: str) -> bool:
        """Return True if this element type should use TRELLIS mesh route."""
        return self.get_route(element_type) == "B"

    def b_class_types(self) -> list[str]:
        """Return all element types routed to B-class."""
        return sorted(k for k, v in self.routing.items() if v == "B")

    def a_class_types(self) -> list[str]:
        """Return all element types routed to A-class."""
        return sorted(k for k, v in self.routing.items() if v == "A")


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""
    vlm: ModelConfig
    revit_mcp: RevitMCPConfig
    falcon: FalconConfig
    trellis: TrellisConfig
    viewer_service: ViewerServiceConfig = field(default_factory=ViewerServiceConfig)
    element_routing: ElementRoutingConfig = field(default_factory=ElementRoutingConfig)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and parse ``config.json``.

    If the file doesn't exist, returns sensible defaults (Ollama local).
    """
    p = Path(path) if path else _CONFIG_PATH
    if p.exists():
        raw = json.loads(p.read_text("utf-8"))
    else:
        raw = {}

    vlm_raw = raw.get("vlm", {})
    mcp_raw = raw.get("revit_mcp", {})
    falcon_raw = raw.get("falcon", {})
    trellis_raw = raw.get("trellis", {})
    viewer_raw = raw.get("viewer_service", {})
    routing_raw = raw.get("element_routing", {})

    return AppConfig(
        vlm=ModelConfig(
            provider=vlm_raw.get("provider", "ollama"),
            api_base=vlm_raw.get("api_base", "http://127.0.0.1:11434/v1"),
            model=vlm_raw.get("model", "gemma4:12b"),
            api_key=vlm_raw.get("api_key", ""),
        ),
        revit_mcp=RevitMCPConfig(
            command=mcp_raw.get("command", "node"),
            args=mcp_raw.get("args", []),
            timeout=int(mcp_raw.get("timeout", 120)),
        ),
        falcon=FalconConfig(
            host=falcon_raw.get("host", "127.0.0.1"),
            port=int(falcon_raw.get("port", 18390)),
            timeout=int(falcon_raw.get("timeout", 300)),
        ),
        trellis=TrellisConfig(
            host=trellis_raw.get("host", "127.0.0.1"),
            port=trellis_raw.get("port", 18391),
            model=trellis_raw.get("model", "microsoft/TRELLIS-image-large"),
            timeout=trellis_raw.get("timeout", 1800),
        ),
        viewer_service=ViewerServiceConfig(
            host=viewer_raw.get("host", "127.0.0.1"),
            port=int(viewer_raw.get("port", 18083)),
            viewer_port=int(viewer_raw.get("viewer_port", 18081)),
        ),
        element_routing=ElementRoutingConfig(
            routing=routing_raw if routing_raw else ElementRoutingConfig().routing,
        ),
    )



def save_config(config: AppConfig) -> None:
    """Write config back to config.json."""
    data = {
        "vlm": {
            "provider": config.vlm.provider,
            "api_base": config.vlm.api_base,
            "model": config.vlm.model,
            "api_key": config.vlm.api_key,
        },
        "revit_mcp": {
            "command": config.revit_mcp.command,
            "args": config.revit_mcp.args,
            "timeout": config.revit_mcp.timeout,
        },
        "falcon": {
            "host": config.falcon.host,
            "port": config.falcon.port,
            "timeout": config.falcon.timeout,
        },
        "trellis": {
            "host": config.trellis.host,
            "port": config.trellis.port,
            "model": config.trellis.model,
            "timeout": config.trellis.timeout,
        },
        "viewer_service": {
            "host": config.viewer_service.host,
            "port": config.viewer_service.port,
            "viewer_port": config.viewer_service.viewer_port,
        },
        "element_routing": dict(config.element_routing.routing),
    }
    _CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")



def test_vlm_connection(api_base: str, api_key: str, model: str) -> str:
    """Test VLM connectivity with a tiny image. Returns status string.

    Sends a 1x1 white pixel to verify the endpoint accepts vision input.
    """
    try:
        import base64
        from io import BytesIO
        from openai import OpenAI

        # 1x1 white PNG
        from PIL import Image
        buf = BytesIO()
        Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        client = OpenAI(base_url=api_base, api_key=api_key or "empty", timeout=30)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Reply in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content or ""
        return f"✅ VLM 连接成功，模型回复: {reply.strip()[:30]}"
    except Exception as e:
        return f"❌ VLM 连接失败: {e}"
