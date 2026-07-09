"""Centralized configuration loader for the 3DGS→BIM pipeline.

Reads ``config.json`` from the project root. All components (pipeline VLM,
AI agent LLM, Revit MCP server) read their settings from this single file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = ROOT / "config.json"


@dataclass(frozen=True)
class ModelConfig:
    """One LLM/VLM endpoint."""
    provider: str = "ollama"
    api_base: str = "http://127.0.0.1:11434"
    model: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class RevitMCPConfig:
    """Revit MCP server launch parameters."""
    command: str = "node"
    args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrellisConfig:
    """TRELLIS mesh generation HTTP server config."""

    host: str = "127.0.0.1"
    port: int = 8391
    model: str = "microsoft/TRELLIS-image-large"
    timeout: int = 1800


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
    llm: ModelConfig
    revit_mcp: RevitMCPConfig
    trellis: TrellisConfig
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
    llm_raw = raw.get("llm", {})
    mcp_raw = raw.get("revit_mcp", {})
    trellis_raw = raw.get("trellis", {})
    routing_raw = raw.get("element_routing", {})

    return AppConfig(
        vlm=ModelConfig(
            provider=vlm_raw.get("provider", "ollama"),
            api_base=vlm_raw.get("api_base", "http://127.0.0.1:11434/v1"),
            model=vlm_raw.get("model", "gemma4:12b"),
            api_key=vlm_raw.get("api_key", ""),
        ),
        llm=ModelConfig(
            provider=llm_raw.get("provider", "ollama"),
            api_base=llm_raw.get("api_base", "http://127.0.0.1:11434"),
            model=llm_raw.get("model", "qwen2.5:32b"),
            api_key=llm_raw.get("api_key", ""),
        ),
        revit_mcp=RevitMCPConfig(
            command=mcp_raw.get("command", "node"),
            args=mcp_raw.get("args", []),
        ),
        trellis=TrellisConfig(
            host=trellis_raw.get("host", "127.0.0.1"),
            port=trellis_raw.get("port", 8391),
            model=trellis_raw.get("model", "microsoft/TRELLIS-image-large"),
            timeout=trellis_raw.get("timeout", 1800),
        ),
        element_routing=ElementRoutingConfig(
            routing=routing_raw if routing_raw else ElementRoutingConfig().routing,
        ),
    )


def get_llm_model(config: AppConfig | None = None):
    """Create a smolagents model instance from config (OpenAI-compatible API).

    Sets ``max_retries=1`` on the underlying OpenAI client so that rate-limit
    (429) and auth errors fail fast instead of blocking the UI for minutes
    during exponential backoff.
    """
    from smolagents import OpenAIServerModel

    cfg = config or load_config()
    m = cfg.llm

    return OpenAIServerModel(
        model_id=m.model,
        api_base=m.api_base,
        api_key=m.api_key or "empty",
        max_retries=1,
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
        "llm": {
            "provider": config.llm.provider,
            "api_base": config.llm.api_base,
            "model": config.llm.model,
            "api_key": config.llm.api_key,
        },
        "revit_mcp": {
            "command": config.revit_mcp.command,
            "args": config.revit_mcp.args,
        },
        "trellis": {
            "host": config.trellis.host,
            "port": config.trellis.port,
            "model": config.trellis.model,
            "timeout": config.trellis.timeout,
        },
        "element_routing": dict(config.element_routing.routing),
    }
    _CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


def test_llm_connection(api_base: str, api_key: str, model: str) -> str:
    """Test LLM connectivity. Returns status string."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key or "empty")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content or ""
        return f"✅ 连接成功，模型回复: {reply.strip()[:30]}"
    except Exception as e:
        return f"❌ 连接失败: {e}"


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
