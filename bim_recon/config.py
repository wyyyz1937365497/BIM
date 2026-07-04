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
class AppConfig:
    """Top-level application config."""
    vlm: ModelConfig
    llm: ModelConfig
    revit_mcp: RevitMCPConfig


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

    return AppConfig(
        vlm=ModelConfig(
            provider=vlm_raw.get("provider", "ollama"),
            api_base=vlm_raw.get("api_base", "http://127.0.0.1:11434"),
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
    )


def get_llm_model(config: AppConfig | None = None):
    """Create a smolagents model instance from config (OpenAI-compatible API)."""
    from smolagents import OpenAIServerModel

    cfg = config or load_config()
    m = cfg.llm

    # Always use OpenAIServerModel — works with Ollama (/v1), OpenAI, Azure, etc.
    return OpenAIServerModel(
        model_id=m.model,
        api_base=m.api_base,
        api_key=m.api_key or "empty",
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
