"""Test smolagents + Revit MCP connectivity.

Usage:
    python scripts/test_agent.py

Prerequisites:
    - Revit running with MCP plugin loaded (port 8080)
    - config.json with LLM settings
    - smolagents installed: pip install smolagents[mcp]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.config import load_config, get_llm_model

# ── 1. Check Revit port ──────────────────────────────────────────────────
import socket

def check_port(port: int, host: str = "127.0.0.1", timeout: float = 3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

print("=" * 60)
print("1. Checking Revit MCP plugin (port 8080)...")
if check_port(8080):
    print("   ✅ Port 8080 is open — Revit plugin is reachable")
else:
    print("   ❌ Port 8080 is NOT reachable — start Revit + MCP plugin first")
    sys.exit(1)

# ── 2. Load config ───────────────────────────────────────────────────────
print("\n2. Loading config.json...")
cfg = load_config()
print(f"   LLM: {cfg.llm.model} @ {cfg.llm.api_base}")
print(f"   MCP: {cfg.revit_mcp.command} {cfg.revit_mcp.args[0]}")

# ── 3. Connect to Revit MCP server ──────────────────────────────────────
print("\n3. Connecting to Revit MCP server (stdio)...")
from smolagents import ToolCollection, ToolCallingAgent
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command=cfg.revit_mcp.command,
    args=cfg.revit_mcp.args,
)

# MUST keep the context manager alive at module scope
global _cm
_cm = ToolCollection.from_mcp(server_params, trust_remote_code=True)
tools_col = _cm.__enter__()  # type: ignore[attr-defined]
tools = tools_col.tools  # type: ignore[attr-defined]
print(f"   ✅ Connected! Loaded {len(tools)} tools:")
for t in tools:
    print(f"      - {t.name}")

# ── 4. Create agent ─────────────────────────────────────────────────────
print("\n4. Creating ToolCallingAgent...")
model = get_llm_model(cfg)
print(f"   Model: {type(model).__name__}")

agent = ToolCallingAgent(
    tools=tools,
    model=model,
    instructions="你是一个 Revit 自动化助手。所有坐标单位为毫米(mm)。",
    max_steps=5,
)
print("   ✅ Agent created")

# ── 5. Test: say_hello ──────────────────────────────────────────────────
print("\n5. Testing tool call: say_hello...")
try:
    result = agent.run("请在 Revit 中显示一个问候消息，内容为 'MCP 连接测试成功'")
    print(f"   ✅ Agent returned: {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ── 6. Test: get_current_view_info ──────────────────────────────────────
print("\n6. Testing tool call: get_current_view_info...")
try:
    result2 = agent.run("获取当前视图信息")
    print(f"   ✅ Agent returned: {str(result2)[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Done. If both tests passed, MCP connectivity is working.")
_cm.__exit__(None, None, None)  # type: ignore[attr-defined]
