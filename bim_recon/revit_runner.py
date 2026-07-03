"""Revit C# script runner — load templates from revit_scripts/ and dispatch.

This module provides a thin abstraction over the ``send_code_to_revit`` MCP
tool. It loads C# script templates from ``revit_scripts/``, lets the caller
inject parameters, and returns the MCP tool's result.

Two usage modes:

**1. Direct (from AI agent context)**

    Read the ``.cs`` file from ``revit_scripts/`` and call
    ``revit_send_code_to_revit`` directly — the runner is not needed.

**2. Pipeline (automated)**

    .. code-block:: python

        from bim_recon.revit_runner import RevitScriptRunner

        runner = RevitScriptRunner()

        # Query available door types
        result = runner.run("query_family_types", parameters=["OST_Doors"])

        # Create a custom-sized door
        result = runner.run("create_custom_door", parameters=[
            host_wall_id,           # long
            loc_x_ft,               # double (feet)
            loc_y_ft,               # double (feet)
            sill_height_ft,         # double (feet)
            width_ft,               # double (feet)
            height_ft,              # double (feet)
            False,                  # facingFlipped
        ])

The runner delegates actual code execution to the MCP tool; it does NOT
compile or run C# itself. In a pipeline context without MCP connectivity,
use :meth:`RevitScriptRunner.load_code` to get the raw C# string and send
it manually.

Unit conversion helpers (metric ↔ Revit internal feet) are provided for
convenience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "revit_scripts"

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

M_TO_FT = 3.280839895013123
MM_TO_FT = 1.0 / 304.8
FT_TO_M = 1.0 / M_TO_FT
FT_TO_MM = 304.8


def m_to_ft(m: float) -> float:
    """Metres → Revit internal feet."""
    return m * M_TO_FT


def mm_to_ft(mm: float) -> float:
    """Millimetres → Revit internal feet."""
    return mm * MM_TO_FT


def ft_to_mm(ft: float) -> float:
    """Revit internal feet → millimetres."""
    return ft * FT_TO_MM


def ft_to_m(ft: float) -> float:
    """Revit internal feet → metres."""
    return ft * FT_TO_M


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

class RevitScriptRunner:
    """Load C# scripts and dispatch to ``send_code_to_revit``.

    Parameters
    ----------
    scripts_dir
        Directory containing ``.cs`` script templates. Defaults to
        ``<project_root>/revit_scripts``.
    mcp_sender
        Optional callable that receives ``(code, parameters)`` and returns
        a result dict. When ``None``, :meth:`run` returns the code + params
        without executing (useful for inspection or when the caller has its
        own MCP dispatch mechanism).
    """

    def __init__(
        self,
        scripts_dir: Optional[Path] = None,
        mcp_sender: Optional[Any] = None,
    ) -> None:
        self.scripts_dir = Path(scripts_dir or SCRIPTS_DIR)
        self._mcp_sender = mcp_sender
        self._cache: Dict[str, str] = {}

    # -- Core API ----------------------------------------------------------

    def run(
        self,
        script_name: str,
        parameters: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a named script with the given parameters.

        Parameters
        ----------
        script_name
            Script name without extension (e.g. ``"create_custom_door"``).
        parameters
            Positional parameters passed to the C# ``Execute`` method as
            ``object[]``.

        Returns
        -------
        dict
            The result from ``send_code_to_revit`` (contains ``success``,
            ``result`` or ``errorMessage``).
        """
        code = self.load_code(script_name)
        params = list(parameters) if parameters else []

        if self._mcp_sender is not None:
            return self._mcp_sender(code=code, parameters=params)

        # No sender configured — return code + params for manual dispatch
        return {
            "_note": "No MCP sender configured. Call revit_send_code_to_revit manually.",
            "code": code,
            "parameters": params,
        }

    def load_code(self, script_name: str) -> str:
        """Load and return the C# code for a named script.

        Caches the result for subsequent calls.
        """
        if script_name in self._cache:
            return self._cache[script_name]

        path = self.scripts_dir / f"{script_name}.cs"
        if not path.exists():
            raise FileNotFoundError(
                f"Revit script not found: {path}\n"
                f"Available: {self.list_scripts()}"
            )
        code = path.read_text(encoding="utf-8")
        self._cache[script_name] = code
        return code

    def list_scripts(self) -> List[str]:
        """Return all available script names (without extension)."""
        return sorted(
            p.stem for p in self.scripts_dir.glob("*.cs")
            if not p.name.startswith("_")
        )

    # -- Convenience helpers -----------------------------------------------

    def create_door(
        self,
        host_wall_id: int,
        x_m: float,
        y_m: float,
        sill_m: float,
        width_m: float,
        height_m: float,
        facing_flipped: bool = False,
    ) -> Dict[str, Any]:
        """Create a custom-sized door (metric units)."""
        return self.run("create_custom_door", parameters=[
            host_wall_id,
            m_to_ft(x_m),
            m_to_ft(y_m),
            m_to_ft(sill_m),
            m_to_ft(width_m),
            m_to_ft(height_m),
            facing_flipped,
        ])

    def create_window(
        self,
        host_wall_id: int,
        x_m: float,
        y_m: float,
        sill_m: float,
        width_m: float,
        height_m: float,
        facing_flipped: bool = False,
    ) -> Dict[str, Any]:
        """Create a custom-sized window (metric units)."""
        return self.run("create_custom_window", parameters=[
            host_wall_id,
            m_to_ft(x_m),
            m_to_ft(y_m),
            m_to_ft(sill_m),
            m_to_ft(width_m),
            m_to_ft(height_m),
            facing_flipped,
        ])

    def query_family_types(self, category: str = "OST_Doors") -> Dict[str, Any]:
        """List all family types for a category."""
        return self.run("query_family_types", parameters=[category])

    def delete_by_category(
        self,
        category: str,
        name_prefix: str = "",
    ) -> Dict[str, Any]:
        """Delete elements by category, optionally filtered by type name prefix."""
        return self.run("delete_elements_by_category", parameters=[
            category, name_prefix,
        ])

    def create_walls_from_metric(
        self,
        walls: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Batch create walls from metric coordinate dicts.

        Each dict: ``{"x1", "y1", "x2", "y2", "thickness", "height"}`` (metres).
        """
        json_str = json.dumps(walls)
        return self.run("create_walls_from_json", parameters=[json_str])
