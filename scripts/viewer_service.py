"""Run the asynchronous SceneSplat Mini Viewer manager API."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

from bim_recon.config import load_config
from bim_recon.viewer_service import app


if __name__ == "__main__":
    config = load_config().viewer_service
    uvicorn.run(app, host=config.host, port=config.port)
