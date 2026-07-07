"""CLI for generating B-class complex component meshes via TRELLIS HTTP server."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.config import load_config
from bim_recon.trellis_client import TrellisClient, TrellisMeshRequest


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Parsed command-line arguments."""

    image: Path
    output_dir: Path
    name: str
    seed: int
    simplify: float
    texture_size: int


def parse_args(argv: list[str] | None = None) -> CliArgs:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate a GLB mesh through TRELLIS HTTP server")
    parser.add_argument("--image", required=True, type=Path, help="Input object image path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated GLB/PLY/preview")
    parser.add_argument("--name", default="trellis_mesh", help="Output artifact stem")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    ns = parser.parse_args(argv)
    return CliArgs(
        image=ns.image,
        output_dir=ns.output_dir,
        name=ns.name,
        seed=ns.seed,
        simplify=ns.simplify,
        texture_size=ns.texture_size,
    )


def main(argv: list[str] | None = None) -> int:
    """Run TRELLIS mesh generation and print JSON result."""
    args = parse_args(argv)
    cfg = load_config().trellis
    client = TrellisClient(host=cfg.host, port=cfg.port, timeout=cfg.timeout)
    result = client.generate_mesh(TrellisMeshRequest(
        image_path=args.image,
        output_dir=args.output_dir,
        name=args.name,
        seed=args.seed,
        simplify=args.simplify,
        texture_size=args.texture_size,
    ))
    print(json.dumps({
        "glb_path": str(result.glb_path),
        "gaussian_path": str(result.gaussian_path) if result.gaussian_path else None,
        "preview_path": str(result.preview_path) if result.preview_path else None,
        "seed": result.seed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
