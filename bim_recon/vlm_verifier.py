"""VLM-verified element extraction via Ollama.

Two-stage element detection pipeline:

  Stage 1 (candidate generation): feat.pt semantic labels → candidate
  locations via radar scan. High recall, low precision.

  Stage 2 (VLM verification): for each candidate, render a targeted RGB
  image from 3DGS at the polar-derived viewpoint, then ask an Ollama VLM
  to confirm or reject. High precision.

The polar-to-viewpoint mapping is the key mathematical bridge: the radar
scan's azimuth angle θ directly determines the camera direction, and the
distance r determines where to aim.

Usage::

    from bim_recon.candidate_extractor import Candidate
    from bim_recon.vlm_verifier import verify_candidates

    results = verify_candidates(
        candidates, scene, scan_center, floor_z, output_dir,
        element_class="door",
    )
    confirmed = [r for r in results if r.confirmed]
"""
from __future__ import annotations

import base64
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bim_recon.candidate_extractor import Candidate
    from bim_recon.gs_scene import GSScene


@dataclass
class VerificationResult:
    """Result of VLM verification for a single candidate."""

    candidate: Any  # Candidate, avoided at runtime for circular import
    confirmed: Optional[bool]   # True / False / None (error)
    vlm_response: str
    image_path: str
    eye: List[float]
    target: List[float]
    fov: float
    theta: float
    r: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict() if hasattr(self.candidate, "to_dict") else None,
            "confirmed": self.confirmed,
            "vlm_response": self.vlm_response,
            "image_path": self.image_path,
            "eye": [round(v, 4) for v in self.eye],
            "target": [round(v, 4) for v in self.target],
            "fov": self.fov,
            "theta": round(self.theta, 2),
            "r": round(self.r, 4),
        }


# ---------------------------------------------------------------------------
# Pure-math: polar → camera viewpoint mapping
# ---------------------------------------------------------------------------

def compute_polar(
    world_x: float, world_y: float,
    scan_center: Tuple[float, float],
) -> Tuple[float, float]:
    """Compute polar coordinates (θ, r) from scan center to a world point.

    Returns:
        (theta_degrees, distance_meters) — θ in [0, 360), r ≥ 0.
    """
    cx, cy = scan_center
    dx = world_x - cx
    dy = world_y - cy
    r = math.sqrt(dx * dx + dy * dy)
    theta = math.degrees(math.atan2(dy, dx)) % 360.0
    return theta, r


def candidate_to_viewpoint(
    world_x: float,
    world_y: float,
    h_min: float,
    h_max: float,
    scan_center: Tuple[float, float],
    floor_z: float,
    eye_height: float = 1.5,
    fov: float = 60.0,
) -> Tuple[List[float], List[float], float]:
    """Map a candidate's polar position to a 3DGS camera pose.

    The camera is placed at the scan center (room center) at human eye
    height, aimed at the candidate's world position at its mid-height.

    Args:
        world_x, world_y: Candidate center in world XY.
        h_min, h_max: Candidate height range above floor (meters).
        scan_center: (cx, cy) room center.
        floor_z: Floor level world coordinate.
        eye_height: Camera height above floor (default 1.5m).
        fov: Field of view degrees.

    Returns:
        (eye, target, fov) — eye=[x,y,z], target=[x,y,z], fov=float.
    """
    cx, cy = scan_center
    h_mid = (h_min + h_max) / 2.0
    eye = [cx, cy, floor_z + eye_height]
    target = [world_x, world_y, floor_z + h_mid]
    return eye, target, fov


# ---------------------------------------------------------------------------
# Ollama VLM query
# ---------------------------------------------------------------------------

def query_ollama(
    image_path: str,
    prompt: str,
    model: str = "gemma4:12b",
    host: str = "localhost",
    port: int = 11434,
    timeout: int = 120,
) -> str:
    """Send an image + prompt to a local Ollama VLM and return the response.

    Uses the Ollama REST API (``POST /api/generate``). The image is
    base64-encoded inline.

    Args:
        image_path: Path to the PNG image file.
        prompt: Text prompt for the VLM.
        model: Ollama model name (must support vision).
        host, port: Ollama server address.
        timeout: Request timeout in seconds.

    Returns:
        The VLM's text response.

    Raises:
        urllib.error.URLError: If Ollama is unreachable.
        KeyError: If the response lacks a ``response`` field.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }).encode()

    url = f"http://{host}:{port}/api/generate"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
    return result.get("response", "")


def _build_prompt(element_class: str) -> str:
    """Build a structured VLM verification prompt."""
    return (
        f"This image is rendered from inside a room. "
        f"Is there a {element_class.upper()} visible in this image? "
        f"Answer with CONFIRMED or REJECTED on the first line, "
        f"then briefly describe what you see."
    )


def _parse_vlm_response(response: str) -> Tuple[Optional[bool], str]:
    """Parse VLM response into (confirmed, raw_text).

    Looks for CONFIRMED or REJECTED in the first line.
    """
    first_line = response.strip().split("\n")[0].upper()
    if "CONFIRMED" in first_line:
        return True, response
    if "REJECTED" in first_line:
        return False, response
    return None, response


# ---------------------------------------------------------------------------
# Full pipeline: render + VLM verify
# ---------------------------------------------------------------------------

def verify_candidates(
    candidates: List[Any],
    scene: Any,
    scan_center: Tuple[float, float],
    floor_z: float,
    output_dir: Path,
    element_class: str = "door",
    ollama_model: str = "gemma4:12b",
    ollama_host: str = "localhost",
    ollama_port: int = 11434,
    image_width: int = 800,
    image_height: int = 600,
    fov: float = 60.0,
    skip_vlm: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[VerificationResult]:
    """Render targeted images for candidates and verify via Ollama VLM.

    For each candidate:
      1. Compute camera pose from polar coordinates.
      2. Render a clean RGB image from 3DGS.
      3. Save image to ``output_dir``.
      4. Query Ollama VLM for confirmation.

    Args:
        candidates: List of :class:`Candidate` objects.
        scene: :class:`GSScene` with original weights loaded.
        scan_center: (cx, cy) room center.
        floor_z: Floor level world coordinate.
        output_dir: Directory to save rendered images.
        element_class: Element type for VLM prompt (e.g. "door").
        ollama_model: Ollama model name.
        skip_vlm: If True, only render images without VLM queries.

    Returns:
        List of :class:`VerificationResult`.
    """
    from bim_recon.gs_scene import look_at_pose
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = _build_prompt(element_class)
    results: List[VerificationResult] = []

    for i, cand in enumerate(candidates):
        eye, target, used_fov = candidate_to_viewpoint(
            cand.world_x, cand.world_y,
            cand.h_min, cand.h_max,
            scan_center, floor_z, fov=fov,
        )

        pose = look_at_pose(
            (eye[0], eye[1], eye[2]),
            (target[0], target[1], target[2]),
            up=(0.0, 0.0, 1.0),
        )
        render_result = scene.render(
            pose, width=image_width, height=image_height,
            fov_degrees=used_fov,
        )

        wall_tag = f"w{cand.wall_idx}" if cand.wall_idx is not None else "free"
        img_name = f"verify_{element_class}_{i}_{wall_tag}.png"
        img_path = str(output_dir / img_name)
        img = Image.fromarray(
            (render_result.colors * 255).clip(0, 255).astype(np.uint8)
        )
        img.save(img_path)

        vlm_text = ""
        confirmed: Optional[bool] = None
        if not skip_vlm:
            try:
                vlm_text = query_ollama(
                    img_path, prompt, ollama_model,
                    ollama_host, ollama_port,
                )
                confirmed, _ = _parse_vlm_response(vlm_text)
            except Exception as e:
                vlm_text = f"ERROR: {e}"

        theta, r = compute_polar(cand.world_x, cand.world_y, scan_center)
        result = VerificationResult(
            candidate=cand,
            confirmed=confirmed,
            vlm_response=vlm_text,
            image_path=img_name,
            eye=eye,
            target=target,
            fov=used_fov,
            theta=theta,
            r=r,
        )
        results.append(result)

        if progress_callback:
            progress_callback(i, len(candidates), result)

    return results
