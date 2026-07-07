"""VLM-verified element extraction via OpenAI-compatible API.

Two-stage element detection pipeline:

  Stage 1 (candidate generation): feat.pt semantic labels → candidate
  locations via radar scan. High recall, low precision.

  Stage 2 (VLM verification): for each candidate, render a targeted RGB
  image from 3DGS at the polar-derived viewpoint, then ask a VLM
  (GPT-4o, GLM-4V, Qwen-VL, Ollama vision, etc.) to confirm or reject.
  High precision.

The polar-to-viewpoint mapping is the key mathematical bridge: the radar
scan's azimuth angle θ directly determines the camera direction, and the
distance r determines where to aim.

VLM config is loaded from ``config.json`` (see :mod:`bim_recon.config`).
The ``vlm.api_base`` should point to the ``/v1`` (or equivalent) endpoint
that supports the OpenAI Chat Completions format with ``image_url``.

Usage::

    from bim_recon.candidate_extractor import Candidate
    from bim_recon.vlm_verifier import verify_candidates

    results = verify_candidates(
        candidates, scene, scan_center, floor_z, output_dir,
        element_class="door",
        vlm_api_base="https://open.bigmodel.cn/api/paas/v4",
        vlm_model="glm-4v",
        vlm_api_key="your-key",
    )
    confirmed = [r for r in results if r.confirmed]
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass
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
    up_axis: int = 2,
) -> Tuple[List[float], List[float], float]:
    """Map a candidate's polar position to a 3DGS camera pose.

    The camera is placed at the scan center (room center) at human eye
    height, aimed at the candidate's world position at its mid-height.
    Supports any up-axis (0=x, 1=y, 2=z) by building eye/target/up
    via axis-index assignment, consistent with VirtualScanner.

    Args:
        world_x, world_y: Candidate center in world horizontal plane.
        h_min, h_max: Candidate height range above floor (meters).
        scan_center: (cx, cy) room center in horizontal plane.
        floor_z: Floor level world coordinate (on the up-axis).
        eye_height: Camera height above floor (default 1.5m).
        fov: Field of view degrees.
        up_axis: Which world axis is vertical (0=x, 1=y, 2=z).

    Returns:
        (eye, target, fov) — eye=[x,y,z], target=[x,y,z], fov=float.
    """
    cx, cy = scan_center
    h_mid = (h_min + h_max) / 2.0
    h_axes = [i for i in range(3) if i != up_axis]

    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = cx
    eye[h_axes[1]] = cy
    eye[up_axis] = floor_z + eye_height

    target = [0.0, 0.0, 0.0]
    target[h_axes[0]] = world_x
    target[h_axes[1]] = world_y
    target[up_axis] = floor_z + h_mid

    return eye, target, fov


# ---------------------------------------------------------------------------
# OpenAI-compatible VLM query
# ---------------------------------------------------------------------------

def query_vlm(
    image_path: str,
    prompt: str,
    api_base: str,
    model: str,
    api_key: str = "",
    timeout: int = 120,
    max_tokens: int = 200,
) -> str:
    """Send an image + prompt to an OpenAI-compatible VLM and return the response.

    Uses the standard Chat Completions API (``POST /chat/completions``) with
    ``image_url`` containing a base64 data URL. Works with any provider that
    supports the OpenAI vision format:

    - OpenAI: ``gpt-4o``, ``gpt-4-turbo``
    - 智谱 ZAI: ``glm-4v``, ``glm-4o``
    - Qwen: ``qwen-vl-max``, ``qwen-vl-plus``
    - Ollama: ``gemma4:12b`` (via ``/v1`` endpoint)
    - DeepSeek, Moonshot, etc.

    Args:
        image_path: Path to the PNG image file.
        prompt: Text prompt for the VLM.
        api_base: API base URL including version path
                  (e.g. ``https://api.openai.com/v1``, ``http://localhost:11434/v1``).
        model: Model name.
        api_key: API key (empty string for local servers like Ollama).
        timeout: Request timeout in seconds.
        max_tokens: Max tokens in the response.

    Returns:
        The VLM's text response.

    Raises:
        Exception: If the API call fails (network error, auth error, etc.).
    """
    from openai import OpenAI

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(base_url=api_base, api_key=api_key or "empty", timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def query_ollama(
    image_path: str,
    prompt: str,
    model: str = "gemma4:12b",
    host: str = "localhost",
    port: int = 11434,
    timeout: int = 120,
) -> str:
    """Backward-compatible wrapper — delegates to :func:`query_vlm`.

    Converts Ollama host/port to the OpenAI-compatible ``/v1`` endpoint.
    """
    return query_vlm(
        image_path,
        prompt,
        api_base=f"http://{host}:{port}/v1",
        model=model,
        api_key="",
        timeout=timeout,
    )


def _build_prompt(element_class: str, vlm_hint: str = "") -> str:
    """Build a structured VLM verification prompt.

    Args:
        element_class: Element name (e.g. "door").
        vlm_hint: Extra context to help the VLM (e.g. "a door with frame and handle").
    """
    hint_clause = f" ({vlm_hint})" if vlm_hint else ""
    return (
        f"This image is rendered from inside a room. "
        f"Is there a {element_class.upper()}{hint_clause} visible in this image? "
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
    vlm_api_base: str = "http://127.0.0.1:11434/v1",
    vlm_model: str = "gemma4:12b",
    vlm_api_key: str = "",
    vlm_timeout: int = 120,
    image_width: int = 800,
    image_height: int = 600,
    fov: float = 60.0,
    up_axis: int = 2,
    vlm_hint: str = "",
    skip_vlm: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[VerificationResult]:
    """Render targeted images for candidates and verify via OpenAI-compatible VLM.

    For each candidate:
      1. Compute camera pose from polar coordinates.
      2. Render a clean RGB image from 3DGS.
      3. Save image to ``output_dir``.
      4. Query VLM for confirmation.

    Args:
        candidates: List of :class:`Candidate` objects.
        scene: :class:`GSScene` with original weights loaded.
        scan_center: (cx, cy) room center in horizontal plane.
        floor_z: Floor level world coordinate (on the up-axis).
        output_dir: Directory to save rendered images.
        element_class: Element type for VLM prompt (e.g. "door").
        vlm_api_base: OpenAI-compatible API base URL (e.g. ``https://api.openai.com/v1``).
        vlm_model: VLM model name (e.g. ``gpt-4o``, ``glm-4v``, ``gemma4:12b``).
        vlm_api_key: API key (empty for local Ollama).
        vlm_timeout: Per-request timeout in seconds.
        up_axis: Which world axis is vertical (0=x, 1=y, 2=z).
        skip_vlm: If True, only render images without VLM queries.

    Returns:
        List of :class:`VerificationResult`.
    """
    from bim_recon.gs_scene import look_at_pose
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build up vector based on up_axis (consistent with VirtualScanner)
    up_vec = [0.0, 0.0, 0.0]
    up_vec[up_axis] = 1.0

    prompt = _build_prompt(element_class, vlm_hint)

    # Phase 1: Render all images sequentially (GPU is single-threaded)
    rendered: List[tuple] = []  # (index, candidate, img_path, eye, target, used_fov)
    for i, cand in enumerate(candidates):
        eye, target, used_fov = candidate_to_viewpoint(
            cand.world_x, cand.world_y,
            cand.h_min, cand.h_max,
            scan_center, floor_z, fov=fov,
            up_axis=up_axis,
        )
        pose = look_at_pose(
            (eye[0], eye[1], eye[2]),
            (target[0], target[1], target[2]),
            up=(up_vec[0], up_vec[1], up_vec[2]),
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
        rendered.append((i, cand, img_path, eye, target, used_fov))

    # Phase 2: Query VLM in parallel (HTTP I/O bound, no CUDA involvement)
    def _query_one(item):
        idx, cand, path = item[0], item[1], item[2]
        if skip_vlm:
            return idx, "", None
        try:
            vlm_text = query_vlm(
                path, prompt, vlm_api_base, vlm_model, vlm_api_key, vlm_timeout
            )
            confirmed, _ = _parse_vlm_response(vlm_text)
            return idx, vlm_text, confirmed
        except Exception as e:
            return idx, f"ERROR: {e}", None

    vlm_results: dict = {}
    if not skip_vlm and rendered:
        with ThreadPoolExecutor(max_workers=min(8, len(rendered))) as pool:
            futures = {pool.submit(_query_one, r): r[0] for r in rendered}
            for fut in futures:
                idx, vlm_text, confirmed = fut.result()
                vlm_results[idx] = (vlm_text, confirmed)

    # Phase 3: Assemble results in order
    results: List[VerificationResult] = []
    for i, cand, img_path, eye, target, used_fov in rendered:
        wall_tag = f"w{cand.wall_idx}" if cand.wall_idx is not None else "free"
        img_name = f"verify_{element_class}_{i}_{wall_tag}.png"
        vlm_text, confirmed = vlm_results.get(i, ("", None))

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
