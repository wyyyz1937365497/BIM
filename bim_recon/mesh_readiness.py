"""Mesh readiness check: ensure rendered images are suitable for TRELLIS 3D
reconstruction before proceeding to Falcon masking + TRELLIS generation.

Problem solved: The VLM verification image ("is this a chair?") is taken from
a fixed viewpoint (room center) that may not show the complete object or may
have a poor angle. TRELLIS needs a clear, complete, well-framed object image.

Strategy:
  1. Render from multiple angles around the candidate (front, ±30°)
  2. Ask VLM: "Is the complete object visible and well-framed for 3D creation?"
  3. Pick the best angle (or skip if none are suitable)

Usage (in pipeline)::

    from bim_recon.mesh_readiness import render_and_check_mesh_readiness

    result = render_and_check_mesh_readiness(
        scene, candidate_world_x, candidate_world_y,
        candidate_h_min, candidate_h_max,
        scan_center, floor_z, ceiling_z,
        element_class="furniture",
        vlm_api_base=..., vlm_model=..., vlm_api_key=...,
        output_dir=Path("output/trellis_meshes"),
    )
    if result.is_ready:
        clean_image = result.best_image_path  # send to Falcon → TRELLIS
    else:
        print(f"Skipped: {result.reason}")
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bim_recon.vlm_verifier import query_vlm


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MeshReadinessResult:
    """Result of the mesh readiness check for one candidate.

    Attributes:
        is_ready: True if at least one angle passed VLM readiness check.
        best_image_path: Path to the best rendered image (or None if not ready).
        best_angle: Horizontal angle offset of the best image (degrees).
        scores: Dict mapping angle (degrees) to VLM readiness verdict.
        reason: Why the candidate was rejected (if is_ready is False).
    """

    is_ready: bool
    best_image_path: Path | None
    best_angle: float
    scores: dict[float, str]
    reason: str


# ---------------------------------------------------------------------------
# Multi-angle rendering
# ---------------------------------------------------------------------------

def render_multi_angle(
    scene: Any,
    world_x: float,
    world_y: float,
    h_min: float,
    h_max: float,
    scan_center: tuple[float, float],
    floor_z: float,
    up_axis: int = 2,
    angles: list[float] | None = None,
    num_steps: int = 3,
    eye_distance: float | None = None,
    fov: float = 50.0,
    width: int = 800,
    height: int = 800,
    output_dir: Path | None = None,
    name_prefix: str = "candidate",
) -> list[tuple[float, Path]]:
    """Render a candidate from multiple horizontal angles.

    The camera orbits around the candidate at a fixed distance, at multiple
    horizontal offset angles. A square image is used so the object has room
    in all directions.

    Args:
        scene: GSScene instance.
        world_x, world_y: Candidate center in world horizontal plane (meters).
        h_min, h_max: Candidate height range above floor (meters).
        scan_center: (cx, cy) room center — used to determine primary direction.
        floor_z: Floor level Z coordinate.
        up_axis: Which axis is vertical (0=x, 1=y, 2=z).
        angles: List of angle offsets in degrees (e.g. [-30, 0, 30]).
                If None, generates ``num_steps`` angles evenly spaced.
        num_steps: Number of angles if ``angles`` is None.
        eye_distance: Distance from camera to candidate. If None, uses the
                      distance from scan_center to candidate.
        fov: Field of view (narrower than VLM's 60° for tighter framing).
        width, height: Image resolution (square by default for TRELLIS).
        output_dir: Directory to save rendered images.
        name_prefix: Filename prefix for saved images.

    Returns:
        List of (angle_degrees, image_path) tuples.
    """
    from bim_recon.gs_scene import look_at_pose

    if angles is None:
        if num_steps == 1:
            angles = [0.0]
        else:
            half_span = 30.0
            angles = list(np.linspace(-half_span, half_span, num_steps))

    # Compute primary viewing direction (from candidate toward room center)
    cx, cy = scan_center
    dx = cx - world_x
    dy = cy - world_y
    primary_angle = math.degrees(math.atan2(dy, dx))

    # Eye distance: distance from candidate to room center, or override
    if eye_distance is None:
        eye_distance = math.sqrt(dx * dx + dy * dy)
        # Clamp to reasonable range for good framing
        eye_distance = max(1.0, min(eye_distance, 5.0))

    h_mid = (h_min + h_max) / 2.0
    h_axes = [i for i in range(3) if i != up_axis]

    results: list[tuple[float, Path]] = []

    for angle_offset in angles:
        # Orbit around candidate: rotate primary direction by angle_offset
        total_angle = math.radians(primary_angle + angle_offset)
        eye_h_x = world_x + eye_distance * math.cos(total_angle)
        eye_h_y = world_y + eye_distance * math.sin(total_angle)

        eye = [0.0, 0.0, 0.0]
        eye[h_axes[0]] = eye_h_x
        eye[h_axes[1]] = eye_h_y
        eye[up_axis] = floor_z + h_mid  # camera at object mid-height

        target = [0.0, 0.0, 0.0]
        target[h_axes[0]] = world_x
        target[h_axes[1]] = world_y
        target[up_axis] = floor_z + h_mid

        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0

        pose = look_at_pose(
            (eye[0], eye[1], eye[2]),
            (target[0], target[1], target[2]),
            up=(up_vec[0], up_vec[1], up_vec[2]),
        )
        render_result = scene.render(pose, width=width, height=height, fov_degrees=fov)

        img = Image.fromarray(
            (render_result.colors * 255).clip(0, 255).astype(np.uint8)
        )

        if output_dir is not None:
            img_path = output_dir / f"{name_prefix}_angle{angle_offset:+.0f}.png"
            img.save(str(img_path))
            results.append((angle_offset, img_path))
        else:
            import io
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".png"))
            img.save(str(tmp))
            results.append((angle_offset, tmp))

    return results


# ---------------------------------------------------------------------------
# VLM readiness check
# ---------------------------------------------------------------------------

def _build_mesh_readiness_prompt(element_class: str) -> str:
    """Build a VLM prompt for checking 3D reconstruction suitability.

    Different from the verification prompt ("is this a door?"), this asks
    whether the image is good enough to create a 3D model from.
    """
    return (
        f"This image shows a {element_class.upper()} in a room. "
        f"I want to create a 3D model of this {element_class} from this image. "
        f"Please check:\n"
        f"1. Is the COMPLETE {element_class} visible (not cut off at edges)?\n"
        f"2. Is the viewing angle suitable for 3D reconstruction?\n"
        f"3. Is the {element_class} the main focus of the image?\n\n"
        f"First line: READY or NOT_READY\n"
        f"Second line: brief reason (e.g. 'complete object, good angle' or "
        f"'object partially cut off at right edge')"
    )


def _parse_mesh_readiness(response: str) -> tuple[bool, str]:
    """Parse VLM readiness response into (is_ready, reason).

    Looks for READY or NOT_READY on the first line.
    """
    lines = response.strip().split("\n")
    first_line = lines[0].upper().strip() if lines else ""

    if "READY" in first_line and "NOT" not in first_line:
        reason = lines[1].strip() if len(lines) > 1 else "approved"
        return True, reason

    if "NOT_READY" in first_line or "NOT READY" in first_line:
        reason = lines[1].strip() if len(lines) > 1 else "rejected"
        return False, reason

    # Ambiguous response — be conservative
    return False, f"ambiguous VLM response: {response[:50]}"


# ---------------------------------------------------------------------------
# Combined: render + check
# ---------------------------------------------------------------------------

def render_and_check_mesh_readiness(
    scene: Any,
    world_x: float,
    world_y: float,
    h_min: float,
    h_max: float,
    scan_center: tuple[float, float],
    floor_z: float,
    element_class: str,
    vlm_api_base: str,
    vlm_model: str,
    vlm_api_key: str,
    output_dir: Path,
    name_prefix: str = "candidate",
    up_axis: int = 2,
    angles: list[float] | None = None,
    ceiling_z: float | None = None,
    vlm_timeout: int = 120,
) -> MeshReadinessResult:
    """Render from multiple angles and check mesh readiness via VLM.

    Renders 3 angles by default (-30°, 0°, +30°), asks VLM for each,
    and returns the best ready image. If none pass, returns is_ready=False.

    Args:
        scene: GSScene instance for rendering.
        world_x, world_y: Candidate center in world coordinates.
        h_min, h_max: Candidate height range above floor.
        scan_center: (cx, cy) room center.
        floor_z: Floor level.
        element_class: Element name (e.g. "furniture") for VLM prompt.
        vlm_api_base, vlm_model, vlm_api_key: OpenAI-compatible VLM config.
        output_dir: Where to save rendered images.
        name_prefix: Filename prefix.
        up_axis: Vertical axis (0/1/2).
        angles: Custom angle list (degrees). None = [-30, 0, 30].
        ceiling_z: Used for eye height clamping (optional).

    Returns:
        MeshReadinessResult with best image path or rejection reason.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Render multiple angles
    renders = render_multi_angle(
        scene, world_x, world_y, h_min, h_max,
        scan_center, floor_z, up_axis=up_axis,
        angles=angles, fov=50.0,
        width=800, height=800,
        output_dir=output_dir,
        name_prefix=name_prefix,
    )

    # Step 2: VLM check each angle
    prompt = _build_mesh_readiness_prompt(element_class)
    scores: dict[float, str] = {}

    for angle, img_path in renders:
        try:
            vlm_text = query_vlm(
                str(img_path), prompt,
                api_base=vlm_api_base,
                model=vlm_model,
                api_key=vlm_api_key,
                timeout=vlm_timeout,
            )
            is_ready, reason = _parse_mesh_readiness(vlm_text)
            scores[angle] = f"{'READY' if is_ready else 'NOT_READY'}: {reason}"
        except Exception as e:
            scores[angle] = f"ERROR: {e}"

    # Step 3: Pick the first READY angle (prefer 0° = front)
    ready_angles = [
        (a, p) for a, p in renders
        if scores.get(a, "").startswith("READY")
    ]

    if not ready_angles:
        all_reasons = "; ".join(f"{a}°: {s}" for a, s in sorted(scores.items()))
        return MeshReadinessResult(
            is_ready=False,
            best_image_path=None,
            best_angle=0.0,
            scores=scores,
            reason=f"No angle passed readiness check. {all_reasons}",
        )

    # Prefer angle closest to 0 (front view)
    best_angle, best_path = min(ready_angles, key=lambda x: abs(x[0]))

    return MeshReadinessResult(
        is_ready=True,
        best_image_path=best_path,
        best_angle=best_angle,
        scores=scores,
        reason=f"best angle {best_angle}°: {scores[best_angle]}",
    )
