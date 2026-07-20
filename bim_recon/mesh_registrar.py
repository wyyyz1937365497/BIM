"""Mesh placement: transform TRELLIS-generated meshes into the 3DGS scene
coordinate system and register them in Revit via DirectShape.

Pipeline flow::

    1. TRELLIS generates a GLB from a VLM-verified candidate image
    2. This module loads the GLB mesh vertices/faces
    3. Computes the placement transform:
       a. TRELLIS output is in a normalized bounding box centered at origin
       b. The VLM image was rendered from a known camera (eye/target) in 3DGS coords
       c. We know the candidate's world position (world_x, world_y, floor_z)
       d. Scale: match the mesh's largest dimension to the candidate's physical size
       e. Translation: place the mesh centroid at the candidate's world position
       f. Rotation: align the mesh's view-facing axis with the camera viewing direction
    4. Send transformed vertices + faces to Revit via DirectShape C# script

Coordinate systems:
    - 3DGS scene: meters, origin at SfM/COLMAP arbitrary point
    - TRELLIS output: normalized [-1, 1] bounding box, origin at mesh center
    - Revit internal: feet (1 m = 3.28084 ft)

Usage::

    from bim_recon.mesh_registrar import MeshPlacement, compute_placement_transform

    placement = MeshPlacement(
        glb_path=Path("output/chair.glb"),
        world_x=-3.5, world_y=1.2,
        floor_z=-0.05, ceiling_z=2.5,
        element_width_m=0.6, element_height_m=0.8,
        up_axis=2,
        category="OST_GenericModel",
        name="Chair #0",
    )
    transform = compute_placement_transform(placement)
    result = register_mesh_in_revit(placement, transform)
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MeshPlacement:
    """All information needed to place a TRELLIS mesh in the 3DGS scene.

    Attributes:
        glb_path: Path to the generated GLB file.
        world_x, world_y: Candidate center in 3DGS horizontal plane (meters).
        floor_z: Floor level Z coordinate in 3DGS (meters).
        ceiling_z: Ceiling level Z coordinate in 3DGS (meters).
        element_width_m: Physical width of the object (meters), from detection.
        element_height_m: Physical height of the object (meters), from detection.
        up_axis: Which world axis is vertical (0=x, 1=y, 2=z).
        yaw_degrees: Yaw around the world up axis, in degrees, with positive
            values rotating the object **clockwise when viewed from above**
            (looking down the up axis). Corrects the constant offset between
            TRELLIS's image-space facing and the 3DGS scene convention; adjust
            per-placement if a future scene needs a different correction.
            Default 90.0.
        rotation_override: Optional full 3x3 world rotation. When provided it
            replaces the axis-remap plus yaw rotation used by the legacy path.
        translation_offset: World-space residual translation in meters.
        scale_multiplier: Positive residual multiplier applied after the
            detected physical-width scale.
        preserve_floor_contact: Keep the mesh base on ``floor_z`` after a
            rotation or scale override.
        category: Revit built-in category for DirectShape.
        name: Human-readable name for the DirectShape element.
    """

    glb_path: Path
    world_x: float
    world_y: float
    floor_z: float
    ceiling_z: float
    element_width_m: float
    element_height_m: float
    up_axis: int = 2
    yaw_degrees: float = 90.0
    rotation_override: tuple[tuple[float, float, float], ...] | None = None
    translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_multiplier: float = 1.0
    preserve_floor_contact: bool = True
    category: str = "OST_GenericModel"
    name: str = "B-class Mesh"


@dataclass(frozen=True, slots=True)
class MeshTransform:
    """Transform from TRELLIS normalized space to 3DGS world space.

    The transform is: world_point = scale * R @ mesh_point + translation
    where R is a 3x3 rotation matrix, scale is a uniform scalar.

    Attributes:
        scale: Uniform scale factor (meters per TRELLIS unit).
        rotation: 3x3 rotation matrix (row-major).
        translation: 3D offset in meters (3DGS world space).
        vertices_world: (N, 3) transformed vertices in meters.
        faces: (M, 3) triangle index array.
        mesh_extents: Per-axis extent of the raw TRELLIS-space mesh bounding
            box, as (x, y, z). Useful for telling which mesh axis is the
            "long" one before any rotation is applied.
        mesh_center: Centroid of the TRELLIS-space bounding box (x, y, z).
        principal_axis_mesh: Unit vector along the mesh's longest extent in
            TRELLIS space (PCA on centered vertices). For a long sofa this
            points down the long axis; for a symmetric cube it is arbitrary.
        principal_axis_world: The same direction after applying ``rotation``,
            in 3DGS world space. Compare against the scene's known long-axis
            direction to diagnose yaw errors.
        principal_axis_angle_deg: Horizontal-plane angle (degrees, CCW from
            the world's first horizontal axis) of ``principal_axis_world``.
            The single most diagnostic number for rotation issues.
    """

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    vertices_world: np.ndarray  # (N, 3) in meters
    faces: np.ndarray  # (M, 3) int32
    mesh_extents: tuple[float, float, float]
    mesh_center: tuple[float, float, float]
    principal_axis_mesh: tuple[float, float, float]
    principal_axis_world: tuple[float, float, float]
    principal_axis_angle_deg: float


# ---------------------------------------------------------------------------
# Object extraction: Falcon mask → clean RGBA image (transparent background)
# ---------------------------------------------------------------------------

def extract_object_from_render(
    rendered_image: Image.Image,
    falcon_detections: list[dict],
    padding: float = 0.05,
) -> Image.Image | None:
    """Extract a clean object image from a rendered scene using Falcon detections.

    Takes a full-scene render + Falcon's detection results, picks the largest
    detection, crops to its mask bbox (with padding), and makes the background
    transparent. The result is a clean RGBA image suitable for TRELLIS.

    Args:
        rendered_image: Full-scene RGB render (PIL Image).
        falcon_detections: List of detection dicts from FalconClient.segment().
            Each must have ``mask_bbox`` with ``{x, y, w, h}`` normalized [0,1],
            or at minimum ``bbox``.
        padding: Extra padding around the crop as fraction of image size.

    Returns:
        RGBA PIL Image with transparent background, cropped to the object.
        None if no valid detection.
    """
    if not falcon_detections:
        return None

    # Pick the detection with the largest mask area (most likely the main object)
    best = max(
        falcon_detections,
        key=lambda d: d.get("mask_area_ratio", 0) or 0,
    )

    # Use mask_bbox if available (tighter), else fall back to detection bbox
    bbox = best.get("mask_bbox") or best.get("bbox")
    if not bbox:
        return None

    w_img, h_img = rendered_image.size

    # Convert normalized bbox to pixel coordinates with padding
    x0 = max(0, int((bbox["x"] - bbox["w"] / 2 - padding) * w_img))
    y0 = max(0, int((bbox["y"] - bbox["h"] / 2 - padding) * h_img))
    x1 = min(w_img, int((bbox["x"] + bbox["w"] / 2 + padding) * w_img))
    y1 = min(h_img, int((bbox["y"] + bbox["h"] / 2 + padding) * h_img))

    if x1 <= x0 or y1 <= y0:
        return None

    # Crop to the object region
    cropped = rendered_image.crop((x0, y0, x1, y1))

    # Create alpha channel: use the bbox region as opaque, edges fade out.
    # Since Falcon only gives bbox (not pixel-level mask), we use a simple
    # center-weighted alpha that makes corners semi-transparent.
    # For production, request task="segmentation" from Falcon to get pixel masks.
    rgba = cropped.convert("RGBA")
    alpha = _create_center_weighted_alpha(rgba.size)
    rgba.putalpha(alpha)

    return rgba


def extract_object_from_segmentation(
    rendered_image: Image.Image,
    mask_rle: dict | None = None,
    mask_bbox: dict | None = None,
    padding: float = 0.05,
) -> Image.Image | None:
    """Extract object using pixel-level segmentation mask (if available).

    When Falcon returns a segmentation mask (not just bbox), this produces
    a pixel-accurate alpha channel.

    Args:
        rendered_image: Full-scene render.
        mask_rle: RLE-encoded mask from Falcon (if available). Not currently
            passed by the HTTP client, but supported for future use.
        mask_bbox: Normalized bbox {x, y, w, h} from Falcon mask_bbox.
        padding: Extra padding fraction.

    Returns:
        RGBA image with transparent background, or None.
    """
    if mask_bbox is None:
        return None

    w_img, h_img = rendered_image.size
    x0 = max(0, int((mask_bbox["x"] - mask_bbox["w"] / 2 - padding) * w_img))
    y0 = max(0, int((mask_bbox["y"] - mask_bbox["h"] / 2 - padding) * h_img))
    x1 = min(w_img, int((mask_bbox["x"] + mask_bbox["w"] / 2 + padding) * w_img))
    y1 = min(h_img, int((mask_bbox["y"] + mask_bbox["h"] / 2 + padding) * h_img))

    if x1 <= x0 or y1 <= y0:
        return None

    cropped = rendered_image.crop((x0, y0, x1, y1))
    rgba = cropped.convert("RGBA")

    # If we have RLE mask, apply it directly
    if mask_rle is not None:
        try:
            from pycocotools import mask as mask_utils
            counts = mask_rle["counts"]
            if isinstance(counts, str):
                counts = counts.encode("utf-8")
            mask_arr = mask_utils.decode({
                "counts": counts,
                "size": mask_rle["size"],
            })
            # Crop the mask to the same region
            mask_crop = mask_arr[y0:y1, x0:x1]
            alpha = Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")
            rgba.putalpha(alpha)
            return rgba
        except ImportError:
            pass

    # Fall back to center-weighted alpha
    alpha = _create_center_weighted_alpha(rgba.size)
    rgba.putalpha(alpha)
    return rgba


def _create_center_weighted_alpha(size: tuple[int, int]) -> Image.Image:
    """Create a center-weighted alpha channel (opaque center, faded edges).

    This is a simple approximation when pixel-level masks aren't available.
    Uses a radial gradient: alpha = 255 at center, fading to ~50 at corners.
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    # Distance from center, normalized to [0, 1]
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    # Alpha: 255 at center, linearly fading to 80 at corners
    alpha = np.clip(255 - dist * 175, 80, 255).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


# ---------------------------------------------------------------------------
# GLB parsing (minimal — extracts mesh vertices + faces from binary glTF)
# ---------------------------------------------------------------------------

def parse_glb_vertices_faces(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract vertices (N,3) and faces (M,3) from a GLB file.

    GLB = binary glTF. Structure:
      - 12-byte header (magic, version, length)
      - JSON chunk (scene structure)
      - BIN chunk (vertex/face data)

    This is a minimal parser that handles the common case:
      - Single mesh, single primitive
      - POSITION accessor (VEC3 float32)
      - INDICES accessor (SCALAR uint16/uint32)

    For complex GLBs (multiple meshes, morph targets, etc.), use trimesh.
    """
    try:
        import trimesh
        mesh = trimesh.load(str(glb_path), force='mesh')
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces') and len(mesh.vertices) > 0:
            return (
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32),
            )
    except Exception:
        pass

    # Fallback: minimal binary glTF parser
    return _parse_glb_binary(glb_path)


def _parse_glb_binary(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal GLB binary parser for single-mesh files."""
    with open(glb_path, 'rb') as f:
        data = f.read()

    # GLB header: magic(4) + version(4) + length(4) = 12 bytes
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError(f"Not a GLB file: magic=0x{magic:08X}")

    offset = 12
    json_data = None
    bin_data = None

    while offset < length:
        chunk_length, chunk_type = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk_body = data[offset:offset + chunk_length]
        offset += chunk_length

        if chunk_type == 0x4E4F534A:  # 'JSON'
            json_data = json.loads(chunk_body.decode('utf-8'))
        elif chunk_type == 0x004E4942:  # 'BIN\0'
            bin_data = chunk_body

    if json_data is None or bin_data is None:
        raise ValueError("GLB missing JSON or BIN chunk")

    # Find mesh → primitives → attributes.POSITION and indices
    meshes = json_data.get('meshes', [])
    if not meshes:
        raise ValueError("GLB has no meshes")

    primitives = meshes[0].get('primitives', [])
    if not primitives:
        raise ValueError("GLB mesh has no primitives")

    prim = primitives[0]
    accessors = json_data.get('accessors', [])
    buffer_views = json_data.get('bufferViews', [])

    def _read_accessor(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = buffer_views[acc['bufferView']]
        offset_in_view = acc.get('byteOffset', 0)
        start = bv['byteOffset'] + offset_in_view
        comp_type = acc['componentType']
        count = acc['count']
        type_str = acc['type']

        if type_str == 'VEC3' and comp_type == 5126:  # FLOAT
            arr = np.frombuffer(bin_data, dtype=np.float32, count=count * 3, offset=start)
            return arr.reshape(-1, 3)
        elif type_str == 'SCALAR' and comp_type == 5123:  # UNSIGNED_SHORT
            return np.frombuffer(bin_data, dtype=np.uint16, count=count, offset=start).astype(np.int32)
        elif type_str == 'SCALAR' and comp_type == 5125:  # UNSIGNED_INT
            return np.frombuffer(bin_data, dtype=np.uint32, count=count, offset=start).astype(np.int32)
        else:
            raise ValueError(f"Unsupported accessor: type={type_str}, compType={comp_type}")

    # POSITION
    pos_idx = prim['attributes']['POSITION']
    vertices = _read_accessor(pos_idx)

    # INDICES (optional — if absent, use sequential indices)
    if 'indices' in prim:
        faces = _read_accessor(prim['indices']).reshape(-1, 3)
    else:
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)

    return vertices.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Transform computation
# ---------------------------------------------------------------------------

def compute_placement_transform(placement: MeshPlacement) -> MeshTransform:
    """Compute the transform from TRELLIS normalized space to 3DGS world space.

    Strategy:
      1. Load mesh vertices from GLB
      2. Compute mesh bounding box in TRELLIS space
      3. Scale: fit the mesh so its largest horizontal dimension matches
         the detected element width (element_width_m)
      4. Translation: place the mesh centroid at (world_x, world_y, floor_z)
         on the up_axis
      5. Rotation: axis-remap (TRELLIS Y-up → 3DGS up_axis), then yaw around
         the world up axis by ``placement.yaw_degrees`` (positive = clockwise
         from above) to correct the image-space → scene facing offset.
      6. Diagnostics: PCA on centered vertices yields the mesh's principal
         axis (longest extent); we report it in both mesh and world space so
         rotation mismatches can be traced post-hoc.
    """
    vertices, faces = parse_glb_vertices_faces(placement.glb_path)

    # Mesh bounding box in TRELLIS space
    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2.0
    mesh_extents = mesh_max - mesh_min

    # Determine which mesh axes are horizontal vs vertical
    # TRELLIS convention: Y-up (standard for image-to-3D models)
    # We need to remap to the 3DGS up_axis
    h_axes_mesh = [i for i in range(3) if i != 1]  # mesh X and Z are horizontal
    v_axis_mesh = 1  # mesh Y is vertical

    # Scale: fit mesh to detected physical size
    # Use the larger of (mesh horizontal extent) as reference
    mesh_h_extent = max(mesh_extents[h_axes_mesh[0]], mesh_extents[h_axes_mesh[1]])
    if mesh_h_extent < 1e-6:
        mesh_h_extent = 1.0  # avoid div-by-zero for degenerate meshes

    scale = placement.element_width_m / mesh_h_extent

    # Also clamp mesh height to ceiling
    mesh_v_extent = mesh_extents[v_axis_mesh]
    scaled_height = mesh_v_extent * scale
    room_height = placement.ceiling_z - placement.floor_z
    if room_height > 0 and scaled_height > room_height:
        scale = scale * (room_height / scaled_height)

    # Build rotation matrix: axis-remap plus legacy yaw, unless a learned
    # full rotation override is supplied.
    if placement.rotation_override is not None:
        rotation = np.asarray(placement.rotation_override, dtype=np.float32)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_override must be a 3x3 matrix")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_override must contain finite values")
        if abs(float(np.linalg.det(rotation))) < 1e-6:
            raise ValueError("rotation_override must be non-singular")
    else:
        axis_remap = _build_axis_remap_rotation(placement.up_axis)
        yaw = _build_yaw_rotation(placement.up_axis, placement.yaw_degrees)
        rotation = yaw @ axis_remap

    # Center the mesh at origin in TRELLIS space, then apply rotation + scale.
    centered = vertices - mesh_center
    rotated = centered @ rotation.T
    scale *= float(placement.scale_multiplier)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale_multiplier must produce a positive finite scale")
    scaled = rotated * scale

    # Translation: place the mesh centroid at candidate's world position.
    translation = np.zeros(3, dtype=np.float32)
    h_axes_world = [i for i in range(3) if i != placement.up_axis]
    translation[h_axes_world[0]] = placement.world_x
    translation[h_axes_world[1]] = placement.world_y
    if placement.preserve_floor_contact:
        # Rotation can change the vertical extent, so use the actual transformed
        # minimum rather than assuming the unrotated mesh height.
        vertical = scaled[:, placement.up_axis]
        translation[placement.up_axis] = placement.floor_z - float(vertical.min())
    else:
        mesh_height_scaled = mesh_extents[v_axis_mesh] * scale
        translation[placement.up_axis] = placement.floor_z + mesh_height_scaled / 2.0
    translation += np.asarray(placement.translation_offset, dtype=np.float32)

    if placement.preserve_floor_contact:
        # A learned vertical offset is allowed, but keep the minimum at the
        # requested floor after the residual horizontal translation is applied.
        translation[placement.up_axis] += placement.floor_z - float(
            (scaled + translation)[..., placement.up_axis].min()
        )

    vertices_world = scaled + translation

    # Diagnostics: principal axis via PCA. For a long object this is the long
    # axis; for a symmetric object it is arbitrary. We track it in both mesh
    # and world space so rotation bugs (e.g. the long axis ending up 45° off
    # from the scene's real long axis) can be diagnosed from logged data.
    principal_mesh = _principal_axis(centered)
    principal_world = rotation @ principal_mesh
    norm_pw = float(np.linalg.norm(principal_world))
    if norm_pw > 1e-9:
        principal_world = principal_world / norm_pw
    principal_angle = _horizontal_angle_deg(principal_world, placement.up_axis)

    return MeshTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        vertices_world=vertices_world.astype(np.float32),
        faces=faces,
        mesh_extents=(float(mesh_extents[0]), float(mesh_extents[1]), float(mesh_extents[2])),
        mesh_center=(float(mesh_center[0]), float(mesh_center[1]), float(mesh_center[2])),
        principal_axis_mesh=(float(principal_mesh[0]), float(principal_mesh[1]), float(principal_mesh[2])),
        principal_axis_world=(float(principal_world[0]), float(principal_world[1]), float(principal_world[2])),
        principal_axis_angle_deg=float(principal_angle),
    )


def _build_axis_remap_rotation(up_axis: int) -> np.ndarray:
    """Build rotation matrix to remap TRELLIS Y-up to 3DGS up_axis.

    TRELLIS convention: X=right, Y=up, Z=forward
    3DGS convention depends on up_axis:
      - up_axis=2 (Z-up): X→X, Y→Z, Z→(-Y)  [Y-up → Z-up rotation]
      - up_axis=1 (Y-up): identity (already Y-up)
      - up_axis=0 (X-up): X→Y, Y→X, Z→(-Z)   [rare]
    """
    if up_axis == 1:
        return np.eye(3, dtype=np.float32)
    elif up_axis == 2:
        # Y-up → Z-up: rotate -90° around X axis
        # new_X = old_X, new_Y = old_Z, new_Z = -old_Y... wait
        # Actually: world_X = mesh_X, world_Y = -mesh_Z, world_Z = mesh_Y
        # Hmm, let's think carefully.
        # TRELLIS: mesh(X=right, Y=up, Z=toward viewer)
        # 3DGS Z-up: world(X=east, Y=north, Z=up)
        # We want: world_X = mesh_X, world_Z = mesh_Y, world_Y = -mesh_Z
        # So: [1, 0, 0]  maps mesh→world for X
        #     [0, 0, -1] maps mesh→world for Y (mesh Y becomes world Z via the third row)
        # Let's build this as a matrix where world = R @ mesh:
        # R = [[1, 0, 0],
        #      [0, 0, -1],
        #      [0, 1, 0]]
        # Check: R @ [0,1,0] = [0,0,1] → mesh Y-up becomes world Z-up ✓
        # Check: R @ [0,0,1] = [0,-1,0] → mesh Z-forward becomes world -Y ✓
        return np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
    elif up_axis == 0:
        # X-up: mesh Y → world X (up), mesh Z → world Y, mesh X → world Z
        # det = 1 (proper rotation)
        return np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32)
    else:
        raise ValueError(f"Invalid up_axis: {up_axis}")


def _build_yaw_rotation(up_axis: int, yaw_degrees: float) -> np.ndarray:
    """Build a yaw rotation matrix around the world up axis.

    Positive ``yaw_degrees`` rotates the object **clockwise when viewed from
    above** (i.e. looking down the world up axis toward the scene). This is
    the intuitive top-down-map convention: 90° sends +X (east) → -Y (south)
    for Z-up scenes.

    Internally that maps to a negative angle around the right-hand-rule
    +up axis, since positive RH rotation appears counter-clockwise from
    above.
    """
    angle = -np.radians(yaw_degrees)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if up_axis == 2:  # Z-up: rotate around Z
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ], dtype=np.float32)
    elif up_axis == 1:  # Y-up: rotate around Y
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float32)
    elif up_axis == 0:  # X-up: rotate around X
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c],
        ], dtype=np.float32)
    else:
        raise ValueError(f"Invalid up_axis: {up_axis}")


# ---------------------------------------------------------------------------
# Placement diagnostics (mesh PCA + transform summary for logging)
# ---------------------------------------------------------------------------

def _principal_axis(vertices_centered: np.ndarray) -> np.ndarray:
    """Return the unit vector along the mesh's longest extent (PCA).

    Computes the 3D covariance of ``vertices_centered`` and returns the
    eigenvector with the largest eigenvalue. Sign is arbitrary (PCA axes
    are oriented up to ±1); callers only care about the line direction.

    For <2 vertices or a degenerate (zero covariance) cloud, falls back to
    +X so downstream math remains well-defined.
    """
    if vertices_centered.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cov = np.cov(vertices_centered, rowvar=False)
    if cov.shape != (3, 3) or not np.any(np.abs(cov) > 1e-12):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    try:
        _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    principal = eigvecs[:, -1]
    norm = float(np.linalg.norm(principal))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (principal / norm).astype(np.float32)


def _horizontal_angle_deg(axis_world: np.ndarray, up_axis: int) -> float:
    """Angle of ``axis_world`` in the horizontal plane, in degrees.

    Returns ``atan2(second_h, first_h)`` where ``first_h``/``second_h`` are
    the two non-up components, in degrees, in the range (-180, 180]. This is
    the compass-like direction the axis points in the floor plan.
    """
    h_axes = [i for i in range(3) if i != up_axis]
    x = float(axis_world[h_axes[0]])
    y = float(axis_world[h_axes[1]])
    return float(np.degrees(np.arctan2(y, x)))


def serialize_placement_diagnostics(
    placement: MeshPlacement,
    transform: MeshTransform,
) -> dict[str, Any]:
    """Build a JSON-safe diagnostic dict describing one placement.

    Captures every value that influences the final Revit DirectShape pose:
    the raw placement inputs, the mesh's TRELLIS-space extents and principal
    axis, and the resulting world-space transform including the world-space
    principal axis. Designed to be merged into workflow manifests or Gradio
    output JSON for post-hoc debugging of rotation/position issues.
    """
    return {
        "placement_input": {
            "world_x": placement.world_x,
            "world_y": placement.world_y,
            "floor_z": placement.floor_z,
            "ceiling_z": placement.ceiling_z,
            "element_width_m": placement.element_width_m,
            "element_height_m": placement.element_height_m,
            "up_axis": placement.up_axis,
            "yaw_degrees": placement.yaw_degrees,
            "name": placement.name,
            "category": placement.category,
        },
        "mesh_analysis_trellis_space": {
            "extents_x_y_z": list(transform.mesh_extents),
            "centroid_x_y_z": list(transform.mesh_center),
            "principal_axis_x_y_z": list(transform.principal_axis_mesh),
            "note": "extents/axis in TRELLIS convention (Y is up)",
        },
        "transform_output": {
            "scale": float(transform.scale),
            "rotation_matrix_row_major": [
                float(v) for row in transform.rotation for v in row
            ],
            "translation_x_y_z": [float(v) for v in transform.translation],
            "principal_axis_world_x_y_z": list(transform.principal_axis_world),
            "principal_axis_world_horizontal_angle_deg": (
                transform.principal_axis_angle_deg
            ),
            "note": (
                "principal_axis_world_horizontal_angle_deg is the floor-plan "
                "direction of the mesh's longest extent after transform; "
                "compare to the scene's actual long-axis direction to find "
                "yaw corrections."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Auto-yaw via silhouette matching (render-and-compare)
# ---------------------------------------------------------------------------

def find_best_yaw_silhouette(
    glb_path: Path,
    cutout_alpha: np.ndarray,
    norm_bbox: dict,
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
    camera_up_axis: int,
    camera_fov: float,
    camera_img_w: int,
    camera_img_h: int,
    world_pos: tuple[float, float, float],
    element_width_m: float,
    *,
    up_axis: int = 2,
    cutout_padding: float = 0.08,
    coarse_step: float = 10.0,
    fine_step: float = 1.0,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """Find the yaw that best aligns the mesh with the original cutout image.

    Strategy (analysis by synthesis):
      For each candidate yaw, project the mesh to the original camera's image
      plane, build a silhouette in the cutout's coordinate frame, and compute
      IoU with the cutout's alpha mask. The yaw with the highest IoU wins.

    This is robust to TRELLIS's canonical-pose convention failures (e.g. cubic
    objects generated at an arbitrary angle) because it directly measures
      "does the reconstruction, placed at this yaw, look like what we saw?"

    Args:
        glb_path: TRELLIS-generated GLB file.
        cutout_alpha: (H, W) mask from the cutout's alpha channel. Any non-zero
            pixel counts as occupied.
        norm_bbox: Falcon bbox in normalized image coords. Must contain
            ``{"x", "y", "w", "h"}`` where x/y are the CENTER (not corner),
            matching the convention used by ``backproject_mask_centre`` and
            the Gradio cutout pipeline.
        camera_eye, camera_target: World-space camera pose used to capture the
            original rendering. The mesh is projected through this camera.
        camera_up_axis: World up axis at capture time (usually 2 for Z-up).
        camera_fov: Camera vertical FOV in degrees.
        camera_img_w, camera_img_h: Dimensions of the original rendering. Must
            match the actual image used for Falcon segmentation; non-square
            images are handled correctly (focal derived from the vertical FOV
            and ``camera_img_h``, x-centering uses ``camera_img_w``).
        world_pos: (x, y, z) world position where the mesh should be placed
            for projection. Use the depth-backprojected mask centre.
        element_width_m: Physical width for scale matching (same value used
            in :class:`MeshPlacement`).
        up_axis: World up axis used by the placement transform.
        cutout_padding: Fraction of bbox size used as padding when the cutout
            was cropped (default 0.08 = 8%). Used to reconstruct the crop
            region in the full image.
        coarse_step, fine_step: Two-stage search granularity in degrees.

    Returns:
        Dict with:
          ``best_yaw``: the recovered yaw_degrees.
          ``best_iou``: IoU at the best yaw (0-1; >0.3 is typically a confident
            match, <0.15 suggests the silhouette geometry is ambiguous).
          ``coarse_scores``: list of {yaw, iou} for the coarse scan (debugging).
          ``method``: ``"silhouette_iou"``.
    """
    vertices, faces = parse_glb_vertices_faces(glb_path)

    # Densify: sparse meshes (e.g. test cubes with 8 verts) don't produce
    # usable silhouettes from vertex projection alone. Sample surface points
    # when the vertex count is too low.
    if len(vertices) < 2000 and len(faces) > 0:
        vertices = _sample_surface_points(vertices, faces, target_count=5000)

    centered = vertices - vertices.mean(axis=0)

    # Scale: match element_width_m to the mesh's largest horizontal extent
    # (TRELLIS Y-up → mesh horizontal axes are X and Z)
    mesh_h_extent = max(centered[:, 0].ptp(), centered[:, 2].ptp())
    if mesh_h_extent < 1e-6:
        mesh_h_extent = 1.0
    scale = element_width_m / mesh_h_extent

    # Camera frame (same convention as render_element_front_view)
    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[camera_up_axis] = 1.0
    eye = np.array(camera_eye, dtype=np.float64)
    target = np.array(camera_target, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up_vec)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)
    # Focal length: FOV is vertical → derive from img_h. Square pixels →
    # focal_x == focal_y. x-centering uses img_w, y-centering uses img_h.
    focal_y = 0.5 * camera_img_h / np.tan(np.radians(camera_fov) / 2.0)
    focal_x = focal_y  # square pixels
    center_x = camera_img_w / 2.0
    center_y = camera_img_h / 2.0

    # Reconstruct the crop region in full-image pixel coords.
    # norm_bbox uses center-x/center-y convention; padding expands each side.
    # x-coordinates scale by img_w, y-coordinates by img_h (non-square aware).
    bx_c = norm_bbox.get("x", 0.5) * camera_img_w
    by_c = norm_bbox.get("y", 0.5) * camera_img_h
    bw = max(norm_bbox.get("w", 0.2) * camera_img_w, 4.0)
    bh = max(norm_bbox.get("h", 0.2) * camera_img_h, 4.0)
    pad_x = bw * cutout_padding
    pad_y = bh * cutout_padding
    crop_x0 = bx_c - bw / 2.0 - pad_x
    crop_y0 = by_c - bh / 2.0 - pad_y
    crop_w = max(int(bw + 2 * pad_x), 4)
    crop_h = max(int(bh + 2 * pad_y), 4)

    # Resize cutout_alpha to match the reconstructed crop dimensions
    from PIL import Image as _PILImage
    alpha_img = _PILImage.fromarray(
        (cutout_alpha > 0).astype(np.uint8) * 255, mode="L",
    ).resize((crop_w, crop_h), _PILImage.BILINEAR)
    cutout_mask = np.array(alpha_img, dtype=bool)

    axis_remap = _build_axis_remap_rotation(up_axis)
    world_pos_arr = np.array(world_pos, dtype=np.float64)

    def silhouette_at_yaw(yaw_deg: float) -> np.ndarray:
        """Project the yawed mesh to the cutout frame and build a binary mask."""
        yaw = np.radians(yaw_deg)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rot = centered.copy()
        rot[:, 0] = c * centered[:, 0] + s * centered[:, 2]
        rot[:, 2] = -s * centered[:, 0] + c * centered[:, 2]
        world = (rot @ axis_remap.T) * scale + world_pos_arr

        rel = world - eye
        z_cam = rel @ fwd
        x_cam = rel @ right
        y_cam = rel @ down
        valid = z_cam > 0.1
        if not np.any(valid):
            return np.zeros((crop_h, crop_w), dtype=bool)

        px_full = (x_cam[valid] / z_cam[valid]) * focal_x + center_x
        py_full = (y_cam[valid] / z_cam[valid]) * focal_y + center_y

        # Map from full-image coords to crop-local coords
        px_crop = px_full - crop_x0
        py_crop = py_full - crop_y0

        in_bounds = (
            (px_crop >= 0) & (px_crop < crop_w) &
            (py_crop >= 0) & (py_crop < crop_h)
        )
        mask = np.zeros((crop_h, crop_w), dtype=bool)
        if np.any(in_bounds):
            mask[py_crop[in_bounds].astype(int), px_crop[in_bounds].astype(int)] = True

        # Light dilation to fill gaps between projected vertices
        dilated = mask.copy()
        dilated[:-1, :] |= mask[1:, :]
        dilated[1:, :] |= mask[:-1, :]
        dilated[:, :-1] |= mask[:, 1:]
        dilated[:, 1:] |= mask[:, :-1]
        return dilated

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / max(union, 1)

    # Stage 1: coarse scan over the full circle
    coarse_yaws = np.arange(0.0, 360.0, coarse_step)
    coarse_scores = [
        (float(y), _iou(cutout_mask, silhouette_at_yaw(float(y))))
        for y in coarse_yaws
    ]
    coarse_best = max(coarse_scores, key=lambda t: t[1])

    # Stage 2: fine scan around the coarse winner
    fine_yaws = np.arange(
        coarse_best[0] - coarse_step, coarse_best[0] + coarse_step + fine_step, fine_step,
    )
    fine_scores = [
        (float(y), _iou(cutout_mask, silhouette_at_yaw(float(y))))
        for y in fine_yaws
    ]
    best_yaw, best_iou = max(fine_scores, key=lambda t: t[1])

    if debug_dir is not None:
        _save_yaw_debug_visualization(
            debug_dir=debug_dir,
            cutout_mask=cutout_mask,
            silhouette_at_yaw=silhouette_at_yaw,
            best_yaw=best_yaw,
            coarse_scores=coarse_scores,
            crop_w=crop_w, crop_h=crop_h,
            crop_x0=crop_x0, crop_y0=crop_y0,
            focal_x=focal_x, focal_y=focal_y,
            center_x=center_x, center_y=center_y,
            camera_img_w=camera_img_w, camera_img_h=camera_img_h,
            norm_bbox=norm_bbox,
        )

    return {
        "best_yaw": float(best_yaw),
        "best_iou": float(best_iou),
        "coarse_scores": [{"yaw": y, "iou": i} for y, i in coarse_scores],
        "method": "silhouette_iou",
    }


def _save_yaw_debug_visualization(
    *,
    debug_dir: Path,
    cutout_mask: np.ndarray,
    silhouette_at_yaw,
    best_yaw: float,
    coarse_scores: list,
    crop_w: int, crop_h: int,
    crop_x0: float, crop_y0: float,
    focal_x: float, focal_y: float,
    center_x: float, center_y: float,
    camera_img_w: int, camera_img_h: int,
    norm_bbox: dict,
) -> None:
    """Save debug images for silhouette-based yaw matching.

    Writes to ``debug_dir``:
      ``cutout_mask.png``      — the Falcon alpha mask (binary, in crop frame)
      ``projected_best.png``   — projected mesh silhouette at the best yaw
      ``overlay_best.png``     — 3-color overlay: green=both, red=cutout only, blue=projected only
      ``projection_geometry.json`` — crop/focal/bbox numbers for sanity-checking
    """
    from PIL import Image as _PIL
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cutout mask (grayscale PNG)
    _PIL.fromarray((cutout_mask.astype(np.uint8)) * 255, mode="L").save(debug_dir / "cutout_mask.png")

    # 2. Projected silhouette at the best yaw
    best_sil = silhouette_at_yaw(best_yaw)
    _PIL.fromarray((best_sil.astype(np.uint8)) * 255, mode="L").save(debug_dir / "projected_best.png")

    # 3. Overlay: 3-color (RGB)
    h, w = cutout_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    both = cutout_mask & best_sil
    cutout_only = cutout_mask & ~best_sil
    proj_only = ~cutout_mask & best_sil
    overlay[both] = [0, 255, 0]        # green = overlap
    overlay[cutout_only] = [255, 0, 0]  # red = cutout but not projected
    overlay[proj_only] = [0, 0, 255]    # blue = projected but not cutout
    _PIL.fromarray(overlay, mode="RGB").save(debug_dir / "overlay_best.png")

    # 4. Projection geometry for sanity-checking
    import json as _json
    geom = {
        "camera_img_w": camera_img_w,
        "camera_img_h": camera_img_h,
        "focal_x": focal_x,
        "focal_y": focal_y,
        "center_x": center_x,
        "center_y": center_y,
        "norm_bbox": norm_bbox,
        "crop": {
            "x0": crop_x0, "y0": crop_y0,
            "w": crop_w, "h": crop_h,
        },
        "cutout_mask": {
            "shape": [int(cutout_mask.shape[0]), int(cutout_mask.shape[1])],
            "filled_pixels": int(cutout_mask.sum()),
        },
        "best_projected_silhouette": {
            "filled_pixels": int(best_sil.sum()),
        },
        "best_yaw": best_yaw,
        "coarse_top5": sorted(coarse_scores, key=lambda t: -t[1])[:5],
    }
    (debug_dir / "projection_geometry.json").write_text(
        _json.dumps(geom, indent=2), encoding="utf-8",
    )



def _sample_surface_points(
    vertices: np.ndarray, faces: np.ndarray, *, target_count: int = 5000,
) -> np.ndarray:
    """Uniformly sample points from the mesh surface (area-weighted).

    Used to densify sparse meshes so silhouette projection produces a
    recognizable shape rather than a handful of isolated pixels.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if total < 1e-12:
        return vertices
    probs = areas / total
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    tri_idx = rng.choice(len(faces), size=target_count, p=probs)
    r1 = rng.uniform(0.0, 1.0, target_count)
    r2 = rng.uniform(0.0, 1.0, target_count)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    return (u[:, None] * v0[tri_idx] +
            v[:, None] * v1[tri_idx] +
            w[:, None] * v2[tri_idx]).astype(np.float32)

# ---------------------------------------------------------------------------
# Revit registration (DirectShape via C# script)
# ---------------------------------------------------------------------------

def register_mesh_in_revit(
    placement: MeshPlacement,
    transform: MeshTransform,
) -> dict[str, Any]:
    """Build the DirectShape mesh payload and persist it for the compiled tool.

    The compiled ``create_directshape_from_mesh`` MCP tool is invoked by the
    caller via the MCP gateway (file-path mode). This helper only writes the
    Revit-ready payload to a temp file and returns its path so the caller can
    dispatch ``{"meshFile": payload_path}`` in any environment.

    Args:
        placement: The placement specification (name, category, etc.).
        transform: The computed mesh transform (vertices in meters, faces).

    Returns:
        Dict with keys: ``status`` ("formatted"), ``payload_path``,
        ``vertex_count``, ``face_count``, ``script_name``.
    """
    meters_to_feet = 3.280839895013123

    # Convert vertices from meters to feet for Revit internal units
    vertices_feet = (transform.vertices_world * meters_to_feet).round(6)
    verts_flat = vertices_feet.flatten().tolist()
    faces_flat = transform.faces.flatten().tolist()

    payload = {
        "name": placement.name,
        "category": placement.category,
        "vertices": verts_flat,
        "faces": faces_flat,
    }

    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w",
    ) as tmp:
        json.dump(payload, tmp)
        payload_path = tmp.name

    return {
        "status": "formatted",
        "payload_path": payload_path,
        "vertex_count": len(vertices_feet),
        "face_count": len(transform.faces),
        "script_name": "create_directshape_from_mesh",
    }
