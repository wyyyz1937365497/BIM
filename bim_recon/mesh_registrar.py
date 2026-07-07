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

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    """

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    vertices_world: np.ndarray  # (N, 3) in meters
    faces: np.ndarray  # (M, 3) int32


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
      5. Rotation: identity (TRELLIS front faces +Z by convention;
         camera alignment is handled by the rendering step)
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

    # Build rotation matrix: remap TRELLIS (X-right, Y-up, Z-forward) to 3DGS
    # For 3DGS up_axis=2 (Z-up): mesh X→world X, mesh Y→world Z, mesh Z→world Y(neg)
    # For 3DGS up_axis=1 (Y-up): mesh X→world X, mesh Y→world Y, mesh Z→world Z(neg)
    # For 3DGS up_axis=0 (X-up): mesh X→world Y, mesh Y→world X, mesh Z→world Z(neg)
    #
    # We use a simple axis-swap rotation (no arbitrary rotation needed since
    # the camera-facing direction was handled by the VLM image capture).
    rotation = _build_axis_remap_rotation(placement.up_axis)

    # Center the mesh at origin in TRELLIS space, then apply rotation + scale
    centered = vertices - mesh_center
    rotated = centered @ rotation.T
    scaled = rotated * scale

    # Translation: place centroid at candidate's world position on the floor
    translation = np.zeros(3, dtype=np.float32)
    h_axes_world = [i for i in range(3) if i != placement.up_axis]
    translation[h_axes_world[0]] = placement.world_x
    translation[h_axes_world[1]] = placement.world_y
    # Place mesh base at floor level (shift up by half the scaled height)
    mesh_height_scaled = mesh_extents[v_axis_mesh] * scale
    translation[placement.up_axis] = placement.floor_z + mesh_height_scaled / 2.0

    vertices_world = scaled + translation

    return MeshTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        vertices_world=vertices_world.astype(np.float32),
        faces=faces,
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


# ---------------------------------------------------------------------------
# Revit registration (DirectShape via C# script)
# ---------------------------------------------------------------------------

def register_mesh_in_revit(
    placement: MeshPlacement,
    transform: MeshTransform,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Send the transformed mesh to Revit as a DirectShape via C# script.

    If *runner* is a :class:`~bim_recon.revit_runner.RevitScriptRunner` with an
    MCP sender configured, the mesh is immediately created in Revit.

    If *runner* is ``None`` (e.g. when Revit is not running or the pipeline is
    executed headless), the formatted payload and C# code are returned for
    later manual dispatch via ``send_code_to_revit``.

    Args:
        placement: The placement specification (name, category, etc.).
        transform: The computed mesh transform (vertices in meters, faces).
        runner: Optional ``RevitScriptRunner`` instance. If provided and has
            an MCP sender, the DirectShape is created immediately.

    Returns:
        Dict with keys: ``status``, ``vertex_count``, ``face_count``, and
        either ``element_id`` (if Revit was called) or ``payload_json`` +
        ``script_name`` (for manual dispatch).
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

    base_info = {
        "vertex_count": len(vertices_feet),
        "face_count": len(transform.faces),
    }

    if runner is not None:
        result = runner.run(
            "create_directshape_from_mesh",
            parameters=[json.dumps(payload)],
        )
        if "_note" in result:
            # No MCP sender configured — return for manual dispatch
            return {**base_info, "status": "formatted", "payload_json": json.dumps(payload)}
        return {**base_info, "status": "ok", **result}

    return {
        **base_info,
        "status": "formatted",
        "payload_json": json.dumps(payload),
        "script_name": "create_directshape_from_mesh",
    }
