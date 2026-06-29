"""3DGS scene loader and renderer.

Loads a nerfstudio-exported Gaussian Splatting PLY file and provides
gsplat-based rendering (RGB + expected depth) from arbitrary camera poses.

Camera convention: OpenCV / COLMAP (+x right, +y down, +z forward).
All camera-to-world transforms are 4x4 matrices in column-major convention
used by gsplat (``viewmats`` is world-to-camera).
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gsplat import rasterization

# SH C0 constant — converts DC coefficient to linear RGB.
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class CameraPose:
    """A camera pose in world space (camera-to-world transform).

    The camera looks down +z_local, +x_local points right, +y_local points down
    (OpenCV / COLMAP convention).
    """

    position: Tuple[float, float, float]
    # Camera-to-world rotation as a quaternion (w, x, y, z), must be normalized.
    quaternion_wxyz: Tuple[float, float, float, float]

    def to_viewmat(self) -> np.ndarray:
        """Return the 4x4 world-to-camera (view) matrix, row-major."""
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = _quat_to_rotmat(self.quaternion_wxyz)
        c2w[:3, 3] = self.position
        # view = inverse(camera_to_world)
        viewmat = np.linalg.inv(c2w)
        return viewmat.astype(np.float32)


def _quat_to_rotmat(q: Tuple[float, float, float, float]) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def look_at_pose(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> CameraPose:
    """Construct a CameraPose looking from ``eye`` toward ``target``.

    Note: assumes the world up axis is +y. The camera's local up is -y
    (because +y_local is down in OpenCV convention), so we pass the world
    up vector directly; the resulting rotation handles the sign.
    """
    eye_v = np.asarray(eye, dtype=np.float32)
    target_v = np.asarray(target, dtype=np.float32)
    up_v = np.asarray(up, dtype=np.float32)

    forward = target_v - eye_v
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up_v)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)  # +y points down in OpenCV

    # camera-to-world rotation: columns are right, down, forward
    c2w_rot = np.stack([right, down, forward], axis=1)
    quat = _rotmat_to_quat(c2w_rot)
    return CameraPose(position=(float(eye_v[0]), float(eye_v[1]), float(eye_v[2])), quaternion_wxyz=quat)


def _rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return (w, x, y, z)


def fov_to_intrinsics(fov_degrees: float, width: int, height: int) -> np.ndarray:
    """Build a 3x3 pinhole intrinsics matrix from horizontal FOV.

    The focal length is derived from the horizontal FOV and applied to both
    axes (square pixels assumed), with the principal point at the image center.
    """
    fx = 0.5 * width / math.tan(0.5 * math.radians(fov_degrees))
    return np.array(
        [[fx, 0.0, width / 2.0],
         [0.0, fx, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@dataclass
class RenderResult:
    """The output of :meth:`GSScene.render`.

    ``colors`` is HxWx3 float32 in [0, 1] (clamped). ``depth`` is HxW float32
    metric depth in scene units (metres if the SfM model is metric); pixels with
    no Gaussians have depth 0. ``alpha`` is HxW float32 in [0, 1] indicating
    the accumulated opacity, useful to tell rendered pixels from background.
    """

    colors: np.ndarray  # (H, W, 3) float32
    depth: np.ndarray   # (H, W) float32
    alpha: np.ndarray   # (H, W) float32


@dataclass
class GSScene:
    """A loaded 3D Gaussian Splatting scene, renderable via gsplat.

    The PLY format follows nerfstudio's ``ExportGaussianSplat`` layout
    (SH DC coefficients, logit opacity, log scales, (w,x,y,z) quaternions).
    """

    means: torch.Tensor       # (N, 3) float32 on device
    quats: torch.Tensor       # (N, 4) float32 (w, x, y, z)
    scales: torch.Tensor      # (N, 3) float32 (positive, metric)
    opacities: torch.Tensor   # (N,) float32 in [0, 1]
    colors: torch.Tensor      # (N, 3) float32 linear RGB in [0, 1]
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_ply(cls, ply_path: str | Path, device: Optional[torch.device] = None) -> "GSScene":
        """Load a nerfstudio-exported splat PLY file."""
        path = Path(ply_path)
        if not path.exists():
            raise FileNotFoundError(f"PLY file not found: {path}")
        device = device or torch.device("cuda")
        data = _parse_ply_binary(path)
        return cls._from_parsed(data, device)

    @classmethod
    def from_synthetic(
        cls,
        means: np.ndarray,
        colors_rgb: np.ndarray,
        scales: Optional[np.ndarray] = None,
        opacities: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> "GSScene":
        """Build a synthetic scene for testing (no PLY needed).

        ``means`` is (N, 3) positions, ``colors_rgb`` is (N, 3) linear RGB in
        [0, 1]. Scales default to 0.05 and opacities to 1.0.
        """
        device = device or torch.device("cuda")
        means_np = np.asarray(means, dtype=np.float32)
        n = means_np.shape[0]
        colors_np = np.asarray(colors_rgb, dtype=np.float32)
        scales_np = (
            np.asarray(scales, dtype=np.float32)
            if scales is not None
            else np.full((n, 3), 0.05, dtype=np.float32)
        )
        opacities_np = (
            np.asarray(opacities, dtype=np.float32)
            if opacities is not None
            else np.ones(n, dtype=np.float32)
        )
        quats_np = np.zeros((n, 4), dtype=np.float32)
        quats_np[:, 0] = 1.0  # identity quaternion (w, x, y, z)
        return cls(
            means=torch.from_numpy(means_np).to(device),
            quats=torch.from_numpy(quats_np).to(device),
            scales=torch.from_numpy(scales_np).to(device),
            opacities=torch.from_numpy(opacities_np).to(device),
            colors=torch.from_numpy(colors_np).to(device),
            device=device,
        )

    @classmethod
    def _from_parsed(cls, data: Dict[str, np.ndarray], device: torch.device) -> "GSScene":
        """Convert parsed PLY fields to gsplat-ready tensors."""
        means = data["positions"].astype(np.float32)
        # opacity: logit -> probability
        opacity_logit = data["opacity"].reshape(-1).astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
        # scales: log space -> linear
        scales = np.exp(data["scales"].astype(np.float32))
        # quaternions: (w, x, y, z) — normalize defensively
        quats = data["rotations"].astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(norms, 1e-12, None)
        # SH DC -> linear RGB color
        sh_dc = data["sh_dc"].astype(np.float32).reshape(-1, 3)
        colors = np.clip(SH_C0 * sh_dc + 0.5, 0.0, 1.0)
        return cls(
            means=torch.from_numpy(means).to(device),
            quats=torch.from_numpy(quats).to(device),
            scales=torch.from_numpy(scales).to(device),
            opacities=torch.from_numpy(opacities).to(device),
            colors=torch.from_numpy(colors).to(device),
            device=device,
        )

    # ---- queries ------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) of Gaussian means in world space."""
        mn = self.means.min(dim=0).values.cpu().numpy()
        mx = self.means.max(dim=0).values.cpu().numpy()
        return mn, mx

    # ---- rendering ----------------------------------------------------------

    def render(
        self,
        pose: CameraPose,
        width: int,
        height: int,
        fov_degrees: float = 60.0,
    ) -> RenderResult:
        """Render the scene from a single camera pose.

        Returns RGB (HxWx3 in [0,1]), metric depth (HxW; 0 where empty), and
        alpha (HxW in [0,1]). Uses ``render_mode='RGB+ED'`` so the depth is
        the expected (alpha-normalized) metric depth, not raw accumulation.
        """
        viewmat = torch.from_numpy(pose.to_viewmat()).to(self.device).unsqueeze(0)
        K = torch.from_numpy(fov_to_intrinsics(fov_degrees, width, height)).to(self.device).unsqueeze(0)

        render_colors, render_alphas, _meta = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            render_mode="RGB+ED",
        )
        # render_colors: (1, H, W, 4) -> RGB(3) + expected_depth(1)
        # render_alphas: (1, H, W, 1)
        colors = render_colors[0, :, :, :3].clamp(0.0, 1.0).cpu().numpy()
        depth = render_colors[0, :, :, 3].cpu().numpy()
        alpha = render_alphas[0, :, :, 0].cpu().numpy()
        # Zero-out depth where alpha is effectively zero (background).
        depth = np.where(alpha > 1e-3, depth, 0.0).astype(np.float32)
        return RenderResult(colors=colors.astype(np.float32), depth=depth, alpha=alpha.astype(np.float32))

    def render_batch(
        self,
        poses: List[CameraPose],
        width: int,
        height: int,
        fov_degrees: float = 60.0,
    ) -> List[RenderResult]:
        """Render multiple poses. Currently loops; can be batched later."""
        return [self.render(p, width, height, fov_degrees) for p in poses]

    # ---- selection ----------------------------------------------------------

    def select_by_mask(
        self,
        pose: CameraPose,
        mask: np.ndarray,
        width: int,
        height: int,
        fov_degrees: float = 60.0,
    ) -> np.ndarray:
        """Return the indices of Gaussians contributing to ``mask`` pixels.

        ``mask`` is a boolean HxW array (True = selected pixels). We render the
        scene, inspect the per-Gaussian visibility via the alpha intersection
        meta, and return the indices of Gaussians whose projection overlaps any
        selected pixel. This is the bridge between a 2D VLM mask and the 3D
        Gaussians, enabling cluster selection.
        """
        if mask.shape != (height, width):
            raise ValueError(f"mask shape {mask.shape} != ({height}, {width})")
        viewmat = torch.from_numpy(pose.to_viewmat()).to(self.device).unsqueeze(0)
        K = torch.from_numpy(fov_to_intrinsics(fov_degrees, width, height)).to(self.device).unsqueeze(0)
        _rc, _ra, meta = rasterization(
            self.means, self.quats, self.scales, self.opacities, self.colors,
            viewmats=viewmat, Ks=K, width=width, height=height,
            render_mode="RGB",
        )
        # meta["means2d"]: (K, 2) projected 2D centers for the K visible Gaussians.
        # meta["gaussian_ids"]: (K,) index into the original N Gaussians.
        means2d = meta["means2d"].cpu().numpy()  # (K, 2)
        gaussian_ids = meta["gaussian_ids"].cpu().numpy()  # (K,)
        if means2d.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        px = np.round(means2d[:, 0]).astype(np.int64)
        py = np.round(means2d[:, 1]).astype(np.int64)
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        selected = mask[py, px]
        return gaussian_ids[selected]


# ---- PLY parsing -----------------------------------------------------------

def _parse_ply_binary(path: Path) -> Dict[str, np.ndarray]:
    """Parse a binary little-endian PLY (nerfstudio splat format).

    Returns a dict with keys:
      - ``positions``: (N, 3) float32
      - ``sh_dc``: (N, 3) float32 (if SH mode) — absent in RGB mode
      - ``rgb``: (N, 3) uint8 (if RGB mode) — absent in SH mode
      - ``opacity``: (N, 1) float32 (logit space, as stored)
      - ``scales``: (N, 3) float32 (log space, as stored)
      - ``rotations``: (N, 4) float32 (w, x, y, z)
    """
    with open(path, "rb") as f:
        header_bytes = _read_ply_header(f)
        props, count, format_kind = _parse_ply_header_text(header_bytes)
        # format_kind looks like "binary_little_endian 1.0"; match the leading token.
        if not format_kind.startswith("binary_little_endian"):
            raise NotImplementedError(f"PLY format not supported: {format_kind}")
        return _read_ply_binary_data(f, props, count)


def _read_ply_header(f) -> bytes:
    """Read the PLY header as bytes (up to and including 'end_header\\n')."""
    header_lines: List[bytes] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF before end_header")
        header_lines.append(line)
        if line.strip() == b"end_header":
            break
    return b"".join(header_lines)


def _parse_ply_header_text(header: bytes) -> Tuple[List[str], int, str]:
    """Extract element properties, vertex count, and format from the header."""
    lines = header.decode("ascii", errors="replace").splitlines()
    format_kind = "unknown"
    count = 0
    in_vertex = False
    props: List[str] = []
    for line in lines:
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "format":
            format_kind = " ".join(toks[1:])
        elif toks[0] == "element" and len(toks) >= 3 and toks[1] == "vertex":
            count = int(toks[2])
            in_vertex = True
            props = []
        elif toks[0] == "property" and in_vertex:
            # e.g. "property float x" or "property uchar red"
            kind = toks[1]
            name = toks[2]
            props.append(f"{kind}:{name}")
        elif toks[0] == "element" and toks[1] != "vertex":
            in_vertex = False
    return props, count, format_kind


_PLY_FLOAT_NAMES = {"x", "y", "z", "nx", "ny", "nz", "opacity",
                    "scale_0", "scale_1", "scale_2",
                    "rot_0", "rot_1", "rot_2", "rot_3",
                    "red", "green", "blue"}
_PLY_SH_DC_NAMES = {"f_dc_0", "f_dc_1", "f_dc_2"}
_PLY_SH_REST_PREFIX = "f_rest_"


def _read_ply_binary_data(f, props: List[str], count: int) -> Dict[str, np.ndarray]:
    """Parse the binary vertex data block per the property list."""
    # Build a structured dtype. numpy accepts (name, dtype_str) tuples for
    # scalar fields; we keep the kind separately for post-processing.
    fields: List[Tuple[str, str]] = []
    for prop in props:
        kind, name = prop.split(":", 1)
        if kind == "float":
            fields.append((name, "<f4"))
        elif kind in ("uchar", "uint8"):
            fields.append((name, "u1"))
        elif kind in ("double",):
            fields.append((name, "<f8"))
        elif kind in ("int", "int32"):
            fields.append((name, "<i4"))
        else:
            raise NotImplementedError(f"PLY property kind not supported: {kind}")
    dtype = np.dtype(fields)
    raw = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    names = raw.dtype.names
    if names is None:
        raise RuntimeError("PLY structured array has no named fields")

    out: Dict[str, np.ndarray] = {}
    # positions
    out["positions"] = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)
    # opacity
    if "opacity" in names:
        out["opacity"] = raw["opacity"].reshape(-1, 1).astype(np.float32)
    # scales (log space preserved; exp applied later by GSScene)
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        out["scales"] = np.stack([raw["scale_0"], raw["scale_1"], raw["scale_2"]], axis=-1).astype(np.float32)
    # rotations
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        out["rotations"] = np.stack(
            [raw["rot_0"], raw["rot_1"], raw["rot_2"], raw["rot_3"]], axis=-1
        ).astype(np.float32)
    # color: prefer SH DC for nerfstudio exports
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        out["sh_dc"] = np.stack([raw["f_dc_0"], raw["f_dc_1"], raw["f_dc_2"]], axis=-1).astype(np.float32)
    elif {"red", "green", "blue"}.issubset(names):
        out["rgb"] = np.stack([raw["red"], raw["green"], raw["blue"]], axis=-1).astype(np.uint8)
    return out
