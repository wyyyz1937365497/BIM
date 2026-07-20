"""Learned residual pose refinement for TRELLIS B-class meshes.

The deterministic mesh registrar remains the source of the initial placement.
This module optionally compares the observed 3DGS RGB-D-mask crop with a
rendering of that placement and predicts bounded rotation, translation, and
uniform-scale residuals. Low-confidence predictions fall back to the original
placement.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bim_recon.mesh_registrar import (
    MeshPlacement,
    MeshTransform,
    compute_placement_transform,
    parse_glb_vertices_faces,
)


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """Observed object crop and the camera that produced it."""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    norm_bbox: tuple[float, float, float, float]
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    camera_up: tuple[float, float, float]
    camera_fov: float
    camera_image_size: tuple[int, int]
    up_axis: int = 2
    crop_padding: float = 0.08


@dataclass(frozen=True, slots=True)
class PoseQuality:
    silhouette_iou: float
    depth_agreement: float
    score: float


@dataclass(frozen=True, slots=True)
class PoseRefinementResult:
    """Accepted placement or an explicit deterministic fallback."""

    placement: MeshPlacement
    accepted: bool
    confidence: float
    iterations: int
    initial_quality: PoseQuality
    refined_quality: PoseQuality
    fallback_reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "accepted": self.accepted,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "fallback_reason": self.fallback_reason,
            "initial_quality": asdict(self.initial_quality),
            "refined_quality": asdict(self.refined_quality),
            "refined_pose": {
                "rotation_override": self.placement.rotation_override,
                "translation_offset": list(self.placement.translation_offset),
                "scale_multiplier": self.placement.scale_multiplier,
            },
        }


class ImageEncoder(nn.Module):
    """Compact shared encoder for RGB, relative depth, and mask channels."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(image)
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, features.mean(dim=(2, 3))


class MeshEncoder(nn.Module):
    """PointNet-style encoder for normalized points and normals."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points.transpose(1, 2)).amax(dim=2)


class PoseRefinerNet(nn.Module):
    """Multi-input residual pose network with auxiliary mask/depth heads."""

    metadata_dim = 24

    def __init__(self, channels: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(channels)
        self.mesh_encoder = MeshEncoder(channels)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.metadata_dim, 128),
            nn.SiLU(),
            nn.Linear(128, channels),
            nn.SiLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            channels, num_heads=4, batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 5, 384),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.SiLU(),
        )
        self.rotation_head = nn.Linear(256, 6)
        self.translation_head = nn.Linear(256, 3)
        self.log_scale_head = nn.Linear(256, 1)
        self.confidence_head = nn.Linear(256, 1)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(256, 64 * 8 * 8),
            nn.SiLU(),
        )
        self.auxiliary_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1),
        )
        self._initialize_residual_heads()

    def _initialize_residual_heads(self) -> None:
        for head in (self.rotation_head, self.translation_head, self.log_scale_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        observed: torch.Tensor,
        candidate: torch.Tensor,
        mesh_features: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        observed_tokens, observed_global = self.image_encoder(observed)
        candidate_tokens, candidate_global = self.image_encoder(candidate)
        attended, _ = self.cross_attention(
            observed_tokens, candidate_tokens, candidate_tokens,
            need_weights=False,
        )
        attended_global = attended.mean(dim=1)
        mesh_global = self.mesh_encoder(mesh_features)
        metadata_global = self.metadata_encoder(metadata)
        fused = self.fusion(torch.cat([
            observed_global,
            candidate_global,
            attended_global,
            mesh_global,
            metadata_global,
        ], dim=1))
        auxiliary = self.auxiliary_decoder(
            self.auxiliary_head(fused).view(-1, 64, 8, 8),
        )
        return {
            "rotation_6d": self.rotation_head(fused),
            "translation_raw": self.translation_head(fused),
            "log_scale_raw": self.log_scale_head(fused),
            "confidence_logit": self.confidence_head(fused).squeeze(1),
            "mask_logit": auxiliary[:, 0],
            "depth_pred": auxiliary[:, 1],
        }


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. continuous 6D rotation representation to SO(3)."""

    first = rotation_6d[..., 0:3]
    second = rotation_6d[..., 3:6]
    basis_x = F.normalize(first, dim=-1)
    basis_y = F.normalize(
        second - (basis_x * second).sum(dim=-1, keepdim=True) * basis_x,
        dim=-1,
    )
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-1)


def rotation_geodesic_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def pose_refiner_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Training loss for positive and deliberately mismatched mesh pairs."""

    matched = batch["matched"].float()
    denominator = matched.sum().clamp_min(1.0)
    predicted_rotation = rotation_6d_to_matrix(outputs["rotation_6d"])
    rotation = (
        rotation_geodesic_loss(predicted_rotation, batch["rotation_target"]) * matched
    ).sum() / denominator
    translation = (
        F.smooth_l1_loss(
            outputs["translation_raw"], batch["translation_target"], reduction="none",
        ).mean(dim=1) * matched
    ).sum() / denominator
    log_scale = (
        F.smooth_l1_loss(
            outputs["log_scale_raw"].squeeze(1),
            batch["log_scale_target"],
            reduction="none",
        ) * matched
    ).sum() / denominator
    confidence = F.binary_cross_entropy_with_logits(
        outputs["confidence_logit"], matched,
    )

    target_mask = F.interpolate(
        batch["observed"][:, 4:5], outputs["mask_logit"].shape[-2:], mode="nearest",
    ).squeeze(1)
    mask_bce = F.binary_cross_entropy_with_logits(outputs["mask_logit"], target_mask)
    mask_probability = torch.sigmoid(outputs["mask_logit"])
    intersection = (mask_probability * target_mask).sum(dim=(1, 2))
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (mask_probability.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2)) + 1.0)
    ).mean()

    target_depth = F.interpolate(
        batch["observed"][:, 3:4], outputs["depth_pred"].shape[-2:],
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    valid_depth = target_mask > 0.5
    if valid_depth.any():
        depth = F.smooth_l1_loss(
            outputs["depth_pred"][valid_depth], target_depth[valid_depth],
        )
    else:
        depth = outputs["depth_pred"].sum() * 0.0

    parts = {
        "rotation": rotation,
        "translation": translation,
        "log_scale": log_scale,
        "confidence": confidence,
        "mask_bce": mask_bce,
        "mask_dice": dice,
        "depth": depth,
    }
    total = (
        2.0 * rotation
        + translation
        + 0.5 * log_scale
        + 0.5 * confidence
        + 0.25 * mask_bce
        + 0.25 * dice
        + 0.25 * depth
    )
    return total, parts


def sample_mesh_surface(
    glb_path: Path,
    count: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic surface points, normals, and normalized features."""

    vertices, faces = parse_glb_vertices_faces(glb_path)
    triangles = vertices[faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-10
    if not np.any(valid):
        center = vertices.mean(axis=0)
        extent = max(float(np.ptp(vertices, axis=0).max()), 1e-6)
        indices = np.arange(count) % len(vertices)
        points = vertices[indices].astype(np.float32)
        normals = np.zeros_like(points)
        features = np.concatenate(((points - center) / extent, normals), axis=1)
        return points, normals, features.astype(np.float32)

    triangles = triangles[valid]
    cross = cross[valid]
    probabilities = double_area[valid] / double_area[valid].sum()
    seed_bytes = hashlib.sha1(str(glb_path.resolve()).encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    chosen = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    selected = triangles[chosen]
    u = rng.random(count)
    v = rng.random(count)
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    points = (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)
    normals_all = cross / np.linalg.norm(cross, axis=1, keepdims=True).clip(1e-12)
    normals = normals_all[chosen].astype(np.float32)
    center = points.mean(axis=0)
    extent = max(float(np.ptp(points, axis=0).max()), 1e-6)
    features = np.concatenate(((points - center) / extent, normals), axis=1)
    return points, normals, features.astype(np.float32)


def _crop_geometry(observation: PoseObservation) -> tuple[float, float, float, float]:
    width, height = observation.camera_image_size
    x, y, bbox_width, bbox_height = observation.norm_bbox
    padding = observation.crop_padding
    x0 = (x - bbox_width / 2.0 - padding) * width
    y0 = (y - bbox_height / 2.0 - padding) * height
    crop_width = max((bbox_width + 2.0 * padding) * width, 4.0)
    crop_height = max((bbox_height + 2.0 * padding) * height, 4.0)
    return x0, y0, crop_width, crop_height


def render_mesh_channels(
    points_mesh: np.ndarray,
    normals_mesh: np.ndarray,
    transform: MeshTransform,
    observation: PoseObservation,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Software-render normal RGB, metric depth, and silhouette into the crop."""

    points_world = (
        (points_mesh - np.asarray(transform.mesh_center, dtype=np.float32))
        @ transform.rotation.T
    ) * transform.scale + transform.translation
    normals_world = normals_mesh @ transform.rotation.T

    eye = np.asarray(observation.camera_eye, dtype=np.float64)
    target = np.asarray(observation.camera_target, dtype=np.float64)
    up = np.asarray(observation.camera_up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)

    relative = points_world - eye
    z_camera = relative @ forward
    x_camera = relative @ right
    y_camera = relative @ down
    image_width, image_height = observation.camera_image_size
    focal = 0.5 * image_height / math.tan(math.radians(observation.camera_fov) / 2.0)
    pixel_x = x_camera / np.maximum(z_camera, 1e-6) * focal + image_width / 2.0
    pixel_y = y_camera / np.maximum(z_camera, 1e-6) * focal + image_height / 2.0
    crop_x0, crop_y0, crop_width, crop_height = _crop_geometry(observation)
    output_x = (pixel_x - crop_x0) / crop_width * output_size
    output_y = (pixel_y - crop_y0) / crop_height * output_size

    valid = (
        (z_camera > 0.05)
        & (output_x >= -1)
        & (output_x < output_size + 1)
        & (output_y >= -1)
        & (output_y < output_size + 1)
    )
    rgb = np.zeros((output_size, output_size, 3), dtype=np.float32)
    depth = np.zeros((output_size, output_size), dtype=np.float32)
    mask = np.zeros((output_size, output_size), dtype=np.float32)
    if not np.any(valid):
        return rgb, depth, mask

    px = output_x[valid].astype(np.int32)
    py = output_y[valid].astype(np.int32)
    z_values = z_camera[valid].astype(np.float32)
    colors = ((normals_world[valid] + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)
    order = np.argsort(z_values)[::-1]
    px, py, z_values, colors = px[order], py[order], z_values[order], colors[order]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sx = px + dx
            sy = py + dy
            inside = (sx >= 0) & (sx < output_size) & (sy >= 0) & (sy < output_size)
            rgb[sy[inside], sx[inside]] = colors[inside]
            depth[sy[inside], sx[inside]] = z_values[inside]
            mask[sy[inside], sx[inside]] = 1.0
    return rgb, depth, mask


def _resize_array(array: np.ndarray, size: int, mode: str) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor.permute(2, 0, 1)[None]
    return F.interpolate(
        tensor,
        size=(size, size),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )[0]

def crop_observation_channels(
    observation: PoseObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop aligned full-frame channels to the Falcon bbox plus padding."""

    x0, y0, crop_width, crop_height = _crop_geometry(observation)
    image_height, image_width = observation.depth.shape
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(image_width, int(math.ceil(x0 + crop_width)))
    bottom = min(image_height, int(math.ceil(y0 + crop_height)))
    if right <= left or bottom <= top:
        raise ValueError("Pose observation bbox produces an empty crop")
    return (
        observation.rgb[top:bottom, left:right],
        observation.depth[top:bottom, left:right],
        observation.mask[top:bottom, left:right],
    )


def build_image_tensor(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    size: int,
    depth_reference: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Build a 5-channel network tensor and return its metric depth reference."""

    rgb_array = np.asarray(rgb)
    if rgb_array.ndim == 2:
        rgb_array = np.repeat(rgb_array[:, :, None], 3, axis=2)
    if rgb_array.shape[2] > 3:
        rgb_array = rgb_array[:, :, :3]
    rgb_float = rgb_array.astype(np.float32)
    if rgb_float.max(initial=0.0) > 1.0:
        rgb_float /= 255.0
    mask_float = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    depth_float = np.asarray(depth, dtype=np.float32)
    valid = (depth_float > 0.05) & (mask_float > 0.5)
    if depth_reference is None:
        depth_reference = float(np.median(depth_float[valid])) if np.any(valid) else 1.0
    relative_depth = np.zeros_like(depth_float)
    positive = depth_float > 0.05
    relative_depth[positive] = np.clip(
        depth_float[positive] / max(depth_reference, 1e-4) - 1.0, -2.0, 2.0,
    )
    rgb_tensor = _resize_array(rgb_float, size, "bilinear")
    depth_tensor = _resize_array(relative_depth, size, "bilinear")
    mask_tensor = _resize_array(mask_float, size, "nearest")
    return torch.cat((rgb_tensor, depth_tensor, mask_tensor), dim=0), depth_reference


def _metadata_vector(
    observation: PoseObservation,
    placement: MeshPlacement,
    transform: MeshTransform,
    quality: PoseQuality,
) -> np.ndarray:
    eye = np.asarray(observation.camera_eye, dtype=np.float32)
    target = np.asarray(observation.camera_target, dtype=np.float32)
    direction = target - eye
    distance = float(np.linalg.norm(direction))
    direction /= distance + 1e-12
    bbox = np.asarray(observation.norm_bbox, dtype=np.float32)
    up_one_hot = np.zeros(3, dtype=np.float32)
    up_one_hot[placement.up_axis] = 1.0
    values = np.concatenate([
        direction,
        np.array([distance, observation.camera_fov / 180.0], dtype=np.float32),
        bbox,
        up_one_hot,
        np.asarray(transform.translation, dtype=np.float32),
        np.array([
            transform.scale,
            placement.element_width_m,
            placement.element_height_m,
            placement.floor_z,
            placement.ceiling_z,
            quality.silhouette_iou,
            quality.depth_agreement,
            quality.score,
            1.0,
        ], dtype=np.float32),
    ])
    if values.shape != (PoseRefinerNet.metadata_dim,):
        raise RuntimeError(f"metadata shape mismatch: {values.shape}")
    return values


def _quality(
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_depth: np.ndarray,
) -> PoseQuality:
    observed = observed_mask > 0.5
    candidate = candidate_mask > 0.5
    union = int((observed | candidate).sum())
    silhouette_iou = float((observed & candidate).sum() / max(union, 1))
    overlap = observed & candidate & (observed_depth > 0.05) & (candidate_depth > 0.05)
    if np.any(overlap):
        reference = max(float(np.median(observed_depth[overlap])), 0.1)
        median_error = float(np.median(np.abs(
            observed_depth[overlap] - candidate_depth[overlap]
        )))
        depth_agreement = float(math.exp(-median_error / (0.1 * reference + 1e-6)))
    else:
        depth_agreement = 0.0
    score = 0.6 * silhouette_iou + 0.4 * depth_agreement
    return PoseQuality(silhouette_iou, depth_agreement, score)


def _clamp_rotation(rotation: np.ndarray, max_degrees: float) -> np.ndarray:
    trace = float(np.trace(rotation))
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    maximum = math.radians(max_degrees)
    if angle <= maximum or angle < 1e-8:
        return rotation.astype(np.float32)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    x, y, z = axis
    c, s = math.cos(maximum), math.sin(maximum)
    one_minus_c = 1.0 - c
    return np.array([
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
    ], dtype=np.float32)


def project_rotation_to_gravity(rotation: np.ndarray, up_axis: int) -> np.ndarray:
    """Keep TRELLIS mesh Y vertical while retaining the predicted heading."""

    world_up = np.zeros(3, dtype=np.float64)
    world_up[up_axis] = 1.0
    mesh_x = np.asarray(rotation[:, 0], dtype=np.float64)
    mesh_x -= world_up * float(mesh_x @ world_up)
    if np.linalg.norm(mesh_x) < 1e-8:
        mesh_z = np.asarray(rotation[:, 2], dtype=np.float64)
        mesh_x = np.cross(world_up, mesh_z)
    mesh_x /= np.linalg.norm(mesh_x) + 1e-12
    mesh_y = world_up
    mesh_z = np.cross(mesh_x, mesh_y)
    mesh_z /= np.linalg.norm(mesh_z) + 1e-12
    return np.stack((mesh_x, mesh_y, mesh_z), axis=1).astype(np.float32)


def _rotation_tuple(rotation: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in rotation)


class PoseRefiner:
    """Checkpoint-backed inference wrapper with deterministic fallback policy."""

    def __init__(self, config: Any):
        self.config = config
        requested_device = str(config.device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = PoseRefinerNet().to(self.device)
        checkpoint_path = Path(config.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Pose refiner checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._mesh_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def refine_placement(
        self,
        placement: MeshPlacement,
        observation: PoseObservation,
    ) -> PoseRefinementResult:
        mask_pixels = int((observation.mask > 0.5).sum())
        depth_ratio = float((observation.depth > 0.05).mean())
        if mask_pixels < int(self.config.min_mask_pixels):
            return self._fallback(placement, observation, "insufficient_mask")
        if depth_ratio < float(self.config.min_depth_ratio):
            return self._fallback(placement, observation, "insufficient_depth")

        points, normals, mesh_features = self._mesh_data(placement.glb_path)
        initial_transform = compute_placement_transform(placement)
        initial_rgb, initial_depth, initial_mask = render_mesh_channels(
            points, normals, initial_transform, observation, int(self.config.input_size),
        )
        observed_rgb, observed_depth_metric, observed_mask_metric = (
            crop_observation_channels(observation)
        )
        observed_tensor, depth_reference = build_image_tensor(
            observed_rgb, observed_depth_metric, observed_mask_metric,
            int(self.config.input_size),
        )
        observed_mask = _resize_array(
            (observed_mask_metric > 0.5).astype(np.float32),
            int(self.config.input_size), "nearest",
        )[0].numpy()
        observed_depth = _resize_array(
            observed_depth_metric.astype(np.float32),
            int(self.config.input_size), "bilinear",
        )[0].numpy()
        initial_quality = _quality(
            observed_mask, observed_depth, initial_mask, initial_depth,
        )

        current = placement
        current_transform = initial_transform
        confidence = 0.0
        iterations_run = 0
        for _ in range(max(1, int(self.config.iterations))):
            candidate_rgb, candidate_depth, candidate_mask = render_mesh_channels(
                points, normals, current_transform, observation,
                int(self.config.input_size),
            )
            candidate_tensor, _ = build_image_tensor(
                candidate_rgb, candidate_depth, candidate_mask,
                int(self.config.input_size), depth_reference,
            )
            candidate_quality = _quality(
                observed_mask, observed_depth, candidate_mask, candidate_depth,
            )
            metadata = _metadata_vector(
                observation, current, current_transform, candidate_quality,
            )
            with torch.inference_mode():
                outputs = self.model(
                    observed_tensor[None].to(self.device),
                    candidate_tensor[None].to(self.device),
                    torch.from_numpy(mesh_features)[None].to(self.device),
                    torch.from_numpy(metadata)[None].to(self.device),
                )
            residual_rotation = rotation_6d_to_matrix(outputs["rotation_6d"])[0].cpu().numpy()
            residual_rotation = _clamp_rotation(
                residual_rotation, float(self.config.max_rotation_degrees),
            )
            refined_rotation = residual_rotation @ current_transform.rotation
            if bool(self.config.gravity_locked):
                refined_rotation = project_rotation_to_gravity(
                    refined_rotation, placement.up_axis,
                )
            translation_delta = (
                torch.tanh(outputs["translation_raw"])[0].cpu().numpy()
                * float(self.config.max_translation_m)
            )
            scale_delta = math.exp(float(
                torch.tanh(outputs["log_scale_raw"])[0, 0].cpu()
                * float(self.config.max_log_scale)
            ))
            previous_offset = np.asarray(current.translation_offset, dtype=np.float32)
            current = replace(
                current,
                rotation_override=_rotation_tuple(refined_rotation),
                translation_offset=tuple(float(v) for v in previous_offset + translation_delta),
                scale_multiplier=float(current.scale_multiplier * scale_delta),
                preserve_floor_contact=bool(self.config.floor_contact),
            )
            current_transform = compute_placement_transform(current)
            confidence = float(torch.sigmoid(outputs["confidence_logit"])[0].cpu())
            iterations_run += 1

        _, refined_depth, refined_mask = render_mesh_channels(
            points, normals, current_transform, observation,
            int(self.config.input_size),
        )
        refined_quality = _quality(
            observed_mask, observed_depth, refined_mask, refined_depth,
        )
        accepted = (
            confidence >= float(self.config.confidence_threshold)
            and refined_quality.score >= float(self.config.min_quality_score)
            and refined_quality.score + float(self.config.quality_tolerance)
            >= initial_quality.score
        )
        if not accepted:
            reasons = []
            if confidence < float(self.config.confidence_threshold):
                reasons.append("low_confidence")
            if refined_quality.score < float(self.config.min_quality_score):
                reasons.append("low_quality")
            if refined_quality.score + float(self.config.quality_tolerance) < initial_quality.score:
                reasons.append("quality_regression")
            return PoseRefinementResult(
                placement=placement,
                accepted=False,
                confidence=confidence,
                iterations=iterations_run,
                initial_quality=initial_quality,
                refined_quality=refined_quality,
                fallback_reason="+".join(reasons) or "policy_rejected",
            )
        return PoseRefinementResult(
            placement=current,
            accepted=True,
            confidence=confidence,
            iterations=iterations_run,
            initial_quality=initial_quality,
            refined_quality=refined_quality,
        )

    def _mesh_data(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolved = path.resolve()
        if resolved not in self._mesh_cache:
            self._mesh_cache[resolved] = sample_mesh_surface(resolved)
        return self._mesh_cache[resolved]

    def _fallback(
        self,
        placement: MeshPlacement,
        observation: PoseObservation,
        reason: str,
    ) -> PoseRefinementResult:
        empty = PoseQuality(0.0, 0.0, 0.0)
        return PoseRefinementResult(
            placement=placement,
            accepted=False,
            confidence=0.0,
            iterations=0,
            initial_quality=empty,
            refined_quality=empty,
            fallback_reason=reason,
        )


def create_pose_refiner(config: Any) -> PoseRefiner | None:
    """Return a configured refiner, or None when the feature is disabled."""

    if config is None or not bool(config.enabled):
        return None
    if not str(config.checkpoint).strip():
        raise ValueError("pose_refiner.enabled requires pose_refiner.checkpoint")
    return PoseRefiner(config)


def load_pose_observation(
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    norm_bbox: tuple[float, float, float, float],
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    camera_fov: float,
    camera_image_size: tuple[int, int],
    up_axis: int,
) -> PoseObservation:
    """Load aligned full-frame RGB, depth, and mask observation assets."""

    from PIL import Image

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError(
            f"Pose observation shapes differ: rgb={rgb.shape[:2]}, "
            f"depth={depth.shape}, mask={mask.shape}"
        )
    if (rgb.shape[1], rgb.shape[0]) != tuple(camera_image_size):
        raise ValueError(
            f"Pose camera image size {camera_image_size} differs from RGB "
            f"size {(rgb.shape[1], rgb.shape[0])}"
        )
    return PoseObservation(
        rgb=rgb,
        depth=depth,
        mask=mask,
        norm_bbox=norm_bbox,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_up=camera_up,
        camera_fov=camera_fov,
        camera_image_size=camera_image_size,
        up_axis=up_axis,
    )


__all__ = [
    "PoseObservation",
    "PoseQuality",
    "PoseRefinementResult",
    "PoseRefinerNet",
    "PoseRefiner",
    "create_pose_refiner",
    "crop_observation_channels",
    "build_image_tensor",
    "load_pose_observation",
    "pose_refiner_loss",
    "project_rotation_to_gravity",
    "render_mesh_channels",
    "rotation_6d_to_matrix",
    "rotation_geodesic_loss",
    "sample_mesh_surface",
]
