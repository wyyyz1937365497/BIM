"""Procedural training and evaluation data for the B-class pose refiner."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from bim_recon.pose_refiner import PoseRefinerNet, build_image_tensor


@dataclass(frozen=True, slots=True)
class SyntheticPoseConfig:
    image_size: int = 128
    mesh_points: int = 1024
    max_rotation_degrees: float = 20.0
    max_translation_m: float = 0.25
    max_log_scale: float = 0.15
    mismatch_probability: float = 0.2
    occlusion_probability: float = 0.35
    depth_noise_m: float = 0.015


def _yaw_matrix(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _inverse_tanh_bounded(value: np.ndarray | float, maximum: float) -> np.ndarray:
    normalized = np.asarray(value, dtype=np.float32) / max(maximum, 1e-8)
    return np.arctanh(np.clip(normalized, -0.98, 0.98)).astype(np.float32)


def _polygon_mask(
    size: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    corners = []
    for local_x, local_y in (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ):
        corners.append((
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        ))
    image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(image).polygon(corners, fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _render_object(
    rng: np.random.Generator,
    size: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_degrees: float,
    depth_m: float,
    color: np.ndarray,
    occlude: bool,
    depth_noise_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _polygon_mask(
        size, center_x, center_y, width, height, angle_degrees,
    )
    if occlude:
        occ_width = int(rng.uniform(0.12, 0.32) * size)
        occ_height = int(rng.uniform(0.12, 0.32) * size)
        occ_x = int(rng.uniform(0.15, 0.75) * size)
        occ_y = int(rng.uniform(0.15, 0.75) * size)
        mask[
            occ_y:min(size, occ_y + occ_height),
            occ_x:min(size, occ_x + occ_width),
        ] = 0.0

    yy, xx = np.mgrid[:size, :size]
    background = np.stack((
        0.08 + 0.10 * xx / max(size - 1, 1),
        0.10 + 0.08 * yy / max(size - 1, 1),
        np.full((size, size), 0.12, dtype=np.float32),
    ), axis=2).astype(np.float32)
    normal_shading = (
        0.78 + 0.18 * (xx / max(size - 1, 1))
        - 0.08 * (yy / max(size - 1, 1))
    ).astype(np.float32)
    rgb = background
    rgb = rgb * (1.0 - mask[:, :, None]) + (
        color[None, None, :] * normal_shading[:, :, None]
    ) * mask[:, :, None]
    rgb += rng.normal(0.0, 0.012, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    depth = np.zeros((size, size), dtype=np.float32)
    depth_values = depth_m + rng.normal(
        0.0, depth_noise_m, (size, size),
    ).astype(np.float32)
    depth[mask > 0.5] = depth_values[mask > 0.5]
    return rgb, depth, mask


def _sample_cuboid_features(
    rng: np.random.Generator,
    extents: np.ndarray,
    count: int,
) -> np.ndarray:
    face_axes = rng.integers(0, 3, size=count)
    face_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count)
    points = rng.uniform(-0.5, 0.5, size=(count, 3)).astype(np.float32) * extents
    normals = np.zeros((count, 3), dtype=np.float32)
    rows = np.arange(count)
    points[rows, face_axes] = face_signs * extents[face_axes] / 2.0
    normals[rows, face_axes] = face_signs
    normalized = points / max(float(extents.max()), 1e-6)
    return np.concatenate((normalized, normals), axis=1).astype(np.float32)


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    y_indices, x_indices = np.nonzero(mask > 0.5)
    height, width = mask.shape
    if len(x_indices) == 0:
        return 0.5, 0.5, 0.1, 0.1
    x0, x1 = int(x_indices.min()), int(x_indices.max()) + 1
    y0, y1 = int(y_indices.min()), int(y_indices.max()) + 1
    return (
        (x0 + x1) / (2.0 * width),
        (y0 + y1) / (2.0 * height),
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def make_synthetic_pose_sample(
    seed: int,
    config: SyntheticPoseConfig,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    size = int(config.image_size)
    extents = rng.uniform([0.45, 0.45, 0.45], [1.6, 1.3, 1.6]).astype(np.float32)
    base_width = float(rng.uniform(0.28, 0.52) * size)
    base_height = float(rng.uniform(0.28, 0.58) * size)
    base_center = np.array([
        rng.uniform(0.43, 0.57) * size,
        rng.uniform(0.43, 0.57) * size,
    ], dtype=np.float32)
    base_depth = float(rng.uniform(2.0, 4.5))
    base_angle = float(rng.uniform(-35.0, 35.0))
    color = rng.uniform(0.25, 0.9, size=3).astype(np.float32)

    matched = bool(rng.random() >= config.mismatch_probability)
    rotation_degrees = float(rng.uniform(
        -config.max_rotation_degrees * 0.9,
        config.max_rotation_degrees * 0.9,
    ))
    translation = rng.uniform(
        -config.max_translation_m * 0.85,
        config.max_translation_m * 0.85,
        size=3,
    ).astype(np.float32)
    log_scale = float(rng.uniform(
        -config.max_log_scale * 0.85,
        config.max_log_scale * 0.85,
    ))

    candidate_width = base_width
    candidate_height = base_height
    candidate_center = base_center.copy()
    candidate_angle = base_angle
    candidate_color = color.copy()
    if not matched:
        candidate_width *= float(rng.uniform(0.55, 1.55))
        candidate_height *= float(rng.uniform(0.55, 1.55))
        candidate_center += rng.uniform(-0.22, 0.22, size=2) * size
        candidate_angle += float(rng.uniform(35.0, 120.0))
        candidate_color = rng.uniform(0.15, 0.95, size=3).astype(np.float32)

    candidate_rgb, candidate_depth, candidate_mask = _render_object(
        rng,
        size,
        float(candidate_center[0]),
        float(candidate_center[1]),
        candidate_width,
        candidate_height,
        candidate_angle,
        base_depth,
        candidate_color,
        False,
        config.depth_noise_m * 0.25,
    )

    pixel_shift = translation[:2] / max(config.max_translation_m, 1e-6) * (0.12 * size)
    observed_scale = math.exp(log_scale)
    observed_rgb, observed_depth, observed_mask = _render_object(
        rng,
        size,
        float(base_center[0] + pixel_shift[0]),
        float(base_center[1] + pixel_shift[1]),
        base_width * observed_scale,
        base_height * observed_scale,
        base_angle + rotation_degrees,
        base_depth + float(translation[2]),
        color,
        bool(rng.random() < config.occlusion_probability),
        config.depth_noise_m,
    )

    observed_tensor, depth_reference = build_image_tensor(
        observed_rgb, observed_depth, observed_mask, size,
    )
    candidate_tensor, _ = build_image_tensor(
        candidate_rgb, candidate_depth, candidate_mask, size, depth_reference,
    )
    mesh_features = _sample_cuboid_features(rng, extents, config.mesh_points)
    bbox = _bbox_from_mask(observed_mask)
    intersection = np.logical_and(observed_mask > 0.5, candidate_mask > 0.5).sum()
    union = np.logical_or(observed_mask > 0.5, candidate_mask > 0.5).sum()
    silhouette_iou = float(intersection / max(int(union), 1))
    metadata = np.concatenate((
        np.array([0.0, 0.0, 1.0, base_depth, 50.0 / 180.0], dtype=np.float32),
        np.asarray(bbox, dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, base_depth], dtype=np.float32),
        np.array([
            1.0,
            extents[0],
            extents[1],
            0.0,
            3.0,
            silhouette_iou,
            1.0 if matched else 0.0,
            0.6 * silhouette_iou + 0.4 * float(matched),
            1.0,
        ], dtype=np.float32),
    ))
    if metadata.shape != (PoseRefinerNet.metadata_dim,):
        raise RuntimeError(f"synthetic metadata shape mismatch: {metadata.shape}")

    if matched:
        rotation_target = _yaw_matrix(rotation_degrees)
        translation_target = _inverse_tanh_bounded(
            translation, config.max_translation_m,
        )
        log_scale_target = float(_inverse_tanh_bounded(
            log_scale, config.max_log_scale,
        ))
    else:
        rotation_target = np.eye(3, dtype=np.float32)
        translation_target = np.zeros(3, dtype=np.float32)
        log_scale_target = 0.0
        rotation_degrees = 0.0
        translation = np.zeros(3, dtype=np.float32)
        log_scale = 0.0

    return {
        "observed": observed_tensor,
        "candidate": candidate_tensor,
        "mesh_features": torch.from_numpy(mesh_features),
        "metadata": torch.from_numpy(metadata),
        "matched": torch.tensor(float(matched), dtype=torch.float32),
        "rotation_target": torch.from_numpy(rotation_target),
        "translation_target": torch.from_numpy(translation_target),
        "log_scale_target": torch.tensor(log_scale_target, dtype=torch.float32),
        "rotation_degrees": torch.tensor(rotation_degrees, dtype=torch.float32),
        "translation_m": torch.from_numpy(translation.astype(np.float32)),
        "log_scale": torch.tensor(log_scale, dtype=torch.float32),
    }


class SyntheticPoseDataset(Dataset):
    """Deterministic procedural dataset indexed by seed."""

    def __init__(
        self,
        length: int,
        seed: int = 0,
        config: SyntheticPoseConfig | None = None,
    ):
        self.length = int(length)
        self.seed = int(seed)
        self.config = config or SyntheticPoseConfig()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return make_synthetic_pose_sample(self.seed + int(index) * 104729, self.config)


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_pose_model(
    model: PoseRefinerNet,
    loader: Any,
    device: torch.device,
    config: SyntheticPoseConfig,
    confidence_threshold: float = 0.65,
) -> dict[str, float]:
    model.eval()
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    log_scale_errors: list[float] = []
    confidence_correct = 0
    accepted_matches = 0
    matched_count = 0
    sample_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(
                batch["observed"],
                batch["candidate"],
                batch["mesh_features"],
                batch["metadata"],
            )
            confidence = torch.sigmoid(outputs["confidence_logit"])
            predicted_match = confidence >= confidence_threshold
            actual_match = batch["matched"] > 0.5
            confidence_correct += int((predicted_match == actual_match).sum().item())
            sample_count += int(actual_match.numel())
            accepted_matches += int((predicted_match & actual_match).sum().item())
            matched_count += int(actual_match.sum().item())
            if actual_match.any():
                predicted_rotation = outputs["rotation_6d"][actual_match]
                predicted_matrix = _rotation_6d_to_matrix_local(predicted_rotation)
                target_matrix = batch["rotation_target"][actual_match]
                relative = predicted_matrix.transpose(-1, -2) @ target_matrix
                cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
                angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
                rotation_errors.extend(angle.cpu().tolist())

                predicted_translation = (
                    torch.tanh(outputs["translation_raw"][actual_match])
                    * config.max_translation_m
                )
                translation_error = torch.linalg.norm(
                    predicted_translation - batch["translation_m"][actual_match], dim=1,
                )
                translation_errors.extend(translation_error.cpu().tolist())
                predicted_log_scale = (
                    torch.tanh(outputs["log_scale_raw"][actual_match, 0])
                    * config.max_log_scale
                )
                scale_error = torch.abs(
                    predicted_log_scale - batch["log_scale"][actual_match]
                )
                log_scale_errors.extend(scale_error.cpu().tolist())

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "samples": float(sample_count),
        "matched_samples": float(matched_count),
        "rotation_error_deg_mean": mean(rotation_errors),
        "translation_error_m_mean": mean(translation_errors),
        "log_scale_error_mean": mean(log_scale_errors),
        "confidence_accuracy": confidence_correct / max(sample_count, 1),
        "accepted_match_rate": accepted_matches / max(matched_count, 1),
        "fallback_match_rate": 1.0 - accepted_matches / max(matched_count, 1),
    }


def _rotation_6d_to_matrix_local(rotation_6d: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(rotation_6d[..., 0:3], dim=-1)
    second_raw = rotation_6d[..., 3:6]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


__all__ = [
    "SyntheticPoseConfig",
    "SyntheticPoseDataset",
    "evaluate_pose_model",
    "make_synthetic_pose_sample",
    "move_batch",
]
