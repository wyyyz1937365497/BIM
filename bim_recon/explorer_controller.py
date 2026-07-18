"""Deterministic controller for B-class 3DGS exploration and detection."""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from bim_recon.falcon_client import FalconClient
from bim_recon.gs_scene import CameraPose, GSScene, look_at_pose


@dataclass(frozen=True, slots=True)
class ExplorerCamera:
    eye: tuple[float, float, float]
    yaw_degrees: float = 0.0
    fov: float = 60.0
    look_at: tuple[float, float, float] | None = None


@dataclass
class ExplorerController:
    """Own the heavy scene/Falcon resources and expose explicit scan operations."""

    scene: GSScene
    falcon: FalconClient
    output_dir: Path
    width: int = 1024
    height: int = 768
    duplicate_distance: float = 0.3
    eye: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw: float = 0.0
    fov: float = 60.0
    up_axis: int = 2
    view_counter: int = 0
    object_counter: int = 0
    found: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        ply_path: Path,
        feat_path: Path | None,
        falcon_host: str,
        falcon_port: int,
        output_dir: Path,
        *,
        width: int = 1024,
        height: int = 768,
    ) -> "ExplorerController":
        scene = GSScene.from_ply(ply_path, feat_path=feat_path)
        falcon = FalconClient(host=falcon_host, port=falcon_port, timeout=300)
        if not falcon.health():
            raise RuntimeError(
                f"Falcon server unreachable at {falcon_host}:{falcon_port}"
            )
        return cls(
            scene=scene,
            falcon=falcon,
            output_dir=output_dir,
            width=width,
            height=height,
        )

    def initialize(self, camera: ExplorerCamera) -> str:
        self.eye = [float(value) for value in camera.eye]
        self.fov = camera.fov
        means = self.scene.means
        quantile_indices = torch.linspace(
            0,
            means.shape[0] - 1,
            min(means.shape[0], 100_000),
            device=means.device,
        ).long()
        sample = means[quantile_indices]
        extents = (
            torch.quantile(sample, 0.95, dim=0)
            - torch.quantile(sample, 0.05, dim=0)
        )
        self.up_axis = int(torch.argmin(extents).item())
        if camera.look_at is None:
            self.yaw = math.radians(camera.yaw_degrees)
        else:
            first, second = _horizontal_axes(self.up_axis)
            delta_first = camera.look_at[first] - camera.eye[first]
            delta_second = camera.look_at[second] - camera.eye[second]
            self.yaw = math.atan2(delta_second, delta_first)
        png, _render, _pose = self.render_current()
        return self.save_view(png, "_initial")

    def render_current(self) -> tuple[bytes, Any, CameraPose]:
        target = self._target_from_yaw()
        pose = look_at_pose(
            eye=tuple(self.eye),
            target=tuple(target),
            up=self._up_vector(),
        )
        result = self.scene.render(pose, self.width, self.height, self.fov)
        colors = np.clip(result.colors * 255, 0, 255).astype(np.uint8)
        return _encode_png(colors), result, pose

    def scan_current(self, labels: Sequence[str]) -> dict[str, Any]:
        """Render once, query Falcon once per label, and tag every new detection."""
        png, render, pose = self.render_current()
        source_path = self.save_view(png, "_scan")
        image = PILImage.open(io.BytesIO(png)).convert("RGB")
        detections: list[dict[str, Any]] = []
        tagged: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        for raw_label in labels:
            label = raw_label.strip()
            if not label:
                continue
            for detection in self.falcon.segment(image, label, task="detection"):
                item = {
                    "label": label,
                    "bbox": detection.bbox,
                    "mask_area_ratio": detection.mask_area_ratio,
                }
                detections.append(item)
                result = self._tag_detection(
                    label,
                    detection.bbox,
                    png,
                    render,
                    pose,
                )
                if result["status"] == "tagged":
                    tagged.append(result)
                else:
                    duplicates.append(result)
        return {
            "view_path": source_path,
            "yaw_degrees": round(math.degrees(self.yaw) % 360.0, 1),
            "detections": detections,
            "tagged": tagged,
            "duplicates": duplicates,
        }

    def turn(self, yaw_degrees: float) -> None:
        self.yaw += math.radians(float(yaw_degrees))

    def status(self) -> dict[str, Any]:
        means = self.scene.means
        bounds_min = means.min(dim=0).values.detach().cpu().tolist()
        bounds_max = means.max(dim=0).values.detach().cpu().tolist()
        return {
            "camera": {
                "eye": [round(value, 3) for value in self.eye],
                "yaw_degrees": round(math.degrees(self.yaw) % 360.0, 1),
                "fov": self.fov,
                "up_axis": self.up_axis,
            },
            "scene_bounds": {"min": bounds_min, "max": bounds_max},
            "found_count": len(self.found),
        }

    def save_view(self, png: bytes, suffix: str = "") -> str:
        self.view_counter += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"view_{self.view_counter:03d}{suffix}.png"
        path.write_bytes(png)
        return str(path)

    def persist(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "found_objects.json"
        path.write_text(
            json.dumps(self.found, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _tag_detection(
        self,
        label: str,
        bbox: dict[str, float],
        png: bytes,
        render: Any,
        pose: CameraPose,
    ) -> dict[str, Any]:
        position = self._estimate_3d(bbox, pose)
        if position is None:
            position = self._depth_fallback(bbox, render, pose)
        for existing in self.found:
            if existing["label"] != label:
                continue
            distance = float(np.linalg.norm(
                np.asarray(existing["position_3d"], dtype=float)
                - np.asarray(position, dtype=float)
            ))
            if distance < self.duplicate_distance:
                return {
                    "status": "duplicate",
                    "duplicate_of": existing["id"],
                    "label": label,
                    "position_3d": position,
                }
        self.object_counter += 1
        object_id = f"obj_{self.object_counter:03d}"
        best_view = self.save_view(png, f"_{label}_{object_id}")
        tagged = {
            "status": "tagged",
            "id": object_id,
            "label": label,
            "position_3d": [round(float(value), 4) for value in position],
            "best_view": best_view,
            "best_pose": {
                "eye": [round(value, 4) for value in self.eye],
                "yaw_degrees": round(math.degrees(self.yaw) % 360.0, 2),
                "fov": self.fov,
            },
            "bbox": bbox,
            "trellis_status": "pending_approval",
        }
        self.found.append(tagged)
        self.persist()
        return tagged

    def _estimate_3d(
        self,
        bbox: dict[str, float],
        pose: CameraPose,
    ) -> list[float] | None:
        mask = _bbox_mask(bbox, self.width, self.height)
        try:
            gaussian_ids = self.scene.select_by_mask(
                pose,
                mask,
                self.width,
                self.height,
                self.fov,
            )
        except Exception:
            return None
        if len(gaussian_ids) == 0:
            return None
        ids = torch.as_tensor(gaussian_ids, device=self.scene.device)
        return self.scene.means[ids].median(dim=0).values.detach().cpu().tolist()

    def _depth_fallback(
        self,
        bbox: dict[str, float],
        render: Any,
        pose: CameraPose,
    ) -> list[float]:
        pixel_x = min(
            self.width - 1,
            max(0, int((bbox["x"] + bbox["w"] / 2.0) * self.width)),
        )
        pixel_y = min(
            self.height - 1,
            max(0, int((bbox["y"] + bbox["h"] / 2.0) * self.height)),
        )
        depth = float(render.depth[pixel_y, pixel_x])
        if depth <= 0:
            return list(self.eye)
        focal = 0.5 * self.width / math.tan(0.5 * math.radians(self.fov))
        camera_point = np.array([
            (pixel_x - self.width / 2.0) / focal * depth,
            (pixel_y - self.height / 2.0) / focal * depth,
            depth,
        ])
        camera_to_world = pose.to_viewmat()[:3, :3].T
        return (
            camera_to_world @ camera_point + np.asarray(self.eye)
        ).tolist()

    def _target_from_yaw(self) -> list[float]:
        first, second = _horizontal_axes(self.up_axis)
        target = list(self.eye)
        target[first] += math.cos(self.yaw)
        target[second] += math.sin(self.yaw)
        return target

    def _up_vector(self) -> tuple[float, float, float]:
        vector = [0.0, 0.0, 0.0]
        vector[self.up_axis] = 1.0
        return tuple(vector)


def _horizontal_axes(up_axis: int) -> tuple[int, int]:
    if up_axis == 2:
        return 0, 1
    if up_axis == 1:
        return 0, 2
    return 1, 2


def _bbox_mask(
    bbox: dict[str, float],
    width: int,
    height: int,
) -> np.ndarray:
    x0 = max(0, int(bbox["x"] * width))
    y0 = max(0, int(bbox["y"] * height))
    x1 = min(width, int((bbox["x"] + bbox["w"]) * width))
    y1 = min(height, int((bbox["y"] + bbox["h"]) * height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _encode_png(colors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(colors, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
