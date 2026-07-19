"""BIM pipeline runner — callable generator that yields real-time progress.

Both the CLI (``scripts/run_pipeline.py``) and the Gradio UI call
``run_pipeline()`` directly.  No subprocess, no stdout parsing.

Usage (CLI)::

    from bim_recon.pipeline_runner import PipelineConfig, run_pipeline
    config = PipelineConfig(name="splat", elements=["door", "window"], ...)
    for msg, data in run_pipeline(config):
        print(msg)

Usage (Gradio)::

    for msg, data in run_pipeline(config):
        yield (msg, ...)  # update UI components
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import numpy as np

from bim_recon.element_config import get_element_config
from bim_recon.element_merger import (
    clip_opening_to_wall,
    clip_points_to_wall_span,
    merge_detections,
)
from bim_recon.falcon_client import FalconClient
from bim_recon.gs_scene import GSScene
from bim_recon.ring_scanner import render_ring_views, segment_ring_views, render_element_view
from bim_recon.virtual_scanner import VirtualScanner
from bim_recon.vlm_verifier import query_vlm
from bim_recon.wall_line_extractor import (
    extract_wall_lines, multi_height_scan, save_wall_lines_plot, wall_lines_to_json,
)
from bim_recon.wall_geometry import fit_short_corner_walls, merge_overlapping_walls
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from bim_recon.workflow_events import (
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowProgress,
    WorkflowWarning,
    progress_message,
)
from bim_recon.workflow_runtime import stream_workflow_sync

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """All parameters for one pipeline run."""
    name: str
    elements: list[str] = field(default_factory=lambda: ["door", "window"])
    skip_vlm: bool = False
    vlm_api_base: str = ""
    vlm_model: str = ""
    vlm_api_key: str = ""
    falcon_host: str = "127.0.0.1"
    falcon_port: int = 18390
    num_heights: int = 8
    snap_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_scene_files(data_dir: Path) -> tuple[Path, Path | None]:
    """Auto-discover PLY + feat.pt."""
    original = sorted(data_dir.glob("point_cloud_*.ply")) or sorted(data_dir.glob("*.ply"))
    if not original:
        raise FileNotFoundError(f"No PLY files in {data_dir}")
    ply = original[0]
    feat_candidates = sorted(data_dir.glob("*_feat.pt"))
    if not feat_candidates:
        out_dir = data_dir.parent.parent / "output" / data_dir.name
        feat_candidates = sorted(out_dir.glob("*_feat.pt"))
    feat = feat_candidates[0] if feat_candidates else None
    return ply, feat


def detect_coordinate_system(scene: GSScene, label_set: list[str] | None = None) -> dict:
    """Auto-detect up_axis, floor_z, ceiling_z, scan center."""
    has_sem = scene._has_feat and scene.semantic_querier is not None
    if has_sem:
        floor_c = np.array(scene.query_semantics("floor", mode="dominant", label_set=label_set)["centroid"])
        ceiling_c = np.array(scene.query_semantics("ceiling", mode="dominant", label_set=label_set)["centroid"])
        up_axis = int(np.argmax(np.abs(ceiling_c - floor_c)))
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(floor_c[up_axis])
        ceiling_z = float(ceiling_c[up_axis])
        if ceiling_z - floor_z < 1.5:
            coords = scene.means[:, up_axis].cpu().numpy()
            floor_z = float(np.percentile(coords, 1))
            ceiling_z = float(np.percentile(coords, 99))
        center = (float(floor_c[h_axes[0]]), float(floor_c[h_axes[1]]))
    else:
        means = scene.means.cpu().numpy()
        ranges = means.max(axis=0) - means.min(axis=0)
        up_axis = int(np.argmax(ranges))
        h_axes = [i for i in range(3) if i != up_axis]
        floor_z = float(np.percentile(means[:, up_axis], 1))
        ceiling_z = float(np.percentile(means[:, up_axis], 99))
        center = (float(np.median(means[:, h_axes[0]])), float(np.median(means[:, h_axes[1]])))
    return {"up_axis": up_axis, "h_axes": h_axes, "floor_z": floor_z,
            "ceiling_z": ceiling_z, "center": center}


def extract_walls(scans, center, out_dir, labels=None):
    """Extract wall lines from multi-height scan data."""
    wall_lines, wall_pts = extract_wall_lines(scans, labels=labels, center=center)
    output_json = wall_lines_to_json(wall_lines, scans, center)
    (out_dir / "wall_lines.json").write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    save_wall_lines_plot(wall_lines, wall_pts, center,
                         str(out_dir / "wall_lines_topdown.png"),
                         title=f"Walls ({len(wall_lines)} segments)")
    return [{"x1": wl.x1, "y1": wl.y1, "x2": wl.x2, "y2": wl.y2, "length": wl.length}
            for wl in wall_lines]


def snap_wall_endpoints(walls, threshold=0.5, min_length=0.001):
    """Snap endpoints, then replace short corner chamfers with ray intersections."""
    eps = [[w["x1"], w["y1"], w["x2"], w["y2"]] for w in walls]
    changed = True
    for _ in range(10):
        if not changed:
            break
        changed = False
        points = []
        for i, ep in enumerate(eps):
            points.append(("s", i, ep[0], ep[1]))
            points.append(("e", i, ep[2], ep[3]))
        snapped = set()
        for i, (t1, idx1, x1, y1) in enumerate(points):
            if i in snapped:
                continue
            group = [(t1, idx1, x1, y1)]
            for j, (t2, idx2, x2, y2) in enumerate(points[i + 1:], i + 1):
                if j in snapped:
                    continue
                if any(member[1] == idx2 for member in group):
                    continue
                dist = np.hypot(x1 - x2, y1 - y2)
                if 1e-6 < dist < threshold:
                    group.append((t2, idx2, x2, y2))
                    snapped.add(j)
            if len(group) > 1:
                changed = True
                avg_x = sum(p[2] for p in group) / len(group)
                avg_y = sum(p[3] for p in group) / len(group)
                for t, idx, _, _ in group:
                    if t == "s":
                        eps[idx][0] = avg_x
                        eps[idx][1] = avg_y
                    else:
                        eps[idx][2] = avg_x
                        eps[idx][3] = avg_y
    result = []
    for x1, y1, x2, y2 in eps:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= min_length:
            result.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "length": length,
            })
    fitted, _source_indices = fit_short_corner_walls(result)
    merged, _source_groups = merge_overlapping_walls(fitted)
    return merged


# ---------------------------------------------------------------------------
# Main pipeline generator
# ---------------------------------------------------------------------------

class _LoadScene(Event):
    pass


class _DetectCoordinates(Event):
    pass


class _ScanScene(Event):
    pass


class _ExtractWalls(Event):
    pass


class _RenderRing(Event):
    pass


class _SegmentViews(Event):
    pass


class _MergeDetections(Event):
    pass


class _VerifyElements(Event):
    pass


class _FinalizePipeline(Event):
    pass


@dataclass
class _PipelineRuntime:
    config: PipelineConfig
    out_dir: Path | None = None
    falcon: FalconClient | None = None
    ply_path: Path | None = None
    feat_path: Path | None = None
    scene: GSScene | None = None
    labels: list[str] = field(default_factory=list)
    element_labels: list[str] = field(default_factory=list)
    up_axis: int = 2
    floor_z: float = 0.0
    ceiling_z: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    scanner: VirtualScanner | None = None
    scan_3d: Any = None
    scans: list[Any] = field(default_factory=list)
    total_pts: int = 0
    walls_snapped: list[dict[str, Any]] = field(default_factory=list)
    ring_views: list[Any] = field(default_factory=list)
    ring_fov: float = 60.0
    view_dets: list[Any] = field(default_factory=list)
    merged_elements: list[Any] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)


class ReconstructionWorkflow(Workflow):
    """Typed, deterministic orchestration for the complete 3DGS→BIM pipeline."""

    workflow_name = "reconstruction"

    def __init__(self, config: PipelineConfig):
        super().__init__(timeout=None, verbose=False)
        self.state = _PipelineRuntime(config=config)

    def _emit(
        self,
        ctx: Context,
        stage: str,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ctx.write_event_to_stream(WorkflowProgress(
            workflow=self.workflow_name,
            stage=stage,
            message=message,
            current=current,
            total=total,
            payload=payload or {},
        ))

    @step
    async def prepare(
        self, ctx: Context, ev: StartEvent,
    ) -> _LoadScene | StopEvent:
        rt = self.state
        cfg = rt.config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rt.out_dir = ROOT / "output" / cfg.name / timestamp
        rt.out_dir.mkdir(parents=True, exist_ok=True)
        self._emit(ctx, "prepare", "Connecting to Falcon server...")
        rt.falcon = FalconClient(host=cfg.falcon_host, port=cfg.falcon_port)
        if not rt.falcon.health():
            message = (
                f"Falcon server unreachable at "
                f"{cfg.falcon_host}:{cfg.falcon_port}"
            )
            ctx.write_event_to_stream(WorkflowFailed(
                workflow=self.workflow_name,
                stage="prepare",
                message=message,
            ))
            return StopEvent(result={"error": message, "stage": "prepare"})
        return _LoadScene()

    @step
    async def load_scene(
        self, ctx: Context, ev: _LoadScene,
    ) -> _DetectCoordinates:
        rt = self.state
        cfg = rt.config
        data_dir = ROOT / "data" / cfg.name
        rt.ply_path, rt.feat_path = find_scene_files(data_dir)
        self._emit(ctx, "load_scene", f"Loading scene: {rt.ply_path.name}...")
        text_emb = ROOT / "data" / "bim_text_emb.pt"
        class_names = ROOT / "data" / "bim_class_names.json"
        warm: dict[str, str] = {}
        if text_emb.exists() and class_names.exists():
            warm = {
                "text_emb_path": str(text_emb),
                "class_names_path": str(class_names),
            }
        rt.scene = GSScene.from_ply(
            rt.ply_path,
            feat_path=rt.feat_path,
            **warm,
        )
        structural_labels = ["wall", "floor", "ceiling"]
        for element_type in cfg.elements:
            try:
                rt.element_labels.append(
                    get_element_config(element_type).semantic_label
                )
            except KeyError:
                ctx.write_event_to_stream(WorkflowWarning(
                    workflow=self.workflow_name,
                    stage="load_scene",
                    message=f"Unknown element type skipped: {element_type}",
                ))
        rt.labels = list(dict.fromkeys(structural_labels + rt.element_labels))
        self._emit(
            ctx,
            "load_scene",
            f"Loaded {rt.scene.num_gaussians} Gaussians",
        )
        return _DetectCoordinates()

    @step
    async def detect_coordinates(
        self, ctx: Context, ev: _DetectCoordinates,
    ) -> _ScanScene:
        rt = self.state
        assert rt.scene is not None
        self._emit(ctx, "coordinates", "Detecting coordinate system...")
        coords = detect_coordinate_system(rt.scene, label_set=rt.labels)
        rt.up_axis = coords["up_axis"]
        rt.floor_z = coords["floor_z"]
        rt.ceiling_z = coords["ceiling_z"]
        rt.center = tuple(coords["center"])
        rt.scanner = VirtualScanner(
            rt.scene,
            up_axis=rt.up_axis,
            labels=rt.labels,
        )
        self._emit(
            ctx,
            "coordinates",
            (
                f"up_axis={rt.up_axis} floor={rt.floor_z:.2f} "
                f"ceiling={rt.ceiling_z:.2f}"
            ),
        )
        return _ScanScene()

    @step
    async def scan_scene(
        self, ctx: Context, ev: _ScanScene,
    ) -> _ExtractWalls:
        rt = self.state
        cfg = rt.config
        assert rt.scanner is not None
        self._emit(ctx, "spherical_scan", "3D spherical scan...")
        rt.scan_3d = rt.scanner.scan_3d(
            rt.center,
            rt.floor_z,
            rt.ceiling_z,
            n_azimuth_views=12,
            n_elevation_bands=5,
            width=512,
            fov=45.0,
        )
        floor_ref, ceiling_ref = VirtualScanner.detect_floor_ceiling(
            rt.scan_3d,
            labels=rt.labels,
        )
        if (
            abs(floor_ref - rt.floor_z) < 0.5
            and abs(ceiling_ref - rt.ceiling_z) < 0.5
        ):
            rt.floor_z, rt.ceiling_z = floor_ref, ceiling_ref
        self._emit(
            ctx,
            "spherical_scan",
            (
                f"{len(rt.scan_3d.points_3d)} points, "
                f"floor={rt.floor_z:.2f} ceiling={rt.ceiling_z:.2f}"
            ),
        )
        self._emit(
            ctx,
            "horizontal_scan",
            f"Scanning {cfg.num_heights} heights...",
        )
        rt.scans = multi_height_scan(
            rt.scanner,
            rt.center,
            rt.floor_z,
            rt.ceiling_z,
            num_heights=cfg.num_heights,
            num_views=8,
            width=512,
        )
        rt.total_pts = sum(len(scan.angles_deg) for scan in rt.scans)
        self._emit(
            ctx,
            "horizontal_scan",
            f"Scan complete: {rt.total_pts} points",
        )
        return _ExtractWalls()

    @step
    async def extract_walls_step(
        self, ctx: Context, ev: _ExtractWalls,
    ) -> _RenderRing:
        rt = self.state
        assert rt.out_dir is not None
        self._emit(ctx, "walls", "Extracting walls...")
        walls = extract_walls(
            rt.scans,
            np.asarray(rt.center),
            rt.out_dir,
            labels=rt.labels,
        )
        rt.walls_snapped = snap_wall_endpoints(
            walls,
            rt.config.snap_threshold,
        )
        (rt.out_dir / "wall_lines_snapped.json").write_text(
            json.dumps(rt.walls_snapped, indent=2),
            encoding="utf-8",
        )
        self._emit(
            ctx,
            "walls",
            f"Extracted {len(rt.walls_snapped)} wall segments",
        )
        return _RenderRing()

    @step
    async def render_ring(
        self, ctx: Context, ev: _RenderRing,
    ) -> _SegmentViews:
        rt = self.state
        assert rt.scene is not None and rt.out_dir is not None
        mid_z = (rt.floor_z + rt.ceiling_z) / 2.0
        n_ring = 8
        self._emit(
            ctx,
            "ring_scan",
            f"Rendering {n_ring} views × {rt.ring_fov:.0f}°...",
        )
        rt.ring_views = render_ring_views(
            rt.scene,
            rt.center,
            mid_z,
            up_axis=rt.up_axis,
            n_views=n_ring,
            fov=rt.ring_fov,
            img_size=768,
        )
        ring_dir = rt.out_dir / "ring_views"
        ring_dir.mkdir(exist_ok=True)
        from PIL import Image as PILImage
        for view in rt.ring_views:
            PILImage.fromarray(view.image).save(
                ring_dir / f"view_{view.idx:02d}_{view.azimuth_deg:.0f}.png"
            )
        self._emit(
            ctx,
            "ring_scan",
            f"Rendered {len(rt.ring_views)} views",
        )
        return _SegmentViews()

    @step
    async def segment_views(
        self, ctx: Context, ev: _SegmentViews,
    ) -> _MergeDetections:
        rt = self.state
        assert rt.falcon is not None and rt.out_dir is not None
        self._emit(ctx, "segmentation", "Falcon segmentation per view...")
        rt.view_dets = segment_ring_views(
            rt.ring_views,
            rt.falcon,
            rt.element_labels,
            center_2d=rt.center,
            floor_z=rt.floor_z,
            ceiling_z=rt.ceiling_z,
            up_axis=rt.up_axis,
        )
        raw_json = [
            {
                "label": detection.label,
                "view": detection.view_idx,
                "azimuth": round(detection.azimuth_deg, 1),
                "world_x": round(detection.world_x, 3),
                "world_y": round(detection.world_y, 3),
                "width_m": round(detection.width_m, 3),
            }
            for detection in rt.view_dets
        ]
        (rt.out_dir / "ring_raw_detections.json").write_text(
            json.dumps(raw_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._emit(
            ctx,
            "segmentation",
            f"Detected {len(rt.view_dets)} raw objects across views",
        )
        return _MergeDetections()

    @step
    async def merge_detections_step(
        self, ctx: Context, ev: _MergeDetections,
    ) -> _VerifyElements:
        rt = self.state
        self._emit(ctx, "merge", "Merging detections wall-aware...")
        merge_input = [
            {
                "element_class": detection.label,
                "world_x": detection.world_x,
                "world_y": detection.world_y,
                "sill_height": detection.sill_height,
                "header_height": detection.header_height,
                "width_m": detection.width_m,
                "confidence": detection.centrality,
            }
            for detection in rt.view_dets
        ]
        rt.merged_elements = merge_detections(
            merge_input,
            rt.center,
            up_axis=rt.up_axis,
            merge_threshold=1.5,
            height_tolerance=0.5,
            walls=rt.walls_snapped,
        )
        for element in rt.merged_elements:
            point_sets = [
                rt.view_dets[index].mask_points_xy
                for index in element.source_indices
                if (
                    index < len(rt.view_dets)
                    and rt.view_dets[index].mask_points_xy is not None
                )
            ]
            combined = np.vstack(point_sets) if point_sets else None
            has_host_wall = (
                element.wall_idx is not None
                and 0 <= element.wall_idx < len(rt.walls_snapped)
            )
            is_hosted_opening = element.element_class.casefold() in {
                "door", "window",
            }
            if is_hosted_opening and has_host_wall:
                wall = rt.walls_snapped[element.wall_idx]
                wall_start = np.array(
                    [wall["x1"], wall["y1"]], dtype=np.float64,
                )
                wall_end = np.array(
                    [wall["x2"], wall["y2"]], dtype=np.float64,
                )
                wall_vector = wall_end - wall_start
                wall_length = float(np.linalg.norm(wall_vector))
                if wall_length > 1e-9 and combined is not None:
                    wall_direction = wall_vector / wall_length
                    projections = (combined - wall_start) @ wall_direction
                    observed_start = float(np.percentile(projections, 3))
                    observed_end = float(np.percentile(projections, 97))
                    element.width_m = max(
                        element.width_m,
                        observed_end - observed_start,
                    )
                    observed_center = wall_start + (
                        (observed_start + observed_end) / 2.0
                    ) * wall_direction
                    element.world_x = float(observed_center[0])
                    element.world_y = float(observed_center[1])

                (
                    element.world_x,
                    element.world_y,
                    element.width_m,
                    opening_start,
                    opening_end,
                ) = clip_opening_to_wall(
                    element.world_x,
                    element.world_y,
                    element.width_m,
                    wall,
                )
                offset_x = element.world_x - float(rt.center[0])
                offset_y = element.world_y - float(rt.center[1])
                element.theta_center = (
                    math.degrees(math.atan2(offset_y, offset_x)) % 360.0
                )
                element.r_mean = math.hypot(offset_x, offset_y)

                if wall_length > 1e-9:
                    for source_index in element.source_indices:
                        if source_index >= len(rt.view_dets):
                            continue
                        points = rt.view_dets[source_index].mask_points_xy
                        if points is None:
                            continue
                        rt.view_dets[source_index].mask_points_xy = (
                            clip_points_to_wall_span(
                                points,
                                wall,
                                opening_start,
                                opening_end,
                            )
                        )
                continue

            if combined is None or len(point_sets) < 2:
                continue
            if has_host_wall:
                wall = rt.walls_snapped[element.wall_idx]
                wall_start = np.array([wall["x1"], wall["y1"]])
                wall_end = np.array([wall["x2"], wall["y2"]])
                wall_direction = wall_end - wall_start
                wall_length = np.linalg.norm(wall_direction)
                if wall_length > 1e-6:
                    wall_direction /= wall_length
                    projections = (combined - wall_start) @ wall_direction
                    true_width = float(
                        np.percentile(projections, 97)
                        - np.percentile(projections, 3)
                    )
                    if true_width > element.width_m:
                        element.width_m = true_width
            element.world_x = float(np.median(combined[:, 0]))
            element.world_y = float(np.median(combined[:, 1]))
        self._emit(
            ctx,
            "merge",
            (
                f"{len(rt.view_dets)} raw → "
                f"{len(rt.merged_elements)} unique"
            ),
        )
        return _VerifyElements()

    @step
    async def verify_elements(
        self, ctx: Context, ev: _VerifyElements,
    ) -> _FinalizePipeline:
        rt = self.state
        cfg = rt.config
        assert rt.scene is not None and rt.out_dir is not None
        verify_dir = rt.out_dir / "verify_merged"
        verify_dir.mkdir(exist_ok=True)
        mid_z = (rt.floor_z + rt.ceiling_z) / 2.0
        total = len(rt.merged_elements)
        from PIL import Image as PILImage

        for index, element in enumerate(rt.merged_elements):
            self._emit(
                ctx,
                "vlm_verification",
                f"{element.element_class} θ={element.theta_center:.1f}° rendering...",
                current=index + 1,
                total=total,
            )
            element_view = render_element_view(
                rt.scene,
                element.world_x,
                element.world_y,
                width_m=element.width_m,
                height_m=max(element.element_height, 0.5),
                mid_z=mid_z,
                center_2d=rt.center,
                up_axis=rt.up_axis,
                img_size=768,
                margin=0.5,
            )
            if element_view is None:
                ctx.write_event_to_stream(WorkflowWarning(
                    workflow=self.workflow_name,
                    stage="vlm_verification",
                    message=f"Render failed for {element.element_class} #{index}",
                ))
                continue
            image_name = f"merged_{index}_{element.element_class}.png"
            PILImage.fromarray(element_view.image).save(verify_dir / image_name)
            if cfg.skip_vlm:
                vlm_ok, vlm_response = True, "skipped"
            else:
                try:
                    hint = get_element_config(element.element_class).vlm_hint
                except KeyError:
                    hint = element.element_class
                prompt = (
                    f"Look at this image carefully. Is there {hint} in this image? "
                    "Answer with YES or NO only."
                )
                try:
                    vlm_response = query_vlm(
                        str(verify_dir / image_name),
                        prompt,
                        cfg.vlm_api_base,
                        cfg.vlm_model,
                        cfg.vlm_api_key,
                        timeout=30,
                    )
                except Exception as exc:
                    vlm_response = ""
                    ctx.write_event_to_stream(WorkflowWarning(
                        workflow=self.workflow_name,
                        stage="vlm_verification",
                        message=f"VLM request failed: {exc}",
                    ))
                response_lower = vlm_response.lower().strip()
                vlm_ok = any(
                    keyword in response_lower
                    for keyword in (
                        "yes", "是", "有", "确认", "confir", "correct",
                        "true", "indeed", "确实", "存在",
                    )
                )
                if (
                    not vlm_ok
                    and element.element_class in response_lower
                ):
                    vlm_ok = True
            rt.verification_results.append({
                **element.to_dict(),
                "image_path": image_name,
                "vlm_response": vlm_response,
                "fov_deg": element_view.fov_deg,
                "vlm_confirmed": vlm_ok,
            })
            self._emit(
                ctx,
                "vlm_verification",
                "CONFIRMED" if vlm_ok else "REJECTED",
                current=index + 1,
                total=total,
            )
        return _FinalizePipeline()

    @step
    async def finalize(
        self, ctx: Context, ev: _FinalizePipeline,
    ) -> StopEvent:
        rt = self.state
        cfg = rt.config
        assert (
            rt.scene is not None
            and rt.out_dir is not None
            and rt.ply_path is not None
        )
        self._emit(ctx, "finalize", "Centering room at origin...")
        center_x, center_y = float(rt.center[0]), float(rt.center[1])
        for wall in rt.walls_snapped:
            wall["x1"] -= center_x
            wall["y1"] -= center_y
            wall["x2"] -= center_x
            wall["y2"] -= center_y
        for result in rt.verification_results:
            result["world_x"] -= center_x
            result["world_y"] -= center_y
        for element in rt.merged_elements:
            element.world_x -= center_x
            element.world_y -= center_y
        rt.ceiling_z -= rt.floor_z
        rt.floor_z = 0.0
        rt.center = (0.0, 0.0)
        (rt.out_dir / "wall_lines_snapped.json").write_text(
            json.dumps(rt.walls_snapped, indent=2),
            encoding="utf-8",
        )

        all_results: dict[str, dict[str, Any]] = {}
        for element_type in cfg.elements:
            try:
                element_config = get_element_config(element_type)
            except KeyError:
                continue
            entries = [
                result for result in rt.verification_results
                if result["element_class"] == element_config.semantic_label
            ]
            confirmed_entries = [
                result for result in entries
                if result.get("vlm_confirmed")
            ]
            all_results[element_type] = {
                "total_candidates": len(rt.view_dets),
                "after_prefilter": len(rt.merged_elements),
                "confirmed": len(confirmed_entries),
                "results": [
                    {
                        "confirmed": result.get("vlm_confirmed", False),
                        "candidate": {
                            "element_class": result["element_class"],
                            "world_x": result["world_x"],
                            "world_y": result["world_y"],
                            "theta_center": result["theta_center"],
                            "r_mean": result["r_mean"],
                            "width_m": result["width_m"],
                            "wall_idx": result.get("wall_idx"),
                        },
                        "height_detection": {
                            "sill_height": result["sill_height"],
                            "header_height": result["header_height"],
                            "element_height": result["element_height"],
                            "width_m": result["width_m"],
                        },
                        "image_path": result["image_path"],
                        "vlm_response": result.get("vlm_response", ""),
                    }
                    for result in entries
                ],
            }
            element_json = {
                "scene": cfg.name,
                "element": element_type,
                "ply_used": rt.ply_path.name,
                "vlm_model": cfg.vlm_model if not cfg.skip_vlm else None,
                **all_results[element_type],
            }
            (rt.out_dir / element_config.output_json_name).write_text(
                json.dumps(element_json, indent=2),
                encoding="utf-8",
            )

        merged_json = {
            "raw_count": len(rt.view_dets),
            "merged_count": len(rt.merged_elements),
            "confirmed_count": sum(
                1 for result in rt.verification_results
                if result.get("vlm_confirmed")
            ),
            "total_vlm_results": len(rt.verification_results),
            "merged": [element.to_dict() for element in rt.merged_elements],
            "confirmed": rt.verification_results,
        }
        (rt.out_dir / "merged_elements.json").write_text(
            json.dumps(merged_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._emit(ctx, "radars", "Generating radar plots...")
        _generate_radars(
            rt.out_dir,
            rt.view_dets,
            rt.merged_elements,
            rt.walls_snapped,
            rt.center,
            rt.ring_views,
            rt.ring_fov,
        )
        report = {
            "scene": cfg.name,
            "ply": rt.ply_path.name,
            "num_gaussians": rt.scene.num_gaussians,
            "coordinate_system": {
                "up_axis": rt.up_axis,
                "floor_z": rt.floor_z,
                "ceiling_z": rt.ceiling_z,
                "center": list(rt.center),
            },
            "scan": {
                "num_heights": cfg.num_heights,
                "total_points": rt.total_pts,
                "scan_3d_points": len(rt.scan_3d.points_3d),
            },
            "walls": {"count": len(rt.walls_snapped)},
            "elements": all_results,
            "merged_elements": {
                "count": len(rt.merged_elements),
                "confirmed": sum(
                    1 for result in rt.verification_results
                    if result.get("vlm_confirmed")
                ),
            },
            "vlm_model": cfg.vlm_model if not cfg.skip_vlm else None,
        }
        (rt.out_dir / "pipeline_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        result = {
            "out_dir": str(rt.out_dir),
            "walls": rt.walls_snapped,
            "confirmed_count": sum(
                item.get("confirmed", 0) for item in all_results.values()
            ),
            "report": report,
        }
        ctx.write_event_to_stream(WorkflowCompleted(
            workflow=self.workflow_name,
            result=result,
        ))
        return StopEvent(result=result)


def run_pipeline(
    config: PipelineConfig,
) -> Generator[tuple[str, dict], None, None]:
    """Run :class:`ReconstructionWorkflow` through the legacy sync stream API."""
    workflow = ReconstructionWorkflow(config)
    try:
        for event in stream_workflow_sync(workflow):
            if isinstance(event, (WorkflowProgress, WorkflowWarning, WorkflowFailed)):
                yield progress_message(event), event.payload
            elif isinstance(event, WorkflowCompleted):
                yield "Pipeline complete!", event.result
    except Exception as exc:
        yield f"ERROR [workflow] {exc}", {}


# ---------------------------------------------------------------------------
# Radar plot generation
# ---------------------------------------------------------------------------

def _generate_radars(out_dir, view_dets, merged_elements, walls_snapped,
                     center, ring_views, ring_fov):
    """Generate Cartesian top-down radar plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    cx, cy = float(center[0]), float(center[1])

    def _draw_walls(ax):
        for wl in walls_snapped:
            x1, y1 = wl["x1"] - cx, wl["y1"] - cy
            x2, y2 = wl["x2"] - cx, wl["y2"] - cy
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, zorder=5, alpha=0.7)

    def _draw_pca_line(ax, pts, color, label=None):
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=3, alpha=0.4, zorder=7)
        if len(pts) > 3:
            centered = pts - pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[0]
            c = pts.mean(axis=0)
            p1 = c + Vt[0] * proj.min()
            p2 = c + Vt[0] * proj.max()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color,
                    linewidth=2.5, zorder=8, label=label)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color=color, markersize=5, zorder=9)

    max_r = max(
        max((abs(wl["x1"] - cx) for wl in walls_snapped), default=5),
        max((abs(wl["x2"] - cx) for wl in walls_snapped), default=5), 5)

    # Radar 1: Raw detections
    if view_dets:
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        _draw_walls(ax)
        lc = {"door": "red", "window": "blue", "column": "gray"}
        for di, d in enumerate(view_dets):
            color = lc.get(d.label, "green")
            if d.mask_points_xy is not None:
                _draw_pca_line(ax, d.mask_points_xy - np.array([cx, cy]), color,
                               label=f"{d.label} v{d.view_idx}" if di < 12 else None)
            else:
                ax.scatter(d.world_x - cx, d.world_y - cy, c=color, s=30, zorder=7)
        ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
        ax.set_aspect("equal")
        ax.set_xlim(-max_r - 1, max_r + 1)
        ax.set_ylim(-max_r - 1, max_r + 1)
        ax.set_title(f"Ring Raw Detections ({len(view_dets)} masks)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.savefig(str(out_dir / "radar_ring_raw.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Radar 2: Merged elements
    if merged_elements:
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        _draw_walls(ax)
        for v in ring_views:
            az = math.radians(v.azimuth_deg)
            hfov = math.radians(ring_fov / 2)
            max_d = float(np.percentile(v.depth[v.depth > 0.1], 90)) if (v.depth > 0.1).any() else 5.0
            ax.add_patch(Wedge((0, 0), max_d, math.degrees(az - hfov),
                               math.degrees(az + hfov), alpha=0.04,
                               color="lightblue", zorder=1))
        palette = plt.cm.Set1(np.linspace(0, 1, max(len(merged_elements), 1)))
        for mi, me in enumerate(merged_elements):
            color = palette[mi % len(palette)]
            src_pts = []
            for si in me.source_indices:
                if si < len(view_dets) and view_dets[si].mask_points_xy is not None:
                    src_pts.append(view_dets[si].mask_points_xy - np.array([cx, cy]))
            if src_pts:
                _draw_pca_line(ax, np.vstack(src_pts), color,
                               label=f"{me.element_class} ({me.num_sources} masks)")
            else:
                ax.scatter(me.world_x - cx, me.world_y - cy, c=[color],
                           s=100, marker="*", zorder=8)
        ax.plot(0, 0, "k^", markersize=12, zorder=10, label="Camera")
        ax.set_aspect("equal")
        ax.set_xlim(-max_r - 1, max_r + 1)
        ax.set_ylim(-max_r - 1, max_r + 1)
        ax.set_title(f"Merged Elements ({len(merged_elements)} unique)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(str(out_dir / "radar_merged.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
