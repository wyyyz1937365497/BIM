"""Multi-view ring scanner with overlapping views + per-view segmentation.

Designed flow (per user spec)::

    1.  Render N overlapping views around the room (each ~30-45° FOV).
    2.  Run Falcon open-vocabulary segmentation on EACH view.
    3.  Back-project every mask/pixel hit to world XY via rendered depth.
    4.  Merge all detections in polar (θ, r) space — a large window split
        across 2-3 adjacent views becomes ONE element.
    5.  For each merged element, pick the single existing view that captures
        it most centrally, crop tightly, and send to VLM for confirmation.

This module provides the rendering, back-projection, and view-selection
plumbing.  Falcon segmentation and VLM calls are injected as callables so
the module stays testable without those services.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RingView:
    """One rendered view in the ring scan."""

    idx: int                         # 0-based view index
    azimuth_deg: float               # camera azimuth (degrees, 0=+x)
    image: np.ndarray                # (H, W, 3) uint8 RGB
    depth: np.ndarray                # (H, W) float32 metric depth
    alpha: np.ndarray                # (H, W) float32
    # Camera params for pixel→world back-projection
    eye: np.ndarray                  # (3,)
    forward: np.ndarray              # (3,) unit forward
    right: np.ndarray                # (3,) unit right
    up: np.ndarray                   # (3,) unit up (world up)
    fx: float
    fy: float
    cx_pix: float
    cy_pix: float
    width: int
    height: int
    fov_deg: float

    # Cached Falcon detections (filled by segment_ring_views)
    detections: List[dict] = field(default_factory=list)


@dataclass
class ViewDetection:
    """A single detection from one ring view, with world coordinates."""

    view_idx: int
    azimuth_deg: float               # view's azimuth
    label: str
    # Normalised bbox [0,1]
    bbox: Dict[str, float]           # {x, y, w, h} normalised centre + extent
    # World position (from depth back-projection of mask centroid)
    world_x: float
    world_y: float
    world_z: float
    # Metric extents
    sill_height: float
    header_height: float
    element_height: float
    width_m: float
    depth_m: float
    # Mask pixels in image coords (for cropping later)
    pixel_x_min: int
    pixel_x_max: int
    pixel_y_min: int
    pixel_y_max: int
    # Quality: how central the detection is in this view (0-1, higher=better)
    centrality: float
    # Full backprojected mask point cloud in world XY (for radar visualization)
    mask_points_xy: Optional[np.ndarray] = None  # (K, 2)


# ---------------------------------------------------------------------------
# Ring rendering
# ---------------------------------------------------------------------------

def render_ring_views(
    scene,
    center_2d: Tuple[float, float],
    mid_z: float,
    up_axis: int = 2,
    n_views: int = 12,
    fov: float = 45.0,
    img_size: int = 768,
) -> List[RingView]:
    """Render *n_views* overlapping cameras covering 360°.

    Each camera is placed at ``(center_2d[0], center_2d[1], mid_z)`` and
    looks outward at equally-spaced azimuth angles.  With 12 views and
    45° FOV, each point is covered by ≥ 1.5 views on average, ensuring
    overlap for robust merging.

    Args:
        scene: GSScene with ``render()`` and ``render_validated()``.
        center_2d: (x, y) room centre in world horizontal plane.
        mid_z: Camera height (world up-axis coordinate).
        up_axis: Which axis is vertical.
        n_views: Number of azimuth views (≥ 8 for full 360° coverage).
        fov: Horizontal FOV per view in degrees.
        img_size: Output image size (square).

    Returns:
        List of RingView, one per azimuth.
    """
    from bim_recon.gs_scene import look_at_pose, GSScene

    h_axes = [i for i in range(3) if i != up_axis]
    cx, cy = float(center_2d[0]), float(center_2d[1])
    az_step = 360.0 / n_views

    views: List[RingView] = []
    for i in range(n_views):
        az = i * az_step
        az_rad = math.radians(az)

        eye = [0.0, 0.0, 0.0]
        eye[h_axes[0]] = cx
        eye[h_axes[1]] = cy
        eye[up_axis] = mid_z

        target = [0.0, 0.0, 0.0]
        target[h_axes[0]] = cx + math.cos(az_rad)
        target[h_axes[1]] = cy + math.sin(az_rad)
        target[up_axis] = mid_z

        up = [0.0, 0.0, 0.0]
        up[up_axis] = 1.0

        pose = look_at_pose(tuple(eye), tuple(target), tuple(up))
        result, reason, _ = GSScene.render_validated(
            scene, pose, img_size, img_size, fov)

        if result is None:
            views.append(None)  # type: ignore  # placeholder; caller skips
            continue

        # Camera basis vectors for back-projection
        viewmat = pose.to_viewmat()
        R_c2w = viewmat[:3, :3].T.astype(np.float64)
        forward = R_c2w[:, 2]
        right = R_c2w[:, 0]
        up_vec = R_c2w[:, 1]

        fx = 0.5 * img_size / math.tan(0.5 * math.radians(fov))

        views.append(RingView(
            idx=i,
            azimuth_deg=az,
            image=(result.colors * 255).clip(0, 255).astype(np.uint8),
            depth=result.depth,
            alpha=result.alpha,
            eye=np.array(eye, dtype=np.float64),
            forward=forward,
            right=right,
            up=up_vec,
            fx=fx,
            fy=fx,
            cx_pix=img_size / 2.0,
            cy_pix=img_size / 2.0,
            width=img_size,
            height=img_size,
            fov_deg=fov,
        ))

    # Filter out None placeholders
    return [v for v in views if v is not None]


# ---------------------------------------------------------------------------
# Per-view segmentation + back-projection
# ---------------------------------------------------------------------------

def _pixel_to_world(
    view: RingView, u: float, v: float, depth: float,
    h_axes: List[int], up_axis: int,
) -> np.ndarray:
    """Back-project pixel (u, v) at given depth to world XYZ."""
    x_cam = (u - view.cx_pix) / view.fx * depth
    y_cam = (v - view.cy_pix) / view.fy * depth
    P_cam = np.array([x_cam, y_cam, depth])
    R_c2w = np.stack([view.right, view.up, view.forward], axis=1)
    return R_c2w @ P_cam + view.eye


def segment_ring_views(
    views: List[RingView],
    falcon_client,
    query_labels: Sequence[str],
    center_2d: Tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    up_axis: int = 2,
) -> List[ViewDetection]:
    """Run Falcon segmentation on each view and back-project to world coords.

    For each view, calls ``falcon_client.segment(image, label)`` for each
    label in *query_labels*.  Each detection mask is back-projected using
    the view's depth buffer to obtain world XYZ and metric extents.

    Args:
        views: List of RingView from ``render_ring_views()``.
        falcon_client: Object with ``segment(image, text, task)`` method
            returning a list of detection objects with ``mask_bbox``,
            ``bbox``, ``mask_rle``, ``mask_size``.
        query_labels: Element labels to search for (e.g. ["door", "window"]).
        center_2d, floor_z, ceiling_z, up_axis: Room geometry.

    Returns:
        List of ViewDetection, one per raw detection across all views.
    """
    import base64 as _b64
    h_axes = [i for i in range(3) if i != up_axis]
    all_dets: List[ViewDetection] = []

    for view in views:
        from PIL import Image as _PIL
        img = _PIL.fromarray(view.image)

        for label in query_labels:
            try:
                raw_dets = falcon_client.segment(img, label, task="segmentation")
            except Exception:
                continue

            for det in raw_dets:
                mb = det.mask_bbox or det.bbox
                W, H = view.width, view.height

                # Decode mask if available
                mask_arr = None
                if det.mask_rle and det.mask_size:
                    try:
                        from pycocotools import mask as mask_utils
                        counts = _b64.b64decode(det.mask_rle)
                        mask_arr = mask_utils.decode(
                            {"counts": counts, "size": det.mask_size})
                        mh, mw = det.mask_size
                        if mh != H or mw != W:
                            mask_arr = np.array(
                                _PIL.fromarray(mask_arr).resize((W, H), _PIL.NEAREST))
                    except Exception:
                        mask_arr = None

                # Collect pixel coordinates
                if mask_arr is not None:
                    ys, xs = np.where(mask_arr)
                    if len(ys) < 5:
                        continue
                else:
                    # Use bbox to define pixel region
                    cx_n = max(0.0, min(1.0, mb["x"]))
                    cy_n = max(0.0, min(1.0, mb["y"]))
                    hw_n = max(0.0, min(1.0, mb["w"] / 2))
                    hh_n = max(0.0, min(1.0, mb["h"] / 2))
                    xs = np.arange(
                        max(0, int((cx_n - hw_n) * W)),
                        min(W, int((cx_n + hw_n) * W) + 1),
                    )
                    ys_full = np.arange(
                        max(0, int((cy_n - hh_n) * H)),
                        min(H, int((cy_n + hh_n) * H) + 1),
                    )
                    xs, ys = np.meshgrid(xs, ys_full)
                    xs, ys = xs.ravel(), ys.ravel()

                # Back-project mask pixels via depth
                depths = view.depth[ys, xs].astype(np.float64)
                valid = depths > 0.1
                if valid.sum() < 5:
                    continue
                xs_v, ys_v, ds_v = xs[valid], ys[valid], depths[valid]

                worlds = np.stack([
                    _pixel_to_world(view, float(u), float(v), float(d),
                                    h_axes, up_axis)
                    for u, v, d in zip(xs_v, ys_v, ds_v)
                ])  # (K, 3)

                # Metric extents
                wz = worlds[:, up_axis]
                sill_h = float(np.percentile(wz, 5)) - floor_z
                header_h = float(np.percentile(wz, 95)) - floor_z
                wh0 = worlds[:, h_axes[0]]
                wh1 = worlds[:, h_axes[1]]
                width_m = float(np.percentile(wh0, 95) - np.percentile(wh0, 5) +
                                np.percentile(wh1, 95) - np.percentile(wh1, 5)) / 2
                width_m = max(width_m, 0.1)
                depth_m = float(np.percentile(ds_v, 95) - np.percentile(ds_v, 5))

                # World centroid
                centroid = np.median(worlds, axis=0)

                # Centrality: how close the detection centre is to image centre
                det_cx = float(np.mean(xs_v)) / W
                det_cy = float(np.mean(ys_v)) / H
                centrality = 1.0 - math.hypot(det_cx - 0.5, det_cy - 0.5) * 2.0
                centrality = max(0.0, min(1.0, centrality))

                # Pixel bounds for later cropping
                px_min = int(xs_v.min())
                px_max = int(xs_v.max())
                py_min = int(ys_v.min())
                py_max = int(ys_v.max())

                # Normalised bbox
                bbox = {
                    "x": float(np.mean(xs_v)) / W,
                    "y": float(np.mean(ys_v)) / H,
                    "w": float(px_max - px_min) / W,
                    "h": float(py_max - py_min) / H,
                }

                all_dets.append(ViewDetection(
                    view_idx=view.idx,
                    azimuth_deg=view.azimuth_deg,
                    label=label,
                    bbox=bbox,
                    world_x=float(centroid[h_axes[0]]),
                    world_y=float(centroid[h_axes[1]]),
                    world_z=float(centroid[up_axis]),
                    sill_height=max(0.0, sill_h),
                    header_height=max(sill_h, header_h),
                    element_height=max(0.0, header_h - sill_h),
                    width_m=width_m,
                    depth_m=depth_m,
                    pixel_x_min=px_min,
                    pixel_x_max=px_max,
                    pixel_y_min=py_min,
                    pixel_y_max=py_max,
                    centrality=centrality,
                    mask_points_xy=worlds[:, h_axes].copy(),
                ))

                # Also store in the view for later crop access
                view.detections.append({
                    "label": label,
                    "view_idx": view.idx,
                    "bbox": bbox,
                    "px_min": px_min, "px_max": px_max,
                    "py_min": py_min, "py_max": py_max,
                    "centrality": centrality,
                    "world_x": float(centroid[h_axes[0]]),
                    "world_y": float(centroid[h_axes[1]]),
                })

    return all_dets


# ---------------------------------------------------------------------------
# Best-view selection for VLM
# ---------------------------------------------------------------------------

def render_element_view(
    scene,
    world_x: float,
    world_y: float,
    width_m: float,
    height_m: float,
    mid_z: float,
    center_2d: Tuple[float, float],
    up_axis: int = 2,
    img_size: int = 768,
    margin: float = 0.5,
) -> Optional[RingView]:
    """Render a targeted view that fully captures a specific element.

    The camera is placed at the room centre, looking directly at
    (world_x, world_y).  The FOV is **automatically adjusted** so that
    the element's width and height fit entirely within the frame,
    plus *margin* metres of context on each side.

    Args:
        scene: GSScene instance.
        world_x, world_y: Element centre in world horizontal plane.
        width_m: Element width in metres.
        height_m: Element height in metres (sill to header).
        mid_z: Camera height (world up-axis coordinate).
        center_2d: (cx, cy) room centre.
        up_axis: Vertical axis index.
        img_size: Output image size (square).
        margin: Extra metres of context beyond element edges.

    Returns:
        RingView with the rendered image, or None if the viewpoint
        is invalid (empty / too close).
    """
    from bim_recon.gs_scene import look_at_pose, GSScene

    h_axes = [i for i in range(3) if i != up_axis]
    cx, cy = float(center_2d[0]), float(center_2d[1])

    # Distance from camera to element
    dist = math.hypot(world_x - cx, world_y - cy)
    dist = max(dist, 0.5)  # safety clamp

    # Auto-FOV: cover the larger of (element width + margin) or
    # (element height + margin), ensuring the full element is visible.
    half_w = width_m / 2.0 + margin
    half_h = height_m / 2.0 + margin
    fov_w = math.degrees(2.0 * math.atan(half_w / dist))
    fov_h = math.degrees(2.0 * math.atan(half_h / dist))
    fov = min(max(fov_w, fov_h, 20.0), 75.0)  # clamp 20°–75°

    eye = [0.0, 0.0, 0.0]
    eye[h_axes[0]] = cx
    eye[h_axes[1]] = cy
    eye[up_axis] = mid_z

    target = [0.0, 0.0, 0.0]
    target[h_axes[0]] = float(world_x)
    target[h_axes[1]] = float(world_y)
    target[up_axis] = mid_z

    up = [0.0, 0.0, 0.0]
    up[up_axis] = 1.0

    pose = look_at_pose(tuple(eye), tuple(target), tuple(up))
    result, reason, _ = GSScene.render_validated(
        scene, pose, img_size, img_size, fov)
    if result is None:
        return None

    az = math.degrees(math.atan2(
        world_y - cy, world_x - cx)) % 360.0
    viewmat = pose.to_viewmat()
    R_c2w = viewmat[:3, :3].T.astype(np.float64)
    fx = 0.5 * img_size / math.tan(0.5 * math.radians(fov))

    return RingView(
        idx=-1,
        azimuth_deg=az,
        image=(result.colors * 255).clip(0, 255).astype(np.uint8),
        depth=result.depth,
        alpha=result.alpha,
        eye=np.array(eye, dtype=np.float64),
        forward=R_c2w[:, 2],
        right=R_c2w[:, 0],
        up=R_c2w[:, 1],
        fx=fx, fy=fx,
        cx_pix=img_size / 2.0,
        cy_pix=img_size / 2.0,
        width=img_size,
        height=img_size,
        fov_deg=fov,
    )




