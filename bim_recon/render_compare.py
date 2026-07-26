"""Constrained render-compare Sim(3) registration for B-class meshes.

Aligns a TRELLIS-generated mesh to the backprojected observation (Falcon
instance mask + 3DGS metric depth) by optimizing a *constrained* pose:
yaw + uniform scale + planar translation, with floor contact enforced as a
hard constraint. Uses Open3D ``RaycastingScene`` for real triangle
rasterization (silhouette + depth), not point projection.

Method (attachment-1 §四). The search optimizes the energy

    E = λ_m·(1 − IoU(M_mesh, M_obs)) + λ_d·mean Huber(D_mesh − D_3DGS ; in M_obs)

with floor contact enforced by ``MeshPlacement.preserve_floor_contact`` (so the
E_floor term is a hard constraint, zero by construction). Every candidate pose
is built with the *same* ``compute_placement_transform`` used for final
placement, so the search and landing conventions cannot diverge (attachment
item 2). Free parameters: ξ = (yaw, scale, t_x, t_y); δz is fixed at 0.

Depth convention: 3DGS renders *forward-axis (z) depth* (see
``_unproject_mask_points``), so rasterized ``t_hit`` (Euclidean ray length) is
converted to forward depth via ``t_hit · (ray_dir · forward)`` before compare.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bim_recon.mesh_registrar import (
    MeshPlacement,
    compute_placement_transform,
    parse_glb_vertices_faces,
    _sample_surface_points,
)
from bim_recon.trellis_registration import robust_mask_anchor, _unproject_mask_points


# ---------------------------------------------------------------------------
# Camera rays
# ---------------------------------------------------------------------------

def _camera_frame(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (eye, forward, right, down, fov) for the pinhole camera.

    Uses the ACTUAL ``camera["up"]`` vector (same as ``gs_scene.look_at_pose``),
    NOT a basis-vector reconstruction from up_axis — those differ when the
    viewer's up isn't axis-aligned, which rotates the whole frame and
    backprojects to the wrong 3D position.
    """
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up_raw = camera.get("up")
    if up_raw is not None:
        up = np.asarray(up_raw, dtype=np.float64)
    else:
        up_axis = int(camera.get("up_axis", 2))
        up = np.zeros(3, dtype=np.float64)
        up[up_axis] = 1.0
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(forward, right)
    fov = float(camera.get("fov_degrees", camera.get("fov", 45.0)))
    return eye, forward, right, down, fov


def _rays_for_grid(
    camera: dict[str, Any], full_img_w: int, full_img_h: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build per-pixel ray origins/dirs on a strided grid of the FULL image.

    Rays are computed at full-image pixel coords ``(stride*ir, stride*ic)``
    with the full-image focal, so the low-res rasterization aligns exactly with
    ``obs[::stride, ::stride]``. Returns (origins, dirs, forward, rw, rh).
    """
    eye, forward, right, down, fov = _camera_frame(camera)
    rh = max(full_img_h // stride, 2)
    rw = max(full_img_w // stride, 2)
    # FOV is horizontal (matches gs_scene.fov_to_intrinsics): focal from width.
    focal_x = 0.5 * full_img_w / math.tan(math.radians(fov) / 2.0)
    focal_y = focal_x  # square pixels
    rows = (np.arange(rh) * stride + stride // 2).astype(np.float64)
    cols = (np.arange(rw) * stride + stride // 2).astype(np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    x_cam = (cc - full_img_w / 2.0) / focal_x
    y_cam = (rr - full_img_h / 2.0) / focal_y
    dirs = (right[None, None, :] * x_cam[:, :, None]
            + down[None, None, :] * y_cam[:, :, None]
            + forward[None, None, :])
    dirs = dirs / (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12)
    origins = np.broadcast_to(eye, dirs.shape)
    return (origins.reshape(-1, 3), dirs.reshape(-1, 3), forward, rw, rh)


# ---------------------------------------------------------------------------
# Rasterization (Open3D RaycastingScene)
# ---------------------------------------------------------------------------

def rasterize_pose(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    camera: dict[str, Any],
    full_img_w: int,
    full_img_h: int,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Render mesh (already in world coords) → (silhouette bool[H,W], depth[H,W]).

    ``depth`` is the forward-axis (z) distance, matching the 3DGS depth
    convention. Misses carry depth 0 and silhouette False.
    """
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(
        np.ascontiguousarray(vertices_world, dtype=np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(
        np.ascontiguousarray(faces, dtype=np.int32))
    scene.add_triangles(mesh_t)

    origins, dirs, forward, rw, rh = _rays_for_grid(camera, full_img_w, full_img_h, stride)
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy().astype(np.float64).reshape(rh, rw)
    hit = np.isfinite(t_hit) & (t_hit > 0)
    cos_angle = dirs.reshape(rh, rw, 3) @ forward
    depth = np.where(hit, t_hit * cos_angle, 0.0)
    return hit, depth


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _energy(
    sil_mesh: np.ndarray,
    depth_mesh: np.ndarray,
    mask_obs: np.ndarray,
    depth_obs: np.ndarray,
    *,
    lam_m: float = 1.0,
    lam_d: float = 3.0,
) -> tuple[float, float, float, float]:
    """Return (E, IoU, depth_mae_m, coverage).

    The depth term uses raw MAE (metres), not Huber — Huber with a 5 cm delta
    made the depth contribution <1% of the total energy, so the optimizer was
    effectively silhouette-only and couldn't use the 3D surface shape to
    discriminate yaw. With lam_d=3, a 10 cm depth error contributes 0.3 (vs
    ~0.4 for 1-IoU), so both terms drive the optimization.
    """
    inter = int((sil_mesh & mask_obs).sum())
    union = int((sil_mesh | mask_obs).sum())
    iou = inter / max(union, 1)
    valid = mask_obs & sil_mesh & np.isfinite(depth_obs) & (depth_obs > 0.05)
    coverage = int(valid.sum()) / max(int(mask_obs.sum()), 1)
    if int(valid.sum()) > 0:
        depth_mae = float(np.abs(depth_mesh[valid] - depth_obs[valid]).mean())
    else:
        depth_mae = 0.5  # strong penalty for no overlap
    energy = lam_m * (1.0 - iou) + lam_d * depth_mae
    return float(energy), float(iou), depth_mae, float(coverage)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RenderCompareResult:
    """Outcome of the constrained render-compare optimization."""
    yaw_degrees: float
    scale_multiplier: float
    translation_xy: tuple[float, float]
    world_xy: tuple[float, float]
    iou: float
    depth_mae_m: float
    coverage: float
    accepted: bool
    fallback_reason: str | None = None
    coarse_scores: list[dict] = field(default_factory=list)
    rotation_override: tuple[tuple[float, float, float], ...] | None = None
    icp_fitness: float = 0.0
    icp_rmse: float = 0.0

    def diagnostics(self) -> dict:
        return {
            "method": "render_compare_constrained",
            "yaw_degrees": round(self.yaw_degrees, 3),
            "scale_multiplier": round(self.scale_multiplier, 4),
            "translation_xy": [round(float(v), 4) for v in self.translation_xy],
            "world_xy": [round(float(v), 4) for v in self.world_xy],
            "iou": round(self.iou, 4),
            "depth_mae_m": round(self.depth_mae_m, 4),
            "coverage": round(self.coverage, 4),
            "accepted": bool(self.accepted),
            "fallback_reason": self.fallback_reason,
            "coarse_scores": self.coarse_scores,
            "icp_fitness": round(self.icp_fitness, 4),
            "icp_rmse": round(self.icp_rmse, 4),
            "has_rotation_override": self.rotation_override is not None,
        }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _bbox_center_anchor(depth: np.ndarray, norm_bbox: dict, camera: dict[str, Any]) -> np.ndarray:
    """Fallback anchor: unproject the bbox center at the bbox-median depth."""
    img_h, img_w = depth.shape
    cx = float(norm_bbox.get("x", 0.5)) * img_w
    cy = float(norm_bbox.get("y", 0.5)) * img_h
    w = max(float(norm_bbox.get("w", 0.2)) * img_w, 2.0)
    h = max(float(norm_bbox.get("h", 0.2)) * img_h, 2.0)
    x0 = int(max(cx - w / 2, 0)); x1 = int(min(cx + w / 2, img_w))
    y0 = int(max(cy - h / 2, 0)); y1 = int(min(cy + h / 2, img_h))
    window = depth[y0:y1, x0:x1]
    finite = window[np.isfinite(window) & (window > 0.05)]
    d = float(np.median(finite)) if finite.size else 1.0
    eye, forward, right, down, fov = _camera_frame(camera)
    focal_y = 0.5 * img_w / math.tan(math.radians(fov) / 2.0)  # FOV is horizontal
    x_cam = (cx - img_w / 2.0) / focal_y
    y_cam = (cy - img_h / 2.0) / focal_y
    return eye + right * (x_cam * d) + down * (y_cam * d) + forward * d


def _placement_for(
    glb_path: Path, anchor: np.ndarray, h_axes: list[int],
    *, floor_z: float, ceiling_z: float, element_width_m: float,
    element_height_m: float, up_axis: int, yaw: float, scale: float,
    tx: float, ty: float,
) -> MeshPlacement:
    return MeshPlacement(
        glb_path=glb_path,
        world_x=float(anchor[h_axes[0]] + tx),
        world_y=float(anchor[h_axes[1]] + ty),
        floor_z=floor_z, ceiling_z=ceiling_z,
        element_width_m=element_width_m, element_height_m=element_height_m,
        up_axis=up_axis, yaw_degrees=float(yaw),
        scale_multiplier=float(scale), preserve_floor_contact=True,
    )


def optimize_placement(
    glb_path: Path | str,
    observation: dict[str, Any],
    *,
    floor_z: float,
    ceiling_z: float,
    up_axis: int,
    element_width_m: float,
    element_height_m: float,
    stride: int = 4,
    yaw_coarse: float = 5.0,
    yaw_fine: float = 0.5,
    scale_steps: tuple[float, ...] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
    translate_step: float = 0.05,
    translate_radius_m: float = 0.3,
    accept_iou: float = 0.20,
    accept_depth_mae: float = 0.10,
    outer_iters: int = 2,
    debug_dir: Path | str | None = None,
) -> RenderCompareResult:
    """Constrained render-compare alignment of a TRELLIS mesh to the observation.

    ``observation`` keys: ``camera``, ``depth`` (3DGS, full frame),
    ``mask`` (Falcon, full frame, bool/uint8), ``norm_bbox``.

    Optimizes yaw → scale → planar translation (coarse-to-fine, coordinate
    descent, ``outer_iters`` passes), then applies quality gating. Every
    candidate uses ``compute_placement_transform`` for convention consistency.
    """
    glb_path = Path(glb_path)
    vertices, faces = parse_glb_vertices_faces(glb_path)
    camera = observation["camera"]
    depth_full = np.asarray(observation["depth"], dtype=np.float32)
    mask_full = np.asarray(observation["mask"]) > 0
    if mask_full.shape != depth_full.shape:
        raise ValueError("observation mask and depth shapes differ")
    norm_bbox = observation["norm_bbox"]
    img_h, img_w = depth_full.shape

    # Downsampled observation: sample at block CENTERS (rays fire at
    # stride*ir + stride//2) and crop to the exact ray-grid dims so the
    # rasterized silhouette/mask shapes always match.
    half = stride // 2
    rh, rw = img_h // stride, img_w // stride
    mask_ds = mask_full[half:half + rh * stride:stride,
                        half:half + rw * stride:stride]
    depth_ds = depth_full[half:half + rh * stride:stride,
                          half:half + rw * stride:stride]
    if (mask_ds.shape[0] < 4 or mask_ds.shape[1] < 4) or not mask_full.any():
        # Tiny or empty mask — nothing to align to.
        return RenderCompareResult(
            yaw_degrees=90.0, scale_multiplier=1.0, translation_xy=(0.0, 0.0),
            world_xy=(0.0, 0.0),
            iou=0.0, depth_mae_m=0.0, coverage=0.0, accepted=False,
            fallback_reason="mask_too_small",
        )

    # Robust visible-surface anchor (item 1); bbox-center fallback.
    anchor = robust_mask_anchor(depth_full, mask_full, camera)
    if anchor is None:
        anchor = _bbox_center_anchor(depth_full, norm_bbox, camera)
    anchor = np.asarray(anchor, dtype=np.float64)
    h_axes = [i for i in range(3) if i != up_axis]

    # Estimate floor_z from the lowest visible mask surface point. The caller's
    # floor_z is often 0 (wrong when the scene origin is offset — the mesh
    # would land above the camera and be invisible).
    _mask_pts = _unproject_mask_points(depth_full, mask_full, camera)
    if _mask_pts.size > 0:
        floor_z = float(np.min(_mask_pts[:, up_axis]))

    # Translation search radius must accommodate the surface-anchor → mesh-center
    # offset (≈ half the object depth), else the optimizer can't push the mesh
    # center back to its true position (attachment §二.3). Scale with object size.
    eff_radius = max(translate_radius_m, 0.75 * element_width_m)

    def score(yaw: float, scale: float, tx: float, ty: float) -> tuple[float, float, float, float]:
        placement = _placement_for(
            glb_path, anchor, h_axes, floor_z=floor_z, ceiling_z=ceiling_z,
            element_width_m=element_width_m, element_height_m=element_height_m,
            up_axis=up_axis, yaw=yaw, scale=scale, tx=tx, ty=ty,
        )
        transform = compute_placement_transform(placement)
        sil, depth = rasterize_pose(
            transform.vertices_world, transform.faces, camera,
            img_w, img_h, stride=stride,
        )
        return _energy(sil, depth, mask_ds, depth_ds)

    # --- Stage 1: coarse yaw scan (scale=1, no translation) ---
    best_yaw, best_scale, best_tx, best_ty = 0.0, 1.0, 0.0, 0.0
    E_best = math.inf
    coarse_scores: list[dict] = []
    coarse_ranked: list[tuple[float, float]] = []  # (E, yaw)
    yaws = np.arange(0.0, 360.0, yaw_coarse)
    for y in yaws:
        E, iou, dmae, cov = score(float(y), 1.0, 0.0, 0.0)
        coarse_scores.append({"yaw": round(float(y), 2), "E": round(E, 4),
                              "iou": round(iou, 4)})
        coarse_ranked.append((E, float(y)))

    # Multi-start fine yaw: refine around the top-5 coarse basins. For elongated
    # or asymmetric objects the IoU+depth landscape is multi-modal; a single
    # start traps in a local optimum and the mesh ends up at the wrong facing.
    coarse_ranked.sort()
    for _, cand_yaw in coarse_ranked[:5]:
        for y in np.arange(cand_yaw - yaw_coarse, cand_yaw + yaw_coarse + yaw_fine, yaw_fine):
            E, *_ = score(float(y), best_scale, best_tx, best_ty)
            if E < E_best:
                E_best = E
                best_yaw = float(y)

    for _ in range(max(1, outer_iters)):
        # Yaw: re-fine around current best (scale/translation may have shifted it).
        for y in np.arange(best_yaw - yaw_coarse, best_yaw + yaw_coarse + yaw_fine, yaw_fine):
            E, *_ = score(float(y), best_scale, best_tx, best_ty)
            if E < E_best:
                E_best = E
                best_yaw = float(y)
        # Scale.
        for s in scale_steps:
            E, *_ = score(best_yaw, float(s), best_tx, best_ty)
            if E < E_best:
                E_best = E
                best_scale = float(s)
        # Planar translation (coordinate descent over a small grid).
        improved = True
        while improved:
            improved = False
            for axis, vals in (
                (0, [best_tx + k * translate_step for k in range(-1, 2)]),
                (1, [best_ty + k * translate_step for k in range(-1, 2)]),
            ):
                for v in vals:
                    if abs(v) > eff_radius + 1e-9:
                        continue
                    tx, ty = (v, best_ty) if axis == 0 else (best_tx, v)
                    E, *_ = score(best_yaw, best_scale, tx, ty)
                    if E < E_best - 1e-6:
                        E_best = E
                        if axis == 0:
                            best_tx = float(v)
                        else:
                            best_ty = float(v)
                        improved = True

    final_E, final_iou, final_dmae, final_cov = score(
        best_yaw, best_scale, best_tx, best_ty)

    # accepted / reason computed after ICP refinement below.
    # --- Phase 2: 3D ICP refinement ---
    result_wx = float(anchor[h_axes[0]] + best_tx)
    result_wy = float(anchor[h_axes[1]] + best_ty)
    icp_fitness = 0.0
    icp_rmse = 0.0
    rotation_override = None
    try:
        p1 = _placement_for(
            glb_path, anchor, h_axes, floor_z=floor_z, ceiling_z=ceiling_z,
            element_width_m=element_width_m, element_height_m=element_height_m,
            up_axis=up_axis, yaw=best_yaw, scale=best_scale,
            tx=best_tx, ty=best_ty,
        )
        p1t = compute_placement_transform(p1)
        icp_out = refine_with_icp(p1t, observation, up_axis=up_axis)
        if icp_out is not None:
            R_refined, world_xy_refined, icp_fitness, icp_rmse = icp_out
            rotation_override = tuple(tuple(float(v) for v in row) for row in R_refined)
            refined_wx = float(world_xy_refined[h_axes[0]])
            refined_wy = float(world_xy_refined[h_axes[1]])
            # Re-score the ICP-refined placement.
            icp_p = MeshPlacement(
                glb_path=glb_path,
                world_x=refined_wx, world_y=refined_wy,
                floor_z=floor_z, ceiling_z=ceiling_z,
                element_width_m=element_width_m, element_height_m=element_height_m,
                up_axis=up_axis,
                rotation_override=rotation_override,
                scale_multiplier=best_scale,
                preserve_floor_contact=True,
            )
            icp_t = compute_placement_transform(icp_p)
            icp_sil, icp_depth = rasterize_pose(
                icp_t.vertices_world, icp_t.faces, camera,
                img_w, img_h, stride=stride,
            )
            final_E, final_iou, final_dmae, final_cov = _energy(
                icp_sil, icp_depth, mask_ds, depth_ds)
            result_wx = refined_wx
            result_wy = refined_wy
    except Exception:
        pass
    if debug_dir is not None:
        try:
            # Round-trip check: backproject mask centroid → 3D → re-project.
            # If the re-projected pixel ≠ mask centroid, the camera model
            # (up vector / focal / frame) is inconsistent with gs_scene.
            import json as _json
            ys_f, xs_f = np.nonzero(mask_full)
            if len(xs_f) > 0:
                mc_col = float(np.median(xs_f))
                mc_row = float(np.median(ys_f))
                mc_r = int(np.clip(mc_row, 0, img_h - 1))
                mc_c = int(np.clip(mc_col, 0, img_w - 1))
                mc_d = float(depth_full[mc_r, mc_c])
                _eye, _fwd, _right, _down, _fov = _camera_frame(camera)
                _focal = 0.5 * img_w / math.tan(math.radians(_fov) / 2.0)
                _anchor3d = (_eye + _right * ((mc_col - img_w / 2) / _focal * mc_d)
                             + _down * ((mc_row - img_h / 2) / _focal * mc_d)
                             + _fwd * mc_d)
                _rel = _anchor3d - _eye
                _z = _rel @ _fwd
                _pcol = (_rel @ _right) / _z * _focal + img_w / 2.0
                _prow = (_rel @ _down) / _z * _focal + img_h / 2.0
                _up_raw = camera.get("up")
                _ua = int(camera.get("up_axis", 2))
                _basis = [0.0, 0.0, 0.0]; _basis[_ua] = 1.0
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                (Path(debug_dir) / "roundtrip.json").write_text(_json.dumps({
                    "camera_eye": [round(float(v), 4) for v in _eye],
                    "camera_target": [round(float(v), 4) for v in camera["target"]],
                    "camera_up": list(_up_raw) if _up_raw is not None else None,
                    "up_axis_basis": _basis,
                    "up_mismatch": (str(list(_up_raw)) != str(_basis)) if _up_raw is not None else False,
                    "fov": _fov, "img_size": [img_w, img_h], "focal": round(_focal, 1),
                    "mask_centroid_px": [round(mc_col, 1), round(mc_row, 1)],
                    "mask_depth_at_centroid": round(mc_d, 4),
                    "anchor_3d": [round(float(v), 4) for v in _anchor3d],
                    "anchor_projected_px": [round(_pcol, 1), round(_prow, 1)],
                    "roundtrip_error_px": round(float(np.hypot(_pcol - mc_col, _prow - mc_row)), 2),
                    "anchor_world_xy": [round(float(_anchor3d[h_axes[0]]), 4),
                                         round(float(_anchor3d[h_axes[1]]), 4)],
                }, indent=2), encoding="utf-8")
            if rotation_override is not None:
                # Use the ICP-refined pose for the debug image.
                _debug_p = MeshPlacement(
                    glb_path=glb_path,
                    world_x=result_wx, world_y=result_wy,
                    floor_z=floor_z, ceiling_z=ceiling_z,
                    element_width_m=element_width_m, element_height_m=element_height_m,
                    up_axis=up_axis,
                    rotation_override=rotation_override,
                    scale_multiplier=best_scale,
                    preserve_floor_contact=True,
                )
                _dump_render_compare_debug_refined(
                    Path(debug_dir), _debug_p, camera=camera,
                    img_w=img_w, img_h=img_h, stride=stride,
                    mask_obs=mask_ds, iou=final_iou, depth_mae=final_dmae,
                    coverage=final_cov,
                )
            else:
                _dump_render_compare_debug(
                    Path(debug_dir), glb_path, anchor, h_axes,
                    floor_z=floor_z, ceiling_z=ceiling_z,
                    element_width_m=element_width_m, element_height_m=element_height_m,
                    up_axis=up_axis, yaw=best_yaw, scale=best_scale,
                    tx=best_tx, ty=best_ty, camera=camera,
                    img_w=img_w, img_h=img_h, stride=stride,
                    mask_obs=mask_ds, iou=final_iou, depth_mae=final_dmae,
                    coverage=final_cov,
                )
        except Exception:
            pass
    accepted = bool(final_iou >= accept_iou and final_dmae <= accept_depth_mae)
    reason = None if accepted else (
        "low_iou" if final_iou < accept_iou else "high_depth_mae"
    )
    return RenderCompareResult(
        yaw_degrees=best_yaw, scale_multiplier=best_scale,
        translation_xy=(round(best_tx, 4), round(best_ty, 4)),
        world_xy=(round(result_wx, 4), round(result_wy, 4)),
        iou=final_iou, depth_mae_m=final_dmae, coverage=final_cov,
        accepted=accepted, fallback_reason=reason,
        coarse_scores=coarse_scores,
        rotation_override=rotation_override,
        icp_fitness=round(float(icp_fitness), 4),
        icp_rmse=round(float(icp_rmse), 4),
    )


def refine_with_icp(
    transform: "MeshTransform",
    observation: dict[str, Any],
    *,
    up_axis: int,
    max_distance: float = 0.20,
    sample_count: int = 5000,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Refine a Phase-1 transform via point-to-point ICP against the 3DGS surface.

    Returns ``(R_refined_3x3, world_xy_2, fitness, rmse)`` or ``None`` if the
    target point cloud is too sparse or ICP degenerates.

    Uses point-to-point (not point-to-plane) because backprojected single-view
    points produce degenerate surface normals that cause point-to-plane to
    diverge to wild translations.
    """
    from bim_recon.trellis_registration import _unproject_mask_points as _ump

    # Target: mask-backprojected 3DGS world points.
    depth_full = np.asarray(observation["depth"], dtype=np.float32)
    mask_full = np.asarray(observation["mask"]) > 0
    camera = observation["camera"]
    target_pts = _ump(depth_full, mask_full, camera)
    if target_pts.size == 0 or len(target_pts) < 200:
        return None

    # Source: mesh surface points in world space.
    mesh_pts = _sample_surface_points(
        transform.vertices_world, transform.faces, target_count=sample_count)

    import open3d as o3d
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(mesh_pts.astype(np.float64))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)

    # ICP: source → target, identity init (both in world space already).
    T_init = np.eye(4)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    try:
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_distance, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
    except RuntimeError:
        return None

    if result.fitness < 0.1:
        return None

    T_icp = np.asarray(result.transformation, dtype=np.float64)
    R_icp = T_icp[:3, :3]
    t_icp = T_icp[:3, 3]

    # Combined rotation: Phase-1 (TRELLIS→world) then ICP delta.
    R_refined = R_icp @ np.asarray(transform.rotation, dtype=np.float64)

    # Refined world center (horizontal).
    refined_vertices = (R_icp @ transform.vertices_world.astype(np.float64).T).T + t_icp
    h_axes = [i for i in range(3) if i != up_axis]
    new_wx = float(np.mean(refined_vertices[:, h_axes[0]]))
    new_wy = float(np.mean(refined_vertices[:, h_axes[1]]))
    world_xy = np.array([new_wx, new_wy], dtype=np.float64)

    return R_refined, world_xy, float(result.fitness), float(result.inlier_rmse)


def _dump_render_compare_debug(
    debug_dir: Path, glb_path: Path, anchor: np.ndarray, h_axes: list[int], *,
    floor_z: float, ceiling_z: float, element_width_m: float, element_height_m: float,
    up_axis: int, yaw: float, scale: float, tx: float, ty: float,
    camera: dict[str, Any], img_w: int, img_h: int, stride: int,
    mask_obs: np.ndarray, iou: float, depth_mae: float, coverage: float,
) -> None:
    """Save a 3-panel debug image: observed mask, rendered silhouette, overlay.

    Lets the user verify whether the optimizer's best pose actually aligns the
    mesh with the observation (overlay should be mostly yellow = overlap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    placement = MeshPlacement(
        glb_path=glb_path,
        world_x=float(anchor[h_axes[0]] + tx),
        world_y=float(anchor[h_axes[1]] + ty),
        floor_z=floor_z, ceiling_z=ceiling_z,
        element_width_m=element_width_m, element_height_m=element_height_m,
        up_axis=up_axis, yaw_degrees=yaw, scale_multiplier=scale,
        preserve_floor_contact=True,
    )
    transform = compute_placement_transform(placement)
    sil, _ = rasterize_pose(
        transform.vertices_world, transform.faces, camera, img_w, img_h, stride=stride)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_obs, cmap="gray"); axes[0].set_title("observed mask")
    axes[1].imshow(sil, cmap="gray"); axes[1].set_title("rendered silhouette")
    overlay = np.zeros((mask_obs.shape[0], mask_obs.shape[1], 3), dtype=np.uint8)
    overlay[mask_obs] = [255, 0, 0]        # observed = red
    overlay[sil] = [0, 255, 0]             # rendered = green (overlap → yellow)
    axes[2].imshow(overlay); axes[2].set_title("overlay (red=obs, green=mesh)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(
        f"yaw={yaw:.1f} scale={scale:.3f} tx,ty=({tx:.2f},{ty:.2f}) "
        f"IoU={iou:.3f} depth_mae={depth_mae*100:.1f}cm cov={coverage:.2f}")
    fig.tight_layout()
    fig.savefig(str(debug_dir / "render_compare_debug.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _dump_render_compare_debug_refined(
    debug_dir: Path, placement: MeshPlacement, *,
    camera: dict[str, Any], img_w: int, img_h: int, stride: int,
    mask_obs: np.ndarray, iou: float, depth_mae: float, coverage: float,
) -> None:
    """Same 3-panel debug image, but takes a full MeshPlacement (for ICP-refined poses)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    transform = compute_placement_transform(placement)
    sil, _ = rasterize_pose(
        transform.vertices_world, transform.faces, camera, img_w, img_h, stride=stride)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_obs, cmap="gray"); axes[0].set_title("observed mask")
    axes[1].imshow(sil, cmap="gray"); axes[1].set_title("rendered silhouette (ICP)")
    overlay = np.zeros((mask_obs.shape[0], mask_obs.shape[1], 3), dtype=np.uint8)
    overlay[mask_obs] = [255, 0, 0]        # observed = red
    overlay[sil] = [0, 255, 0]             # rendered = green (overlap → yellow)
    axes[2].imshow(overlay); axes[2].set_title("overlay (red=obs, green=mesh)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(
        f"ICP-refined  IoU={iou:.3f} depth_mae={depth_mae*100:.1f}cm cov={coverage:.2f}")
    fig.tight_layout()
    fig.savefig(str(debug_dir / "render_compare_debug.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
