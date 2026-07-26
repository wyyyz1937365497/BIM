"""Shared B-class spatial radar renderers.

Two top-down debug radars used by the B-class registration flow:

* ``observation_radar`` — backprojects the Falcon-segmented photo's depth into
  the A-class room frame (where the object sits before any GLB exists).
* ``registration_radar`` — projects the registered GLB's triangles into the
  same room frame (where the placed mesh lands).

Both overlay the A-class walls/elements (``context`` dict) so the B-class
object is seen in its room context. The renderers are pure (numpy + matplotlib)
and gradio-agnostic, shared by the main page and the Registration Lab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bim_recon.mesh_registrar import parse_glb_vertices_faces
from bim_recon.trellis_registration import backproject_observation



_CJK_FONT_CONFIGURED = False


def _ensure_cjk_font() -> None:
    """Pick a CJK-capable matplotlib font so Chinese labels render, not boxes.

    Probed once and cached. Prefers Microsoft YaHei (standard on Windows) with
    DejaVu Sans fallback, and disables the unicode-minus glyph CJK fonts lack.
    """
    global _CJK_FONT_CONFIGURED
    if _CJK_FONT_CONFIGURED:
        return
    import matplotlib
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun",
                      "Microsoft JhengHei", "Noto Sans CJK SC"):
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_CONFIGURED = True

def project_registered_mesh(
    manifest: dict[str, Any],
    center_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return registered GLB vertices/faces projected into A-class room XY."""
    assets = manifest.get("assets", {})
    glb_value = assets.get("glb") or manifest.get("glb_path")
    registration = manifest.get("registration", {})
    placement = registration.get("placement", {})
    placement_input = placement.get("placement_input", {})
    analysis = placement.get("mesh_analysis_trellis_space", {})
    transform = placement.get("transform_output", {})
    if not glb_value or not placement_input or not analysis or not transform:
        raise ValueError("registration manifest 缺少 GLB placement diagnostics")

    glb_path = Path(glb_value)
    if not glb_path.is_absolute():
        manifest_path = manifest.get("manifest_path")
        if manifest_path:
            glb_path = Path(manifest_path).resolve().parent / glb_path
        else:
            glb_path = Path.cwd() / glb_path
    vertices, faces = parse_glb_vertices_faces(glb_path.resolve())

    mesh_center = np.asarray(analysis["centroid_x_y_z"], dtype=np.float64)
    scale = float(transform["scale"])
    rotation = np.asarray(
        transform["rotation_matrix_row_major"], dtype=np.float64
    ).reshape(3, 3)
    translation = np.asarray(
        transform["translation_x_y_z"], dtype=np.float64
    )
    world_vertices = (vertices.astype(np.float64) - mesh_center) @ rotation.T
    world_vertices = world_vertices * scale + translation

    up_axis = int(placement_input.get("up_axis", 2))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    projected = world_vertices[:, horizontal_axes].copy()
    projected[:, 0] -= float(center_offset[0])
    projected[:, 1] -= float(center_offset[1])
    heights = world_vertices[:, up_axis]
    return projected, faces, heights


def draw_mesh_top_projection(
    ax,
    manifest: dict[str, Any],
    center_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the registered GLB's real top-view triangular projection."""
    from matplotlib.collections import PolyCollection

    projected, faces, heights = project_registered_mesh(manifest, center_offset)
    triangles = projected[faces]
    face_heights = heights[faces].mean(axis=1)
    order = np.argsort(face_heights)
    collection = PolyCollection(
        triangles[order],
        array=face_heights[order],
        cmap="magma",
        edgecolors=(0.35, 0.02, 0.12, 0.10),
        linewidths=0.18,
        alpha=0.76,
        zorder=7,
        label="新增 GLB 顶视投影",
    )
    ax.add_collection(collection)
    return projected[:, 0], projected[:, 1]


def draw_spatial_context(
    context: dict[str, Any] | None,
    manifest: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
):
    """Draw A-class geometry with one explicitly selected B-class stage."""
    if not context:
        return None
    import matplotlib
    matplotlib.use("Agg")
    _ensure_cjk_font()
    import matplotlib.pyplot as plt

    walls = context.get("walls", [])
    elements = context.get("elements", [])
    center = context.get("center_offset", [0.0, 0.0])
    cx, cy = float(center[0]), float(center[1])

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#f4f6f5")
    ax.set_facecolor("#f4f6f5")
    for index, wall in enumerate(walls):
        ax.plot(
            [float(wall["x1"]), float(wall["x2"])],
            [float(wall["y1"]), float(wall["y2"])],
            color="#202a27",
            linewidth=4,
            solid_capstyle="round",
            label="墙" if index == 0 else None,
            zorder=3,
        )

    element_style = {
        "door": ("#d4513c", "s", "门"),
        "window": ("#287f9d", "D", "窗"),
    }
    seen_labels: set[str] = set()
    for element in elements:
        kind = str(element.get("element_class", ""))
        color, marker, display_name = element_style.get(
            kind, ("#78847f", "o", kind or "构件")
        )
        ax.scatter(
            float(element["world_x"]),
            float(element["world_y"]),
            color=color,
            marker=marker,
            s=90,
            edgecolors="white",
            linewidths=1.2,
            label=display_name if display_name not in seen_labels else None,
            zorder=5,
        )
        seen_labels.add(display_name)

    ax.scatter(0.0, 0.0, color="#202a27", marker="^", s=110,
               label="A 类扫描中心", zorder=6)

    observation_xs: list[float] = []
    observation_ys: list[float] = []
    if observation:
        target = observation["horizontal_position"]
        camera_xy = observation["camera_horizontal_position"]
        tx, ty = float(target[0]) - cx, float(target[1]) - cy
        cam_x, cam_y = float(camera_xy[0]) - cx, float(camera_xy[1]) - cy
        width = max(
            float(observation["estimated_dimensions_m"].get("width", 0.1)),
            0.1,
        )
        mask_points = np.asarray(
            observation.get("mask_points_horizontal", []), dtype=np.float64
        )
        if mask_points.ndim == 2 and mask_points.shape[1] == 2 and len(mask_points):
            mask_x = mask_points[:, 0] - cx
            mask_y = mask_points[:, 1] - cy
            ax.scatter(
                mask_x, mask_y, s=5, color="#f6a623", alpha=0.28,
                edgecolors="none", label="Falcon mask 深度反投影点云", zorder=6,
            )
        else:
            mask_x = np.empty(0, dtype=np.float64)
            mask_y = np.empty(0, dtype=np.float64)
            ax.add_patch(plt.Circle(
                (tx, ty), width / 2.0, facecolor="#f6b44b", edgecolor="#9b4f00",
                linewidth=2.2, alpha=0.45, label="分割照片估算宽度", zorder=7,
            ))
        ax.plot(
            [cam_x, tx], [cam_y, ty], color="#e08a24", linewidth=2,
            linestyle="--", alpha=0.9, label="相机反投影视线", zorder=6,
        )
        ax.scatter(
            cam_x, cam_y, color="#e08a24", marker="^", s=120,
            edgecolors="white", linewidths=1.2, label="捕获相机", zorder=7,
        )
        ax.scatter(
            tx, ty, color="#9b4f00", marker="X", s=180,
            edgecolors="white", linewidths=1.2, label="照片反投影中心", zorder=8,
        )
        ax.annotate(
            f"反投影\n({tx:.2f}, {ty:.2f}) m\n距离 {float(observation['distance_m']):.2f} m",
            (tx, ty), xytext=(10, 12), textcoords="offset points",
            fontsize=9, color="#713900", weight="bold",
        )
        observation_xs = [cam_x, tx - width / 2.0, tx + width / 2.0]
        observation_ys = [cam_y, ty - width / 2.0, ty + width / 2.0]
        if mask_x.size:
            observation_xs.extend([float(mask_x.min()), float(mask_x.max())])
            observation_ys.extend([float(mask_y.min()), float(mask_y.max())])


    placement_input: dict[str, Any] = {}
    mesh_xs: np.ndarray | None = None
    mesh_ys: np.ndarray | None = None
    if manifest:
        registration = manifest.get("registration", {})
        placement = registration.get("placement", {})
        placement_input = placement.get("placement_input", {})
        transform = placement.get("transform_output", {})
        if placement_input:
            gx = float(placement_input["world_x"]) - cx
            gy = float(placement_input["world_y"]) - cy
            width = max(float(placement_input.get("element_width_m", 0.5)), 0.1)
            angle = np.deg2rad(float(
                transform.get("principal_axis_world_horizontal_angle_deg", 0.0)
            ))
            try:
                mesh_xs, mesh_ys = draw_mesh_top_projection(
                    ax, manifest, (cx, cy)
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                radius = width / 2.0
                ax.add_patch(plt.Circle(
                    (gx, gy), radius, fill=False, edgecolor="#b0163f",
                    linewidth=2.5, linestyle="--", zorder=7,
                    label="GLB 估算占地",
                ))
            ax.scatter(gx, gy, color="#b0163f", marker="*", s=280,
                       edgecolors="white", linewidths=1.5,
                       label="当前 GLB 中心", zorder=8)
            axis_length = max(width * 0.8, 0.45)
            ax.arrow(
                gx, gy, axis_length * np.cos(angle), axis_length * np.sin(angle),
                width=0.018, head_width=0.14, head_length=0.16,
                length_includes_head=True, color="#b0163f", zorder=8,
            )
            ax.annotate(
                f"GLB\n({gx:.2f}, {gy:.2f}) m",
                (gx, gy), xytext=(10, 12), textcoords="offset points",
                fontsize=9, color="#7f0f31", weight="bold",
            )

    xs = [float(v[key]) for v in walls for key in ("x1", "x2")]
    ys = [float(v[key]) for v in walls for key in ("y1", "y2")]
    xs.extend(observation_xs)
    ys.extend(observation_ys)
    if manifest and placement_input:
        xs.append(gx)
        ys.append(gy)
        if mesh_xs is not None and mesh_ys is not None:
            xs.extend(mesh_xs.tolist())
            ys.extend(mesh_ys.tolist())
    if xs and ys:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 2.0)
        margin = max(0.8, span * 0.18)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#bfc9c4", linewidth=0.7, alpha=0.55)
    ax.set_xlabel("房间 X / m")
    ax.set_ylabel("房间 Y / m")
    if observation:
        ax.set_title("分割照片深度反投影位置")
    elif manifest:
        ax.set_title("GLB 配准后场景顶视投影")
    else:
        ax.set_title("A 类房间布局")
    ax.legend(loc="best", frameon=True, framealpha=0.94)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = rgba[:, :, :3].copy()
    plt.close(fig)
    return image


def observation_radar(
    render_state: dict[str, Any] | None,
    detection_info: dict[str, Any] | None,
    context: dict[str, Any] | None,
):
    """Render the segmented-photo backprojection before any GLB exists."""
    if not context:
        return None
    if not render_state or not detection_info:
        return draw_spatial_context(context)
    try:
        observation = backproject_observation(render_state, detection_info)
        return draw_spatial_context(context, observation=observation)
    except (KeyError, TypeError, ValueError):
        return draw_spatial_context(context)


def registration_radar(
    manifest: dict[str, Any] | None,
    context: dict[str, Any] | None,
):
    """Render the registered GLB top-view projection in the room frame."""
    return draw_spatial_context(context, manifest)
