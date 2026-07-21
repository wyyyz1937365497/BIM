# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender 4.3 UI for B-class TRELLIS-to-3DGS pose annotations."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel

from .annotation_io import (
    AnnotationError,
    ROLE_GS_REFERENCE,
    ROLE_KEY,
    ROLE_OBSERVATION_CAMERA,
    ROLE_TRELLIS_PROXY,
    export_scene_manifest,
    validate_scene,
)

bl_info = {
    "name": "BIM Pose Annotation",
    "author": "TJ BIM Project",
    "version": (0, 1, 0),
    "blender": (4, 3, 0),
    "location": "3D Viewport > Sidebar > BIM Pose",
    "description": "Align TRELLIS geometry to a 3DGS reference and export ground truth",
    "category": "3D View",
}

STATUS_ITEMS = (
    ("draft", "Draft", "Alignment is still being edited"),
    ("approved", "Approved", "Alignment is ready for dataset export"),
    ("review", "Needs Review", "Alignment requires another review"),
    ("rejected", "Rejected", "TRELLIS geometry is unsuitable for pose training"),
)


def _active_object(context: bpy.types.Context) -> bpy.types.Object | None:
    return context.active_object


def _set_role(obj: bpy.types.Object, role: str) -> None:
    obj[ROLE_KEY] = role


def _object_status(obj: bpy.types.Object) -> str:
    value = str(obj.get("bim_annotation_status", "draft"))
    return value if value in {item[0] for item in STATUS_ITEMS} else "draft"


class BIMPOSE_OT_mark_gs_reference(Operator):
    bl_idname = "bim_pose.mark_gs_reference"
    bl_label = "Mark Active as GS Reference"
    bl_description = "Use the active object as the sole raw 3DGS coordinate reference"
    bl_options = {"REGISTER", "UNDO"}

    source_ply: StringProperty(name="Source PLY", subtype="FILE_PATH")
    metric_scale_validated: bpy.props.BoolProperty(
        name="Metric Scale Validated",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _active_object(context) is not None

    def invoke(self, context, event):
        obj = _active_object(context)
        self.source_ply = str(obj.get("bim_source_ply", ""))
        self.metric_scale_validated = bool(
            obj.get("bim_metric_scale_validated", False)
        )
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = _active_object(context)
        for candidate in context.scene.objects:
            if candidate.get(ROLE_KEY) == ROLE_GS_REFERENCE:
                del candidate[ROLE_KEY]
        _set_role(obj, ROLE_GS_REFERENCE)
        obj["bim_source_ply"] = self.source_ply
        obj["bim_metric_scale_validated"] = self.metric_scale_validated
        self.report({"INFO"}, f"Marked {obj.name} as GS reference")
        return {"FINISHED"}


class BIMPOSE_OT_mark_trellis_proxy(Operator):
    bl_idname = "bim_pose.mark_trellis_proxy"
    bl_label = "Mark Active as TRELLIS GT Proxy"
    bl_description = "Mark an unmodified raw-coordinate mesh as one annotated TRELLIS variant"
    bl_options = {"REGISTER", "UNDO"}

    object_id: StringProperty(name="Object ID")
    class_name: StringProperty(name="Class Name")
    variant_id: StringProperty(name="Variant ID")
    source_view_id: StringProperty(name="Source View ID")
    source_glb: StringProperty(name="Source GLB", subtype="FILE_PATH")
    proxy_source: StringProperty(name="Proxy Source", subtype="FILE_PATH")
    status: EnumProperty(name="Status", items=STATUS_ITEMS, default="draft")
    quality: FloatProperty(name="Quality", min=0.0, max=1.0, default=0.0)
    notes: StringProperty(name="Notes")

    @classmethod
    def poll(cls, context):
        obj = _active_object(context)
        return obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        obj = _active_object(context)
        self.object_id = str(obj.get("bim_object_id", ""))
        self.class_name = str(obj.get("bim_class_name", ""))
        self.variant_id = str(obj.get("bim_variant_id", ""))
        self.source_view_id = str(obj.get("bim_source_view_id", ""))
        self.source_glb = str(obj.get("bim_source_glb", ""))
        self.proxy_source = str(obj.get("bim_proxy_source", ""))
        self.status = _object_status(obj)
        self.quality = float(obj.get("bim_annotation_quality", 0.0))
        self.notes = str(obj.get("bim_annotation_notes", ""))
        return context.window_manager.invoke_props_dialog(self, width=560)

    def execute(self, context):
        obj = _active_object(context)
        _set_role(obj, ROLE_TRELLIS_PROXY)
        obj["bim_object_id"] = self.object_id.strip()
        obj["bim_class_name"] = self.class_name.strip()
        obj["bim_variant_id"] = self.variant_id.strip()
        obj["bim_source_view_id"] = self.source_view_id.strip()
        obj["bim_source_glb"] = self.source_glb
        obj["bim_proxy_source"] = self.proxy_source
        obj["bim_annotation_status"] = self.status
        obj["bim_annotation_quality"] = self.quality
        obj["bim_annotation_notes"] = self.notes
        self.report({"INFO"}, f"Marked {obj.name} as TRELLIS proxy")
        return {"FINISHED"}


class BIMPOSE_OT_mark_observation_camera(Operator):
    bl_idname = "bim_pose.mark_observation_camera"
    bl_label = "Mark Active as Observation Camera"
    bl_description = "Associate a Blender camera and aligned observation assets with one object"
    bl_options = {"REGISTER", "UNDO"}

    target_object_id: StringProperty(name="Target Object ID")
    view_id: StringProperty(name="View ID")
    rgb_path: StringProperty(name="RGB", subtype="FILE_PATH")
    depth_path: StringProperty(name="Depth NPY", subtype="FILE_PATH")
    mask_path: StringProperty(name="Mask", subtype="FILE_PATH")
    bbox_x: FloatProperty(name="BBox Center X", min=0.0, max=1.0, default=0.5)
    bbox_y: FloatProperty(name="BBox Center Y", min=0.0, max=1.0, default=0.5)
    bbox_width: FloatProperty(name="BBox Width", min=0.0, max=1.0, default=0.0)
    bbox_height: FloatProperty(name="BBox Height", min=0.0, max=1.0, default=0.0)

    @classmethod
    def poll(cls, context):
        obj = _active_object(context)
        return obj is not None and obj.type == "CAMERA"

    def invoke(self, context, event):
        obj = _active_object(context)
        bbox = obj.get("bim_norm_bbox", (0.5, 0.5, 0.0, 0.0))
        self.target_object_id = str(obj.get("bim_target_object_id", ""))
        self.view_id = str(obj.get("bim_view_id", obj.name))
        self.rgb_path = str(obj.get("bim_rgb_path", ""))
        self.depth_path = str(obj.get("bim_depth_path", ""))
        self.mask_path = str(obj.get("bim_mask_path", ""))
        self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height = (
            float(value) for value in bbox
        )
        return context.window_manager.invoke_props_dialog(self, width=560)

    def execute(self, context):
        obj = _active_object(context)
        _set_role(obj, ROLE_OBSERVATION_CAMERA)
        obj["bim_target_object_id"] = self.target_object_id.strip()
        obj["bim_view_id"] = self.view_id.strip()
        obj["bim_rgb_path"] = self.rgb_path
        obj["bim_depth_path"] = self.depth_path
        obj["bim_mask_path"] = self.mask_path
        obj["bim_norm_bbox"] = (
            self.bbox_x,
            self.bbox_y,
            self.bbox_width,
            self.bbox_height,
        )
        self.report({"INFO"}, f"Marked {obj.name} as observation camera")
        return {"FINISHED"}


class BIMPOSE_OT_clear_role(Operator):
    bl_idname = "bim_pose.clear_role"
    bl_label = "Clear BIM Pose Role"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_object(context)
        return obj is not None and ROLE_KEY in obj

    def execute(self, context):
        obj = _active_object(context)
        del obj[ROLE_KEY]
        self.report({"INFO"}, f"Cleared role from {obj.name}")
        return {"FINISHED"}


class BIMPOSE_OT_validate_scene(Operator):
    bl_idname = "bim_pose.validate_scene"
    bl_label = "Validate Annotation Scene"
    bl_description = "Check roles, identifiers, transforms and source paths"

    def execute(self, context):
        errors, warnings = validate_scene(context.scene)
        context.scene["bim_last_validation"] = (
            f"{len(errors)} error(s), {len(warnings)} warning(s)"
        )
        if errors:
            self.report({"ERROR"}, errors[0])
            for message in errors:
                print(f"BIM_POSE_ERROR: {message}")
            for message in warnings:
                print(f"BIM_POSE_WARNING: {message}")
            return {"CANCELLED"}
        for message in warnings:
            print(f"BIM_POSE_WARNING: {message}")
        self.report(
            {"WARNING" if warnings else "INFO"},
            f"Valid scene: {len(warnings)} warning(s)",
        )
        return {"FINISHED"}


class BIMPOSE_OT_export_manifest(Operator):
    bl_idname = "bim_pose.export_manifest"
    bl_label = "Export Annotation Manifest"
    bl_description = "Validate and export TRELLIS-to-3DGS pose ground truth"

    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        default_name = str(getattr(context.scene, "bim_scene_id", "scene")) or "scene"
        if not self.filepath:
            blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
            self.filepath = str(blend_dir / f"{default_name}_pose_annotations.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            path = export_scene_manifest(context.scene, self.filepath, strict=True)
        except AnnotationError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.scene["bim_last_manifest"] = str(path)
        self.report({"INFO"}, f"Exported {path.name}")
        return {"FINISHED"}


class BIMPOSE_PT_annotation(Panel):
    bl_label = "BIM Pose Annotation"
    bl_idname = "BIMPOSE_PT_annotation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BIM Pose"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "bim_scene_id", text="Scene ID")
        layout.prop(scene, "bim_up_axis", text="Up Axis")

        active = _active_object(context)
        box = layout.box()
        box.label(text=f"Active: {active.name if active else 'None'}")
        if active:
            role = str(active.get(ROLE_KEY, "unassigned"))
            box.label(text=f"Role: {role}")
            box.operator(BIMPOSE_OT_mark_gs_reference.bl_idname)
            box.operator(BIMPOSE_OT_mark_trellis_proxy.bl_idname)
            if active.type == "CAMERA":
                box.operator(BIMPOSE_OT_mark_observation_camera.bl_idname)
            if ROLE_KEY in active:
                box.operator(BIMPOSE_OT_clear_role.bl_idname)

        layout.separator()
        layout.operator(BIMPOSE_OT_validate_scene.bl_idname, icon="CHECKMARK")
        layout.operator(BIMPOSE_OT_export_manifest.bl_idname, icon="EXPORT")
        layout.label(
            text=str(scene.get("bim_last_validation", "Not validated")),
            icon="INFO",
        )
        last_manifest = str(scene.get("bim_last_manifest", ""))
        if last_manifest:
            layout.label(text=Path(last_manifest).name, icon="FILE_TICK")


def _ensure_scene_defaults() -> None:
    bpy.types.Scene.bim_scene_id = StringProperty(
        name="Scene ID",
        description="Stable scene identifier used in the dataset manifest",
        default="",
    )
    bpy.types.Scene.bim_up_axis = IntProperty(
        name="Up Axis",
        description="3DGS world up axis: 0=X, 1=Y, 2=Z",
        default=2,
        min=0,
        max=2,
    )


def _remove_scene_properties() -> None:
    if hasattr(bpy.types.Scene, "bim_scene_id"):
        del bpy.types.Scene.bim_scene_id
    if hasattr(bpy.types.Scene, "bim_up_axis"):
        del bpy.types.Scene.bim_up_axis


CLASSES = (
    BIMPOSE_OT_mark_gs_reference,
    BIMPOSE_OT_mark_trellis_proxy,
    BIMPOSE_OT_mark_observation_camera,
    BIMPOSE_OT_clear_role,
    BIMPOSE_OT_validate_scene,
    BIMPOSE_OT_export_manifest,
    BIMPOSE_PT_annotation,
)


def register():
    _ensure_scene_defaults()
    for cls in CLASSES:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _remove_scene_properties()


if __name__ == "__main__":
    register()
