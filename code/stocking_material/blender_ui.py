"""Blender sidebar for rebuilding geometry and exporting geometric features."""

import json
import traceback
from pathlib import Path

import bpy
from bpy.props import FloatProperty, IntProperty, PointerProperty, StringProperty

from .parameters import Parameters
from .scene import PARAMETERS_KEY, build_scene


STOCKING_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = STOCKING_ROOT.parent
PARAMETER_NAMES = (
    "a", "h", "d", "R", "rowOffset", "scale_mm", "nRows", "nLoops",
    "samples_per_loop", "tube_sides", "width", "height",
)


class STOCKING_PG_settings(bpy.types.PropertyGroup):
    a: FloatProperty(name="圆润程度 a", default=1.5, min=0.0, precision=3)
    h: FloatProperty(name="高度振幅 h", default=4.0, min=0.001, precision=3)
    d: FloatProperty(name="交织深度振幅 d", default=1.0, min=0.001, precision=3)
    R: FloatProperty(name="圆管半径 R", default=0.5, min=0.001, precision=3)
    rowOffset: FloatProperty(name="横列间距", default=4.5, min=0.001, precision=3)
    scale_mm: FloatProperty(name="每单位对应毫米", default=0.05, min=0.000001, precision=5)
    nRows: IntProperty(name="预览行数", default=5, min=1, max=100)
    nLoops: IntProperty(name="每行线圈数", default=5, min=1, max=100)
    samples_per_loop: IntProperty(name="每圈采样点", default=192, min=16, max=4096)
    tube_sides: IntProperty(name="圆管边数", default=24, min=6, max=256)
    width: IntProperty(name="特征图宽度", default=256, min=8, max=8192)
    height: IntProperty(name="特征图高度", default=256, min=8, max=8192)
    asset_dir: StringProperty(
        name="资产目录", subtype="DIR_PATH",
        default=str(STOCKING_ROOT / "assets" / "plain-knit-v1"),
    )
    preview_dir: StringProperty(
        name="预览目录", subtype="DIR_PATH",
        default=str(PROJECT_ROOT / "render" / "plain-knit-v1"),
    )
    status: StringProperty(name="状态", default="就绪")

    def parameters(self):
        return Parameters.from_dict({name: getattr(self, name) for name in PARAMETER_NAMES})


class STOCKING_OT_rebuild(bpy.types.Operator):
    bl_idname = "stocking_material.rebuild"
    bl_label = "重建三维针织样片"
    bl_description = "按当前参数生成连续纱线与圆管表面"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.stocking_material
        try:
            info = build_scene(settings.parameters(), scene=context.scene)
            settings.status = "几何已生成：{:,} 个三角形".format(info["triangle_count"])
            for obj in context.selected_objects:
                obj.select_set(False)
            info["mesh_object"].select_set(True)
            context.view_layer.objects.active = info["mesh_object"]
            self.report({"INFO"}, settings.status)
        except Exception as exc:
            traceback.print_exc()
            settings.status = "错误：" + str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class STOCKING_OT_bake_export(bpy.types.Operator):
    bl_idname = "stocking_material.bake_export"
    bl_label = "采样并导出 ID / P / N / T"
    bl_description = "生成正反两面的特征资产与预览图，保存到下方目录"

    def execute(self, context):
        from .pipeline import build_asset

        settings = context.scene.stocking_material
        try:
            if not settings.asset_dir.strip() or not settings.preview_dir.strip():
                raise ValueError("请先指定资产目录和预览目录")
            p = settings.parameters()
            context.scene[PARAMETERS_KEY] = json.dumps(p.to_dict(), ensure_ascii=False)
            build_asset(
                p,
                Path(bpy.path.abspath(settings.asset_dir)),
                Path(bpy.path.abspath(settings.preview_dir)),
                render_preview=False,
                save_blend=False,
            )
            settings.status = "已导出：" + str(Path(bpy.path.abspath(settings.asset_dir)))
            self.report({"INFO"}, "正反面 ID / P / N / T 特征已导出")
        except Exception as exc:
            traceback.print_exc()
            settings.status = "错误：" + str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class STOCKING_PT_authoring(bpy.types.Panel):
    bl_label = "平针制作 · ID / P / N / T"
    bl_idname = "STOCKING_PT_authoring"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stocking"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stocking_material
        layout.label(text="Crane 中心线 · 静态样片")
        shape = layout.box()
        shape.label(text="纱线结构")
        for name in ("a", "h", "d", "R", "rowOffset", "scale_mm"):
            shape.prop(settings, name)
        shape.label(text="针距：{:.4f} mm".format(6.283185307179586 * settings.scale_mm))
        quality = layout.box()
        quality.label(text="几何精度与采样尺寸")
        for names in (("nRows", "nLoops"), ("samples_per_loop", "tube_sides"), ("width", "height")):
            row = quality.row(align=True)
            for name in names:
                row.prop(settings, name)
        layout.operator(STOCKING_OT_rebuild.bl_idname, icon="MESH_GRID")
        output = layout.box()
        output.label(text="输出位置")
        output.prop(settings, "asset_dir")
        output.prop(settings, "preview_dir")
        output.operator(STOCKING_OT_bake_export.bl_idname, icon="EXPORT")
        layout.label(text="保存 .blend 可保留当前样片与参数。")
        layout.label(text=settings.status)


CLASSES = (STOCKING_PG_settings, STOCKING_OT_rebuild, STOCKING_OT_bake_export, STOCKING_PT_authoring)


@bpy.app.handlers.persistent
def _restore_saved_parameters(_unused=None):
    for scene in bpy.data.scenes:
        raw = scene.get(PARAMETERS_KEY)
        if not raw:
            continue
        try:
            p = Parameters.from_dict(json.loads(raw))
            for name in PARAMETER_NAMES:
                setattr(scene.stocking_material, name, getattr(p, name))
        except (ValueError, TypeError, KeyError) as exc:
            scene.stocking_material.status = "保存的参数未能载入：" + str(exc)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.stocking_material = PointerProperty(type=STOCKING_PG_settings)
    if _restore_saved_parameters not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_restore_saved_parameters)
    _restore_saved_parameters()


def unregister():
    if _restore_saved_parameters in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_saved_parameters)
    if hasattr(bpy.types.Scene, "stocking_material"):
        del bpy.types.Scene.stocking_material
    for cls in reversed(CLASSES):
        if getattr(cls, "is_registered", False):
            bpy.utils.unregister_class(cls)
