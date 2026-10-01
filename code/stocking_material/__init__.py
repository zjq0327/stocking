"""Parametric plain-knit authoring and two-sided geometric feature baking."""

bl_info = {
    "name": "Stocking Material Authoring",
    "author": "Stocking project; centerline model by Keenan Crane",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Stocking",
    "description": "Create plain-knit yarn geometry and bake front/back ID, P, N, T",
    "category": "Material",
}


def register():
    from . import blender_ui

    blender_ui.register()


def unregister():
    from . import blender_ui

    blender_ui.unregister()
