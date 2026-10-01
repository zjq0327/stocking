"""Open this file in Blender's Text Editor and choose Run Script.

Registration lasts for the current Blender session and does not change startup
preferences or install packages. Run this again after reopening Blender.
"""

import importlib
from pathlib import Path
import sys

import bpy
from mathutils import Vector


code_dir = Path(__file__).resolve().parent
sys.pycache_prefix = str(code_dir.parent.parent / "build-support" / "stocking-material" / "pycache")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

if "stocking_material" in sys.modules:
    module = sys.modules["stocking_material"]
    if hasattr(bpy.types.Scene, "stocking_material"):
        module.unregister()
else:
    module = importlib.import_module("stocking_material")
module.register()


def _authoring_areas():
    """Return visible 3D areas belonging to the current authoring scene."""
    scene = bpy.context.scene
    windows = bpy.context.window_manager.windows
    screens = [window.screen for window in windows if window.scene == scene]
    if not screens and bpy.context.screen is not None:
        screens = [bpy.context.screen]
    return [area for screen in screens for area in screen.areas if area.type == "VIEW_3D"]


def _frame_existing_swatch():
    """Set viewport properties directly; no context-sensitive UI operators."""
    from stocking_material.scene import OWNER_KEY, OWNER_VALUE, PARAMETERS_KEY

    scene = bpy.context.scene
    if not scene.get(PARAMETERS_KEY):
        return False
    collections = [collection for collection in scene.collection.children
                   if collection.get(OWNER_KEY) == OWNER_VALUE]
    yarns = [obj for collection in collections for obj in collection.objects
             if obj.type == "MESH" and obj.get(OWNER_KEY) == OWNER_VALUE]
    if not yarns:
        return False
    points = [obj.matrix_world @ Vector(corner) for obj in yarns for corner in obj.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (low + high) * 0.5
    span = max(max(high - low), 0.001)
    camera = next((obj for collection in collections for obj in collection.objects
                   if obj.type == "CAMERA"), None)
    for area in _authoring_areas():
        space = area.spaces.active
        space.show_region_ui = True
        space.clip_start = max(span * 0.0001, 0.000001)
        space.clip_end = max(span * 100, 10.0)
        space.overlay.show_extras = False
        space.shading.type = "SOLID"
        space.shading.color_type = "MATERIAL"
        view = space.region_3d
        if view is not None:
            view.view_perspective = "ORTHO"
            view.view_location = center
            view.view_distance = span * 2.0
            if camera is not None:
                view.view_rotation = camera.matrix_world.to_quaternion()
        area.tag_redraw()
    return True


def _select_stocking_sidebar():
    # The sidebar categories become available after the first UI draw. This
    # short timer only selects a tab; it never repeatedly changes the view.
    _select_stocking_sidebar.attempts += 1
    selected = False
    for area in _authoring_areas():
        for region in area.regions:
            if region.type == "UI" and hasattr(region, "active_panel_category"):
                try:
                    region.active_panel_category = "Stocking"
                    selected = True
                    area.tag_redraw()
                except (TypeError, ValueError, RuntimeError):
                    pass
    if not selected and _select_stocking_sidebar.attempts < 6:
        return 0.2
    return None


_select_stocking_sidebar.attempts = 0
if _frame_existing_swatch() and not bpy.app.background:
    bpy.app.timers.register(_select_stocking_sidebar, first_interval=0.2)
print("丝袜材质制作面板已就绪：3D 视图 > N 侧栏 > Stocking")
