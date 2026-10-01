"""Blender preview scene. All created objects belong to one owned collection."""

import json
from pathlib import Path

import bpy
from mathutils import Vector

from .geometry import generate_preview_geometry


OWNER_KEY = "stocking_material_owner"
OWNER_VALUE = "plain-knit-v1"
PARAMETERS_KEY = "stocking_material_parameters"


def _remove_owned_collection(scene):
    """Remove only this tool's collection linked to the requested scene."""
    for collection in list(scene.collection.children):
        if collection.get(OWNER_KEY) != OWNER_VALUE:
            continue
        if scene.camera is not None and scene.camera in collection.objects.values():
            scene.camera = None
        scene.collection.children.unlink(collection)
        # Collections linked elsewhere may be used by other scenes. Leave them intact.
        if collection.users:
            continue
        objects = list(collection.objects)
        bpy.data.collections.remove(collection)
        for obj in objects:
            if obj.users:
                continue
            data = obj.data
            bpy.data.objects.remove(obj)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Light):
                    bpy.data.lights.remove(data)


def _object(collection, name, data):
    obj = bpy.data.objects.new(name, data)
    obj[OWNER_KEY] = OWNER_VALUE
    collection.objects.link(obj)
    return obj


def _aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material():
    name = "Stocking | Yarn preview"
    material = next(
        (m for m in bpy.data.materials if m.get(OWNER_KEY) == OWNER_VALUE), None
    )
    if material is None:
        material = bpy.data.materials.new(name)
        material[OWNER_KEY] = OWNER_VALUE
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.68, 0.82, 0.91, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.36
        material.diffuse_color = (0.68, 0.82, 0.91, 1.0)
    return material


def build_scene(p, geometry=None, scene=None):
    """Create a millimeter preview; return Blender objects and mesh statistics.

    Existing objects are retained. Only a collection marked with OWNER_KEY is
    replaced. Parameters are stored as JSON on the scene for reopening the file.
    This function never saves, renames, or renders the current document.
    """
    p.validate()
    scene = scene or bpy.context.scene
    geometry = geometry if geometry is not None else generate_preview_geometry(p)
    _remove_owned_collection(scene)
    collection = bpy.data.collections.new("Stocking | Plain knit")
    collection[OWNER_KEY] = OWNER_VALUE
    scene.collection.children.link(collection)

    mesh = bpy.data.meshes.new("Stocking | Yarn surface")
    mesh.from_pydata(geometry.vertices.tolist(), [], geometry.triangles.tolist())
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if hasattr(mesh, "normals_split_custom_set_from_vertices"):
        mesh.normals_split_custom_set_from_vertices(geometry.normals.tolist())
    # Preserve the actual unit yarn direction in the saved mesh as an attribute.
    tangent_attribute = mesh.attributes.new("yarn_tangent", "FLOAT_VECTOR", "POINT")
    tangent_attribute.data.foreach_set("vector", geometry.tangents.ravel())
    id_attribute = mesh.attributes.new("yarn_type_id", "INT", "FACE")
    id_attribute.data.foreach_set("value", geometry.yarn_ids.ravel())
    yarn = _object(collection, "Stocking | Plain-knit yarn", mesh)
    yarn.data.materials.append(_material())
    yarn["coordinates"] = "local XYZ in millimeters"
    yarn[PARAMETERS_KEY] = json.dumps(p.to_dict(), ensure_ascii=False)

    low = Vector(geometry.vertices.min(axis=0).tolist())
    high = Vector(geometry.vertices.max(axis=0).tolist())
    center = (low + high) * 0.5
    span = max((high - low).x, (high - low).y, 0.01)
    camera_data = bpy.data.cameras.new("Stocking | Preview camera")
    camera = _object(collection, "Stocking | Preview camera", camera_data)
    camera.location = center + Vector((0.20 * span, -0.62 * span, 1.6 * span))
    _aim(camera, center)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.35
    camera_data.clip_start = max(span * 0.0001, 0.000001)
    camera_data.clip_end = max(span * 100, 10.0)

    for name, offset, power, size in (
        ("Key", (-0.5, -0.8, 1.2), 100.0, 1.0),
        ("Fill", (0.8, 0.6, 0.7), 45.0, 0.9),
    ):
        light_data = bpy.data.lights.new("Stocking | " + name, "AREA")
        light_data.energy = power * span * span
        light_data.shape = "DISK"
        light_data.size = span * size
        light = _object(collection, "Stocking | " + name, light_data)
        light.location = center + Vector(offset) * span
        _aim(light, center)

    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene[PARAMETERS_KEY] = json.dumps(p.to_dict(), ensure_ascii=False)
    if scene.camera is None or scene.camera.get(OWNER_KEY) == OWNER_VALUE:
        scene.camera = camera
    return {
        "scene": scene,
        "collection": collection,
        "mesh_object": yarn,
        "camera": camera,
        "vertex_count": len(geometry.vertices),
        "triangle_count": len(geometry.triangles),
    }


def render_preview(scene_info, preview_dir):
    """Render just this tool's collection without changing the current scene."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    output = preview_dir / "geometry.png"
    scene = bpy.data.scenes.new("Stocking | Temporary preview")
    world = bpy.data.worlds.new("Stocking | Temporary world")
    try:
        scene.collection.children.link(scene_info["collection"])
        scene.camera = scene_info["camera"]
        scene.world = world
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        background.inputs["Color"].default_value = (0.055, 0.075, 0.10, 1.0)
        background.inputs["Strength"].default_value = 0.45
        engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        scene.render.resolution_x = 900
        scene.render.resolution_y = 900
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = False
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        bpy.data.scenes.remove(scene)
        if world.users == 0:
            bpy.data.worlds.remove(world)
    return output
