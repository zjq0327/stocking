"""Save the extracted frame-996 stocking as a single, ready-to-view Blender scene.

Run with Blender after loading assets/authored-target-v2/authored_target.blend.
The source scene and authored-target-v2 assets are read only.
"""

import hashlib
import json
import shutil
from pathlib import Path

import bpy


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
SOURCE = ROOT / "stocking/assets/authored-target-v2"
OUTPUT = ROOT / "stocking/assets/authored-target-posed-v1"
RENDER = ROOT / "render/authored-target-posed-v1/stocking-original-pose.png"
SCENE_NAME = "Authored target | current"
TARGET_NAME = "Stocking target | current"
DECORATION_NAME = "Decoration excluded from target | current"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    source_blend = SOURCE / "authored_target.blend"
    if Path(bpy.data.filepath).resolve() != source_blend.resolve():
        raise ValueError(f"Open the audited v2 preview first: {source_blend}")
    metadata_path = SOURCE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "stocking.authored_target.v2" or metadata.get("source_frame") != 996:
        raise ValueError("Expected the extracted frame-996 v2 target")
    original_blend = Path(metadata["source_file"])
    if sha256(original_blend) != metadata["source_sha256"]:
        raise ValueError("The original .blend changed after v2 extraction")
    input_hashes = {
        "original_blend_sha256": metadata["source_sha256"],
        "v2_blend_sha256": sha256(source_blend),
        "v2_metadata_sha256": sha256(metadata_path),
        "v2_npz_sha256": sha256(SOURCE / "authored_target.npz"),
        "v2_current_obj_sha256": sha256(SOURCE / "authored_target_current_mm.obj"),
    }

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.camera is None:
        raise ValueError("Current-pose scene or camera is missing")
    target = scene.objects.get(TARGET_NAME)
    decoration = scene.objects.get(DECORATION_NAME)
    if target is None or target.type != "MESH" or decoration is None or decoration.type != "MESH":
        raise ValueError("Current-pose target or toe decoration is missing")
    if len(target.data.vertices) != 60260 or len(target.data.polygons) != 120202:
        raise ValueError("Current-pose mesh no longer matches the audited v2 topology")

    # Make current the only scene and remove datablocks for the unposed sock.
    if bpy.context.window:
        bpy.context.window.scene = scene
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other, do_unlink=True)
    for obj in list(bpy.data.objects):
        if obj.name not in scene.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if len(bpy.data.scenes) != 1 or any("unposed" in obj.name.lower() for obj in bpy.data.objects):
        raise RuntimeError("An unposed scene or object remains")
    scene.frame_set(996)
    scene["source_frame"] = 996

    # Keep captions in the render, while presenting a clean, selected mesh in 3D views.
    for obj in scene.objects:
        if any(label in obj.name for label in (" | Top band", " | Bottom band", " | Title", " | Subtitle")):
            obj.hide_viewport = True
            obj.hide_render = False
        obj.select_set(False)
    target.hide_viewport = False
    target.select_set(True)
    scene.view_layers[0].objects.active = target
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.region_3d.view_perspective = "CAMERA"
                space.region_3d.view_camera_zoom = 0
                space.overlay.show_extras = False
                space.shading.type = "MATERIAL"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    RENDER.parent.mkdir(parents=True, exist_ok=True)
    output_blend = OUTPUT / "stocking-original-pose.blend"
    output_obj = OUTPUT / "stocking-original-pose_mm.obj"
    shutil.copyfile(SOURCE / "authored_target_current_mm.obj", output_obj)
    scene.render.filepath = str(RENDER)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    bpy.context.preferences.filepaths.save_version = 0
    preview_type = bpy.context.preferences.filepaths.file_preview_type
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), copy=True)
    finally:
        bpy.context.preferences.filepaths.file_preview_type = preview_type
    if not output_blend.is_file() or not RENDER.is_file():
        raise RuntimeError("Pose blend or preview render was not written")
    if sha256(original_blend) != input_hashes["original_blend_sha256"] or sha256(source_blend) != input_hashes["v2_blend_sha256"]:
        raise RuntimeError("A source .blend changed during export")

    posed_validation = metadata["validation"]["positions_mm"]
    result = {
        "schema": "stocking.authored_target_posed.v1",
        "pose": "current / original bent-leg pose",
        "source_frame": 996,
        "source_v2_metadata": str(metadata_path),
        "source_v2_npz": str(SOURCE / "authored_target.npz"),
        "input_hashes": input_hashes,
        "outputs": {
            "blend": str(output_blend), "obj_mm": str(output_obj), "preview_png": str(RENDER)
        },
        "output_hashes": {
            "blend_sha256": sha256(output_blend),
            "obj_mm_sha256": sha256(output_obj),
            "preview_png_sha256": sha256(RENDER),
        },
        "scene": SCENE_NAME,
        "target_object": TARGET_NAME,
        "toe_decoration_object": DECORATION_NAME,
        "camera_direction": [-1.2, 1.7, 0.55],
        "coordinate_contract": "OBJ and source NPZ in millimetres; Blender mesh vertices in metres",
        "uv_contract": "Retained source texture UV; not knitting material coordinates",
        "geometry_role": "Extracted and locally repaired worn macro target; not unworn X0 or equilibrium yarn mesh",
        "posed_intersection_limitations": posed_validation["intersections"],
        "physical_solver_ready": False,
        "material_X0_correspondence_available": False,
        "dressing_equilibrium_solved": False,
    }
    write_json(OUTPUT / "metadata.json", result)
    print("POSED_TARGET_EXPORTED", json.dumps(result["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
