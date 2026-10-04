"""Generate an independent unworn stocking initial shell and optional preview.

Pure Python: python run_reference.py -- --preset presets/garment-reference.json
Blender: blender -b --python run_reference.py -- --save-blend NEW.blend --render OUT.png
No existing scene is edited; preview is created in a separate scene.
"""
import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")

from stocking_material.asset_io import save_garment_pair
from stocking_material.procedural_garment import (
    ProceduralGarmentParameters, generate_reference_garment, export_reference_obj, create_reference_scene,
)


def main():
    parser = argparse.ArgumentParser(description="Generate an uncalibrated complete unworn stocking X0 shell")
    parser.add_argument("--preset", type=Path, default=CODE / "presets" / "garment-reference.json")
    parser.add_argument("--output", type=Path, default=CODE.parent / "assets" / "procedural-reference-v1")
    parser.add_argument("--save-blend", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--render-detail", type=Path, help="optional foot/heel/toe topology close-up")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    params = ProceduralGarmentParameters.load(args.preset)
    pair = generate_reference_garment(params)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = save_garment_pair(pair, args.output / "garment_reference")
    outputs["obj"] = export_reference_obj(pair, args.output / "garment_reference.obj")
    parameter_path = args.output / "parameters.json"
    parameter_path.write_text(json.dumps(params.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    outputs["parameters"] = str(parameter_path)
    if args.save_blend or args.render or args.render_detail:
        try:
            import bpy
        except ImportError:
            parser.error("--save-blend/--render require Blender; NPZ/JSON/OBJ have already been generated")
        scene, obj = create_reference_scene(pair, params)
        if args.save_blend:
            target = args.save_blend.resolve()
            if bpy.data.filepath and Path(bpy.data.filepath).resolve() == target:
                raise ValueError("Save to a new .blend copy; the currently loaded document cannot be overwritten")
            target.parent.mkdir(parents=True, exist_ok=True)
            original_filepath = bpy.data.filepath
            preview_setting = bpy.context.preferences.filepaths.file_preview_type
            bpy.context.preferences.filepaths.file_preview_type = "NONE"
            try:
                bpy.ops.wm.save_as_mainfile(filepath=str(target), copy=True)
            finally:
                bpy.context.preferences.filepaths.file_preview_type = preview_setting
            if bpy.data.filepath != original_filepath:
                raise RuntimeError("Blender copy save unexpectedly changed the current document filepath")
            outputs["blend"] = str(target)
        if args.render:
            args.render.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(args.render.resolve())
            bpy.ops.render.render(write_still=True, scene=scene.name)
            outputs["render"] = str(args.render.resolve())
        if args.render_detail:
            from mathutils import Vector
            span = params.foot_length_mm + .5 * params.ankle_width_mm
            target = Vector((.45 * params.foot_length_mm, 0., .4 * params.heel_bend_radius_mm))
            scene.camera.location = target + Vector((-.2 * span, -1.6 * span, .8 * span))
            scene.camera.rotation_euler = (target - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.camera.data.ortho_scale = span * 1.25
            scene.render.resolution_x, scene.render.resolution_y = 1200, 900
            args.render_detail.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(args.render_detail.resolve())
            bpy.ops.render.render(write_still=True, scene=scene.name)
            outputs["render_detail"] = str(args.render_detail.resolve())
    print(json.dumps(dict(outputs=outputs, topology=pair.metadata["topology_summary"],
                          reference_role="generated_initial", synthetic_uncalibrated=True), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
