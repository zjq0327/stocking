"""Run with Blender: blender -b --factory-startup --python run.py -- [options]."""
from pathlib import Path
import argparse
import sys

CODE = Path(__file__).resolve().parent
sys.pycache_prefix = str(CODE.parent.parent / "build-support" / "stocking-material" / "pycache")
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def main():
    import bpy
    from stocking_material.parameters import Parameters
    from stocking_material.pipeline import build_asset
    parser = argparse.ArgumentParser(description="Bake a Crane plain-knit surface into two-sided feature maps")
    parser.add_argument("--preset", type=Path, default=CODE / "presets" / "plain-knit.json")
    parser.add_argument("--output", type=Path, default=CODE.parent / "assets" / "plain-knit-v1")
    parser.add_argument("--preview", type=Path, default=CODE.parent.parent / "render" / "plain-knit-v1")
    parser.add_argument("--resolution", type=int, help="Override both output dimensions")
    parser.add_argument("--render", action="store_true", help="Also render a 3D swatch preview")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    params = Parameters.load(args.preset)
    if args.resolution is not None:
        from dataclasses import replace
        params = replace(params, width=args.resolution, height=args.resolution)
    # Batch output gets its own scene, leaving any existing scene untouched.
    scene = bpy.data.scenes.new("Stocking | Plain-knit asset")
    bpy.context.window.scene = scene
    from stocking_material import register
    register()
    build_asset(params, args.output, args.preview, render_preview=args.render, save_blend=True)


if __name__ == "__main__":
    main()
