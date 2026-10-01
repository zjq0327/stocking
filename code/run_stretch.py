"""Blender batch entry: blender -b --factory-startup --python run_stretch.py -- [options]."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
sys.pycache_prefix = str(CODE.parent.parent / "build-support" / "stocking-material" / "pycache")
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def main():
    import bpy
    from stocking_material.parameters import Parameters
    from stocking_material.stretch_parameters import StretchParameters
    from stocking_material.stretch_pipeline import build_stretched_asset

    parser = argparse.ArgumentParser(description="Solve periodic yarn stretch and bake reference/final ID, P, N, T")
    parser.add_argument("--preset", type=Path, default=CODE / "presets" / "plain-knit.json")
    parser.add_argument("--stretch-preset", type=Path, default=CODE / "presets" / "stretch-x.json")
    parser.add_argument("--lambda-x", type=float, help="Override the horizontal prescribed stretch ratio")
    parser.add_argument("--lambda-y", type=float, help="Override the vertical prescribed stretch ratio")
    parser.add_argument("--output", type=Path, default=CODE.parent / "assets" / "plain-knit-stretch-v2")
    parser.add_argument("--preview", type=Path, default=CODE.parent.parent / "render" / "plain-knit-stretch-v2")
    parser.add_argument("--resolution", type=int, help="Override both dimensions of both comparison states")
    parser.add_argument("--render", action="store_true", help="Also render the final 3D yarn swatch")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    params = Parameters.load(args.preset)
    settings = StretchParameters.load(args.stretch_preset)
    if args.resolution is not None:
        params = replace(params, width=args.resolution, height=args.resolution)
    overrides = {name: value for name, value in (("lambda_x", args.lambda_x), ("lambda_y", args.lambda_y)) if value is not None}
    if overrides:
        settings = replace(settings, **overrides)
    scene = bpy.data.scenes.new("Stocking | Stretched yarn asset")
    bpy.context.window.scene = scene
    from stocking_material import register
    register()
    build_stretched_asset(params, settings, args.output, args.preview,
                          render_preview=args.render, save_blend=True)


if __name__ == "__main__":
    main()
