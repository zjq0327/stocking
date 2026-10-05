"""Blender batch entry: import one static leg-and-foot OBJ and fit a stocking.

Example from the project root::

  blender -b --factory-startup --python stocking/code/run_single_leg.py -- \
      --input leg.obj --output stocking/assets/single-leg-fit-v1 \
      --mm-per-unit 1000 --clearance-mm 0.6

The output argument is a parent folder; each source file receives a unique
child asset directory.  No armature or second leg is required.
"""

import argparse
import json
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")

from stocking_material.single_leg_pipeline import generate_single_leg_from_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="one OBJ leg mesh including the foot")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "stocking/assets/single-leg-fit-v1")
    parser.add_argument("--mm-per-unit", type=float, default=1000.,
                        help="millimetres represented by one imported OBJ coordinate")
    parser.add_argument("--clearance-mm", type=float, default=.6)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    result = generate_single_leg_from_obj(
        args.input, args.output, clearance_mm=args.clearance_mm,
        mm_per_unit=args.mm_per_unit)
    print("SINGLE_LEG_STOCKING_GENERATED", json.dumps({
        "output": str(result["paths"]["metadata.json"].parent),
        "mesh_vertices": result["report"]["mesh_vertices"],
        "mesh_triangles": result["report"]["mesh_triangles"],
        "fitting": result["report"]["fitting"],
    }, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
