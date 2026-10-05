"""Build a self-contained single-leg .blend from a completed asset directory.

Run in a separate Blender background process, for example::

    blender -b --factory-startup --python stocking/code/run_single_leg_scene.py -- \
        --asset-dir stocking/assets/single-leg-fit-v1/example

The parent Blender document is never loaded.  Input arrays use millimetres;
Blender mesh coordinates use metres.  Only the input leg and fitted X1 are
placed in the saved scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")

from stocking_material.asset_io import load_garment_pair
from stocking_material.fitting_preview import _mesh_object, _plain_material, _shell_material


def _load_target(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing single-leg target: {path}")
    with np.load(path, allow_pickle=False) as source:
        if not {"positions_mm", "triangles"} <= set(source.files):
            raise ValueError("target.npz must contain positions_mm and triangles")
        raw_positions = source["positions_mm"]
        raw_triangles = source["triangles"]
    if raw_positions.ndim != 2 or raw_positions.shape[1] != 3 or len(raw_positions) < 3:
        raise ValueError("target positions_mm must be a nonempty (N, 3) array")
    if raw_positions.dtype.kind not in "fiu":
        raise ValueError("target positions_mm must contain real numbers")
    positions = np.asarray(raw_positions, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("target positions_mm contains nonfinite coordinates")
    if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or len(raw_triangles) < 1:
        raise ValueError("target triangles must be a nonempty (M, 3) array")
    if raw_triangles.dtype.kind not in "iu":
        raise ValueError("target triangles must contain integer vertex indices")
    triangles = np.asarray(raw_triangles, dtype=np.int64)
    if np.any(triangles < 0) or np.any(triangles >= len(positions)):
        raise ValueError("target triangles reference an out-of-range vertex")
    if np.any((triangles[:, 0] == triangles[:, 1]) |
              (triangles[:, 1] == triangles[:, 2]) |
              (triangles[:, 2] == triangles[:, 0])):
        raise ValueError("target triangles contain a repeated vertex")
    return positions, triangles


def build_scene(asset_dir):
    """Load validated arrays, reset Blender, and save only their two meshes."""
    import bpy

    directory = Path(asset_dir).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Asset directory does not exist: {directory}")
    pair = load_garment_pair(directory / "garment_pair")
    positions, triangles = _load_target(directory / "target.npz")

    # This script runs as a child Blender process.  Resetting here also makes
    # its output independent of startup-file preferences and enabled add-ons.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if scene is None or len(bpy.data.scenes) != 1 or len(bpy.data.objects) != 0:
        raise RuntimeError("Blender did not provide an empty factory scene")
    scene.name = "Stocking | Single leg fitted X1"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"

    body_material = _plain_material("Single leg | input surface", (.55, .36, .28), .78)
    shell_material = _shell_material()
    body = _mesh_object(scene, "Single leg | input", positions, triangles,
                        material=body_material)
    shell = _mesh_object(scene, "Stocking | fitted X1", pair.current_positions_mm,
                         pair.triangles, pair=pair, material=shell_material)
    body["stocking_role"] = "single_leg_input_target"
    shell["stocking_role"] = "single_leg_static_X1"
    scene["stocking_generation"] = "single static leg geometric fitting"
    shell.select_set(True)
    scene.view_layers[0].objects.active = shell

    extent = np.vstack((positions, pair.current_positions_mm))
    lower, upper = extent.min(axis=0) / 1000.0, extent.max(axis=0) / 1000.0
    center = (lower + upper) / 2.0
    distance = max(1.0, float(np.linalg.norm(upper - lower) * 1.7))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.region_3d.view_location = center
                space.region_3d.view_distance = distance
                space.shading.color_type = "MATERIAL"

    output = directory / "single-leg-fit.blend"
    if output.exists():
        raise FileExistsError(f"Scene file already exists: {output}")
    temporary = directory / ".single-leg-fit-building.blend"
    if temporary.exists():
        raise FileExistsError(f"Temporary scene file already exists: {temporary}")
    try:
        bpy.context.preferences.filepaths.file_preview_type = "NONE"
        result = bpy.ops.wm.save_as_mainfile(filepath=str(temporary), check_existing=False)
        if "FINISHED" not in result or not temporary.is_file():
            raise RuntimeError("Blender did not save the isolated scene")
        temporary.replace(output)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise
    return {
        "output": str(output),
        "target_vertices": int(len(positions)),
        "stocking_vertices": int(len(pair.current_positions_mm)),
        "stocking_triangles": int(len(pair.triangles)),
        "scene_count": int(len(bpy.data.scenes)),
        "object_count": int(len(bpy.data.objects)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True,
                        help="asset folder containing garment_pair.npz/json and target.npz")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    result = build_scene(args.asset_dir)
    print("SINGLE_LEG_SCENE_SAVED " + json.dumps(result, ensure_ascii=False, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
