"""Bind the independent unworn X0 to both authored stocking target poses.

Blender -b --factory-startup --python run_authored_fitting.py -- [--output DIR]
The output is a geometric candidate with no yarn or dressing equilibrium.
"""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")

from stocking_material.asset_io import save_garment_pair
from stocking_material.authored_registration import register_authored_pair
from stocking_material.deformation_field import compute_deformation_field
from stocking_material.procedural_garment import ProceduralGarmentParameters, generate_reference_garment


ACCEPTED_DIMENSIONS = {
    "leg_length_mm": 640., "foot_length_mm": 150.,
    "heel_bend_radius_mm": 40., "toe_length_mm": 35.,
    "circumference_segments": 128, "leg_segments": 128,
    "heel_segments": 32, "foot_segments": 32, "toe_segments": 16,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def save_field(field, directory, basename):
    arrays = {item.name: getattr(field, item.name) for item in fields(field)
              if isinstance(getattr(field, item.name), np.ndarray)}
    path = directory / (basename + ".npz")
    np.savez_compressed(path, **arrays)
    write_json(directory / (basename + ".json"), {"metadata": field.metadata,
               "array_file": path.name, "array_sha256": sha256(path),
               "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for name, value in arrays.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "stocking/assets/authored-fit-v1")
    parser.add_argument("--dense-angles", type=int, default=256)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.dense_angles < 128 or args.dense_angles > 1024:
        parser.error("--dense-angles must be 128..1024")

    source = ROOT / "stocking/assets/authored-target-v2"
    source_npz = source / "authored_target.npz"
    source_metadata = source / "metadata.json"
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata.get("schema") != "stocking.authored_target.v2" or metadata.get("source_frame") != 996:
        raise ValueError("The target must be the audited frame-996 authored-target-v2")
    if sha256(Path(metadata["source_file"])) != metadata["source_sha256"]:
        raise ValueError("Original .blend changed since target extraction")
    with np.load(source_npz, allow_pickle=False) as archive:
        target = {key: archive[key].copy() for key in
                  ("triangles", "unposed_positions_mm", "positions_mm", "component")}
    preset = ROOT / "stocking/code/presets/garment-reference.json"
    parameter_values = json.loads(preset.read_text(encoding="utf-8"))
    parameter_values.update(ACCEPTED_DIMENSIONS)
    params = ProceduralGarmentParameters.from_dict(parameter_values)
    params_dict = params.to_dict()
    source_hashes = {"original_blend_sha256": metadata["source_sha256"],
                     "authored_target_npz_sha256": sha256(source_npz),
                     "authored_target_metadata_sha256": sha256(source_metadata),
                     "reference_preset_sha256": sha256(preset)}
    results = {}
    for side in ("left", "right"):
        print(f"[authored-fit] registering {side}", flush=True)
        reference = generate_reference_garment(params)
        reference.metadata["side"] = side
        result = register_authored_pair(reference, target, metadata, side,
                                        dense_angles=args.dense_angles)
        field_u = compute_deformation_field(result.pair_unposed)
        field_c = compute_deformation_field(result.pair_current)
        results[side] = (reference, result, field_u, field_c)
        print(f"[authored-fit] {side} ready; current flips={result.report['current']['nonpositive_target_normal_faces']}",
              flush=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "stocking.authored_fit.v1", "stage": "geometric_candidate",
                "target_frame": 996, "source_hashes": source_hashes,
                "reference_parameters": params_dict, "dense_angles": args.dense_angles,
                "target": str(source_npz), "sides": {},
                "target_existing_intersections": metadata["validation"]["positions_mm"]["intersections"],
                "dressing_equilibrium_solved": False, "yarn_equilibrium_solved": False,
                "material_calibrated": False,
                "scope": "X0 material shell fitted to the authored surface in unposed and frame-996 poses; no mechanics"}
    for side, (reference, result, field_u, field_c) in results.items():
        folder = output / side
        folder.mkdir(parents=True, exist_ok=True)
        save_garment_pair(reference, folder / "garment_reference")
        save_garment_pair(result.pair_unposed, folder / "garment_pair_unposed")
        save_garment_pair(result.pair_current, folder / "garment_pair")
        np.savez_compressed(folder / "binding.npz", **result.binding)
        write_json(folder / "binding.json", {"schema": "stocking.authored_target_binding.v1",
                   "target_npz_sha256": source_hashes["authored_target_npz_sha256"],
                   "target_triangle_index": "global face index in authored-target-v2/authored_target.npz",
                   "target_barycentric": "same three weights on unposed and frame-996 target triangles",
                   "normal_offset_mm": "zero in this geometric-candidate version",
                   "method_codes": result.report["method_codes"],
                   "array_sha256": sha256(folder / "binding.npz")})
        save_field(field_u, folder, "deformation_field_unposed")
        save_field(field_c, folder, "deformation_field")
        np.savez_compressed(folder / "quality_unposed.npz", **result.quality_unposed)
        np.savez_compressed(folder / "quality_current.npz", **result.quality_current)
        write_json(folder / "validation.json", result.report)
        manifest["sides"][side] = {"reference_version": reference.metadata["reference_version"],
                                   "material_version": reference.metadata["material_version"],
                                   "material_points": len(reference.material_point_ids),
                                   "faces": len(reference.triangles),
                                   "validation": result.report,
                                   "hashes": {name: sha256(folder / name) for name in
                                              ("garment_reference.npz", "garment_pair_unposed.npz",
                                               "garment_pair.npz", "binding.npz",
                                               "deformation_field_unposed.npz", "deformation_field.npz")}}
    write_json(output / "manifest.json", manifest)
    report = ROOT / "build-support/authored-stocking-target/authored-fitting-report.json"
    write_json(report, manifest)
    if sha256(Path(metadata["source_file"])) != source_hashes["original_blend_sha256"] or sha256(source_npz) != source_hashes["authored_target_npz_sha256"]:
        raise RuntimeError("An input changed during fitting")
    print("AUTHORED_FIT_EXPORTED", json.dumps({"output": str(output), "report": str(report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
