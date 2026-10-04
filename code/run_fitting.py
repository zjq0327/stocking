"""Blender batch: complete independent synthetic X0 -> user model X1 fit.

blender -b SOURCE.blend --python stocking/code/run_fitting.py -- [options]
Writes new assets and a new .blend copy. Never overwrites the loaded source.
This validates geometric correspondence/contact samples, not physical donning.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import bpy
import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")

from stocking_material.asset_io import save_garment_pair
from stocking_material.body_target import extract_body_target
from stocking_material.deformation_field import compute_deformation_field
from stocking_material.geometric_fitting import fit_reference_to_body
from stocking_material.procedural_garment import ProceduralGarmentParameters, generate_reference_garment
from stocking_material.fitting_preview import create_fit_preview, render_fit_previews


def main():
    parser = argparse.ArgumentParser(description="Kinematic fitting verification of a complete independent sock shell")
    parser.add_argument("--body", default="body_base")
    parser.add_argument("--rig", default="rig")
    parser.add_argument("--frame", type=int, default=996)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--clearance-mm", type=float, default=.6)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--target-pose", choices=("current", "rest"), default="current")
    parser.add_argument("--report-contact-conflicts", action="store_true", help="keep and report original-pose conflicts; never label these as passed")
    parser.add_argument("--circumference-segments", type=int, default=128)
    parser.add_argument("--leg-segments", type=int, default=128)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    asset_name = "stocking-fit-v1" if args.target_pose == "rest" else "stocking-fit-current-pose-v1"
    args.output = args.output or CODE.parent / "assets" / asset_name
    args.render_dir = args.render_dir or ROOT / "render" / asset_name
    if not np.isfinite(args.clearance_mm) or args.clearance_mm <= 0:
        parser.error("--clearance-mm must be a finite positive millimetre value")
    source_file = bpy.data.filepath
    target = extract_body_target(bpy.data.objects[args.body], bpy.data.objects[args.rig], frame=args.frame)
    target["target_pose"] = args.target_pose
    if args.target_pose == "rest":
        target["positions_mm"] = target["reference_positions_mm"].copy()
        for landmarks in target["landmarks_by_side"].values():
            landmarks["current"] = {key: value.copy() for key,value in landmarks["rest"].items()}
    # Use an independent synthetic X0 and record all dimensions. The body is
    # deliberately not used to manufacture a fake measured/reference shape.
    parameters = ProceduralGarmentParameters(circumference_segments=args.circumference_segments,
                                             leg_segments=args.leg_segments,
                                             heel_segments=32, foot_segments=32, toe_segments=16)
    reference = generate_reference_garment(parameters)
    pairs, reports, results, field_arrays = {}, {}, {}, {}
    for side in ("left", "right"):
        result = fit_reference_to_body(reference, parameters, target, side, clearance_mm=args.clearance_mm,
                                       progress=lambda msg: print(msg, flush=True), diagnostic_path=ROOT/"build-support"/"stocking-fit"/(side+"-fit-diagnostic.npz"),
                                       report_contact_conflicts=args.report_contact_conflicts)
        field = compute_deformation_field(result.pair)
        arrays = {item.name: getattr(field, item.name) for item in fields(field) if isinstance(getattr(field, item.name), np.ndarray)}
        result.report["course_stretch_percentiles"] = np.percentile(field.course_stretch, [0, 5, 50, 95, 100]).tolist()
        result.report["wale_stretch_percentiles"] = np.percentile(field.wale_stretch, [0, 5, 50, 95, 100]).tolist()
        result.report["max_absolute_shear_degrees"] = float(np.rad2deg(np.abs(field.shear_angle_rad)).max())
        pairs[side], reports[side] = result.pair, result.report
        results[side], field_arrays[side] = result, arrays
        print("FIT_SIDE", side, json.dumps(result.report), flush=True)
    # Finish both sides before replacing any previously completed assets.
    # A strict fit failure must not leave an old manifest beside mixed poses.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata.json").unlink(missing_ok=True)
    (args.output / "parameters.json").write_text(json.dumps(parameters.to_dict(), indent=2), encoding="utf8")
    np.savez_compressed(args.output / "collision_body.npz", positions_mm=target["positions_mm"], triangles=target["triangles"])
    np.savez_compressed(args.output / "body_binding_domain.npz", reference_positions_mm=target["reference_positions_mm"],
                        current_positions_mm=target["positions_mm"], triangles=target["triangles"], vertex_side=target["vertex_side"])
    for side, result in results.items():
        folder = args.output / side
        folder.mkdir(parents=True, exist_ok=True)
        save_garment_pair(reference, folder / "garment_reference")
        save_garment_pair(result.pair, folder / "garment_pair")
        np.savez_compressed(folder / "deformation_field.npz", **field_arrays[side])
        np.savez_compressed(folder / "contact_samples.npz", **result.contact_samples)
        (folder / "fit_report.json").write_text(json.dumps(result.report, indent=2, allow_nan=False), encoding="utf8")
    scene, objects = create_fit_preview(target["positions_mm"], target["triangles"], pairs)
    objects["_body_vertex_side"] = target["vertex_side"].copy()
    objects["_contact_samples"] = {side: result.contact_samples for side,result in results.items()}
    outputs = {} if args.no_render else render_fit_previews(scene, objects, args.render_dir)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    blend_path = (args.output / "fitting-validation.blend").resolve()
    if source_file and Path(source_file).resolve() == blend_path:
        raise ValueError("save target must differ from original source")
    preview_type = bpy.context.preferences.filepaths.file_preview_type
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
    finally:
        bpy.context.preferences.filepaths.file_preview_type = preview_type
    if bpy.data.filepath != source_file:
        raise RuntimeError("copy save unexpectedly changed source filepath")
    manifest = dict(schema="stocking.geometric_fit_verification", schema_version=1,
                    source_file=source_file, frame=args.frame, body=args.body, length_unit="mm",
                    fit_status="geometric_fit_with_contact_conflicts" if any(r["fit_status"] != "geometric_fit" or r["other_leg_sample_penetration_count"] for r in reports.values()) else "geometric_fit",
                    target_pose=args.target_pose, other_leg_contacts_passed=all(r["other_leg_sample_penetration_count"] == 0 for r in reports.values()),
                    synthetic_uncalibrated=True, dressing_equilibrium_solved=False,
                    material_calibrated=False, yarn_equilibrium_solved=False,
                    original_source_overwritten=False, reference_version=reference.metadata["reference_version"],
                    physical_vertices_per_sock=len(reference.material_point_ids), triangles_per_sock=len(reference.triangles),
                    side_reports=reports, renders=outputs, blend=str(blend_path))
    (args.output / "metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print("FIT_COMPLETED", json.dumps(manifest, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
