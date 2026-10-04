"""Compute a matched local X0/X1, camera-dependent yarn coverage prototype."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")

from stocking_material.asset_io import load_garment_pair
from stocking_material.authored_yarn import AuthoredYarnProvider
from stocking_material.deformation_field import compute_deformation_field
from stocking_material.parameters import Parameters
from stocking_material.yarn_coverage import orthographic_project, rasterize_capsule_union


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def generate_source(provider, output, name, row_half, loop_half):
    rows = np.arange(-819 - row_half, -819 + row_half + 1)
    logical_loops = np.arange(16 - loop_half, 16 + loop_half + 1)
    loops = np.unique(np.mod(logical_loops, provider.phase.course_repeat_count))
    reference, reference_info = provider.centerline_chunk("reference", rows, loops)
    current, current_info = provider.centerline_chunk("current", rows, loops)
    common = ("edges", "global_sample_id", "material_q_mm", "material_face_index", "material_barycentric")
    for key in common:
        if not np.array_equal(reference[key], current[key]):
            raise ValueError("the two states have different material addresses: " + key)
    if reference_info["unmapped_boundary_samples"] or current_info["unmapped_boundary_samples"]:
        raise ValueError("selected source rectangle loses material geometry")
    arrays = {key: reference[key] for key in common}
    for state, data in (("reference", reference), ("current", current)):
        for key in ("positions_mm", "surface_normals"):
            arrays[state + "_" + key] = data[key]
    path = output / (name + "_geometry.npz")
    np.savez_compressed(path, **arrays)
    record = {"file": path.name, "sha256": sha256(path), "rows": [int(rows[0]), int(rows[-1])],
              "loops": [int(logical_loops[0]), int(logical_loops[-1])], "loop_indices": loops.tolist(),
              "loop_selection": "logical local range wrapped to the frozen periodic course",
              "points": len(reference["positions_mm"]),
              "segments": len(reference["edges"]), "same_material_addresses": True,
              "reference_generation": reference_info, "current_generation": current_info}
    return arrays, record


def difference(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"mean_absolute": float(delta.mean()), "p95_absolute": float(np.quantile(delta, .95)),
            "maximum_absolute": float(delta.max()), "different_pixels": int(np.count_nonzero(delta))}


def add_pixel_footprint_levels(output):
    """Aggregate the same finest subray grid into four-times-larger pixels."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    for case in manifest["cases"]:
        path = output / case["npz_file"]
        with np.load(path, allow_pickle=False) as raw:
            data = {key: raw[key].copy() for key in raw.files}
        counts = data["hit_count_32"]
        height, width = counts.shape
        if height % 4 or width % 4:
            raise ValueError("far-footprint comparison requires image dimensions divisible by four")
        far_counts = counts.reshape(height // 4, 4, width // 4, 4).sum(axis=(1, 3), dtype=np.uint32)
        data["far_hit_count"] = far_counts
        data["far_coverage"] = far_counts.astype(np.float32) / np.float32(32 * 32 * 4 * 4)
        np.savez_compressed(path, **data)
        case["sha256"] = sha256(path)
        case["far_footprint"] = {"resolution": [height // 4, width // 4], "linear_pixel_scale": 4,
            "samples_per_far_pixel": 16384, "effective_subsamples_per_axis": 128,
            "method": "sum fine-grid hit counts over 4x4 pixel blocks; same geometric subrays",
            "native_pixel_mm": [case["fov_mm"][0] / width, case["fov_mm"][1] / height],
            "far_pixel_mm": [case["fov_mm"][0] / (width // 4), case["fov_mm"][1] / (height // 4)],
            "mean_C": float(data["far_coverage"].mean()), "coverage_key": "far_coverage"}
    manifest["pixel_footprint_filtering_generated"] = True
    write_json(output / "manifest.json", manifest)


def select_common_interior(parent, output, border):
    """Export one common interior viewport; retain unsuccessful full-field diagnostics."""
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("interior output already exists; use a new version directory")
    original = json.loads((parent / "manifest.json").read_text(encoding="utf8"))
    if border < 1 or not original["all_cases_generated"]:
        raise ValueError("a completed parent and positive common border are required")
    manifest = copy.deepcopy(original)
    height, width = original["cases"][0]["resolution"]
    bounds = [border, height-border, border, width-border]
    new_height, new_width = height-2*border, width-2*border
    if min(new_height, new_width) <= 0 or new_height % 4 or new_width % 4:
        raise ValueError("interior dimensions must be positive multiples of four")
    output.mkdir(parents=True)
    diagnostics = output / "full-fov-diagnostics"
    diagnostics.mkdir()
    shutil.copy2(parent / "manifest.json", diagnostics / "manifest.json")
    for source in original["source_windows"].values():
        shutil.copy2(parent / source["file"], output / source["file"])
    manifest["parent_full_field_manifest_file"] = "full-fov-diagnostics/manifest.json"
    manifest["parent_full_field_manifest_sha256"] = sha256(parent / "manifest.json")
    manifest["crop_bounds_pixels"] = bounds
    manifest["viewport_selection"] = {
        "method": "same symmetric interior crop in every state and angle",
        "border_pixels_each_side": border,
        "parent_resolution": [height, width],
        "parent_full_frame_window_guard_passed": original["expanded_window_stable_all_cases"],
        "physical_pixel_pitch_and_subrays_unchanged": True,
        "selection_scope": "bounded local cloth source only; no global source completeness proof",
        "reason": "preserve full-field edge differences while examining a common interior stable between two finite source windows"}
    manifest["parent_full_field_elapsed_seconds"] = manifest.pop("elapsed_seconds", None)
    manifest["prototype_scope"] += "; common central interior viewport with retained full-field failures"
    roi = (slice(border, height-border), slice(border, width-border))
    for case in manifest["cases"]:
        if case["resolution"] != [height, width]:
            raise ValueError("all cases must share the parent viewport")
        parent_path = parent / case["npz_file"]
        shutil.copy2(parent_path, diagnostics / case["npz_file"])
        with np.load(parent_path, allow_pickle=False) as raw:
            data = {key: raw[key][roi].copy() for key in raw.files
                    if key.startswith(("coverage_", "hit_count_", "expanded_"))}
        path = output / case["npz_file"]
        np.savez_compressed(path, **data)
        case["parent_full_field_npz_file"] = "full-fov-diagnostics/" + case["npz_file"]
        case["npz_path"], case["sha256"] = str(path), sha256(path)
        old_fov = case["fov_mm"]
        new_fov = [old_fov[0]*new_width/width, old_fov[1]*new_height/height]
        case["resolution"] = [new_height, new_width]
        case["fov_mm"] = new_fov
        case["camera"]["image_size"] = [new_height, new_width]
        case["camera"]["fov_mm"] = new_fov
        case["mean_C"] = float(data["coverage_32"].mean())
        case["convergence"] = {
            "8_to_16": difference(data["coverage_8"], data["coverage_16"]),
            "16_to_32": difference(data["coverage_16"], data["coverage_32"])}
        case["expanded_window_difference"] = difference(data["coverage_32"], data["expanded_coverage_32"])
        case["expanded_window_stable"] = case["expanded_window_difference"]["different_pixels"] == 0
        case.pop("far_footprint", None)
        for level, metadata in list(case["level_metadata"].items()) + [("expanded", case["expanded_level_metadata"])]:
            sample_level = 32 if level == "expanded" else int(level)
            count_key = "expanded_hit_count_32" if level == "expanded" else "hit_count_"+level
            c_key = "expanded_coverage_32" if level == "expanded" else "coverage_"+level
            for statistic in ("visited_segment_count", "outside_raster_segment_count",
                              "clipped_expanded_aabb_segment_count", "point_projection_segment_count",
                              "tested_segment_sample_pairs", "retained_subpixel_mask_bytes"):
                metadata.pop(statistic, None)
            metadata.update(image_size_height_width=[new_height, new_width], fov_width_height_mm=new_fov,
                            subpixel_hit_count=int(data[count_key].sum(dtype=np.uint64)),
                            mean_coverage=float(data[c_key].mean()),
                            viewport_export="exact parent interior subrays; full raster work statistics retained in parent manifest")
    manifest["expanded_window_stable_all_cases"] = all(case["expanded_window_stable"] for case in manifest["cases"])
    write_json(output / "manifest.json", manifest)
    add_pixel_footprint_levels(output)
    print("YARN_COVERAGE_INTERIOR_COMPLETE", json.dumps({"output": str(output), "bounds": bounds,
          "stable_all_cases": manifest["expanded_window_stable_all_cases"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, default=ROOT / "stocking/assets/authored-fit-v1/left/garment_pair.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "stocking/assets/yarn-coverage-v4")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--fov-mm", type=float, default=1.)
    parser.add_argument("--main-half", type=int, default=19)
    parser.add_argument("--expanded-half", type=int, default=23)
    parser.add_argument("--filter-existing", action="store_true", help="add larger-pixel comparisons to a completed local prototype")
    parser.add_argument("--crop-existing", type=Path, help="select a common interior and retain full-field diagnostics")
    parser.add_argument("--border", type=int, default=8)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:])
    if args.crop_existing:
        select_common_interior(args.crop_existing, args.output, args.border)
        return
    if args.filter_existing:
        add_pixel_footprint_levels(args.output.resolve())
        print("YARN_COVERAGE_FOOTPRINT_COMPLETE", str(args.output.resolve()), flush=True)
        return
    if args.size < 1 or not 0 < args.fov_mm or not 0 <= args.main_half < args.expanded_half <= 30:
        raise ValueError("invalid image, FOV, or nested material window bounds")
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("output already exists; use a new version directory")
    output.mkdir(parents=True, exist_ok=True)
    pair = load_garment_pair(args.pair)
    frozen_hash = sha256(args.pair)
    parameters = Parameters(samples_per_loop=192)
    provider = AuthoredYarnProvider(pair, parameters, representation="capsule_union")
    field = compute_deformation_field(pair)
    core, core_info = generate_source(provider, output, "main", args.main_half, args.main_half)
    expanded, expanded_info = generate_source(provider, output, "expanded", args.expanded_half, args.expanded_half)
    central, _ = provider.centerline_chunk("reference", range(-820, -817), range(15, 18))
    central_current, _ = provider.centerline_chunk("current", range(-820, -817), range(15, 18))
    face = int(central["material_face_index"][len(central["material_face_index"]) // 2])
    radius = parameters.R * parameters.scale_mm
    manifest = {"schema": "stocking.camera_yarn_coverage.v1", "prototype_scope": "matched finite material window, opaque geometric yarn, orthographic camera, camera-aligned rear skin plane",
                "pair_path": str(args.pair.resolve()), "pair_sha256": frozen_hash,
                "sock": pair.metadata["side"], "parameters": parameters.to_dict(),
                "radius_mm": radius, "source_windows": {"main": core_info, "expanded": expanded_info},
                "central_frame_face": face, "cases": [], "row_zero": "image top (+camera up)",
                "C_definition": "fraction of equal-weight camera subrays hitting capsule union before the rear skin plane",
                "C_uses_geometric_hit_not_PNT_valid": True, "no_second_F": True,
                "mechanics_solved": False, "full_stocking_coverage": False,
                "real_skin_geometry_used": False, "global_source_completeness_certified": False,
                "fiber_transmission_modeled": False}
    started = time.perf_counter()
    for state, central_data, frame in (("reference", central, field.reference_frames[face]),
                                        ("current", central_current, field.current_frames[face])):
        center = central_data["positions_mm"].mean(axis=0)
        points, edges = core[state + "_positions_mm"], core["edges"]
        larger_points, larger_edges = expanded[state + "_positions_mm"], expanded["edges"]
        for angle in (0, 45, 75):
            theta = np.deg2rad(angle)
            right = np.cos(theta) * frame[:, 0] - np.sin(theta) * frame[:, 2]
            up = frame[:, 1]
            toward = np.sin(theta) * frame[:, 0] + np.cos(theta) * frame[:, 2]
            projected = orthographic_project(larger_points, center, right, up, toward)
            camera = {"center_mm": center.tolist(), "right": right.tolist(), "up": up.tolist(),
                      "toward_camera": toward.tolist(), "camera_depth_mm": float(projected[:, 2].max()+radius+.1),
                      "skin_depth_mm": float(projected[:, 2].min()-radius-.1),
                      "image_size": [args.size, args.size], "fov_mm": [args.fov_mm, args.fov_mm]}
            kw = dict(center_mm=center, right=right, up=up, toward_camera=toward,
                      image_size=tuple(camera["image_size"]), fov_mm=tuple(camera["fov_mm"]),
                      camera_depth_mm=camera["camera_depth_mm"], skin_depth_mm=camera["skin_depth_mm"])
            saved, level_info = {}, {}
            name = state + "-" + str(angle)
            for level in (8, 16, 32):
                tick = time.perf_counter()
                result = rasterize_capsule_union(points[edges[:, 0]], points[edges[:, 1]], radius,
                                                  supersample=level, **kw)
                saved["coverage_" + str(level)] = result["coverage"]
                saved["hit_count_" + str(level)] = result["hit_count"]
                level_info[str(level)] = result["metadata"]
                print("YARN_COVERAGE_LEVEL", json.dumps({"case": name, "S": level, "mean_C": float(result["coverage"].mean()),
                      "seconds": time.perf_counter()-tick}), flush=True)
            larger = rasterize_capsule_union(larger_points[larger_edges[:, 0]], larger_points[larger_edges[:, 1]], radius,
                                             supersample=32, **kw)
            saved["expanded_coverage_32"] = larger["coverage"]
            saved["expanded_hit_count_32"] = larger["hit_count"]
            path = output / (name + ".npz")
            np.savez_compressed(path, **saved)
            guard = difference(saved["coverage_32"], larger["coverage"])
            convergence = {"8_to_16": difference(saved["coverage_8"], saved["coverage_16"]),
                           "16_to_32": difference(saved["coverage_16"], saved["coverage_32"])}
            record = {"name": name, "state": state, "angle_deg": angle, "npz_path": str(path), "npz_file": path.name,
                      "sha256": sha256(path), "coverage_key": "coverage_32", "samples_per_axis": 32,
                      "resolution": [args.size, args.size], "fov_mm": [args.fov_mm, args.fov_mm], "camera": camera,
                      "mean_C": float(saved["coverage_32"].mean()), "convergence": convergence,
                      "expanded_window_difference": guard, "expanded_window_stable": guard["different_pixels"] == 0,
                      "level_metadata": level_info, "expanded_level_metadata": larger["metadata"]}
            manifest["cases"].append(record)
            write_json(output / "manifest.inprogress.json", manifest)
            print("YARN_COVERAGE_CASE", json.dumps({"name": name, "guard": guard, "convergence": convergence}), flush=True)
    if sha256(args.pair) != frozen_hash:
        raise ValueError("frozen material pair changed during coverage generation")
    manifest["all_cases_generated"] = len(manifest["cases"]) == 6
    manifest["expanded_window_stable_all_cases"] = all(case["expanded_window_stable"] for case in manifest["cases"])
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "manifest.json", manifest)
    (output / "manifest.inprogress.json").unlink()
    add_pixel_footprint_levels(output)
    print("YARN_COVERAGE_COMPLETE", json.dumps({"output": str(output), "cases": len(manifest["cases"]),
          "expanded_window_stable": manifest["expanded_window_stable_all_cases"], "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
