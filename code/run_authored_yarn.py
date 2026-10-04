"""Build geometric plain-knit tubes or stream full material centerlines.

Examples (from project root):
  python stocking/code/run_authored_yarn.py --side left --mode estimate
  python stocking/code/run_authored_yarn.py --side left --mode patch --face 100
  python stocking/code/run_authored_yarn.py --side left --mode stream-centerlines --max-new-chunks 4
  python stocking/code/run_authored_yarn.py --side left --mode window-patch --representation capsule_union --row-min -3429 --row-max -3427 --loop-min 278 --loop-max 280

Streaming has no hidden resolution reduction. ``--max-new-chunks`` explicitly
leaves a partial manifest; omit it to continue every remaining chunk.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")

from stocking_material.asset_io import load_garment_pair  # noqa: E402
from stocking_material.authored_yarn import AuthoredYarnProvider  # noqa: E402
from stocking_material.parameters import Parameters  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def source_identity(pair_prefix, yarn_parameters, provider):
    prefix = Path(pair_prefix)
    envelope = json.loads(prefix.with_suffix(".json").read_text(encoding="utf8"))
    return {
        "pair_npz_sha256": envelope["array_sha256"],
        "input_snapshot_prefix": str(prefix.resolve()),
        "pair_reference_version": provider.pair.metadata["reference_version"],
        "pair_material_version": provider.pair.metadata["material_version"],
        "yarn_parameters": yarn_parameters.to_dict(),
        "global_phase": provider.phase.metadata,
        "geometry_contract": "C=bary @ X + h * interpolated_surface_normal; no second F",
    }


def freeze_pair_input(source_prefix, output):
    """Copy a changing candidate to a content-addressed, validated input."""
    source = Path(source_prefix)
    source_npz = source.with_suffix(".npz")
    source_json = source.with_suffix(".json")
    snapshots = Path(output) / "input_snapshot"
    for attempt in range(3):
        envelope = json.loads(source_json.read_text(encoding="utf8"))
        expected_sha = envelope["array_sha256"]
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("source pair envelope has no valid array hash")
        target = snapshots / expected_sha / "garment_pair"
        target_npz = target.with_suffix(".npz")
        target_json = target.with_suffix(".json")
        if target_npz.exists() and target_json.exists():
            try:
                load_garment_pair(target)
                if sha256(target_npz) == expected_sha:
                    return target
            except ValueError:
                pass
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_npz = target.with_name(target.name + ".npz.tmp")
        temporary_json = target.with_name(target.name + ".json.tmp")
        shutil.copyfile(source_npz, temporary_npz)
        shutil.copyfile(source_json, temporary_json)
        copied = json.loads(temporary_json.read_text(encoding="utf8"))
        if copied.get("array_sha256") != expected_sha or sha256(temporary_npz) != expected_sha:
            temporary_npz.unlink(missing_ok=True)
            temporary_json.unlink(missing_ok=True)
            continue
        temporary_npz.replace(target_npz)
        temporary_json.replace(target_json)
        load_garment_pair(target)
        return target
    raise ValueError("source pair changed during snapshot copy three times; retry after candidate settles")


def stream_centerlines(provider, output, configuration, identity, *, rows_per_chunk,
                       loops_per_chunk, max_new_chunks, row_window=None,
                       loop_window=None):
    estimate = provider.estimate()
    full_first_row = estimate["first_candidate_wale_row"]
    full_last_row = estimate["last_candidate_wale_row"]
    course_count = provider.phase.course_repeat_count
    first_row, last_row = row_window or (full_first_row, full_last_row)
    first_loop, last_loop = loop_window or (0, course_count - 1)
    if (first_row < full_first_row or last_row > full_last_row or first_row > last_row
            or first_loop < 0 or last_loop >= course_count or first_loop > last_loop):
        raise ValueError("requested row/loop window exceeds the global material chart")
    row_ranges = [(a, min(last_row + 1, a + rows_per_chunk))
                  for a in range(first_row, last_row + 1, rows_per_chunk)]
    loop_ranges = [(a, min(last_loop + 1, a + loops_per_chunk))
                   for a in range(first_loop, last_loop + 1, loops_per_chunk)]
    total = len(row_ranges) * len(loop_ranges)
    output.mkdir(parents=True, exist_ok=True)
    expected = {"schema": "stocking.authored_yarn_stream.v1", "identity": identity,
                "configuration": configuration, "rows_per_chunk": rows_per_chunk,
                "loops_per_chunk": loops_per_chunk, "total_chunks": total,
                "full_stream_requested": max_new_chunks is None and row_window is None and loop_window is None,
                "stream_content": "stable-ID centerline nodes and material edges; tube mesh built on demand",
                "coverage_semantics": "chunk count and material sample mapping only; not feature coverage or physical validity"}
    if row_window is not None or loop_window is not None:
        expected["explicit_window"] = {"row_min": first_row, "row_max": last_row,
                                       "loop_min": first_loop, "loop_max": last_loop}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf8"))
        for key in ("schema", "identity", "configuration", "rows_per_chunk",
                    "loops_per_chunk", "total_chunks", "explicit_window"):
            if previous.get(key) != expected.get(key):
                raise ValueError(f"existing centerline stream has different {key}; select a new output")
    else:
        write_json(manifest_path, expected | {"completed_chunks": 0, "stream_complete": False})
    seam_npz = output / "toe-seam.npz"
    seam_json = output / "toe-seam.json"
    if seam_npz.exists() or seam_json.exists():
        if not seam_npz.exists() or not seam_json.exists():
            raise ValueError("incomplete toe seam stream; inspect before resuming")
        seam_report = json.loads(seam_json.read_text(encoding="utf8"))
        if (seam_report.get("identity") != identity
                or seam_report.get("configuration") != configuration
                or seam_report.get("npz_sha256") != sha256(seam_npz)):
            raise ValueError("stale or damaged toe seam stream; select a new output")
    else:
        if configuration == "both":
            seam0, seam0_coverage = provider.seam_centerline("reference")
            seam1, seam1_coverage = provider.seam_centerline("current")
            seam_keys = ("material_edge_point_ids", "material_edge_fraction",
                         "global_seam_sample_id", "edges")
            if any(not np.array_equal(seam0[key], seam1[key]) for key in seam_keys):
                raise ValueError("X0 and X1 toe seam lost material correspondence")
            seam_arrays = {key: seam0[key] for key in seam_keys}
            seam_arrays.update(reference_positions_mm=seam0["positions_mm"],
                               current_positions_mm=seam1["positions_mm"],
                               reference_surface_normals=seam0["surface_normals"],
                               current_surface_normals=seam1["surface_normals"])
            seam_coverage = dict(seam0_coverage, configuration="both",
                                 same_material_edge_ids_and_fractions=True,
                                 current_sample_count=seam1_coverage["sample_count"])
        else:
            seam_arrays, seam_coverage = provider.seam_centerline(configuration)
        write_npz(seam_npz, seam_arrays)
        seam_report = {"schema": "stocking.authored_yarn_seam_file.v1",
                       "identity": identity, "configuration": configuration,
                       "npz_sha256": sha256(seam_npz), "coverage": seam_coverage,
                       "closure_representation": "synthetic distributed seam centerline; separate from course window",
                       "toe_seam_tube_quality_checked": False}
        write_json(seam_json, seam_report)
    created = completed = mapped = selected = 0
    for row_a, row_b in row_ranges:
        for loop_a, loop_b in loop_ranges:
            name = f"r{row_a:+06d}_{row_b-1:+06d}_c{loop_a:04d}_{loop_b-1:04d}"
            npz = output / (name + ".npz")
            sidecar = output / (name + ".json")
            if npz.exists() or sidecar.exists():
                if not npz.exists() or not sidecar.exists():
                    raise ValueError(f"incomplete old chunk {name}; inspect before resuming")
                saved = json.loads(sidecar.read_text(encoding="utf8"))
                if saved.get("identity") != identity or saved.get("configuration") != configuration or saved.get("npz_sha256") != sha256(npz):
                    raise ValueError(f"stale or damaged old chunk {name}; select a new output")
                completed += 1
                mapped += int(saved["coverage"]["mapped_samples"])
                selected += int(saved["coverage"]["selected_samples"])
                continue
            if max_new_chunks is not None and created >= max_new_chunks:
                break
            if configuration == "both":
                reference, reference_coverage = provider.centerline_chunk(
                    "reference", range(row_a, row_b), range(loop_a, loop_b))
                current, current_coverage = provider.centerline_chunk(
                    "current", range(row_a, row_b), range(loop_a, loop_b))
                shared_keys = ("material_face_index", "material_barycentric",
                               "material_q_mm", "global_sample_id", "edges")
                if any(not np.array_equal(reference[key], current[key]) for key in shared_keys):
                    raise ValueError("X0 and X1 streamed centerlines lost identical material correspondence")
                arrays = {key: reference[key] for key in shared_keys}
                arrays.update(reference_positions_mm=reference["positions_mm"],
                              current_positions_mm=current["positions_mm"],
                              reference_surface_normals=reference["surface_normals"],
                              current_surface_normals=current["surface_normals"])
                coverage = dict(reference_coverage,
                                configuration="both",
                                same_material_sample_ids_and_barycentrics=True,
                                current_mapped_samples=current_coverage["mapped_samples"],
                                current_unmapped_boundary_samples=current_coverage["unmapped_boundary_samples"])
            else:
                arrays, coverage = provider.centerline_chunk(
                    configuration, range(row_a, row_b), range(loop_a, loop_b))
            write_npz(npz, arrays)
            write_json(sidecar, {"schema": "stocking.authored_yarn_centerline_file.v1",
                                 "identity": identity, "configuration": configuration,
                                 "npz_sha256": sha256(npz), "coverage": coverage})
            created += 1
            completed += 1
            mapped += int(coverage["mapped_samples"])
            selected += int(coverage["selected_samples"])
        if max_new_chunks is not None and created >= max_new_chunks:
            break
    result = expected | {"completed_chunks": completed, "created_this_run": created,
                         "stream_complete": completed == total,
                         "full_sock_stream_complete": completed == total and row_window is None and loop_window is None,
                         "mapped_samples_in_completed_chunks": mapped,
                         "selected_samples_in_completed_chunks": selected,
                         "mapped_fraction_in_completed_chunks": mapped / selected if selected else None,
                         "estimated_full_stream": estimate,
                         "production_feature_coverage_established": False,
                         "tube_geometry_fully_exported": False,
                         "toe_seam_centerline_exported": True,
                         "toe_seam_centerline_samples": seam_report["coverage"]["sample_count"],
                         "toe_seam_tube_geometry_separately_validated": False}
    write_json(manifest_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Geometric authored plain-knit yarn on a material garment pair")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--pair", type=Path, help="GarmentMeshPair prefix; default authored-fit-v1/<side>/garment_pair")
    parser.add_argument("--output", type=Path, help="output directory; default authored-yarn-v1/<side>")
    parser.add_argument("--mode", choices=("estimate", "patch", "window-patch", "stream-centerlines"), default="estimate")
    parser.add_argument("--configuration", choices=("reference", "current", "both"), default="both")
    parser.add_argument("--face", type=int, action="append", help="material face index for patch mode; repeatable")
    parser.add_argument("--halo-mm", type=float, default=0.3, help="world ray/support reach beyond yarn height/radius")
    parser.add_argument("--samples-per-loop", type=int, default=192)
    parser.add_argument("--tube-sides", type=int, default=24)
    parser.add_argument("--max-tube-vertices", type=int, default=10_000_000)
    parser.add_argument("--representation", choices=("strict_tube", "capsule_union"), default="strict_tube")
    parser.add_argument("--capsule-cap-rings", type=int, default=3)
    parser.add_argument("--rows-per-chunk", type=int, default=4)
    parser.add_argument("--loops-per-chunk", type=int, default=32)
    parser.add_argument("--max-new-chunks", type=int, help="explicit partial streaming limit; omit for all remaining")
    parser.add_argument("--row-min", type=int, help="explicit inclusive row window for local stream")
    parser.add_argument("--row-max", type=int, help="explicit inclusive row window for local stream")
    parser.add_argument("--loop-min", type=int, help="explicit inclusive loop window for local stream")
    parser.add_argument("--loop-max", type=int, help="explicit inclusive loop window for local stream")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    if args.mode == "patch" and not args.face:
        parser.error("--mode patch requires at least one --face")
    if args.mode == "patch" and args.configuration == "both":
        parser.error("--mode patch requires --configuration reference or current")
    if args.rows_per_chunk < 1 or args.loops_per_chunk < 1 or (args.max_new_chunks is not None and args.max_new_chunks < 1):
        parser.error("chunk sizes and --max-new-chunks must be positive")
    if (args.row_min is None) != (args.row_max is None) or (args.loop_min is None) != (args.loop_max is None):
        parser.error("each row/loop window requires both its minimum and maximum")
    if args.mode not in ("stream-centerlines", "window-patch") and any(value is not None for value in
                                                   (args.row_min, args.row_max, args.loop_min, args.loop_max)):
        parser.error("row/loop windows apply only to stream-centerlines or window-patch")
    if args.mode == "window-patch" and any(value is None for value in
                                           (args.row_min, args.row_max, args.loop_min, args.loop_max)):
        parser.error("window-patch requires explicit --row-min/max and --loop-min/max")
    pair_prefix = args.pair or ROOT / "stocking" / "assets" / "authored-fit-v1" / args.side / "garment_pair"
    output = args.output or ROOT / "stocking" / "assets" / "authored-yarn-v1" / args.side
    if args.mode != "estimate":
        pair_prefix = freeze_pair_input(pair_prefix, output)
    pair = load_garment_pair(pair_prefix)
    parameters = replace(Parameters(), samples_per_loop=args.samples_per_loop,
                         tube_sides=args.tube_sides)
    provider = AuthoredYarnProvider(pair, parameters,
                                    max_tube_vertices=args.max_tube_vertices,
                                    representation=args.representation,
                                    capsule_cap_rings=args.capsule_cap_rings)
    identity = source_identity(pair_prefix, parameters, provider)
    if args.representation == "capsule_union":
        identity["representation"] = args.representation
        identity["capsule_cap_rings"] = args.capsule_cap_rings
    estimate = provider.estimate()
    if args.mode == "estimate":
        print(json.dumps({"identity": identity, "estimate": estimate}, ensure_ascii=True, indent=2))
        return
    if args.mode == "patch":
        result = provider(args.configuration, args.face, args.halo_mm)
        geometry = result["geometry"]
        output.mkdir(parents=True, exist_ok=True)
        name = f"{args.configuration}-faces-" + "-".join(str(face) for face in sorted(set(args.face)))
        target = output / (name + ".npz")
        write_npz(target, {"vertices_mm": geometry.vertices,
                           "triangles": geometry.triangles,
                           "normals": geometry.normals,
                           "tangents": geometry.tangents,
                           **({"triangle_tangent_valid": result["triangle_tangent_valid"]}
                              if "triangle_tangent_valid" in result else {})})
        report = {"schema": "stocking.authored_yarn_geometric_patch.v1",
                  "identity": identity, "npz_sha256": sha256(target),
                  "face_indices": sorted(set(args.face)),
                  "coverage": result["coverage"],
                  "full_sock_feature_coverage": False,
                  "synthetic_toe_layout": True,
                  "material_calibrated": False,
                  "dressing_equilibrium_solved": False,
                  "yarn_equilibrium_solved": False}
        write_json(output / (name + ".json"), report)
        print(json.dumps({"output": str(target), "coverage": report["coverage"]}, ensure_ascii=True))
        return
    if args.mode == "window-patch":
        states = ("reference", "current") if args.configuration == "both" else (args.configuration,)
        shared = None
        for configuration in states:
            result = provider.window_geometry(configuration,
                                              range(args.row_min, args.row_max + 1),
                                              range(args.loop_min, args.loop_max + 1))
            yarn = result["centerline"]
            shared_keys = ("material_q_mm", "material_face_index", "material_barycentric",
                           "global_sample_id", "edges")
            if shared is None:
                shared = {key: yarn[key] for key in shared_keys}
            elif any(not np.array_equal(shared[key], yarn[key]) for key in shared_keys):
                raise ValueError("window X0/X1 material correspondence differs")
            geometry = result["geometry"]
            name = (f"{configuration}-{args.representation}-rows-{args.row_min}-{args.row_max}"
                    f"-loops-{args.loop_min}-{args.loop_max}")
            target = output / (name + ".npz")
            arrays = {"vertices_mm": geometry.vertices, "triangles": geometry.triangles,
                      "normals": geometry.normals, "tangents": geometry.tangents,
                      "centerline_positions_mm": yarn["positions_mm"],
                      "centerline_surface_normals": yarn["surface_normals"],
                      **{("centerline_" + key): yarn[key] for key in shared_keys}}
            if "triangle_tangent_valid" in result:
                arrays["triangle_tangent_valid"] = result["triangle_tangent_valid"]
            write_npz(target, arrays)
            report = {"schema": "stocking.authored_yarn_window_patch.v1",
                      "identity": identity, "npz_sha256": sha256(target),
                      "representative_face_indices_only": sorted(set(args.face or [])),
                      "coverage": result["coverage"],
                      "same_X0_X1_material_correspondence_checked": args.configuration == "both",
                      "full_sock_feature_coverage": False,
                      "toe_seam_included": False}
            write_json(output / (name + ".json"), report)
            print(json.dumps({"output": str(target), "coverage": report["coverage"]}, ensure_ascii=True))
        return
    report = stream_centerlines(provider, output / "centerline-stream" / args.configuration,
                                args.configuration, identity, rows_per_chunk=args.rows_per_chunk,
                                loops_per_chunk=args.loops_per_chunk,
                                max_new_chunks=args.max_new_chunks,
                                row_window=(args.row_min, args.row_max) if args.row_min is not None else None,
                                loop_window=(args.loop_min, args.loop_max) if args.loop_min is not None else None)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
