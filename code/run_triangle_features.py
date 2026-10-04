"""Estimate or resume independent dense triangle P/N/T/valid generation.

Use Blender for --bake; --estimate-only works with a NumPy Python runtime.
Example: blender -b --python run_triangle_features.py -- --pair PAIR
    --output OUTPUT --face-start 0 --face-count 1 --max-new-chunks 1 --bake
"""
from pathlib import Path
import argparse
import json
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")

from stocking_material.asset_io import load_garment_pair
from stocking_material.parameters import Parameters
from stocking_material.triangle_features import (prepare_triangle_sampling, bake_triangle_features,
    plan_triangle_export_budget, _write_json)


def main():
    parser = argparse.ArgumentParser(description="Independent triangle features from actual geometric yarn tubes; no equilibrium")
    parser.add_argument("--pair", type=Path, help="versioned garment_pair NPZ/JSON or basename")
    parser.add_argument("--budget-pair", type=Path, action="append", help="repeat for the socks included in one overall byte budget")
    parser.add_argument("--budget-name", action="append", help="repeat optional names in the same order as budget-pair")
    parser.add_argument("--budget-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--reserve-bytes", type=int, default=100_000_000)
    parser.add_argument("--budget-plan", type=Path, help="write a formal density/byte plan; no features are generated")
    parser.add_argument("--use-budget-plan", type=Path, help="read frozen spacing/chunking/per-sock byte limit for --pair")
    parser.add_argument("--budget-sock", help="sock name in --use-budget-plan")
    parser.add_argument("--export-byte-limit", type=int, help="hard per-output-directory ceiling, requires compact coverage")
    parser.add_argument("--compact-coverage", action="store_true", help="bounded per-chunk coverage summaries, no per-request JSON")
    parser.add_argument("--output", type=Path, help="new/resumable formal feature asset directory")
    parser.add_argument("--parameters", type=Path, help="plain-knit geometry parameter JSON")
    parser.add_argument("--samples-per-loop", type=int)
    parser.add_argument("--tube-sides", type=int)
    parser.add_argument("--course-repeat-count", type=int)
    parser.add_argument("--spacing-mm", type=float)
    parser.add_argument("--samples-per-radius", type=float, default=2.)
    parser.add_argument("--ray-offset-mm", type=float)
    parser.add_argument("--halo-mm", type=float)
    parser.add_argument("--max-samples-per-chunk", type=int, default=65536)
    parser.add_argument("--max-faces-per-chunk", type=int, default=1)
    parser.add_argument("--geometry-world-tile-mm", type=float, default=.5)
    parser.add_argument("--geometry-material-tile-mm", type=float, default=1.)
    parser.add_argument("--max-samples-per-geometry-request", type=int, default=2048)
    parser.add_argument("--representation", choices=("strict_tube", "capsule_union", "analytic_capsule_union"), default="strict_tube")
    parser.add_argument("--capsule-cap-rings", type=int, default=3)
    parser.add_argument("--max-tube-vertices", type=int, default=3_000_000)
    parser.add_argument("--face-start", type=int, default=0)
    parser.add_argument("--face-count", type=int, help="omit for remaining/all faces")
    parser.add_argument("--max-new-chunks", type=int, help="bounded run, then resume with identical arguments")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--bake", action="store_true", help="explicitly generate data; default is estimate only")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    values = json.loads(args.parameters.read_text(encoding="utf-8-sig")) if args.parameters else {}
    if args.samples_per_loop is not None:
        values["samples_per_loop"] = args.samples_per_loop
    if args.tube_sides is not None:
        values["tube_sides"] = args.tube_sides
    parameters = Parameters.from_dict(values)
    radius = parameters.R * parameters.scale_mm
    if args.budget_pair:
        if args.pair or args.bake:
            parser.error("budget planning uses --budget-pair without --pair or --bake")
        names = args.budget_name or ["sock-%d" % index for index in range(len(args.budget_pair))]
        if len(names) != len(args.budget_pair) or len(set(names)) != len(names):
            parser.error("budget-name must provide unique names for every budget-pair")
        pairs = {name: load_garment_pair(path) for name, path in zip(names, args.budget_pair)}
        # Bundle metadata remains bounded only with many faces per storage chunk.
        max_faces = 256 if args.max_faces_per_chunk == 1 else args.max_faces_per_chunk
        plan = plan_triangle_export_budget(pairs, budget_bytes=args.budget_bytes,
            reserve_bytes=args.reserve_bytes, yarn_radius_mm=radius, preferred_spacing_mm=args.spacing_mm,
            max_samples_per_chunk=args.max_samples_per_chunk, max_faces_per_chunk=max_faces)
        plan["inputs"] = {name: {"pair": str(path.resolve()),
            "array_sha256": json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["array_sha256"]}
            for name, path in zip(names, args.budget_pair)}
        plan["geometry"] = {"representation": args.representation, "parameters": parameters.to_dict(),
                            "course_repeat_count": args.course_repeat_count}
        if args.budget_plan:
            args.budget_plan.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.budget_plan, plan)
        print("TRIANGLE_FEATURE_BUDGET", json.dumps(plan, ensure_ascii=True))
        return
    if args.pair is None:
        parser.error("provide --pair for estimate/bake or repeat --budget-pair for a global plan")
    pair = load_garment_pair(args.pair)
    if args.use_budget_plan:
        plan = json.loads(args.use_budget_plan.read_text(encoding="utf-8"))
        if args.budget_sock not in plan["socks"]:
            parser.error("--budget-sock must name a sock in the frozen plan")
        from stocking_material.triangle_features import pair_fingerprint
        sock = plan["socks"][args.budget_sock]
        if sock["pair_fingerprint"] != pair_fingerprint(pair):
            parser.error("budget plan pair identity differs from --pair")
        args.spacing_mm = plan["spacing_mm"]
        args.max_samples_per_chunk = plan["max_samples_per_chunk"]
        args.max_faces_per_chunk = plan["max_faces_per_chunk"]
        args.compact_coverage = True
        args.export_byte_limit = sock["export_byte_limit"]
        if args.representation != plan["geometry"]["representation"] or parameters.to_dict() != plan["geometry"]["parameters"]:
            parser.error("geometry representation/parameters differ from the frozen budget plan")
        if args.course_repeat_count != plan["geometry"]["course_repeat_count"]:
            parser.error("course repeat count differs from the frozen budget plan")
    sampling = prepare_triangle_sampling(pair, yarn_radius_mm=radius,
        spacing_mm=args.spacing_mm, samples_per_radius=args.samples_per_radius)
    if args.face_start < 0 or args.face_start >= len(pair.triangles):
        parser.error("face-start outside the material mesh")
    count = len(pair.triangles) - args.face_start if args.face_count is None else args.face_count
    if count < 1 or args.face_start + count > len(pair.triangles):
        parser.error("face-count outside the material mesh")
    import numpy as np
    faces = np.arange(args.face_start, args.face_start + count, dtype=np.int64)
    from stocking_material.authored_yarn import AuthoredYarnProvider
    provider = AuthoredYarnProvider(pair, parameters,
        course_repeat_count=args.course_repeat_count, max_tube_vertices=args.max_tube_vertices,
        representation="capsule_union" if args.representation == "analytic_capsule_union" else args.representation,
        capsule_cap_rings=args.capsule_cap_rings)
    if args.bake and args.representation == "analytic_capsule_union" and not callable(getattr(provider, "sample_capsule_rays", None)):
        parser.error("analytic capsule callback is unavailable; no fallback to huge triangle geometry")
    estimate = {"selected_scope": sampling.estimate(faces), "all_faces_scope": sampling.estimate(),
                "global_yarn_estimate": provider.estimate(), "features_generated_by_estimate": False}
    print("TRIANGLE_FEATURE_ESTIMATE", json.dumps(estimate, ensure_ascii=False))
    if args.estimate_only or not args.bake:
        return
    if args.output is None:
        parser.error("--bake requires an explicit --output")
    ray = args.ray_offset_mm if args.ray_offset_mm is not None else abs(parameters.d * parameters.scale_mm) + 2 * radius
    halo = args.halo_mm if args.halo_mm is not None else ray + radius
    manifest = bake_triangle_features(pair, args.output, provider,
        yarn_radius_mm=radius, ray_offset_mm=ray, halo_mm=halo,
        geometry_identity={"provider": "AuthoredYarnProvider-v1", "parameters": parameters.to_dict(),
                           "representation": args.representation, "capsule_cap_rings": args.capsule_cap_rings,
                           "course_repeat_count": provider.phase.course_repeat_count,
                           "phase_metadata": provider.phase.metadata},
        spacing_mm=args.spacing_mm, samples_per_radius=args.samples_per_radius,
        max_samples_per_chunk=args.max_samples_per_chunk, max_faces_per_chunk=args.max_faces_per_chunk,
        geometry_world_tile_mm=args.geometry_world_tile_mm, geometry_material_tile_mm=args.geometry_material_tile_mm,
        max_samples_per_geometry_request=args.max_samples_per_geometry_request,
        compact_coverage=args.compact_coverage, export_byte_limit=args.export_byte_limit,
        face_indices=faces, max_new_chunks=args.max_new_chunks,
        progress=lambda state: print("TRIANGLE_FEATURE_PROGRESS", json.dumps(state), flush=True))
    print("TRIANGLE_FEATURE_RESULT", json.dumps({key: manifest[key] for key in (
        "completed_samples", "completed_faces", "requested_scope_complete",
        "requested_scope_full", "full_surface_features_generated")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
