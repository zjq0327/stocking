"""Supervise resumable Blender shards and strictly merge a frozen feature plan.

Run with a NumPy Python runtime.  Default prints a frozen job plan only; --run
launches at most eight hidden single-thread children and merges verified shards.
--merge-only performs all source/content/budget checks without launching Blender.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support" / "stocking-material" / "pycache")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from stocking_material.asset_io import load_garment_pair
from stocking_material.triangle_features import prepare_triangle_sampling, _sha256, _write_json
from stocking_material.parallel_triangle_features import (inspect_feature_directory,
    plan_shards, merge_feature_shards, supervise_shard_commands, THREAD_ENVIRONMENT,
    record_closure_provenance)


def print_progress(state):
    print("PARALLEL_TRIANGLE_PROGRESS", json.dumps(state, ensure_ascii=False), flush=True)


def frozen_parameters(plan, cache):
    path = cache / "geometry-parameters.json"
    values = plan["geometry"]["parameters"]
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != values:
            raise ValueError("existing parallel geometry parameters differ from frozen plan")
    else:
        _write_json(path, values)
    return path


def child_command(blender, budget_path, plan, sock, parameters, output, *,
        baseline=None, face_start=0, face_count=None, max_new_chunks=None):
    geometry = plan["geometry"]
    request = baseline["geometry_requests"] if baseline else {
        "world_tile_mm": 2., "material_tile_mm": 2., "max_samples_per_geometry_request": 2048}
    command = [str(blender), "--background", "--factory-startup", "--threads", "1",
        "--python-exit-code", "1", "--python", str(CODE / "run_triangle_features.py"), "--",
        "--pair", str(Path(plan["inputs"][sock]["pair"])), "--output", str(output),
        "--parameters", str(parameters), "--use-budget-plan", str(budget_path),
        "--budget-sock", sock, "--representation", geometry["representation"],
        "--samples-per-radius", str(baseline["samples_per_radius"] if baseline else 2.),
        "--ray-offset-mm", str(baseline["ray_offset_mm"] if baseline else .1),
        "--halo-mm", str(baseline["halo_mm"] if baseline else .125),
        "--capsule-cap-rings", str(baseline["geometry_identity"]["capsule_cap_rings"] if baseline else 3),
        "--geometry-world-tile-mm", str(request["world_tile_mm"]),
        "--geometry-material-tile-mm", str(request["material_tile_mm"]),
        "--max-samples-per-geometry-request", str(request["max_samples_per_geometry_request"]),
        "--face-start", str(face_start), "--bake"]
    if geometry["course_repeat_count"] is not None:
        command.extend(["--course-repeat-count", str(geometry["course_repeat_count"])])
    if face_count is not None:
        command.extend(["--face-count", str(face_count)])
    if max_new_chunks is not None:
        command.extend(["--max-new-chunks", str(max_new_chunks)])
    return command


def pin_or_restore_plan(path, baseline, sock, directory, shard_root, blocks_per_shard):
    manifest = baseline["manifest"]
    common_digest = hashlib.sha256(json.dumps(baseline["common_identity"],
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["common_identity_sha256"] != common_digest
                or saved["domain_sha256"] != manifest["domain_sha256"]
                or saved["generation_signature"] != manifest["generation_signature"]
                or Path(saved["output"]).resolve() != directory.resolve()
                or saved["blocks_per_shard"] != blocks_per_shard):
            raise ValueError("existing parallel job plan belongs to different inputs, scope or chunking")
        prefix = saved["committed_prefix"]
        if len(manifest["chunks"]) < len(prefix):
            raise ValueError("formal committed prefix shrank since the parallel plan")
        for actual, pinned in zip(manifest["chunks"], prefix):
            if actual["file"] != pinned["file"] or actual["sha256"] != pinned["sha256"]:
                raise ValueError("formal committed prefix differs from the frozen parallel plan")
        expected_start = len(prefix)
        for job in saved["jobs"]:
            if (job["sock"] != sock or job["global_chunk_start"] != expected_start
                    or job["global_chunk_end"] >= len(baseline["layout"])):
                raise ValueError("frozen parallel job plan has a gap or invalid block range")
            selected = baseline["layout"][expected_start:job["global_chunk_end"] + 1]
            if (job["face_start"] != selected[0]["face_start"]
                    or job["face_count"] != selected[-1]["face_end"] - selected[0]["face_start"] + 1
                    or Path(job["output"]).resolve().parent != shard_root.resolve()):
                raise ValueError("frozen parallel job face interval/output differs from canonical blocks")
            expected_start = job["global_chunk_end"] + 1
        if expected_start != len(baseline["layout"]):
            raise ValueError("frozen parallel job plan does not cover the full remaining surface")
        return saved
    jobs = plan_shards(directory, shard_root, blocks_per_shard=blocks_per_shard)
    for job in jobs:
        job["sock"] = sock
    saved = {"schema": "stocking.parallel_triangle_job_plan.v1", "sock": sock,
        "output": str(directory.resolve()), "blocks_per_shard": blocks_per_shard,
        "common_identity_sha256": common_digest, "domain_sha256": manifest["domain_sha256"],
        "generation_signature": manifest["generation_signature"],
        "committed_prefix": [{"file": chunk["file"], "sha256": chunk["sha256"]}
                             for chunk in manifest["chunks"]], "jobs": jobs}
    _write_json(path, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-plan", type=Path, required=True)
    parser.add_argument("--blender", type=Path, help="Blender executable; required for --run")
    parser.add_argument("--output-root", type=Path, help="formal left/right destination; defaults to budget-plan parent")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "build-support/stocking-material/parallel-budget-features")
    parser.add_argument("--sock", action="append", help="repeat sock names; defaults to all frozen plan socks")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blocks-per-shard", type=int, default=26)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="launch/resume children, validate all exits, then merge")
    mode.add_argument("--merge-only", action="store_true", help="validate and merge already completed frozen shards")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be from one to eight")
    if args.blocks_per_shard < 1:
        parser.error("blocks-per-shard must be positive")
    if args.run and (not args.blender or not args.blender.is_file()):
        parser.error("--run requires an existing --blender executable")
    budget_path = args.budget_plan.resolve()
    plan = json.loads(budget_path.read_text(encoding="utf-8"))
    names = args.sock or list(plan["socks"])
    if not names or len(set(names)) != len(names) or any(name not in plan["socks"] for name in names):
        parser.error("sock names must be unique socks from the frozen budget plan")
    output_root = (args.output_root or budget_path.parent).resolve()
    cache_root = args.cache_root.resolve()
    if cache_root == output_root or cache_root.is_relative_to(output_root):
        parser.error("parallel cache must be outside the formal asset/export root")
    cache_root.mkdir(parents=True, exist_ok=True)
    parameters = frozen_parameters(plan, cache_root)
    baselines, sessions = {}, {}
    for sock in names:
        pair_path = Path(plan["inputs"][sock]["pair"])
        if _sha256(pair_path) != plan["inputs"][sock]["array_sha256"]:
            raise ValueError("frozen garment-pair array SHA changed: " + sock)
        pair = load_garment_pair(pair_path)
        sampling = prepare_triangle_sampling(pair, yarn_radius_mm=plan["yarn_radius_mm"],
                                             spacing_mm=plan["spacing_mm"])
        directory = output_root / sock
        if not (directory / "triangle_features.json").exists():
            if not args.run:
                raise ValueError("initialize the formal full-scope manifest with --run before planning: " + sock)
            # A zero-chunk original-writer call establishes full scope and domain.
            directory.mkdir(parents=True, exist_ok=True)
            bootstrap = {"sock": sock, "face_start": 0, "face_count": len(pair.triangles),
                         "output": str(directory)}
            supervise_shard_commands([bootstrap], lambda job: child_command(args.blender,
                budget_path, plan, sock, parameters, directory, max_new_chunks=0),
                lambda job: bool(inspect_feature_directory(directory, expected_sampling=sampling,
                    require_full_scope=True, allow_course_roundoff=True)), cache_root / (sock + "-bootstrap-report.json"),
                workers=1, progress=print_progress)
        baseline = inspect_feature_directory(directory, expected_sampling=sampling,
            require_full_scope=True, allow_course_roundoff=True)
        manifest = baseline["manifest"]
        if (manifest["pair_fingerprint"] != plan["socks"][sock]["pair_fingerprint"]
                or manifest.get("export_byte_limit") != plan["socks"][sock]["export_byte_limit"]
                or not manifest.get("compact_coverage")
                or manifest["max_faces_per_chunk"] != plan["max_faces_per_chunk"]
                or manifest["max_samples_per_chunk"] != plan["max_samples_per_chunk"]
                or manifest["geometry_identity"]["parameters"] != plan["geometry"]["parameters"]
                or manifest["geometry_identity"]["representation"] != plan["geometry"]["representation"]):
            raise ValueError("formal baseline identity/budget differs from frozen plan: " + sock)
        if manifest.get("geometry_requests", {}).get("protocol") != "sample_capsule_rays":
            raise ValueError("parallel budget path requires the frozen analytic capsule callback")
        identity_digest = hashlib.sha256(json.dumps(baseline["common_identity"],
            sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        session = cache_root / (sock + "-" + identity_digest)
        session.mkdir(parents=True, exist_ok=True)
        saved = pin_or_restore_plan(session / "job-plan.json", baseline, sock,
            directory, session / "shards", args.blocks_per_shard)
        baselines[sock], sessions[sock] = baseline, saved
    all_jobs = []
    # Interleave socks so all eight workers advance both assets from the start.
    for index in range(max((len(session["jobs"]) for session in sessions.values()), default=0)):
        for sock in names:
            if index < len(sessions[sock]["jobs"]):
                all_jobs.append(sessions[sock]["jobs"][index])
    print("PARALLEL_TRIANGLE_PLAN", json.dumps({"workers": args.workers,
        "blocks_per_shard": args.blocks_per_shard, "thread_environment": THREAD_ENVIRONMENT,
        "child_count": len(all_jobs), "jobs": all_jobs}, ensure_ascii=False), flush=True)
    if not args.run and not args.merge_only:
        return
    if args.run:
        def validate_child(job):
            baseline = baselines[job["sock"]]
            checked = inspect_feature_directory(job["output"], expected_sampling=baseline["sampling"],
                expected_common_identity=baseline["common_identity"], full_layout=baseline["layout"])
            if (checked["manifest"]["requested_faces"][0] != job["face_start"]
                    or len(checked["manifest"]["requested_faces"]) != job["face_count"]):
                raise ValueError("child output requested a different face interval")
            return checked["complete"]
        supervise_shard_commands(all_jobs, lambda job: child_command(args.blender, budget_path,
            plan, job["sock"], parameters, job["output"], baseline=baselines[job["sock"]]["manifest"],
            face_start=job["face_start"], face_count=job["face_count"]), validate_child,
            cache_root / "children-report.json", workers=args.workers, progress=print_progress)
    for sock in names:
        manifest = merge_feature_shards(output_root / sock,
            [job["output"] for job in sessions[sock]["jobs"]], progress=print_progress)
        audit_root = ROOT / "build-support/stocking-material/budget-features"
        reports = [audit_root / name for name in ("reference-tail-intersections.json",
            "tail-director-frames-audit.json", "capsule-crease-cache-verification.json")]
        if "sharp_reference_toe_closure" in manifest and all(path.is_file() for path in reports):
            manifest = record_closure_provenance(output_root / sock, reports)
        inspected = inspect_feature_directory(output_root / sock,
            expected_sampling=baselines[sock]["sampling"], require_full_scope=True, require_complete=True)
        print("PARALLEL_TRIANGLE_RESULT", json.dumps({"sock": sock,
            "completed_faces": manifest["completed_faces"], "completed_samples": manifest["completed_samples"],
            "verified_full_surface_features_generated": inspected["manifest"]["full_surface_features_generated"]}), flush=True)


if __name__ == "__main__":
    main()
