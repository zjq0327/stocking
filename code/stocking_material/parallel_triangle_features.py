"""Resume safe independent triangle-feature shards and strict full-domain merge.

The existing feature writer remains the only ray/feature implementation.  This
module validates its committed data, partitions whole storage blocks, supervises
isolated Blender processes, and merges identical sampling addresses.  Completion
is computed from NPZ contents, never inferred from subprocess success or flags.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .triangle_features import (TriangleSampling, CONFIGURATIONS, SIDES,
    _directory_bytes, _sha256, _write_json, face_sample_barycentric,
    iter_sample_chunks)


IDENTITY_FIELDS = ("geometry_identity", "ray_offset_mm", "halo_mm",
                   "max_samples_per_chunk", "max_faces_per_chunk")
OPTIONAL_IDENTITY_FIELDS = ("geometry_requests", "compact_coverage", "export_byte_limit")
THREAD_ENVIRONMENT = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}


def _plain_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + " differs")


def _safe_file(directory, name):
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or name in (".", ".."):
        raise ValueError("manifest files must be direct children of their asset directory")
    result = directory / relative
    if result.is_symlink():
        raise ValueError("asset manifest cannot refer to a symbolic link")
    return result


def common_identity(manifest, domain_metadata):
    """Recover exactly the writer's signed identity except requested scope."""
    identity = copy.deepcopy(domain_metadata)
    for key in IDENTITY_FIELDS:
        identity[key] = copy.deepcopy(manifest[key])
    for key in OPTIONAL_IDENTITY_FIELDS:
        if key in manifest:
            identity[key] = copy.deepcopy(manifest[key])
    return identity


def _full_layout(sampling, manifest):
    layout = []
    for index, (faces, local, _) in enumerate(iter_sample_chunks(sampling,
            max_samples=manifest["max_samples_per_chunk"],
            max_faces=manifest["max_faces_per_chunk"])):
        layout.append({"global_index": index,
            "start": int(sampling.sample_offsets[faces[0]] + local[0]),
            "end": int(sampling.sample_offsets[faces[-1]] + local[-1]),
            "samples": len(faces), "face_start": int(faces[0]),
            "face_end": int(faces[-1]), "first_local": int(local[0]),
            "last_local": int(local[-1])})
    return layout


def _verify_feature_arrays(raw, count, record):
    for configuration in CONFIGURATIONS:
        coverage = record["geometry_coverage"][configuration]
        if not coverage.get("geometry_available"):
            raise ValueError("committed chunk is missing geometry support")
        if "samples_intersected" in coverage and coverage["samples_intersected"] != count:
            raise ValueError("geometry coverage did not intersect every stored sample")
        for side in SIDES:
            prefix = configuration + "_" + side + "_"
            valid = raw[prefix + "valid"]
            if valid.dtype != np.bool_ or valid.shape != (count,):
                raise ValueError("invalid feature mask dtype/shape")
            for field in ("P_offset_mm", "N_local", "T_local"):
                values = raw[prefix + field]
                if (values.dtype != np.float32 or values.shape != (count, 3)
                        or not np.isfinite(values).all() or np.any(values[~valid] != 0)):
                    raise ValueError("invalid P/N/T feature dtype, shape or values")
            stats = record["stats"][configuration + "_" + side]
            if stats["sample_count"] != count or stats["hit_samples"] != int(valid.sum()):
                raise ValueError("saved feature statistics differ from actual samples")
            for field in ("ray_origin_outside", "ray_tangent_invalid"):
                if prefix + field in raw.files:
                    values = raw[prefix + field]
                    if values.dtype != np.bool_ or values.shape != (count,):
                        raise ValueError("invalid ray diagnostics dtype/shape")
                    invalid = ~values if field == "ray_origin_outside" else values
                    if np.any(valid & invalid):
                        raise ValueError("valid feature has an invalid ray origin/tangent")


def _sharp_reference_closure(sampling):
    triangles = sampling.arrays["triangles"]
    xyz = sampling.arrays["reference_positions_mm"][triangles]
    geometric = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    director = sampling.arrays["reference_vertex_normals"][triangles].sum(axis=1)
    denominator = np.linalg.norm(geometric, axis=1) * np.linalg.norm(director, axis=1)
    dot = np.divide(np.einsum("ij,ij->i", geometric, director), denominator,
        out=np.full(len(triangles), -1., dtype=np.float64), where=denominator > 1e-28)
    faces = np.flatnonzero(dot <= 0).astype(int).tolist()
    return {"schema": "stocking.sharp_reference_toe_closure.v1", "configuration": "reference",
        "nonpositive_face_to_vertex_director_faces": faces, "face_count": len(faces),
        "minimum_face_to_vertex_director_dot": float(dot.min()),
        "ray_side_rule": "front/back follow the continuous normalized area-weighted interpolated vertex director",
        "sharp_crease_policy": "analytic capsule support permits declared closure director creases; strict tube support keeps its orientation rejection",
        "reference_positions_unchanged": True,
        "pair_fingerprint": sampling.metadata["pair_fingerprint"]}


def inspect_feature_directory(directory, *, expected_sampling=None,
        expected_common_identity=None, require_full_scope=False,
        require_complete=False, full_layout=None, allow_course_roundoff=False):
    """Read, hash and validate actual domain/chunk contents in bounded memory.

    A shard must request one contiguous face interval aligned with the global
    writer storage blocks.  Each committed chunk must be the next whole global
    block; each face-local barycentric sample is checked, including real holes.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "triangle_features.json").read_text(encoding="utf-8"))
    domain_path = _safe_file(directory, manifest["domain_file"])
    _plain_equal(_sha256(domain_path), manifest["domain_sha256"], "sampling domain hash")
    with np.load(domain_path, allow_pickle=False) as raw:
        domain_metadata = json.loads(str(raw["metadata_json"]))
        requested = raw["requested_faces"].copy()
        arrays = {key: raw[key].copy() for key in raw.files
                  if key not in ("metadata_json", "requested_faces")}
    sampling = TriangleSampling(arrays, domain_metadata)
    if expected_sampling is not None:
        _plain_equal(domain_metadata, expected_sampling.metadata, "sampling metadata")
        _plain_equal(set(arrays), set(expected_sampling.arrays) - {"requested_faces"}, "sampling array inventory")
        for key, value in arrays.items():
            target = expected_sampling.arrays[key]
            matches = np.array_equal(value, target)
            if (not matches and allow_course_roundoff
                    and key in ("reference_face_courses", "current_face_courses")):
                # Rebuilding an existing Blender domain in a different NumPy
                # runtime can round the batched 2x2 material inverse differently.
                # This option is for original pair-to-domain verification only;
                # source shards still compare every saved domain value exactly.
                matches = value.shape == target.shape and np.allclose(value, target,
                    rtol=0., atol=1e-13, equal_nan=False)
            if value.dtype != target.dtype or not matches:
                raise ValueError("sampling domain array differs: " + key)
    face_count = len(arrays["triangles"])
    if (requested.ndim != 1 or requested.dtype.kind not in "iu" or not len(requested)
            or requested[0] < 0 or requested[-1] >= face_count
            or not np.array_equal(requested, np.arange(requested[0], requested[-1] + 1))):
        raise ValueError("shard requests must be a contiguous unique in-range face interval")
    _plain_equal(manifest["requested_faces"], requested.tolist(), "manifest/domain requested faces")
    for key, value in domain_metadata.items():
        _plain_equal(manifest[key], value, "manifest/domain " + key)
    identity = common_identity(manifest, domain_metadata)
    signed = dict(identity, requested_faces=requested.tolist())
    signature = hashlib.sha256(json.dumps(signed, sort_keys=True, allow_nan=False).encode()).hexdigest()
    _plain_equal(manifest["generation_signature"], signature, "generation signature")
    if expected_common_identity is not None:
        _plain_equal(identity, expected_common_identity, "pair/grid/yarn/ray/window identity")
    full = np.array_equal(requested, np.arange(face_count))
    if require_full_scope and not full:
        raise ValueError("destination must request the true full surface")
    offsets = sampling.sample_offsets
    if (offsets.dtype != np.int64 or offsets.shape != (face_count + 1,)
            or offsets[0] != 0 or np.any(np.diff(offsets) < 3)):
        raise ValueError("invalid material sample offsets")
    layout = _full_layout(sampling, manifest) if full_layout is None else full_layout
    by_start = {block["start"]: block for block in layout}
    interval_start, interval_end = int(offsets[requested[0]]), int(offsets[requested[-1] + 1])
    if interval_start not in by_start or interval_end not in {block["end"] + 1 for block in layout}:
        raise ValueError("shard boundaries do not align with full-domain storage blocks")
    requested_samples = interval_end - interval_start
    _plain_equal(manifest["estimate"]["requested_samples"], requested_samples, "requested sample estimate")
    _plain_equal(manifest["estimate"]["requested_faces"], len(requested), "requested face estimate")
    next_address, completed_faces = interval_start, 0
    chunks = []
    for number, record in enumerate(manifest["chunks"]):
        path = _safe_file(directory, record["file"])
        _plain_equal(record["file"], "chunk-%06d.npz" % number, "local chunk filename/order")
        _plain_equal(_sha256(path), record["sha256"], "feature chunk hash " + record["file"])
        if next_address not in by_start:
            raise ValueError("chunk does not begin at the next global storage block")
        block = by_start[next_address]
        if block["end"] >= interval_end:
            raise ValueError("chunk exceeds its requested face interval")
        with np.load(path, allow_pickle=False) as raw:
            faces, local, bary = [raw[key] for key in ("face_indices", "face_local_indices", "barycentric")]
            if (faces.dtype != np.int32 or local.dtype != np.int64 or bary.dtype != np.float64
                    or faces.shape != (block["samples"],) or local.shape != faces.shape
                    or bary.shape != (len(faces), 3) or np.any(faces < requested[0])
                    or np.any(faces > requested[-1]) or np.any(local < 0)
                    or np.any(local >= np.diff(offsets)[faces])):
                raise ValueError("invalid face/local/barycentric sample address layout")
            addresses = offsets[faces] + local
            if not np.array_equal(addresses, np.arange(block["start"], block["end"] + 1)):
                raise ValueError("actual sample addresses have a gap, duplicate or permutation")
            for face in np.unique(faces):
                selected = faces == face
                if not np.array_equal(bary[selected], face_sample_barycentric(sampling, face, local[selected])):
                    raise ValueError("actual barycentric samples differ from the frozen material grid")
            _plain_equal(record["samples"], len(faces), "chunk sample count")
            _plain_equal(record["faces"], np.unique(faces).tolist(), "chunk face summary")
            _plain_equal(record["first_address"], [int(faces[0]), int(local[0])], "chunk first address")
            _plain_equal(record["last_address"], [int(faces[-1]), int(local[-1])], "chunk last address")
            _verify_feature_arrays(raw, len(faces), record)
            completed_faces += len(np.unique(faces[local == np.diff(offsets)[faces] - 1]))
        chunks.append(dict(block, record=record, path=path))
        next_address = block["end"] + 1
    completed_samples = next_address - interval_start
    complete = completed_samples == requested_samples
    for key, value in (("completed_samples", completed_samples), ("completed_faces", completed_faces),
            ("requested_scope_full", bool(full)), ("requested_scope_complete", complete),
            ("full_surface_features_generated", bool(full and complete))):
        _plain_equal(manifest[key], value, "actual data versus " + key)
    if require_complete and not complete:
        raise ValueError("shard has not generated every requested sample")
    limit = manifest.get("export_byte_limit")
    if limit is not None and _directory_bytes(directory) > limit:
        raise ValueError("feature directory exceeds its frozen export byte limit")
    return {"directory": directory, "manifest": manifest, "sampling": sampling,
            "common_identity": identity, "chunks": chunks, "layout": layout,
            "next_address": next_address, "interval_start": interval_start,
            "interval_end": interval_end, "complete": complete}


def plan_shards(output_directory, shard_root, *, blocks_per_shard=26):
    """Plan bounded whole-face shards after the validated committed prefix."""
    if type(blocks_per_shard) is not int or blocks_per_shard < 1:
        raise ValueError("blocks_per_shard must be a positive integer")
    inspected = inspect_feature_directory(output_directory, require_full_scope=True)
    sampling, manifest, layout = inspected["sampling"], inspected["manifest"], inspected["layout"]
    block_faces = manifest["max_faces_per_chunk"]
    for block in layout:
        if (block["first_local"] != 0
                or block["last_local"] != np.diff(sampling.sample_offsets)[block["face_end"]] - 1
                or block["face_start"] % block_faces != 0
                or block["face_end"] - block["face_start"] + 1 != block_faces
                   and block["face_end"] + 1 != len(sampling.arrays["triangles"])):
            raise ValueError("parallel whole-face scheduling requires complete aligned storage blocks")
    prefix = len(inspected["chunks"])
    jobs = []
    for start in range(prefix, len(layout), blocks_per_shard):
        selected = layout[start:start + blocks_per_shard]
        face_start, face_end = selected[0]["face_start"], selected[-1]["face_end"]
        count = face_end - face_start + 1
        jobs.append({"face_start": face_start, "face_count": count,
            "global_chunk_start": start, "global_chunk_end": selected[-1]["global_index"],
            "samples": sum(block["samples"] for block in selected),
            "output": str((Path(shard_root) / ("shard-f%06d-n%06d" % (face_start, count))).resolve()),
            "existing_prefix_chunks": prefix})
    return jobs


def merge_feature_shards(output_directory, shard_directories, *, progress=None):
    """Verify all sources before committing a true full-domain manifest.

    Existing committed bytes and records are retained.  Interrupted copies are
    adopted only when their hashes equal the verified source.  Every mutation
    preserves the allocated per-sock byte guard; committed flags always describe
    the actual contiguous samples present in the current manifest.
    """
    baseline = inspect_feature_directory(output_directory, require_full_scope=True)
    directory, sampling = baseline["directory"], baseline["sampling"]
    candidates = {chunk["global_index"]: chunk for chunk in baseline["chunks"]}
    source_info = []
    for source in shard_directories:
        inspected = inspect_feature_directory(source, expected_sampling=sampling,
            expected_common_identity=baseline["common_identity"], require_complete=True,
            full_layout=baseline["layout"])
        source_info.append({"domain_sha256": inspected["manifest"]["domain_sha256"],
            "generation_signature": inspected["manifest"]["generation_signature"],
            "requested_face_range": [inspected["manifest"]["requested_faces"][0],
                                     inspected["manifest"]["requested_faces"][-1]]})
        for chunk in inspected["chunks"]:
            index = chunk["global_index"]
            if index in candidates and candidates[index]["record"]["sha256"] != chunk["record"]["sha256"]:
                raise ValueError("different content supplied for the same global storage block")
            candidates.setdefault(index, chunk)
    expected_indices = list(range(len(baseline["layout"])))
    if sorted(candidates) != expected_indices:
        missing = sorted(set(expected_indices) - set(candidates))
        raise ValueError("cannot produce full features: missing global storage blocks " + str(missing))
    manifest = copy.deepcopy(baseline["manifest"])
    prefix = len(manifest["chunks"])
    limit = manifest.get("export_byte_limit")
    manifest["valid_semantics"] = ("first hit within +/-ray_offset of specified smooth-normal ray; "
        "false means no hit or separately diagnosed invalid origin/tangent; missing geometry aborts uncommitted chunk")
    if manifest.get("geometry_requests", {}).get("protocol") == "sample_capsule_rays":
        manifest["actual_intersection_error_model"] = {
            "method": "float64 analytic ray/polyline capsule union",
            "tube_polygon_and_Blender_BVH_used": False,
            "sources": ["centerline polyline discretization", "piecewise shell and interpolated director",
                "finite near-band/local material support", "material sample spacing",
                "float64 algebra and boundary tolerances", "float32 stored feature quantization"],
            "legacy_error_sources_scope": "generic sampling metadata; actual method described here"}
        if manifest.get("geometry_identity", {}).get("provider") == "AuthoredYarnProvider-v1":
            closure = _sharp_reference_closure(sampling)
            previous = manifest.get("sharp_reference_toe_closure", {})
            if (previous.get("pair_fingerprint") == closure["pair_fingerprint"]
                    and previous.get("nonpositive_face_to_vertex_director_faces") == closure["nonpositive_face_to_vertex_director_faces"]):
                for key in ("pair_file_sha256", "evidence_reports", "evidence_scope"):
                    if key in previous:
                        closure[key] = copy.deepcopy(previous[key])
            manifest["sharp_reference_toe_closure"] = closure
    if prefix == len(expected_indices):
        # Still validate every provided source above.  A repeated completed merge
        # does not rewrite provenance or change any committed bytes.
        if manifest != baseline["manifest"]:
            _write_json(directory / "triangle_features.json", manifest, byte_limit=limit)
        return manifest
    new_bytes = 0
    for index in range(prefix, len(expected_indices)):
        chunk = candidates[index]
        target = directory / ("chunk-%06d.npz" % index)
        if target.exists():
            if _sha256(target) != chunk["record"]["sha256"]:
                raise ValueError("uncommitted destination file has different content: " + target.name)
        else:
            new_bytes += chunk["path"].stat().st_size
    final = copy.deepcopy(manifest)
    for index in range(prefix, len(expected_indices)):
        record = copy.deepcopy(candidates[index]["record"])
        record["file"] = "chunk-%06d.npz" % index
        final["chunks"].append(record)
    final.update(completed_samples=int(sampling.sample_offsets[-1]),
        completed_faces=len(sampling.arrays["triangles"]), requested_scope_complete=True,
        full_surface_features_generated=True)
    final.pop("last_failure", None)
    final["parallel_merge"] = {"schema": "stocking.parallel_triangle_merge.v1",
        "validated_actual_addresses": True, "validated_barycentric_grid": True,
        "validated_source_hashes": True, "initial_committed_prefix_chunks": prefix,
        "source_shards": source_info}
    final_bytes = len((json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    existing_metadata_bytes = (directory / "triangle_features.json").stat().st_size
    if limit is not None and _directory_bytes(directory) + new_bytes + final_bytes - existing_metadata_bytes > limit:
        raise ValueError("merged full features would exceed the frozen per-sock export byte limit")
    # All hash/address/domain/identity and byte preflight is complete before any write.
    for index in range(prefix, len(expected_indices)):
        chunk = candidates[index]
        target = directory / ("chunk-%06d.npz" % index)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                shutil.copyfile(chunk["path"], temporary)
                if _sha256(temporary) != chunk["record"]["sha256"]:
                    raise ValueError("source changed while copying storage block")
                if limit is not None and _directory_bytes(directory) + temporary.stat().st_size + final_bytes > limit:
                    raise ValueError("per-sock export byte limit exceeded before committing storage block")
                # Do not replace a different file if an external process created it.
                if target.exists():
                    if _sha256(target) != chunk["record"]["sha256"]:
                        raise ValueError("destination changed while merging storage block")
                    temporary.unlink()
                else:
                    temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        record = copy.deepcopy(chunk["record"])
        record["file"] = target.name
        manifest["chunks"].append(record)
        manifest["completed_samples"] = chunk["end"] + 1
        manifest["completed_faces"] = chunk["face_end"] + 1
        complete = manifest["completed_samples"] == int(sampling.sample_offsets[-1])
        manifest["requested_scope_complete"] = complete
        manifest["full_surface_features_generated"] = complete
        manifest.pop("last_failure", None)
        _write_json(directory / "triangle_features.json", manifest, byte_limit=limit)
        if progress:
            progress({"event": "merge_chunk_committed", "global_chunk": index,
                "completed_samples": manifest["completed_samples"],
                "completed_faces": manifest["completed_faces"],
                "requested_scope_complete": complete})
    _write_json(directory / "triangle_features.json", final, byte_limit=limit)
    return final


def record_closure_provenance(output_directory, report_paths):
    """Attach hashed authored-closure evidence without changing feature identity.

    Evidence is bound to the sock's frozen pair NPZ SHA.  It retains each report's
    scope and limitations; the reports do not certify global yarn first hits.
    Only metadata is written, using the existing per-sock byte guard.
    """
    directory = Path(output_directory).resolve()
    path = directory / "triangle_features.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("full_surface_features_generated"):
        raise ValueError("closure provenance requires the already verified full feature manifest")
    if "sharp_reference_toe_closure" not in manifest:
        raise ValueError("verified feature manifest is missing its explicit closure semantics")
    plan = json.loads((directory.parent / "budget-plan.json").read_text(encoding="utf-8"))
    sock = directory.name
    _plain_equal(plan["socks"][sock]["pair_fingerprint"], manifest["pair_fingerprint"], "closure pair identity")
    expected_sha = plan["inputs"][sock]["array_sha256"]
    _plain_equal(_sha256(plan["inputs"][sock]["pair"]), expected_sha, "closure frozen pair file SHA")
    root = Path(__file__).resolve().parents[3]
    reports = []
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        sha = _sha256(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        socks = report["socks"]
        record = socks[sock] if isinstance(socks, dict) else next(item for item in socks if item["sock"] == sock)
        if "pair_file_sha256" in record:
            _plain_equal(record["pair_file_sha256"], expected_sha, "closure evidence frozen pair SHA")
        else:
            _plain_equal(record["pair_sha256_before"], expected_sha, "closure evidence input pair SHA")
            _plain_equal(record["pair_sha256_after"], expected_sha, "closure evidence unchanged pair SHA")
        if "sampling_domain_sha256" in record:
            _plain_equal(record["sampling_domain_sha256"], manifest["domain_sha256"], "closure evidence sampling domain SHA")
        if "budget_plan_sha256" in report:
            _plain_equal(report["budget_plan_sha256"], _sha256(directory.parent / "budget-plan.json"), "closure evidence budget SHA")
        if "passed" in report and report["passed"] is not True:
            raise ValueError("closure regression report did not pass")
        _plain_equal(_sha256(report_path), sha, "closure report changed during read")
        reports.append({"path": report_path.relative_to(root).as_posix()
            if report_path.is_relative_to(root) else str(report_path), "sha256": sha})
    updated = copy.deepcopy(manifest)
    updated["sharp_reference_toe_closure"].update(pair_file_sha256=expected_sha,
        evidence_reports=reports,
        evidence_scope="float64 director/frame bounds and geometric incidence plus real capsule/cache regression; no exact arithmetic or global yarn first-hit certificate")
    if updated != manifest:
        _write_json(path, updated, byte_limit=manifest.get("export_byte_limit"))
    return updated


def supervise_shard_commands(jobs, command_builder, validator, report_path, *,
        workers=8, progress=None):
    """Run bounded hidden children, recording every exit and completed shard.

    ``validator(job)`` must verify complete data rather than trust an exit code.
    Existing complete shards are skipped; unfinished shards use writer resume.
    A failed child prevents merge even when Blender itself returned zero.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between one and eight")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **THREAD_ENVIRONMENT)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results, active, pending = [], {}, list(jobs)
    report = {"schema": "stocking.parallel_triangle_children.v1", "workers": workers,
        "thread_environment": THREAD_ENVIRONMENT, "jobs": results,
        "all_children_observed": False, "passed": False}
    def save():
        _write_json(report_path, report)
    def notify(state):
        if progress:
            progress(state)
    save()
    last_progress = 0.
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                result = {"sock": job.get("sock"), "face_start": job["face_start"],
                    "face_count": job["face_count"], "output": job["output"],
                    "exit_code": None, "validated_complete": False}
                results.append(result)
                output = Path(job["output"])
                if (output / "triangle_features.json").exists():
                    try:
                        if validator(job):
                            result.update(skipped_existing_complete=True, validated_complete=True)
                            save()
                            notify({"event": "shard_existing_complete", **result})
                            continue
                    except Exception as exc:
                        result.update(error="existing shard validation failed: " + str(exc))
                        save()
                        continue
                log_path = report_path.parent / ("%s-%s-f%06d.log" % (stamp, job.get("sock", "sock"), job["face_start"]))
                log = log_path.open("ab")
                result["log"] = str(log_path.resolve())
                result["command"] = command_builder(job)
                started = time.monotonic()
                try:
                    child = subprocess.Popen(result["command"], stdout=log, stderr=subprocess.STDOUT,
                        env=environment, creationflags=flags)
                    result["pid"] = child.pid
                    active[child.pid] = (child, log, started, job, result)
                    notify({"event": "shard_started", "pid": child.pid,
                        "sock": job.get("sock"), "face_start": job["face_start"], "face_count": job["face_count"]})
                except Exception as exc:
                    log.close()
                    result["error"] = "child launch failed: " + str(exc)
                save()
            for pid, (child, log, started, job, result) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                child.wait()
                log.close()
                result.update(exit_code=int(code), elapsed_seconds=time.monotonic() - started)
                if code == 0:
                    try:
                        result["validated_complete"] = bool(validator(job))
                        if not result["validated_complete"]:
                            result["error"] = "child exited zero with incomplete actual samples"
                    except Exception as exc:
                        result["error"] = "child output validation failed: " + str(exc)
                else:
                    result["error"] = "child exited with code %d" % code
                del active[pid]
                save()
                notify({"event": "shard_finished", **result})
            if time.monotonic() - last_progress >= 30:
                running = []
                for child, _, started, job, _ in active.values():
                    state = {"pid": child.pid, "sock": job.get("sock"), "face_start": job["face_start"],
                             "elapsed_seconds": time.monotonic() - started}
                    manifest_path = Path(job["output"]) / "triangle_features.json"
                    if manifest_path.exists():
                        try:
                            value = json.loads(manifest_path.read_text(encoding="utf-8"))
                            state.update(completed_samples=value["completed_samples"],
                                completed_faces=value["completed_faces"])
                        except (OSError, ValueError, KeyError):
                            pass
                    running.append(state)
                notify({"event": "parallel_progress", "pending": len(pending), "running": running,
                    "complete": sum(bool(r["validated_complete"]) for r in results)})
                last_progress = time.monotonic()
            if active:
                time.sleep(.25)
    except BaseException:
        report["interrupted"] = True
        for child, _, _, _, _ in active.values():
            child.terminate()
        for child, log, started, _, result in active.values():
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
            log.close()
            result.update(exit_code=int(code), elapsed_seconds=time.monotonic() - started,
                error="supervisor interrupted; child stopped and observed")
        report["all_children_observed"] = True
        save()
        raise
    report["all_children_observed"] = True
    report["passed"] = len(results) == len(jobs) and all(r["validated_complete"] for r in results)
    save()
    if not report["passed"]:
        raise RuntimeError("one or more shards failed; inspect child report: " + str(report_path))
    return report
