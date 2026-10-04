"""Portable, byte-bounded export of the authored geometric stocking data.

Only committed, complete feature manifests are accepted.  A package contains
the two material mesh pairs, their binding/strain data, actual feature chunks,
and a checksum inventory.  Historical experiments and high-density tube meshes
are not copied.  The cap covers the ZIP and its adjacent verification JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

from .asset_io import load_garment_pair
from .triangle_features import pair_fingerprint, prepare_triangle_sampling, face_sample_barycentric

SIDES = ("left", "right")
REPORT_RESERVE = 64_000
FIT_FILES = (
    "garment_reference.npz", "garment_reference.json",
    "garment_pair_unposed.npz", "garment_pair_unposed.json",
    "garment_pair.npz", "garment_pair.json", "binding.npz", "binding.json",
    "deformation_field.npz", "deformation_field.json",
    "deformation_field_unposed.npz", "deformation_field_unposed.json",
    "quality_current.npz", "quality_unposed.npz", "validation.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _member_path(directory, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("manifest file is not a relative asset path: " + name)
    path = (directory / Path(*relative.parts)).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("manifest asset missing/outside its directory: " + name)
    return path


def _validate_full_samples(folder, metadata, pair):
    """Check persisted addresses and channels, not just completion booleans."""
    sampling = prepare_triangle_sampling(pair, yarn_radius_mm=metadata["yarn_radius_mm"],
                                         spacing_mm=metadata["spacing_mm"])
    faces_all = np.arange(len(pair.triangles), dtype=np.int64)
    if not np.array_equal(metadata["requested_faces"], faces_all):
        raise ValueError("feature requested_faces is not the complete material surface")
    domain_path = _member_path(folder, metadata["domain_file"])
    if sha256_file(domain_path) != metadata["domain_sha256"]:
        raise ValueError("committed sampling domain changed")
    with np.load(domain_path, allow_pickle=False) as domain:
        if not np.array_equal(domain["requested_faces"], faces_all):
            raise ValueError("sampling domain contains only a partial surface")
        domain_metadata = json.loads(str(domain["metadata_json"]))
        if (domain_metadata.get("pair_fingerprint") != pair_fingerprint(pair)
                or domain_metadata.get("pair_fingerprint") != metadata["pair_fingerprint"]
                or domain_metadata.get("spacing_mm") != metadata["spacing_mm"]
                or domain_metadata.get("yarn_radius_mm") != metadata["yarn_radius_mm"]
                or domain_metadata.get("id_channels") != []):
            raise ValueError("sampling domain metadata differs from the feature/pair identity")
        for name, expected in sampling.arrays.items():
            if name not in domain:
                raise ValueError("persisted sampling domain is missing: " + name)
            actual = domain[name]
            derived_float = name in ("reference_vertex_normals", "current_vertex_normals",
                                     "reference_face_courses", "current_face_courses", "edge_longest_mm")
            equal = (np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
                     if derived_float and actual.shape == expected.shape
                     else np.array_equal(actual, expected))
            if not equal:
                raise ValueError("persisted sampling domain differs from the pair/grid: " + name)
    offsets = sampling.sample_offsets
    expected_count = int(offsets[-1])
    if (metadata["completed_samples"] != expected_count
            or metadata["estimate"]["requested_samples"] != expected_count):
        raise ValueError("feature completed/estimated samples disagree with the complete domain")
    completed = 0
    counts = np.diff(offsets)
    for chunk in metadata["chunks"]:
        source = _member_path(folder, chunk["file"])
        if sha256_file(source) != chunk["sha256"]:
            raise ValueError("committed feature chunk changed")
        with np.load(source, allow_pickle=False) as values:
            faces = values["face_indices"]
            local = values["face_local_indices"]
            bary = values["barycentric"]
            length = len(faces)
            if (faces.ndim != 1 or local.shape != faces.shape
                    or faces.dtype.kind not in "iu" or local.dtype.kind not in "iu"
                    or not length or bary.shape != (length, 3)
                    or np.any(faces < 0) or np.any(faces >= len(counts))
                    or np.any(local < 0) or np.any(local >= counts[faces])):
                raise ValueError("invalid feature sample address arrays")
            global_address = offsets[faces] + local
            if not np.array_equal(global_address, np.arange(completed, completed + length)):
                raise ValueError("feature chunks have missing, repeated or out-of-order sample addresses")
            if (chunk["samples"] != length or chunk["first_address"] != [int(faces[0]), int(local[0])]
                    or chunk["last_address"] != [int(faces[-1]), int(local[-1])]):
                raise ValueError("feature chunk address summary disagrees with saved arrays")
            for face in np.unique(faces):
                chosen = np.flatnonzero(faces == face)
                expected = face_sample_barycentric(sampling, int(face), local[chosen])
                if not np.allclose(bary[chosen], expected, rtol=0, atol=1e-12):
                    raise ValueError("feature barycentric addresses disagree with their material sample grid")
            for configuration in ("reference", "current"):
                for side in ("front", "back"):
                    prefix = configuration + "_" + side + "_"
                    valid = values[prefix + "valid"]
                    if valid.dtype != np.bool_ or valid.shape != (length,):
                        raise ValueError("valid must be bool per material sample")
                    channels = {}
                    for name in ("P_offset_mm", "N_local", "T_local"):
                        channel = values[prefix + name]
                        if (channel.dtype != np.float32 or channel.shape != (length, 3)
                                or not np.isfinite(channel).all() or np.any(channel[~valid] != 0)):
                            raise ValueError("invalid feature channel values: " + prefix + name)
                        channels[name] = channel
                    normal, tangent = channels["N_local"][valid], channels["T_local"][valid]
                    if (np.any(np.abs(np.linalg.norm(normal, axis=1) - 1) > 2e-5)
                            or np.any(np.abs(np.linalg.norm(tangent, axis=1) - 1) > 2e-5)
                            or np.any(np.abs(np.einsum("ij,ij->i", normal, tangent)) > 2e-5)):
                        raise ValueError("valid feature directions are not unit/orthogonal")
                    for name, required in (("ray_hit", True), ("ray_origin_outside", True),
                                           ("ray_tangent_invalid", False)):
                        if prefix + name in values and np.any(values[prefix + name][valid] != required):
                            raise ValueError("valid feature contradicts ray diagnostic " + name)
            completed += length
    if completed != expected_count:
        raise ValueError("saved feature chunks do not cover every material sample")


def plan_authored_export(fit_directory, feature_directory, *, maximum_bytes=1_000_000_000,
                         extras=()):
    """Validate identities/full coverage before making a portable inventory."""
    if isinstance(maximum_bytes, bool) or maximum_bytes <= REPORT_RESERVE:
        raise ValueError("maximum_bytes must allow a package and verification report")
    fit = Path(fit_directory).resolve()
    features = Path(feature_directory).resolve()
    files = []
    scopes = {}

    def add(source, member, expected_hash=None):
        source = Path(source)
        digest = sha256_file(source)
        if expected_hash is not None and digest != expected_hash:
            raise ValueError("committed asset checksum changed: " + str(source))
        if any(item["member"] == member for item in files):
            raise ValueError("duplicate package member: " + member)
        files.append({"source": str(source), "member": member,
                      "bytes": source.stat().st_size, "sha256": digest})

    for side in SIDES:
        pair = load_garment_pair(fit / side / "garment_pair")
        if pair.metadata.get("side") != side:
            raise ValueError(side + " material mesh pair has a different sock side identity")
        folder = features / side
        metadata = _read_json(folder / "triangle_features.json")
        if not all(metadata.get(key) is True for key in (
                "requested_scope_full", "requested_scope_complete", "full_surface_features_generated")):
            raise ValueError(side + " features are incomplete; partial data cannot be packaged as full stockings")
        if metadata.get("completed_faces") != len(pair.triangles):
            raise ValueError(side + " feature face count disagrees with the material mesh")
        if metadata.get("pair_fingerprint") != pair_fingerprint(pair):
            raise ValueError(side + " features belong to a different material mesh pair")
        if metadata.get("id_channels") != []:
            raise ValueError("the requested export contains no optical ID channels")
        _validate_full_samples(folder, metadata, pair)
        for name in FIT_FILES:
            add(fit / side / name, "fit/" + side + "/" + name)
        add(folder / "triangle_features.json", "features/" + side + "/triangle_features.json")
        name = metadata["domain_file"]
        add(_member_path(folder, name), "features/" + side + "/" + name,
            metadata["domain_sha256"])
        for chunk in metadata["chunks"]:
            add(_member_path(folder, chunk["file"]), "features/" + side + "/" + chunk["file"],
                chunk["sha256"])
        scopes[side] = {"faces": len(pair.triangles), "samples": metadata["completed_samples"],
                        "spacing_mm": metadata["spacing_mm"],
                        "pair_npz_sha256": sha256_file(fit / side / "garment_pair.npz"),
                        "full_surface_features_generated": True,
                        "geometry_identity": metadata.get("geometry_identity"),
                        "sampling_error_bound_certified": metadata.get("sampling_error_bound_certified", False)}
    add(fit / "manifest.json", "fit/manifest.json")
    budget = features / "budget-plan.json"
    if budget.exists():
        add(budget, "features/budget-plan.json")
    for source, member in extras:
        relative = PurePosixPath(member)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("extra member must be a relative package path")
        add(source, str(relative))
    source_bytes = sum(item["bytes"] for item in files)
    # ZIP_STORED does not inflate assets.  Two ZIP headers per file, a modest
    # inventory allowance, and an external report have explicit reservations.
    upper_bound = source_bytes + sum(1024 + 2 * len(item["member"].encode("utf-8"))
                                    for item in files) + 250_000 + REPORT_RESERVE
    if upper_bound > maximum_bytes:
        raise ValueError(f"export would exceed byte cap: conservative {upper_bound:,} > {maximum_bytes:,}")
    return {"schema": "stocking.authored_export.v1", "maximum_bytes": int(maximum_bytes),
            "source_asset_bytes": source_bytes, "conservative_package_bytes": upper_bound,
            "sides": scopes, "files": files,
            "configuration": ["reference", "current"], "feature_sides": ["front", "back"],
            "feature_channels": ["P_offset_mm", "N_local", "T_local", "valid"],
            "id_channels": [], "mechanics_validated": False,
            "scope": "synthetic plain-knit geometry mapped to authored frame 996",
            "full_high_density_tube_mesh_included": False}


def write_authored_export(plan, output_directory):
    """Write, re-read all checksums, then publish within the strict byte cap."""
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "stocking-geometry-1gb.zip"
    report_path = output / "verification.json"
    temporary = output / "stocking-geometry-1gb.zip.partial"
    if archive.exists() or report_path.exists() or temporary.exists():
        raise ValueError("export output already exists; choose a new version directory")
    inventory = {key: value for key, value in plan.items() if key != "files"}
    inventory["files"] = [{key: value for key, value in item.items() if key != "source"}
                          for item in plan["files"]]
    payload = json.dumps(inventory, ensure_ascii=False, indent=2).encode("utf-8")
    budget = plan["maximum_bytes"]
    if len(payload) > 250_000:
        raise ValueError("inventory exceeds its reserved storage")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as writer:
            writer.writestr("export.json", payload)
            for item in plan["files"]:
                source = Path(item["source"])
                if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
                    raise ValueError("input changed after inventory: " + str(source))
                writer.write(source, item["member"])
                if temporary.stat().st_size > budget - REPORT_RESERVE:
                    raise ValueError("actual archive exceeded byte cap before finalization")
        if temporary.stat().st_size > budget - REPORT_RESERVE:
            raise ValueError("actual finalized ZIP exceeded byte cap")
        with zipfile.ZipFile(temporary) as reader:
            for item in plan["files"]:
                digest = hashlib.sha256()
                with reader.open(item["member"]) as source:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError("export readback checksum mismatch: " + item["member"])
        report = {"schema": "stocking.authored_export_verification.v1",
                  "maximum_bytes": budget, "archive": archive.name,
                  "archive_bytes": temporary.stat().st_size,
                  "archive_sha256": sha256_file(temporary), "readback_all_members_passed": True,
                  "file_count": len(plan["files"]), "sides": {
                      side: {key: value for key, value in plan["sides"][side].items()
                             if key != "geometry_identity"} for side in SIDES}}
        report["total_export_bytes"] = 0
        report["within_byte_cap"] = True
        for _ in range(6):
            report_payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
            actual = temporary.stat().st_size + len(report_payload)
            if actual == report["total_export_bytes"]:
                break
            report["total_export_bytes"] = actual
            report["within_byte_cap"] = actual <= budget
        else:
            raise ValueError("verification size did not reach a fixed point")
        if not report["within_byte_cap"] or len(report_payload) > REPORT_RESERVE:
            raise ValueError("package plus verification JSON exceeded byte cap")
        temporary.replace(archive)
        report_path.write_bytes(report_payload)
        return report
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
