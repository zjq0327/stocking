"""Export verified local yarn C assets without exceeding the shared 1 GB cap.

The previous authored stocking ZIP and its verification JSON are read-only.
Their actual bytes plus this supplemental ZIP and its adjacent report must
remain within the same cap.  The default invocation only validates/plans;
pass --write to create a new, checksummed ZIP after all QA gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
REPORT_RESERVE = 64_000
INVENTORY_RESERVE = 250_000
MODULES = ("__init__.py", "yarn_coverage.py", "analytic_yarn_queries.py",
           "authored_yarn.py", "asset_io.py", "garment_data.py",
           "parameters.py", "geometry.py", "deformation_field.py")
AUXILIARY_SOURCES = ("verify-stocking-yarn-coverage.py",
                     "verify-stocking-yarn-coverage-actual.py",
                     "render-stocking-yarn-coverage.py",
                     "plot-stocking-yarn-coverage.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def member_path(directory, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in name:
        raise ValueError("asset member must be a relative POSIX path: " + name)
    path = (directory / Path(*relative.parts)).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("asset missing or outside its directory: " + name)
    return path


def _node_gate(audit):
    return (audit.get("one_factor_link") is True and
            audit.get("extra_alpha_multiplier") is False and
            audit.get("material_alpha") == 1. and audit.get("yarn_color_alpha") == 1. and
            audit.get("texture_colorspace") == "Non-Color" and
            audit.get("packed_image") is True)


def _skip_asset(relative):
    parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    return (bool(parts & {"__pycache__", "backups", "backup", "archives", "archive"}) or
            name.endswith((".zip", ".7z", ".rar", ".tar", ".gz", ".bak", ".backup",
                           ".blend1", ".blend2", ".partial", ".tmp", "~")) or
            ".inprogress." in name)


def plan_coverage_export(asset_directory, old_export_directory, *,
                         actual_report, shader_report, core_report,
                         preview_directory, maximum_bytes=1_000_000_000,
                         include_pair=True):
    """Require actual coverage/shader QA and inventory the committed inputs."""
    if (not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or
            maximum_bytes <= REPORT_RESERVE + INVENTORY_RESERVE):
        raise ValueError("maximum_bytes must allow archives plus a verification report")
    assets = Path(asset_directory).resolve()
    old = Path(old_export_directory).resolve()
    previews = Path(preview_directory).resolve()
    manifest_path = assets / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    for key in ("all_cases_generated", "expanded_window_stable_all_cases",
                "pixel_footprint_filtering_generated", "C_uses_geometric_hit_not_PNT_valid", "no_second_F"):
        if manifest.get(key) is not True:
            raise ValueError("coverage manifest gate is not true: " + key)
    expected_cases = {(state, angle) for state in ("reference", "current") for angle in (0., 45., 75.)}
    cases = manifest.get("cases", [])
    if len(cases) != 6 or {(case["state"], float(case["angle_deg"])) for case in cases} != expected_cases:
        raise ValueError("coverage requires the six expected state/angle cases")
    if len({case["name"] for case in cases}) != 6:
        raise ValueError("coverage case names are repeated")
    for case in cases:
        if case.get("expanded_window_stable") is not True:
            raise ValueError("case local source expansion guard failed: " + case["name"])
        source = member_path(assets, case["npz_file"])
        if sha256_file(source) != case["sha256"]:
            raise ValueError("coverage case checksum changed: " + case["name"])
        with np.load(source, allow_pickle=False) as arrays:
            for samples in (8, 16, 32):
                coverage, counts = arrays["coverage_" + str(samples)], arrays["hit_count_" + str(samples)]
                if (coverage.dtype != np.float32 or counts.dtype != np.uint32 or
                        coverage.shape != tuple(case["resolution"]) or counts.shape != coverage.shape or
                        not np.isfinite(coverage).all() or np.any(counts > samples ** 2) or
                        not np.array_equal(coverage, counts.astype(np.float32) / np.float32(samples ** 2))):
                    raise ValueError("saved coverage/count values disagree: " + case["name"])
            if (not np.array_equal(arrays["hit_count_32"], arrays["expanded_hit_count_32"]) or
                    not np.array_equal(arrays["coverage_32"], arrays["expanded_coverage_32"])):
                raise ValueError("saved local window union differs: " + case["name"])
    for source in manifest["source_windows"].values():
        if sha256_file(member_path(assets, source["file"])) != source["sha256"]:
            raise ValueError("source geometry checksum changed")
    parent_name = manifest.get("parent_full_field_manifest_file")
    if parent_name and sha256_file(member_path(assets, parent_name)) != manifest["parent_full_field_manifest_sha256"]:
        raise ValueError("retained full-field diagnostic manifest changed")

    actual = read_json(actual_report)
    actual_scope = actual.get("actual", {})
    if (actual.get("passed") is not True or actual_scope.get("manifest_sha256") != manifest_hash or
            actual_scope.get("all_local_source_windows_observed_stable") is not True or
            actual_scope.get("frozen_pair_sha256") != manifest["pair_sha256"]):
        raise ValueError("actual geometry QA is missing, failed, or belongs to a different manifest")
    if {item["name"] for item in actual_scope.get("cases", [])} != {case["name"] for case in cases}:
        raise ValueError("actual QA does not cover every exported case")
    core = read_json(core_report)
    if core.get("passed") is not True:
        raise ValueError("capsule coverage core verification failed")
    shader = read_json(shader_report)
    if (shader.get("passed") is not True or shader.get("saved_blend_reopened") is not True or
            Path(shader["source_manifest"]).resolve() != manifest_path.resolve()):
        raise ValueError("shader QA failed or references a different asset manifest")
    shader_cases = shader.get("cases", [])
    if len(shader_cases) != 6 or {item["name"] for item in shader_cases} != {case["name"] for case in cases}:
        raise ValueError("shader QA does not cover every exported case")
    by_name = {case["name"]: case for case in cases}
    for item in shader_cases:
        case = by_name[item["name"]]
        if (not _node_gate(item.get("node_audit", {})) or not _node_gate(item.get("saved_blend_node_audit", {})) or
                item.get("coverage_key") != "coverage_32" or item.get("shape") != case["resolution"] or
                abs(float(item["mean_C"]) - float(case["mean_C"])) > 1e-7):
            raise ValueError("shader node/data identity gate failed: " + item["name"])

    old_archive, old_report_path = old / "stocking-geometry-1gb.zip", old / "verification.json"
    previous = read_json(old_report_path)
    old_archive_hash = sha256_file(old_archive)
    old_archive_bytes, old_report_bytes = old_archive.stat().st_size, old_report_path.stat().st_size
    if (previous.get("readback_all_members_passed") is not True or
            previous.get("archive_sha256") != old_archive_hash or
            previous.get("archive_bytes") != old_archive_bytes or
            previous.get("total_export_bytes") != old_archive_bytes + old_report_bytes):
        raise ValueError("the preserved stocking export differs from its verification")
    old_identity = {"archive_source": str(old_archive), "report_source": str(old_report_path),
                    "archive_bytes": old_archive_bytes, "verification_bytes": old_report_bytes,
                    "archive_sha256": old_archive_hash, "verification_sha256": sha256_file(old_report_path),
                    "total_bytes": old_archive_bytes + old_report_bytes}
    files = []

    def add(source, member, expected_hash=None):
        source = Path(source).resolve()
        relative = PurePosixPath(member)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("package member must be a relative path")
        member = str(relative)
        if any(item["member"] == member for item in files):
            raise ValueError("duplicate package member: " + member)
        digest = sha256_file(source)
        if expected_hash is not None and digest != expected_hash:
            raise ValueError("source identity changed: " + str(source))
        files.append({"source": str(source), "member": member,
                      "bytes": source.stat().st_size, "sha256": digest})

    # Retain the failed parent full-field diagnostics inside the final asset.
    # They document finite-support limits; they are not replaced by passing
    # cropped results and do not turn this into full-garment coverage.
    for path in sorted(assets.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(assets)
        if _skip_asset(relative):
            continue
        if not path.resolve().is_relative_to(assets):
            raise ValueError("asset symlink resolves outside its directory")
        add(path, "stocking/assets/" + assets.name + "/" + relative.as_posix())
    add(CODE / "YARN_COVERAGE_METHOD.md", "docs/YARN_COVERAGE_METHOD.md")
    for name in ("run_yarn_coverage.py", "run_yarn_coverage_export.py"):
        add(CODE / name, "stocking/code/" + name)
    for name in MODULES:
        add(CODE / "stocking_material" / name, "stocking/code/stocking_material/" + name)
    for name in AUXILIARY_SOURCES:
        add(ROOT / "测试代码" / name, "测试代码/" + name)
    for path, name in ((core_report, "core-verification.json"),
                       (actual_report, "actual-coverage-verification.json"),
                       (shader_report, "coverage-shader-verification.json")):
        add(path, "checks/" + name)
    for name in ("coverage-comparison.png", "footprint-comparison.png"):
        add(previews / name, "preview/" + name)
    pair_path = Path(manifest["pair_path"]).resolve()
    if sha256_file(pair_path) != manifest["pair_sha256"]:
        raise ValueError("frozen mesh pair changed")
    if include_pair:
        add(pair_path, "stocking/assets/authored-fit-v1/left/garment_pair.npz", manifest["pair_sha256"])
        add(pair_path.with_suffix(".json"), "stocking/assets/authored-fit-v1/left/garment_pair.json")
    source_bytes = sum(item["bytes"] for item in files)
    upper_bound = source_bytes + sum(1024 + 2 * len(item["member"].encode("utf-8")) for item in files) + (
        INVENTORY_RESERVE + REPORT_RESERVE)
    combined_bound = old_identity["total_bytes"] + upper_bound
    if combined_bound > maximum_bytes:
        raise ValueError(f"preserved export plus supplement exceeds cap: {combined_bound:,} > {maximum_bytes:,}")
    return {"schema": "stocking.local_yarn_coverage_export.v1", "maximum_combined_bytes": maximum_bytes,
            "asset_version": assets.name, "source_manifest_sha256": manifest_hash,
            "source_asset_bytes": source_bytes, "conservative_supplement_bytes": upper_bound,
            "conservative_combined_bytes": combined_bound,
            "preserved_stocking_export": old_identity, "files": files,
            "scope": manifest["prototype_scope"], "case_names": [case["name"] for case in cases],
            "all_cases_generated": True, "local_expanded_window_guards_passed": True,
            "actual_geometry_qa_passed": True, "shader_qa_passed": True,
            "full_stocking_coverage": False, "global_source_completeness_certified": False,
            "real_skin_geometry_used": False, "fiber_transmission_modeled": False,
            "frozen_pair_included": bool(include_pair),
            "archive_layout": "workspace-relative assets and code; source metadata absolute paths are provenance",
            "code_snapshot_scope": "coverage generation, analytic queries and QA sources with their local Python dependencies; relocation of every entry point is not certified",
            "runtime_requirements": ["Python with NumPy", "Blender for preview rerender", "Pillow for contact sheets"]}


def _old_export_unchanged(plan):
    old = plan["preserved_stocking_export"]
    for source_key, bytes_key, hash_key in (("archive_source", "archive_bytes", "archive_sha256"),
                                           ("report_source", "verification_bytes", "verification_sha256")):
        path = Path(old[source_key])
        if path.stat().st_size != old[bytes_key] or sha256_file(path) != old[hash_key]:
            raise ValueError("preserved old export changed during supplemental packaging")


def write_coverage_export(plan, output_directory):
    """Read back every ZIP member and count all four actual delivery files."""
    output = Path(output_directory).resolve()
    archive = output / "yarn-coverage-supplement.zip"
    report_path = output / "verification.json"
    temporary = output / "yarn-coverage-supplement.zip.partial"
    report_temporary = output / "verification.json.partial"
    if any(path.exists() for path in (archive, report_path, temporary, report_temporary)):
        raise ValueError("supplement output already exists; select a new version directory")
    _old_export_unchanged(plan)
    budget = plan["maximum_combined_bytes"]
    old_bytes = plan["preserved_stocking_export"]["total_bytes"]
    inventory = {key: value for key, value in plan.items() if key != "files"}
    inventory["preserved_stocking_export"] = {key: value for key, value in plan["preserved_stocking_export"].items()
                                             if not key.endswith("_source")}
    inventory["files"] = [{key: value for key, value in item.items() if key != "source"} for item in plan["files"]]
    payload = json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if len(payload) > INVENTORY_RESERVE:
        raise ValueError("inventory exceeds reserved storage")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as writer:
        writer.writestr("export.json", payload)
        for item in plan["files"]:
            source = Path(item["source"])
            if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
                raise ValueError("input changed after inventory: " + str(source))
            writer.write(source, item["member"])
            if old_bytes + temporary.stat().st_size + REPORT_RESERVE > budget:
                raise ValueError("actual shared export exceeded byte cap while writing")
    archive_bytes = temporary.stat().st_size
    if old_bytes + archive_bytes + REPORT_RESERVE > budget:
        raise ValueError("actual shared export exceeded byte cap after ZIP finalization")
    expected = {item["member"]: item for item in plan["files"]}
    with zipfile.ZipFile(temporary) as reader:
        if set(reader.namelist()) != set(expected) | {"export.json"} or len(reader.namelist()) != len(expected) + 1:
            raise ValueError("ZIP has unexpected or duplicate members")
        if reader.read("export.json") != payload:
            raise ValueError("ZIP inventory readback differs")
        for name, item in expected.items():
            digest = hashlib.sha256()
            size = 0
            with reader.open(name) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("ZIP entry readback mismatch: " + name)
    _old_export_unchanged(plan)
    report = {"schema": "stocking.local_yarn_coverage_export_verification.v1",
              "archive": archive.name, "archive_bytes": archive_bytes,
              "archive_sha256": sha256_file(temporary), "source_manifest_sha256": plan["source_manifest_sha256"],
              "file_count": len(plan["files"]), "zip_member_count": len(plan["files"]) + 1,
              "readback_all_members_passed": True, "inventory_readback_passed": True,
              "preserved_old_export_unchanged": True,
              "preserved_old_export": inventory["preserved_stocking_export"],
              "maximum_combined_bytes": budget, "supplement_export_bytes": 0,
              "combined_export_bytes": 0, "within_combined_byte_cap": True,
              "full_stocking_coverage": False, "actual_geometry_qa_passed": True, "shader_qa_passed": True}
    for _ in range(8):
        report_payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        supplement_bytes = archive_bytes + len(report_payload)
        actual_combined = old_bytes + supplement_bytes
        if (supplement_bytes == report["supplement_export_bytes"] and
                actual_combined == report["combined_export_bytes"]):
            break
        report["supplement_export_bytes"] = supplement_bytes
        report["combined_export_bytes"] = actual_combined
        report["within_combined_byte_cap"] = actual_combined <= budget
    else:
        raise ValueError("verification report bytes did not reach a fixed point")
    if not report["within_combined_byte_cap"] or len(report_payload) > REPORT_RESERVE:
        raise ValueError("actual four-file delivery exceeded shared byte cap")
    report_temporary.write_bytes(report_payload)
    # Publish only after both files and every readback check are ready.  An
    # unsuccessful run retains its own .partial files for inspection; it
    # never removes assets, previous exports, or backups.
    temporary.replace(archive)
    report_temporary.replace(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "stocking/assets/yarn-coverage-v5")
    parser.add_argument("--old-export", type=Path, default=ROOT / "stocking/assets/authored-export-1gb-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "stocking/assets/yarn-coverage-export-v1")
    parser.add_argument("--actual-report", type=Path)
    parser.add_argument("--shader-report", type=Path)
    parser.add_argument("--core-report", type=Path,
                        default=ROOT / "build-support/stocking-material/yarn-coverage/core-verification.json")
    parser.add_argument("--preview-directory", type=Path)
    parser.add_argument("--maximum-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--omit-pair", action="store_true")
    parser.add_argument("--write", action="store_true")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    checks = ROOT / "build-support/stocking-material" / args.assets.name
    actual = args.actual_report or checks / "actual-coverage-verification.json"
    shader = args.shader_report or checks / "coverage-shader-verification.json"
    previews = args.preview_directory or ROOT / "render" / args.assets.name
    plan = plan_coverage_export(args.assets, args.old_export, actual_report=actual,
        shader_report=shader, core_report=args.core_report, preview_directory=previews,
        maximum_bytes=args.maximum_bytes, include_pair=not args.omit_pair)
    print("YARN_COVERAGE_EXPORT_PLAN", json.dumps({key: value for key, value in plan.items() if key != "files"},
                                                 ensure_ascii=True), flush=True)
    if args.write:
        report = write_coverage_export(plan, args.output)
        print("YARN_COVERAGE_EXPORT_RESULT", json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
