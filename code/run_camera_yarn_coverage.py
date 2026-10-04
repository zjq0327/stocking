"""Bake the original camera's geometric yarn C for body and toe shaders."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")
from stocking_material.camera_yarn_coverage import bake_camera_coverage, build_native_kernel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=ROOT / "stocking/assets/original-yarn-shader-v1/mapping-manifest.json")
    parser.add_argument("--masks", type=Path, default=ROOT / "build-support/yarn-shader-integration-v1/phase-mask-index.json")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = parser.parse_args()
    manifest = json.loads(args.mapping.read_text(encoding="utf8"))
    mask_index = json.loads(args.masks.read_text(encoding="utf8"))
    assert mask_index["bitorder"] == "little"
    assert manifest["period_mm"] == mask_index["period_mm"]
    masks = []
    for side in mask_index["sides"]:
        path = Path(side["file"])
        assert sha(path) == side["sha256"]
        masks.append(np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(side["shape"])))
    build_native_kernel()
    records = []
    for obj in manifest["objects"]:
        source_file = args.mapping.parent / obj["file"]
        assert sha(source_file) == obj["sha256"]
        with np.load(source_file) as f:
            source = {k: f[k] for k in f.files}
        levels, details = {}, {}
        for sample in args.levels:
            arrays, metadata = bake_camera_coverage(source, masks, manifest["world_to_clip"], manifest["width"], manifest["height"],
                period_mm=manifest["period_mm"], grid=mask_index["grid"], supersample=sample, workers=args.workers)
            levels[sample] = arrays
            details[str(sample)] = metadata
            print(json.dumps({"object": obj["name"], "supersample": sample, "elapsed_seconds": metadata["elapsed_seconds"],
                "mean_C": metadata["mean_C_over_mapped_subrays"], "mapped_pixels": metadata["mapped_pixel_count"]}), flush=True)
        final_level = max(args.levels)
        arrays = dict(levels[final_level])
        for sample in args.levels:
            if sample != final_level:
                arrays.update({k+"_"+str(sample): v for k, v in levels[sample].items()})
        output = args.mapping.parent / (obj["name"] + "-coverage.npz")
        np.savez_compressed(output, **arrays)
        convergence = {}
        for lower, upper in zip(args.levels[:-1], args.levels[1:]):
            selected = (levels[lower]["mapped_count"] > 0) & (levels[upper]["mapped_count"] > 0)
            difference = abs(levels[lower]["coverage"][selected] - levels[upper]["coverage"][selected])
            convergence[str(lower)+"_to_"+str(upper)] = {"scope": "pixels with mapped samples at both levels", "pixel_count": int(selected.sum()),
                "mean_absolute_difference": float(difference.mean()), "p95_absolute_difference": float(np.percentile(difference, 95)),
                "max_absolute_difference": float(difference.max()), "finite_sampling_error_certified": False}
        record = {"object": obj["name"], "file": output.name, "bytes": output.stat().st_size, "sha256": sha(output),
            "source_file": source_file.name, "source_sha256": obj["sha256"], "levels": details,
            "final_supersample_per_axis": final_level, "convergence": convergence}
        records.append(record)
        (args.mapping.parent / (obj["name"] + "-coverage.json")).write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
        print(json.dumps({"object_ready": obj["name"], "output": output.name, "convergence": convergence}), flush=True)
    final = {"schema": "stocking.original_camera_coverage_manifest.v1", "mapping_file": args.mapping.name,
        "mapping_sha256": sha(args.mapping), "phase_mask_index_sha256": sha(args.masks), "objects": records,
        "primary_camera_only": True, "local_affine_periodic_geometry_approximation": True,
        "exact_curved_whole_yarn_skin_raytrace": False, "fiber_transmission_solved": False}
    (args.mapping.parent / "camera-coverage-manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
