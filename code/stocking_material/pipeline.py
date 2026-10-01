"""One shared authoring pipeline for the batch entry point and Blender panel."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .geometry import generate_tile_geometry
from .bake import bake_geometry
from .validation import validate_mesh, validate_features, validate_periodicity, validate_clearance


def build_asset(params, asset_dir, preview_dir, render_preview=False, save_blend=False):
    import bpy
    from .export import export_features, verify_roundtrip
    from .scene import build_scene, render_preview as render_scene
    start = time.perf_counter()
    params.validate()
    asset_dir, preview_dir = Path(asset_dir).resolve(), Path(preview_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    log = lambda message: print("[stocking] " + message, flush=True)
    log("Generating periodic tube geometry")
    geometry = generate_tile_geometry(params)
    checks = {"mesh": validate_mesh(geometry), "clearance": validate_clearance(params)}
    features = bake_geometry(geometry, params, progress=log)
    checks["features"] = validate_features(features, params)
    checks["periodicity"] = validate_periodicity(geometry, params)
    log("Exporting numerical features and previews")
    paths = export_features(features, params, asset_dir, preview_dir,
                            extra_metadata={
                                "blender_version": bpy.app.version_string,
                                "geometry_reference": {"author": "Keenan Crane", "year": 2023,
                                    "title": "A Simple Parametric Model of Plain-Knit Yarns",
                                    "url": "https://github.com/keenancrane/plain-knit-yarn",
                                    "code_license": "MIT"},
                                "yarn_types": {"0": "empty", "1": "single circular tube"},
                                "sampling": {"pattern": "pixel center", "samples_per_pixel": 1,
                                             "pixel_center_xy_mm": "((i+0.5)*Lx/W, (j+0.5)*Ly/H)"}})
    checks["roundtrip"] = verify_roundtrip(features, asset_dir)
    if not checks["roundtrip"]["passed"]:
        raise RuntimeError("Feature export round-trip validation failed: " +
                           "; ".join(checks["roundtrip"]["errors"]))
    scene_info = build_scene(params)
    if save_blend:
        # Save a copy; an existing interactive user's current document stays current.
        blend_path = asset_dir / "swatch.blend"
        # Skip OS thumbnail cache generation; keep generated previews in render/.
        preferences = bpy.context.preferences.filepaths
        old_preview_type = preferences.file_preview_type
        preferences.file_preview_type = "NONE"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
        finally:
            preferences.file_preview_type = old_preview_type
        paths["blend"] = str(blend_path)
    if render_preview:
        log("Rendering 3D swatch preview")
        paths["swatch_preview"] = str(render_scene(scene_info, preview_dir))
    result = {"paths": paths, "validation": checks,
              "elapsed_seconds": round(time.perf_counter() - start, 3)}
    root = Path(__file__).resolve().parents[3]
    report_dir = root / "build-support" / "stocking-material"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / (asset_dir.name + "-validation.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Completed in {result['elapsed_seconds']} s: {asset_dir}")
    return result
