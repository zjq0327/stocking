"""Quasistatic yarn authoring: solve, rebuild, compare and export two states."""

from pathlib import Path
import json
import time
import traceback

import numpy as np


class DeformedParameters:
    """Use a solved sampling period while preserving the v1 authoring snapshot."""

    def __init__(self, params, period_mm):
        self._params = params
        self.period_mm = tuple(float(value) for value in period_mm)
        self.validate()

    def __getattr__(self, name):
        return getattr(self._params, name)

    def validate(self):
        self._params.validate()
        period = np.asarray(self.period_mm)
        if period.shape != (2,) or not np.isfinite(period).all() or np.any(period <= 0):
            raise ValueError("Deformed sampling period must contain two positive finite lengths")

    def to_dict(self):
        # parameters.json remains the initial authoring input. The actual period
        # is separately and explicitly recorded in metadata and centerlines.npz.
        return self._params.to_dict()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def _validate_solution(params, settings, solution):
    """Reject incomplete, failed, or mismatched cached solver results."""
    from .stretch_solver import initial_nodes, segment_ends, RodModel

    shape = (settings.nodes, 3)
    arrays = {}
    for name in ("initial_nodes_mm", "reference_nodes_mm", "nodes_mm"):
        data = np.asarray(getattr(solution, name), dtype=np.float64)
        if data.shape != shape or not np.isfinite(data).all():
            raise ValueError(f"Solver result {name} must be finite with shape {shape}")
        arrays[name] = data
    reference_period = np.asarray(solution.reference_period_mm, dtype=np.float64)
    final_period = np.asarray(solution.period_mm, dtype=np.float64)
    if reference_period.shape != (2,) or not np.allclose(reference_period, params.period_mm, rtol=1e-12, atol=0):
        raise ValueError("Cached solver reference period does not match the authoring parameters")
    expected_period = reference_period * [settings.lambda_x, settings.lambda_y]
    if final_period.shape != (2,) or not np.allclose(final_period, expected_period, rtol=1e-12, atol=0):
        raise ValueError("Cached solver final period does not match the requested stretches")
    expected_initial = initial_nodes(params, settings.nodes)
    if not np.allclose(arrays["initial_nodes_mm"], expected_initial, rtol=1e-12,
                       atol=max(reference_period) * 1e-12):
        raise ValueError("Cached solver initial nodes do not match the authoring parameters")
    rest = np.asarray(solution.rest_lengths_mm, dtype=np.float64)
    expected_rest = np.linalg.norm(segment_ends(arrays["initial_nodes_mm"], reference_period)
                                   - arrays["initial_nodes_mm"], axis=1)
    if rest.shape != (settings.nodes,) or not np.allclose(rest, expected_rest, rtol=1e-12, atol=0):
        raise ValueError("Cached solver rest lengths do not match its initial nodes")
    states = np.asarray(solution.states, dtype=np.float64)
    expected_count = settings.load_steps + 1
    if states.shape != (expected_count, *shape) or not np.isfinite(states).all():
        raise ValueError("Solver result must contain reference plus every prescribed load state")
    if len(solution.history) != expected_count:
        raise ValueError("Solver history and saved states have different lengths")
    if not np.array_equal(states[0], arrays["reference_nodes_mm"]) or not np.array_equal(states[-1], arrays["nodes_mm"]):
        raise ValueError("Saved endpoint states do not match reference/final solver nodes")
    periods = []
    model = RodModel(rest, params.R * params.scale_mm, settings)
    for index, stat in enumerate(solution.history):
        if not stat.get("converged", False):
            raise RuntimeError(f"Solver state {index} is not converged")
        residual, strain = float(stat["residual_force_N"]), float(stat["max_axial_strain"])
        if not np.isfinite([residual, strain]).all() or residual > settings.gradient_tolerance or strain > settings.max_strain:
            raise RuntimeError(f"Solver state {index} exceeds the residual or axial-strain tolerance")
        period = np.asarray(stat["period_mm"], dtype=np.float64)
        fraction = index / settings.load_steps
        expected = reference_period * (1.0 + fraction * (np.array([settings.lambda_x, settings.lambda_y]) - 1.0))
        if period.shape != (2,) or not np.allclose(period, expected, rtol=1e-12, atol=0):
            raise ValueError(f"Solver state {index} has an unexpected prescribed period")
        # Cached claims cannot stand in for forces under the requested material.
        energy, _, actual = model.energy_gradient(states[index], period, components=True)
        actual["energy_N_mm"] = energy
        for name, value in actual.items():
            recorded = float(stat.get(name, np.nan))
            if (not np.isfinite([value, recorded]).all()
                    or not np.isclose(recorded, value, rtol=1e-6, atol=1e-14)):
                raise ValueError(f"Cached state {index} history {name} does not match this material and geometry")
        if (actual["residual_force_N"] > settings.gradient_tolerance
                or actual["max_axial_strain"] > settings.max_strain):
            raise RuntimeError(f"Cached state {index} is not an equilibrium for this material")
        periods.append(period)
    arrays.update(rest_lengths_mm=rest, states=states, periods=np.asarray(periods),
                  reference_period_mm=reference_period, period_mm=final_period)
    return arrays


def _bake_state(params, settings, nodes, period, asset_dir, preview_dir, label, log):
    import bpy
    from .deformed_geometry import generate_deformed_geometry
    from .bake import bake_geometry
    from .export import export_features, verify_roundtrip
    from .validation import validate_mesh, validate_features, validate_periodicity

    sampling = DeformedParameters(params, period)
    log(f"Rebuilding and baking {label}: {params.width} x {params.height}")
    geometry = generate_deformed_geometry(nodes, period, params.R * params.scale_mm,
                                         params.tube_sides, tile=True)
    checks = {"mesh": validate_mesh(geometry)}
    features = bake_geometry(geometry, sampling, progress=log)
    checks["features"] = validate_features(features, sampling)
    checks["periodicity"] = validate_periodicity(geometry, sampling)
    paths = export_features(features, sampling, asset_dir, preview_dir, extra_metadata={
        "authoring_version": "stretch-v2",
        "authoring_state": label,
        "authoring_parameters_role": "parameters.json is the original Crane shape; sample coordinates use period_mm below",
        "initial_authoring_period_mm": list(params.period_mm),
        "actual_sampling_period_mm": list(period),
        "stretch_parameters": settings.to_dict(),
        "surface_reconstruction": "Circular rings at solver nodes; discrete parallel transport with periodic holonomy correction",
        "radius_model": "constant circular radius; no Poisson thinning",
        "radius_mm": params.R * params.scale_mm,
        "physics_scope": "Simplified isotropic rod equilibrium with axial, bending and periodic contact terms; no friction or twist DOF; demonstration coefficients",
        "blender_version": bpy.app.version_string,
    })
    checks["roundtrip"] = verify_roundtrip(features, asset_dir)
    if not checks["roundtrip"]["passed"]:
        raise RuntimeError(f"{label} exported data failed round-trip validation: {checks['roundtrip']['errors']}")
    return paths, checks


def build_stretched_asset(params, settings, asset_dir, preview_dir,
                          render_preview=False, save_blend=False, solution=None):
    """Export relaxed reference and final loaded states; raise on any failure.

    All internal load-step centerlines are saved for inspection, but only the
    two comparison states are baked. Passing ``solution`` avoids a second solve.
    The current .blend filename is unchanged, including when a copy is saved.
    """
    import bpy
    from .stretch_solver import solve_stretch
    from .deformed_geometry import generate_deformed_geometry
    from .scene import build_scene, render_preview as render_scene

    started = time.perf_counter()
    asset_dir, preview_dir = Path(asset_dir).resolve(), Path(preview_dir).resolve()
    report_dir = Path(__file__).resolve().parents[3] / "build-support" / "stocking-material"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (asset_dir.name + "-stretch-validation.json")
    result = {"passed": False, "paths": {}, "validation": {}, "report": str(report_path)}
    log = lambda message: print("[stocking/stretch] " + message, flush=True)
    try:
        params.validate()
        settings.validate()
        if solution is None:
            solution = solve_stretch(params, settings, progress=log)
        arrays = _validate_solution(params, settings, solution)
        asset_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Keep a converged solution even if a later mesh/export operation fails.
        # The validation report remains passed=False until the entire run ends.
        np.savez_compressed(asset_dir / "centerlines.npz", **arrays)
        _write_json(asset_dir / "stretch_parameters.json", settings.to_dict())
        _write_json(asset_dir / "solver_history.json", solution.history)

        reference_paths, reference_checks = _bake_state(
            params, settings, arrays["reference_nodes_mm"], arrays["reference_period_mm"],
            asset_dir / "reference", preview_dir / "reference", "relaxed_reference", log)
        final_paths, final_checks = _bake_state(
            params, settings, arrays["nodes_mm"], arrays["period_mm"],
            asset_dir, preview_dir, "final_loaded", log)
        result["paths"] = {**final_paths, "reference": reference_paths}
        result["validation"].update(reference=reference_checks, final=final_checks)

        centerlines_path = asset_dir / "centerlines.npz"
        np.savez_compressed(centerlines_path, **arrays)
        with np.load(centerlines_path, allow_pickle=False) as restored:
            exact = set(restored.files) == set(arrays) and all(np.array_equal(restored[key], value) for key, value in arrays.items())
        if not exact:
            raise RuntimeError("Centerline NPZ failed round-trip validation")
        result["validation"]["centerlines_roundtrip"] = {"passed": True, "state_count": len(arrays["states"])}
        _write_json(asset_dir / "stretch_parameters.json", settings.to_dict())
        _write_json(asset_dir / "solver_history.json", solution.history)
        result["paths"].update(centerlines=str(centerlines_path),
                               stretch_parameters=str(asset_dir / "stretch_parameters.json"),
                               solver_history=str(asset_dir / "solver_history.json"))

        preview_geometry = generate_deformed_geometry(
            arrays["nodes_mm"], arrays["period_mm"], params.R * params.scale_mm,
            params.tube_sides, rows=params.nRows, loops=params.nLoops)
        scene_info = build_scene(params, geometry=preview_geometry)
        scene = scene_info["scene"]
        scene["stocking_stretch_parameters"] = json.dumps(settings.to_dict(), ensure_ascii=False)
        scene["stocking_stretch_asset_dir"] = str(asset_dir)
        scene["stocking_stretch_preview_dir"] = str(preview_dir)
        scene["stocking_stretch_period_mm"] = list(arrays["period_mm"])
        scene["stocking_material_geometry_state"] = "final_loaded_stretch_v2"
        if save_blend:
            blend_path = asset_dir / "swatch.blend"
            preferences = bpy.context.preferences.filepaths
            old_preview_type = preferences.file_preview_type
            preferences.file_preview_type = "NONE"
            try:
                bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
            finally:
                preferences.file_preview_type = old_preview_type
            result["paths"]["blend"] = str(blend_path)
        if render_preview:
            result["paths"]["swatch_preview"] = str(render_scene(scene_info, preview_dir))
        result["comparison"] = {
            "reference_period_mm": arrays["reference_period_mm"].tolist(),
            "final_period_mm": arrays["period_mm"].tolist(),
            "reference_projected_hit_fraction": reference_checks["features"]["front"]["hit_fraction"],
            "final_projected_hit_fraction": final_checks["features"]["front"]["hit_fraction"],
            "interpretation": "Same pixel resolution and normalized tile coordinates; physical tile dimensions differ. Hit fraction is geometric coverage, not optical opacity.",
        }
        result["solver"] = {"states": len(solution.states), "final": _plain(solution.history[-1])}
        result["passed"] = True
        log(f"Completed reference/final export: {asset_dir}")
        return result
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _write_json(report_path, result)
