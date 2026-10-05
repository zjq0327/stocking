"""Generate one static stocking shell from one imported leg-and-foot mesh.

The input mesh is a geometric target, not an unworn stocking or a physical
dressing simulation.  This module keeps the procedural X0 topology and material
coordinates while producing a fitted X1 in the supplied static pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import numpy as np

from .asset_io import save_garment_pair
from .procedural_garment import ProceduralGarmentParameters, generate_reference_garment
from .single_leg_target import analyze_single_leg_mesh
from .single_leg_fitting import fit_static_single_leg


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("Cannot determine an anatomical direction from the input mesh")
    return vector / length


def _span(points, axis):
    projection = np.asarray(points, dtype=np.float64) @ _unit(axis)
    return float(np.ptp(projection))


def _reference_parameters(positions, analysis):
    """Choose a declared synthetic X0 scale from the target's measured extent."""
    cuff = np.asarray(analysis["cuff_center_mm"], dtype=np.float64)
    ankle = np.asarray(analysis["ankle_center_mm"], dtype=np.float64)
    toe = np.asarray(analysis["toe_point_mm"], dtype=np.float64)
    down = _unit(analysis["down_axis"])
    forward = _unit(analysis["foot_axis"])
    lateral = _unit(np.cross(down, forward))
    cuff_points = positions[np.asarray(analysis["cuff_loop"], dtype=np.int64)]
    foot_points = positions[np.asarray(analysis["foot_vertex_indices"], dtype=np.int64)]
    centerline = np.asarray(analysis["leg_centerline_mm"], dtype=np.float64)
    leg_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    foot_reach = float((toe-ankle) @ forward)
    if not 150. <= leg_length <= 1600. or not 70. <= foot_reach <= 450.:
        raise ValueError(
            f"Implausible leg/foot dimensions ({leg_length:.1f}/{foot_reach:.1f} mm); "
            "check the millimetres-per-model-unit setting and cuff/toe inference"
        )
    cuff_width = max(_span(cuff_points, lateral), _span(cuff_points, forward))
    foot_width = _span(foot_points, lateral)
    if not 20. <= cuff_width <= 400. or not 20. <= foot_width <= 250.:
        raise ValueError("Cuff or foot width is implausible; check the model units and inferred axes")
    # X0 is intentionally a synthetic starting shape; its dimensions are not
    # treated as measurements of a real unworn stocking.
    leg_width = max(25., .85*cuff_width)
    ankle_width = max(20., min(.7*cuff_width, .75*foot_width))
    reference_foot_width = max(25., .9*foot_width)
    heel_radius = max(.61*max(ankle_width, reference_foot_width),
                      min(55., .11*leg_length))
    toe_length = min(45., max(18., .22*foot_reach))
    params = ProceduralGarmentParameters(
        leg_length_mm=max(leg_length+heel_radius, heel_radius+leg_width+10.),
        leg_width_mm=leg_width,
        ankle_width_mm=ankle_width,
        foot_length_mm=max(foot_reach+.5*reference_foot_width,
                           heel_radius+toe_length+.55*reference_foot_width),
        foot_width_mm=reference_foot_width,
        flattening=.16,
        heel_bend_radius_mm=heel_radius,
        toe_length_mm=toe_length,
        circumference_segments=64,
        leg_segments=96,
        heel_segments=24,
        foot_segments=32,
        toe_segments=16,
    )
    params.validate()
    return params


def _export_obj(path, pair):
    """Write the fitted shell in millimetres, with per-corner material UV."""
    positions = pair.current_positions_mm
    triangles = pair.triangles
    face_uv = pair.triangle_material_uv
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Stocking X1 geometric shell; XYZ coordinates in millimetres\n")
        for point in positions:
            stream.write("v {:.9f} {:.9f} {:.9f}\n".format(*point))
        for uv in face_uv.reshape(-1, 2):
            stream.write("vt {:.9f} {:.9f}\n".format(*uv))
        for index, triangle in enumerate(triangles):
            stream.write("f " + " ".join(
                f"{int(vertex)+1}/{3*index+corner+1}"
                for corner, vertex in enumerate(triangle)) + "\n")


def _make_preview_scene(positions_mm, triangles, pair):
    import bpy
    from .fitting_preview import _mesh_object, _plain_material, _shell_material

    scene = bpy.data.scenes.new("Stocking | Single leg fitted X1")
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.
    skin = _plain_material("Single leg | input surface", (.55, .36, .28), .78)
    shell = _shell_material()
    body = _mesh_object(scene, "Single leg | input", positions_mm, triangles, material=skin)
    stocking = _mesh_object(scene, "Stocking | fitted X1", pair.current_positions_mm,
                            pair.triangles, pair=pair, material=shell)
    body.hide_render = True
    body.hide_viewport = True
    stocking["stocking_role"] = "single_leg_static_X1"
    scene["stocking_generation"] = "single static leg geometric fitting"
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    return scene, stocking


def generate_single_leg_from_arrays(positions_mm, triangles, output_dir, *,
                                    clearance_mm=0.6, source_name="single_leg"):
    """Fit one procedural X0 and write an independently inspectable X1 asset.

    Coordinates are in millimetres. ``output_dir`` must not already contain an
    asset; work is staged beside it and only published after validation.
    """
    import bpy

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not np.isfinite(clearance_mm) or not 0. < clearance_mm <= 10.:
        raise ValueError("clearance_mm must be in (0, 10] mm")
    positions = np.asarray(positions_mm, dtype=np.float64)
    faces = np.asarray(triangles)
    analysis = analyze_single_leg_mesh(positions, faces)
    positions = np.asarray(analysis.get("positions_mm", positions), dtype=np.float64)
    faces = np.asarray(analysis.get("triangles", faces), dtype=np.int32)
    params = _reference_parameters(positions, analysis)
    reference = generate_reference_garment(params)
    pair, fit_report, contact = fit_static_single_leg(
        reference, positions, faces, analysis, clearance_mm=clearance_mm)
    pair.validate()

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = PROJECT_ROOT / "build-support" / "single-leg" / "builds" / uuid.uuid4().hex
    stage.mkdir(parents=True)
    save_garment_pair(reference, stage / "garment_reference")
    save_garment_pair(pair, stage / "garment_pair")
    np.savez_compressed(stage / "contact_samples.npz", **contact)
    np.savez_compressed(stage / "target.npz", positions_mm=positions, triangles=faces)
    _export_obj(stage / "stocking-x1.obj", pair)
    analysis_report = analysis.get("report", {})
    metadata = {
        "schema": "stocking.single_leg_fit.v1",
        "source_name": str(source_name),
        "length_unit": "mm",
        "target_input_sha256": hashlib.sha256(positions.tobytes()+faces.tobytes()).hexdigest(),
        "reference_parameters": params.to_dict(),
        "clearance_mm": float(clearance_mm),
        "target_analysis": analysis_report,
        "fitting": fit_report,
        "mesh_vertices": int(len(pair.current_positions_mm)),
        "mesh_triangles": int(len(pair.triangles)),
        "dressing_equilibrium_solved": False,
        "material_calibrated": False,
        "outputs": ["garment_reference.npz", "garment_reference.json",
                    "garment_pair.npz", "garment_pair.json", "contact_samples.npz",
                    "target.npz",
                    "stocking-x1.obj", "single-leg-fit.blend", "metadata.json"],
    }
    (stage / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    scene_script = PROJECT_ROOT / "stocking" / "code" / "run_single_leg_scene.py"
    command = [bpy.app.binary_path, "--background", "--factory-startup", "--threads", "4",
               "--python-exit-code", "1", "--python", str(scene_script), "--",
               "--asset-dir", str(stage)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                            text=True, timeout=300, check=False)
    if result.returncode != 0 or not (stage / "single-leg-fit.blend").is_file():
        raise RuntimeError("Isolated Blender scene export failed; build kept at "
                           f"{stage}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    prior_scene = bpy.context.window.scene if bpy.context.window is not None else None
    scene, mesh_object = _make_preview_scene(positions, faces, pair)
    publish = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        shutil.copytree(stage, publish)
        publish.rename(output)
    except Exception:
        if publish.exists() and publish.parent.resolve() == output.parent.resolve():
            shutil.rmtree(publish)
        if prior_scene is not None and bpy.context.window is not None:
            bpy.context.window.scene = prior_scene
        for obj in tuple(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(scene)
        raise
    if stage.parent.resolve() == (PROJECT_ROOT / "build-support" / "single-leg" / "builds").resolve():
        shutil.rmtree(stage, ignore_errors=True)
    paths = {name: output / name for name in metadata["outputs"]}
    return {"mesh_object": mesh_object, "scene": scene, "pair": pair,
            "report": metadata, "paths": paths}


def generate_single_leg_from_obj(filepath, output_dir, clearance_mm=0.6, *,
                                 mm_per_unit=1000.0):
    """Import a single OBJ mesh, then generate a unique asset under output_dir."""
    import bpy

    source = Path(filepath).resolve()
    if source.suffix.lower() != ".obj" or not source.is_file():
        raise ValueError("Select an existing OBJ file containing one leg-and-foot mesh")
    if not np.isfinite(mm_per_unit) or not 0.01 <= mm_per_unit <= 1_000_000.:
        raise ValueError("mm_per_unit must be finite and between 0.01 and 1,000,000")
    before = set(bpy.data.objects)
    try:
        # Treat OBJ vertex coordinates as the declared model coordinate frame.
        # Blender's default -Z/Y import basis silently rotates the input.
        bpy.ops.wm.obj_import(filepath=str(source), forward_axis="Y", up_axis="Z")
        imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
        if len(imported) != 1:
            raise ValueError(f"OBJ must import as exactly one mesh object; found {len(imported)}")
        obj = imported[0]
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            positions = np.asarray([tuple(evaluated.matrix_world @ vertex.co)
                                    for vertex in mesh.vertices], dtype=np.float64) * mm_per_unit
            triangles = np.asarray([tuple(face.vertices) for face in mesh.loop_triangles], dtype=np.int32)
        finally:
            evaluated.to_mesh_clear()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", source.stem).strip("-") or "leg"
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()[:8]
        base = Path(output_dir).resolve()
        candidate = base / f"{stem}-{fingerprint}"
        index = 2
        while candidate.exists():
            candidate = base / f"{stem}-{fingerprint}-{index}"
            index += 1
        return generate_single_leg_from_arrays(positions, triangles, candidate,
                                               clearance_mm=clearance_mm, source_name=source.name)
    except Exception:
        for obj in tuple(bpy.data.objects):
            if obj not in before:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise
