"""Fixed-camera pixel filtering of the locally affine periodic yarn union.

This is a camera bake, not a UV texture valid for every camera. A macro mesh
selects the frontmost original object triangle. Perspective-correct material
coordinates then address a binary phase mask computed from the 3-D yarn.
The phase mask uses a constant face Jacobian/normal/view and finite 128-square
phase sampling. Curved-garment yarn self-occlusion and real skin rays are not
solved here. Images are row-major, top-down, linear numerical data.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np

_LIBRARIES = {}


def native_paths():
    source = Path(__file__).with_suffix(".cpp")
    identity = hashlib.sha256(source.read_bytes() + b"camera-yarn-v1-msvc-O2-strict").hexdigest()[:20]
    directory = source.parents[3] / "build-support/stocking-material/camera-coverage-native" / identity
    return source, directory, directory / "camera_yarn_coverage.dll"


def build_native_kernel():
    source, directory, library = native_paths()
    if library.is_file():
        return library
    root = Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC")
    versions = sorted((p for p in root.glob("*") if (p / "bin/Hostx64/x64/cl.exe").is_file()),
                      key=lambda p: tuple(map(int, p.name.split("."))))
    if not versions:
        raise RuntimeError("No existing MSVC compiler found")
    msvc = versions[-1]
    kits = Path("C:/Program Files (x86)/Windows Kits/10")
    sdks = sorted((p for p in (kits / "Include").glob("*") if (p / "ucrt/stdlib.h").is_file()),
                  key=lambda p: tuple(map(int, p.name.split("."))))
    if not sdks:
        raise RuntimeError("No existing Windows SDK found")
    sdk = sdks[-1]
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(msvc / "bin/Hostx64/x64/cl.exe"), "/nologo", "/LD", "/O2", "/fp:strict", "/EHsc", "/std:c++17", "/MT",
               "/I" + str(msvc / "include"), "/I" + str(sdk / "ucrt"), "/I" + str(sdk / "shared"), "/I" + str(sdk / "um"),
               str(source), "/Fo" + str(directory / "camera_yarn_coverage.obj"),
               "/Fd" + str(directory / "camera_yarn_coverage.pdb"), "/link", "/OUT:" + str(library),
               "/LIBPATH:" + str(msvc / "lib/x64"), "/LIBPATH:" + str(kits / "Lib" / sdk.name / "ucrt/x64"),
               "/LIBPATH:" + str(kits / "Lib" / sdk.name / "um/x64")]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, encoding="utf8", errors="replace")
    (directory / "build.log").write_text(result.stdout + result.stderr, encoding="utf8")
    if result.returncode or not library.is_file():
        raise RuntimeError("Native camera coverage build failed: " + str(directory / "build.log"))
    return library


def _load():
    _, _, path = native_paths()
    if not path.is_file():
        raise RuntimeError("Explicitly call build_native_kernel before camera bake")
    key = str(path)
    if key not in _LIBRARIES:
        lib = ctypes.CDLL(key)
        d = ctypes.POINTER(ctypes.c_double)
        i = ctypes.POINTER(ctypes.c_int)
        b = ctypes.POINTER(ctypes.c_ubyte)
        u = ctypes.POINTER(ctypes.c_ushort)
        lib.stocking_camera_coverage.argtypes = [ctypes.c_int, d, d, i, i, b, b, b,
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, u, u, u, d]
        lib.stocking_camera_coverage.restype = ctypes.c_int
        _LIBRARIES[key] = lib
    return _LIBRARIES[key]


def project_vertices(positions_world_mm, world_to_clip, width, height):
    """Project world positions using the supplied Blender clip matrix."""
    p = np.asarray(positions_world_mm, dtype=np.float64)
    matrix = np.asarray(world_to_clip, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Expected world vertices[N,3] and world_to_clip[4,4]")
    if not np.isfinite(p).all() or not np.isfinite(matrix).all():
        raise ValueError("Nonfinite projection data")
    return np.column_stack((p, np.ones(len(p)))) @ matrix.T


def _clip_z_polygon(clip, q, sign):
    """Clip homogeneous polygon against z+w or w-z, carrying affine q."""
    signed = clip[:, 3] + sign * clip[:, 2]
    output, values = [], []
    for i in range(len(clip)):
        j = (i - 1) % len(clip)
        inside_i, inside_j = signed[i] >= 0, signed[j] >= 0
        if inside_i != inside_j:
            t = signed[j] / (signed[j] - signed[i])
            output.append(clip[j] + t * (clip[i] - clip[j]))
            values.append(q[j] + t * (q[i] - q[j]))
        if inside_i:
            output.append(clip[i]); values.append(q[i])
    return np.asarray(output).reshape((-1, 4)), np.asarray(values).reshape((-1, 2))


def prepare_triangles(source, world_to_clip, width, height):
    """Generate clipped screen triangles and their material attributes."""
    triangles = np.asarray(source["triangles"], dtype=np.int32)
    q = np.asarray(source["triangle_q_mm"], dtype=np.float64)
    positions = np.asarray(source["positions_world_mm"], dtype=np.float64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or q.shape != (len(triangles), 3, 2):
        raise ValueError("Invalid source triangles or material q")
    if triangles.min() < 0 or triangles.max() >= len(positions) or not np.isfinite(q).all():
        raise ValueError("Invalid source indices or nonfinite q")
    clip = project_vertices(positions, world_to_clip, width, height)[triangles]
    planes = np.stack((clip[:, :, 3] + clip[:, :, 2], clip[:, :, 3] - clip[:, :, 2]), axis=2)
    inside = np.all(planes >= 0, axis=(1, 2)) & np.all(clip[:, :, 3] > 0, axis=1)
    rejected = np.any(np.all(planes < 0, axis=1), axis=1)
    ids = list(np.flatnonzero(inside))
    clips = [clip[inside]]
    qs = [q[inside]]
    for f in np.flatnonzero(~inside & ~rejected):
        polygon, values = _clip_z_polygon(clip[f], q[f], 1)
        if len(polygon) < 3:
            continue
        polygon, values = _clip_z_polygon(polygon, values, -1)
        for j in range(1, len(polygon)-1):
            item = polygon[[0, j, j+1]]
            if np.any(item[:, 3] <= 0):
                continue
            clips.append(item[None]); qs.append(values[[0, j, j+1]][None]); ids.append(int(f))
    if not ids:
        raise ValueError("Source has no triangles in camera clip depth range")
    clip = np.concatenate(clips)
    q = np.ascontiguousarray(np.concatenate(qs))
    ids = np.asarray(ids, dtype=np.int32)
    inverse_w = 1 / clip[:, :, 3]
    ndc = clip[:, :, :3] * inverse_w[:, :, None]
    screen = np.empty(clip.shape, dtype=np.float64)
    screen[:, :, 0] = (ndc[:, :, 0] + 1) * (width / 2)
    screen[:, :, 1] = (1 - ndc[:, :, 1]) * (height / 2)
    screen[:, :, 2] = ndc[:, :, 2]
    screen[:, :, 3] = inverse_w
    return {"screen": np.ascontiguousarray(screen), "q": q,
        "face": np.ascontiguousarray(np.asarray(source["triangle_material_face_index"], dtype=np.int32)[ids]),
        "sock": np.ascontiguousarray(np.asarray(source["triangle_sock_index"], dtype=np.int32)[ids]),
        "mapped": np.ascontiguousarray(np.asarray(source["triangle_mapped"], dtype=np.uint8)[ids]),
        "source_triangle_index": ids, "source_triangle_count": len(triangles),
        "clipped_source_triangle_count": int(np.count_nonzero(~inside & ~rejected)),
        "rejected_source_triangle_count": int(np.count_nonzero(rejected))}


def bake_camera_coverage(source, masks, world_to_clip, width, height, *,
                         period_mm=(0.3141592653589793, 0.225), grid=128,
                         supersample=4, workers=None):
    """Return conditional C and mapping fraction suitable for one shader mix.

    old/new shader mix = (1-mapped)*old_C + mapped*coverage. The coverage
    numerator is binary yarn hits among mapped subrays, and the denominator
    is mapped subrays, avoiding an extra mapped factor at the mapping edge.
    """
    if not all(isinstance(v, int) and v > 0 for v in (width, height, grid, supersample)):
        raise ValueError("Invalid image/mask resolution")
    # The C++ kernel retains only a 32-pixel tile per worker, never the full
    # supersampled frame. 32x sampling of a 1080x1920 camera is therefore
    # practical without storing two billion subray records.
    if supersample > 64 or width*height*supersample**2 > 3_000_000_000:
        raise ValueError("Camera raster sample budget exceeded")
    masks = tuple(np.asarray(m, dtype=np.uint8) for m in masks)
    if len(masks) != 2 or masks[0].shape != masks[1].shape or masks[0].ndim != 2 or masks[0].shape[1] != grid*grid//8:
        raise ValueError("Expected two packed masks[material_faces,grid²/8]")
    masks = tuple(np.ascontiguousarray(m) for m in masks)
    prepared = prepare_triangles(source, world_to_clip, width, height)
    f = len(prepared["face"])
    mapped = prepared["mapped"] > 0
    if np.any((prepared["face"][mapped] < 0) | (prepared["face"][mapped] >= len(masks[0]))) or np.any((prepared["sock"][mapped] < 0) | (prepared["sock"][mapped] > 1)):
        raise ValueError("Mapped triangle refers to absent material face or sock")
    hits = np.zeros((height, width), dtype=np.uint16)
    mapped_counts = np.zeros_like(hits)
    surface_counts = np.zeros_like(hits)
    stats = np.zeros(4, dtype=np.float64)
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    workers = min(16, os.cpu_count() or 1) if workers is None else int(workers)
    if not 1 <= workers <= 64:
        raise ValueError("Invalid worker count")
    started = time.perf_counter()
    result = _load().stocking_camera_coverage(f, dp(prepared["screen"]), dp(prepared["q"]),
        ip(prepared["face"]), ip(prepared["sock"]), bp(prepared["mapped"]), bp(masks[0]), bp(masks[1]),
        len(masks[0]), grid, *period_mm, width, height, supersample, workers,
        up(hits), up(mapped_counts), up(surface_counts), dp(stats))
    if not result:
        raise RuntimeError("Native camera raster rejected inputs")
    denominator = np.float32(supersample**2)
    conditional = np.divide(hits, mapped_counts, out=np.zeros_like(hits, dtype=np.float32), where=mapped_counts > 0)
    arrays = {"coverage": conditional, "mapped": mapped_counts.astype(np.float32)/denominator,
        "coverage_unconditional": hits.astype(np.float32)/denominator,
        "hit_count": hits, "mapped_count": mapped_counts, "surface_count": surface_counts,
        "surface": surface_counts.astype(np.float32)/denominator}
    metadata = {"schema": "stocking.camera_yarn_coverage.v1", "width": width, "height": height,
        "supersample_per_axis": supersample, "samples_per_pixel": supersample**2,
        "phase_grid_per_axis": grid, "phase_mask_bitorder": "little",
        "phase_mask_flattening": "y_wale*grid+x_course", "period_mm": list(period_mm),
        "world_to_clip": np.asarray(world_to_clip).tolist(), "image_row_direction": "top-down",
        "conditional_C_definition": "binary_yarn_hit_count/mapped_subray_count (zero if unmapped)",
        "mapped_definition": "mapped_subray_count/samples_per_pixel",
        "shader_expression": "(1-mapped)*old_C+mapped*coverage",
        "source_triangle_count": prepared["source_triangle_count"], "raster_triangle_count": f,
        "clipped_source_triangle_count": prepared["clipped_source_triangle_count"],
        "rejected_source_triangle_count": prepared["rejected_source_triangle_count"],
        "mapped_subray_count": int(mapped_counts.sum()), "yarn_hit_subray_count": int(hits.sum()),
        "surface_subray_count": int(surface_counts.sum()),
        "mapped_pixel_count": int(np.count_nonzero(mapped_counts)),
        "mean_C_over_mapped_subrays": float(hits.sum()/mapped_counts.sum()) if mapped_counts.sum() else 0,
        "elapsed_seconds": time.perf_counter()-started,
        "triangle_sample_tests": int(stats[0]), "nearest_surface_updates": int(stats[1]),
        "phase_mask_lookups": int(stats[2]), "tiles": int(stats[3]),
        "native_kernel_identity": native_paths()[1].name,
        "deformation_applied_once": True, "camera_direction_scope": "fixed original primary camera",
        "approximation": "nearest macro triangle, perspective-correct material q, per-material-face local affine periodic projected yarn phase mask",
        "curved_whole_garment_exact": False, "real_skin_occlusion_used": False,
        "fiber_transmission_solved": False, "mechanical_equilibrium_solved": False}
    return arrays, metadata
