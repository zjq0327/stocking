"""Deterministic orthographic feature sampling of a periodic yarn surface.

All positions are in millimetres. Row zero is the lowest Y, on both sides.
This records geometric first hits; it does not evaluate a lighting shader.
"""
from __future__ import annotations

import numpy as np


def make_bvh(geometry):
    from mathutils.bvhtree import BVHTree
    # One explicit triangle per polygon keeps returned indices aligned with attrs.
    return BVHTree.FromPolygons(geometry.vertices.tolist(),
                               geometry.triangles.tolist(), all_triangles=True)


def sample_points(geometry, bvh, xy, side):
    """Sample arbitrary XY locations; return one coherent attribute record per ray."""
    if side not in {"front", "back"}:
        raise ValueError("side must be front or back")
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    count = len(xy)
    zlo, zhi = float(geometry.vertices[:, 2].min()), float(geometry.vertices[:, 2].max())
    # BVH intersections use float32. A fixed millimetre offset would erase
    # small Z differences when the user authors a microscopic-scale sample.
    margin = zhi - zlo
    if not margin > 0:
        raise ValueError("Geometry must have a nonzero Z extent")
    origin_z = zhi + margin if side == "front" else zlo - margin
    direction = (0.0, 0.0, -1.0 if side == "front" else 1.0)
    distance = zhi - zlo + 2.0 * margin
    indices = np.full(count, -1, dtype=np.int32)
    positions = np.zeros((count, 3), dtype=np.float64)
    for k, (x, y) in enumerate(xy):
        position, _face_normal, triangle_index, _distance = bvh.ray_cast(
            (float(x), float(y), origin_z), direction, distance)
        if triangle_index is not None:
            indices[k] = triangle_index
            positions[k] = position
    valid = indices >= 0
    result = {"id": np.zeros(count, dtype=np.uint32), "valid": valid,
              "P": positions.astype(np.float32),
              "N": np.zeros((count, 3), dtype=np.float32),
              "T": np.zeros((count, 3), dtype=np.float32)}
    if not valid.any():
        return result
    triangles = geometry.triangles[indices[valid]]
    corners = geometry.vertices[triangles]
    e0, e1 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    q = positions[valid] - corners[:, 0]
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    d00, d01, d11 = dot(e0, e0), dot(e0, e1), dot(e1, e1)
    d20, d21 = dot(q, e0), dot(q, e1)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= 0):
        raise ValueError("Degenerate triangle encountered during interpolation")
    w1 = (d11 * d20 - d01 * d21) / denominator
    w2 = (d00 * d21 - d01 * d20) / denominator
    weights = np.column_stack((1.0 - w1 - w2, w1, w2))
    # Float32 ray/triangle intersections can fall microscopically outside an edge.
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    normal = np.einsum("ij,ijk->ik", weights, geometry.normals[triangles])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tangent = np.einsum("ij,ijk->ik", weights, geometry.tangents[triangles])
    tangent -= dot(tangent, normal)[:, None] * normal
    lengths = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(lengths < 1e-10):
        raise ValueError("Invalid longitudinal tangent after interpolation")
    tangent /= lengths
    result["id"][valid] = geometry.yarn_ids[indices[valid]]
    result["N"][valid] = normal.astype(np.float32)
    result["T"][valid] = tangent.astype(np.float32)
    return result


def bake_geometry(geometry, params, progress=None):
    """Return front/back id, valid, P, N, T arrays, indexed [y, x, component]."""
    width, height = params.width, params.height
    lx, ly = params.period_mm
    x = (np.arange(width, dtype=np.float64) + 0.5) * lx / width
    y = (np.arange(height, dtype=np.float64) + 0.5) * ly / height
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    bvh = make_bvh(geometry)
    features = {}
    for side in ("front", "back"):
        if progress:
            progress(f"Sampling {side}: {width} x {height}")
        samples = sample_points(geometry, bvh, xy, side)
        features[side] = {
            key: value.reshape((height, width) + (() if value.ndim == 1 else (3,)))
            for key, value in samples.items()}
    return features
