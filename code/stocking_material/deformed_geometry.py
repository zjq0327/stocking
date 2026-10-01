"""Rebuild periodic circular yarn tubes directly from deformed centerline nodes.

Coordinates, periods and radius are all millimeters. The input contains one
period without its repeated endpoint. Its last edge ends at nodes[0]+(Lx,0,0).
Discrete parallel transport supports zero curvature; a distributed rotation
removes the frame holonomy so repeating periods have identical cross sections.
This module checks local tube regularity, not global contact or equilibrium.
"""

from numbers import Integral, Real
import math
import sys

import numpy as np

from .geometry import Geometry


def _positive_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _transport(normal, tangent_before, tangent_after):
    """Shortest rotation between two unit tangents, followed by cleanup."""
    cosine = float(np.clip(np.dot(tangent_before, tangent_after), -1.0, 1.0))
    if cosine <= -1.0 + 1e-10:
        raise ValueError("Adjacent node tangents reverse direction; refine or repair the centerline")
    axis_sine = np.cross(tangent_before, tangent_after)
    first_cross = np.cross(axis_sine, normal)
    rotated = normal + first_cross + np.cross(axis_sine, first_cross) / (1.0 + cosine)
    rotated -= np.dot(rotated, tangent_after) * tangent_after
    length = np.linalg.norm(rotated)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Cannot construct a stable frame for this centerline")
    return rotated / length


def _periodic_bishop_frame(nodes, lx, radius):
    """Return tangent, normal-axis, binormal-axis for the unrepeated nodes."""
    endpoint = nodes[0] + np.array([lx, 0.0, 0.0])
    edges = np.diff(np.vstack((nodes, endpoint)), axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    numeric_scale = max(lx, radius, float(np.max(np.abs(nodes))))
    if (not np.isfinite(lengths).all()
            or np.any(lengths <= max(numeric_scale * sys.float_info.epsilon * 64, sys.float_info.min))):
        raise ValueError("Centerline has repeated or numerically indistinguishable adjacent nodes")
    edge_tangents = edges / lengths[:, None]
    previous = np.roll(edge_tangents, 1, axis=0)
    cosine = np.clip(np.einsum("ij,ij->i", previous, edge_tangents), -1.0, 1.0)
    if np.any(cosine <= -1.0 + 1e-10):
        raise ValueError("Centerline contains a reversing cusp; a circular tube is undefined there")
    tangent = previous + edge_tangents
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    # Turning angle divided by dual edge length, using tan(theta/2), is a
    # conservative discrete local-radius test. Zero curvature remains valid.
    dual_lengths = 0.5 * (lengths + np.roll(lengths, 1))
    curvature = 2.0 * np.sqrt(np.maximum(0.0, (1.0 - cosine) / (1.0 + cosine))) / dual_lengths
    radius_curvature = radius * curvature
    if not np.isfinite(radius_curvature).all() or np.any(radius_curvature >= 1.0 - 1e-6):
        raise ValueError("Tube radius is too large for the deformed centerline curvature "
                         f"(max R*kappa={float(np.max(radius_curvature)):.6g}); reduce radius or repair the shape")

    reference = np.eye(3)[int(np.argmin(np.abs(tangent[0])))]
    initial = reference - np.dot(reference, tangent[0]) * tangent[0]
    initial /= np.linalg.norm(initial)
    raw = np.empty((len(nodes) + 1, 3), dtype=np.float64)
    raw[0] = initial
    for i in range(1, len(nodes) + 1):
        raw[i] = _transport(raw[i - 1], tangent[i - 1], tangent[i % len(nodes)])
    correction = math.atan2(float(np.dot(tangent[0], np.cross(raw[-1], raw[0]))),
                            float(np.dot(raw[-1], raw[0])))
    arclength = np.r_[0.0, np.cumsum(lengths)]
    angles = correction * arclength[:-1] / arclength[-1]
    u = (raw[:-1] * np.cos(angles)[:, None]
         + np.cross(tangent, raw[:-1]) * np.sin(angles)[:, None])
    u -= np.einsum("ij,ij->i", u, tangent)[:, None] * tangent
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return tangent, u, v


def generate_deformed_geometry(nodes_mm, period_mm, radius_mm, tube_sides,
                               rows=5, loops=5, tile=False) -> Geometry:
    """Generate a continuous periodic tube mesh with unit N/T vertex attributes.

    ``tile=True`` covers the closed rectangle [0,Lx] x [0,Ly], including seam
    probes. Neighbor counts derive from the actual centerline bounds, including
    the periodic last edge. ``rows``/``loops`` only size the finite preview.
    The mesh has no end caps; preview row ends are deliberately open.
    """
    try:
        nodes = np.array(nodes_mm, dtype=np.float64, copy=True)
        period = np.asarray(period_mm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nodes_mm and period_mm must be numeric arrays") from exc
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes_mm must be a finite (N,3) array with N >= 3, without a repeated endpoint")
    if period.shape != (2,) or not np.isfinite(period).all() or np.any(period <= 0):
        raise ValueError("period_mm must contain two finite positive lengths (Lx,Ly)")
    lx, ly = map(float, period)
    radius = _positive_number(radius_mm, "radius_mm")
    tube_sides = _integer(tube_sides, "tube_sides", 6)
    rows, loops = _integer(rows, "rows", 1), _integer(loops, "loops", 1)
    if not isinstance(tile, (bool, np.bool_)):
        raise ValueError("tile must be a boolean")
    tangent, u, v = _periodic_bishop_frame(nodes, lx, radius)
    phi = np.arange(tube_sides) * (2.0 * math.pi / tube_sides)
    normals_one = (u[:, None, :] * np.cos(phi)[None, :, None]
                   + v[:, None, :] * np.sin(phi)[None, :, None])
    normals_one /= np.linalg.norm(normals_one, axis=2, keepdims=True)
    vertices_one = nodes[:, None, :] + radius * normals_one
    tangent_one = np.broadcast_to(tangent[:, None, :], normals_one.shape)

    if tile:
        # Include the implicit final node: its last-edge triangles can extend
        # beyond the bounds of the supplied node array.
        complete_nodes = np.vstack((nodes, nodes[0] + np.array([lx, 0.0, 0.0])))
        low, high = complete_nodes.min(axis=0) - radius, complete_nodes.max(axis=0) + radius
        first_loop = math.ceil(np.nextafter(-high[0] / lx, -np.inf)) - 1
        last_loop = math.floor(np.nextafter((lx - low[0]) / lx, np.inf)) + 1
        first_row = math.ceil(np.nextafter(-high[1] / ly, -np.inf))
        last_row = math.floor(np.nextafter((ly - low[1]) / ly, np.inf))
    else:
        first_loop, last_loop = 0, loops - 1
        first_row, last_row = 0, rows - 1
    loop_count, row_count = last_loop - first_loop + 1, last_row - first_row + 1
    ring_count = loop_count * len(nodes) + 1
    points_per_row = ring_count * tube_sides
    if row_count <= 0 or row_count * points_per_row > np.iinfo(np.int32).max:
        raise ValueError("Requested neighbor geometry exceeds the supported vertex-index range")

    shifts = np.zeros((loop_count, 1, 1, 3), dtype=np.float64)
    shifts[:, 0, 0, 0] = np.arange(first_loop, last_loop + 1, dtype=np.float64) * lx
    ring_vertices = (vertices_one[None, :, :, :] + shifts).reshape(-1, tube_sides, 3)
    endpoint_ring = vertices_one[0] + np.array([(last_loop + 1) * lx, 0.0, 0.0])
    ring_vertices = np.concatenate((ring_vertices, endpoint_ring[None, :, :]), axis=0)
    ring_normals = np.concatenate((np.tile(normals_one, (loop_count, 1, 1)), normals_one[:1]), axis=0)
    ring_tangents = np.concatenate((np.tile(tangent_one, (loop_count, 1, 1)), tangent_one[:1]), axis=0)

    row_shifts = np.zeros((row_count, 1, 3), dtype=np.float64)
    row_shifts[:, 0, 1] = np.arange(first_row, last_row + 1, dtype=np.float64) * ly
    vertices = (ring_vertices.reshape(1, -1, 3) + row_shifts).reshape(-1, 3)
    normals = np.tile(ring_normals.reshape(-1, 3), (row_count, 1))
    tangents = np.tile(ring_tangents.reshape(-1, 3), (row_count, 1))

    starts = np.arange(ring_count - 1, dtype=np.int64)[:, None] * tube_sides
    sides = np.arange(tube_sides, dtype=np.int64)[None, :]
    q00, q01 = starts + sides, starts + (sides + 1) % tube_sides
    q10, q11 = q00 + tube_sides, q01 + tube_sides
    base_triangles = np.stack((np.stack((q00, q01, q10), axis=-1),
                               np.stack((q01, q11, q10), axis=-1)), axis=-2).reshape(-1, 3)
    offsets = np.arange(row_count, dtype=np.int64)[:, None, None] * points_per_row
    triangles = (base_triangles[None, :, :] + offsets).reshape(-1, 3).astype(np.int32)
    corners = vertices[triangles]
    face_cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    facing = np.einsum("ij,ij->i", face_cross, normals[triangles].sum(axis=1))
    if not np.isfinite(vertices).all() or not np.isfinite(facing).all() or np.any(facing <= 0):
        raise ValueError("Deformed tube contains degenerate or inverted triangles; refine nodes or reduce radius")
    return Geometry(vertices, triangles, normals, tangents, np.ones(len(triangles), dtype=np.uint32))
