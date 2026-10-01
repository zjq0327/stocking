"""Periodic plain-knit centerlines and open circular tubes, in millimetres.

Centerline and analytic Frenet frame follow Keenan Crane (2023),
"A Simple Parametric Model of Plain-Knit Yarns":
https://github.com/keenancrane/plain-knit-yarn
The original implementation is MIT-licensed.  Periodic neighbor generation,
millimetre conversion and tube triangulation are this project's additions.
There is no force, stretch, collision response, or physical relaxation here.
"""

from dataclasses import dataclass
import math

import numpy as np

from .parameters import Parameters


_TAU = 2.0 * math.pi


@dataclass
class Geometry:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    yarn_ids: np.ndarray


def centerline(t, p: Parameters, row: int = 0) -> np.ndarray:
    """Return C(t) with shape ``(..., 3)`` in mm; t is dimensionless.

    The row argument translates by row * rowOffset in the wale (y) direction.
    C(t + 2*pi) = C(t) + (period_mm[0], 0, 0).
    """
    t = np.asarray(t, dtype=np.float64)
    return np.stack((t + p.a * np.sin(2.0 * t),
                     p.h * np.cos(t) + row * p.rowOffset,
                     p.d * np.cos(2.0 * t)), axis=-1) * p.scale_mm


def centerline_derivatives(t, p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic first and second derivatives with respect to t, in mm."""
    t = np.asarray(t, dtype=np.float64)
    velocity = np.stack((1.0 + 2.0 * p.a * np.cos(2.0 * t),
                         -p.h * np.sin(t),
                         -2.0 * p.d * np.sin(2.0 * t)), axis=-1)
    acceleration = np.stack((-4.0 * p.a * np.sin(2.0 * t),
                             -p.h * np.cos(t),
                             -4.0 * p.d * np.cos(2.0 * t)), axis=-1)
    return velocity * p.scale_mm, acceleration * p.scale_mm


def _unit(v: np.ndarray, label: str) -> np.ndarray:
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(f"degenerate {label}; choose less extreme shape parameters")
    return v / lengths


def frenet_frame(t, p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T, principal normal, binormal), all periodic unit vectors.

    This cross-product form is equivalent to normalizing the analytic dT/dt
    used in Crane's reference.  It does not numerically difference samples.
    """
    velocity, acceleration = centerline_derivatives(t, p)
    tangent = _unit(velocity, "centerline tangent")
    binormal = _unit(np.cross(velocity, acceleration), "Frenet frame")
    principal_normal = _unit(np.cross(binormal, tangent), "principal normal")
    return tangent, principal_normal, binormal


def _tube_shape_check(p: Parameters) -> dict[str, float]:
    """Sample analytic curvature to reject a locally folded circular tube.

    This is a local regularity check, not a global self-intersection guarantee.
    A tube must have R * curvature < 1 before triangles are generated.
    """
    t = np.linspace(0.0, _TAU, max(4096, 4 * p.samples_per_loop), endpoint=False)
    velocity, acceleration = centerline_derivatives(t, p)
    speed = np.linalg.norm(velocity, axis=-1)
    cross_length = np.linalg.norm(np.cross(velocity, acceleration), axis=-1)
    if (not np.all(np.isfinite(speed)) or np.any(speed <= 0.0)
            or not np.all(np.isfinite(cross_length)) or np.any(cross_length <= 0.0)):
        raise ValueError("shape parameters produce a degenerate analytic frame")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        curvature = cross_length / (speed * speed * speed)
        radius_curvature = p.R * p.scale_mm * curvature
    if not np.all(np.isfinite(radius_curvature)):
        raise ValueError("shape parameters exceed the supported numeric range")
    maximum = float(np.max(radius_curvature))
    if maximum >= 1.0 - 1e-6:
        raise ValueError(
            f"R is too large for the centerline curvature (max R*kappa={maximum:.6g}); "
            "reduce R or change the loop shape"
        )
    return {"sampled_max_radius_curvature": maximum,
            "sampled_min_speed_mm": float(np.min(speed))}


def _generate(p: Parameters, rows: range, first_loop: int, loop_count: int) -> Geometry:
    p.validate()
    _tube_shape_check(p)
    rings = loop_count * p.samples_per_loop + 1
    points_per_row = rings * p.tube_sides
    total_points = len(rows) * points_per_row
    if total_points > np.iinfo(np.int32).max:
        raise ValueError("geometry exceeds the int32 vertex-index limit")

    # Include the final ring to avoid shortening each row by one sample.
    t = first_loop * _TAU + np.arange(rings, dtype=np.float64) * (_TAU / p.samples_per_loop)
    phi = np.arange(p.tube_sides, dtype=np.float64) * (_TAU / p.tube_sides)
    tangent, e2, e3 = frenet_frame(t, p)
    radial = (e2[:, None, :] * np.cos(phi)[None, :, None]
              + e3[:, None, :] * np.sin(phi)[None, :, None])
    radial = _unit(radial, "tube normal")
    base_vertices = (centerline(t, p)[:, None, :]
                     + (p.R * p.scale_mm) * radial).reshape(-1, 3)
    row_tangents = np.broadcast_to(tangent[:, None, :], radial.shape).reshape(-1, 3)
    row_normals = radial.reshape(-1, 3)
    offsets = np.zeros((len(rows), 1, 3), dtype=np.float64)
    offsets[:, 0, 1] = np.asarray(rows, dtype=np.float64) * p.rowOffset * p.scale_mm
    vertices = (base_vertices[None, :, :] + offsets).reshape(-1, 3)
    normals = np.tile(row_normals, (len(rows), 1))
    tangents = np.tile(row_tangents, (len(rows), 1))

    ring_start = np.arange(rings - 1, dtype=np.int64)[:, None] * p.tube_sides
    sides = np.arange(p.tube_sides, dtype=np.int64)[None, :]
    q00 = ring_start + sides
    q01 = ring_start + (sides + 1) % p.tube_sides
    q10 = q00 + p.tube_sides
    q11 = q01 + p.tube_sides
    # phi x t points outward; t x phi would reverse the entire tube.
    row_triangles = np.stack((np.stack((q00, q01, q10), axis=-1),
                              np.stack((q01, q11, q10), axis=-1)), axis=-2).reshape(-1, 3)
    index_offsets = np.arange(len(rows), dtype=np.int64)[:, None, None] * points_per_row
    triangles = (row_triangles[None, :, :] + index_offsets).reshape(-1, 3).astype(np.int32)
    yarn_ids = np.ones(len(triangles), dtype=np.uint32)

    # A coarse polygonal mesh can invert even when the smooth tube is regular.
    edges1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    face_cross = np.cross(edges1, edges2)
    facing = np.einsum("ij,ij->i", face_cross, np.sum(normals[triangles], axis=1))
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(facing)) or np.any(facing <= 0.0):
        raise ValueError("tube triangles are degenerate or inverted; increase sampling or change shape")
    return Geometry(vertices, triangles, normals, tangents, yarn_ids)


def generate_tile_geometry(params: Parameters) -> Geometry:
    """Generate neighbors covering tile x=[0,Lx], y=[0,Ly], without end caps.

    |C.x - t| <= |a| and |C.y - row*w| <= h.  Adding R gives conservative
    tube bounds.  Rows whose bounds touch the tile are retained.  Whole extra
    loops put open ends strictly outside the tile and include every possible
    contributing point, even for shapes that backtrack along x.
    """
    params.validate()
    reach = params.h + params.R
    lower = -reach / params.rowOffset
    upper = 1.0 + reach / params.rowOffset
    first_row = math.ceil(np.nextafter(lower, -np.inf))
    last_row = math.floor(np.nextafter(upper, np.inf))
    pad = math.floor((abs(params.a) + params.R) / _TAU) + 1
    return _generate(params, range(first_row, last_row + 1), -pad, 1 + 2 * pad)


def generate_preview_geometry(params: Parameters) -> Geometry:
    """Generate exactly nRows rows of nLoops periods, with open boundary ends."""
    params.validate()
    return _generate(params, range(params.nRows), 0, params.nLoops)


def geometry_diagnostics(geometry: Geometry, params: Parameters) -> dict:
    """Return JSON-ready local quality metrics; does not test global collisions."""
    face_vertices = geometry.vertices[geometry.triangles]
    face_cross = np.cross(face_vertices[:, 1] - face_vertices[:, 0],
                          face_vertices[:, 2] - face_vertices[:, 0])
    twice_area = np.linalg.norm(face_cross, axis=1)
    face_normal = face_cross / twice_area[:, None]
    average_normal = _unit(np.sum(geometry.normals[geometry.triangles], axis=1), "face normal average")
    return {
        "vertex_count": int(len(geometry.vertices)),
        "triangle_count": int(len(geometry.triangles)),
        "bounds_mm": [geometry.vertices.min(axis=0).tolist(), geometry.vertices.max(axis=0).tolist()],
        "period_mm": list(params.period_mm),
        "min_triangle_area_mm2": float(0.5 * twice_area.min()),
        "min_outward_normal_dot": float(np.min(np.sum(face_normal * average_normal, axis=1))),
        "max_normal_length_error": float(np.max(np.abs(np.linalg.norm(geometry.normals, axis=1) - 1.0))),
        "max_tangent_length_error": float(np.max(np.abs(np.linalg.norm(geometry.tangents, axis=1) - 1.0))),
        "max_normal_tangent_dot": float(np.max(np.abs(np.sum(geometry.normals * geometry.tangents, axis=1)))),
        "global_collision_check": "not_performed",
        **_tube_shape_check(params),
    }
