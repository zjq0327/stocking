"""Streamable geometric plain-knit embedding on an authored garment pair.

This module does not solve rod equilibrium.  Its material chart is built once
from X0 ring arc length and a fixed integer course count.  For every centerline
sample it locates the material triangle containing q and evaluates
``C = bary @ X + h * interpolated_surface_normal``.  X0 and X1 are sampled
separately; neither receives an additional deformation gradient.

The closed toe uses the procedural shell's distributed seam.  Course yarns
terminate at the toe chart boundary; a separate synthetic seam tube may be
added.  This is a geometric layout, not an inferred knit/short-row pattern.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math
import time

import numpy as np

from .geometry import Geometry
from .parameters import Parameters


TAU = 2.0 * math.pi


def _unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(length).all() or np.any(length <= 1e-14):
        raise ValueError("geometric yarn has a zero or nonfinite frame vector")
    return v / length


def _vertex_normals(positions, triangles):
    tri = positions[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    result = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(result, triangles[:, corner], cross)
    return _unit(result)


@dataclass
class AuthoredPhase:
    face_q_mm: np.ndarray
    course_repeat_count: int
    course_width_mm: float
    wale_length_mm: float
    toe_seam_mask: np.ndarray
    seam_vertex_start: int
    metadata: dict


def build_authored_phase(pair, yarn_parameters=None, *, course_repeat_count=None):
    """Build a global q chart from original X0 lengths, not texture UV scale."""
    p = Parameters() if yarn_parameters is None else yarn_parameters
    p.validate()
    pair.validate()
    source = pair.metadata.get("parameters", {})
    if pair.metadata.get("generator_version") != "procedural-reference-v1":
        raise ValueError("authored yarn requires a procedural-reference-v1 X0 chart")
    count = source.get("circumference_segments")
    regions = pair.metadata.get("ring_regions")
    if not isinstance(count, int) or count < 4 or not isinstance(regions, list) or not regions:
        raise ValueError("X0 ring structure is missing")
    ring_count = len(regions)
    seam_start = count * ring_count
    expected = seam_start + count // 2 + 1
    if len(pair.material_point_ids) != expected or not np.array_equal(pair.material_point_ids, np.arange(expected)):
        raise ValueError("X0 material IDs do not match procedural ring/seam topology")
    ring = pair.reference_positions_mm[:seam_start].reshape(ring_count, count, 3)
    k = np.arange(count)
    closure = pair.reference_positions_mm[seam_start + np.minimum(k, count - k)]
    all_rings = np.concatenate((ring, closure[None]), axis=0)
    edges = np.linalg.norm(np.roll(all_rings, -1, axis=1) - all_rings, axis=-1)
    circumference = edges.sum(axis=1)
    if not np.isfinite(edges).all() or np.any(edges <= 0):
        raise ValueError("X0 ring contains degenerate material edges")
    if course_repeat_count is None:
        course_repeat_count = max(1, int(math.floor(circumference[0] / p.period_mm[0] + 0.5)))
    if isinstance(course_repeat_count, bool) or not isinstance(course_repeat_count, (int, np.integer)) or course_repeat_count < 1:
        raise ValueError("course_repeat_count must be a positive integer")
    arc = np.concatenate((np.zeros((ring_count + 1, 1)), np.cumsum(edges, axis=1)), axis=1)
    arc /= circumference[:, None]
    arc[:, -1] = 1.0
    uv = pair.triangle_material_uv
    corner_k_float = uv[..., 0] * count
    corner_k = np.rint(corner_k_float).astype(np.int64)
    if np.max(np.abs(corner_k_float - corner_k)) > 1e-8 or np.any((corner_k < 0) | (corner_k > count)):
        raise ValueError("X0 UV corners do not identify exact ring meridians")
    corner_rows = np.minimum(pair.triangles // count, ring_count)
    radius = float(source["heel_bend_radius_mm"])
    length = (float(source["leg_length_mm"]) - radius
              + 0.5 * math.pi * radius
              + float(source["foot_length_mm"]) - radius)
    q = np.empty_like(uv)
    q[..., 0] = arc[corner_rows, corner_k] * course_repeat_count * p.period_mm[0]
    q[..., 1] = -uv[..., 1] * length
    a = q[:, 0]
    e = np.stack((q[:, 1] - a, q[:, 2] - a), axis=-1)
    det = np.linalg.det(e)
    if not np.isfinite(q).all() or np.any(np.abs(det) <= 1e-14):
        raise ValueError("global authored material q chart is singular")
    toe_mask = np.any(pair.triangles >= seam_start, axis=1)
    metadata = {
        "schema": "stocking.authored_yarn_phase.v1",
        "length_unit": "mm",
        "course_repeat_count": int(course_repeat_count),
        "course_width_mm": float(course_repeat_count * p.period_mm[0]),
        "wale_length_mm": float(length),
        "reference_ring_circumference_mm": [float(circumference.min()), float(circumference.max())],
        "material_chart": "global X0 ring arc fraction times integer course count; minus X0 centerline length",
        "toe_layout": "synthetic distributed closure seam; no short rows or recovered stitch instructions",
        "yarn_mechanics_solved": False,
        "macro_deformation_applied_once": True,
    }
    return AuthoredPhase(q, int(course_repeat_count), float(course_repeat_count * p.period_mm[0]),
                         float(length), toe_mask, seam_start, metadata)


class _MaterialLocator:
    """Exact barycentric lookup for q samples using a sparse 2D grid."""

    def __init__(self, face_q_mm, grid_mm):
        self.q = np.asarray(face_q_mm, dtype=np.float64)
        self.grid_mm = float(grid_mm)
        if self.grid_mm <= 0:
            raise ValueError("material locator grid must be positive")
        self.base = self.q[:, 0]
        edges = np.stack((self.q[:, 1] - self.base, self.q[:, 2] - self.base), axis=-1)
        self.inverse = np.linalg.inv(edges)
        self.min = self.q.min(axis=1)
        self.max = self.q.max(axis=1)
        buckets = defaultdict(list)
        for index, (lo, hi) in enumerate(zip(self.min, self.max)):
            lower = np.floor(lo / self.grid_mm).astype(np.int64)
            upper = np.floor(hi / self.grid_mm).astype(np.int64)
            for ix in range(int(lower[0]), int(upper[0]) + 1):
                for iy in range(int(lower[1]), int(upper[1]) + 1):
                    buckets[(ix, iy)].append(index)
        self.buckets = {key: np.asarray(value, dtype=np.int32) for key, value in buckets.items()}

    def locate(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("material q points must be [N,2]")
        face = np.full(len(points), -1, dtype=np.int32)
        bary = np.zeros((len(points), 3), dtype=np.float64)
        cells = np.floor(points / self.grid_mm).astype(np.int64)
        groups = defaultdict(list)
        for index, cell in enumerate(cells):
            groups[(int(cell[0]), int(cell[1]))].append(index)
        for cell, indices in groups.items():
            candidates = self.buckets.get(cell)
            if candidates is None:
                continue
            candidates = np.unique(candidates)
            for start in range(0, len(indices), 512):
                selected = np.asarray(indices[start:start + 512], dtype=np.int64)
                delta = points[selected, None, :] - self.base[None, candidates, :]
                uv = np.einsum("cij,pcj->pci", self.inverse[candidates], delta)
                w = 1.0 - uv[..., 0] - uv[..., 1]
                inside = ((uv[..., 0] >= -2e-10) & (uv[..., 1] >= -2e-10)
                          & (w >= -2e-10) & (uv[..., 0] <= 1.0 + 2e-10)
                          & (uv[..., 1] <= 1.0 + 2e-10))
                has = inside.any(axis=1)
                choice = inside.argmax(axis=1)
                chosen = selected[has]
                face[chosen] = candidates[choice[has]]
                bary[chosen, 0] = w[has, choice[has]]
                bary[chosen, 1:] = uv[has, choice[has]]
        return face, bary


def _discrete_curvature_ratio(centers, edges, radius_mm):
    if len(centers) == 0 or len(edges) == 0:
        return 0.0
    incoming = np.zeros_like(centers)
    outgoing = np.zeros_like(centers)
    for a, b in edges:
        d = centers[b] - centers[a]
        outgoing[a] += d
        incoming[b] += d
    left = np.linalg.norm(incoming, axis=1)
    right = np.linalg.norm(outgoing, axis=1)
    chord = np.linalg.norm(incoming + outgoing, axis=1)
    interior = (left > 1e-14) & (right > 1e-14)
    curvature = np.zeros(len(centers), dtype=np.float64)
    curvature[interior] = (2 * np.linalg.norm(np.cross(incoming[interior], outgoing[interior]), axis=1)
                           / (left[interior] * right[interior] * chord[interior]))
    return float(np.max(radius_mm * curvature))


def _tube_mesh(centers, shell_normals, edges, radius_mm, sides):
    if len(centers) == 0 or len(edges) == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return (Geometry(empty3, np.empty((0, 3), dtype=np.int32), empty3.copy(), empty3.copy(),
                         np.empty(0, dtype=np.uint32)),
                {"maximum_radius_times_discrete_curvature": 0.0,
                 "minimum_face_outward_dot": None,
                 "nonlocal_self_intersection_check": "not_performed"})
    tangent = np.zeros_like(centers)
    for a, b in edges:
        d = centers[b] - centers[a]
        tangent[a] += d
        tangent[b] += d
    maximum_curvature_ratio = _discrete_curvature_ratio(centers, edges, radius_mm)
    if not math.isfinite(maximum_curvature_ratio) or maximum_curvature_ratio >= 1.0 - 1e-6:
        raise ValueError(f"embedded yarn centerline bends tighter than its radius: "
                         f"max radius*discrete_curvature={maximum_curvature_ratio:.6g}")
    tangent = _unit(tangent)
    radial1 = _unit(np.cross(shell_normals, tangent))
    radial2 = _unit(np.cross(tangent, radial1))
    phi = TAU * np.arange(sides) / sides
    radial = (radial1[:, None, :] * np.cos(phi)[None, :, None]
              + radial2[:, None, :] * np.sin(phi)[None, :, None])
    vertices = (centers[:, None, :] + radius_mm * radial).reshape(-1, 3)
    normals = radial.reshape(-1, 3)
    tangents = np.broadcast_to(tangent[:, None, :], radial.shape).reshape(-1, 3).copy()
    edge = np.asarray(edges, dtype=np.int64)
    side = np.arange(sides, dtype=np.int64)
    a = edge[:, 0, None] * sides + side
    b = edge[:, 1, None] * sides + side
    an = edge[:, 0, None] * sides + (side + 1) % sides
    bn = edge[:, 1, None] * sides + (side + 1) % sides
    faces = np.stack((np.stack((a, an, b), axis=-1),
                      np.stack((an, bn, b), axis=-1)), axis=-2).reshape(-1, 3)
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("one tube chunk exceeds int32 indices; reduce chunk size")
    minimum_outward = math.inf
    for start in range(0, len(faces), 250_000):
        tri = faces[start:start + 250_000]
        xyz = vertices[tri]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        outward = np.einsum("ij,ij->i", normal, normals[tri].sum(axis=1))
        if not np.isfinite(outward).all() or np.any(outward <= 0):
            raise ValueError("embedded tube contains degenerate or inward-facing triangles")
        minimum_outward = min(minimum_outward, float(outward.min()))
    quality = {"maximum_radius_times_discrete_curvature": maximum_curvature_ratio,
               "minimum_face_outward_dot": minimum_outward,
               "nonlocal_self_intersection_check": "not_performed"}
    return (Geometry(vertices, faces.astype(np.int32), normals, tangents,
                     np.ones(len(faces), dtype=np.uint32)), quality)


class CapsuleSupport:
    """Exact centerline capsule union membership for ray-origin rejection.

    The polygonal capsule mesh is inscribed in this analytic support.  A ray
    starting outside the analytic union is also outside the polygonal mesh;
    its first front-facing hit is on the polygonal union's outer boundary.
    """

    def __init__(self, centers, edges, radius_mm, triangle_tangent_valid):
        self.centers_mm = np.asarray(centers, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.int32)
        self.radius_mm = float(radius_mm)
        self.triangle_tangent_valid = np.asarray(triangle_tangent_valid, dtype=bool)
        self.grid_mm = 2 * self.radius_mm
        buckets = defaultdict(list)
        endpoint = self.centers_mm[self.edges]
        lo = np.floor((endpoint.min(axis=1) - self.radius_mm) / self.grid_mm).astype(np.int64)
        hi = np.floor((endpoint.max(axis=1) + self.radius_mm) / self.grid_mm).astype(np.int64)
        for i in range(len(self.edges)):
            for ix in range(int(lo[i, 0]), int(hi[i, 0]) + 1):
                for iy in range(int(lo[i, 1]), int(hi[i, 1]) + 1):
                    for iz in range(int(lo[i, 2]), int(hi[i, 2]) + 1):
                        buckets[(ix, iy, iz)].append(i)
        self._buckets = {key: np.asarray(value, dtype=np.int32) for key, value in buckets.items()}

    def origins_outside(self, points_mm, *, tolerance_mm=1e-6):
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("capsule origin points must be finite [N,3]")
        if tolerance_mm < 0 or tolerance_mm >= self.radius_mm:
            raise ValueError("capsule origin tolerance must be 0..radius")
        outside = np.ones(len(points), dtype=bool)
        cells = np.floor(points / self.grid_mm).astype(np.int64)
        for i, cell in enumerate(cells):
            candidates = self._buckets.get(tuple(int(v) for v in cell))
            if candidates is None:
                continue
            segment = self.centers_mm[self.edges[candidates]]
            a, b = segment[:, 0], segment[:, 1]
            ab = b - a
            denominator = np.einsum("ij,ij->i", ab, ab)
            t = np.clip(np.einsum("ij,ij->i", points[i] - a, ab) / denominator, 0., 1.)
            distance_sq = np.einsum("ij,ij->i", points[i] - a - t[:, None] * ab,
                                    points[i] - a - t[:, None] * ab)
            outside[i] = bool(np.all(distance_sq > (self.radius_mm + tolerance_mm)**2))
        return outside


class _FaceAABBIndex:
    """Conservative world-grid accelerator for local shell-face queries."""

    def __init__(self, minimum, maximum, grid_mm=10.0):
        self.minimum = minimum
        self.maximum = maximum
        self.grid_mm = float(grid_mm)
        lower = np.floor(minimum / self.grid_mm).astype(np.int32)
        upper = np.floor(maximum / self.grid_mm).astype(np.int32)
        buckets = defaultdict(list)
        large = []
        for face, (lo, hi) in enumerate(zip(lower, upper)):
            span = hi.astype(np.int64) - lo.astype(np.int64) + 1
            if int(np.prod(span)) > 128:
                large.append(face)
                continue
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        buckets[(ix, iy, iz)].append(face)
        self.buckets = {key: np.asarray(value, dtype=np.int32) for key, value in buckets.items()}
        self.large_faces = np.asarray(large, dtype=np.int32)

    def query(self, query_min, query_max):
        lower = np.floor(query_min / self.grid_mm).astype(np.int32)
        upper = np.floor(query_max / self.grid_mm).astype(np.int32)
        span = upper.astype(np.int64) - lower.astype(np.int64) + 1
        if int(np.prod(span)) > 512:
            candidates = np.arange(len(self.minimum), dtype=np.int32)
        else:
            parts = [self.large_faces]
            for ix in range(int(lower[0]), int(upper[0]) + 1):
                for iy in range(int(lower[1]), int(upper[1]) + 1):
                    for iz in range(int(lower[2]), int(upper[2]) + 1):
                        found = self.buckets.get((ix, iy, iz))
                        if found is not None:
                            parts.append(found)
            candidates = np.unique(np.concatenate(parts))
        if not len(candidates):
            return candidates
        near = np.all((self.maximum[candidates] >= query_min)
                      & (self.minimum[candidates] <= query_max), axis=1)
        return candidates[near]


def _capsule_union_mesh(centers, edges, radius_mm, sides, cap_rings, max_vertices):
    """Unmerged polygonal capsules whose first outside hit is a union hit."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        geometry = Geometry(empty3, np.empty((0, 3), dtype=np.int32), empty3.copy(),
                            empty3.copy(), np.empty(0, dtype=np.uint32))
        return geometry, CapsuleSupport(centers, edges, radius_mm, np.empty(0, dtype=bool)), {
            "maximum_original_radius_times_discrete_curvature": 0.0,
            "minimum_face_outward_dot": None,
        }
    ring_count = 2 + 2 * cap_rings
    per_segment_vertices = ring_count * sides + 2
    planned = len(edges) * per_segment_vertices
    if planned > max_vertices or planned > np.iinfo(np.int32).max:
        raise ValueError(f"capsule patch needs {planned:,} vertices, over limit {max_vertices:,}; split material face chunk")
    vparts, nparts, tparts, fparts = [], [], [], []
    angle = TAU * np.arange(sides) / sides
    j = np.arange(sides, dtype=np.int32)
    jn = (j + 1) % sides
    local_faces = [np.column_stack((np.zeros(sides, dtype=np.int32), 1 + jn, 1 + j))]
    for r in range(ring_count - 1):
        first = 1 + r * sides
        second = first + sides
        local_faces.append(np.column_stack((first + j, first + jn, second + j)))
        local_faces.append(np.column_stack((first + jn, second + jn, second + j)))
    last_ring = 1 + (ring_count - 1) * sides
    local_faces.append(np.column_stack((last_ring + j, last_ring + jn,
                                        np.full(sides, per_segment_vertices - 1, dtype=np.int32))))
    local_faces = np.concatenate(local_faces)
    local_tangent_valid = np.ones(len(local_faces), dtype=bool)
    local_tangent_valid[:sides] = False
    local_tangent_valid[-sides:] = False
    for edge_number, (ia, ib) in enumerate(edges):
        a, b = centers[ia], centers[ib]
        segment = b - a
        length = float(np.linalg.norm(segment))
        if not math.isfinite(length) or length <= 1e-14:
            raise ValueError("capsule support contains a zero-length material segment")
        tangent = segment / length
        axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        radial1 = _unit(np.cross(axis, tangent))
        radial2 = _unit(np.cross(tangent, radial1))
        radial = np.cos(angle)[:, None] * radial1 + np.sin(angle)[:, None] * radial2
        internal = np.arange(1, cap_rings + 1) * (math.pi / 2) / (cap_rings + 1)
        profile = [(-radius_mm * math.cos(theta), radius_mm * math.sin(theta),
                    -math.cos(theta), math.sin(theta)) for theta in internal]
        profile.append((0., radius_mm, 0., 1.))
        profile.append((length, radius_mm, 0., 1.))
        profile.extend((length + radius_mm * math.cos(theta), radius_mm * math.sin(theta),
                        math.cos(theta), math.sin(theta)) for theta in internal[::-1])
        rings = []
        normals = []
        projected_tangents = []
        for axial, radial_distance, normal_t, normal_r in profile:
            direction = normal_t * tangent + normal_r * radial
            rings.append(a + axial * tangent + radial_distance * radial)
            normals.append(direction)
            projection = tangent[None, :] - normal_t * direction
            projected_tangents.append(_unit(projection))
        pole_a = a - radius_mm * tangent
        pole_b = b + radius_mm * tangent
        vertices = np.vstack((pole_a[None], *rings, pole_b[None]))
        vertex_normals = np.vstack((-tangent[None], *normals, tangent[None]))
        vertex_tangents = np.vstack((radial1[None], *projected_tangents, radial1[None]))
        offset = edge_number * per_segment_vertices
        vparts.append(vertices)
        nparts.append(vertex_normals)
        tparts.append(vertex_tangents)
        fparts.append(local_faces + offset)
    vertices = np.concatenate(vparts)
    normals = np.concatenate(nparts)
    tangents = np.concatenate(tparts)
    faces = np.concatenate(fparts)
    triangle_tangent_valid = np.tile(local_tangent_valid, len(edges))
    minimum_outward = math.inf
    for start in range(0, len(faces), 250_000):
        tri = faces[start:start + 250_000]
        xyz = vertices[tri]
        geometric = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        outward = np.einsum("ij,ij->i", geometric, normals[tri].sum(axis=1))
        if not np.isfinite(outward).all() or np.any(outward <= 0):
            raise ValueError("capsule union has degenerate or inward-facing triangles")
        minimum_outward = min(minimum_outward, float(outward.min()))
    quality = {
        "maximum_original_radius_times_discrete_curvature": _discrete_curvature_ratio(centers, edges, radius_mm),
        "minimum_face_outward_dot": minimum_outward,
        "pole_fan_triangles_tangent_undefined": int((~triangle_tangent_valid).sum()),
        "polygonal_radial_hausdorff_upper_bound_mm": float(radius_mm * (
            2 - math.cos(math.pi / sides)
              - math.cos(math.pi / (4 * (cap_rings + 1))))),
        "representation": "unmerged polygonal capsule union; analytic capsule membership for origin rejection",
        "global_self_intersection_check": "not_performed",
    }
    geometry = Geometry(vertices, faces, normals, tangents, np.ones(len(faces), dtype=np.uint32))
    support = CapsuleSupport(centers, edges, radius_mm, triangle_tangent_valid)
    return geometry, support, quality


class AuthoredYarnProvider:
    """Generate real local tube geometry for material-face feature consumers.

    ``halo_mm`` is an added world-space ray/reconstruction reach. Candidate
    material faces are selected by both 3D overlap and chart distance scaled
    by the inverse local q->X metric.  It is never a fixed number of periods.
    The caller must perform a larger-halo convergence check near folds.
    """

    def __init__(self, pair, parameters=None, *, course_repeat_count=None,
                 max_tube_vertices=10_000_000, representation="strict_tube",
                 capsule_cap_rings=3):
        self.pair = pair.validate()
        self.parameters = Parameters() if parameters is None else parameters
        self.parameters.validate()
        self.phase = build_authored_phase(pair, self.parameters,
                                          course_repeat_count=course_repeat_count)
        self.locator = _MaterialLocator(self.phase.face_q_mm,
                                        max(self.parameters.period_mm) * 4)
        self.max_tube_vertices = int(max_tube_vertices)
        if self.max_tube_vertices < self.parameters.tube_sides * 2:
            raise ValueError("max_tube_vertices is too small for one yarn segment")
        if representation not in ("strict_tube", "capsule_union"):
            raise ValueError("representation must be strict_tube or capsule_union")
        if isinstance(capsule_cap_rings, bool) or not isinstance(capsule_cap_rings, int) or capsule_cap_rings < 1:
            raise ValueError("capsule_cap_rings must be a positive integer")
        self.representation = representation
        self.capsule_cap_rings = capsule_cap_rings
        self._positions = {
            "reference": self.pair.reference_positions_mm,
            "current": self.pair.current_positions_mm,
        }
        self._normals = {key: _vertex_normals(value, pair.triangles)
                         for key, value in self._positions.items()}
        self._bounds = {key: (value[pair.triangles].min(axis=1), value[pair.triangles].max(axis=1))
                        for key, value in self._positions.items()}
        q = self.phase.face_q_mm
        self._q_min = q.min(axis=1)
        self._q_max = q.max(axis=1)
        qedge = np.stack((q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=-1)
        self._inv_q = np.linalg.inv(qedge)
        self._q_base = q[:, 0]
        self._metric = {}
        for key, value in self._positions.items():
            xedge = np.stack((value[pair.triangles[:, 1]] - value[pair.triangles[:, 0]],
                              value[pair.triangles[:, 2]] - value[pair.triangles[:, 0]]), axis=-1)
            derivative = xedge @ self._inv_q
            self._metric[key] = {
                "minimum_stretch": np.linalg.svd(derivative, compute_uv=False)[:, 1],
                "world_to_q_inverse": np.linalg.pinv(derivative),
            }
        self._face_spatial_index = {}
        # Feature tiles repeatedly request overlapping material cells.  Cache
        # their exact mapped 64/192-node segments, bounded independently of
        # the full stocking size.  The cache is per configuration and never
        # changes a material q or the corresponding X0/X1 embedding.
        self._segment_cell_cache = OrderedDict()
        self._segment_cell_cache_limit = 20_000
        self._last_coverage = None

    def estimate(self):
        p = self.parameters
        first_row = int(math.ceil((-self.phase.wale_length_mm - p.h * p.scale_mm) / p.period_mm[1]))
        last_row = int(math.floor((p.h * p.scale_mm) / p.period_mm[1]))
        row_count = last_row - first_row + 1
        loops = row_count * self.phase.course_repeat_count
        samples = loops * p.samples_per_loop
        return {
            "course_repeat_count": self.phase.course_repeat_count,
            "first_candidate_wale_row": first_row,
            "last_candidate_wale_row": last_row,
            "candidate_wale_rows": row_count,
            "nominal_loops": loops,
            "nominal_centerline_samples": samples,
            "nominal_tube_vertices": samples * p.tube_sides,
            "tube_positions_float64_GiB": samples * p.tube_sides * 3 * 8 / 2**30,
            "tube_PNT_float64_GiB": samples * p.tube_sides * 3 * 3 * 8 / 2**30,
            "tube_triangles_int32_GiB": samples * p.tube_sides * 2 * 3 * 4 / 2**30,
            "both_state_centerline_positions_float64_GiB": samples * 2 * 3 * 8 / 2**30,
            "scope": "candidate fixed-count course rows; boundary samples are clipped by material chart; excludes seam tube",
        }

    def centerline_chunk(self, configuration, rows, loops):
        """Return stable-ID material centerlines for a row/loop rectangle.

        Adjacent chunks duplicate endpoint samples with the same global sample
        ID, so they can be welded exactly when streamed to a mesh consumer.
        The output is geometry only and does not imply collision equilibrium.
        """
        if configuration not in self._positions:
            raise ValueError("configuration must be 'reference' or 'current'")
        row_values = sorted(set(int(x) for x in rows))
        loop_values = sorted(set(int(x) for x in loops))
        if not row_values or not loop_values or loop_values[0] < 0 or loop_values[-1] >= self.phase.course_repeat_count:
            raise ValueError("rows and loops must be nonempty, with loops in the global course range")
        p = self.parameters
        ring_samples = self.phase.course_repeat_count * p.samples_per_loop
        centers, normals, faces, barycentrics, global_ids, material_q, edges = [], [], [], [], [], [], []
        selected_count = unmapped_count = 0
        for row in row_values:
            ids = set()
            requested_edges = []
            for loop in loop_values:
                start = loop * p.samples_per_loop
                for sample in range(p.samples_per_loop):
                    a = start + sample
                    b = (a + 1) % ring_samples
                    ids.add(a)
                    ids.add(b)
                    requested_edges.append((a, b))
            ordered = np.asarray(sorted(ids), dtype=np.int64)
            c, n, valid, face, bary = self._centerline_points(row, ordered, configuration)
            selected_count += len(ordered)
            unmapped_count += int((~valid).sum())
            t = TAU * ordered / p.samples_per_loop
            q = np.column_stack((p.scale_mm * (t + p.a * np.sin(2 * t)),
                                 row * p.period_mm[1] + p.h * p.scale_mm * np.cos(t)))
            q[:, 0] = np.mod(q[:, 0], self.phase.course_width_mm)
            offset = sum(len(part) for part in centers)
            keep = np.flatnonzero(valid)
            compact = np.full(len(ordered), -1, dtype=np.int64)
            compact[keep] = np.arange(len(keep), dtype=np.int64) + offset
            lookup = {int(gid): index for index, gid in enumerate(ordered)}
            centers.append(c[keep])
            normals.append(n[keep])
            faces.append(face[keep])
            barycentrics.append(bary[keep])
            material_q.append(q[keep])
            global_ids.append(row * ring_samples + ordered[keep])
            for a, b in requested_edges:
                ia, ib = compact[lookup[a]], compact[lookup[b]]
                if ia >= 0 and ib >= 0:
                    edges.append((int(ia), int(ib)))
        result = {
            "positions_mm": np.concatenate(centers),
            "surface_normals": np.concatenate(normals),
            "material_face_index": np.concatenate(faces),
            "material_barycentric": np.concatenate(barycentrics),
            "material_q_mm": np.concatenate(material_q),
            "global_sample_id": np.concatenate(global_ids),
            "edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        }
        coverage = {
            "schema": "stocking.authored_centerline_chunk.v1",
            "configuration": configuration,
            "first_row": row_values[0], "last_row": row_values[-1],
            "first_loop": loop_values[0], "last_loop": loop_values[-1],
            "selected_samples": selected_count,
            "mapped_samples": len(result["positions_mm"]),
            "unmapped_boundary_samples": unmapped_count,
            "mapped_segments": len(edges),
            "no_double_F": True,
            "mechanical_equilibrium_solved": False,
            "tube_quality_checked": False,
        }
        return result, coverage

    def window_geometry(self, configuration, rows, loops):
        """Build a material window; no full-face or global ray coverage claim."""
        arrays, centerline_coverage = self.centerline_chunk(configuration, rows, loops)
        centers = arrays["positions_mm"]
        edges = arrays["edges"]
        if not len(edges):
            raise ValueError("material window contains no complete mapped yarn segments")
        radius = self.parameters.R * self.parameters.scale_mm
        if self.representation == "strict_tube":
            geometry, quality = _tube_mesh(centers, arrays["surface_normals"], edges,
                                           radius, self.parameters.tube_sides)
            support = None
        else:
            geometry, support, quality = _capsule_union_mesh(
                centers, edges, radius, self.parameters.tube_sides,
                self.capsule_cap_rings, self.max_tube_vertices)
        coverage = {
            "schema": "stocking.authored_yarn_material_window.v1",
            "configuration": configuration,
            "representation": self.representation,
            "centerline": centerline_coverage,
            "tube_vertex_count": int(len(geometry.vertices)),
            "tube_triangle_count": int(len(geometry.triangles)),
            "tube_quality": quality,
            "shell_orientation": self._orientation_metrics(
                configuration, np.unique(arrays["material_face_index"])),
            "toe_seam_included": False,
            "full_face_ray_coverage": False,
            "global_nearband_certified": False,
            "material_force_equilibrium_solved": False,
        }
        result = {"geometry": geometry, "centerline": arrays, "coverage": coverage}
        if support is not None:
            result.update(origin_outside=support.origins_outside,
                          triangle_tangent_valid=support.triangle_tangent_valid,
                          capsule_support=support)
        return result

    def seam_centerline(self, configuration):
        """Synthetic distributed toe closure with common X0/X1 edge fractions."""
        if configuration not in self._positions:
            raise ValueError("configuration must be reference or current")
        start = self.phase.seam_vertex_start
        original = self._positions["reference"][start:]
        current = self._positions["current"][start:]
        seam = self._positions[configuration][start:]
        normal = self._normals[configuration][start:]
        step = self.parameters.period_mm[0] / self.parameters.samples_per_loop
        positions, normals, material_edges, fraction, sample_ids = [], [], [], [], []
        edges = []
        global_index = 0
        for a in range(len(seam) - 1):
            length = max(np.linalg.norm(original[a + 1] - original[a]),
                         np.linalg.norm(current[a + 1] - current[a]))
            count = max(1, int(math.ceil(length / step)))
            values = np.arange(0 if a == 0 else 1, count + 1, dtype=np.float64) / count
            xyz = seam[a][None] * (1 - values[:, None]) + seam[a + 1][None] * values[:, None]
            unit_normal = _unit(normal[a][None] * (1 - values[:, None])
                                + normal[a + 1][None] * values[:, None])
            positions.append(xyz)
            normals.append(unit_normal)
            material_edges.append(np.broadcast_to([start + a, start + a + 1], (len(values), 2)).copy())
            fraction.append(values)
            sample_ids.append(np.arange(global_index, global_index + len(values), dtype=np.int64))
            last = global_index + len(values)
            edges.extend((i, i + 1) for i in range(max(0, global_index - int(a > 0)), last - 1))
            global_index = last
        arrays = {
            "positions_mm": np.concatenate(positions),
            "surface_normals": np.concatenate(normals),
            "material_edge_point_ids": np.concatenate(material_edges),
            "material_edge_fraction": np.concatenate(fraction),
            "global_seam_sample_id": np.concatenate(sample_ids),
            "edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        }
        coverage = {
            "schema": "stocking.authored_yarn_seam_centerline.v1",
            "configuration": configuration,
            "sample_count": len(arrays["positions_mm"]),
            "segment_count": len(arrays["edges"]),
            "synthetic_distributed_closure": True,
            "material_correspondence": "same seam material edge IDs and interpolation fractions on X0/X1",
            "knit_short_rows_recovered": False,
            "mechanical_equilibrium_solved": False,
        }
        return arrays, coverage

    def _candidate_faces(self, configuration, requested, halo_mm):
        p = self.parameters
        bounds_min, bounds_max = self._bounds[configuration]
        extra = halo_mm + abs(p.d * p.scale_mm) + p.R * p.scale_mm
        candidates = set(int(x) for x in requested)
        qmin, qmax = self._q_min, self._q_max
        minimum_stretch = self._metric[configuration]["minimum_stretch"][requested]
        if np.any(minimum_stretch <= 1e-8):
            raise ValueError("current material chart is too compressed to bound local yarn halo")
        for face, sigma in zip(requested, minimum_stretch):
            spatial = np.all((bounds_max >= bounds_min[face] - extra)
                             & (bounds_min <= bounds_max[face] + extra), axis=1)
            q_margin = extra / sigma
            chart = np.all((qmax >= qmin[face] - q_margin)
                           & (qmin <= qmax[face] + q_margin), axis=1)
            candidates.update(np.flatnonzero(spatial | chart).tolist())
        return np.asarray(sorted(candidates), dtype=np.int32), float(minimum_stretch.min())

    def _orientation_metrics(self, configuration, faces):
        faces = np.asarray(faces, dtype=np.int64)
        if not len(faces):
            return {"minimum_face_to_smoothed_normal_dot": None,
                    "nonpositive_face_orientation_count": 0}
        triangles = self.pair.triangles[faces]
        xyz = self._positions[configuration][triangles]
        geometric = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        length = np.linalg.norm(geometric, axis=1)
        smooth = self._normals[configuration][triangles].sum(axis=1)
        smooth_length = np.linalg.norm(smooth, axis=1)
        dot = np.divide(np.einsum("ij,ij->i", geometric, smooth),
                        length * smooth_length,
                        out=np.full(len(faces), -1., dtype=np.float64),
                        where=(length > 1e-14) & (smooth_length > 1e-14))
        return {"minimum_face_to_smoothed_normal_dot": float(dot.min()),
                "nonpositive_face_orientation_count": int(np.count_nonzero(dot <= 0)),
                "nonpositive_face_indices": np.asarray(faces)[dot <= 0].astype(int).tolist()}

    def _candidate_cells(self, faces, configuration, halo_mm):
        p = self.parameters
        q = self.phase.face_q_mm
        sigma = self._metric[configuration]["minimum_stretch"][faces]
        if np.any(sigma <= 1e-8):
            raise ValueError("candidate material chart is too compressed for bounded yarn generation")
        xwave = abs(p.a * p.scale_mm)
        ywave = p.h * p.scale_mm
        reach = halo_mm + abs(p.d * p.scale_mm) + p.R * p.scale_mm
        cells = defaultdict(set)
        for face, stretch in zip(faces, sigma):
            bound = reach / stretch
            lo = q[face].min(axis=0) - bound
            hi = q[face].max(axis=0) + bound
            first_loop = int(math.floor((lo[0] - xwave) / p.period_mm[0]))
            last_loop = int(math.floor((hi[0] + xwave) / p.period_mm[0]))
            first_row = int(math.ceil((lo[1] - ywave) / p.period_mm[1]))
            last_row = int(math.floor((hi[1] + ywave) / p.period_mm[1]))
            for row in range(first_row, last_row + 1):
                for loop in range(first_loop, last_loop + 1):
                    cells[row].add(loop % self.phase.course_repeat_count)
        return cells, float(sigma.min())

    def for_samples(self, configuration, faces, barycentric, halo_mm, *, plan_only=False,
                    allow_creased_director=False):
        """Local actual-tube query around specified shell samples.

        This narrows coarse source faces with their affine q-to-X inverse and
        the world query box.  It does not certify that a long, nonlocal tube
        chord from an otherwise remote face never crosses the ray band.
        """
        if configuration not in self._positions:
            raise ValueError("configuration must be reference or current")
        face = np.asarray(faces, dtype=np.int64)
        bary = np.asarray(barycentric, dtype=np.float64)
        if (face.ndim != 1 or not len(face) or bary.shape != (len(face), 3)
                or np.any(face < 0) or np.any(face >= len(self.pair.triangles))
                or not np.isfinite(bary).all()
                or np.any(bary < -1e-9) or np.any(bary > 1 + 1e-9)
                or np.max(np.abs(bary.sum(axis=1) - 1)) > 1e-9):
            raise ValueError("for_samples requires valid face[N] and barycentric[N,3]")
        if not math.isfinite(halo_mm) or halo_mm <= 0:
            raise ValueError("halo_mm must be a finite positive world-space reach")
        if type(allow_creased_director) is not bool:
            raise ValueError("allow_creased_director must be a bool")
        if allow_creased_director and self.representation != "capsule_union":
            raise ValueError("creased directors require the explicitly declared capsule representation")
        p = self.parameters
        position = self._positions[configuration]
        query = np.einsum("ni,nij->nj", bary, position[self.pair.triangles[face]])
        query_min, query_max = query.min(axis=0), query.max(axis=0)
        reach = halo_mm + abs(p.d * p.scale_mm) + p.R * p.scale_mm
        bound_min, bound_max = self._bounds[configuration]
        if configuration not in self._face_spatial_index:
            self._face_spatial_index[configuration] = _FaceAABBIndex(bound_min, bound_max)
        candidates = self._face_spatial_index[configuration].query(query_min - reach,
                                                                     query_max + reach)
        if not len(candidates):
            raise ValueError("no material shell face near supplied query samples")
        xyz_corners = np.asarray([[x, y, z]
                                  for x in (query_min[0], query_max[0])
                                  for y in (query_min[1], query_max[1])
                                  for z in (query_min[2], query_max[2])], dtype=np.float64)
        q = self.phase.face_q_mm
        cells = defaultdict(set)
        used_faces = []
        used_sigma = []
        xwave = abs(p.a * p.scale_mm)
        ywave = p.h * p.scale_mm
        sigma_by_face = self._metric[configuration]["minimum_stretch"]
        inverse_by_face = self._metric[configuration]["world_to_q_inverse"]
        for index in candidates:
            tri = self.pair.triangles[index]
            sigma = float(sigma_by_face[index])
            if sigma <= 1e-8:
                raise ValueError("nearby material face is too compressed to bound sample yarn support")
            inverse = inverse_by_face[index]
            projected = (xyz_corners - position[tri[0]]) @ inverse.T + q[index, 0]
            lo = np.maximum(self._q_min[index], projected.min(axis=0) - reach / sigma)
            hi = np.minimum(self._q_max[index], projected.max(axis=0) + reach / sigma)
            if np.any(lo > hi):
                continue
            used_faces.append(index)
            used_sigma.append(sigma)
            first_loop = int(math.floor((lo[0] - xwave) / p.period_mm[0]))
            last_loop = int(math.floor((hi[0] + xwave) / p.period_mm[0]))
            first_row = int(math.ceil((lo[1] - ywave) / p.period_mm[1]))
            last_row = int(math.floor((hi[1] + ywave) / p.period_mm[1]))
            for row in range(first_row, last_row + 1):
                for loop in range(first_loop, last_loop + 1):
                    cells[row].add(loop % self.phase.course_repeat_count)
        if not cells:
            raise ValueError("sample neighborhood has no material course cells")
        shell_orientation = self._orientation_metrics(configuration, used_faces)
        if shell_orientation["nonpositive_face_orientation_count"] and not allow_creased_director:
            raise ValueError("sample yarn support includes shell faces opposing their smoothed normals")
        seam = position[self.phase.seam_vertex_start:]
        segment_min = np.minimum(seam[:-1], seam[1:])
        segment_max = np.maximum(seam[:-1], seam[1:])
        seam_near = np.all((segment_max >= query_min - reach)
                           & (segment_min <= query_max + reach), axis=1)
        seam_segments = np.flatnonzero(seam_near).tolist()
        potential_samples = sum(len(loops) * p.samples_per_loop + 1 for loops in cells.values())
        seam0 = self._positions["reference"][self.phase.seam_vertex_start:]
        seam1 = self._positions["current"][self.phase.seam_vertex_start:]
        seam_step = p.period_mm[0] / p.samples_per_loop
        seam_sample_upper = sum(max(1, int(math.ceil(max(
            np.linalg.norm(seam0[a + 1] - seam0[a]),
            np.linalg.norm(seam1[a + 1] - seam1[a])) / seam_step)))
            for a in seam_segments) + int(bool(seam_segments))
        plan_coverage = {
            "schema": "stocking.authored_yarn_sample_support_plan.v1",
            "configuration": configuration,
            "query_sample_count": int(len(face)),
            "query_world_bounds_mm": [query_min.tolist(), query_max.tolist()],
            "world_halo_mm": float(halo_mm),
            "nearby_shell_face_count": int(len(candidates)),
            "projected_support_face_count": int(len(used_faces)),
            "material_cell_count": int(sum(len(loops) for loops in cells.values())),
            "minimum_q_to_X_stretch": float(min(used_sigma)),
            "potential_centerline_samples": int(potential_samples + seam_sample_upper),
            "synthetic_toe_seam_segments": len(seam_segments),
            "shell_orientation": shell_orientation,
            "creased_director_allowed": bool(allow_creased_director),
            "ray_side_rule": "area-weighted interpolated vertex director, including declared sharp closure creases",
            "global_tube_edge_nearband_completeness_certified": False,
        }
        if plan_only:
            return {"cells": cells, "seam_segments": seam_segments,
                    "coverage": plan_coverage}
        if self.representation == "strict_tube":
            potential_vertices = (potential_samples + seam_sample_upper) * p.tube_sides
        else:
            per_segment = (2 + 2 * self.capsule_cap_rings) * p.tube_sides + 2
            potential_vertices = (potential_samples + seam_sample_upper) * per_segment
        if potential_vertices > self.max_tube_vertices:
            raise ValueError(f"sample yarn window may need {potential_vertices:,} vertices, "
                             f"above max_tube_vertices={self.max_tube_vertices:,}; split sample tile")
        geometry, support, quality, sampled, unmapped, seam_added = self._build(
            configuration, cells, seam_segments=seam_segments)
        if not len(geometry.triangles):
            raise ValueError("sample yarn window contains no mapped tube triangles")
        coverage = {
            "schema": "stocking.authored_yarn_sample_window.v1",
            "configuration": configuration,
            "representation": self.representation,
            "query_sample_count": int(len(face)),
            "query_world_bounds_mm": [query_min.tolist(), query_max.tolist()],
            "world_halo_mm": float(halo_mm),
            "nearby_shell_face_count": int(len(candidates)),
            "projected_support_face_count": int(len(used_faces)),
            "material_cell_count": int(sum(len(loops) for loops in cells.values())),
            "minimum_q_to_X_stretch": float(min(used_sigma)),
            "selected_centerline_samples": sampled,
            "unmapped_boundary_centerline_samples": unmapped,
            "synthetic_toe_seam_segments_added": seam_added,
            "tube_vertex_count": int(len(geometry.vertices)),
            "tube_triangle_count": int(len(geometry.triangles)),
            "shell_orientation": shell_orientation,
            "tube_quality": quality,
            "no_double_F": True,
            "material_force_equilibrium_solved": False,
            "global_tube_edge_nearband_completeness_certified": False,
            "scope": "sample world box with per-face affine q inverse; nonlocal tube chords unbounded",
        }
        result = {"geometry": geometry, "coverage": coverage}
        if support is not None:
            result.update(origin_outside=support.origins_outside,
                          triangle_tangent_valid=support.triangle_tangent_valid,
                          capsule_support=support)
        return result

    def _segments_for_plan(self, configuration, plan):
        """Material-identical sampled centerline segments without tube faces."""
        parts_a, parts_b = [], []
        selected = mapped = 0
        ring_samples = self.phase.course_repeat_count * self.parameters.samples_per_loop
        samples_per_loop = self.parameters.samples_per_loop
        for row, loops in sorted(plan["cells"].items()):
            ordered_loops = sorted(loops)
            # Pin this row's requested values until its segments are assembled.
            # Filling missing cells can evict an older requested cell from the
            # bounded LRU; those evictions must affect reuse, never this query.
            row_data = {}
            for loop in ordered_loops:
                key = (configuration, row, loop)
                if key in self._segment_cell_cache:
                    row_data[loop] = self._segment_cell_cache[key]
                    self._segment_cell_cache.move_to_end(key)
            missing = [loop for loop in ordered_loops if loop not in row_data]
            if missing:
                chunk, _ = self.centerline_chunk(configuration, [row], missing)
                edge = chunk["edges"]
                xyz = chunk["positions_mm"]
                local_gid = chunk["global_sample_id"] - row * ring_samples
                starts = local_gid[edge[:, 0]] // samples_per_loop
                valid_gid = set(local_gid.tolist())
                for loop in missing:
                    keep = starts == loop
                    ids = (np.arange(loop * samples_per_loop,
                                     loop * samples_per_loop + samples_per_loop + 1)
                           % ring_samples)
                    valid_ids = ids[np.fromiter((int(gid) in valid_gid for gid in ids),
                                                dtype=bool, count=len(ids))]
                    key = (configuration, row, loop)
                    value = (
                        xyz[edge[keep, 0]].copy(), xyz[edge[keep, 1]].copy(),
                        valid_ids)
                    row_data[loop] = value
                    self._segment_cell_cache[key] = value
                    if len(self._segment_cell_cache) > self._segment_cell_cache_limit:
                        self._segment_cell_cache.popitem(last=False)
            selected_ids = []
            mapped_ids = []
            for loop in ordered_loops:
                key = (configuration, row, loop)
                segment_a, segment_b, valid_ids = row_data[loop]
                if key in self._segment_cell_cache:
                    self._segment_cell_cache.move_to_end(key)
                if len(segment_a):
                    parts_a.append(segment_a)
                    parts_b.append(segment_b)
                selected_ids.append((np.arange(loop * samples_per_loop,
                                               loop * samples_per_loop + samples_per_loop + 1)
                                     % ring_samples))
                mapped_ids.append(valid_ids)
            selected += len(np.unique(np.concatenate(selected_ids)))
            mapped += len(np.unique(np.concatenate(mapped_ids)))
        if plan["seam_segments"]:
            start = self.phase.seam_vertex_start
            seam = self._positions[configuration][start:]
            original = self._positions["reference"][start:]
            current = self._positions["current"][start:]
            step = self.parameters.period_mm[0] / self.parameters.samples_per_loop
            for segment in plan["seam_segments"]:
                longest = max(np.linalg.norm(original[segment + 1] - original[segment]),
                              np.linalg.norm(current[segment + 1] - current[segment]))
                count = max(1, int(math.ceil(longest / step)))
                w = np.linspace(0., 1., count + 1)
                xyz = seam[segment][None] * (1 - w[:, None]) + seam[segment + 1][None] * w[:, None]
                parts_a.append(xyz[:-1])
                parts_b.append(xyz[1:])
        if not parts_a:
            raise ValueError("sample support has no mapped centerline segments")
        a = np.concatenate(parts_a)
        b = np.concatenate(parts_b)
        length = np.linalg.norm(b - a, axis=1)
        if not np.isfinite(length).all() or np.any(length <= 1e-14):
            raise ValueError("sample support contains zero or nonfinite material segments")
        return a, b, {"selected_centerline_samples": int(selected),
                      "mapped_centerline_samples": int(mapped),
                      "unmapped_boundary_centerline_samples": int(selected - mapped),
                      "segment_count": int(len(a))}

    def sample_capsule_rays(self, configuration, faces, barycentric,
                            surface_mm, frames, ray_offset_mm, halo_mm):
        """Stream real material segments into analytic capsule first hits.

        No tube vertices or triangles are constructed.  The world AABB index
        is complete for the material segments selected by the local chart
        plan; completeness against omitted distant material chords remains
        explicitly unproved.
        """
        if self.representation != "capsule_union":
            raise ValueError("analytic capsule sampling requires representation='capsule_union'")
        if (not math.isfinite(ray_offset_mm) or ray_offset_mm <= 0
                or ray_offset_mm > halo_mm):
            raise ValueError("ray_offset_mm must be positive and at most halo_mm")
        from .analytic_yarn_queries import cast_capsule_segments

        start_time = time.perf_counter()
        surface = np.asarray(surface_mm, dtype=np.float64)
        basis = np.asarray(frames, dtype=np.float64)
        count = len(np.asarray(faces))
        if (surface.shape != (count, 3) or basis.shape != (count, 3, 3)
                or not np.isfinite(surface).all() or not np.isfinite(basis).all()):
            raise ValueError("sample_capsule_rays requires finite surface[N,3] and frames[N,3,3]")
        # An analytic capsule remains a defined radius-neighborhood of C(q)
        # at a sharp shell crease.  Its outward N comes from the capsule hit,
        # not from assuming the smoothed shell director agrees with every
        # incident face winding.  Default mesh/strict-tube queries keep that
        # hemisphere guard.  Singular directors/frames still fail explicitly.
        plan = self.for_samples(configuration, faces, barycentric, halo_mm,
                                plan_only=True, allow_creased_director=True)
        expected_surface = np.einsum("ni,nij->nj", np.asarray(barycentric, dtype=np.float64),
                                     self._positions[configuration][self.pair.triangles[np.asarray(faces)]])
        if np.max(np.abs(expected_surface - surface)) > 1e-7:
            raise ValueError("sample surface positions do not match the supplied garment pair/configuration")
        plan_seconds = time.perf_counter() - start_time
        segment_a, segment_b, network_coverage = self._segments_for_plan(configuration, plan)
        network_seconds = time.perf_counter() - start_time - plan_seconds
        radius = self.parameters.R * self.parameters.scale_mm
        segment_min = np.minimum(segment_a, segment_b)
        segment_max = np.maximum(segment_a, segment_b)
        shell_normal = basis[:, :, 2]
        origins = np.vstack((surface + ray_offset_mm * shell_normal,
                             surface - ray_offset_mm * shell_normal))
        directions = np.vstack((-shell_normal, shell_normal))
        ends = origins + 2 * ray_offset_mm * directions
        candidate_offsets = [0]
        candidate_parts = []
        # A tiny query tile is cheaper to filter with vectorized AABB tests
        # than to rebuild a Python grid for every overlapping material plan.
        direct_filter = 2 * count * len(segment_a) <= 2_000_000
        index = None if direct_filter else _FaceAABBIndex(
            segment_min, segment_max, grid_mm=max(2., 2 * ray_offset_mm))
        for ray_start, ray_end in zip(origins, ends):
            query_min = np.minimum(ray_start, ray_end) - radius
            query_max = np.maximum(ray_start, ray_end) + radius
            if direct_filter:
                candidates = np.flatnonzero(np.all((segment_max >= query_min)
                                                    & (segment_min <= query_max), axis=1)).astype(np.int32)
            else:
                candidates = index.query(query_min, query_max)
            candidate_parts.append(candidates)
            candidate_offsets.append(candidate_offsets[-1] + len(candidates))
        candidate_indices = np.concatenate(candidate_parts).astype(np.int32)
        spatial_seconds = time.perf_counter() - start_time - plan_seconds - network_seconds
        cast = cast_capsule_segments(origins, directions, 2 * ray_offset_mm,
                                     segment_a, segment_b, radius,
                                     candidate_offsets=np.asarray(candidate_offsets, dtype=np.int64),
                                     candidate_indices=candidate_indices,
                                     max_pairs_per_batch=32000)
        cast_seconds = time.perf_counter() - start_time - plan_seconds - network_seconds - spatial_seconds
        features = {}
        for side, slc in (("front", slice(0, count)), ("back", slice(count, 2 * count))):
            valid = cast["valid"][slc]
            displacement = cast["P_mm"][slc] - surface
            local_position = np.einsum("nji,nj->ni", basis, displacement)
            local_normal = np.einsum("nji,nj->ni", basis, cast["N"][slc])
            local_tangent = np.einsum("nji,nj->ni", basis, cast["T"][slc])
            local_position[~valid] = 0.
            local_normal[~valid] = 0.
            local_tangent[~valid] = 0.
            features[side] = {
                "P_offset_mm": local_position,
                "N_local": local_normal,
                "T_local": local_tangent,
                "valid": valid,
                "ray_hit": cast["hit"][slc],
                "ray_origin_outside": ~cast["origin_inside"][slc],
                "ray_tangent_invalid": (cast["pole_tangent_invalid"][slc]
                                        | cast["tangent_ambiguous"][slc]),
                "ray_tangent_pole": cast["pole_tangent_invalid"][slc],
                "ray_tangent_ambiguous": cast["tangent_ambiguous"][slc],
            }
        coverage = dict(plan["coverage"],
                        schema="stocking.authored_yarn_analytic_capsule_query.v1",
                        representation="analytic_capsule_union",
                        geometry_available=True,
                        **network_coverage,
                        local_segment_AABB_index_complete=True,
                        local_segment_candidate_filter=("vectorized_AABB" if direct_filter else "grid_AABB"),
                        candidate_pair_count=int(cast["candidate_pair_count"]),
                        maximum_candidate_segments_per_ray=int(np.diff(candidate_offsets).max()),
                        front_valid_count=int(features["front"]["valid"].sum()),
                        back_valid_count=int(features["back"]["valid"].sum()),
                        front_inside_origin_count=int(cast["origin_inside"][:count].sum()),
                        back_inside_origin_count=int(cast["origin_inside"][count:].sum()),
                        front_tangent_invalid_count=int(features["front"]["ray_tangent_invalid"].sum()),
                        back_tangent_invalid_count=int(features["back"]["ray_tangent_invalid"].sum()),
                        computation_seconds={"plan": plan_seconds, "network": network_seconds,
                                             "spatial_index": spatial_seconds,
                                             "analytic_cast": cast_seconds,
                                             "total": time.perf_counter() - start_time})
        return {"features": features, "coverage": coverage}

    def _centerline_points(self, row, global_sample_ids, configuration):
        p = self.parameters
        t = TAU * global_sample_ids / p.samples_per_loop
        q = np.column_stack((p.scale_mm * (t + p.a * np.sin(2 * t)),
                             row * p.period_mm[1] + p.h * p.scale_mm * np.cos(t)))
        q[:, 0] = np.mod(q[:, 0], self.phase.course_width_mm)
        face, bary = self.locator.locate(q)
        valid = face >= 0
        centers = np.zeros((len(q), 3), dtype=np.float64)
        normals = np.zeros_like(centers)
        if np.any(valid):
            tris = self.pair.triangles[face[valid]]
            b = bary[valid]
            shell = np.einsum("ni,nij->nj", b, self._positions[configuration][tris])
            normal = _unit(np.einsum("ni,nij->nj", b, self._normals[configuration][tris]))
            h = p.d * p.scale_mm * np.cos(2 * t[valid])
            centers[valid] = shell + h[:, None] * normal
            normals[valid] = normal
        return centers, normals, valid, face, bary

    def _seam_segments(self, configuration, requested, halo_mm):
        p = self.parameters
        positions = self._positions[configuration]
        seam = positions[self.phase.seam_vertex_start:]
        segment_min = np.minimum(seam[:-1], seam[1:])
        segment_max = np.maximum(seam[:-1], seam[1:])
        face_min, face_max = self._bounds[configuration]
        reach = halo_mm + abs(p.d * p.scale_mm) + p.R * p.scale_mm
        selected = set()
        for face in requested:
            near = np.all((segment_max >= face_min[face] - reach)
                          & (segment_min <= face_max[face] + reach), axis=1)
            selected.update(np.flatnonzero(near).tolist())
        return sorted(selected)

    def _build(self, configuration, cells, *, seam_segments):
        p = self.parameters
        samples_per_ring = self.phase.course_repeat_count * p.samples_per_loop
        centers_all, normal_all, edges_all = [], [], []
        count_unmapped = 0
        count_selected = 0
        for row, loops in sorted(cells.items()):
            ids = set()
            edge_ids = []
            for loop in sorted(loops):
                start = loop * p.samples_per_loop
                for i in range(p.samples_per_loop):
                    a = start + i
                    b = (a + 1) % samples_per_ring
                    ids.add(a)
                    ids.add(b)
                    edge_ids.append((a, b))
            if not ids:
                continue
            ordered = np.asarray(sorted(ids), dtype=np.int64)
            c, n, valid, _, _ = self._centerline_points(row, ordered, configuration)
            count_selected += len(ordered)
            count_unmapped += int((~valid).sum())
            lookup = {int(gid): index for index, gid in enumerate(ordered)}
            keep = np.flatnonzero(valid)
            compact = np.full(len(ordered), -1, dtype=np.int64)
            compact[keep] = np.arange(len(keep)) + sum(len(part) for part in centers_all)
            centers_all.append(c[keep])
            normal_all.append(n[keep])
            for a, b in edge_ids:
                ia, ib = compact[lookup[a]], compact[lookup[b]]
                if ia >= 0 and ib >= 0:
                    edges_all.append((int(ia), int(ib)))
        if seam_segments:
            seam = self._positions[configuration][self.phase.seam_vertex_start:]
            seam_normals = self._normals[configuration][self.phase.seam_vertex_start:]
            # Sample the distributed closure at the same maximum centerline
            # step used for regular courses, retaining each physical seam ID.
            step = p.period_mm[0] / p.samples_per_loop
            source_seam = self._positions["reference"][self.phase.seam_vertex_start:]
            target_seam = self._positions["current"][self.phase.seam_vertex_start:]
            parts, normals, stitch_edges = [], [], []
            previous = None
            output_count = 0
            for a in seam_segments:
                longest = max(np.linalg.norm(source_seam[a + 1] - source_seam[a]),
                              np.linalg.norm(target_seam[a + 1] - target_seam[a]))
                count = max(1, int(math.ceil(longest / step)))
                continued = previous is not None and a == previous + 1
                w = np.arange(1 if continued else 0, count + 1, dtype=np.float64) / count
                start = output_count
                parts.append(seam[a][None, :] * (1 - w[:, None]) + seam[a + 1][None, :] * w[:, None])
                normals.append(_unit(seam_normals[a][None, :] * (1 - w[:, None])
                                     + seam_normals[a + 1][None, :] * w[:, None]))
                stitch_edges.extend((start + i, start + i + 1) for i in range(len(w) - 1))
                if continued:
                    stitch_edges.append((start - 1, start))
                output_count += len(w)
                previous = a
            seam_centers = np.concatenate(parts)
            seam_normal = np.concatenate(normals)
            offset = sum(len(part) for part in centers_all)
            centers_all.append(seam_centers)
            normal_all.append(seam_normal)
            edges_all.extend((offset + a, offset + b) for a, b in stitch_edges)
        centers = np.concatenate(centers_all) if centers_all else np.empty((0, 3))
        normals = np.concatenate(normal_all) if normal_all else np.empty((0, 3))
        if self.representation == "strict_tube":
            geometry, quality = _tube_mesh(centers, normals, edges_all,
                                           p.R * p.scale_mm, p.tube_sides)
            support = None
        else:
            geometry, support, quality = _capsule_union_mesh(
                centers, edges_all, p.R * p.scale_mm, p.tube_sides,
                self.capsule_cap_rings, self.max_tube_vertices)
        return geometry, support, quality, count_selected, count_unmapped, len(seam_segments)

    def __call__(self, configuration, face_indices, halo_mm):
        if configuration not in self._positions:
            raise ValueError("configuration must be 'reference' or 'current'")
        requested = np.unique(np.asarray(face_indices, dtype=np.int64))
        if requested.ndim != 1 or len(requested) == 0 or np.any((requested < 0) | (requested >= len(self.pair.triangles))):
            raise ValueError("face_indices must be nonempty valid face indices")
        if not math.isfinite(halo_mm) or halo_mm <= 0:
            raise ValueError("halo_mm must be a finite positive world-space reach")
        candidate_faces, min_requested_stretch = self._candidate_faces(configuration, requested, halo_mm)
        shell_orientation = self._orientation_metrics(configuration, candidate_faces)
        if shell_orientation["nonpositive_face_orientation_count"]:
            raise ValueError("requested yarn patch includes shell faces opposing their smoothed normals")
        cells, min_candidate_stretch = self._candidate_cells(candidate_faces, configuration, halo_mm)
        potential_samples = sum(len(loops) * self.parameters.samples_per_loop + 1 for loops in cells.values())
        seam_segments = self._seam_segments(configuration, requested, halo_mm)
        seam0 = self._positions["reference"][self.phase.seam_vertex_start:]
        seam1 = self._positions["current"][self.phase.seam_vertex_start:]
        seam_step = self.parameters.period_mm[0] / self.parameters.samples_per_loop
        seam_sample_upper = sum(max(1, int(math.ceil(max(
            np.linalg.norm(seam0[a + 1] - seam0[a]),
            np.linalg.norm(seam1[a + 1] - seam1[a])) / seam_step)))
            for a in seam_segments) + int(bool(seam_segments))
        if self.representation == "strict_tube":
            potential_vertices = (potential_samples + seam_sample_upper) * self.parameters.tube_sides
        else:
            per_segment = ((2 + 2 * self.capsule_cap_rings) * self.parameters.tube_sides + 2)
            potential_vertices = (potential_samples + seam_sample_upper) * per_segment
        if potential_vertices > self.max_tube_vertices:
            raise ValueError(f"requested yarn patch may need {potential_vertices:,} "
                             f"tube vertices, above max_tube_vertices={self.max_tube_vertices:,}; split face chunk")
        geometry, support, quality, sampled, unmapped, seam_added = self._build(
            configuration, cells, seam_segments=seam_segments)
        if not len(geometry.triangles):
            raise ValueError("requested face patch contains no mapped yarn tube triangles")
        coverage = {
            "schema": "stocking.authored_yarn_patch_coverage.v1",
            "configuration": configuration,
            "requested_face_count": int(len(requested)),
            "candidate_face_count": int(len(candidate_faces)),
            "candidate_material_cell_count": int(sum(len(loops) for loops in cells.values())),
            "minimum_requested_q_to_X_stretch": min_requested_stretch,
            "minimum_candidate_q_to_X_stretch": min_candidate_stretch,
            "world_halo_mm": float(halo_mm),
            "selected_centerline_samples": sampled,
            "unmapped_boundary_centerline_samples": unmapped,
            "tube_vertex_count": int(len(geometry.vertices)),
            "tube_triangle_count": int(len(geometry.triangles)),
            "synthetic_toe_seam_segments_added": seam_added,
            "material_phase_global": True,
            "representation": self.representation,
            "geometry_built": True,
            "material_force_equilibrium_solved": False,
            "no_double_F": True,
            "tube_quality": quality,
            "shell_orientation": shell_orientation,
            "scope": "local actual tube geometry; global nearband completeness not certified",
        }
        if support is not None:
            coverage.update(capsule_origin_exterior_required=True,
                            capsule_pole_fan_tangent_undefined=True,
                            capsule_overlap_faces_unmerged=True)
        self._last_coverage = coverage
        result = {"geometry": geometry, "coverage": coverage}
        if support is not None:
            result.update(origin_outside=support.origins_outside,
                          triangle_tangent_valid=support.triangle_tangent_valid,
                          capsule_support=support)
        return result


def make_authored_yarn_provider(pair, parameters=None, **kwargs):
    return AuthoredYarnProvider(pair, parameters, **kwargs)
