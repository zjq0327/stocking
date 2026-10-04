"""Rebuild circular tubes from already embedded open yarn centerlines (mm).

No deformation of an old tube: radius is measured in the current physical
space. This is a geometry operation, not a curved-contact equilibrium solver.
Open endpoints must be outside the trusted bake region.
"""
import math
import numpy as np

from .geometry import Geometry
from .deformed_geometry import _transport


def generate_curve_geometry(curves_mm, radius_mm, tube_sides=12, *, frame_reference_rotation=None):
    if isinstance(radius_mm, (bool, np.bool_)) or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be finite and positive")
    if isinstance(tube_sides, (bool, np.bool_)) or not isinstance(tube_sides, (int, np.integer)) or tube_sides < 6:
        raise ValueError("tube_sides must be an integer >= 6")
    vertices, normals, tangents, triangles = [], [], [], []
    rotation = np.eye(3) if frame_reference_rotation is None else np.asarray(frame_reference_rotation,dtype=float)
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-10) or not np.isclose(np.linalg.det(rotation),1,atol=1e-10):
        raise ValueError("frame_reference_rotation must be a proper orthonormal orientation")
    offset = 0
    phi = np.arange(tube_sides) * 2 * math.pi / tube_sides
    for points in curves_mm:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Each curve must be finite (N,3), N >= 3")
        edges = np.diff(points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        scale = max(float(np.max(lengths)), float(radius_mm))
        if np.any(lengths <= scale * 1e-12):
            raise ValueError("Repeated or indistinguishable adjacent yarn nodes")
        edge_t = edges / lengths[:, None]
        cosine = np.einsum("ij,ij->i", edge_t[:-1], edge_t[1:])
        if np.any(cosine <= -1 + 1e-10):
            raise ValueError("Reversing centerline cusp")
        curvature = 2 * np.sqrt(np.maximum(0, (1-cosine) / (1+cosine))) / (.5*(lengths[:-1]+lengths[1:]))
        if np.any(radius_mm * curvature >= 1 - 1e-6):
            raise ValueError("Tube radius exceeds the local centerline curvature bound")
        t = np.vstack((edge_t[0], edge_t[:-1]+edge_t[1:], edge_t[-1]))
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        # Anchor polygon phase to the material chart. Choosing a world axis
        # afresh would spin a finite-sided circular tube after rigid motion.
        axis = rotation[:,np.argmin(np.abs(t[0] @ rotation))]
        u = np.empty_like(points)
        u[0] = axis - np.dot(axis, t[0]) * t[0]
        u[0] /= np.linalg.norm(u[0])
        for i in range(1, len(points)):
            u[i] = _transport(u[i-1], t[i-1], t[i])
        v = np.cross(t, u)
        n = u[:, None, :] * np.cos(phi)[None, :, None] + v[:, None, :] * np.sin(phi)[None, :, None]
        p = points[:, None, :] + radius_mm * n
        starts = np.arange(len(points)-1)[:, None] * tube_sides
        sides = np.arange(tube_sides)[None, :]
        a, b = starts+sides, starts+(sides+1)%tube_sides
        c, d = a+tube_sides, b+tube_sides
        tri = np.stack((np.stack((a,b,c),-1),np.stack((b,d,c),-1)), -2).reshape(-1,3)
        flat_p, flat_n = p.reshape(-1,3), n.reshape(-1,3)
        cross = np.cross(flat_p[tri[:,1]]-flat_p[tri[:,0]], flat_p[tri[:,2]]-flat_p[tri[:,0]])
        if np.any(np.einsum("ij,ij->i", cross, flat_n[tri].sum(axis=1)) <= 0):
            raise ValueError("Rebuilt tube has degenerate or inverted triangles")
        vertices.append(flat_p)
        normals.append(flat_n)
        tangents.append(np.broadcast_to(t[:,None,:], n.shape).reshape(-1,3))
        triangles.append(tri+offset)
        offset += len(flat_p)
    if not vertices:
        raise ValueError("At least one centerline is required")
    tri = np.concatenate(triangles).astype(np.int32)
    return Geometry(np.concatenate(vertices), tri, np.concatenate(normals),
                    np.concatenate(tangents), np.ones(len(tri), dtype=np.uint32))
