"""Local geometric orientation repair on a fixed target surface.

Signed triangle areas give a geometric descent direction. There is no yarn
energy, mass, force, pressure, material law or time integration here. A repair
is an additional choice of material correspondence; target bindings are
updated explicitly and diagnostics are always returned, including failure.
"""
import numpy as np


def _normals(points, faces):
    tri = points[faces]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 1e-20):
        raise ValueError("orientation target contains a degenerate face")
    return cross/lengths[:, None]


def _weights(point, tri):
    e = np.stack((tri[1]-tri[0], tri[2]-tri[0]), axis=1)
    uv = np.linalg.solve(e.T@e, e.T@(point-tri[0]))
    result = np.array([1.-uv.sum(), uv[0], uv[1]])
    result = np.maximum(result, 0.)
    return result/result.sum()


def repair_orientation_on_surface(positions_mm, triangles, target_positions_mm,
                                  target_triangles, target_face_index,
                                  target_barycentric, allowed_target_faces, *,
                                  fixed_vertices=(), max_iterations=160,
                                  maximum_step_mm=.8, progress=None):
    """Repair locally inverted faces, recording the new tri/bary binding.

    Each move is limited and reprojected to the same declared target component.
    This cannot certify global one-to-one mapping or eliminate contact. Caller
    must run independent orientation/coverage and stretch checks afterward.
    The input arrays are never modified. No fallback result is marked passed.
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    points = np.array(positions_mm, dtype=np.float64, copy=True)
    faces = np.asarray(triangles, dtype=np.int64)
    target = np.asarray(target_positions_mm, dtype=np.float64)
    target_faces = np.asarray(target_triangles, dtype=np.int64)
    allowed = np.asarray(allowed_target_faces, dtype=np.int64)
    indices = np.array(target_face_index, dtype=np.int64, copy=True)
    bary = np.array(target_barycentric, dtype=np.float64, copy=True)
    if len(allowed) == 0 or not np.all(np.isin(indices, allowed)):
        raise ValueError("repair binding must stay in its allowed target component")
    fixed = np.zeros(len(points), dtype=bool)
    fixed[np.asarray(fixed_vertices, dtype=np.int64)] = True
    normals = _normals(target, target_faces)
    tree = BVHTree.FromPolygons([Vector(p) for p in target], target_faces[allowed].tolist(), all_triangles=True)
    initial = points.copy()

    def evaluate(values, bound):
        tri = values[faces]
        u, v = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        n = normals[bound][faces].sum(axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
        area = np.einsum("ij,ij->i", np.cross(u, v), n)
        length2 = np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)
        # A small positive angle margin avoids declaring zero-area faces fixed.
        floor = np.maximum(length2*1e-4, 1e-10)
        bad = area <= floor
        score = float(np.sum(np.maximum(0., floor-area)/np.maximum(length2, 1e-12)))
        return u, v, n, area, floor, bad, score

    history = []
    state = evaluate(points, indices)
    history.append({"iteration": 0, "orientation_bad_faces": int(state[5].sum()), "score": state[6]})
    for iteration in range(1, int(max_iterations)+1):
        u, v, n, area, floor, bad, score = state
        if not np.any(bad):
            break
        selected = np.flatnonzero(bad)
        g1, g2 = np.cross(v[selected], n[selected]), np.cross(n[selected], u[selected])
        gradient = np.stack((-g1-g2, g1, g2), axis=1)
        movable = ~fixed[faces[selected]]
        gradient *= movable[..., None]
        denominator = np.maximum(np.sum(gradient*gradient, axis=(1, 2)), 1e-20)
        corrections = gradient * ((floor[selected]-area[selected])/denominator)[:, None, None]
        displacement = np.zeros_like(points)
        count = np.zeros(len(points))
        for corner in range(3):
            np.add.at(displacement, faces[selected, corner], corrections[:, corner])
            np.add.at(count, faces[selected, corner], movable[:, corner])
        active = np.flatnonzero((count > 0) & ~fixed)
        if len(active) == 0:
            break
        displacement[active] /= count[active, None]
        vertex_n = normals[indices[active]]
        displacement[active] -= np.einsum("ij,ij->i", displacement[active], vertex_n)[:, None]*vertex_n
        length = np.linalg.norm(displacement[active], axis=1)
        displacement[active] *= np.minimum(1., maximum_step_mm/np.maximum(length, 1e-20))[:, None]
        accepted = False
        for factor in (.8, .4, .2, .1, .05):
            candidate, candidate_indices, candidate_bary = points.copy(), indices.copy(), bary.copy()
            for index in active:
                hit, _, local_face, _ = tree.find_nearest(Vector(points[index]+factor*displacement[index]))
                if hit is None:
                    raise ValueError("orientation repair projection missed the declared surface")
                face = int(allowed[local_face])
                weights = _weights(np.asarray(hit), target[target_faces[face]])
                candidate[index] = weights@target[target_faces[face]]
                candidate_indices[index], candidate_bary[index] = face, weights
            candidate_state = evaluate(candidate, candidate_indices)
            if candidate_state[6] < score-1e-12:
                points, indices, bary, state = candidate, candidate_indices, candidate_bary, candidate_state
                accepted = True
                break
        history.append({"iteration": iteration, "orientation_bad_faces": int(state[5].sum()),
                        "score": state[6], "accepted": accepted})
        if progress and (iteration % 10 == 0 or not accepted):
            progress(history[-1])
        if not accepted:
            break
    report = {"method": "local signed-area geometric descent with same-component surface projection",
              "mechanical_equilibrium": False, "history": history,
              "remaining_orientation_bad_faces": int(state[5].sum()),
              "maximum_material_point_reassignment_mm": float(np.linalg.norm(points-initial, axis=1).max()),
              "scope": "local orientation diagnostic; no global bijection, self-intersection or calibration certification"}
    return points, indices, bary, report
