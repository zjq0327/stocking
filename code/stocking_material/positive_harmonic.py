"""Positive graph harmonic parameterization of a triangulated disk.

This chooses a geometric correspondence to a supplied convex boundary. It is
not a fabric force equilibrium. NumPy conjugate gradients solves the two
Dirichlet coordinate systems without SciPy or source mesh modification.
"""
import numpy as np


def positive_harmonic_disk(points, triangles, boundary_vertices, boundary_xy, *,
                           tolerance=2e-11, max_iterations=5000,
                           weight_exponent=1., progress=None):
    """Return (xy[N,2], report) for a manifold disk and convex boundary.

    Positive symmetric edge weights are inverse physical edge length raised
    to ``weight_exponent``. Input boundary order must follow physical edges.
    The caller controls material boundary identity and disk orientation.
    Finite coordinates and convergence are checked; tiny or folded parameter
    triangles are reported separately, not hidden by a nearest query.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles)
    boundary = np.asarray(boundary_vertices)
    q = np.asarray(boundary_xy, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] not in (2, 3) or not np.isfinite(points).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu"
            or triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("harmonic disk needs finite points and valid integer triangles")
    if (boundary.ndim != 1 or boundary.dtype.kind not in "iu" or len(boundary) < 3
            or len(np.unique(boundary)) != len(boundary) or np.any(boundary < 0)
            or np.any(boundary >= len(points)) or q.shape != (len(boundary), 2)
            or not np.isfinite(q).all()):
        raise ValueError("invalid ordered harmonic disk boundary")
    if tolerance <= 0 or max_iterations < 1 or not 0 <= weight_exponent <= 2:
        raise ValueError("invalid harmonic iteration/weight configuration")
    all_edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    unique, counts = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2) or len(points)-len(unique)+len(triangles) != 1:
        raise ValueError("harmonic domain must be a manifold disk")
    boundary_edges = unique[counts == 1]
    declared_edges = np.sort(np.stack((boundary, np.roll(boundary, -1)), axis=1), axis=1)
    ordered_edges = lambda e: e[np.lexsort((e[:, 1], e[:, 0]))]
    if not np.array_equal(ordered_edges(boundary_edges), ordered_edges(declared_edges)):
        raise ValueError("declared harmonic boundary does not match the disk's only boundary loop")
    qe = np.roll(q, -1, axis=0)-q
    next_qe = np.roll(qe, -1, axis=0)
    turns = qe[:, 0]*next_qe[:, 1]-qe[:, 1]*next_qe[:, 0]
    if not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError("harmonic boundary must be strictly convex in the supplied order")
    length = np.linalg.norm(points[unique[:, 1]]-points[unique[:, 0]], axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("harmonic mesh has indistinguishable edge endpoints")
    weight = length**(-weight_exponent)
    src, dst = np.r_[unique[:, 0], unique[:, 1]], np.r_[unique[:, 1], unique[:, 0]]
    weights = np.r_[weight, weight]
    degree = np.bincount(src, weights=weights, minlength=len(points))
    fixed = np.zeros(len(points), dtype=bool)
    fixed[boundary] = True
    interior = np.flatnonzero(~fixed)
    lookup = np.full(len(points), -1, dtype=np.int64)
    lookup[interior] = np.arange(len(interior))
    xy = np.zeros((len(points), 2), dtype=np.float64)
    xy[boundary] = q
    interior_src = lookup[src]
    interior_dst = lookup[dst]
    within = (interior_src >= 0) & (interior_dst >= 0)
    inner_src, inner_dst, inner_weights = interior_src[within], interior_dst[within], weights[within]
    to_boundary = (interior_src >= 0) & (interior_dst < 0)
    rhs = np.stack([np.bincount(interior_src[to_boundary],
        weights=weights[to_boundary]*xy[dst[to_boundary], column], minlength=len(interior))
        for column in range(2)], axis=1)
    d = degree[interior]
    if np.any(d <= 0):
        raise ValueError("harmonic mesh has an isolated interior vertex")

    def multiply(values):
        neighbor_sum = np.stack([np.bincount(inner_src,
            weights=inner_weights*values[inner_dst, column], minlength=len(interior))
            for column in range(2)], axis=1)
        return d[:, None]*values-neighbor_sum

    x = np.zeros_like(rhs)
    residual = rhs.copy()
    z = residual/d[:, None]
    direction = z.copy()
    dot = np.sum(residual*z, axis=0)
    scale = np.maximum(np.linalg.norm(rhs, axis=0), 1.)
    relative = np.linalg.norm(residual, axis=0)/scale
    history = []
    iteration = 0
    for iteration in range(1, int(max_iterations)+1):
        if np.all(relative <= tolerance):
            break
        active = relative > tolerance
        product = multiply(direction)
        denominator = np.sum(direction*product, axis=0)
        if np.any(denominator[active] <= 0):
            raise ValueError("positive harmonic system lost positive definiteness")
        alpha = np.divide(dot, denominator, out=np.zeros(2), where=active)
        x += direction*alpha
        residual -= product*alpha
        relative = np.linalg.norm(residual, axis=0)/scale
        z = residual/d[:, None]
        new_dot = np.sum(residual*z, axis=0)
        beta = np.divide(new_dot, dot, out=np.zeros(2), where=active & (np.abs(dot) > 1e-300))
        direction = z+direction*beta
        direction[:, relative <= tolerance] = 0.
        dot = new_dot
        if iteration % 100 == 0:
            entry = {"iteration": iteration, "relative_residual": relative.tolist()}
            history.append(entry)
            if progress:
                progress(entry)
    if not np.all(relative <= tolerance):
        raise ValueError("positive harmonic coordinates did not converge: " + str(relative.tolist()))
    xy[interior] = x
    uv = xy[triangles]
    e1, e2 = uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]
    area = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    sign = 1. if np.median(area) > 0 else -1.
    report = {"method": "positive inverse-edge-length graph harmonic disk",
              "weight_exponent": weight_exponent, "iterations": iteration,
              "relative_residual": relative.tolist(), "history": history,
              "parameter_orientation_sign": sign,
              "nonpositive_parameter_faces": int(np.sum(sign*area <= 0)),
              "minimum_absolute_parameter_area2": float(np.abs(area).min()),
              "boundary_vertex_count": len(boundary), "interior_vertex_count": len(interior),
              "mechanical_equilibrium": False}
    if not np.isfinite(xy).all() or report["nonpositive_parameter_faces"]:
        raise ValueError("harmonic disk is folded or degenerate after numerical solve: " + str(report))
    return xy, report
