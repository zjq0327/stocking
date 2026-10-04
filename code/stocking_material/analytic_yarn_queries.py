"""Bounded-memory analytic ray queries of real polyline yarn capsules.

Each centerline edge is a radius-R capsule: finite cylinder plus two
hemispherical caps. This module queries the geometric union directly and
does not allocate a tube mesh or replace the authored centerline with a
periodic atlas. Candidate selection may be supplied as CSR ray-to-segment
indices; completeness of those candidates belongs to the caller.
"""

import numpy as np


_CYLINDER = 0
_CAP_A = 1
_CAP_B = 2


def _pair_first_hit(origin, direction, limit, a, b, radius_mm, pole_tolerance):
    """Vectorized first hit and origin membership for aligned ray/segment pairs."""
    count = len(origin)
    edge = b-a
    edge_length = np.linalg.norm(edge, axis=1)
    if np.any(edge_length <= 1e-12):
        raise ValueError("capsule centerline contains a zero-length segment")
    axis = edge/edge_length[:, None]
    offset = origin-a
    axial = np.einsum("ij,ij->i", offset, axis)
    projected = a+np.clip(axial, 0., edge_length)[:, None]*axis
    origin_distance2 = np.einsum("ij,ij->i", origin-projected, origin-projected)
    boundary_tolerance = max(1e-9, radius_mm*1e-8)
    inside = origin_distance2 <= (radius_mm+boundary_tolerance)**2

    # Cylinder wall: solve |(O-A)+tD projected perpendicular to axis|²=R².
    along_ray = np.einsum("ij,ij->i", direction, axis)
    radial_origin = offset-axial[:, None]*axis
    radial_ray = direction-along_ray[:, None]*axis
    qa = np.einsum("ij,ij->i", radial_ray, radial_ray)
    qb = 2*np.einsum("ij,ij->i", radial_origin, radial_ray)
    qc = np.einsum("ij,ij->i", radial_origin, radial_origin)-radius_mm**2
    discriminant = qb*qb-4*qa*qc
    safe = (qa > 1e-14) & (discriminant >= -1e-12)
    root = np.sqrt(np.maximum(discriminant, 0.))
    denominator = np.where(safe, 2*qa, 1.)
    cylinder_times = np.column_stack(((-qb-root)/denominator, (-qb+root)/denominator))
    cylinder_axis = axial[:, None]+cylinder_times*along_ray[:, None]
    cylinder_ok = safe[:, None] & (cylinder_axis >= -1e-9) & (
        cylinder_axis <= edge_length[:, None]+1e-9)

    def cap_roots(center, first):
        relative = origin-center
        qb_sphere = np.einsum("ij,ij->i", relative, direction)
        qc_sphere = np.einsum("ij,ij->i", relative, relative)-radius_mm**2
        disc = qb_sphere**2-qc_sphere
        roots = np.column_stack((-qb_sphere-np.sqrt(np.maximum(disc, 0.)),
                                 -qb_sphere+np.sqrt(np.maximum(disc, 0.))))
        axis_coordinate = axial[:, None]+roots*along_ray[:, None]
        if first:
            cap_ok = axis_coordinate <= 1e-9
        else:
            cap_ok = axis_coordinate >= edge_length[:, None]-1e-9
        return roots, (disc >= -1e-12)[:, None] & cap_ok

    cap_a_times, cap_a_ok = cap_roots(a, True)
    cap_b_times, cap_b_ok = cap_roots(b, False)
    times = np.column_stack((cylinder_times, cap_a_times, cap_b_times))
    allowed = np.column_stack((cylinder_ok, cap_a_ok, cap_b_ok))
    allowed &= np.isfinite(times) & (times >= 0.) & (times <= limit[:, None]+1e-9)
    times = np.where(allowed, times, np.inf)
    piece_root = np.argmin(times, axis=1)
    hit_t = times[np.arange(count), piece_root]
    hit = np.isfinite(hit_t)
    safe_t = np.where(hit, hit_t, 0.)
    point = origin+safe_t[:, None]*direction
    piece = np.where(piece_root < 2, _CYLINDER,
                     np.where(piece_root < 4, _CAP_A, _CAP_B)).astype(np.int8)
    center = np.where((piece == _CAP_A)[:, None], a,
                      np.where((piece == _CAP_B)[:, None], b,
                               a+np.clip(axial+safe_t*along_ray, 0., edge_length)[:, None]*axis))
    normal = point-center
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
    tangent = axis.copy()
    sphere = piece != _CYLINDER
    if np.any(sphere):
        projected_axis = axis[sphere]-np.einsum("ij,ij->i", axis[sphere], normal[sphere])[:, None]*normal[sphere]
        projected_length = np.linalg.norm(projected_axis, axis=1)
        tangent[sphere] = projected_axis/np.maximum(projected_length[:, None], 1e-30)
    pole = hit & sphere & (np.linalg.norm(tangent, axis=1) < pole_tolerance)
    # The normalized projection above may be numerically nonzero at a pole;
    # detect the unnormalized magnitude directly as well.
    if np.any(hit & sphere):
        cross_length = np.linalg.norm(np.cross(axis, normal), axis=1)
        pole |= hit & sphere & (cross_length <= pole_tolerance)
    tangent[pole] = 0.
    point[~hit], normal[~hit], tangent[~hit] = 0., 0., 0.
    return hit_t, point, normal, tangent, piece, hit, inside, pole


def _pair_batches(ray_count, segment_count, offsets, indices, maximum):
    if offsets is None:
        ray_block = max(1, maximum//max(1, min(segment_count, maximum)))
        segment_block = min(segment_count, maximum)
        for first_ray in range(0, ray_count, ray_block):
            rays = np.arange(first_ray, min(first_ray+ray_block, ray_count), dtype=np.int32)
            for first_segment in range(0, segment_count, segment_block):
                segments = np.arange(first_segment, min(first_segment+segment_block, segment_count),
                                     dtype=np.int32)
                yield np.repeat(rays, len(segments)), np.tile(segments, len(rays))
    else:
        for first in range(0, len(indices), maximum):
            last = min(first+maximum, len(indices))
            pair_index = np.arange(first, last, dtype=np.int64)
            rays = np.searchsorted(offsets, pair_index, side="right")-1
            yield rays.astype(np.int32), indices[first:last]


def cast_capsule_segments(origins_mm, directions, max_distance_mm, segment_a_mm,
                          segment_b_mm, radius_mm, *, candidate_offsets=None,
                          candidate_indices=None, max_pairs_per_batch=65536,
                          pole_tolerance=1e-6):
    """Cast rays against the union of supplied finite centerline capsules.

    Directions are normalized internally, so all ``t_mm`` values are physical
    millimetres. If either candidate CSR array is supplied, both are needed:
    offsets[N+1] and segment indices[K] encode the local segments for each ray.
    Candidate indices should be unique within each ray. A missing segment can
    invalidate a coverage claim, even when all returned algebra is exact.

    ``hit`` means a first boundary hit from outside the union. ``valid`` also
    requires a unique PNT tangent. On a hemispherical cap the tangent is the
    centerline axis projected to the sphere tangent plane; at its exact pole
    it has no unique direction and is marked invalid. Equal-distance hits on
    a smooth shared endpoint use the projected average tangent; a sharp joint
    or disagreement in surface normal is marked tangent-ambiguous.
    """
    origins = np.asarray(origins_mm, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    first = np.asarray(segment_a_mm, dtype=np.float64)
    second = np.asarray(segment_b_mm, dtype=np.float64)
    if (origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape
            or first.ndim != 2 or first.shape[1] != 3 or second.shape != first.shape
            or not len(first) or not np.isfinite(origins).all()
            or not np.isfinite(directions).all() or not np.isfinite(first).all()
            or not np.isfinite(second).all()):
        raise ValueError("finite ray[N,3] and nonempty segment[E,3] arrays are required")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be finite and positive")
    if not isinstance(max_pairs_per_batch, int) or max_pairs_per_batch < 1:
        raise ValueError("max_pairs_per_batch must be a positive integer")
    if not np.isfinite(pole_tolerance) or not 0 < pole_tolerance < .1:
        raise ValueError("pole_tolerance must be in (0,0.1)")
    length = np.linalg.norm(directions, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("ray direction cannot be zero")
    directions = directions/length[:, None]
    limit = np.broadcast_to(np.asarray(max_distance_mm, dtype=np.float64), (len(origins),))
    if not np.isfinite(limit).all() or np.any(limit <= 0):
        raise ValueError("max_distance_mm must be positive for every ray")
    if (candidate_offsets is None) != (candidate_indices is None):
        raise ValueError("candidate_offsets and candidate_indices must be supplied together")
    if candidate_offsets is not None:
        offsets = np.asarray(candidate_offsets)
        indices = np.asarray(candidate_indices)
        if (offsets.shape != (len(origins)+1,) or offsets.dtype.kind not in "iu"
                or indices.ndim != 1 or indices.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != len(indices)
                or np.any(np.diff(offsets) < 0) or np.any(indices < 0)
                or np.any(indices >= len(first))):
            raise ValueError("invalid CSR ray-to-segment candidates")
        offsets = offsets.astype(np.int64, copy=False)
        indices = indices.astype(np.int32, copy=False)
        pair_count = int(len(indices))
    else:
        offsets = indices = None
        pair_count = int(len(origins)*len(first))

    ray_count = len(origins)
    best_t = np.full(ray_count, np.inf)
    best_point = np.zeros((ray_count, 3))
    best_normal = np.zeros((ray_count, 3))
    best_tangent = np.zeros((ray_count, 3))
    best_segment = np.full(ray_count, -1, dtype=np.int32)
    best_piece = np.full(ray_count, -1, dtype=np.int8)
    best_pole = np.zeros(ray_count, dtype=bool)
    inside = np.zeros(ray_count, dtype=bool)
    tie_count = np.zeros(ray_count, dtype=np.int32)
    tie_normal_sum = np.zeros((ray_count, 3))
    tie_tangent_sum = np.zeros((ray_count, 3))
    tie_tolerance_mm = 1e-8
    for ray_ids, segment_ids in _pair_batches(ray_count, len(first), offsets, indices,
                                               max_pairs_per_batch):
        if not len(ray_ids):
            continue
        t, point, normal, tangent, piece, pair_hit, pair_inside, pole = _pair_first_hit(
            origins[ray_ids], directions[ray_ids], limit[ray_ids],
            first[segment_ids], second[segment_ids], float(radius_mm), pole_tolerance)
        inside[ray_ids[pair_inside]] = True
        hits = np.flatnonzero(pair_hit)
        if not len(hits):
            continue
        batch_min = np.full(ray_count, np.inf)
        np.minimum.at(batch_min, ray_ids[hits], t[hits])
        finite_difference = np.abs(np.where(pair_hit, t, 0.)
                                   - np.where(pair_hit, batch_min[ray_ids], 0.))
        tied = pair_hit & (finite_difference <= tie_tolerance_mm)
        batch_ties = np.zeros(ray_count, dtype=np.int32)
        np.add.at(batch_ties, ray_ids[tied], 1)
        batch_normal_sum = np.zeros((ray_count, 3))
        batch_tangent_sum = np.zeros((ray_count, 3))
        np.add.at(batch_normal_sum, ray_ids[tied], normal[tied])
        np.add.at(batch_tangent_sum, ray_ids[tied], tangent[tied])
        batch_pole = np.zeros(ray_count, dtype=bool)
        batch_pole[ray_ids[tied & pole]] = True
        order = np.lexsort((segment_ids[hits], t[hits], ray_ids[hits]))
        ordered = hits[order]
        unique = np.r_[True, ray_ids[ordered[1:]] != ray_ids[ordered[:-1]]]
        chosen = ordered[unique]
        rays = ray_ids[chosen]
        better = t[chosen] < best_t[rays]-tie_tolerance_mm
        equal = np.abs(t[chosen]-best_t[rays]) <= tie_tolerance_mm
        if np.any(equal):
            tie_count[rays[equal]] += batch_ties[rays[equal]]
            tie_normal_sum[rays[equal]] += batch_normal_sum[rays[equal]]
            tie_tangent_sum[rays[equal]] += batch_tangent_sum[rays[equal]]
            best_pole[rays[equal]] |= batch_pole[rays[equal]]
        if np.any(better):
            replace = chosen[better]
            target = ray_ids[replace]
            best_t[target] = t[replace]
            best_point[target] = point[replace]
            best_normal[target] = normal[replace]
            best_tangent[target] = tangent[replace]
            best_segment[target] = segment_ids[replace]
            best_piece[target] = piece[replace]
            best_pole[target] = pole[replace]
            tie_count[target] = batch_ties[target]
            tie_normal_sum[target] = batch_normal_sum[target]
            tie_tangent_sum[target] = batch_tangent_sum[target]
            best_pole[target] |= batch_pole[target]
    hit = np.isfinite(best_t) & ~inside
    pole_invalid = hit & best_pole
    tie_scale = np.maximum(tie_count.astype(np.float64), 1.)
    normal_coherence = np.linalg.norm(tie_normal_sum, axis=1)/tie_scale
    tangent_coherence = np.linalg.norm(tie_tangent_sum, axis=1)/tie_scale
    # At a polyline joint two coincident endpoint spheres share the same
    # geometric normal. Smooth adjacent axes admit the projected average as
    # the discrete centerline tangent; a sharp joint remains invalid.
    ambiguous = hit & (tie_count > 1) & ((normal_coherence < np.cos(np.deg2rad(1.)))
                                           | (tangent_coherence < np.cos(np.deg2rad(10.))))
    smooth_joint = hit & (tie_count > 1) & ~ambiguous & ~pole_invalid
    if np.any(smooth_joint):
        best_normal[smooth_joint] = tie_normal_sum[smooth_joint]/np.linalg.norm(
            tie_normal_sum[smooth_joint], axis=1)[:, None]
        averaged = tie_tangent_sum[smooth_joint]
        averaged -= np.einsum("ij,ij->i", averaged, best_normal[smooth_joint])[:, None]*best_normal[smooth_joint]
        best_tangent[smooth_joint] = averaged/np.linalg.norm(averaged, axis=1)[:, None]
    valid = hit & ~pole_invalid & ~ambiguous
    best_t[~hit] = np.inf
    best_point[~hit], best_normal[~hit], best_tangent[~hit] = 0., 0., 0.
    best_segment[~hit], best_piece[~hit], tie_count[~hit] = -1, -1, 0
    best_tangent[~valid] = 0.
    return {"hit": hit, "valid": valid, "t_mm": best_t, "P_mm": best_point,
            "N": best_normal, "T": best_tangent, "origin_inside": inside,
            "pole_tangent_invalid": pole_invalid,
            "tangent_ambiguous": ambiguous, "tie_count": tie_count,
            "segment_index": best_segment, "piece_code": best_piece,
            "candidate_pair_count": pair_count,
            "analytic_capsule_union": True,
            "supplied_segments_exhaustively_queried": candidate_offsets is None,
            "global_candidate_completeness_certified": False}
