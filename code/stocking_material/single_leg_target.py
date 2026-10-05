"""Inspect a static, complete single-leg-and-foot surface in world millimetres.

The input has one open cuff and a closed foot.  No rig, vertex groups, world
axis, or left/right convention is assumed.  The resulting landmarks are
geometric estimates for a kinematic stocking fit, not anatomical labels.
"""

from __future__ import annotations

import heapq

import numpy as np


def _unit(vector, name):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError(f"Cannot determine {name}: its direction has zero length.")
    return np.asarray(vector, dtype=np.float64) / length


def _topology(positions, triangles):
    """Check an orientable disk and return its directed boundary cycle."""
    count = len(positions)
    if len(np.unique(triangles)) != count:
        raise ValueError("The leg mesh has isolated or unused vertices.")
    edges = {}
    parent = np.arange(count, dtype=np.int64)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for face in triangles:
        a, b, c = map(int, face)
        for start, end in ((a, b), (b, c), (c, a)):
            ra, rb = root(start), root(end)
            if ra != rb:
                parent[rb] = ra
            key = (min(start, end), max(start, end))
            sign = 1 if start < end else -1
            previous = edges.get(key)
            if previous is None:
                edges[key] = (1, sign, start, end)
            elif previous[0] == 1:
                if previous[1] == sign:
                    raise ValueError("Adjacent triangles have inconsistent winding.")
                edges[key] = (2, 0, 0, 0)
            else:
                raise ValueError("The leg mesh has a nonmanifold edge used by over two faces.")
    if len({root(index) for index in range(count)}) != 1:
        raise ValueError("The leg mesh must be one connected surface.")
    boundary = [(record[2], record[3]) for record in edges.values() if record[0] == 1]
    if len(boundary) < 8:
        raise ValueError("The leg mesh needs one cuff boundary with at least eight edges.")
    successors = {}
    incoming = {}
    for start, end in boundary:
        if start in successors or end in incoming:
            raise ValueError("The open cuff boundary branches or contains multiple loops.")
        successors[start] = end
        incoming[end] = start
    if set(successors) != set(incoming):
        raise ValueError("The open cuff boundary contains a broken chain.")
    start = min(successors)
    loop = []
    cursor = start
    while cursor not in loop:
        loop.append(cursor)
        cursor = successors[cursor]
    if cursor != start or len(loop) != len(boundary):
        raise ValueError("The leg mesh must have exactly one open cuff boundary.")
    euler = count - len(edges) + len(triangles)
    if euler != 1:
        raise ValueError("The leg mesh must have disk topology: one cuff, no handles or extra holes.")
    return np.asarray(loop, dtype=np.int64), np.asarray(list(edges), dtype=np.int64)


def _geodesic_distances(positions, edge_pairs, cuff_loop):
    """Multi-source mesh-edge distance from the cuff, independent of pose."""
    adjacency = [[] for _ in range(len(positions))]
    for a, b in edge_pairs:
        a, b = int(a), int(b)
        length = float(np.linalg.norm(positions[a] - positions[b]))
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    distances = np.full(len(positions), np.inf, dtype=np.float64)
    distances[cuff_loop] = 0.0
    queue = [(0.0, int(index)) for index in cuff_loop]
    heapq.heapify(queue)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance > distances[vertex]:
            continue
        for neighbor, length in adjacency[vertex]:
            proposed = distance + length
            if proposed < distances[neighbor]:
                distances[neighbor] = proposed
                heapq.heappush(queue, (proposed, neighbor))
    if not np.isfinite(distances).all():
        raise ValueError("Some leg vertices cannot be reached from the cuff.")
    return distances


def _section(positions, triangles, scalar, distance):
    """Length-weighted centre and RMS spread of a plane/triangle section."""
    t = scalar[triangles]
    candidates = np.flatnonzero((t.min(axis=1) < distance) & (t.max(axis=1) > distance))
    if len(candidates) < 4:
        return None
    faces = triangles[candidates]
    levels = t[candidates]
    next_levels = levels[:, [1, 2, 0]]
    crossed = ((levels < distance) & (next_levels >= distance)) | ((next_levels < distance) & (levels >= distance))
    keep = crossed.sum(axis=1) == 2
    if np.count_nonzero(keep) < 4:
        return None
    faces, levels, next_levels, crossed = faces[keep], levels[keep], next_levels[keep], crossed[keep]
    fraction = np.divide(distance - levels, next_levels - levels,
                         out=np.zeros_like(levels), where=crossed)
    points = positions[faces]
    ends = positions[faces[:, [1, 2, 0]]]
    hits = (points + fraction[..., None] * (ends - points))[crossed].reshape(-1, 2, 3)
    lengths = np.linalg.norm(hits[:, 1] - hits[:, 0], axis=1)
    valid = lengths > 1e-10
    if np.count_nonzero(valid) < 4:
        return None
    hits, lengths = hits[valid], lengths[valid]
    centers = hits.mean(axis=1)
    center = np.average(centers, axis=0, weights=lengths)
    squared = np.sum((hits - center) ** 2, axis=2).mean(axis=1)
    spread = float(np.sqrt(np.average(squared, weights=lengths)))
    return center, spread


def analyze_single_leg_mesh(positions_mm, triangles):
    """Return a validated surface, cuff, leg path, ankle, ball and toe estimates.

    ``positions_mm`` is an ``(N, 3)`` array in world millimetres and
    ``triangles`` is an ``(M, 3)`` vertex-index array.  The surface must be a
    connected, orientable topological disk with its only boundary at the cuff.
    Consistently inward input winding is reversed in the returned triangles;
    mixed winding and ambiguous foot geometry raise ``ValueError``.

    Array-valued fields remain NumPy arrays for downstream fitting.  Only the
    nested ``report`` dictionary is intended for direct JSON serialization.
    """
    positions = np.asarray(positions_mm, dtype=np.float64)
    raw_faces = np.asarray(triangles)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 16:
        raise ValueError("Single-leg positions must have shape (N, 3), N >= 16.")
    if not np.isfinite(positions).all():
        raise ValueError("Single-leg positions contain NaN or infinity.")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) < 16:
        raise ValueError("Single-leg triangles must have shape (M, 3), M >= 16.")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("Single-leg triangle indices must be integers.")
    faces = raw_faces.astype(np.int64, copy=True)
    if np.any(faces < 0) or np.any(faces >= len(positions)):
        raise ValueError("Single-leg triangles reference a vertex outside the mesh.")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) |
              (faces[:, 2] == faces[:, 0])):
        raise ValueError("Single-leg mesh has a triangle with repeated vertices.")
    span = np.ptp(positions, axis=0)
    diagonal = float(np.linalg.norm(span))
    if diagonal < 1e-3 or diagonal > 1e6:
        raise ValueError("Single-leg model has implausible millimetre-scale dimensions.")
    p = positions[faces]
    double_areas = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if np.any(double_areas <= max(diagonal * diagonal * 1e-14, 1e-12)):
        raise ValueError("Single-leg mesh contains zero-area or extremely small triangles.")

    cuff_loop, edge_pairs = _topology(positions, faces)
    cuff = positions[cuff_loop]
    cuff_center = cuff.mean(axis=0)
    _, cuff_singular, cuff_basis = np.linalg.svd(cuff - cuff_center, full_matrices=False)
    cuff_diameter = float(2.0 * np.sqrt(np.max(np.sum((cuff - cuff_center) ** 2, axis=1))))
    plane_error = float(np.sqrt(np.mean(((cuff - cuff_center) @ cuff_basis[2]) ** 2)))
    if cuff_singular[1] < 0.07 * cuff_singular[0] or plane_error > 0.12 * cuff_diameter:
        raise ValueError("The cuff opening is too narrow or nonplanar to infer a leg axis.")
    # Use the first interior ring to choose the cuff normal's sign.  The
    # whole-leg centroid can lie on the wrong side when a knee is very bent.
    cuff_mask = np.zeros(len(positions), dtype=bool)
    cuff_mask[cuff_loop] = True
    near = np.unique(faces[np.any(cuff_mask[faces], axis=1)])
    interior = near[~cuff_mask[near]]
    if len(interior) < 4:
        raise ValueError("The cuff has too few adjacent interior vertices.")
    inward = positions[interior].mean(axis=0) - cuff_center
    plane_axis = cuff_basis[2]
    if inward @ plane_axis < 0:
        plane_axis = -plane_axis
    if inward @ plane_axis < 1e-5 * cuff_diameter:
        raise ValueError("The cuff does not have a clear inward direction.")
    down_axis = _unit(plane_axis, "the inward leg axis")

    # Cap the single opening virtually.  A positive closed signed volume
    # means the input triangles face outwards.  Reverse a uniformly inward
    # shell, which is common after a negative-scale/mirror transform.
    relative = positions - cuff_center
    q = relative[faces]
    volume6 = float(np.einsum("ij,ij->", q[:, 0], np.cross(q[:, 1], q[:, 2])))
    # The fan triangles (cuff_center, next, current) have zero individual
    # triple product in cuff-centred coordinates, so volume6 above is enough.
    if abs(volume6) < diagonal ** 3 * 1e-8:
        raise ValueError("The mesh volume is too small to establish outward winding.")
    flipped = volume6 < 0
    if flipped:
        faces = faces[:, [0, 2, 1]]
        cuff_loop = cuff_loop[::-1].copy()
    volume_mm3 = abs(volume6) / 6.0

    # Plane slices along the cuff normal cease to be single leg sections in a
    # bent pose.  Distances along mesh edges remain monotonic from cuff to toe.
    surface_distance = _geodesic_distances(positions, edge_pairs, cuff_loop)
    far_distance = float(np.percentile(surface_distance, 99.8))
    if far_distance < 2.2 * cuff_diameter:
        raise ValueError("The cuff-to-toe surface path is too short for a full leg and foot.")
    vertex_area = np.zeros(len(positions), dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_area, faces[:, corner], double_areas / 6.0)
    shell_levels = np.linspace(.06 * far_distance, .94 * far_distance, 45)
    half_width = max(.018 * far_distance, .08 * cuff_diameter)
    shell_centers = []
    spreads = []
    shell_sizes = []
    for level in shell_levels:
        indices = np.flatnonzero(abs(surface_distance - level) <= half_width)
        shell_sizes.append(int(len(indices)))
        if len(indices) < 12:
            shell_centers.append(None)
            spreads.append(np.nan)
            continue
        weights = vertex_area[indices]
        center = np.average(positions[indices], axis=0, weights=weights)
        spread = np.sqrt(np.average(np.sum((positions[indices] - center) ** 2, axis=1),
                                    weights=weights))
        shell_centers.append(center)
        spreads.append(float(spread))
    spreads = np.asarray(spreads, dtype=np.float64)
    valid = np.isfinite(spreads)
    shaft = valid & (shell_levels >= .10 * far_distance) & (shell_levels <= .60 * far_distance)
    if np.count_nonzero(shaft) < 15:
        raise ValueError("Too few reliable cuff-distance bands to trace the leg shaft.")
    ankle_candidates = np.flatnonzero(valid & (shell_levels >= .63 * far_distance) &
                                      (shell_levels <= .85 * far_distance))
    if len(ankle_candidates) < 5:
        raise ValueError("The distal leg-to-foot region is insufficiently sampled.")
    ankle_index = int(ankle_candidates[np.argmin(spreads[ankle_candidates])])
    ankle_depth = float(shell_levels[ankle_index])
    ankle_center = shell_centers[ankle_index]
    toe_band = surface_distance >= np.percentile(surface_distance, 99)
    toe_point = np.average(positions[toe_band], axis=0, weights=vertex_area[toe_band])
    foot_axis = _unit(toe_point - ankle_center, "the ankle-to-toe axis")
    toe_reach = float(np.linalg.norm(toe_point - ankle_center))
    if toe_reach < .50 * cuff_diameter:
        raise ValueError("The inferred toe is too close to the ankle for a complete foot.")
    before = np.flatnonzero(valid & (shell_levels >= ankle_depth - .20 * far_distance) &
                            (shell_levels <= ankle_depth - .12 * far_distance))
    if not len(before):
        raise ValueError("Cannot establish the lower-leg direction before the ankle.")
    shin_center = shell_centers[int(before[-1])]
    shin_axis = _unit(ankle_center - shin_center, "the lower-leg axis")
    foot_bend = float(np.linalg.norm((toe_point - ankle_center) -
                                    ((toe_point - ankle_center) @ shin_axis) * shin_axis))
    after = valid & (shell_levels > ankle_depth + .04 * far_distance) & (
        shell_levels <= .92 * far_distance)
    spread_growth = (float(np.nanmax(spreads[after]) / spreads[ankle_index])
                     if np.any(after) else 0.0)
    if foot_bend < .10 * cuff_diameter and spread_growth < 1.08:
        raise ValueError("The distal mesh resembles a capped leg; a distinct foot is not identifiable.")
    foot_scalar = (positions - ankle_center) @ foot_axis
    ball_section = _section(positions, faces, foot_scalar, .65 * toe_reach)
    ball_center = ball_section[0] if ball_section is not None else ankle_center + .65 * (toe_point - ankle_center)
    foot_indices = np.flatnonzero(surface_distance >= ankle_depth - .12 * cuff_diameter)
    if len(foot_indices) < 24:
        raise ValueError("Too few vertices form the inferred foot region.")

    centerline = [cuff_center]
    for level, center in zip(shell_levels, shell_centers):
        if center is not None and .09 * far_distance <= level < ankle_depth:
            centerline.append(center)
    centerline.append(ankle_center)
    centerline = np.asarray(centerline, dtype=np.float64)
    report = {
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(faces)),
        "edge_count": int(len(edge_pairs)),
        "boundary_loop_count": 1,
        "cuff_vertex_count": int(len(cuff_loop)),
        "euler_characteristic": 1,
        "winding_flipped": bool(flipped),
        "enclosed_volume_mm3": float(volume_mm3),
        "cuff_diameter_mm": float(cuff_diameter),
        "cuff_planarity_rms_mm": float(plane_error),
        "cuff_axis_extent_mm": float(np.percentile((positions - cuff_center) @ down_axis, 99.8)),
        "leg_depth_mm": float(far_distance),
        "ankle_depth_mm": float(ankle_depth),
        "toe_reach_mm": float(toe_reach),
        "foot_vertices": int(len(foot_indices)),
        "foot_bend_mm": float(foot_bend),
        "post_ankle_spread_ratio": float(spread_growth),
        "ankle_section_spread_mm": float(spreads[ankle_index]),
        "shaft_section_count": int(np.count_nonzero(shaft)),
    }
    return {
        "positions_mm": positions,
        "triangles": faces,
        "cuff_loop": cuff_loop,
        "cuff_center_mm": cuff_center,
        "down_axis": down_axis,
        "foot_axis": foot_axis,
        "ankle_center_mm": ankle_center,
        "ball_center_mm": ball_center,
        "toe_point_mm": toe_point,
        "foot_vertex_indices": foot_indices,
        "leg_centerline_mm": centerline,
        "report": report,
    }
