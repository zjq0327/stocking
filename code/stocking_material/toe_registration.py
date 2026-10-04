"""Register the source toe disk to a clipped authored-target toe disk.

All returned bindings address faces of the unchanged authored-target-v2 mesh.
The clipped mesh is only an auxiliary parameter domain; each clipped corner
stores its original target face and barycentric coordinate exactly.
"""

from collections import Counter, defaultdict
import math

import numpy as np

from .positive_harmonic import positive_harmonic_disk


def _unit(value):
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length < 1e-12 or not np.isfinite(length):
        raise ValueError("Zero or invalid toe-cut direction")
    return value/length


def _toe_cut_plane(boundary_points, target_points):
    center = np.mean(boundary_points, axis=0)
    centered = boundary_points-center
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = right[-1]
    tip = target_points[np.argmin(target_points[:, 1])]
    if np.dot(normal, tip-center) < 0:
        normal = -normal
    residual = centered@normal
    return center, normal, {"maximum_boundary_plane_residual_mm": float(np.max(np.abs(residual))),
                            "rms_boundary_plane_residual_mm": float(np.sqrt(np.mean(residual**2)))}


def _clip_target_cap(target_points, target_faces, allowed_faces, plane_center, plane_normal):
    """Keep the toe side of a plane, welding crossings by original edge ID."""
    cap_vertices = []
    key_to_local = {}
    cap_triangles = []
    cap_parent_face = []
    cap_parent_bary = []
    plane_distance = (target_points-plane_center)@plane_normal
    epsilon = 1e-9

    def vertex(key, position):
        if key not in key_to_local:
            key_to_local[key] = len(cap_vertices)
            cap_vertices.append(np.asarray(position, dtype=np.float64))
        return key_to_local[key]

    for global_face in allowed_faces:
        original_ids = target_faces[global_face]
        polygon = []
        for corner, vertex_id in enumerate(original_ids):
            weight = np.zeros(3)
            weight[corner] = 1.
            polygon.append((("v", int(vertex_id)), target_points[vertex_id], weight,
                            float(plane_distance[vertex_id])))

        def intersection(first, second):
            if abs(first[3]) <= epsilon:
                return first
            if abs(second[3]) <= epsilon:
                return second
            t = first[3]/(first[3]-second[3])
            v0, v1 = int(first[0][1]), int(second[0][1])
            key = ("e", min(v0, v1), max(v0, v1))
            return (key, first[1]+t*(second[1]-first[1]),
                    first[2]+t*(second[2]-first[2]), 0.)

        clipped = []
        for index in range(3):
            first, second = polygon[index-1], polygon[index]
            inside_first, inside_second = first[3] >= -epsilon, second[3] >= -epsilon
            if inside_first and inside_second:
                clipped.append(second)
            elif inside_first and not inside_second:
                clipped.append(intersection(first, second))
            elif not inside_first and inside_second:
                clipped.extend((intersection(first, second), second))
        compact = []
        for corner in clipped:
            if not compact or corner[0] != compact[-1][0]:
                compact.append(corner)
        if len(compact) > 1 and compact[0][0] == compact[-1][0]:
            compact.pop()
        if len(compact) < 3:
            continue
        for j in range(1, len(compact)-1):
            corners = (compact[0], compact[j], compact[j+1])
            indices = tuple(vertex(corner[0], corner[1]) for corner in corners)
            if len(set(indices)) < 3:
                continue
            triangle = np.asarray([cap_vertices[i] for i in indices])
            area2 = np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
            if area2 < 1e-10:
                continue
            cap_triangles.append(indices)
            cap_parent_face.append(int(global_face))
            cap_parent_bary.append([corner[2] for corner in corners])
    points = np.asarray(cap_vertices, dtype=np.float64)
    faces = np.asarray(cap_triangles, dtype=np.int32)
    parent = np.asarray(cap_parent_face, dtype=np.int32)
    bary = np.asarray(cap_parent_bary, dtype=np.float64)
    if len(points) < 3 or len(faces) < 1:
        raise ValueError("Toe cut produced no target cap")
    return points, faces, parent, bary


def _single_boundary_loop(triangles, vertex_count):
    counts = Counter()
    for a, b, c in triangles:
        for x, y in ((a, b), (b, c), (c, a)):
            counts[(min(int(x), int(y)), max(int(x), int(y)))] += 1
    if any(count > 2 for count in counts.values()):
        raise ValueError("Clipped toe cap contains nonmanifold edges")
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    adjacent = defaultdict(list)
    for a, b in boundary_edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    if not boundary_edges or any(len(neighbors) != 2 for neighbors in adjacent.values()):
        raise ValueError("Clipped toe cap boundary is not a simple loop")
    start = min(adjacent)
    loop = [start]
    previous = -1
    current = start
    while True:
        neighbors = adjacent[current]
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == start:
            break
        if following in loop:
            raise ValueError("Clipped toe cap boundary self-repeats")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacent):
        raise ValueError("Clipped toe cap has multiple boundary loops")
    used = np.unique(triangles)
    if len(used) != vertex_count:
        raise ValueError("Clipped toe cap has orphan vertices")
    # Euler characteristic 1 is required for a disk with one boundary.
    euler = vertex_count-len(counts)+len(triangles)
    if euler != 1:
        raise ValueError(f"Clipped toe cap is not a disk; Euler characteristic {euler}")
    return np.asarray(loop, dtype=np.int32), {"vertices": int(vertex_count),
                                              "edges": len(counts), "faces": len(triangles),
                                              "boundary_vertices": len(loop),
                                              "euler_characteristic": int(euler)}


def _project_loop_arclength(points, loop, query):
    curve = points[loop]
    following = np.roll(curve, -1, axis=0)
    edge = following-curve
    lengths = np.linalg.norm(edge, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("Toe cut boundary contains collapsed edges")
    edge_length2 = lengths*lengths
    cumulative = np.r_[0., np.cumsum(lengths)]
    arclength = np.empty(len(query))
    distance = np.empty(len(query))
    for i, point in enumerate(query):
        t = np.clip(np.einsum("ij,ij->i", point-curve, edge)/edge_length2, 0., 1.)
        closest = curve+t[:, None]*edge
        distance2 = np.einsum("ij,ij->i", point-closest, point-closest)
        index = int(np.argmin(distance2))
        arclength[i] = cumulative[index]+t[index]*lengths[index]
        distance[i] = math.sqrt(float(distance2[index]))
    return arclength, distance, cumulative


def _harmonic_boundary_angles(cap_points, loop, source_ring_points):
    # The cut plane is only a few mm from the already registered material
    # ring; use its projected arclength to align the target disk boundary.
    forward, distance, cumulative = _project_loop_arclength(cap_points, loop, source_ring_points)
    total = cumulative[-1]
    forward_unwrapped = np.unwrap(2*math.pi*(forward-forward[0])/total)
    reverse_loop = np.r_[loop[:1], loop[:0:-1]]
    reverse, reverse_distance, reverse_cumulative = _project_loop_arclength(cap_points, reverse_loop,
                                                                             source_ring_points)
    reverse_unwrapped = np.unwrap(2*math.pi*(reverse-reverse[0])/reverse_cumulative[-1])
    forward_bad = int(np.count_nonzero(np.diff(forward_unwrapped) <= 0))
    reverse_bad = int(np.count_nonzero(np.diff(reverse_unwrapped) <= 0))
    if reverse_bad < forward_bad:
        loop, forward, distance, cumulative = reverse_loop, reverse, reverse_distance, reverse_cumulative
        forward_unwrapped, forward_bad = reverse_unwrapped, reverse_bad
    if forward_bad:
        raise ValueError(f"Target toe-cut perimeter disagrees with material boundary order at {forward_bad} edges")
    if forward_unwrapped[-1] >= 2*math.pi:
        raise ValueError("Material boundary makes more than one turn around toe cut")
    source_angles = forward_unwrapped-forward_unwrapped[0]
    target_angles = np.mod(2*math.pi*(cumulative[:-1]-forward[0])/cumulative[-1], 2*math.pi)
    source_xy = np.column_stack((np.cos(source_angles), np.sin(source_angles)))
    target_xy = np.column_stack((np.cos(target_angles), np.sin(target_angles)))
    return loop, source_xy, target_xy, {"boundary_projection_max_mm": float(distance.max()),
                                        "boundary_projection_p95_mm": float(np.percentile(distance, 95)),
                                        "boundary_material_phase_last_radians": float(source_angles[-1])}


def _disk_query_index(xy, triangles, cell_count=64):
    """Build a compact lookup of UV triangles by their unit-disk bounding cells."""
    grid = defaultdict(list)
    tri = xy[triangles]
    for face_id, points in enumerate(tri):
        low = np.clip(np.floor((points.min(axis=0)+1)*cell_count/2).astype(int), 0, cell_count-1)
        high = np.clip(np.floor((points.max(axis=0)+1)*cell_count/2).astype(int), 0, cell_count-1)
        for x in range(low[0], high[0]+1):
            for y in range(low[1], high[1]+1):
                grid[(x,y)].append(face_id)
    return grid, tri, cell_count


def _disk_query(index, point):
    grid, tri, size = index
    cell = np.clip(np.floor((point+1)*size/2).astype(int), 0, size-1)
    candidates = grid.get((int(cell[0]), int(cell[1])), ())
    for face_id in candidates:
        a, b, c = tri[face_id]
        v0, v1, v2 = b-a, c-a, point-a
        den = v0[0]*v1[1]-v0[1]*v1[0]
        if abs(den) < 1e-15:
            continue
        v = (v2[0]*v1[1]-v2[1]*v1[0])/den
        w = (v0[0]*v2[1]-v0[1]*v2[0])/den
        bary = np.array([1-v-w, v, w])
        if np.min(bary) >= -1e-7:
            return face_id, np.clip(bary, 0., 1.)/np.clip(bary, 0., 1.).sum()
    return None, None


def register_toe_disk(reference, target_unposed, target_triangles, allowed_faces,
                      binding_face, binding_bary, toe_start_ring=192):
    """Replace toe interiors with a shared positive harmonic-disk map.

    Inputs and returned face IDs index the original full target triangles.
    The already fitted toe-boundary ring is kept fixed in material identity.
    """
    target_unposed = np.asarray(target_unposed, dtype=np.float64)
    target_triangles = np.asarray(target_triangles, dtype=np.int32)
    allowed_faces = np.asarray(allowed_faces, dtype=np.int32)
    updated_face = np.asarray(binding_face, dtype=np.int32).copy()
    updated_bary = np.asarray(binding_bary, dtype=np.float64).copy()
    circumference = int(reference.metadata["parameters"]["circumference_segments"])
    ring_count = len(reference.metadata["ring_regions"])
    base = toe_start_ring*circumference
    if toe_start_ring >= ring_count or len(reference.reference_positions_mm) <= base+circumference:
        raise ValueError("Reference has no toe patch at the requested ring")
    boundary_indices = np.arange(base, base+circumference, dtype=np.int32)
    boundary_points = np.einsum("ni,nij->nj", updated_bary[boundary_indices],
                                target_unposed[target_triangles[updated_face[boundary_indices]]])
    side_vertices = np.unique(target_triangles[allowed_faces])
    plane_center, plane_normal, plane_report = _toe_cut_plane(boundary_points,
                                                                target_unposed[side_vertices])
    target_cap_points, target_cap_faces, parent_face, parent_bary = _clip_target_cap(
        target_unposed, target_triangles, allowed_faces, plane_center, plane_normal)
    target_loop, target_topology = _single_boundary_loop(target_cap_faces, len(target_cap_points))
    target_loop, source_circle, target_circle, phase_report = _harmonic_boundary_angles(
        target_cap_points, target_loop, boundary_points)

    patch_face_mask = np.all(reference.triangles >= base, axis=1)
    source_faces = (reference.triangles[patch_face_mask]-base).astype(np.int32)
    source_points = reference.reference_positions_mm[base:]
    source_loop, source_topology = _single_boundary_loop(source_faces, len(source_points))
    if len(source_loop) != circumference or set(source_loop.tolist()) != set(range(circumference)):
        raise ValueError("X0 toe patch boundary differs from the last straight-foot ring")
    target_xy, target_harmonic_report = positive_harmonic_disk(
        target_cap_points, target_cap_faces, target_loop, target_circle)
    source_xy, source_harmonic_report = positive_harmonic_disk(
        source_points, source_faces, np.arange(circumference, dtype=np.int32), source_circle)
    target_tri_xy = target_xy[target_cap_faces]
    target_area2 = np.cross(target_tri_xy[:,1]-target_tri_xy[:,0],
                            target_tri_xy[:,2]-target_tri_xy[:,0])
    source_tri_xy = source_xy[source_faces]
    source_area2 = np.cross(source_tri_xy[:,1]-source_tri_xy[:,0],
                            source_tri_xy[:,2]-source_tri_xy[:,0])
    if np.any(np.abs(target_area2) < 1e-12) or np.any(np.abs(source_area2) < 1e-12):
        raise ValueError("Toe harmonic disk contains singular triangles")
    if np.any(np.sign(target_area2) != np.sign(target_area2[0])) or np.any(np.sign(source_area2) != np.sign(source_area2[0])):
        raise ValueError("Toe harmonic disk is not consistently oriented")
    index = _disk_query_index(target_xy, target_cap_faces)
    query_misses = 0
    for local_vertex in range(circumference, len(source_points)):
        cap_face, uv_bary = _disk_query(index, source_xy[local_vertex])
        if cap_face is None:
            query_misses += 1
            continue
        global_vertex = base+local_vertex
        updated_face[global_vertex] = parent_face[cap_face]
        original_bary = uv_bary@parent_bary[cap_face]
        original_bary = np.clip(original_bary, 0., 1.)
        updated_bary[global_vertex] = original_bary/original_bary.sum()
    if query_misses:
        raise ValueError(f"Toe harmonic disk missed {query_misses} target faces in its own domain")
    report = {"method": "positive harmonic source/target disks via clipped target toe cap",
              "plane_center_mm": plane_center.tolist(), "plane_normal_toward_toe": plane_normal.tolist(),
              "cut": plane_report, "target_topology": target_topology, "source_topology": source_topology,
              "boundary_match": phase_report,
              "target_harmonic": target_harmonic_report, "source_harmonic": source_harmonic_report,
              "target_disk_min_abs_area2": float(np.min(np.abs(target_area2))),
              "source_disk_min_abs_area2": float(np.min(np.abs(source_area2))),
              "boundary_ring_kept_fixed": True,
              "bound_interior_points": int(len(source_points)-circumference)}
    return updated_face, updated_bary, report
