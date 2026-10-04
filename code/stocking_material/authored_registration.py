"""Geometric correspondence from a procedural unworn shell to an authored sock.

The target's unposed and frame-996 arrays share one triangle topology.  Every
material vertex is bound once to a target triangle and barycentric position in
the unposed array, then transported with that same identity to frame 996.
This is a kinematic construction, not a dressing or yarn equilibrium solver.
"""

from dataclasses import dataclass
import copy
import math

import numpy as np

from .garment_data import GarmentMeshPair
from .surface_registration_refinement import repair_orientation_on_surface
from .toe_registration import register_toe_disk


@dataclass
class RegistrationResult:
    binding: dict
    pair_unposed: GarmentMeshPair
    pair_current: GarmentMeshPair
    report: dict
    quality_unposed: dict
    quality_current: dict


def _unit(value):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero or nonfinite registration vector")
    return value / norm


def _barycentric(point, triangle):
    a, b, c = triangle
    v0, v1, v2 = b-a, c-a, point-a
    d00, d01, d11 = v0@v0, v0@v1, v1@v1
    denominator = d00*d11-d01*d01
    if denominator <= 1e-20:
        raise ValueError("Target triangle is singular")
    d20, d21 = v2@v0, v2@v1
    v = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    result = np.array([1-v-w, v, w])
    if np.min(result) < -1e-4 or np.max(result) > 1+1e-4:
        raise ValueError(f"Projected point fell outside its target triangle: bary={result.tolist()}, point={point.tolist()}")
    result = np.clip(result, 0, 1)
    return result/result.sum()


def _pchip(x, y, samples):
    """Small monotone cubic mapping between anatomical arc-length anchors."""
    x, y, samples = np.asarray(x), np.asarray(y), np.asarray(samples)
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("Registration path anchors must be strictly increasing")
    h = np.diff(x)
    delta = np.diff(y)/h
    slope = np.empty(len(x))
    slope[0], slope[-1] = delta[0], delta[-1]
    for i in range(1, len(x)-1):
        left, right = delta[i-1], delta[i]
        w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
        slope[i] = (w1+w2)/(w1/left+w2/right)
    section = np.clip(np.searchsorted(x, samples, side="right")-1, 0, len(h)-1)
    t = (samples-x[section])/h[section]
    output = ((2*t**3-3*t**2+1)*y[section]
              +(t**3-2*t**2+t)*h[section]*slope[section]
              +(-2*t**3+3*t**2)*y[section+1]
              +(t**3-t**2)*h[section]*slope[section+1])
    return np.clip(output, y[0], y[-1])


def _section_center(vertices, faces, axis, value, *, extra=None):
    """Center of actual triangle/plane intersections, independent of vertex density."""
    points = vertices[faces]
    intersection = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        first, second = points[:, a], points[:, b]
        da, db = first[:, axis]-value, second[:, axis]-value
        crossing = (da*db < 0) | (np.abs(da) < 1e-9) | (np.abs(db) < 1e-9)
        safe = crossing & (np.abs(da-db) > 1e-12)
        fraction = da[safe]/(da[safe]-db[safe])
        intersection.append(first[safe]+fraction[:, None]*(second[safe]-first[safe]))
    samples = np.concatenate(intersection)
    if extra is not None:
        samples = samples[extra(samples)]
    if len(samples) < 16:
        raise ValueError(f"Target plane section axis {axis}, value {value} is under-sampled")
    center = (samples.min(axis=0)+samples.max(axis=0))/2
    center[axis] = value
    return center


def _path_for_component(all_vertices, component_faces, component_vertices, cuff, params):
    """Use actual target shell slices, with one smooth cuff-to-toe arc coordinate."""
    cuff_points = all_vertices[cuff]
    cuff_center = (cuff_points.min(axis=0)+cuff_points.max(axis=0))/2
    cuff_center[2] = cuff_points[:, 2].mean()
    z_levels = np.r_[np.arange(cuff_center[2]-15, 70, -15), 70.]
    centers = [cuff_center]
    for z in z_levels:
        centers.append(_section_center(all_vertices, component_faces, 2, z))
    leg_end_index = len(centers)-1
    # A fitted Hermite elbow joins the actual z=70 leg and y=-10 foot
    # cross-section centers with their measured incoming/outgoing tangents.
    # Directly splicing z and y section midpoints gives a medial path kink
    # exactly where material meridians turn from leg to foot.
    elbow_start = centers[-1]
    elbow_end = _section_center(all_vertices, component_faces, 1, -10., extra=lambda p: p[:, 2] < 65.)
    next_foot = _section_center(all_vertices, component_faces, 1, -20., extra=lambda p: p[:, 2] < 65.)
    incoming = _unit(elbow_start-centers[-2])
    outgoing = _unit(next_foot-elbow_end)
    handle_length = min(20., np.linalg.norm(elbow_end-elbow_start)/3.)
    control_1 = elbow_start+handle_length*incoming
    control_2 = elbow_end-handle_length*outgoing
    for t in np.linspace(0., 1., 17)[1:]:
        centers.append(((1-t)**3*elbow_start + 3*(1-t)**2*t*control_1
                        + 3*(1-t)*t*t*control_2 + t**3*elbow_end))
    heel_end_index = len(centers)-1
    centers.append(next_foot)
    for y in (-35., -50., -65., -80., -95.):
        centers.append(_section_center(all_vertices, component_faces, 1, y, extra=lambda p: p[:, 2] < 65.))
    toe_start_index = len(centers)-2  # the -80 mm foot cross-section
    toe = component_vertices[component_vertices[:, 1] < component_vertices[:, 1].min()+1.5].mean(axis=0)
    centers.append(toe)
    raw = np.asarray(centers)
    path_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    if np.any(np.diff(path_s) <= 1e-6):
        raise ValueError("Registration center path has a repeated point")
    uniform_s = np.linspace(0, path_s[-1], max(256, int(math.ceil(path_s[-1]))+1))
    sampled = np.column_stack([np.interp(uniform_s, path_s, raw[:, axis]) for axis in range(3)])
    kernel_x = np.arange(-12, 13, dtype=float)
    kernel = np.exp(-.5*(kernel_x/4)**2)
    kernel /= kernel.sum()
    smooth = np.column_stack([np.convolve(np.pad(sampled[:, axis], (12,12), mode="edge"), kernel, mode="valid")
                              for axis in range(3)])
    smooth[0], smooth[-1] = raw[0], raw[-1]
    straight_leg = params["leg_length_mm"]-params["heel_bend_radius_mm"]
    heel_length = math.pi*params["heel_bend_radius_mm"]/2
    straight_foot = params["foot_length_mm"]-params["heel_bend_radius_mm"]-params["toe_length_mm"]
    source_anchors = np.cumsum([0., straight_leg, heel_length, straight_foot, params["toe_length_mm"]])
    target_anchors = path_s[[0, leg_end_index, heel_end_index, toe_start_index, len(path_s)-1]]
    if np.any(np.diff(target_anchors) <= 0):
        raise ValueError("Target anatomy anchors are not ordered along the center path")
    return uniform_s, smooth, source_anchors, target_anchors, toe


def _path_sample(uniform_s, path, value):
    center = np.array([np.interp(value, uniform_s, path[:, i]) for i in range(3)])
    before = np.array([np.interp(max(0., value-2.), uniform_s, path[:, i]) for i in range(3)])
    after = np.array([np.interp(min(uniform_s[-1], value+2.), uniform_s, path[:, i]) for i in range(3)])
    return center, _unit(after-before)


def _arc_fraction(points):
    lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Reference or target ring has a zero-length edge")
    return np.r_[0., np.cumsum(lengths)]/lengths.sum()


def _cuff_binding(reference_ring, target_vertices, loop, target_triangles, edge_faces, major, minor):
    points = target_vertices[loop]
    center = (points.min(axis=0)+points.max(axis=0))/2
    anchor = int(np.argmax((points-center)@major))
    loop = np.roll(loop, -anchor)
    points = target_vertices[loop]
    if (points[len(points)//4]-center)@minor < 0:
        loop = np.r_[loop[:1], loop[:0:-1]]
        points = target_vertices[loop]
    target_fraction = _arc_fraction(points)
    reference_fraction = _arc_fraction(reference_ring)[:-1]
    edge_index = np.clip(np.searchsorted(target_fraction, reference_fraction, side="right")-1, 0, len(loop)-1)
    t = (reference_fraction-target_fraction[edge_index])/(target_fraction[edge_index+1]-target_fraction[edge_index])
    output_faces = np.empty(len(reference_ring), dtype=np.int32)
    weights = np.zeros((len(reference_ring), 3))
    for k, (edge, fraction) in enumerate(zip(edge_index, t)):
        v0, v1 = int(loop[edge]), int(loop[(edge+1)%len(loop)])
        face = edge_faces.get((min(v0,v1), max(v0,v1)))
        if face is None:
            raise ValueError("A cuff loop edge has no target face")
        output_faces[k] = face
        triangle = target_triangles[face]
        weights[k, np.flatnonzero(triangle == v0)[0]] = 1-fraction
        weights[k, np.flatnonzero(triangle == v1)[0]] = fraction
    return output_faces, weights


def _face_normals(positions, triangles):
    points = positions[triangles]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Target has degenerate triangles")
    return cross/lengths[:, None]


def _orientation_dots(positions, reference_faces, target_positions, target_triangles,
                      target_face_index):
    points = positions[reference_faces]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    normal = cross/np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-30)
    target_normal = _face_normals(target_positions, target_triangles)
    expected = target_normal[target_face_index][reference_faces].sum(axis=1)
    expected /= np.maximum(np.linalg.norm(expected, axis=1)[:, None], 1e-30)
    return np.einsum("ij,ij->i", normal, expected)


def _improve_same_face_pose_margin(reference_faces, target_unposed, target_current,
                                   target_triangles, face_index, bary, unposed_points,
                                   current_points, movable, *, minimum_dot=.01):
    """Increase a near-tangent pose margin with a tiny same-face bary shift."""
    all_current = _orientation_dots(current_points, reference_faces, target_current,
                                    target_triangles, face_index)
    initial_minimum = float(all_current.min())
    if initial_minimum >= minimum_dot:
        return {"attempted": False, "accepted": False, "initial_minimum_dot": initial_minimum}
    normal_u = _face_normals(target_unposed, target_triangles)
    normal_c = _face_normals(target_current, target_triangles)
    best = None
    for vertex in movable:
        incident = np.flatnonzero(np.any(reference_faces == vertex, axis=1))
        local_faces = reference_faces[incident]
        expected_u = normal_u[face_index][local_faces].sum(axis=1)
        expected_c = normal_c[face_index][local_faces].sum(axis=1)
        expected_u /= np.maximum(np.linalg.norm(expected_u, axis=1)[:, None], 1e-30)
        expected_c /= np.maximum(np.linalg.norm(expected_c, axis=1)[:, None], 1e-30)
        base_tri_u = unposed_points[local_faces]
        base_tri_c = current_points[local_faces]
        corner_mask = local_faces == vertex
        target_face = int(face_index[vertex])
        triangle_u = target_unposed[target_triangles[target_face]]
        triangle_c = target_current[target_triangles[target_face]]
        for du in np.linspace(-.05, .05, 11):
            for dv in np.linspace(-.05, .05, 11):
                proposal = bary[vertex]+np.array([du, dv, -du-dv])
                if np.min(proposal) < 0 or np.max(proposal) > 1:
                    continue
                point_u, point_c = proposal@triangle_u, proposal@triangle_c
                move_u = float(np.linalg.norm(point_u-unposed_points[vertex]))
                move_c = float(np.linalg.norm(point_c-current_points[vertex]))
                if move_u > .5 or move_c > .5:
                    continue
                tri_u, tri_c = base_tri_u.copy(), base_tri_c.copy()
                tri_u[corner_mask], tri_c[corner_mask] = point_u, point_c
                cross_u = np.cross(tri_u[:,1]-tri_u[:,0], tri_u[:,2]-tri_u[:,0])
                cross_c = np.cross(tri_c[:,1]-tri_c[:,0], tri_c[:,2]-tri_c[:,0])
                dots_u = np.einsum("ij,ij->i", cross_u, expected_u)/np.maximum(
                    np.linalg.norm(cross_u, axis=1), 1e-30)
                dots_c = np.einsum("ij,ij->i", cross_c, expected_c)/np.maximum(
                    np.linalg.norm(cross_c, axis=1), 1e-30)
                if np.min(dots_u) <= 0 or np.min(dots_c) <= 0:
                    continue
                score = float(np.min(dots_c)-.001*move_c)
                if best is None or score > best[0]:
                    best = (score, int(vertex), proposal, point_u, point_c, move_u, move_c)
    if best is None:
        return {"attempted": True, "accepted": False, "initial_minimum_dot": initial_minimum,
                "reason": "no feasible same-face bary shift"}
    score, vertex, proposal, point_u, point_c, move_u, move_c = best
    old_bary = bary[vertex].copy()
    old_u, old_c = unposed_points[vertex].copy(), current_points[vertex].copy()
    bary[vertex], unposed_points[vertex], current_points[vertex] = proposal, point_u, point_c
    final_u = _orientation_dots(unposed_points, reference_faces, target_unposed,
                                 target_triangles, face_index)
    final_c = _orientation_dots(current_points, reference_faces, target_current,
                                 target_triangles, face_index)
    accepted = bool(np.all(final_u > 0) and np.all(final_c >= minimum_dot))
    if not accepted:
        bary[vertex], unposed_points[vertex], current_points[vertex] = old_bary, old_u, old_c
    return {"attempted": True, "accepted": accepted, "initial_minimum_dot": initial_minimum,
            "proposed_minimum_current_dot": float(final_c.min()),
            "proposed_minimum_unposed_dot": float(final_u.min()),
            "material_vertex": vertex, "bary_delta": (proposal-old_bary).tolist(),
            "current_move_mm": move_c, "unposed_move_mm": move_u,
            "requirement": f"both poses positive and current minimum dot >= {minimum_dot}"}


def _quality(reference, positions, target_positions, target_triangles, target_face_index,
             target_bvh, *, target_face_indices):
    faces = reference.triangles
    p = positions[faces]
    cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normal = cross/np.maximum(area2[:, None], 1e-30)
    target_normal = _face_normals(target_positions, target_triangles)
    target_at_nodes = target_normal[target_face_index]
    average = target_at_nodes[faces].sum(axis=1)
    average /= np.maximum(np.linalg.norm(average, axis=1)[:, None], 1e-30)
    orientation_dot = np.einsum("ij,ij->i", normal, average)
    reference_faces = reference.reference_positions_mm[faces]
    reference_area2 = np.linalg.norm(np.cross(reference_faces[:, 1]-reference_faces[:, 0],
                                               reference_faces[:, 2]-reference_faces[:, 0]), axis=1)
    area_ratio = area2/reference_area2
    # Independent target-to-candidate and candidate-to-target checks on a
    # deterministic face-center subset; full binding distances are exact.
    sample = np.linspace(0, len(target_face_indices)-1, min(8192, len(target_face_indices)), dtype=np.int32)
    target_centers = target_positions[target_triangles[target_face_indices[sample]]].mean(axis=1)
    from mathutils import Vector
    candidate_bvh = _mesh_bvh(positions, faces)
    target_to_candidate = np.array([candidate_bvh.find_nearest(Vector(point))[3] for point in target_centers])
    candidate_centers = p[np.linspace(0, len(p)-1, min(8192, len(p)), dtype=np.int32)].mean(axis=1)
    candidate_to_target = np.array([target_bvh.find_nearest(Vector(point))[3] for point in candidate_centers])
    return {
        "orientation_dot": orientation_dot,
        "area_ratio": area_ratio,
        "target_to_candidate_distance_mm": target_to_candidate,
        "candidate_to_target_distance_mm": candidate_to_target,
    }


def _mesh_bvh(positions, triangles):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([Vector(row) for row in positions], triangles.tolist(), all_triangles=True)


def _summary(quality):
    def describe(values):
        return {"min": float(np.min(values)), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
    return {
        "independent_target_normal_dot": describe(quality["orientation_dot"]),
        "nonpositive_target_normal_faces": int(np.count_nonzero(quality["orientation_dot"] <= 0)),
        "area_ratio_to_X0": describe(quality["area_ratio"]),
        "target_to_candidate_distance_mm_sample": describe(quality["target_to_candidate_distance_mm"]),
        "candidate_to_target_distance_mm_sample": describe(quality["candidate_to_target_distance_mm"]),
    }


def register_authored_pair(reference: GarmentMeshPair, target: dict, target_metadata: dict,
                           side: str, *, dense_angles=256):
    """Return stable target binding and two same-material geometric candidates.

    Requires Blender's bundled NumPy/mathutils.  The source target must be the
    audited v2 topology; no body mesh is used for the surface fit.
    """
    from mathutils import Vector

    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    reference.validate()
    unposed = np.asarray(target["unposed_positions_mm"], dtype=np.float64)
    current = np.asarray(target["positions_mm"], dtype=np.float64)
    triangles = np.asarray(target["triangles"], dtype=np.int32)
    components = np.asarray(target["component"], dtype=np.int32)
    if unposed.shape != current.shape or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Target poses must share one triangular topology")
    component_ids = np.unique(components)
    if len(component_ids) != 2:
        raise ValueError("Expected exactly two authored stocking components")
    component_x = {int(i): float(np.mean(unposed[components == i, 0])) for i in component_ids}
    chosen_component = max(component_x, key=component_x.get) if side == "left" else min(component_x, key=component_x.get)
    selected = np.all(components[triangles] == chosen_component, axis=1)
    global_faces = np.flatnonzero(selected).astype(np.int32)
    if not len(global_faces):
        raise ValueError("The chosen target component has no triangles")
    bvh = _mesh_bvh(unposed, triangles[global_faces])
    loops = target_metadata["validation"]["unposed_positions_mm"]["cuff_loops"]
    cuff = next((np.asarray(row, dtype=np.int32) for row in loops if components[row[0]] == chosen_component), None)
    if cuff is None:
        raise ValueError("Target component has no identified cuff boundary")
    edge_faces = {}
    for face in global_faces:
        a, b, c = triangles[face]
        for first, second in ((a,b),(b,c),(c,a)):
            edge_faces[(min(int(first),int(second)), max(int(first),int(second)))] = int(face)

    params = reference.metadata["parameters"]
    count = int(params["circumference_segments"])
    ring_count = len(reference.metadata["ring_regions"])
    toe_first_ring = reference.metadata["ring_regions"].index("toe")
    boundary_ring = toe_first_ring-1
    seam_start = ring_count*count
    if len(reference.reference_positions_mm) != seam_start+count//2+1:
        raise ValueError("Reference material ring layout differs from the declared topology")
    component_vertex_ids = np.flatnonzero(components == chosen_component)
    component_vertices = unposed[component_vertex_ids]
    uniform_s, path, source_anchors, target_anchors, tip = _path_for_component(
        unposed, triangles[global_faces], component_vertices, cuff, params)
    source_s = reference.material_uv[:seam_start:count, 1]*source_anchors[-1]
    target_s = _pchip(source_anchors, target_anchors, source_s)

    result_faces = np.empty(len(reference.reference_positions_mm), dtype=np.int32)
    result_bary = np.empty((len(result_faces), 3))
    method = np.empty(len(result_faces), dtype=np.uint8)
    ring_index = np.full(len(result_faces), -1, dtype=np.int32)
    ray_misses = 0
    upper_projection_max_mm = 0.
    continuation_projection_max_mm = 0.
    cuff_center = path[0]
    upper_transition_s = 70.
    previous_major = None
    for row in range(toe_first_ring):
        center, tangent = _path_sample(uniform_s, path, target_s[row])
        major = np.array([0., -1., 0.]) if previous_major is None else previous_major
        major -= tangent*(major@tangent)
        if np.linalg.norm(major) < 1e-5:
            major = np.array([0., 0., 1.])-tangent*tangent[2]
        major = _unit(major)
        minor = _unit(np.cross(-tangent, major))
        ids = slice(row*count, (row+1)*count)
        ring_index[ids] = row
        reference_ring = reference.reference_positions_mm[ids]
        if row == 0:
            result_faces[ids], result_bary[ids] = _cuff_binding(
                reference_ring, unposed, cuff, triangles, edge_faces, major, minor)
            method[ids] = 0
            cuff_points = np.einsum("ni,nij->nj", result_bary[ids], unposed[triangles[result_faces[ids]]])
            previous_major = major.copy()
            continue
        if target_s[row] <= upper_transition_s:
            # The authored cuff is tilted by about 16 mm. A flat horizontal
            # ray ring would cross the open boundary at some meridians. Carry
            # each cuff-edge point downward on its own local height and bind
            # the continuation explicitly to the same target component.
            fade = min(1., target_s[row]/upper_transition_s)
            fade = fade*fade*(3.-2.*fade)
            for k in range(count):
                guess = cuff_points[k]+(center-cuff_center)
                guess[2] -= fade*(cuff_points[k, 2]-cuff_center[2])
                hit, _, local_face, distance = bvh.find_nearest(Vector(guess))
                if hit is None or distance > 24.:
                    raise ValueError(f"Upper cuff continuation failed at ring {row}, meridian {k}: {distance}")
                upper_projection_max_mm = max(upper_projection_max_mm, float(distance))
                face = int(global_faces[local_face])
                material = row*count+k
                result_faces[material] = face
                result_bary[material] = _barycentric(np.asarray(hit), unposed[triangles[face]])
                method[material] = 3
            previous_major = major.copy()
            continue
        # Parameterize the complete target intersection by physical perimeter
        # rather than propagating each meridian's previous radial length. The
        # latter can pinch neighboring material IDs onto one sole triangle.
        dense_count = max(dense_angles, 2*count)
        dense_hit = np.empty((dense_count, 3))
        for j in range(dense_count):
            angle = 2*math.pi*j/dense_count
            direction = _unit(math.cos(angle)*major+math.sin(angle)*minor)
            hit, hit_normal, _, _ = bvh.ray_cast(Vector(center), Vector(direction), 160.)
            if hit is None or hit_normal.dot(Vector(direction)) <= 0:
                ray_misses += 1
                raise ValueError(f"No outward target section hit at ring {row}, sample {j}")
            dense_hit[j] = np.asarray(hit)
        dense_fraction = _arc_fraction(dense_hit)
        source_fraction = _arc_fraction(reference_ring)[:-1]
        for k in range(count):
            edge = int(np.clip(np.searchsorted(dense_fraction, source_fraction[k], side="right")-1,
                               0, dense_count-1))
            t = (source_fraction[k]-dense_fraction[edge])/(dense_fraction[edge+1]-dense_fraction[edge])
            guess = (1-t)*dense_hit[edge]+t*dense_hit[(edge+1)%dense_count]
            hit, _, local_face, distance = bvh.find_nearest(Vector(guess))
            if hit is None or distance > 5.:
                raise ValueError(f"Target perimeter binding failed at ring {row}, meridian {k}: {distance}")
            continuation_projection_max_mm = max(continuation_projection_max_mm, float(distance))
            method_code = 1
            face = int(global_faces[local_face])
            material = row*count+k
            result_faces[material] = face
            result_bary[material] = _barycentric(np.asarray(hit), unposed[triangles[face]])
            method[material] = method_code
        previous_major = major.copy()

    # The full toe is mapped as one disk. The final straight-foot ring is its
    # fixed material boundary; toe rings and seam receive original-target
    # face/bary bindings from the clipped target cap parameterization.
    seam_count = count//2+1
    for row in range(toe_first_ring, ring_count):
        ring_index[row*count:(row+1)*count] = row

    result_faces, result_bary, toe_disk_report = register_toe_disk(
        reference, unposed, triangles, global_faces, result_faces, result_bary,
        toe_start_ring=boundary_ring)
    method[toe_first_ring*count:] = 6

    if not np.isfinite(result_bary).all() or np.any(result_bary < -1e-12) or np.any(np.abs(result_bary.sum(axis=1)-1)>1e-10):
        raise ValueError("Target material binding is not finite barycentric data")
    bound_unposed = np.einsum("ni,nij->nj", result_bary, unposed[triangles[result_faces]])
    bound_current = np.einsum("ni,nij->nj", result_bary, current[triangles[result_faces]])
    pose_repair_report = {"attempted": False, "accepted": False}
    pose_dots = _orientation_dots(bound_current, reference.triangles, current, triangles, result_faces)
    pose_bad = np.flatnonzero(pose_dots <= 0)
    if len(pose_bad) and len(pose_bad) <= 8:
        movable = np.unique(reference.triangles[pose_bad])
        fixed = np.setdiff1d(np.arange(len(bound_current)), movable)
        repaired_current, proposed_face, proposed_bary, local_report = repair_orientation_on_surface(
            bound_current, reference.triangles, current, triangles, result_faces,
            result_bary, global_faces, fixed_vertices=fixed,
            max_iterations=160, maximum_step_mm=.8)
        proposed_unposed = np.einsum("ni,nij->nj", proposed_bary,
                                    unposed[triangles[proposed_face]])
        proposed_current = np.einsum("ni,nij->nj", proposed_bary,
                                    current[triangles[proposed_face]])
        dot_u = _orientation_dots(proposed_unposed, reference.triangles, unposed, triangles,
                                  proposed_face)
        dot_c = _orientation_dots(proposed_current, reference.triangles, current, triangles,
                                  proposed_face)
        maximum_move = float(np.linalg.norm(repaired_current-bound_current, axis=1).max())
        accepted = bool(np.all(dot_u > 0) and np.all(dot_c > 0) and maximum_move <= .5)
        pose_repair_report = {"attempted": True, "accepted": accepted,
                              "initial_nonpositive_faces": int(len(pose_bad)),
                              "affected_material_point_ids": reference.material_point_ids[movable].tolist(),
                              "maximum_current_reassignment_mm": maximum_move,
                              "proposed_minimum_unposed_dot": float(dot_u.min()),
                              "proposed_minimum_current_dot": float(dot_c.min()),
                              "local_signed_area_report": local_report}
        if accepted:
            result_faces, result_bary = proposed_face, proposed_bary
            bound_unposed, bound_current = proposed_unposed, proposed_current
            method[movable] = 7
            pose_repair_report["same_face_margin_search"] = _improve_same_face_pose_margin(
                reference.triangles, unposed, current, triangles, result_faces, result_bary,
                bound_unposed, bound_current, movable)
    final_rings = bound_unposed[:seam_start].reshape(ring_count, count, 3)
    ring_motion_max_mm = float(np.linalg.norm(np.diff(final_rings, axis=0), axis=2).max())
    neighbor_jump_max_mm = float(np.linalg.norm(np.roll(final_rings, -1, axis=1)-final_rings, axis=2).max())
    binding = {
        "target_triangle_index": result_faces,
        "target_barycentric": result_bary,
        "normal_offset_mm": np.zeros(len(result_faces)),
        "unposed_positions_mm": bound_unposed,
        "current_positions_mm": bound_current,
        "target_component": np.full(len(result_faces), chosen_component, dtype=np.int32),
        "material_point_ids": reference.material_point_ids.copy(),
        "binding_method_code": method,
        "reference_ring_index": ring_index,
        "source_centerline_arclength_mm": np.r_[source_s.repeat(count), source_anchors[-1]*np.ones(seam_count)],
        "target_centerline_arclength_mm": np.r_[target_s.repeat(count), target_anchors[-1]*np.ones(seam_count)],
    }

    def make_pair(positions, pose):
        metadata = copy.deepcopy(reference.metadata)
        metadata.update(fit_status="authored_geometric_candidate", target_pose=pose,
                        target_reference_version="authored-target-v2", target_component=int(chosen_component),
                        fitting_method="actual authored shell triangle/bary binding via continuous ring path",
                        dressing_equilibrium_solved=False, yarn_equilibrium_solved=False,
                        material_calibrated=False, physical_contact_passed=False)
        return GarmentMeshPair(reference.reference_positions_mm, positions,
                               reference.triangles, reference.material_uv,
                               reference.material_point_ids, metadata,
                               face_material_uv=reference.face_material_uv)

    pair_u = make_pair(bound_unposed, "unposed_auxiliary")
    pair_c = make_pair(bound_current, "frame_996_current")
    q_u = _quality(reference, bound_unposed, unposed, triangles, result_faces, bvh,
                   target_face_indices=global_faces)
    current_bvh = _mesh_bvh(current, triangles[global_faces])
    q_c = _quality(reference, bound_current, current, triangles, result_faces, current_bvh,
                   target_face_indices=global_faces)
    report = {
        "side": side, "target_component": int(chosen_component),
        "method_codes": {"0": "exact cuff boundary edge", "1": "target section perimeter arclength remap",
                         "3": "bounded tilted-cuff surface continuation",
                         "6": "positive-harmonic target toe disk",
                         "7": "bounded local frame-996 orientation repair"},
        "source_arclength_anchors_mm": source_anchors.tolist(),
        "target_arclength_anchors_mm": target_anchors.tolist(),
        "cuff_target_centroid_mm": unposed[cuff].mean(axis=0).tolist(),
        "cuff_target_vertex_count": int(len(cuff)),
        "target_tip_mm": tip.tolist(),
        "toe_disk_registration": toe_disk_report,
        "pose_orientation_repair": pose_repair_report,
        "ray_misses": int(ray_misses),
        "upper_transition_arclength_mm": upper_transition_s,
        "upper_projection_max_mm": upper_projection_max_mm,
        "continuation_projection_max_mm": continuation_projection_max_mm,
        "ring_motion_max_mm": ring_motion_max_mm,
        "ring_max_neighbor_jump_mm": neighbor_jump_max_mm,
        "unposed": _summary(q_u), "current": _summary(q_c),
        "scope": "geometric candidate; independent target normals are diagnostic, not a no-self-intersection proof",
    }
    return RegistrationResult(binding, pair_u, pair_c, report, q_u, q_c)
