"""Independent X0 -> posed body geometric binding, NOT a dressing solver.

Build material rings on the body's unposed domain, bind by fixed body triangle
barycentrics, then evaluate those bindings on the posed body. This avoids
sweeping a thick tube around an acute posed knee. Circumferential allocation
uses X0 physical arc length, rather than treating ellipse angle as material
length. The foot envelope bridges separate toes; final contact correction is
kinematic. No force, friction, yarn equilibrium or calibration is inferred.
"""
from dataclasses import dataclass
import copy
import math

import numpy as np

from .garment_data import GarmentMeshPair


def unit(value):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError("zero direction in geometric fitting")
    return value / length


def barycentric(point, triangle):
    a, b, c = triangle
    e, f, q = b-a, c-a, point-a
    ee, ef, ff = e@e, e@f, f@f
    det = ee*ff-ef*ef
    if det <= 1e-14:
        raise ValueError("degenerate body binding triangle")
    v = (ff*(q@e)-ef*(q@f))/det
    w = (ee*(q@f)-ef*(q@e))/det
    return np.array((1-v-w, v, w))


class BodySurface:
    """Blender BVH in world millimetres, with fixed triangle provenance."""
    def __init__(self, positions_mm, triangles, current_positions_mm=None):
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        self.positions = np.asarray(positions_mm, dtype=float)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.current = self.positions if current_positions_mm is None else np.asarray(current_positions_mm, dtype=float)
        self.bvh = BVHTree.FromPolygons([Vector(row) for row in self.positions],
                                       self.triangles.tolist(), all_triangles=True)
        p = self.current[self.triangles]
        area_normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        self.current_normals = unit(area_normals)
        vertex_normals = np.zeros_like(self.current)
        for corner in range(3):
            np.add.at(vertex_normals, self.triangles[:,corner], area_normals)
        lengths = np.linalg.norm(vertex_normals,axis=1)
        active = lengths > 1e-12
        vertex_normals[active] /= lengths[active,None]
        self.current_vertex_normals = vertex_normals
        self.miss_count = 0
        self.reverse_ray_count = 0

    def ray_bind(self, origin, direction):
        from mathutils import Vector
        direction = unit(direction)
        point, normal, face, distance = self.bvh.ray_cast(Vector(origin), Vector(direction), 600.)
        if point is None or normal.dot(Vector(direction)) <= 0:
            self.reverse_ray_count += 1
            point, normal, face, distance = self.bvh.ray_cast(Vector(origin+direction*600), Vector(-direction), 1200.)
        if point is None:
            self.miss_count += 1
            raise ValueError("body ray missed: no silent nearest-point reassignment")
        weights = barycentric(np.array(point), self.positions[self.triangles[face]])
        current_point = weights @ self.current[self.triangles[face]]
        normal1 = unit(weights @ self.current_vertex_normals[self.triangles[face]])
        return np.array(point), current_point, normal1, int(face), weights

    def nearest(self, points):
        from mathutils import Vector
        nearest = np.empty_like(points)
        normals = np.empty_like(points)
        indices = np.empty(len(points), dtype=np.int64)
        for index, point in enumerate(points):
            hit, normal, triangle, distance = self.bvh.find_nearest(Vector(point))
            if hit is None:
                raise ValueError("empty body collision surface")
            nearest[index] = hit
            normals[index] = normal
            indices[index] = triangle
        signed = np.einsum("ij,ij->i", points-nearest, normals)
        return nearest, normals, signed, indices


def foot_envelope(target, side, *, posed=False):
    """Convex foot domain with original body vertex IDs, not per-toe wrap."""
    import bmesh
    suffix = ".l" if side == "left" else ".r"
    groups = list(target["group_names"])
    weights = target["group_weights"][:, groups.index("foot"+suffix)] + target["group_weights"][:, groups.index("toes_01"+suffix)]
    selected = np.flatnonzero(weights > .4)
    bm = bmesh.new()
    source = bm.verts.layers.int.new("source_id")
    positions = target["positions_mm"] if posed else target["reference_positions_mm"]
    for index in selected:
        vertex = bm.verts.new(positions[index])
        vertex[source] = int(index)
    bm.verts.ensure_lookup_table()
    result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    faces = [item for item in result["geom"] if isinstance(item, bmesh.types.BMFace)]
    bmesh.ops.triangulate(bm, faces=faces)
    triangles = np.array([[v[source] for v in face.verts] for face in bm.faces], dtype=np.int64)
    bm.free()
    p = positions[triangles]
    center = positions[selected].mean(axis=0)
    flip = np.einsum("ij,ij->i", np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), p.mean(axis=1)-center) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return BodySurface(positions, triangles, target["positions_mm"])


def _polyline(points, count):
    points = np.asarray(points)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distance = np.linspace(0, lengths[-1], count)
    return np.column_stack([np.interp(distance, lengths, points[:, axis]) for axis in range(3)])


def _frame_centers(target, side, params):
    landmarks = target["landmarks_by_side"][side]["rest"]
    cuff, mid, knee, calf, ankle, ball = [np.asarray(landmarks[key], dtype=float) for key in ("cuff", "thighmid", "knee", "calfmid", "ankle", "ball")]
    down, forward = unit(ankle-calf), unit(ball-ankle)
    # A sharp anatomical joint center is not a regular tube centerline. A
    # broad rest-domain heel transition avoids folding the inner meridians.
    heel_trim = min(75., .58*np.linalg.norm(ball-ankle))
    heel_before = ankle-down*heel_trim
    heel_after = ankle+forward*heel_trim
    leg = _polyline((cuff, mid, knee, calf, heel_before), params.leg_segments+1)
    t = np.arange(1, params.heel_segments+1) / params.heel_segments
    heel = (1-t[:, None])**2*heel_before + 2*(1-t[:, None])*t[:, None]*ankle + t[:, None]**2*heel_after
    t = np.arange(1, params.foot_segments+1) / params.foot_segments
    foot = heel_after+(ball-heel_after)*t[:, None]
    centers = np.vstack((leg, heel, foot))
    tangent = unit(np.gradient(centers, axis=0))
    major = forward - tangent[0]*(forward@tangent[0])
    major = unit(major)
    frames = []
    for direction in tangent:
        major = unit(major-direction*(major@direction))
        minor = unit(np.cross(-direction, major))
        frames.append((major.copy(), minor, direction))
    return centers, np.asarray(frames), ball, unit(landmarks["toe_direction"])


def _arc_fractions(points):
    length = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    return np.r_[0., np.cumsum(length)] / length.sum()


def _redistribute_ring(reference, dense_reference, dense_current, dense_normals):
    """Match cumulative physical arc length, phase anchored to the first point."""
    fractions = _arc_fractions(reference)[:-1]
    dense_fraction = _arc_fractions(dense_reference)
    current = np.vstack((dense_current, dense_current[0]))
    normals = np.vstack((dense_normals, dense_normals[0]))
    # Interpolation is a geometric binding mixture; contact is checked later.
    points = np.column_stack([np.interp(fractions, dense_fraction, current[:, axis]) for axis in range(3)])
    outward = unit(np.column_stack([np.interp(fractions, dense_fraction, normals[:, axis]) for axis in range(3)]))
    return points, outward


@dataclass
class FitResult:
    pair: GarmentMeshPair
    report: dict
    contact_samples: dict


def fit_reference_to_body(reference, params, target, side, *, clearance_mm=.6, tolerance_mm=.01, progress=print, diagnostic_path=None,
                          report_contact_conflicts=False):
    """Fixed-topology kinematic fit plus finite-sample clearance correction."""
    reference.validate()
    side_index = 0 if side == "left" else 1
    rows = np.all(target["vertex_side"][target["triangles"]] == side_index, axis=1)
    rest_skin = BodySurface(target["reference_positions_mm"], target["triangles"][rows], target["positions_mm"])
    foot_hull = foot_envelope(target, side)
    posed_foot_hull = foot_envelope(target, side, posed=True)
    collision = BodySurface(target["positions_mm"], target["triangles"][rows])
    other_rows = np.all(target["vertex_side"][target["triangles"]] == 1-side_index, axis=1)
    other_collision = BodySurface(target["positions_mm"], target["triangles"][other_rows])
    centers, frames, ball, toe_forward = _frame_centers(target, side, params)
    circumference = params.circumference_segments
    dense_count = max(256, circumference*4)
    angles = 2*math.pi*np.arange(dense_count)/dense_count
    current = np.empty_like(reference.reference_positions_mm)
    director = np.empty_like(current)
    base_count = len(centers)
    ring_count = len(reference.metadata["ring_regions"])
    progress(f"{side}: binding {ring_count} material rings in unposed body domain")
    for row in range(base_count):
        major, minor, tangent = frames[row]
        dense0, dense1, normals = [], [], []
        # Convex envelope across toes; full skin elsewhere. Hull and skin use
        # the same original body vertex IDs and posed evaluation.
        foot_fraction = np.clip((row-params.leg_segments-params.heel_segments)/params.foot_segments, 0., 1.)
        hull_blend = foot_fraction*foot_fraction*(3-2*foot_fraction)
        for angle in angles:
            ray = major*math.cos(angle)+minor*math.sin(angle)
            p0, p1, normal, _, _ = rest_skin.ray_bind(centers[row], ray)
            if hull_blend > 0:
                h0, h1, hn, _, _ = foot_hull.ray_bind(centers[row], ray)
                p0, p1 = p0*(1-hull_blend)+h0*hull_blend, p1*(1-hull_blend)+h1*hull_blend
                normal = unit(normal*(1-hull_blend)+hn*hull_blend)
            dense0.append(p0); dense1.append(p1); normals.append(normal)
        ids = slice(row*circumference, (row+1)*circumference)
        points, outward = _redistribute_ring(reference.reference_positions_mm[ids], np.asarray(dense0), np.asarray(dense1), np.asarray(normals))
        current[ids], director[ids] = points+clearance_mm*outward, outward
        if row == base_count-1:
            base_angles = np.interp(_arc_fractions(reference.reference_positions_mm[ids])[:-1],
                                    _arc_fractions(np.asarray(dense0)), np.r_[angles, 2*math.pi])
    # A distributed seam cap: forward dome rays bridge separate toes. It is
    # intentionally an enclosing sock toe, not independent skin shrinkwrap.
    major, minor, _ = frames[-1]
    # Keep the cap tangent continuous with the final foot ring. The envelope
    # determines its actual front; a toe control bone may end outside skin.
    toe_forward = unit(frames[-1, 2])
    major = unit(major-toe_forward*(major@toe_forward))
    minor = unit(np.cross(-toe_forward, major))
    tip0, _, _, _, _ = foot_hull.ray_bind(ball, toe_forward)
    toe_length = np.linalg.norm(tip0-ball)
    radius = params.foot_width_mm*.55
    seam = reference.metadata["toe_seam_material_point_ids"]
    def line_fraction(points):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        return distance/distance[-1]
    dense_seam_angles = np.linspace(0., math.pi, dense_count//2+1)
    dense_seam0 = np.asarray([foot_hull.ray_bind(ball, major*radius*math.cos(angle)+toe_forward*toe_length)[0]
                             for angle in dense_seam_angles])
    # One consistent meridian allocation for the entire final cap. Re-solving
    # arc allocation separately on every nearly flattened ring lets a seam
    # endpoint overtake its adjacent material point on a flat hull facet.
    seam_angles = np.interp(line_fraction(reference.reference_positions_mm[seam]),
                            line_fraction(dense_seam0), dense_seam_angles)
    cap_angles = np.r_[seam_angles, 2*math.pi-seam_angles[-2:0:-1]]
    for row in range(base_count, ring_count):
        fraction = (row-base_count+1)/params.toe_segments
        blend = min(1., fraction*2)
        blend = blend*blend*(3-2*blend)
        ring_angles = base_angles*(1-blend)+cap_angles*blend
        for k, angle in enumerate(ring_angles):
            direction = (major*radius*math.cos(angle)+minor*radius*math.sin(angle)*math.cos(fraction*math.pi/2)
                         +toe_forward*toe_length*fraction)
            _, p1, normal, _, _ = foot_hull.ray_bind(ball, direction)
            vertex = row*circumference+k
            current[vertex], director[vertex] = p1+clearance_mm*normal, normal
    for vertex, angle in zip(seam, seam_angles):
        _, p1, normal, _, _ = foot_hull.ray_bind(ball, major*radius*math.cos(angle)+toe_forward*toe_length)
        current[vertex], director[vertex] = p1+clearance_mm*normal, normal
    # Barycentric deformation of a rest convex hull need not remain convex
    # when toe bones move. Correct to the actual posed convex foot envelope.
    foot_start = (params.leg_segments+params.heel_segments)*circumference
    posed_ball = np.asarray(target["landmarks_by_side"][side]["current"]["ball"])
    # Radial envelope offset is continuous across hull facet boundaries.
    # Per-facet normal offsets would create small backwards steps where the
    # final distributed seam has very short longitudinal edges.
    for vertex in range(foot_start, len(current)):
        radial = unit(current[vertex]-clearance_mm*director[vertex]-posed_ball)
        _, surface_point, surface_normal, _, _ = posed_foot_hull.ray_bind(posed_ball, radial)
        fraction = np.clip((vertex//circumference-params.leg_segments-params.heel_segments)/params.foot_segments, 0., 1.)
        blend = fraction*fraction*(3-2*fraction)
        current[vertex] = current[vertex]*(1-blend)+(surface_point+clearance_mm*radial)*blend
        director[vertex] = surface_normal
    # Distributed seam endpoints can put a tiny triangle exactly in a hull
    # meridian plane. Resolve that geometric singularity with a bounded
    # tangential correction; never change triangle winding, IDs or UV.
    orientation_displacement = np.zeros_like(current)
    orientation_repairs = 0
    for repair_iteration in range(60):
        faces = current[reference.triangles]
        area_normals = np.cross(faces[:,1]-faces[:,0], faces[:,2]-faces[:,0])
        _, body_normals, _, _ = collision.nearest(faces.mean(axis=1))
        dots = np.einsum("ij,ij->i", unit(area_normals), body_normals)
        bad = np.flatnonzero(dots <= 0.)
        if report_contact_conflicts:
            bad = bad[bad >= (ring_count-3)*circumference*2]
        if not len(bad):
            break
        if len(bad) > len(reference.triangles)*.03:
            progress(f"{side}: constrained pre-contact inward counts {dict(zip(*np.unique(bad//(circumference*2),return_counts=True)))}")
            if diagnostic_path is not None:
                np.savez_compressed(diagnostic_path, current_positions_mm=current, reference_positions_mm=reference.reference_positions_mm,
                                    triangles=reference.triangles, face_outward_dot=dots, body_face_normals=body_normals,
                                    centers=centers, frames=frames)
            raise ValueError("large folded surface cannot be repaired with local geometric orientation constraints")
        displacement = np.zeros_like(current)
        counts = np.zeros(len(current))
        for face_index in bad:
            a, b, c = reference.triangles[face_index]
            n = body_normals[face_index]
            gradients = np.array([np.cross(current[b]-current[c], n), np.cross(current[c]-current[a], n), np.cross(current[a]-current[b], n)])
            gain = np.sum(gradients*gradients)
            if gain < 1e-16:
                raise ValueError("singular cap orientation constraint")
            desired = .1*np.linalg.norm(area_normals[face_index])
            delta = gradients*(desired-area_normals[face_index]@n)/gain
            length = np.linalg.norm(delta,axis=1).max()
            if length > .08:
                delta *= .08/length
            np.add.at(displacement, [a,b,c], delta)
            np.add.at(counts, [a,b,c], 1.)
            orientation_repairs += 1
        active = counts > 0
        delta = displacement[active]/counts[active,None]
        current[active] += delta
        orientation_displacement[active] += delta
        if repair_iteration % 10 == 0:
            progress(f"{side}: local orientation constraints {repair_iteration+1}, {len(bad)} faces")
    # Vertex projection alone misses penetration by flat face interiors. Test
    # all vertices, unique edges, and centroids, then lift incident vertices.
    edges = np.unique(np.sort(np.vstack((reference.triangles[:, [0,1]], reference.triangles[:, [1,2]], reference.triangles[:, [2,0]])), axis=1), axis=0)
    corrections = []
    for iteration in range(16):
        samples = np.vstack((current, current[edges].mean(axis=1), current[reference.triangles].mean(axis=1)))
        nearest, normals, signed, body_indices = collision.nearest(samples)
        minimum = float(signed.min())
        corrections.append(minimum)
        if minimum >= -tolerance_mm or report_contact_conflicts:
            break
        needed = np.maximum(clearance_mm*.35-signed, 0.)
        displacement = np.zeros_like(current)
        weight = np.zeros(len(current))
        offset = len(current)
        for ids, delta, normal in ((np.arange(len(current))[:,None], needed[:offset], normals[:offset]),
                                   (edges, needed[offset:offset+len(edges)], normals[offset:offset+len(edges)]),
                                   (reference.triangles, needed[offset+len(edges):], normals[offset+len(edges):])):
            active = delta > 0
            for corner in range(ids.shape[1]):
                np.add.at(displacement, ids[active, corner], normal[active]*delta[active, None]*1.25)
                np.add.at(weight, ids[active, corner], 1.)
        active = weight > 0
        current[active] += displacement[active]/weight[active,None]
        worst = int(np.argmin(signed))
        progress(f"{side}: contact correction {iteration+1}, sampled minimum {minimum:.4f} mm at {samples[worst].round(2).tolist()}")
    else:
        raise ValueError(f"{side}: contact did not converge; minimum {minimum:.5f} mm")
    faces = current[reference.triangles]
    face_normals = unit(np.cross(faces[:,1]-faces[:,0], faces[:,2]-faces[:,0]))
    face_body_normals = normals[len(current)+len(edges):]
    outward_dot = np.einsum("ij,ij->i", face_normals, face_body_normals)
    conflicts = bool(np.any(outward_dot <= 0) or np.any(signed < -tolerance_mm))
    metadata = copy.deepcopy(reference.metadata)
    metadata.update(fit_status="geometric_fit_with_contact_conflicts" if conflicts else "geometric_fit", fit_method="unposed body material-ring arc binding + posed barycentric evaluation + sampled clearance correction",
                    side=side, dressing_equilibrium_solved=False, material_calibrated=False,
                    yarn_equilibrium_solved=False, stress_free_assumed=False,
                    toe_method="foot convex envelope with distributed forward dome seam",
                    circumferential_allocation="equal relative physical arc length per ring; kinematic assumption",
                    source_body_object=target.get("body_name", "body_base"), source_frame=target.get("frame", 996),
                    target_pose=target.get("target_pose", "current"), source_pose_contact_conflicts_reported=report_contact_conflicts)
    pair = GarmentMeshPair(reference.reference_positions_mm, current, reference.triangles,
                           reference.material_uv, reference.material_point_ids, metadata,
                           face_material_uv=reference.face_material_uv)
    if np.any(outward_dot <= 0) and not report_contact_conflicts:
        bad = np.flatnonzero(outward_dot <= 0)
        if diagnostic_path is not None:
            np.savez_compressed(diagnostic_path, current_positions_mm=current, reference_positions_mm=reference.reference_positions_mm,
                                triangles=reference.triangles, face_outward_dot=outward_dot, body_face_normals=face_body_normals,
                                centers=centers, frames=frames)
        progress(f"{side}: inward counts by material row {dict(zip(*np.unique(bad//(circumference*2),return_counts=True)))}")
        raise ValueError(f"{side}: {len(bad)} inward triangles; minimum normal dot {outward_dot.min():.6g}; first={bad[:10].tolist()}")
    if not np.any(outward_dot <= 0):
        pair.current_face_directors = face_body_normals.copy()
    pair.validate()
    _, _, other_signed, _ = other_collision.nearest(samples)
    samples_dict = dict(sample_positions_mm=samples, closest_body_positions_mm=nearest,
                        closest_body_normals=normals, signed_distance_mm=signed,
                        closest_body_triangle_index=np.flatnonzero(rows)[body_indices],
                        other_leg_signed_distance_mm=other_signed,
                        sample_kind=np.r_[np.zeros(len(current), dtype=int), np.ones(len(edges), dtype=int), np.full(len(pair.triangles), 2, dtype=int)],
                        sample_source_index=np.r_[np.arange(len(current)), np.arange(len(edges)), np.arange(len(pair.triangles))],
                        face_outward_dot=outward_dot)
    report = dict(schema="stocking.geometric_fit_report", schema_version=1, side=side,
                  fit_status=metadata["fit_status"], length_unit="mm", tolerance_mm=tolerance_mm,
                  requested_clearance_mm=clearance_mm, miss_count=rest_skin.miss_count+foot_hull.miss_count,
                  reverse_ray_count=rest_skin.reverse_ray_count+foot_hull.reverse_ray_count,
                  sample_count=len(samples), sample_min_signed_distance_mm=float(signed.min()),
                  sample_penetration_count=int(np.count_nonzero(signed < -tolerance_mm)),
                  sampled_clearance_percentiles_mm=np.percentile(signed, [0, 5, 50, 95, 100]).tolist(),
                  samples_only=True, triangles_outward_checked=True, minimum_face_outward_dot=float(outward_dot.min()),
                  inward_triangle_count=int(np.count_nonzero(outward_dot <= 0)),
                  source_pose_contact_conflicts_reported=report_contact_conflicts,
                  self_intersection_check_status="not_computed", correction_minimum_history_mm=corrections,
                  collision_component=side+" body skin", other_leg_sample_penetration_count=int(np.count_nonzero(other_signed < -tolerance_mm)),
                  other_leg_minimum_signed_distance_mm=float(other_signed.min()),
                  toe_orientation_constraint_updates=orientation_repairs,
                  maximum_toe_orientation_correction_mm=float(np.linalg.norm(orientation_displacement,axis=1).max()),
                  dressing_equilibrium_solved=False, material_calibrated=False, yarn_equilibrium_solved=False,
                  scope="sampled own-leg body clearance, separate other-leg penetration diagnostic, fixed mesh/UV/IDs and endpoint kinematics; no continuous collision, shell self-contact, friction or physical dressing")
    return FitResult(pair, report, samples_dict)
