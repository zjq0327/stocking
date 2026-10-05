"""Fit an independent stocking X0 to one static leg-and-foot mesh.

The fit is a geometric shell construction in millimetres.  It keeps physical
vertex IDs, triangle order, and material UV from X0; it does not simulate
dressing, fabric forces, or motion to another pose.  Blender is needed only
when this function runs (for its BVH and convex-hull implementations).
"""

import copy
import math

import numpy as np

from .garment_data import GarmentMeshPair
from .geometric_fitting import BodySurface, _polyline, _redistribute_ring, unit
from .procedural_garment import ProceduralGarmentParameters


def _required_vector(analysis, key):
    value = np.asarray(analysis[key], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"single-leg landmark {key} must be a finite 3D point")
    return value


def _foot_hull(positions, indices):
    """Make a closed convex toe/foot envelope from original skin vertices."""
    import bmesh

    indices = np.unique(np.asarray(indices, dtype=np.int64))
    if len(indices) < 8 or np.any(indices < 0) or np.any(indices >= len(positions)):
        raise ValueError("foot_vertex_indices must contain at least eight valid vertices")
    bm = bmesh.new()
    source = bm.verts.layers.int.new("source_id")
    try:
        for index in indices:
            vert = bm.verts.new(tuple(positions[index]))
            vert[source] = int(index)
        bm.verts.ensure_lookup_table()
        result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        hull_faces = [item for item in result["geom"] if isinstance(item, bmesh.types.BMFace)]
        if not hull_faces:
            raise ValueError("foot vertices do not span a three-dimensional convex envelope")
        bmesh.ops.triangulate(bm, faces=hull_faces)
        triangles = np.asarray([[vertex[source] for vertex in face.verts] for face in bm.faces], dtype=np.int64)
    finally:
        bm.free()
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 4:
        raise ValueError("foot convex envelope has no usable triangles")
    face_points = positions[triangles]
    center = positions[indices].mean(axis=0)
    normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    flip = np.einsum("ij,ij->i", normals, face_points.mean(axis=1) - center) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return BodySurface(positions, triangles)


def _frame_centers(analysis, params):
    cuff = _required_vector(analysis, "cuff_center_mm")
    ankle = _required_vector(analysis, "ankle_center_mm")
    ball = _required_vector(analysis, "ball_center_mm")
    down = unit(_required_vector(analysis, "down_axis"))
    forward = unit(_required_vector(analysis, "foot_axis"))
    if abs(float(down @ forward)) > .92:
        raise ValueError("leg and foot axes are too nearly parallel for an L-shaped sock")
    centerline = np.asarray(analysis["leg_centerline_mm"], dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 2 or not np.isfinite(centerline).all():
        raise ValueError("leg_centerline_mm must be finite points ordered from cuff to ankle")
    if np.linalg.norm(centerline[0] - cuff) > max(15., .08 * np.linalg.norm(ankle - cuff)):
        raise ValueError("leg_centerline_mm must start at the cuff center")
    foot_reach = float((ball - ankle) @ forward)
    if foot_reach < 15.:
        raise ValueError("ball center must lie in front of ankle along foot_axis")
    ankle_to_ball = unit(ball - ankle)
    heel_trim = min(75., .58 * foot_reach)
    # The cuff normal is only a global shaft direction.  On a posed, bent leg
    # it can leave the calf entirely near the ankle.  Locate the heel entry
    # along the measured cross-section centerline instead.
    centerline = centerline.copy()
    centerline[0] = cuff
    centerline[-1] = ankle
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))]
    if arc[-1] <= heel_trim + 20. or np.any(np.diff(arc) < 1e-6):
        raise ValueError("leg centerline is too short or contains repeated points")
    heel_arc = arc[-1] - heel_trim
    heel_before = np.array([np.interp(heel_arc, arc, centerline[:, axis]) for axis in range(3)])
    # Keep the quadratic heel tangent aligned with the following foot
    # centerline.  The toe axis can differ substantially from ankle-to-ball
    # direction on a bent foot; using it here makes the first foot ring kink.
    heel_after = ankle + ankle_to_ball * heel_trim
    leg_knots = centerline[arc < heel_arc - 1e-6]
    leg_knots = np.vstack((leg_knots, heel_before))
    if np.any(np.linalg.norm(np.diff(leg_knots, axis=0), axis=1) < 1e-6):
        raise ValueError("leg centerline has repeated points near the ankle")
    leg = _polyline(leg_knots, params.leg_segments + 1)
    t = np.arange(1, params.heel_segments + 1) / params.heel_segments
    heel = ((1-t[:, None])**2 * heel_before
            + 2 * (1-t[:, None]) * t[:, None] * ankle
            + t[:, None]**2 * heel_after)
    t = np.arange(1, params.foot_segments + 1) / params.foot_segments
    foot = heel_after + t[:, None] * (ball - heel_after)
    centers = np.vstack((leg, heel, foot))
    # Wide secants prevent adjacent section planes from pivoting through the
    # same skin vertex on a sharply posed upper leg.
    indices = np.arange(len(centers))
    tangent = unit(centers[np.minimum(indices + 3, len(centers) - 1)]
                   - centers[np.maximum(indices - 3, 0)])
    major = forward - tangent[0] * float(forward @ tangent[0])
    if np.linalg.norm(major) < .1:
        basis = np.eye(3)[int(np.argmin(np.abs(tangent[0])))]
        major = np.cross(tangent[0], basis)
    major = unit(major)
    frames = []
    for direction in tangent:
        major = unit(major - direction * float(major @ direction))
        minor = unit(np.cross(-direction, major))
        frames.append((major.copy(), minor, direction))
    return centers, np.asarray(frames), ball


def _arc_fractions(points):
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if not np.isfinite(lengths).all() or lengths.sum() < 1e-9:
        raise ValueError("a material ring or toe seam has zero length")
    return np.r_[0., np.cumsum(lengths)] / lengths.sum()


def _open_arc_fractions(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if lengths.sum() < 1e-9:
        raise ValueError("toe seam collapsed to a point")
    return np.r_[0., np.cumsum(lengths)] / lengths.sum()


def _samples(current, triangles, edges):
    return np.vstack((current, current[edges].mean(axis=1), current[triangles].mean(axis=1)))


def _unique_edges(triangles):
    pairs = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _cuff_ring(reference_points, positions, cuff_loop, cuff_center, frame, clearance_mm):
    """Bind the first material ring to the actual single open boundary."""
    major, minor, tangent = frame
    ring = positions[np.asarray(cuff_loop, dtype=np.int64)]
    coordinates = np.column_stack(((ring - cuff_center) @ major, (ring - cuff_center) @ minor))
    signed_area = np.sum(coordinates[:, 0] * np.roll(coordinates[:, 1], -1)
                         - coordinates[:, 1] * np.roll(coordinates[:, 0], -1))
    if abs(signed_area) < 1e-8:
        raise ValueError("cuff boundary has nearly zero area in the leg cross section")
    if signed_area < 0:
        ring = ring[::-1]
        coordinates = coordinates[::-1]
    anchor = int(np.argmax(coordinates[:, 0]))
    ring = np.roll(ring, -anchor, axis=0)
    fractions = _arc_fractions(ring)
    closed = np.vstack((ring, ring[0]))
    wanted = _arc_fractions(reference_points)[:-1]
    points = np.column_stack([np.interp(wanted, fractions, closed[:, axis]) for axis in range(3)])
    radial = points - cuff_center
    radial -= np.outer(radial @ tangent, tangent)
    return points + clearance_mm * unit(radial)


def _plane_section_ring(positions, triangles, center, frame, angles):
    """Intersect a skin cross-section plane and cast 2D rays to its near rim.

    This prevents an oblique 3D ray on a bent upper thigh from hitting a
    distant part of the same leg well above or below its intended material
    course.  Several disjoint section loops are allowed; the nearest radial
    crossing around the measured centerline is selected.
    """
    major, minor, tangent = frame
    level = (positions - center) @ tangent
    triangle_level = level[triangles]
    candidates = triangles[(triangle_level.min(axis=1) < 0.) & (triangle_level.max(axis=1) > 0.)]
    if len(candidates) < 4:
        raise ValueError("no closed skin section at this material course")
    distances = level[candidates]
    starts = np.array((0, 1, 2))
    ends = np.array((1, 2, 0))
    crossed = ((distances[:, starts] < 0.) & (distances[:, ends] >= 0.)) | (
        (distances[:, ends] < 0.) & (distances[:, starts] >= 0.))
    valid_faces = crossed.sum(axis=1) == 2
    if np.count_nonzero(valid_faces) < 4:
        raise ValueError("skin section has too few intersected edges")
    candidates = candidates[valid_faces]
    distances = distances[valid_faces]
    crossed = crossed[valid_faces]
    a = positions[candidates[:, starts]]
    b = positions[candidates[:, ends]]
    fraction = np.divide(-distances[:, starts], distances[:, ends] - distances[:, starts],
                         out=np.zeros_like(distances), where=crossed)
    hits = (a + fraction[..., None] * (b - a))[crossed].reshape(-1, 2, 3)
    a2 = np.column_stack(((hits[:, 0] - center) @ major, (hits[:, 0] - center) @ minor))
    b2 = np.column_stack(((hits[:, 1] - center) @ major, (hits[:, 1] - center) @ minor))
    edge = b2 - a2
    points = []
    for angle in angles:
        direction = np.array((math.cos(angle), math.sin(angle)))
        denominator = edge[:, 0] * direction[1] - edge[:, 1] * direction[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            along = -(a2[:, 0] * direction[1] - a2[:, 1] * direction[0]) / denominator
            radial = (a2[:, 0] + along * edge[:, 0]) * direction[0] + (
                a2[:, 1] + along * edge[:, 1]) * direction[1]
        valid = (np.abs(denominator) > 1e-10) & (along >= -1e-8) & (along <= 1. + 1e-8) & (radial > 1e-5)
        if not np.any(valid):
            raise ValueError("skin section is open in a radial direction")
        radius = float(radial[valid].min())
        points.append(center + radius * (major * direction[0] + minor * direction[1]))
    return np.asarray(points)


def _contact_correct(current, triangles, skin, clearance_mm, *, tolerance_mm=.02):
    """Lift vertices using vertex/edge-midpoint/face-centroid signed samples."""
    edges = _unique_edges(triangles)
    history = []
    for iteration in range(16):
        samples = _samples(current, triangles, edges)
        nearest, normals, signed, body_indices = skin.nearest(samples)
        minimum = float(signed.min())
        history.append(minimum)
        if minimum >= -tolerance_mm:
            break
        # A finite-sample clearance correction; this is not continuous
        # collision detection.  The positive target dampens re-penetration.
        needed = np.maximum(.35 * clearance_mm - signed, 0.)
        displacement = np.zeros_like(current)
        weight = np.zeros(len(current))
        sections = ((np.arange(len(current))[:, None], 0),
                    (edges, len(current)),
                    (triangles, len(current) + len(edges)))
        for ids, start in sections:
            delta = needed[start:start + len(ids)]
            directions = normals[start:start + len(ids)]
            active = delta > 0
            for corner in range(ids.shape[1]):
                np.add.at(displacement, ids[active, corner], directions[active] * delta[active, None] * 1.25)
                np.add.at(weight, ids[active, corner], 1.)
        active = weight > 0
        current[active] += displacement[active] / weight[active, None]
    # The samples and closest points must describe the final corrected mesh.
    samples = _samples(current, triangles, edges)
    nearest, normals, signed, body_indices = skin.nearest(samples)
    if float(signed.min()) < -tolerance_mm:
        worst = int(np.argmin(signed))
        raise ValueError(f"sampled skin contact did not converge; minimum {signed.min():.4f} mm "
                         f"at sample {worst} {samples[worst].round(3).tolist()}; "
                         f"history {[round(value, 3) for value in history]}")
    return edges, samples, nearest, normals, signed, body_indices, history


def _self_intersections(current, triangles, *, maximum_to_report=20):
    """Return disjoint-face BVH overlaps; neighboring triangle contact is normal."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromPolygons([Vector(row) for row in current], triangles.tolist(), all_triangles=True)
    overlaps = []
    for left, right in bvh.overlap(bvh):
        if left >= right:
            continue
        if np.intersect1d(triangles[left], triangles[right], assume_unique=True).size:
            continue
        overlaps.append((int(left), int(right)))
        if len(overlaps) >= maximum_to_report:
            break
    return overlaps


def _repair_toe_seam_orientation(current, triangles, ball, first_cap_face):
    """Untangle tiny terminal seam triangles without changing IDs or winding."""
    displacement_total = np.zeros_like(current)
    repaired_faces = 0
    for _ in range(40):
        cap = current[triangles[first_cap_face:]]
        area = np.cross(cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0])
        radial = unit(cap.mean(axis=1) - ball)
        signed = np.einsum("ij,ij->i", area, radial)
        bad = np.flatnonzero(signed <= 1e-8)
        if not len(bad):
            return repaired_faces, float(np.linalg.norm(displacement_total, axis=1).max())
        if len(bad) > 12:
            raise ValueError(f"toe cap has {len(bad)} inward faces, beyond local seam repair")
        displacement = np.zeros_like(current)
        count = np.zeros(len(current))
        for local_id in bad:
            face_id = first_cap_face + int(local_id)
            a, b, c = triangles[face_id]
            normal = radial[local_id]
            gradients = np.asarray((np.cross(current[b] - current[c], normal),
                                    np.cross(current[c] - current[a], normal),
                                    np.cross(current[a] - current[b], normal)))
            gain = float(np.sum(gradients * gradients))
            if gain < 1e-15:
                raise ValueError("toe seam orientation constraint is singular")
            desired = .08 * np.linalg.norm(area[local_id])
            delta = gradients * ((desired - signed[local_id]) / gain)
            maximum = float(np.linalg.norm(delta, axis=1).max())
            if maximum > .12:
                delta *= .12 / maximum
            vertices = np.asarray((a, b, c), dtype=np.int64)
            np.add.at(displacement, vertices, delta)
            np.add.at(count, vertices, 1.)
            repaired_faces += 1
        active = count > 0
        delta = displacement[active] / count[active, None]
        current[active] += delta
        displacement_total[active] += delta
    raise ValueError("toe seam orientation did not converge after local correction")


def fit_static_single_leg(reference, positions_mm, triangles, analysis, *, clearance_mm=.6):
    """Return ``(GarmentMeshPair, report, contact_samples)`` for one static leg.

    ``analysis`` is the result of ``analyze_single_leg_mesh``.  Its centerline
    runs from the sole open cuff toward the ankle.  Foot vertices are used to
    construct a toe envelope that bridges individual toes.
    """
    reference.validate()
    if not np.isfinite(clearance_mm) or clearance_mm <= 0:
        raise ValueError("clearance_mm must be a positive finite millimetre value")
    positions = np.asarray(positions_mm, dtype=float)
    skin_triangles = np.asarray(triangles, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("single-leg positions_mm must be finite [N,3] coordinates")
    if skin_triangles.ndim != 2 or skin_triangles.shape[1] != 3 or len(skin_triangles) < 4:
        raise ValueError("single-leg triangles must be [M,3]")
    if np.any(skin_triangles < 0) or np.any(skin_triangles >= len(positions)):
        raise ValueError("single-leg triangles contain an invalid vertex index")
    input_overlaps = _self_intersections(positions, skin_triangles)
    if input_overlaps:
        raise ValueError(
            "input single-leg skin self-intersects between disjoint triangles "
            f"{input_overlaps[0]}; repair or choose a clean pose before fitting")
    parameters = reference.metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("reference metadata must include procedural garment parameters")
    params = ProceduralGarmentParameters.from_dict(parameters)
    circumference = params.circumference_segments
    base_count = params.leg_segments + params.heel_segments + params.foot_segments + 1
    ring_count = len(reference.metadata.get("ring_regions", ()))
    if ring_count != base_count + params.toe_segments - 1:
        raise ValueError("reference ring topology does not match its saved parameters")
    seam = np.asarray(reference.metadata.get("toe_seam_material_point_ids", ()), dtype=np.int64)
    if len(seam) != circumference // 2 + 1 or len(reference.reference_positions_mm) != ring_count * circumference + len(seam):
        raise ValueError("reference toe seam topology does not match its saved parameters")

    centers, frames, ball = _frame_centers(analysis, params)
    skin = BodySurface(positions, skin_triangles)
    ankle = _required_vector(analysis, "ankle_center_mm")
    toe = _required_vector(analysis, "toe_point_mm")
    foot_axis = unit(_required_vector(analysis, "foot_axis"))
    toe_reach = float((toe - ankle) @ foot_axis)
    foot_ids = np.asarray(analysis["foot_vertex_indices"], dtype=np.int64)
    # A hull of ankle, shin and toes makes an artificial high instep wedge.
    # Only the forefoot is needed to bridge toe clefts and close the cap.
    forefoot_ids = foot_ids[((positions[foot_ids] - ankle) @ foot_axis) >= .35 * toe_reach]
    hull = _foot_hull(positions, forefoot_ids)
    dense_count = max(256, circumference * 4)
    angles = 2 * math.pi * np.arange(dense_count) / dense_count
    current = np.empty_like(reference.reference_positions_mm)
    base_angles = None
    missing_section_rows = []

    for row in range(base_count):
        if row == 0:
            current[:circumference] = _cuff_ring(
                reference.reference_positions_mm[:circumference], positions,
                analysis["cuff_loop"], _required_vector(analysis, "cuff_center_mm"),
                frames[0], clearance_mm)
            continue
        major, minor, _ = frames[row]
        try:
            dense_reference = _plane_section_ring(positions, skin_triangles, centers[row], frames[row], angles)
        except ValueError:
            missing_section_rows.append(row)
            continue
        dense_normals = np.asarray([
            major * math.cos(angle) + minor * math.sin(angle) for angle in angles])
        ids = slice(row * circumference, (row + 1) * circumference)
        points, outward = _redistribute_ring(reference.reference_positions_mm[ids],
                                              dense_reference, dense_reference, dense_normals)
        current[ids] = points + clearance_mm * outward
        if row == base_count - 1:
            base_angles = np.interp(_arc_fractions(reference.reference_positions_mm[ids])[:-1],
                                    _arc_fractions(dense_reference), np.r_[angles, 2 * math.pi])

    if missing_section_rows:
        missing_section_rows = sorted(set(missing_section_rows))
        groups = []
        for row in missing_section_rows:
            if not groups or row != groups[-1][-1] + 1:
                groups.append([row])
            else:
                groups[-1].append(row)
        for group in groups:
            first, last = group[0], group[-1]
            if first < 1 or last + 1 >= base_count or len(group) > 16:
                raise ValueError(f"too many consecutive missing skin sections at rings {first}..{last}")
            before = current[(first - 1) * circumference:first * circumference]
            after = current[(last + 1) * circumference:(last + 2) * circumference]
            for row in group:
                fraction = (row - first + 1) / (last - first + 2)
                current[row * circumference:(row + 1) * circumference] = (
                    (1 - fraction) * before + fraction * after)
    if base_angles is None:
        raise ValueError("last foot section is unavailable; cannot form a toe cap")

    major, minor, foot_tangent = frames[-1]
    toe_forward = unit(foot_tangent)
    major = unit(major - toe_forward * float(major @ toe_forward))
    minor = unit(np.cross(-toe_forward, major))
    tip, _, _, _, _ = hull.ray_bind(ball, toe_forward)
    toe_length = float((tip - ball) @ toe_forward)
    if toe_length < 5.:
        raise ValueError("foot hull does not extend beyond the ball center toward the toes")
    left, _, _, _, _ = hull.ray_bind(ball, major)
    right, _, _, _, _ = hull.ray_bind(ball, -major)
    radius = .9 * min(float(np.linalg.norm(left - ball)), float(np.linalg.norm(right - ball)))
    if radius < 2.:
        raise ValueError("foot hull is too narrow at the ball center")
    dense_seam_angles = np.linspace(0., math.pi, dense_count // 2 + 1)
    dense_seam = np.asarray([hull.ray_bind(ball, major * radius * math.cos(angle) + toe_forward * toe_length)[0]
                             for angle in dense_seam_angles])
    seam_fraction = _open_arc_fractions(dense_seam)
    unique = np.r_[True, np.diff(seam_fraction) > 1e-10]
    seam_angles = np.interp(_open_arc_fractions(reference.reference_positions_mm[seam]),
                            seam_fraction[unique], dense_seam_angles[unique])
    cap_angles = np.r_[seam_angles, 2 * math.pi - seam_angles[-2:0:-1]]
    if len(cap_angles) != circumference:
        raise ValueError("toe cap angle count does not match ring resolution")
    for row in range(base_count, ring_count):
        fraction = (row - base_count + 1) / params.toe_segments
        blend = min(1., 2 * fraction)
        blend = blend**2 * (3 - 2 * blend)
        ring_angles = base_angles * (1 - blend) + cap_angles * blend
        for k, angle in enumerate(ring_angles):
            direction = (major * radius * math.cos(angle)
                         + minor * radius * math.sin(angle) * math.cos(fraction * math.pi / 2)
                         + toe_forward * toe_length * fraction)
            point, _, _, _, _ = hull.ray_bind(ball, direction)
            vertex = row * circumference + k
            current[vertex] = point + clearance_mm * unit(point - ball)
    for vertex, angle in zip(seam, seam_angles):
        point, _, _, _, _ = hull.ray_bind(ball, major * radius * math.cos(angle) + toe_forward * toe_length)
        current[vertex] = point + clearance_mm * unit(point - ball)

    # The skin section at the ball and the enclosing toe hull can be separated
    # by a toe cleft or raised instep.  Spread that displacement over several
    # cap courses instead of introducing it all at their first shared edge.
    cap_transition = min(6, max(2, params.toe_segments // 2))
    last_skin_ring = current[(base_count - 1) * circumference:base_count * circumference].copy()
    for offset in range(1, cap_transition):
        fraction = offset / cap_transition
        blend = fraction * fraction * (3 - 2 * fraction)
        ids = slice((base_count - 1 + offset) * circumference,
                    (base_count + offset) * circumference)
        current[ids] = (1 - blend) * last_skin_ring + blend * current[ids]

    # Cross-section cuts through close skin folds can leave sub-pixel kinks
    # along one material course.  Smooth only their circumferential sampling;
    # the finite skin-contact pass below restores local clearance.
    for row in range(1, min(25, params.leg_segments)):
        ids = slice(row * circumference, (row + 1) * circumference)
        ring = current[ids].copy()
        current[ids] = .75 * ring + .125 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))

    shell_faces = reference.triangles
    edges, samples, nearest, normals, signed, body_indices, history = _contact_correct(
        current, shell_faces, skin, clearance_mm)
    cap_face_start = (base_count - 1) * circumference * 2
    toe_repairs, toe_repair_maximum = _repair_toe_seam_orientation(
        current, shell_faces, ball, cap_face_start)
    if toe_repairs:
        edges, samples, nearest, normals, signed, body_indices, second_history = _contact_correct(
            current, shell_faces, skin, clearance_mm)
        history += second_history
    face_points = current[shell_faces]
    area = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    area_size = np.linalg.norm(area, axis=1)
    if np.any(area_size < 1e-10):
        raise ValueError(f"fitted stocking contains {int(np.count_nonzero(area_size < 1e-10))} degenerate faces")
    face_normals = area / area_size[:, None]
    _, skin_normals, _, _ = skin.nearest(face_points.mean(axis=1))
    foot_face_start = cap_face_start
    expected = skin_normals.copy()
    cap_radial = unit(face_points[foot_face_start:].mean(axis=1) - ball)
    expected[foot_face_start:] = cap_radial
    outward_dot = np.einsum("ij,ij->i", face_normals, expected)
    inward = np.flatnonzero(outward_dot <= 1e-8)
    if len(inward):
        raise ValueError(f"fitted stocking has {len(inward)} inward faces; first face {int(inward[0])}")
    overlaps = _self_intersections(current, shell_faces)
    if overlaps:
        raise ValueError(f"fitted stocking self-intersects between disjoint faces, first pair {overlaps[0]}")

    metadata = copy.deepcopy(reference.metadata)
    metadata.update(fit_status="geometric_fit", fit_method="static single-leg material-ring skin-section binding with convex toe envelope",
                    source_body_object=analysis.get("source_object", "imported_single_leg"), source_frame=None,
                    target_pose="static", requested_clearance_mm=float(clearance_mm),
                    dressing_equilibrium_solved=False, material_calibrated=False,
                    yarn_equilibrium_solved=False, stress_free_assumed=False,
                    toe_method="distal forefoot convex envelope with distributed seam",
                    toe_envelope_forward_start_fraction=.35,
                    circumferential_allocation="relative physical arc length of each X0 ring",
                    interpolated_skin_section_ring_count=len(missing_section_rows))
    pair = GarmentMeshPair(reference.reference_positions_mm, current, shell_faces,
                           reference.material_uv, reference.material_point_ids, metadata,
                           face_material_uv=reference.face_material_uv,
                           current_face_directors=expected)
    contact_samples = dict(sample_positions_mm=samples, closest_body_positions_mm=nearest,
                           closest_body_normals=normals, signed_distance_mm=signed,
                           closest_body_triangle_index=body_indices,
                           sample_kind=np.r_[np.zeros(len(current), dtype=int),
                                             np.ones(len(edges), dtype=int),
                                             np.full(len(shell_faces), 2, dtype=int)],
                           sample_source_index=np.r_[np.arange(len(current)),
                                                     np.arange(len(edges)),
                                                     np.arange(len(shell_faces))],
                           face_outward_dot=outward_dot)
    toe_join_step = np.linalg.norm(
        current[base_count * circumference:(base_count + 1) * circumference]
        - current[(base_count - 1) * circumference:base_count * circumference], axis=1)
    report = dict(schema="stocking.single_leg_fit_report", schema_version=1,
                  fit_status="geometric_fit", length_unit="mm", requested_clearance_mm=float(clearance_mm),
                  tolerance_mm=.02, vertex_count=len(current), triangle_count=len(shell_faces),
                  sample_count=len(samples), sample_min_signed_distance_mm=float(signed.min()),
                  sample_penetration_count=int(np.count_nonzero(signed < -.02)),
                  sampled_clearance_percentiles_mm=np.percentile(signed, [0, 5, 50, 95, 100]).tolist(),
                  samples_only=True, correction_minimum_history_mm=history,
                  triangles_outward_checked=True, minimum_face_outward_dot=float(outward_dot.min()),
                  inward_triangle_count=0, self_intersection_check_status="disjoint-face BVH overlap checked",
                  self_intersection_count=0, input_self_intersection_check_status="disjoint-face BVH overlap checked",
                  input_self_intersection_count=0,
                  foot_hull_ray_miss_count=hull.miss_count,
                  foot_hull_reverse_ray_count=hull.reverse_ray_count,
                  forefoot_hull_source_vertex_count=int(len(forefoot_ids)),
                  toe_envelope_forward_start_fraction=.35,
                  toe_first_course_max_step_mm=float(toe_join_step.max()),
                  interpolated_skin_section_ring_indices=missing_section_rows,
                  toe_orientation_constraint_updates=toe_repairs,
                  maximum_toe_orientation_correction_mm=toe_repair_maximum,
                  synthetic_uncalibrated=True, dressing_equilibrium_solved=False,
                  scope="one static pose; finite vertex/edge/face skin clearance and discrete shell BVH overlap checks")
    return pair, report, contact_samples
