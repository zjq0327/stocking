"""Checks for the authored geometry and its feature data (not a physics solver)."""
from __future__ import annotations

import numpy as np


def validate_clearance(p):
    """Reject sampled global tube overlaps, separately from local curvature.

    Compare one entire centerline period with neighboring rows and periods.
    Nearby points on the same continuous strand are excluded by arc distance;
    their local regularity is checked by geometry._tube_shape_check. This is
    a conservative *sampled diagnostic*, not a continuous collision solver.
    """
    from .geometry import centerline, centerline_derivatives
    import math
    n = max(768, 4 * p.samples_per_loop)
    dt = 2 * math.pi / n
    pad = math.ceil((2 * p.a + 2 * p.R) / (2 * math.pi)) + 1
    tb = np.arange(-pad * n, (pad + 1) * n + 1) * dt
    ta = np.arange(n) * dt
    a = centerline(ta, p)
    velocity, _ = centerline_derivatives(tb, p)
    speed = np.linalg.norm(velocity, axis=1)
    arc = np.r_[0., np.cumsum((speed[:-1] + speed[1:]) * .5 * dt)]
    arc_a = arc[pad * n:pad * n + n]
    # Analytic speed upper bound converts finite sampling into a useful margin.
    error_bound = p.scale_mm * math.sqrt((1 + 2 * p.a) ** 2 + p.h ** 2 + (2 * p.d) ** 2) * dt
    radius = p.R * p.scale_mm
    row_reach = math.ceil((2 * p.h + 2 * p.R) / p.rowOffset)
    minimum = float("inf")
    for row in range(-row_reach, row_reach + 1):
        b = centerline(tb, p, row=row)
        for start in range(0, len(a), 48):
            part = a[start:start + 48]
            distances = np.sum((part[:, None, :] - b[None, :, :]) ** 2, axis=2)
            if row == 0:
                local = np.abs(arc_a[start:start + len(part), None] - arc[None, :]) <= math.pi * radius + error_bound
                distances[local] = np.inf
            minimum = min(minimum, float(np.sqrt(distances.min())))
    gap = minimum - 2 * radius
    if gap < -error_bound:
        raise ValueError(f"Yarn tubes overlap: sampled nonlocal gap {gap:.6g} mm. Reduce R or adjust rowOffset/shape.")
    return {"method": "periodic centerline sampling with local arc exclusion",
            "samples_per_period": n, "min_nonlocal_center_distance_mm": minimum,
            "sampled_surface_gap_mm": gap, "sampling_margin_mm": error_bound,
            "status": "separated_at_sampling_margin" if gap > error_bound else "near_contact_requires_inspection",
            "continuous_collision_solver": False}


def validate_mesh(geometry):
    vertices, triangles = geometry.vertices, geometry.triangles
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError("Mesh is empty or contains non-finite positions")
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    areas2 = np.linalg.norm(cross, axis=1)
    degenerate = int(np.count_nonzero(areas2 < 1e-18))
    outward = np.einsum("ij,ij->i", cross, geometry.normals[triangles].mean(axis=1))
    inward = int(np.count_nonzero(outward <= 0))
    if degenerate or inward:
        raise ValueError(f"Invalid mesh: {degenerate} degenerate, {inward} reversed triangles")
    return {"vertices": len(vertices), "triangles": len(triangles),
            "degenerate_triangles": degenerate, "inward_triangles": inward}


def validate_features(features, p):
    lx, ly = p.period_mm
    x = (np.arange(p.width) + 0.5) * lx / p.width
    y = (np.arange(p.height) + 0.5) * ly / p.height
    xx, yy = np.meshgrid(x, y)
    report = {}
    tolerance = max(lx, ly) * 2e-6
    for side, data in features.items():
        valid = data["valid"]
        if valid.shape != (p.height, p.width) or not np.array_equal(data["id"] != 0, valid):
            raise ValueError("ID and valid mask do not agree")
        if not valid.any():
            raise ValueError("The selected tile has no yarn hits")
        for key in ("P", "N", "T"):
            if not np.isfinite(data[key]).all() or np.any(data[key][~valid] != 0):
                raise ValueError(f"{side}/{key}: invalid values or nonzero empty pixels")
        normal, tangent = data["N"][valid], data["T"][valid]
        norm_error = float(np.abs(np.linalg.norm(normal, axis=1) - 1).max())
        tangent_error = float(np.abs(np.linalg.norm(tangent, axis=1) - 1).max())
        orthogonality = float(np.abs(np.einsum("ij,ij->i", normal, tangent)).max())
        xy_error = float(max(np.abs(data["P"][..., 0][valid] - xx[valid]).max(),
                             np.abs(data["P"][..., 1][valid] - yy[valid]).max()))
        if max(norm_error, tangent_error, orthogonality) > 2e-5 or xy_error > tolerance:
            raise ValueError(f"{side}: inconsistent surface attributes")
        report[side] = {"hit_fraction": float(valid.mean()), "hit_pixels": int(valid.sum()),
                        "normal_length_error": norm_error, "tangent_length_error": tangent_error,
                        "normal_tangent_dot_max": orthogonality, "position_xy_error_mm": xy_error}
    front, back = features["front"], features["back"]
    mask_mismatch = int(np.count_nonzero(front["valid"] != back["valid"]))
    if mask_mismatch:
        raise ValueError(f"Front/back occupancy differs at {mask_mismatch} pixels")
    valid = front["valid"]
    thickness = front["P"][..., 2][valid] - back["P"][..., 2][valid]
    if np.any(thickness < -tolerance):
        raise ValueError("Front hit lies below back hit")
    report["occupancy_mismatch"] = mask_mismatch
    report["min_front_back_depth_mm"] = float(thickness.min())
    return report


def validate_periodicity(geometry, p):
    from .bake import make_bvh, sample_points
    bvh = make_bvh(geometry)
    lx, ly = p.period_mm
    f = (np.arange(47) + 0.371) / 47
    probes = [(np.column_stack((np.zeros_like(f), f * ly)), np.array([lx, 0., 0.])),
              (np.column_stack((f * lx, np.zeros_like(f))), np.array([0., ly, 0.]))]
    report = {"mask_mismatch": 0, "position_error_mm": 0., "normal_error": 0., "tangent_error": 0.}
    for xy, offset in probes:
        for side in ("front", "back"):
            a = sample_points(geometry, bvh, xy, side)
            b = sample_points(geometry, bvh, xy + offset[:2], side)
            report["mask_mismatch"] += int(np.count_nonzero(a["valid"] != b["valid"]))
            valid = a["valid"] & b["valid"]
            if not valid.any():
                continue
            report["position_error_mm"] = max(report["position_error_mm"],
                float(np.linalg.norm(a["P"][valid] + offset - b["P"][valid], axis=1).max()))
            for key, label in (("N", "normal_error"), ("T", "tangent_error")):
                report[label] = max(report[label], float(np.linalg.norm(a[key][valid] - b[key][valid], axis=1).max()))
    if report["mask_mismatch"] or report["position_error_mm"] > max(lx, ly) * 2e-5 or max(report["normal_error"], report["tangent_error"]) > .003:
        raise ValueError(f"Periodic seam check failed: {report}")
    return report
