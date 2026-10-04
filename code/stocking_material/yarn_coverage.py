"""Orthographic pixel coverage of an actual polyline yarn capsule union.

This local prototype assumes an opaque skin plane behind *all* supplied
capsules and a camera plane in front of all of them.  Under those checked
conditions, ray hits equal membership in the projected 2-D capsule union.
Each subpixel contributes one Boolean hit, regardless of yarn overlap or
whether a P/N/T tangent is defined at the intersection.  No optical ID or
surface feature-valid channel participates in this calculation.
"""

import numpy as np


def _camera_frame(center_mm, right, up, toward_camera):
    center = np.asarray(center_mm, dtype=np.float64)
    basis = np.asarray((right, up, toward_camera), dtype=np.float64)
    if center.shape != (3,) or basis.shape != (3, 3):
        raise ValueError("camera center and each basis axis must have three coordinates")
    if not np.isfinite(center).all() or not np.isfinite(basis).all():
        raise ValueError("camera center and basis must be finite")
    if np.max(np.abs(basis @ basis.T - np.eye(3))) > 1e-10:
        raise ValueError("camera basis must be orthonormal")
    if np.linalg.det(basis) < 1 - 1e-10:
        raise ValueError("right, up, toward_camera must form a right-handed basis")
    return center, basis


def orthographic_project(points_mm, center_mm, right, up, toward_camera):
    """Return camera right/up/depth coordinates, in millimetres.

    Positive depth is toward the camera; all rays point in the negative
    depth direction.  Input may have any leading dimensions ending in 3.
    """
    center, basis = _camera_frame(center_mm, right, up, toward_camera)
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite points[...,3] are required")
    return (points - center) @ basis.T


def subpixel_camera_coordinates(image_size, fov_mm, supersample):
    """Return subpixel center x and y arrays, with image rows going downward.

    ``image_size`` is (height, width), ``fov_mm`` is (width, height), and
    ``supersample`` is the grid count along each pixel axis.  A regular
    S-by-S stratified center grid is used; different S grids are not nested.
    """
    dims = tuple(image_size)
    if len(dims) != 2 or any(not isinstance(v, (int, np.integer)) or
                             isinstance(v, (bool, np.bool_)) or v < 1 for v in dims):
        raise ValueError("image_size must be two positive integers (height,width)")
    size = np.asarray(fov_mm, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("fov_mm must be positive finite (width,height)")
    if (not isinstance(supersample, (int, np.integer)) or
            isinstance(supersample, (bool, np.bool_)) or not 1 <= supersample <= 65535):
        raise ValueError("supersample must be an integer in [1,65535]")
    height, width = map(int, dims)
    sx = (np.arange(width * supersample, dtype=np.float64) + .5) * (
        size[0] / (width * supersample)) - size[0] / 2
    sy = size[1] / 2 - (np.arange(height * supersample, dtype=np.float64) + .5) * (
        size[1] / (height * supersample))
    return sx, sy


def projected_capsule_hits(points_xy_mm, segment_a_xy_mm, segment_b_xy_mm,
                           radius_mm, *, max_work_pairs=65536):
    """Return raw projected union hits for arbitrary 2-D query points.

    This query helper bounds point-by-segment temporary arrays and preserves
    point projections.  It uses the same geometric radius and union rule as
    rasterization, and has no P/N/T validity requirement.
    """
    points = np.asarray(points_xy_mm, dtype=np.float64)
    first = np.asarray(segment_a_xy_mm, dtype=np.float64)
    second = np.asarray(segment_b_xy_mm, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or first.ndim != 2 or
            first.shape[1] != 2 or second.shape != first.shape or
            not np.isfinite(points).all() or not np.isfinite(first).all() or
            not np.isfinite(second).all()):
        raise ValueError("finite points[N,2] and matching segment arrays[E,2] are required")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be finite and positive")
    if (not isinstance(max_work_pairs, (int, np.integer)) or
            isinstance(max_work_pairs, (bool, np.bool_)) or max_work_pairs < 1):
        raise ValueError("max_work_pairs must be a positive integer")
    hits = np.zeros(len(points), dtype=bool)
    if not len(first):
        return hits
    segment_block = min(len(first), int(max_work_pairs))
    point_block = max(1, int(max_work_pairs) // segment_block)
    radius2 = float(radius_mm) ** 2
    for pstart in range(0, len(points), point_block):
        pstop = min(pstart + point_block, len(points))
        local_hits = np.zeros(pstop - pstart, dtype=bool)
        for sstart in range(0, len(first), segment_block):
            sstop = min(sstart + segment_block, len(first))
            a = first[None, sstart:sstop, :]
            edge = second[None, sstart:sstop, :] - a
            offset = points[pstart:pstop, None, :] - a
            length2 = np.sum(edge * edge, axis=2)
            fraction = np.clip(np.sum(offset * edge, axis=2) /
                               np.where(length2 == 0., 1., length2), 0., 1.)
            distance2 = np.sum((offset - fraction[..., None] * edge) ** 2, axis=2)
            local_hits |= np.any(distance2 <= radius2, axis=1)
            if local_hits.all():
                break
        hits[pstart:pstop] = local_hits
    return hits


def rasterize_capsule_union(segment_a_mm, segment_b_mm, radius_mm, *,
                           center_mm, right, up, toward_camera,
                           image_size=(128, 128), fov_mm=(1., 1.),
                           supersample=8, camera_depth_mm=None,
                           skin_depth_mm=None, return_subpixel_hits=False,
                           max_raster_samples=64 * 1024 * 1024,
                           max_work_samples=65536):
    """Compute float32 pixel C and integer hit counts with bounded memory.

    Every 3-D centerline segment has the same positive radius.  Its exact
    orthographic silhouette is the 2-D capsule around the projected segment,
    including a circle when a nonzero 3-D segment projects to a point.
    Segment AABBs select local raster rectangles; at most max_work_samples
    temporary distance values are evaluated per block.  The retained union
    mask uses one byte per subpixel and is capped by max_raster_samples.

    Auto camera/skin planes are separated from the whole capsule union.
    Explicit depths must provide the same ordering.  This permits projected
    membership to replace first-hit casts only for this checked setup.
    FOV clipping does not remove any in-frame capsule silhouette: segment
    AABBs are expanded by radius before clipping to the sample raster.
    Completeness of the supplied centerline selection is the caller's duty.
    """
    center, basis = _camera_frame(center_mm, right, up, toward_camera)
    first = np.asarray(segment_a_mm, dtype=np.float64)
    second = np.asarray(segment_b_mm, dtype=np.float64)
    if (first.ndim != 2 or first.shape[1] != 3 or second.shape != first.shape or
            not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("finite segment_a_mm[E,3] and matching segment_b_mm are required")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be finite and positive")
    for name, value in (("max_raster_samples", max_raster_samples),
                        ("max_work_samples", max_work_samples)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(name + " must be a positive integer")
    # Validate dimensions without allocating the potentially large raster.
    # Coordinate vectors themselves are much smaller than the union mask.
    dims = tuple(image_size)
    if len(dims) != 2 or any(not isinstance(v, (int, np.integer)) or
                            isinstance(v, (bool, np.bool_)) or v < 1 for v in dims):
        raise ValueError("image_size must be two positive integers (height,width)")
    if (not isinstance(supersample, (int, np.integer)) or isinstance(supersample, (bool, np.bool_)) or
            not 1 <= supersample <= 65535):
        raise ValueError("supersample must be an integer in [1,65535]")
    height, width = map(int, dims)
    sample_height, sample_width = height * int(supersample), width * int(supersample)
    if sample_height * sample_width > max_raster_samples:
        raise ValueError("subpixel raster exceeds max_raster_samples memory bound")
    x_centers, y_centers = subpixel_camera_coordinates(dims, fov_mm, int(supersample))
    fov = np.asarray(fov_mm, dtype=np.float64)
    projected_a = (first - center) @ basis.T
    projected_b = (second - center) @ basis.T
    if len(first):
        near = float(max(projected_a[:, 2].max(), projected_b[:, 2].max()) + radius_mm)
        far = float(min(projected_a[:, 2].min(), projected_b[:, 2].min()) - radius_mm)
    else:
        near, far = float(radius_mm), -float(radius_mm)
    margin = max(.1, 4 * float(radius_mm))
    camera_depth = near + margin if camera_depth_mm is None else float(camera_depth_mm)
    skin_depth = far - margin if skin_depth_mm is None else float(skin_depth_mm)
    if not np.isfinite(camera_depth) or not np.isfinite(skin_depth):
        raise ValueError("camera and skin plane depths must be finite")
    if camera_depth <= near:
        raise ValueError("camera plane must be strictly in front of every capsule")
    if skin_depth >= far:
        raise ValueError("skin plane must be strictly behind every capsule")

    hits = np.zeros((sample_height, sample_width), dtype=bool)
    dx, dy = fov[0] / sample_width, fov[1] / sample_height
    radius2 = float(radius_mm) ** 2
    tested_samples, visited_segments, outside_segments = 0, 0, 0
    clipped_segments, point_projections = 0, 0
    # Flip projection Y for monotonically increasing raster-index bounds.
    xy_a, xy_b = projected_a[:, :2].copy(), projected_b[:, :2].copy()
    xy_a[:, 1] *= -1
    xy_b[:, 1] *= -1
    image_y = -y_centers
    # Roundoff margin is solely for conservative AABB selection; distance
    # membership uses the specified radius without geometrical inflation.
    index_tolerance = 1e-9
    for a, b in zip(xy_a, xy_b):
        lower, upper = np.minimum(a, b) - radius_mm, np.maximum(a, b) + radius_mm
        i0 = int(np.ceil((lower[0] + fov[0] / 2) / dx - .5 - index_tolerance))
        i1 = int(np.floor((upper[0] + fov[0] / 2) / dx - .5 + index_tolerance)) + 1
        j0 = int(np.ceil((lower[1] + fov[1] / 2) / dy - .5 - index_tolerance))
        j1 = int(np.floor((upper[1] + fov[1] / 2) / dy - .5 + index_tolerance)) + 1
        if i1 <= 0 or i0 >= sample_width or j1 <= 0 or j0 >= sample_height:
            outside_segments += 1
            continue
        if i0 < 0 or i1 > sample_width or j0 < 0 or j1 > sample_height:
            clipped_segments += 1
        i0, i1 = max(i0, 0), min(i1, sample_width)
        j0, j1 = max(j0, 0), min(j1, sample_height)
        if i0 >= i1 or j0 >= j1:
            outside_segments += 1
            continue
        visited_segments += 1
        edge = b - a
        length2 = float(edge @ edge)
        is_point = length2 == 0.
        point_projections += int(is_point)
        columns_per_block = min(i1 - i0, int(max_work_samples))
        rows_per_block = max(1, int(max_work_samples) // columns_per_block)
        for col in range(i0, i1, columns_per_block):
            stop_col = min(col + columns_per_block, i1)
            relative_x = x_centers[col:stop_col][None, :] - a[0]
            for row in range(j0, j1, rows_per_block):
                stop_row = min(row + rows_per_block, j1)
                relative_y = image_y[row:stop_row, None] - a[1]
                if is_point:
                    distance2 = relative_x ** 2 + relative_y ** 2
                else:
                    fraction = np.clip((relative_x * edge[0] + relative_y * edge[1]) / length2, 0., 1.)
                    distance2 = (relative_x - fraction * edge[0]) ** 2 + (relative_y - fraction * edge[1]) ** 2
                selected = distance2 <= radius2
                hits[row:stop_row, col:stop_col] |= selected
                tested_samples += selected.size
    counts = hits.reshape(height, supersample, width, supersample).sum(axis=(1, 3), dtype=np.uint32)
    samples_per_pixel = int(supersample) ** 2
    coverage = counts.astype(np.float32) / np.float32(samples_per_pixel)
    metadata = {
        "method": "orthographic_projected_capsule_boolean_union",
        "coverage_semantics": "fraction of pixel subrays hitting opaque yarn before the opaque rear skin plane",
        "hit_source": "projected geometric capsule membership; independent of PNT valid",
        "union_overlap_rule": "one Boolean hit per subpixel, no repeated counting",
        "projection_exactness": "3D capsule orthographic silhouette equals its 2D projected segment capsule",
        "subpixel_sampling": "regular stratified grid centers, not nested across supersample levels",
        "image_size_height_width": [height, width],
        "fov_width_height_mm": fov.tolist(),
        "supersample_per_axis": int(supersample),
        "samples_per_pixel": samples_per_pixel,
        "image_row_direction": "negative_camera_up",
        "camera_center_mm": center.tolist(),
        "camera_right": basis[0].tolist(),
        "camera_up": basis[1].tolist(),
        "camera_toward": basis[2].tolist(),
        "ray_direction": (-basis[2]).tolist(),
        "camera_plane_depth_mm": camera_depth,
        "skin_plane_depth_mm": skin_depth,
        "capsule_near_bound_depth_mm": near,
        "capsule_far_bound_depth_mm": far,
        "camera_clearance_mm": camera_depth - near,
        "skin_clearance_mm": far - skin_depth,
        "all_capsules_before_skin": True,
        "all_ray_origins_outside_capsules": True,
        "radius_mm": float(radius_mm),
        "input_segment_count": int(len(first)),
        "visited_segment_count": visited_segments,
        "outside_raster_segment_count": outside_segments,
        "clipped_expanded_aabb_segment_count": clipped_segments,
        "point_projection_segment_count": point_projections,
        "expanded_aabb_radius_mm": float(radius_mm),
        "subpixel_hit_count": int(counts.sum(dtype=np.uint64)),
        "mean_coverage": float(counts.sum(dtype=np.uint64) / (height * width * samples_per_pixel)),
        "tested_segment_sample_pairs": int(tested_samples),
        "retained_subpixel_mask_bytes": int(hits.nbytes),
        "max_work_samples": int(max_work_samples),
        "selection_completeness": "caller must supply all capsules whose silhouettes intersect the FOV",
        "skin_model": "flat opaque plane behind all supplied capsules; not the full leg surface",
    }
    result = {"coverage": coverage, "hit_count": counts, "samples_per_pixel": samples_per_pixel,
              "metadata": metadata}
    if return_subpixel_hits:
        result["subpixel_hits"] = hits
    return result
