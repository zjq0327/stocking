"""Dense, independent material-triangle features from actual yarn tubes.

No periodic state atlas, optical ID channel, equilibrium, or second macro F.
The two configurations share material sample addresses.  Interior grids are
chosen in millimetres; shared edges have common physical-edge subdivisions.
Only a sample chunk and its supplied local tube geometry need be in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

from .garment_data import triangle_edges

SCHEMA = "stocking.triangle_features"
VERSION = 1
CONFIGURATIONS = ("reference", "current")
SIDES = ("front", "back")
FIELDS = ("P_offset_mm", "N_local", "T_local")
EDGES = ((0, 1), (1, 2), (2, 0))
MAX_SAMPLE_BYTES = 200  # four PNTvalid groups, addresses, optional ray diagnostic masks


class _HitTrackingBVH:
    """Keep ray-hit triangle addresses solely for optional tangent diagnostics."""
    def __init__(self, source, count, tangent_valid=None):
        self.source = source
        self.indices = np.full(count, -1, dtype=np.int64)
        self.next_index = 0
        self.tangent_valid = tangent_valid

    def ray_cast(self, *args):
        result = self.source.ray_cast(*args)
        if result[2] is not None:
            self.indices[self.next_index] = result[2]
        self.next_index += 1
        if (result[2] is not None and self.tangent_valid is not None
                and not self.tangent_valid[result[2]]):
            # A pole fan does not supply a defined yarn tangent.  Preserve
            # the first-hit diagnostic, but do not let an arbitrary/zero T
            # enter surface_bake's interpolation or seek a deeper surface.
            return (None, None, None, None)
        return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _unit(values, name):
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-14):
        raise ValueError(name + " has undefined directions; repair or exclude this domain explicitly")
    return values / lengths


def pair_fingerprint(pair):
    digest = hashlib.sha256()
    for value in (pair.reference_positions_mm, pair.current_positions_mm,
                  pair.triangles, pair.material_point_ids, pair.triangle_material_uv):
        value = np.ascontiguousarray(value)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class TriangleSampling:
    arrays: dict
    metadata: dict

    @property
    def sample_offsets(self):
        return self.arrays["face_sample_offsets"]

    def estimate(self, face_indices=None):
        selected = validate_faces(self, face_indices)
        counts = np.diff(self.sample_offsets)[selected]
        samples = sum(int(x) for x in counts)
        # Two configurations x two sides x (9 float32 + one bool), plus
        # float64 barycentrics and internal int32 face/int64 sample addresses.
        raw_bytes = samples * (4 * (9 * 4 + 1) + 3 * 8 + 4 + 8)
        return {"requested_faces": len(selected), "requested_samples": samples,
                "raw_sample_bytes_estimate": raw_bytes,
                "compressed_size_known": False, "local_tube_memory_not_included": True,
                "spacing_mm": self.metadata["spacing_mm"],
                "maximum_face_samples": max((int(x) for x in counts), default=0)}


def _sampling_at_spacing(template, spacing):
    """Reuse validated geometry when a budget searches many grid densities."""
    spacing = _positive(spacing, "spacing_mm")
    longest = template.arrays["edge_longest_mm"]
    subdivisions = np.maximum(1, np.ceil(longest.max(axis=1) / spacing)).astype(np.int64)
    divisions = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
    counts = np.maximum(subdivisions - 1, 0) * np.maximum(subdivisions - 2, 0) // 2 + divisions.sum(axis=1)
    arrays = {**template.arrays, "face_subdivisions": subdivisions, "face_edge_divisions": divisions,
              "face_sample_offsets": np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]}
    metadata = {**template.metadata, "spacing_mm": spacing, "density_source": "global_export_byte_budget"}
    return TriangleSampling(arrays, metadata)


def _storage_chunk_count(counts, max_samples, max_faces):
    chunks, samples, faces = 0, 0, 0
    for count in counts:
        if samples and faces == max_faces:
            chunks += 1
            samples, faces = 0, 0
        faces += 1
        remaining = int(count)
        while remaining:
            take = min(remaining, max_samples - samples)
            remaining -= take
            samples += take
            if samples == max_samples:
                chunks += 1
                samples, faces = 0, 1 if remaining else 0
    return chunks + bool(samples)


def estimate_triangle_export(sampling, face_indices=None, *, max_samples_per_chunk=65536,
                             max_faces_per_chunk=256, compact_coverage=True):
    """Conservative *uncompressed* feature/domain/ZIP/metadata estimate.

    Diagnostic masks are budgeted even when a provider does not emit them.
    Compact metadata is required for a fixed full-surface budget; detailed
    per-request provider JSON has no bounded size. Tube meshes are not exported.
    """
    if not compact_coverage:
        raise ValueError("bounded export estimate requires compact_coverage=True")
    for value in (max_samples_per_chunk, max_faces_per_chunk):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("chunk sizes must be positive integers")
    selected = validate_faces(sampling, face_indices)
    counts = np.diff(sampling.sample_offsets)[selected]
    samples = sum(int(n) for n in counts)
    chunks = int(_storage_chunk_count(counts, int(max_samples_per_chunk), int(max_faces_per_chunk)))
    domain_raw = sum(int(array.nbytes) for array in sampling.arrays.values()) + selected.nbytes
    domain_raw += 4 * len(json.dumps(sampling.metadata, ensure_ascii=False))
    feature_raw = MAX_SAMPLE_BYTES * samples
    # .npy header + ZIP member/deflate upper bound, deliberately generous.
    archive_overhead = math.ceil((domain_raw + feature_raw) * .002) + (len(sampling.arrays) + 2 + 35 * chunks) * 1024
    # Bounded per-chunk summaries, face lists, identities and final audit fields.
    metadata_reserve = 2_000_000 + chunks * 65536 + len(selected) * 32
    total = domain_raw + feature_raw + archive_overhead + metadata_reserve
    return {**sampling.estimate(selected), "raw_sample_bytes_including_diagnostics": feature_raw,
            "bytes_per_sample_upper_bound": MAX_SAMPLE_BYTES, "domain_raw_bytes": domain_raw,
            "storage_chunks": chunks, "archive_overhead_reserve_bytes": archive_overhead,
            "metadata_reserve_bytes": metadata_reserve, "conservative_export_bytes": total,
            "minimum_samples_per_face": 3, "compact_coverage_required": True,
            "float32_feature_precision_retained": True,
            "export_includes": ["two configurations x front/back PNTvalid", "optional ray diagnostics", "material sampling domain", "chunk and identity metadata"],
            "export_excludes": ["tube meshes", "centerline streams", "renders", "preexisting candidate assets"]}


def plan_triangle_export_budget(pairs, *, budget_bytes=1_000_000_000, reserve_bytes=100_000_000,
                                yarn_radius_mm=.025, preferred_spacing_mm=None,
                                max_samples_per_chunk=65536, max_faces_per_chunk=256):
    """Choose one global spacing and per-sock byte ceilings without baking.

    At least three independently stored corners remain for every triangle.
    The reserve covers package snapshots/parameters and unplanned metadata;
    it must not be spent on tube geometry without a new explicit budget.
    """
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("pairs must be a nonempty name -> GarmentMeshPair dict")
    if type(budget_bytes) is not int or type(reserve_bytes) is not int or not 0 <= reserve_bytes < budget_bytes:
        raise ValueError("budget and reserve must be integers with 0 <= reserve < budget")
    radius = _positive(yarn_radius_mm, "yarn_radius_mm")
    preferred = radius / 2 if preferred_spacing_mm is None else _positive(preferred_spacing_mm, "preferred_spacing_mm")
    templates = {name: prepare_triangle_sampling(pair, yarn_radius_mm=radius, spacing_mm=preferred)
                 for name, pair in pairs.items()}
    maximum = max(float(s.arrays["edge_longest_mm"].max()) for s in templates.values())
    def at(spacing):
        samples = {name: _sampling_at_spacing(template, spacing) for name, template in templates.items()}
        estimates = {name: estimate_triangle_export(s, max_samples_per_chunk=max_samples_per_chunk,
                    max_faces_per_chunk=max_faces_per_chunk) for name, s in samples.items()}
        return samples, estimates, sum(e["conservative_export_bytes"] for e in estimates.values())
    _, minimum, floor_bytes = at(max(maximum * (1 + 1e-12), preferred))
    available = budget_bytes - reserve_bytes
    if floor_bytes > available:
        raise ValueError(f"budget cannot store all faces even at three corners per face: minimum {floor_bytes + reserve_bytes} bytes")
    low, high = preferred, max(maximum * (1 + 1e-12), preferred)
    _, _, preferred_bytes = at(preferred)
    if preferred_bytes > available:
        for _ in range(48):
            mid = (low + high) / 2
            if at(mid)[2] <= available:
                high = mid
            else:
                low = mid
        spacing = math.ceil(high * 1_000_000) / 1_000_000
    else:
        spacing = preferred
    samplings, estimates, planned = at(spacing)
    remaining = available - planned
    names = list(pairs)
    for index, name in enumerate(names):
        e = estimates[name]
        e["export_byte_limit"] = e["conservative_export_bytes"] + remaining // len(names) + (index < remaining % len(names))
        e["pair_fingerprint"] = samplings[name].metadata["pair_fingerprint"]
        e["minimum_grid_estimate"] = minimum[name]
    return {"schema": "stocking.triangle_feature_budget", "schema_version": 1,
            "budget_bytes": budget_bytes, "decimal_gigabytes": budget_bytes / 1_000_000_000,
            "package_reserve_bytes": reserve_bytes, "spacing_mm": spacing,
            "preferred_spacing_mm": preferred, "yarn_radius_mm": radius,
            "sampling_larger_than_yarn_diameter": spacing > 2 * radius,
            "fine_weave_hole_resolution_certified": False,
            "max_samples_per_chunk": max_samples_per_chunk, "max_faces_per_chunk": max_faces_per_chunk,
            "compact_coverage": True, "socks": estimates,
            "total_samples": sum(e["requested_samples"] for e in estimates.values()),
            "conservative_total_bytes_including_reserve": planned + reserve_bytes,
            "allocated_total_bytes_including_reserve": sum(e["export_byte_limit"] for e in estimates.values()) + reserve_bytes,
            "features_generated": False, "global_spacing_for_both_configurations_and_socks": True,
            "resampling_rule": "recast geometry at the new material samples; reuse saved features only with identical pair/grid/yarn/ray signatures; do not interpolate holes"}


def prepare_triangle_sampling(pair, *, yarn_radius_mm, spacing_mm=None,
                              samples_per_radius=2):
    """Build a common X0/X1 grid, not a fixed N x N grid for every face.

    Each edge's divisions depend on its longest X0/X1 physical length.
    Face interiors use the face's longest edge at the same spacing.  Boundary
    samples are the canonical shared-edge grid, replacing the interior grid's
    boundary.  This retains identical physical samples across shared edges.
    The spacing is a sampling budget, not a guaranteed geometric error bound.
    """
    pair.validate()
    radius = _positive(yarn_radius_mm, "yarn_radius_mm")
    density = _positive(samples_per_radius, "samples_per_radius")
    spacing = radius / density if spacing_mm is None else _positive(spacing_mm, "spacing_mm")
    tri = pair.triangles
    physical_edges = np.stack([tri[:, [a, b]] for a, b in EDGES], axis=1)
    edge_lengths = []
    arrays = {"triangles": tri.copy()}
    for configuration, positions in (("reference", pair.reference_positions_mm),
                                      ("current", pair.current_positions_mm)):
        points = positions[tri]
        edges = triangle_edges(points)
        raw_normal = np.cross(edges[..., 0], edges[..., 1])
        vertex_normals = np.zeros_like(positions)
        for corner in range(3):
            np.add.at(vertex_normals, tri[:, corner], raw_normal)
        vertex_normals = _unit(vertex_normals, configuration + " vertex normal")
        derivatives = edges @ np.linalg.inv(triangle_edges(pair.triangle_material_uv))
        face_courses = _unit(derivatives[..., 0], configuration + " material +u")
        arrays[configuration + "_positions_mm"] = positions.copy()
        arrays[configuration + "_vertex_normals"] = vertex_normals
        arrays[configuration + "_face_courses"] = face_courses
        endpoints = positions[physical_edges]
        edge_lengths.append(np.linalg.norm(endpoints[..., 1, :] - endpoints[..., 0, :], axis=-1))
    longest = np.maximum(*edge_lengths)
    subdivisions = np.maximum(1, np.ceil(longest.max(axis=1) / spacing)).astype(np.int64)
    edge_divisions = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
    if np.any(subdivisions > 10**9):
        raise ValueError("sampling count exceeds the supported int64 layout")
    interior = np.maximum(subdivisions - 1, 0) * np.maximum(subdivisions - 2, 0) // 2
    counts = interior + edge_divisions.sum(axis=1)
    total = sum(int(x) for x in counts)
    if total > np.iinfo(np.int64).max:
        raise ValueError("sample offsets exceed int64")
    arrays.update(edge_longest_mm=longest, face_subdivisions=subdivisions, face_edge_divisions=edge_divisions,
                  face_sample_offsets=np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)])
    metadata = {"schema": SCHEMA, "schema_version": VERSION, "length_unit": "mm",
                "pair_fingerprint": pair_fingerprint(pair),
                "reference_version": pair.metadata["reference_version"],
                "material_version": pair.metadata["material_version"],
                "yarn_radius_mm": radius, "spacing_mm": spacing,
                "density_source": "explicit_spacing_mm" if spacing_mm is not None else "radius_divided_by_samples_per_radius",
                "samples_per_radius": density, "macro_strain_already_applied": True,
                "id_channels": [], "configuration_grid": "shared_material_barycentric_addresses",
                "sample_order": "three corners; edges 01,12,20 excluding corners; positive interior barycentric lattice",
                "frame_rule": "face material +u projected into area-weighted interpolated vertex-normal plane; normal cross course",
                "ray_normal_rule": "normalized barycentric interpolation of area-weighted physical vertex normals",
                "position_semantics": "offset_from_sample_shell_in_sample_orthonormal_frame",
                "sampling_error_bound_certified": False,
                "error_sources": ["centerline sampling", "tube polygon approximation", "smooth-ray shell approximation",
                                  "finite near-band and local material support", "sample spacing", "Blender float32 BVH", "float32 feature quantization"],
                "density_validation": "compare refined spacing and enlarged material support; not yet certified"}
    return TriangleSampling(arrays, metadata)


def validate_faces(sampling, face_indices):
    if face_indices is None:
        return np.arange(len(sampling.arrays["triangles"]), dtype=np.int64)
    result = np.asarray(face_indices)
    if (result.ndim != 1 or result.dtype.kind not in "iu" or len(result) == 0
            or np.any(result < 0) or np.any(result >= len(sampling.arrays["triangles"]))
            or len(np.unique(result)) != len(result)):
        raise ValueError("face_indices must be a nonempty unique in-range integer vector")
    return np.sort(result.astype(np.int64))


def face_sample_barycentric(sampling, face_index, local_indices):
    """Random access to a face's ragged sample lattice without allocating it."""
    face = int(face_index)
    indices = np.asarray(local_indices, dtype=np.int64)
    count = int(sampling.sample_offsets[face + 1] - sampling.sample_offsets[face])
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("face-local sample indices are out of range")
    output = np.zeros((len(indices), 3), dtype=np.float64)
    corners = indices < 3
    output[np.flatnonzero(corners), indices[corners]] = 1
    base = 3
    for (a, b), divisions in zip(EDGES, sampling.arrays["face_edge_divisions"][face]):
        divisions = int(divisions)
        mask = (indices >= base) & (indices < base + divisions - 1)
        t = (indices[mask] - base + 1) / divisions
        output[mask, a], output[mask, b] = 1 - t, t
        base += divisions - 1
    mask = indices >= base
    if mask.any():
        n = int(sampling.arrays["face_subdivisions"][face])
        t = indices[mask] - base
        discriminant = np.maximum(0., float((2 * n - 3)**2) - 8. * t)
        row = np.floor((2 * n - 3 - np.sqrt(discriminant)) / 2).astype(np.int64)
        before = lambda r: r * (2 * n - 3 - r) // 2
        # Correct the floating root at exact row boundaries.
        while np.any(before(row) > t):
            row[before(row) > t] -= 1
        while np.any((row < n - 3) & (before(row + 1) <= t)):
            row[(row < n - 3) & (before(row + 1) <= t)] += 1
        i, j = row + 1, t - before(row) + 1
        output[mask] = np.column_stack((n - i - j, i, j)) / n
    return output


def sample_domain(sampling, face_indices, barycentric, configuration):
    """Physical surface and right-handed sample frames, identical edge rays."""
    if configuration not in CONFIGURATIONS:
        raise ValueError("unknown configuration")
    indices = np.asarray(face_indices, dtype=np.int64)
    weights = np.asarray(barycentric, dtype=np.float64)
    if (indices.ndim != 1 or weights.shape != (len(indices), 3)
            or np.any(indices < 0) or np.any(indices >= len(sampling.arrays["triangles"]))
            or not np.isfinite(weights).all() or np.any(weights < -1e-12)
            or np.any(np.abs(weights.sum(axis=1) - 1) > 1e-12)):
        raise ValueError("invalid material surface samples")
    triangles = sampling.arrays["triangles"][indices]
    surface = np.einsum("ni,nij->nj", weights, sampling.arrays[configuration + "_positions_mm"][triangles])
    normals = _unit(np.einsum("ni,nij->nj", weights,
                            sampling.arrays[configuration + "_vertex_normals"][triangles]), "sample normal")
    course = sampling.arrays[configuration + "_face_courses"][indices].copy()
    course -= np.einsum("ni,ni->n", course, normals)[:, None] * normals
    course = _unit(course, "sample course projection")
    frames = np.stack((course, np.cross(normals, course), normals), axis=-1)
    return surface, frames


def iter_sample_chunks(sampling, face_indices=None, *, max_samples=65536, max_faces=1):
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if isinstance(max_faces, bool) or not isinstance(max_faces, (int, np.integer)) or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    selected = validate_faces(sampling, face_indices)
    faces, local, bary = [], [], []
    accumulated = 0
    accumulated_faces = 0
    for face in selected:
        if accumulated and accumulated_faces == max_faces:
            yield np.concatenate(faces), np.concatenate(local), np.concatenate(bary)
            faces, local, bary, accumulated, accumulated_faces = [], [], [], 0, 0
        accumulated_faces += 1
        count = int(sampling.sample_offsets[face + 1] - sampling.sample_offsets[face])
        start = 0
        while start < count:
            take = min(count - start, int(max_samples) - accumulated)
            indices = np.arange(start, start + take, dtype=np.int64)
            faces.append(np.full(take, face, dtype=np.int32))
            local.append(indices)
            bary.append(face_sample_barycentric(sampling, face, indices))
            start += take
            accumulated += take
            if accumulated == max_samples:
                yield np.concatenate(faces), np.concatenate(local), np.concatenate(bary)
                faces, local, bary, accumulated = [], [], [], 0
                accumulated_faces = 1 if start < count else 0
    if accumulated:
        yield np.concatenate(faces), np.concatenate(local), np.concatenate(bary)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(directory, exclude=()):
    excluded = {Path(path).resolve() for path in exclude}
    return sum(path.stat().st_size for path in Path(directory).rglob("*")
               if path.is_file() and path.resolve() not in excluded and not path.name.endswith(".tmp"))


def _write_npz(path, arrays, *, byte_limit=None, metadata_headroom=0):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    if byte_limit is not None and _directory_bytes(path.parent, (path,)) + temporary.stat().st_size + metadata_headroom > byte_limit:
        temporary.unlink()
        raise ValueError("export byte limit exceeded before committing NPZ; choose a coarser budget plan")
    temporary.replace(path)


def _write_json(path, data, *, byte_limit=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if byte_limit is not None and _directory_bytes(path.parent, (path,)) + temporary.stat().st_size > byte_limit:
        temporary.unlink()
        raise ValueError("export byte limit exceeded before committing metadata")
    temporary.replace(path)


def iter_geometry_sample_groups(sampling, faces, barycentric, configuration, provider, *,
                                world_tile_mm=.5, material_tile_mm=1., max_samples=2048):
    """Partition a storage chunk into bounded actual sample windows.

    Corners and boundary samples keep their stable storage addresses; geometry
    requests are grouped separately by world voxels and (when supplied) the
    provider's material q chart. No entire face box is formed from far corners.
    """
    surface, _ = sample_domain(sampling, faces, barycentric, configuration)
    keys = np.floor(surface / world_tile_mm).astype(np.int64)
    phase = getattr(provider, "phase", None)
    if phase is not None and hasattr(phase, "face_q_mm"):
        q = np.einsum("ni,nij->nj", barycentric, np.asarray(phase.face_q_mm)[faces])
        keys = np.column_stack((keys, np.floor(q / material_tile_mm).astype(np.int64)))
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        indices = order[a:b]
        for start in range(0, len(indices), max_samples):
            yield indices[start:start + max_samples]


def _sample_geometry_request(supplied, surface, frames, ray_offset, halo):
    from .bake import make_bvh
    from .surface_bake import sample_surface_points
    details, outside_check, triangle_tangent_valid = {}, None, None
    if isinstance(supplied, dict):
        geometry, bvh = supplied["geometry"], supplied.get("bvh")
        details = supplied.get("coverage", {})
        outside_check = supplied.get("origin_outside")
        triangle_tangent_valid = supplied.get("triangle_tangent_valid")
    elif isinstance(supplied, tuple):
        geometry, bvh = supplied
    else:
        geometry, bvh = supplied, None
    if details.get("geometry_available") is False:
        raise ValueError("provider reports missing geometry; leave chunk ungenerated")
    empty = not len(geometry.triangles) or not len(geometry.vertices)
    if empty and not details.get("empty_geometry_certified", False):
        raise ValueError("provider returned no generated yarn geometry; this is not a valid=false hole")
    if bvh is None and not empty:
        bvh = make_bvh(geometry)
    if triangle_tangent_valid is not None:
        triangle_tangent_valid = np.asarray(triangle_tangent_valid)
        if (triangle_tangent_valid.dtype != np.bool_
                or triangle_tangent_valid.shape != (len(geometry.triangles),)):
            raise ValueError("triangle_tangent_valid must be bool[tube_triangle_count]")
    output, stats = {}, {}
    for side in SIDES:
        tracked = _HitTrackingBVH(bvh, len(surface), triangle_tangent_valid) if triangle_tangent_valid is not None and not empty else None
        features = ({"valid": np.zeros(len(surface), dtype=bool),
                     **{key: np.zeros((len(surface), 3), dtype=np.float32) for key in FIELDS}}
                    if empty else sample_surface_points(geometry, tracked or bvh, surface, frames, side, ray_offset))
        invalid_tangent = None
        if tracked is not None:
            invalid_tangent = np.zeros(len(surface), dtype=bool)
            hits = tracked.indices >= 0
            invalid_tangent[hits] = ~triangle_tangent_valid[tracked.indices[hits]]
            features["valid"] &= ~invalid_tangent
            features["ray_tangent_invalid"] = invalid_tangent
            features["ray_tangent_check_available"] = np.ones(len(surface), dtype=bool)
        outside = None
        if outside_check is not None:
            if not callable(outside_check):
                raise ValueError("provider origin_outside must be a callable")
            sign = 1 if side == "front" else -1
            origins = surface + sign * ray_offset * frames[:, :, 2]
            outside = np.asarray(outside_check(origins))
            if outside.dtype != np.bool_ or outside.shape != (len(surface),):
                raise ValueError("origin_outside callback must return bool[sample_count]")
            features["valid"] &= outside
            features["ray_origin_outside"] = outside
            features["ray_origin_check_available"] = np.ones(len(surface), dtype=bool)
        for key in FIELDS:
            features[key][~features["valid"]] = 0
        output[side] = features
        stats[side] = {"hit_samples": int(features["valid"].sum()), "sample_count": len(surface),
            "origin_outside_checked": outside is not None,
            "origin_checked_samples": len(surface) if outside is not None else 0,
            "inside_origin_samples": int((~outside).sum()) if outside is not None else None,
            "tangent_checked_samples": len(surface) if triangle_tangent_valid is not None else 0,
            "invalid_tangent_samples": int(invalid_tangent.sum()) if invalid_tangent is not None else None}
    coverage = {"geometry_available": True, "tube_vertices": len(geometry.vertices),
                "tube_triangles": len(geometry.triangles), "material_halo_mm": halo,
                "global_first_hit_certified": False, **details}
    return output, stats, coverage


def _sample_analytic_request(supplied, sample_count, halo):
    """Validate provider-computed exact ray features; never apply macro F."""
    if not isinstance(supplied, dict) or supplied.get("coverage", {}).get("geometry_available") is False:
        raise ValueError("analytic provider reports missing geometry; leave chunk ungenerated")
    output, stats = {}, {}
    for side in SIDES:
        source = supplied["features"][side]
        valid = np.asarray(source["valid"])
        if valid.dtype != np.bool_ or valid.shape != (sample_count,):
            raise ValueError("analytic valid must be bool[sample_count]")
        values = {"valid": valid.copy()}
        for key in FIELDS:
            value = np.asarray(source[key], dtype=np.float32)
            if value.shape != (sample_count, 3) or not np.isfinite(value).all():
                raise ValueError("analytic " + key + " must be finite [sample_count,3]")
            values[key] = value.copy()
        for key in ("ray_origin_outside", "ray_origin_check_available", "ray_tangent_invalid", "ray_tangent_check_available"):
            if key in source:
                value = np.asarray(source[key])
                if value.dtype != np.bool_ or value.shape != (sample_count,):
                    raise ValueError("analytic " + key + " must be bool[sample_count]")
                values[key] = value.copy()
        outside = values.get("ray_origin_outside")
        checked = values.get("ray_origin_check_available", np.ones(sample_count, dtype=bool) if outside is not None else np.zeros(sample_count, dtype=bool))
        tangent_invalid = values.get("ray_tangent_invalid")
        tangent_checked = values.get("ray_tangent_check_available", np.ones(sample_count, dtype=bool) if tangent_invalid is not None else np.zeros(sample_count, dtype=bool))
        if outside is not None:
            values["valid"] &= outside | ~checked
            values["ray_origin_check_available"] = checked.copy()
        if tangent_invalid is not None:
            values["valid"] &= ~(tangent_invalid & tangent_checked)
            values["ray_tangent_check_available"] = tangent_checked.copy()
        hit = values["valid"]
        n, t = values["N_local"][hit], values["T_local"][hit]
        if (np.any(np.abs(np.linalg.norm(n, axis=1) - 1) > 1e-4)
                or np.any(np.abs(np.linalg.norm(t, axis=1) - 1) > 1e-4)
                or np.any(np.abs(np.einsum("ni,ni->n", n, t)) > 1e-4)):
            raise ValueError("analytic valid N/T must be unit and orthogonal")
        for key in FIELDS:
            values[key][~hit] = 0
        output[side] = values
        stats[side] = {"hit_samples": int(hit.sum()), "sample_count": sample_count,
            "origin_outside_checked": bool(checked.all()), "origin_checked_samples": int(checked.sum()),
            "inside_origin_samples": int((~outside & checked).sum()) if outside is not None else None,
            "tangent_checked_samples": int(tangent_checked.sum()),
            "invalid_tangent_samples": int((tangent_invalid & tangent_checked).sum()) if tangent_invalid is not None else None}
    coverage = {"geometry_available": True, "tube_vertices": 0, "tube_triangles": 0,
                "material_halo_mm": halo, "global_first_hit_certified": False,
                "intersection_protocol": "streamed_analytic_capsule_rays", **supplied.get("coverage", {})}
    return output, stats, coverage


def bake_triangle_features(pair, output_dir, geometry_provider, *, yarn_radius_mm,
                           ray_offset_mm, halo_mm, geometry_identity,
                           spacing_mm=None, samples_per_radius=2,
                           max_samples_per_chunk=65536, max_faces_per_chunk=1, face_indices=None,
                           geometry_world_tile_mm=.5, geometry_material_tile_mm=1.,
                           max_samples_per_geometry_request=2048,
                           compact_coverage=False, export_byte_limit=None,
                           max_new_chunks=None, resume=True, progress=None):
    """Stream paired features, with explicit partial/full scope and resume.

    ``geometry_provider(configuration, face_indices, halo_mm)`` returns actual
    Geometry, (Geometry, BVH), or {geometry, bvh?, coverage?}.  The material
    support/halo is the provider's responsibility.  An empty geometry is not
    recorded as a fabric hole.  A local halo's first hit is not a certified
    whole-geometry first hit; preserve provider coverage diagnostics.

    When available, ``provider.for_samples(configuration, faces, bary, halo)``
    replaces whole-face requests.  Storage chunks are independently grouped
    into bounded world/material sample tiles, intersected one request at a
    time, and scattered back to the original addresses.  Missing support
    aborts the uncommitted storage chunk.  Explicitly certified empty support
    may yield real no-hit samples.  Capsule origin/pole diagnostics are kept
    separately from no-hit fabric holes.
    """
    ray_offset = _positive(ray_offset_mm, "ray_offset_mm")
    halo = _positive(halo_mm, "halo_mm")
    if halo < ray_offset:
        raise ValueError("halo_mm must cover at least ray_offset_mm; local geometry cannot cover the prescribed near-band otherwise")
    if (isinstance(max_samples_per_chunk, bool) or not isinstance(max_samples_per_chunk, (int, np.integer))
            or max_samples_per_chunk < 1):
        raise ValueError("max_samples_per_chunk must be a positive integer")
    if (isinstance(max_faces_per_chunk, bool) or not isinstance(max_faces_per_chunk, (int, np.integer))
            or max_faces_per_chunk < 1):
        raise ValueError("max_faces_per_chunk must be a positive integer")
    world_tile = _positive(geometry_world_tile_mm, "geometry_world_tile_mm")
    material_tile = _positive(geometry_material_tile_mm, "geometry_material_tile_mm")
    if (isinstance(max_samples_per_geometry_request, bool)
            or not isinstance(max_samples_per_geometry_request, (int, np.integer))
            or max_samples_per_geometry_request < 1):
        raise ValueError("max_samples_per_geometry_request must be a positive integer")
    sample_provider = getattr(geometry_provider, "for_samples", None)
    analytic_provider = getattr(geometry_provider, "sample_capsule_rays", None)
    analytic = callable(analytic_provider) and getattr(geometry_provider, "representation", "analytic_capsule_union") in ("capsule_union", "analytic_capsule_union")
    windowed = callable(sample_provider) or analytic
    if export_byte_limit is not None:
        if type(export_byte_limit) is not int or export_byte_limit <= 0:
            raise ValueError("export_byte_limit must be a positive integer")
        if not compact_coverage:
            raise ValueError("export_byte_limit requires compact_coverage=True")
    if not isinstance(geometry_identity, dict) or not geometry_identity:
        raise ValueError("geometry_identity must version the supplied yarn geometry/parameters")
    if max_new_chunks is not None and (isinstance(max_new_chunks, bool)
            or not isinstance(max_new_chunks, int) or max_new_chunks < 0):
        raise ValueError("max_new_chunks must be nonnegative or None")
    sampling = prepare_triangle_sampling(pair, yarn_radius_mm=yarn_radius_mm,
        spacing_mm=spacing_mm, samples_per_radius=samples_per_radius)
    selected = validate_faces(sampling, face_indices)
    estimate = sampling.estimate(selected)
    if export_byte_limit is not None:
        estimate["budget_export_estimate"] = estimate_triangle_export(sampling, selected,
            max_samples_per_chunk=max_samples_per_chunk, max_faces_per_chunk=max_faces_per_chunk)
        if estimate["budget_export_estimate"]["conservative_export_bytes"] > export_byte_limit:
            raise ValueError("density exceeds the allocated export byte limit; use the overall budget plan spacing")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if export_byte_limit is not None and _directory_bytes(directory) > export_byte_limit:
        raise ValueError("existing export directory already exceeds its allocated byte limit")
    identity = {**sampling.metadata, "requested_faces": selected.tolist(),
                "geometry_identity": geometry_identity, "ray_offset_mm": ray_offset,
                "halo_mm": halo, "max_samples_per_chunk": int(max_samples_per_chunk),
                "max_faces_per_chunk": int(max_faces_per_chunk)}
    if windowed:
        identity["geometry_requests"] = {"protocol": "sample_capsule_rays" if analytic else "for_samples(configuration,faces,barycentric,halo_mm)",
            "world_tile_mm": world_tile, "material_tile_mm": material_tile,
            "max_samples_per_geometry_request": int(max_samples_per_geometry_request),
            "storage_sample_order": "unchanged; window intersections scatter back to original addresses"}
    if compact_coverage or export_byte_limit is not None:
        identity.update(compact_coverage=bool(compact_coverage), export_byte_limit=export_byte_limit)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest_path = directory / "triangle_features.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("feature manifest exists; choose another output directory or resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_signature") != signature:
            raise ValueError("resume input/grid/yarn/ray identity changed")
        if _sha256(directory / "sampling_domain.npz") != manifest["domain_sha256"]:
            raise ValueError("sampling domain changed since the committed manifest")
        for chunk in manifest["chunks"]:
            if _sha256(directory / chunk["file"]) != chunk["sha256"]:
                raise ValueError("committed feature chunk changed: " + chunk["file"])
    else:
        domain = dict(sampling.arrays, requested_faces=selected,
                      metadata_json=np.array(json.dumps(sampling.metadata, allow_nan=False)))
        _write_npz(directory / "sampling_domain.npz", domain, byte_limit=export_byte_limit, metadata_headroom=2_000_000)
        manifest = {**identity, "generation_signature": signature,
                    "domain_file": "sampling_domain.npz", "domain_sha256": _sha256(directory / "sampling_domain.npz"),
                    "estimate": estimate, "chunks": [], "completed_samples": 0,
                    "completed_faces": 0, "requested_scope_full": len(selected) == len(pair.triangles),
                    "requested_scope_complete": False, "full_surface_features_generated": False,
                    "feature_storage": "signed float32 offsets and unit local N/T; bool valid",
                    "address_storage": "internal face/sample addresses and float64 barycentrics; no optical ID channels",
                    "valid_semantics": "first hit within +/-ray_offset of specified smooth-normal ray; false means no hit or separately diagnosed invalid origin/tangent; missing geometry aborts uncommitted chunk",
                    "near_band_scope": "finite segment of length 2*ray_offset_mm",
                    "geometry_coverage": "provider material support; global first-hit completeness not certified",
                    "global_tube_edge_nearband_completeness": "not_certified",
                    "ray_origin_outside_validation": "provider callback when present; otherwise not_certified",
                    "halo_validation": "not_checked", "sampling_refinement_validation": "not_checked"}
        if analytic:
            manifest["actual_intersection_error_model"] = {
                "method": "float64 analytic ray/polyline capsule union",
                "tube_polygon_and_Blender_BVH_used": False,
                "sources": ["centerline polyline discretization", "piecewise shell and interpolated director",
                    "finite near-band/local material support", "material sample spacing",
                    "float64 algebra and boundary tolerances", "float32 stored feature quantization"],
                "legacy_error_sources_scope": "generic sampling metadata; actual method described here"}
        _write_json(manifest_path, manifest, byte_limit=export_byte_limit)
    new_chunks = 0
    committed = len(manifest["chunks"])
    for number, (faces, local, bary) in enumerate(iter_sample_chunks(sampling, selected,
                            max_samples=max_samples_per_chunk, max_faces=max_faces_per_chunk)):
        if number < committed:
            continue
        if max_new_chunks is not None and new_chunks >= max_new_chunks:
            break
        chunk_started = time.perf_counter()
        arrays = {"face_indices": faces, "face_local_indices": local, "barycentric": bary}
        coverage, stats = {}, {}
        def record_failure(configuration, message):
            manifest["last_failure"] = {"chunk": number, "configuration": configuration,
                "faces": np.unique(faces).tolist(), "geometry_available": False,
                "samples_committed": False, "message": message}
            _write_json(manifest_path, manifest, byte_limit=export_byte_limit)
        for configuration in CONFIGURATIONS:
            surface_all, frames_all = sample_domain(sampling, faces, bary, configuration)
            groups = (iter_geometry_sample_groups(sampling, faces, bary, configuration, geometry_provider,
                        world_tile_mm=world_tile, material_tile_mm=material_tile,
                        max_samples=int(max_samples_per_geometry_request))
                      if windowed else (np.arange(len(faces), dtype=np.int64),))
            records = []
            request_progress_time = time.perf_counter()
            samples_intersected = 0
            if progress:
                progress({"event": "geometry_configuration_start", "chunk": number,
                    "configuration": configuration, "chunk_samples": len(faces),
                    "completed_samples": manifest["completed_samples"]})
            for side in SIDES:
                stats[configuration + "_" + side] = {"hit_samples": 0, "sample_count": len(faces),
                    "origin_outside_checked": True, "origin_checked_samples": 0,
                    "inside_origin_samples": None, "tangent_checked_samples": 0,
                    "invalid_tangent_samples": None}
            for request_number, indices in enumerate(groups):
                try:
                    if analytic:
                        supplied = analytic_provider(configuration, faces[indices].astype(np.int64), bary[indices],
                            surface_all[indices], frames_all[indices], ray_offset, halo)
                        features, request_stats, request_coverage = _sample_analytic_request(supplied, len(indices), halo)
                    else:
                        supplied = (sample_provider(configuration, faces[indices].astype(np.int64), bary[indices], halo)
                                    if windowed else geometry_provider(configuration, np.unique(faces).astype(np.int64), halo))
                        features, request_stats, request_coverage = _sample_geometry_request(
                            supplied, surface_all[indices], frames_all[indices], ray_offset, halo)
                    for side in SIDES:
                        prefix = configuration + "_" + side + "_"
                        for key, value in features[side].items():
                            if prefix + key not in arrays:
                                arrays[prefix + key] = np.zeros((len(faces),) + value.shape[1:], dtype=value.dtype)
                            arrays[prefix + key][indices] = value
                        total = stats[configuration + "_" + side]
                        part = request_stats[side]
                        total["hit_samples"] += part["hit_samples"]
                        total["origin_outside_checked"] &= part["origin_outside_checked"]
                        for key in ("origin_checked_samples", "tangent_checked_samples"):
                            total[key] += part[key]
                        for key in ("inside_origin_samples", "invalid_tangent_samples"):
                            if part[key] is not None:
                                total[key] = (total[key] or 0) + part[key]
                    if windowed:
                        records.append({"request": request_number, "samples": len(indices),
                            "storage_position_range": [int(indices.min()), int(indices.max())],
                            "world_surface_bounds_mm": [surface_all[indices].min(axis=0).tolist(),
                                                        surface_all[indices].max(axis=0).tolist()],
                            "coverage": request_coverage})
                    else:
                        coverage[configuration] = request_coverage
                    del supplied
                    samples_intersected += len(indices)
                    if progress and time.perf_counter() - request_progress_time >= 30:
                        progress({"event": "geometry_request_progress", "chunk": number,
                            "configuration": configuration, "geometry_requests_completed": request_number + 1,
                            "samples_intersected": samples_intersected, "chunk_samples": len(faces),
                            "chunk_elapsed_seconds": time.perf_counter() - chunk_started,
                            "completed_samples": manifest["completed_samples"]})
                        request_progress_time = time.perf_counter()
                except Exception as exc:
                    record_failure(configuration, str(exc))
                    manifest["last_failure"].update(geometry_request=request_number,
                        request_sample_count=len(indices),
                        request_world_surface_bounds_mm=[surface_all[indices].min(axis=0).tolist(),
                                                         surface_all[indices].max(axis=0).tolist()])
                    _write_json(manifest_path, manifest, byte_limit=export_byte_limit)
                    raise
            if windowed:
                coverage[configuration] = {"geometry_available": True,
                    "protocol": "sample_windows", "geometry_requests": len(records),
                    "samples_intersected": sum(r["samples"] for r in records),
                    "maximum_request_samples": max(r["samples"] for r in records),
                    "maximum_tube_vertices_per_request": max(r["coverage"]["tube_vertices"] for r in records),
                    "maximum_tube_triangles_per_request": max(r["coverage"]["tube_triangles"] for r in records),
                    "global_first_hit_certified": False,
                    "intersection_protocol": "streamed_analytic_capsule_rays" if analytic else "tube_BVH"}
                if compact_coverage:
                    coverage[configuration].update(request_details_saved=False,
                        world_surface_bounds_mm=[surface_all.min(axis=0).tolist(), surface_all.max(axis=0).tolist()],
                        provider_global_first_hit_certified=all(r["coverage"].get("global_first_hit_certified", False) for r in records),
                        provider_intersection_protocols=sorted(set(r["coverage"].get("intersection_protocol", "tube_BVH") for r in records)))
                    timing_keys = sorted(set(k for r in records for k in r["coverage"].get("computation_seconds", {})))
                    if timing_keys:
                        coverage[configuration]["provider_computation_seconds"] = {key: sum(float(r["coverage"].get("computation_seconds", {}).get(key, 0)) for r in records) for key in timing_keys}
                    for key in ("candidate_pair_count", "front_inside_origin_count", "back_inside_origin_count", "front_tangent_invalid_count", "back_tangent_invalid_count"):
                        if any(key in r["coverage"] for r in records):
                            coverage[configuration][key] = sum(int(r["coverage"].get(key, 0)) for r in records)
                else:
                    coverage[configuration]["requests"] = records
        filename = "chunk-%06d.npz" % number
        headroom = len(json.dumps(manifest, ensure_ascii=False).encode("utf-8")) * 2 + 65536 + len(np.unique(faces)) * 32
        _write_npz(directory / filename, arrays, byte_limit=export_byte_limit, metadata_headroom=headroom)
        chunk = {"file": filename, "sha256": _sha256(directory / filename), "samples": len(faces),
                 "faces": np.unique(faces).tolist(), "first_address": [int(faces[0]), int(local[0])],
                 "last_address": [int(faces[-1]), int(local[-1])], "geometry_coverage": coverage, "stats": stats}
        manifest["chunks"].append(chunk)
        manifest["completed_samples"] += len(faces)
        ended = np.unique(faces[local == np.diff(sampling.sample_offsets)[faces] - 1])
        manifest["completed_faces"] += len(ended)
        manifest["requested_scope_complete"] = manifest["completed_samples"] == estimate["requested_samples"]
        manifest["full_surface_features_generated"] = bool(manifest["requested_scope_full"] and manifest["requested_scope_complete"])
        manifest.pop("last_failure", None)
        _write_json(manifest_path, manifest, byte_limit=export_byte_limit)
        new_chunks += 1
        if progress:
            progress({"chunk": number, "completed_samples": manifest["completed_samples"],
                      "requested_samples": estimate["requested_samples"],
                      "completed_faces": manifest["completed_faces"],
                      "chunk_seconds": time.perf_counter() - chunk_started,
                      "geometry_requests": sum(c.get("geometry_requests", 1) for c in coverage.values()),
                      "actual_export_bytes": _directory_bytes(directory) if export_byte_limit is not None else None,
                      "export_byte_limit": export_byte_limit,
                      "requested_scope_complete": manifest["requested_scope_complete"]})
    return manifest
