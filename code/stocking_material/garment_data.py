"""Versioned correspondence between a reference and current physical sock shell.

This is a geometry contract, not a dressing solver or a calibrated material.
Positions are millimetres; material UV identifies material and does not imply
physical length. Physical vertices have unique stable IDs. UV seam duplicates
belong to a separate render mesh, or can use the explicit per-face UV chart.
"""

from dataclasses import dataclass
import copy
import json

import numpy as np


GARMENT_SCHEMA = "stocking.garment_mesh_pair"
GARMENT_SCHEMA_VERSION = 1
REFERENCE_ROLES = frozenset(("generated_initial", "measured_finished_unworn",
                           "effective_finished_reference", "synthetic_unworn"))


def default_garment_metadata(reference_version, material_version,
                             reference_role="synthetic_unworn", **extra):
    """Make explicit version/units metadata; no material accuracy is assumed."""
    result = dict(extra)
    result.update(schema=GARMENT_SCHEMA, schema_version=GARMENT_SCHEMA_VERSION,
                  length_unit="mm", reference_version=reference_version,
                  material_version=material_version, reference_role=reference_role)
    return result


def validate_garment_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("garment metadata must be a JSON object")
    if metadata.get("schema") != GARMENT_SCHEMA:
        raise ValueError("unsupported garment schema")
    version = metadata.get("schema_version")
    if type(version) is not int or version != GARMENT_SCHEMA_VERSION:
        raise ValueError("unsupported garment schema_version")
    if metadata.get("length_unit") != "mm":
        raise ValueError("garment length_unit must be 'mm'; convert explicitly before import")
    for key in ("reference_version", "material_version"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata {key} must be a nonempty version string")
    if metadata.get("reference_role") not in REFERENCE_ROLES:
        raise ValueError("unsupported reference_role; an unworn pose is not automatically stress-free")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("garment metadata must contain finite JSON values") from exc


def _float_array(value, name, shape):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    array = np.array(original, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite entries")
    return array


def _integer_array(value, name, shape):
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and integer entries")
    if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must contain nonnegative int64-representable entries")
    return np.array(array, dtype=np.int64, copy=True)


def _triangle_regular(edges, name):
    """Scale-relative conditioning test, invariant under change of length units."""
    if edges.shape[-2:] == (3, 2):
        e, f = edges[..., 0], edges[..., 1]
        twice_area = np.linalg.norm(np.cross(e, f), axis=-1)
    elif edges.shape[-2:] == (2, 2):
        e, f = edges[..., 0], edges[..., 1]
        twice_area = np.abs(e[..., 0] * f[..., 1] - e[..., 1] * f[..., 0])
    else:
        raise ValueError("internal triangle edges have an unexpected shape")
    scale2 = np.maximum.reduce((np.sum(e * e, axis=-1), np.sum(f * f, axis=-1),
                                np.sum((f - e) ** 2, axis=-1)))
    if (not np.isfinite(twice_area).all() or not np.isfinite(scale2).all()
            or np.any(scale2 <= 0) or np.any(twice_area <= 1e-12 * scale2)):
        raise ValueError(f"{name} contains degenerate or numerically singular triangles")


def triangle_edges(values):
    """Return edge column matrices from triangle-corner values [M,3,D]."""
    return np.stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]), axis=-1)


def _validate_topology(triangles):
    if any(len(set(row)) != 3 for row in triangles.tolist()):
        raise ValueError("triangles contain repeated vertices")
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError("triangles contain duplicate faces")
    edges = {}
    for a, b, c in triangles:
        for i, j in ((a, b), (b, c), (c, a)):
            key = (min(int(i), int(j)), max(int(i), int(j)))
            directions = edges.setdefault(key, [])
            directions.append(1 if i < j else -1)
            if len(directions) > 2:
                raise ValueError("physical shell has a nonmanifold edge")
    if any(len(d) == 2 and d[0] == d[1] for d in edges.values()):
        raise ValueError("neighboring faces have inconsistent winding")


def validate_mesh_correspondence(reference_triangles, current_triangles,
                                 reference_material_point_ids, current_material_point_ids):
    """Require existing vertex order and topology; never silently reorder IDs."""
    for before, after, name in ((reference_triangles, current_triangles, "triangles"),
                                (reference_material_point_ids, current_material_point_ids,
                                 "material_point_ids")):
        left, right = np.asarray(before), np.asarray(after)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"reference/current {name} mismatch; explicit correspondence is required")


@dataclass
class GarmentMeshPair:
    reference_positions_mm: np.ndarray
    current_positions_mm: np.ndarray
    triangles: np.ndarray
    material_uv: np.ndarray
    material_point_ids: np.ndarray
    metadata: dict
    face_material_uv: np.ndarray | None = None
    current_face_directors: np.ndarray | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check and own arrays, preserving all explicitly supplied physical IDs.

        Optional directors must be independently supplied outward current-face
        directions. They detect inversion relative to that orientation. Without
        a director/history, a 3D endpoint pair cannot distinguish an inversion
        from a 180-degree rigid rotation; no false guarantee is made here.
        """
        before = np.asarray(self.reference_positions_mm)
        if before.ndim != 2 or before.shape[1] != 3 or len(before) < 3:
            raise ValueError("reference_positions_mm must be [N>=3,3]")
        n = len(before)
        faces = np.asarray(self.triangles)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise ValueError("triangles must be [M>=1,3]")
        m = len(faces)
        self.reference_positions_mm = _float_array(before, "reference_positions_mm", (n, 3))
        self.current_positions_mm = _float_array(self.current_positions_mm, "current_positions_mm", (n, 3))
        self.triangles = _integer_array(faces, "triangles", (m, 3))
        if np.any(self.triangles >= n):
            raise ValueError("triangle vertex index is out of range")
        _validate_topology(self.triangles)
        self.material_uv = _float_array(self.material_uv, "material_uv", (n, 2))
        self.material_point_ids = _integer_array(self.material_point_ids, "material_point_ids", (n,))
        if len(np.unique(self.material_point_ids)) != n:
            raise ValueError("physical material_point_ids must be unique; keep render seam duplication separate")
        if self.face_material_uv is not None:
            self.face_material_uv = _float_array(self.face_material_uv, "face_material_uv", (m, 3, 2))
        chart = self.material_uv[self.triangles] if self.face_material_uv is None else self.face_material_uv
        _triangle_regular(triangle_edges(chart), "material UV charts")
        for name, positions in (("reference shell", self.reference_positions_mm),
                                 ("current shell", self.current_positions_mm)):
            _triangle_regular(triangle_edges(positions[self.triangles]), name)
        validate_garment_metadata(self.metadata)
        self.metadata = copy.deepcopy(self.metadata)
        if self.current_face_directors is not None:
            directors = _float_array(self.current_face_directors, "current_face_directors", (m, 3))
            lengths = np.linalg.norm(directors, axis=-1)
            if not np.isfinite(lengths).all() or np.any(lengths <= 0):
                raise ValueError("current_face_directors must have finite nonzero length")
            directors /= lengths[:, None]
            e = triangle_edges(self.current_positions_mm[self.triangles])
            normals = np.cross(e[..., 0], e[..., 1])
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            if np.any(np.einsum("ij,ij->i", normals, directors) <= 1e-10):
                raise ValueError("current face is inverted or orthogonal relative to its supplied outward director")
            self.current_face_directors = directors
        return self

    @property
    def triangle_material_uv(self):
        """Chart-local corner UV; explicit charts support periodic seams."""
        return self.material_uv[self.triangles] if self.face_material_uv is None else self.face_material_uv

    @classmethod
    def from_meshes(cls, *, reference_triangles, current_triangles,
                    reference_material_point_ids, current_material_point_ids, **kwargs):
        validate_mesh_correspondence(reference_triangles, current_triangles,
                                     reference_material_point_ids, current_material_point_ids)
        return cls(triangles=reference_triangles, material_point_ids=reference_material_point_ids,
                   **kwargs)
