"""Measured kinematics of corresponding triangular shell configurations.

No constitutive material, curvature inference, dressing, or equilibrium solve
is performed. UV is a material chart; all physical derivatives use X0 in mm.
"""

from dataclasses import dataclass

import numpy as np

from .garment_data import GarmentMeshPair, triangle_edges


@dataclass
class DeformationField:
    reference_metric: np.ndarray
    current_metric: np.ndarray
    surface_deformation_gradient: np.ndarray
    reference_frames: np.ndarray
    current_frames: np.ndarray
    course_stretch: np.ndarray
    wale_stretch: np.ndarray
    reference_material_angle_rad: np.ndarray
    current_material_angle_rad: np.ndarray
    shear_angle_rad: np.ndarray
    area_ratio: np.ndarray
    principal_stretches: np.ndarray
    reference_face_centers_mm: np.ndarray
    current_face_centers_mm: np.ndarray
    metadata: dict
    curvature: None = None

    @property
    def F(self):
        return self.surface_deformation_gradient

    @property
    def B1(self):
        return self.current_frames


def _normalize(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("cannot construct a finite surface frame")
    return vectors / lengths


def _frame(derivatives, edges):
    normal = _normalize(np.cross(edges[..., 0], edges[..., 1]))
    course = _normalize(derivatives[..., 0])
    # Roundoff cleanup, not a removal of physical shear from the derivative.
    course = _normalize(course - np.einsum("ij,ij->i", course, normal)[:, None] * normal)
    across = _normalize(np.cross(normal, course))
    return np.stack((course, across, normal), axis=-1)


def _material_angle(derivatives):
    course, wale = derivatives[..., 0], derivatives[..., 1]
    # atan2 is stable for near-orthogonal and acute/obtuse material directions.
    return np.arctan2(np.linalg.norm(np.cross(course, wale), axis=-1),
                      np.einsum("ij,ij->i", course, wale))


def compute_deformation_field(pair: GarmentMeshPair) -> DeformationField:
    """Compute one piecewise-constant deformation state per physical face.

    F[M,3,2] maps physical displacements in the reference orthonormal basis
    [course, reference-normal x course] into current space. Metrics are A^T A,
    with A = dX/d(material UV), and retain the chart's physical scale. Positive
    shear_angle_rad means the angle between +course/+wale has decreased.
    Curvature requires a separately validated neighborhood estimator and is
    explicitly absent rather than filled with misleading zero values.
    """
    pair.validate()
    points0 = pair.reference_positions_mm[pair.triangles]
    points1 = pair.current_positions_mm[pair.triangles]
    edges0, edges1 = triangle_edges(points0), triangle_edges(points1)
    uv_edges = triangle_edges(pair.triangle_material_uv)
    uv_inverse = np.linalg.inv(uv_edges)
    a0, a1 = edges0 @ uv_inverse, edges1 @ uv_inverse
    b0, b1 = _frame(a0, edges0), _frame(a1, edges1)
    reference_chart_in_basis = np.swapaxes(b0[..., :2], -1, -2) @ a0
    gradient = a1 @ np.linalg.inv(reference_chart_in_basis)
    metric0 = np.swapaxes(a0, -1, -2) @ a0
    metric1 = np.swapaxes(a1, -1, -2) @ a1
    stretches = np.linalg.norm(a1, axis=1) / np.linalg.norm(a0, axis=1)
    angle0, angle1 = _material_angle(a0), _material_angle(a1)
    area_ratio = (np.linalg.norm(np.cross(edges1[..., 0], edges1[..., 1]), axis=-1)
                  / np.linalg.norm(np.cross(edges0[..., 0], edges0[..., 1]), axis=-1))
    eigenvalues = np.linalg.eigvalsh(np.swapaxes(gradient, -1, -2) @ gradient)
    if not np.isfinite(gradient).all() or not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("deformation field is numerically singular or nonfinite")
    return DeformationField(
        reference_metric=metric0, current_metric=metric1,
        surface_deformation_gradient=gradient, reference_frames=b0, current_frames=b1,
        course_stretch=stretches[:, 0], wale_stretch=stretches[:, 1],
        reference_material_angle_rad=angle0, current_material_angle_rad=angle1,
        shear_angle_rad=angle0 - angle1, area_ratio=area_ratio,
        principal_stretches=np.sqrt(eigenvalues[:, ::-1]),
        reference_face_centers_mm=points0.mean(axis=1),
        current_face_centers_mm=points1.mean(axis=1),
        metadata={"schema": "stocking.deformation_field", "schema_version": 1,
                  "length_unit": "mm", "angle_unit": "radian",
                  "reference_version": pair.metadata["reference_version"],
                  "material_version": pair.metadata["material_version"],
                  "metric_basis": "material_uv_chart",
                  "F_domain": "reference_orthonormal_course_crosscourse_mm",
                  "B1_columns": ["current_course", "current_normal_cross_course", "current_normal"],
                  "shear_definition": "reference_angle_minus_current_angle",
                  "curvature_status": "not_computed", "status": "kinematics_only"})


def validate_surface_samples(pair, face_indices, barycentric):
    indices = np.asarray(face_indices)
    if indices.dtype.kind not in "iu":
        raise ValueError("face_indices must be integer values")
    if np.any(indices < 0) or np.any(indices >= len(pair.triangles)):
        raise ValueError("surface sample face index is out of range")
    weights = np.asarray(barycentric, dtype=np.float64)
    if weights.shape != indices.shape + (3,) or not np.isfinite(weights).all():
        raise ValueError("barycentric must have face_indices.shape + (3,) finite entries")
    if (np.any(weights < -1e-12) or np.any(weights > 1 + 1e-12)
            or np.any(np.abs(weights.sum(axis=-1) - 1) > 1e-12)):
        raise ValueError("barycentric samples must lie inside a face and sum to one")
    return indices.astype(np.int64), weights


def sample_surface(pair: GarmentMeshPair, face_indices, barycentric, *, current=True):
    """S0(q)/S1(q), with q=(stable face index, reference barycentric weights).

    A face index is meaningful only with the same pair/reference version. No
    remeshing, nearest-point reassignment, or UV physical-length conversion is
    hidden here. Scalars return [3], batches return [...,3].
    """
    indices, weights = validate_surface_samples(pair, face_indices, barycentric)
    positions = pair.current_positions_mm if current else pair.reference_positions_mm
    return np.einsum("...i,...ij->...j", weights, positions[pair.triangles[indices]])


def sample_material_uv(pair: GarmentMeshPair, face_indices, barycentric):
    """Return chart-local material coordinates using the same sample q."""
    indices, weights = validate_surface_samples(pair, face_indices, barycentric)
    return np.einsum("...i,...ij->...j", weights, pair.triangle_material_uv[indices])
