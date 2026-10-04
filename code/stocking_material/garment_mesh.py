"""Explicit import of an existing independent Blender X0/X1 shell pair.

No bpy dependency is imported at module load. This adapter imports authored
mesh data, not modifier/evaluated geometry, and never constructs X0 by scaling
or shrinkwrapping X1. It is not a dressing solver. Prepare physical material
point IDs once, then copy and deform the same topology for the current state.
"""

import math
from numbers import Real

import numpy as np

from .garment_data import GarmentMeshPair, default_garment_metadata, validate_mesh_correspondence


def _mesh_data(obj, role):
    if getattr(obj, "type", None) != "MESH":
        raise ValueError(f"{role} object must be a mesh")
    if any(getattr(modifier, "show_viewport", True) for modifier in getattr(obj, "modifiers", ())):
        raise ValueError(f"{role} has active modifiers; apply to a separate same-topology source copy first")
    mesh = obj.data
    if getattr(mesh, "shape_keys", None) is not None:
        raise ValueError(f"{role} has shape keys; create an explicit source mesh copy of the chosen state")
    return mesh


def _world_positions(obj, mesh, scale):
    transform = np.asarray(obj.matrix_world, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("object matrix_world must be a finite affine transform")
    determinant = np.linalg.det(transform[:3, :3])
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("object transform is singular or mirrored; repair winding/orientation explicitly")
    local = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
    return (local @ transform[:3, :3].T + transform[:3, 3]) * scale


def _material_ids(mesh, attribute_name, explicit, role):
    if explicit is not None:
        return np.asarray(explicit)
    attribute = mesh.attributes.get(attribute_name)
    if (attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT"
            or len(attribute.data) != len(mesh.vertices)):
        raise ValueError(f"{role} needs POINT INT '{attribute_name}' or explicitly supplied physical IDs; "
                         "IDs are never inferred or silently renumbered")
    return np.asarray([entry.value for entry in attribute.data], dtype=np.int64)


def _uv_layer(mesh, name, *, required):
    layer = mesh.uv_layers.get(name) if name is not None else mesh.uv_layers.active
    if required and layer is None:
        raise ValueError("reference mesh needs a material UV layer")
    return layer


def garment_pair_from_objects(reference_object, current_object, *, reference_version,
                              material_version, reference_role="measured_finished_unworn",
                              millimeters_per_blender_unit=None, scene=None,
                              material_point_id_attribute="material_point_id", material_uv_layer=None,
                              reference_material_point_ids=None, current_material_point_ids=None,
                              current_face_directors_world=None, metadata=None):
    """Import independently supplied X0/X1, preserving source material identity.

    Default conversion is scene.unit_settings.scale_length * 1000 (Blender unit
    -> metres -> mm). An explicit mm/BU override is required when a scene uses a
    different convention; this routine never guesses scale from object size.
    Reference polygon triangulation is reused after exact polygon/ID matching,
    avoiding geometry-dependent re-triangulation in X1. Per-loop reference UV
    becomes explicit per-face chart coordinates, including render UV seams.
    Optional independently known current outward directors enable orientation
    inversion checks; endpoint geometry alone does not establish that history.
    """
    if millimeters_per_blender_unit is None:
        if scene is None:
            import bpy
            scene = bpy.context.scene
        millimeters_per_blender_unit = scene.unit_settings.scale_length * 1000.0
    scale = millimeters_per_blender_unit
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("millimeters_per_blender_unit must be a finite positive number")
    scale = float(scale)
    mesh0 = _mesh_data(reference_object, "reference")
    mesh1 = _mesh_data(current_object, "current")
    if len(mesh0.vertices) != len(mesh1.vertices):
        raise ValueError("reference/current physical vertex counts mismatch")
    polygons0 = [tuple(polygon.vertices) for polygon in mesh0.polygons]
    polygons1 = [tuple(polygon.vertices) for polygon in mesh1.polygons]
    if polygons0 != polygons1:
        raise ValueError("reference/current polygon topology or winding mismatch")
    ids0 = _material_ids(mesh0, material_point_id_attribute, reference_material_point_ids, "reference")
    ids1 = _material_ids(mesh1, material_point_id_attribute, current_material_point_ids, "current")
    layer0 = _uv_layer(mesh0, material_uv_layer, required=True)
    layer1 = _uv_layer(mesh1, material_uv_layer if material_uv_layer is not None else layer0.name,
                       required=False)
    uv0 = np.asarray([tuple(entry.uv) for entry in layer0.data], dtype=np.float64)
    if layer1 is not None:
        uv1 = np.asarray([tuple(entry.uv) for entry in layer1.data], dtype=np.float64)
        if uv0.shape != uv1.shape or not np.array_equal(uv0, uv1):
            raise ValueError("current material UV changed; preserve reference UV/material phase")
    mesh0.calc_loop_triangles()
    triangles = np.asarray([tuple(triangle.vertices) for triangle in mesh0.loop_triangles], dtype=np.int64)
    triangle_loops = np.asarray([tuple(triangle.loops) for triangle in mesh0.loop_triangles], dtype=np.int64)
    validate_mesh_correspondence(triangles, triangles, ids0, ids1)
    chart_uv = uv0[triangle_loops]
    # This representative per-vertex coordinate is a marker only. Face-local
    # loop charts are authoritative for all derivatives at an actual UV seam.
    vertex_uv = np.full((len(mesh0.vertices), 2), np.nan)
    for loop in mesh0.loops:
        if not np.isfinite(vertex_uv[loop.vertex_index]).all():
            vertex_uv[loop.vertex_index] = uv0[loop.index]
    if not np.isfinite(vertex_uv).all():
        raise ValueError("physical shell has unused vertices with no material UV")
    extra = {} if metadata is None else dict(metadata)
    expected = default_garment_metadata(reference_version, material_version, reference_role)
    for key, value in expected.items():
        if key in extra and extra.pop(key) != value:
            raise ValueError(f"import metadata {key} conflicts with the explicit import contract")
    extra.update(source_reference_object=reference_object.name,
                 source_current_object=current_object.name,
                 source_material_id_attribute=material_point_id_attribute,
                 source_uv_layer=layer0.name, millimeters_per_blender_unit=scale,
                 correspondence="same ordered physical IDs and polygons",
                 uv_chart_contract="reference per-loop UV; reference triangulation reused",
                 deformation_origin="caller_supplied_current_mesh; not a dressing solve",
                 orientation_validation=("independent_current_directors" if current_face_directors_world is not None
                                         else "topology_winding_and_positive_object_transform_only"))
    return GarmentMeshPair(
        reference_positions_mm=_world_positions(reference_object, mesh0, scale),
        current_positions_mm=_world_positions(current_object, mesh1, scale),
        triangles=triangles, material_uv=vertex_uv, material_point_ids=ids0,
        face_material_uv=chart_uv, current_face_directors=current_face_directors_world,
        metadata=default_garment_metadata(reference_version, material_version, reference_role, **extra))
