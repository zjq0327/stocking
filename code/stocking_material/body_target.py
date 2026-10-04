"""Extract corresponding rest and posed collision surfaces from a Blender rig.

The body's evaluated polygon corners are the topology authority. Blender may
choose different quad diagonals after posing, so both surfaces use the one
triangulation captured in the rest domain.
"""
from __future__ import annotations

import numpy as np


def _snapshot_mesh(body_object, depsgraph, mm_per_unit):
    evaluated = body_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        positions = np.asarray([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices], dtype=np.float64) * mm_per_unit
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError('Evaluated body positions must be finite and nonempty.')
        polygons = tuple(tuple(polygon.vertices) for polygon in mesh.polygons)
        mesh.calc_loop_triangles()
        triangles = np.asarray([tuple(triangle.vertices) for triangle in mesh.loop_triangles], dtype=np.int32)
        groups = tuple(group.name for group in body_object.vertex_groups)
        weights = np.zeros((len(mesh.vertices), len(groups)), dtype=np.float64)
        for vertex in mesh.vertices:
            for entry in vertex.groups:
                weights[vertex.index, entry.group] = entry.weight
        return positions, polygons, triangles, weights, groups
    finally:
        evaluated.to_mesh_clear()


def _landmarks(rig_object, suffix, mm_per_unit, *, posed):
    names = ('thigh_twist', 'thigh_stretch', 'leg_stretch', 'leg_twist', 'foot', 'toes_01')
    bones = rig_object.pose.bones if posed else rig_object.data.bones
    missing = [name + suffix for name in names if name + suffix not in bones]
    if missing:
        raise ValueError('Required deform bones are missing: ' + ', '.join(missing))

    def position(name, endpoint):
        bone = bones[name + suffix]
        point = getattr(bone, endpoint if posed else endpoint + '_local')
        return np.asarray(list(rig_object.matrix_world @ point), dtype=np.float64) * mm_per_unit

    thigh_head = position('thigh_twist', 'head')
    knee = position('thigh_stretch', 'tail')
    toe_delta = position('toes_01', 'tail') - position('toes_01', 'head')
    length = np.linalg.norm(toe_delta)
    if length < 1e-10:
        raise ValueError('Toe bone has zero length.')
    return {
        'cuff': thigh_head + 0.15 * (knee - thigh_head),
        'thighmid': position('thigh_twist', 'tail'),
        'knee': knee,
        'calfmid': position('leg_twist', 'head'),
        'ankle': position('foot', 'head'),
        'ball': position('foot', 'tail'),
        'toe_direction': toe_delta / length,
        'toe_bone_end': position('toes_01', 'tail'),
    }


def extract_body_target(body_object, rig_object, *, frame=996):
    """Return world-mm rest/posed meshes, stable triangles and limb landmarks.

    No file is saved. Frame and all temporarily changed modifier flags are
    restored, including when extraction or correspondence checks fail.
    This extracts the body's unposed domain; it is not a material reference
    or a physical rest state for the separately generated stocking.
    """
    import bpy

    if body_object.type != 'MESH' or rig_object.type != 'ARMATURE':
        raise TypeError('Expected a mesh body and an armature rig.')
    scene = bpy.context.scene
    mm_per_unit = float(scene.unit_settings.scale_length) * 1000.0
    if not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError('Scene unit scale must be positive and finite.')
    original_frame = scene.frame_current
    original_subframe = scene.frame_subframe
    modifiers = [modifier for modifier in body_object.modifiers if modifier.type == 'ARMATURE']
    if not modifiers:
        raise ValueError('Body has no armature modifier to disable for rest-domain extraction.')
    if not any(modifier.object == rig_object for modifier in modifiers):
        raise ValueError('The supplied rig does not deform the supplied body.')
    original_flags = [(modifier, modifier.show_viewport, modifier.show_render) for modifier in modifiers]
    try:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        current = _snapshot_mesh(body_object, bpy.context.evaluated_depsgraph_get(), mm_per_unit)
        current_landmarks = {label: _landmarks(rig_object, suffix, mm_per_unit, posed=True)
                             for label, suffix in (('left', '.l'), ('right', '.r'))}
        for modifier, _, _ in original_flags:
            modifier.show_viewport = False
            modifier.show_render = False
        bpy.context.view_layer.update()
        rest = _snapshot_mesh(body_object, bpy.context.evaluated_depsgraph_get(), mm_per_unit)
        if current[0].shape != rest[0].shape or current[1] != rest[1]:
            raise ValueError('Rest and posed evaluated body vertex/polygon topology differs.')
        if current[4] != rest[4] or not np.array_equal(current[3], rest[3]):
            raise ValueError('Rest and posed evaluated body group identity or weights differ.')
        weights, names = rest[3], rest[4]
        left_indices = [index for index, name in enumerate(names) if name.endswith('.l')]
        right_indices = [index for index, name in enumerate(names) if name.endswith('.r')]
        if not left_indices or not right_indices:
            raise ValueError('Body requires left and right limb vertex groups.')
        vertex_side = np.where(weights[:, left_indices].sum(axis=1) > weights[:, right_indices].sum(axis=1), 0, 1).astype(np.int32)
        landmarks = {label: {'rest': _landmarks(rig_object, suffix, mm_per_unit, posed=False),
                             'current': current_landmarks[label], 'bone_suffix': suffix}
                     for label, suffix in (('left', '.l'), ('right', '.r'))}
        return {
            'positions_mm': current[0],
            'reference_positions_mm': rest[0],
            'triangles': rest[2],
            'vertex_side': vertex_side,
            'group_weights': weights,
            'group_names': np.asarray(names),
            'landmarks_by_side': landmarks,
            'metadata': {
                'frame': int(frame), 'mm_per_blender_unit': mm_per_unit,
                'body_object': body_object.name, 'rig_object': rig_object.name,
                'body_reference_role': 'body_unposed_binding_domain',
                'triangulation': 'fixed_rest_domain',
                'evaluated_polygon_correspondence_verified': True,
                'pose_dependent_loop_triangulation_equal': bool(np.array_equal(current[2], rest[2])),
            },
        }
    finally:
        for modifier, viewport, render in original_flags:
            modifier.show_viewport = viewport
            modifier.show_render = render
        scene.frame_set(original_frame, subframe=original_subframe)
        bpy.context.view_layer.update()
