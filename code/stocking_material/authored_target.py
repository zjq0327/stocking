"""Extract an authored stocking target, retaining source-face provenance.

The unposed sample is the same worn design before armature deformation, NOT
an unworn or stress-free garment X0. Mask cuts are explicit geometry choices,
not an inversion of the optical material and not a material correspondence.
"""
from __future__ import annotations

import numpy as np


def snapshot_authored_object(obj, depsgraph, mm_per_unit):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        triangles = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
        loop_ids = np.array([tuple(t.loops) for t in mesh.loop_triangles], dtype=np.int32)
        positions = np.array([tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices], dtype=float) * mm_per_unit
        attributes = {}
        for name in ('SkinMask', 'StockingMask', 'PreSkinHeight'):
            attr = mesh.attributes.get(name)
            if attr is None:
                continue
            if attr.domain != 'POINT':
                raise ValueError(f'{obj.name}/{name}: expected POINT attribute')
            components = 4 if attr.data_type in ('FLOAT_COLOR', 'BYTE_COLOR') else 1
            data = np.empty(len(attr.data) * components, dtype=float)
            attr.data.foreach_get('color' if components == 4 else 'value', data)
            if components == 4:
                # The source masks are achromatic. Refuse an ambiguous color mask.
                data = data.reshape(-1, 4)[:, :3]
                if np.max(np.ptp(data, axis=1)) > 1e-4:
                    raise ValueError(f'{obj.name}/{name}: non-grayscale mask needs a policy')
                data = data.mean(axis=1)
            attributes[name] = data
        if mesh.uv_layers.active is None:
            raise ValueError(f'{obj.name}: source UV is absent')
        uv = np.array([tuple(entry.uv) for entry in mesh.uv_layers.active.data], dtype=float)
        return {
            'positions_mm': positions, 'triangles': triangles,
            'polygons': tuple(tuple(p.vertices) for p in mesh.polygons),
            'face_source_polygon': np.array([t.polygon_index for t in mesh.loop_triangles], dtype=np.int32),
            'face_uv': uv[loop_ids],
            'material_indices': np.array([t.material_index for t in mesh.loop_triangles], dtype=np.int32),
            'material_names': [m.name if m else '' for m in mesh.materials],
            'attributes': attributes, 'uv_name': mesh.uv_layers.active.name,
        }
    finally:
        evaluated.to_mesh_clear()


def extract_authored_snapshots(scene, names=('body_base', 'toe_stocking')):
    import bpy
    mm_per_unit = float(scene.unit_settings.scale_length) * 1000
    if not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError('Invalid scene scale')
    objects = [scene.objects[name] for name in names]
    # Capture exactly the evaluated geometry visible in the source scene.
    for obj in objects:
        if any(m.show_render != m.show_viewport for m in obj.modifiers):
            raise ValueError(f'{obj.name}: viewport/render modifier flags differ')
        if any(m.type == 'SUBSURF' and m.levels != m.render_levels for m in obj.modifiers):
            raise ValueError(f'{obj.name}: viewport/render subdivision differs')
    current = [snapshot_authored_object(obj, bpy.context.evaluated_depsgraph_get(), mm_per_unit) for obj in objects]
    flags = [(m, m.show_viewport) for obj in scene.objects for m in obj.modifiers if m.type == 'ARMATURE']
    try:
        for modifier, _ in flags:
            modifier.show_viewport = False
        bpy.context.view_layer.update()
        unposed = [snapshot_authored_object(obj, bpy.context.evaluated_depsgraph_get(), mm_per_unit) for obj in objects]
    finally:
        for modifier, visible in flags:
            modifier.show_viewport = visible
        bpy.context.view_layer.update()
    results = []
    for name, a, b in zip(names, unposed, current):
        if a['positions_mm'].shape != b['positions_mm'].shape or a['polygons'] != b['polygons']:
            raise ValueError(f'{name}: posing changed evaluated polygon topology')
        if a['attributes'].keys() != b['attributes'].keys() or any(
                not np.allclose(a['attributes'][key], b['attributes'][key], atol=1e-5, rtol=0)
                for key in a['attributes']):
            raise ValueError(f'{name}: source selection attributes changed with pose')
        a['unposed_positions_mm'] = a.pop('positions_mm')
        a['positions_mm'] = b['positions_mm']
        a['name'] = name
        results.append(a)
    return results


def clip_source(source, inequalities, *, material_names=None):
    """Cut triangle-linear fields at zero, preserving exact parent barycentrics.

    Each inequality is (name, scalar[N]); keep scalar >= 0. Shared edge cuts
    reuse vertices by their original vertex weights; UV remains per corner.
    """
    triangles = source['triangles']
    fields = [np.asarray(value, dtype=float) for _, value in inequalities]
    if any(value.shape != (len(source['positions_mm']),) or not np.isfinite(value).all() for value in fields):
        raise ValueError('Clipping fields must be finite source-vertex values')
    lookup, positions, unposed, parents, weights = {}, [], [], [], []
    faces, face_uv, parent_faces, corner_bary, materials = [], [], [], [], []
    identity = np.eye(3)
    for face_index, tri in enumerate(triangles):
        material = int(source['material_indices'][face_index])
        if material_names is not None and source['material_names'][material] not in material_names:
            continue
        polygon = [row.copy() for row in identity]
        for field in fields:
            if not polygon:
                break
            samples = field[tri]
            clipped = []
            prev = polygon[-1]
            prev_value = float(prev @ samples)
            for point in polygon:
                value = float(point @ samples)
                if (prev_value >= 0) != (value >= 0):
                    t = prev_value / (prev_value - value)
                    clipped.append(prev + t * (point - prev))
                if value >= 0:
                    clipped.append(point)
                prev, prev_value = point, value
            polygon = clipped
        if len(polygon) < 3:
            continue
        for index in range(1, len(polygon) - 1):
            bary = np.array([polygon[0], polygon[index], polygon[index + 1]])
            xyz = bary @ source['unposed_positions_mm'][tri]
            if np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) <= 1e-12:
                continue
            ids = []
            for point in bary:
                key = tuple(sorted((int(v), round(float(w), 13)) for v, w in zip(tri, point) if abs(w) > 1e-13))
                if key not in lookup:
                    lookup[key] = len(positions)
                    positions.append(point @ source['positions_mm'][tri])
                    unposed.append(point @ source['unposed_positions_mm'][tri])
                    parents.append(face_index)
                    weights.append(point)
                ids.append(lookup[key])
            if len(set(ids)) != 3:
                continue
            faces.append(ids)
            face_uv.append(bary @ source['face_uv'][face_index])
            parent_faces.append(face_index)
            corner_bary.append(bary)
            materials.append(material)
    return {'positions_mm': np.asarray(positions), 'unposed_positions_mm': np.asarray(unposed),
            'triangles': np.asarray(faces, dtype=np.int32), 'face_uv': np.asarray(face_uv),
            'vertex_source_triangle': np.asarray(parents, dtype=np.int32),
            'vertex_source_barycentric': np.asarray(weights),
            'face_source_triangle': np.asarray(parent_faces, dtype=np.int32),
            'face_source_barycentric': np.asarray(corner_bary),
            'face_material_index': np.asarray(materials, dtype=np.int32)}


def mesh_topology(positions, triangles):
    """Exact index topology plus geometric area checks; no weld by proximity."""
    edges = {}
    parent = list(range(len(positions)))
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for f, triangle in enumerate(triangles):
        a, b, c = map(int, triangle)
        for x, y in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(x, y), max(x, y)), []).append((x, y, f))
            parent[find(y)] = find(x)
    boundary = [items[0][:2] for items in edges.values() if len(items) == 1]
    next_vertex = {}
    for a, b in boundary:
        next_vertex.setdefault(a, []).append(b)
    loops, visited = [], set()
    for a, b in boundary:
        if (a, b) in visited:
            continue
        loop, start, current, closed = [], a, a, False
        for _ in range(len(boundary) + 1):
            loop.append(current)
            candidates = [v for v in next_vertex.get(current, []) if (current, v) not in visited]
            if len(candidates) != 1:
                break
            following = candidates[0]
            visited.add((current, following))
            current = following
            if current == start:
                closed = True
                break
        loops.append({'vertices': loop, 'closed': closed})
    p = positions[triangles]
    twice_area = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
    used = np.unique(triangles)
    components = {}
    for v in used:
        components.setdefault(find(int(v)), []).append(int(v))
    labels = np.full(len(positions), -1, dtype=np.int32)
    component_report = []
    for i, group in enumerate(sorted(components.values(), key=len, reverse=True)):
        labels[group] = i
        xyz = positions[group]
        component_report.append({'vertices': len(group), 'bounds_mm': [xyz.min(0).tolist(), xyz.max(0).tolist()]})
    report = {'vertices': len(positions), 'triangles': len(triangles), 'components': component_report,
              'boundary_edges': len(boundary), 'boundary_loops': len(loops),
              'closed_boundary_loops': sum(loop['closed'] for loop in loops),
              'nonmanifold_edges': sum(len(items) > 2 for items in edges.values()),
              'inconsistent_winding_edges': sum(len(items) == 2 and items[0][:2] == items[1][:2] for items in edges.values()),
              'duplicate_faces': len(triangles) - len(np.unique(np.sort(triangles, axis=1), axis=0)),
              'unused_vertices': len(positions) - len(used),
              'degenerate_faces': int(np.sum(twice_area <= 1e-12)),
              'minimum_triangle_area_mm2': float(twice_area.min()/2),
              'area_mm2': float(twice_area.sum()/2)}
    return report, loops, labels


def join_authored_patches(body, foot, *, maximum_connector_width_mm=1.0):
    """Join matching ankle loops with an explicitly recorded narrow strip.

    Source vertices are never moved. Pairing is determined in the unposed
    design, carried into the current pose, and checked in both configurations.
    This strip is a target-surface construction, not a physical knitting seam.
    """
    offset = len(body['positions_mm'])
    result = {}
    for key in ('positions_mm', 'unposed_positions_mm', 'vertex_source_triangle',
                'vertex_source_barycentric', 'face_source_triangle', 'face_source_barycentric',
                'face_uv', 'face_material_index'):
        result[key] = np.concatenate((body[key], foot[key]))
    result['triangles'] = np.concatenate((body['triangles'], foot['triangles'] + offset))
    result['vertex_source_object'] = np.concatenate((np.zeros(offset, dtype=np.int32),
                                                    np.ones(len(foot['positions_mm']), dtype=np.int32)))
    result['face_source_object'] = np.concatenate((np.zeros(len(body['triangles']), dtype=np.int32),
                                                  np.ones(len(foot['triangles']), dtype=np.int32)))
    result['face_chart'] = result['face_source_object'].copy()
    _, body_loops, _ = mesh_topology(body['unposed_positions_mm'], body['triangles'])
    _, foot_loops, _ = mesh_topology(foot['unposed_positions_mm'], foot['triangles'])
    if len(body_loops) != 4 or len(foot_loops) != 2 or not all(x['closed'] for x in body_loops + foot_loops):
        raise ValueError('Expected two body cuff/ankle loop pairs and two foot ankle loops')
    used_body, strips, strip_uv, seam_report = set(), [], [], []
    for foot_loop in foot_loops:
        fv = np.asarray(foot_loop['vertices'], dtype=np.int32)
        candidates = []
        for index, loop in enumerate(body_loops):
            if index in used_body or len(loop['vertices']) != len(fv):
                continue
            bv = np.asarray(loop['vertices'], dtype=np.int32)
            for shift in range(len(fv)):
                matched = np.roll(fv[::-1], shift)
                distances = np.linalg.norm(body['unposed_positions_mm'][bv]-foot['unposed_positions_mm'][matched], axis=1)
                candidates.append((float(np.mean(distances**2)), index, bv, matched))
        if not candidates:
            raise ValueError('Ankle loops do not have matching vertex counts')
        _, body_index, bv, fv = min(candidates, key=lambda x: x[0])
        used_body.add(body_index)
        widths = {}
        for key in ('unposed_positions_mm', 'positions_mm'):
            distance = np.linalg.norm(body[key][bv]-foot[key][fv], axis=1)
            if np.max(distance) > maximum_connector_width_mm:
                raise ValueError(f'Ankle connector exceeds {maximum_connector_width_mm} mm in {key}')
            widths[key] = {'minimum_mm': float(distance.min()), 'maximum_mm': float(distance.max()),
                           'rms_mm': float(np.sqrt(np.mean(distance**2)))}
        start = len(strips)
        for i in range(len(bv)):
            j = (i+1) % len(bv)
            strips.extend(((int(bv[i]), int(fv[i])+offset, int(bv[j])),
                           (int(bv[j]), int(fv[i])+offset, int(fv[j])+offset)))
            a, b = i/len(bv), (i+1)/len(bv)
            strip_uv.extend((((a,0),(a,1),(b,0)), ((b,0),(a,1),(b,1))))
        seam_report.append({'body_boundary_vertices': bv.tolist(),
                            'foot_boundary_vertices': (fv+offset).tolist(),
                            'connector_face_start': len(result['triangles'])+start,
                            'connector_face_count': len(strips)-start, 'widths': widths})
    count = len(strips)
    result['triangles'] = np.concatenate((result['triangles'], np.asarray(strips, dtype=np.int32)))
    result['face_uv'] = np.concatenate((result['face_uv'], np.asarray(strip_uv)))
    result['face_source_triangle'] = np.concatenate((result['face_source_triangle'], np.full(count, -1, dtype=np.int32)))
    result['face_source_object'] = np.concatenate((result['face_source_object'], np.full(count, -1, dtype=np.int32)))
    result['face_chart'] = np.concatenate((result['face_chart'], np.repeat([2,3], [x['connector_face_count'] for x in seam_report])))
    result['face_source_barycentric'] = np.concatenate((result['face_source_barycentric'], np.zeros((count,3,3))))
    result['face_material_index'] = np.concatenate((result['face_material_index'], np.full(count, -1, dtype=np.int32)))
    topology, loops, component = mesh_topology(result['unposed_positions_mm'], result['triangles'])
    if (len(topology['components']) != 2 or topology['closed_boundary_loops'] != 2
            or topology['boundary_loops'] != 2 or topology['nonmanifold_edges']
            or topology['inconsistent_winding_edges'] or topology['degenerate_faces']):
        raise ValueError(f'Joined target failed topology checks: {topology}')
    result['component'] = component
    return result, seam_report
