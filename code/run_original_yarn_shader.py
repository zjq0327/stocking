"""Replace only the authored skin/yarn mixture C in a copy of the source scene.

Run with Blender, for example:
  blender --background WhiteStocking_MIZI_Bl45.blend --python-exit-code 1 \
    --python stocking/code/run_original_yarn_shader.py -- --extract

Extraction freezes the evaluated geometry at frame 996. Rendering consumes
body_base-coverage.npz and toe_stocking-coverage.npz (coverage and mapped
float32, top-down arrays at the original camera's full pixel resolution). The
original Alpha network, normal, closures, lighting, masks, and compositor
remain unchanged. The delivered geometric C is specific to this camera.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OBJECT_NAMES = ('body_base', 'toe_stocking')
SOURCE_SHA = 'bec4876808f42a44b231f370404a6a85425c38301af171975ef7108b11d9671b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return [value(x) for x in v]
    except TypeError:
        return str(v)


def node_snapshot(tree):
    nodes = {}
    for n in tree.nodes:
        item = {'type': n.bl_idname, 'mute': n.mute, 'inputs': []}
        for prop in ('operation', 'blend_type', 'data_type', 'use_clamp',
                     'clamp_factor', 'clamp_result', 'attribute_name', 'attribute_type',
                     'layer_name', 'uv_map', 'space', 'invert', 'interpolation',
                     'extension', 'projection'):
            if hasattr(n, prop):
                item[prop] = value(getattr(n, prop))
        if n.type == 'GROUP':
            item['node_tree'] = n.node_tree.name if n.node_tree else None
        if n.type == 'TEX_IMAGE':
            item['image'] = n.image.name if n.image else None
        if n.type == 'VALTORGB':
            item['ramp'] = [(e.position, list(e.color)) for e in n.color_ramp.elements]
            item['ramp_interpolation'] = n.color_ramp.interpolation
        for i, socket in enumerate(n.inputs):
            item['inputs'].append({
                'index': i, 'name': socket.name,
                'default': value(socket.default_value) if hasattr(socket, 'default_value') else None,
                'links': [(l.from_node.name, list(l.from_node.outputs).index(l.from_socket)) for l in socket.links],
            })
        nodes[n.name] = item
    return nodes


def signature(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def settings_snapshot(scene):
    camera = scene.camera
    camera_fields = ('type', 'lens', 'sensor_width', 'sensor_height', 'sensor_fit',
                     'shift_x', 'shift_y', 'clip_start', 'clip_end', 'ortho_scale')
    compositor = getattr(scene, 'compositing_node_group', None) or getattr(scene, 'node_tree', None)
    return {
        'frame': scene.frame_current,
        'unit_scale': scene.unit_settings.scale_length,
        'engine': scene.render.engine,
        'camera': {
            'name': camera.name,
            'matrix_world': value(camera.matrix_world),
            'fields': {p: value(getattr(camera.data, p)) for p in camera_fields},
            'dof': {p: value(getattr(camera.data.dof, p)) for p in ('use_dof', 'focus_distance', 'aperture_fstop')},
        },
        'lights': [{
            'name': o.name, 'type': o.data.type, 'energy': o.data.energy,
            'color': list(o.data.color), 'matrix_world': value(o.matrix_world), 'hide_render': o.hide_render,
            'node_tree': node_snapshot(o.data.node_tree) if o.data.use_nodes else None,
        } for o in scene.objects if o.type == 'LIGHT'],
        'world': node_snapshot(scene.world.node_tree) if scene.world and scene.world.use_nodes else None,
        'compositor': node_snapshot(compositor) if compositor else None,
        'use_compositing': scene.render.use_compositing,
        'film_transparent': scene.render.film_transparent,
        'view': {p: value(getattr(scene.view_settings, p)) for p in ('view_transform', 'look', 'exposure', 'gamma')},
        'pixel_aspect': [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
    }


def render_settings(scene):
    eevee = getattr(scene, 'eevee', None)
    return {
        'resolution': [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
        'eevee': {p: value(getattr(eevee, p)) for p in ('taa_render_samples', 'taa_samples', 'use_raytracing') if hasattr(eevee, p)},
        'cycles': {p: value(getattr(scene.cycles, p)) for p in ('samples', 'seed', 'use_animated_seed')},
    }


def mesh_arrays(mesh, matrix):
    mesh.calc_loop_triangles()
    positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', positions.ravel())
    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', triangles.ravel())
    polygon_index = np.empty(len(mesh.loop_triangles), dtype=np.int32)
    mesh.loop_triangles.foreach_get('polygon_index', polygon_index)
    polygon_material = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get('material_index', polygon_material)
    normals = np.empty_like(positions)
    mesh.vertices.foreach_get('normal', normals.ravel())
    world = np.asarray(matrix, dtype=np.float64)
    positions_mm = (positions.astype(np.float64) @ world[:3, :3].T + world[:3, 3]) * 1000.
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    normals_world = normals.astype(np.float64) @ normal_matrix.T
    normals_world /= np.maximum(np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-30)
    result = {
        'positions_world_mm': positions_mm,
        'positions_local_blender': positions,
        'vertex_normals_world': normals_world.astype(np.float32),
        'triangles': triangles,
        'triangle_polygon_index': polygon_index,
        'triangle_material_index': polygon_material[polygon_index],
        'vertex_id': np.arange(len(mesh.vertices), dtype=np.int32),
        'matrix_world': world,
    }
    for name, key in (('StockingMask', 'stocking_mask'), ('SkinMask', 'skin_mask')):
        a = mesh.attributes.get(name)
        if a is None or a.domain != 'POINT' or a.data_type not in ('FLOAT_COLOR', 'BYTE_COLOR'):
            raise ValueError(f'{name}: expected evaluated POINT color attribute')
        colors = np.empty((len(a.data), 4), dtype=np.float32)
        a.data.foreach_get('color', colors.ravel())
        result[key + '_rgba'] = colors
        result[key] = (colors[:, :3] @ np.array([.2126, .7152, .0722], dtype=np.float32)).astype(np.float32)
    loops = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get('loops', loops.ravel())
    for i, uv in enumerate(mesh.uv_layers):
        values = np.empty((len(uv.data), 2), dtype=np.float32)
        uv.data.foreach_get('uv', values.ravel())
        result[f'uv_layer_{i}_loop'] = values
        result[f'uv_layer_{i}_triangle'] = values[loops]
    return result


def geometry_signature(arrays):
    return {k: hashlib.sha256(arrays[k].tobytes()).hexdigest()
            for k in ('positions_local_blender', 'positions_world_mm', 'triangles',
                      'triangle_material_index', 'stocking_mask_rgba', 'skin_mask_rgba')
            if k in arrays}


def shading_geometry_signature(arrays):
    return {key: hashlib.sha256(values.tobytes()).hexdigest()
            for key, values in arrays.items()
            if key.startswith('uv_layer_') or key == 'vertex_normals_world'}


def ensure_source(source):
    source = Path(source).resolve()
    if sha(source) != SOURCE_SHA:
        raise ValueError('Original source file differs from the frozen authored input')
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        bpy.ops.wm.open_mainfile(filepath=str(source))
    bpy.context.scene.frame_set(996)
    return source


def extract(source, output):
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    output = Path(output).resolve()
    input_dir = output / 'inputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'original-yarn-shader-inputs-v1', 'source_file': str(source),
        'source_sha256': sha(source), 'blender_version': bpy.app.version_string,
        'units': 'positions_world_mm', 'settings': settings_snapshot(scene),
        'render_settings': render_settings(scene), 'objects': {},
    }
    for name in OBJECT_NAMES:
        obj = scene.objects[name]
        evaluated = obj.evaluated_get(dg)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
        try:
            arrays = mesh_arrays(mesh, evaluated.matrix_world)
            path = input_dir / (name + '.npz')
            np.savez_compressed(path, **arrays)
            manifest['objects'][name] = {
                'path': str(path), 'sha256': sha(path), 'vertices': len(mesh.vertices),
                'triangles': len(mesh.loop_triangles), 'geometry_signatures': geometry_signature(arrays),
                'materials': [m.name if m else None for m in mesh.materials],
                'uv_layers': [uv.name for uv in mesh.uv_layers],
                'attributes': [{'name': a.name, 'domain': a.domain, 'type': a.data_type} for a in mesh.attributes],
            }
        finally:
            evaluated.to_mesh_clear()
        print('EXTRACTED_C_INPUT', name, manifest['objects'][name]['vertices'], flush=True)
    if sha(source) != manifest['source_sha256']:
        raise AssertionError('Source file changed during extraction')
    json_write(input_dir / 'manifest.json', manifest)
    print('C_INPUT_EXTRACTION_COMPLETE', input_dir / 'manifest.json', flush=True)


def load_coverage(path, shape):
    with np.load(path, allow_pickle=False) as data:
        c = np.asarray(data['coverage'], dtype=np.float32)
        mapped = np.asarray(data['mapped'], dtype=np.float32)
    if c.shape != shape or mapped.shape != shape:
        raise ValueError(f'{path}: expected top-down camera arrays with shape {shape}')
    if not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 1):
        raise ValueError('Coverage must be finite and within [0, 1]')
    if not np.isfinite(mapped).all() or np.any(mapped < 0) or np.any(mapped > 1):
        raise ValueError('mapped must be a finite support fraction within [0, 1]')
    return c, mapped


def coverage_image(name, values, path):
    height, width = values.shape
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = values[::-1, :, None]
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_scene = bpy.data.scenes.new(name + ' float32 save settings')
    save_scene.render.image_settings.file_format = 'OPEN_EXR'
    save_scene.render.image_settings.color_mode = 'RGBA'
    save_scene.render.image_settings.color_depth = '32'
    image.save_render(str(path), scene=save_scene)
    bpy.data.scenes.remove(save_scene)
    bpy.data.images.remove(image)
    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = name
    image.colorspace_settings.name = 'Non-Color'
    pixels = np.empty(height * width * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    actual = pixels.reshape(height, width, 4)[::-1, :, 0]
    if not np.array_equal(actual, values):
        raise AssertionError('Float32 EXR changed numeric C or mapped values')
    image.pack()
    return image


def image_values(image):
    width, height = image.size
    pixels = np.empty(height * width * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(height, width, 4)[::-1, :, 0].copy()


def render_footprint(c, mapped, percentage):
    if percentage == 100:
        return c, mapped.astype(np.float32)
    if percentage != 50 or c.shape[0] % 2 or c.shape[1] % 2:
        raise ValueError('Camera-specific map previews support exactly 50% or 100% resolution')
    height, width = c.shape
    mask = mapped.astype(np.float32)
    averaged_mask = mask.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    weighted_c = (c * mask).reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    filtered_c = np.divide(weighted_c, averaged_mask, out=np.zeros_like(weighted_c), where=averaged_mask > 0)
    return filtered_c, averaged_mask


def geometric_material(original, name='fixture', c_image=None, mapped_image=None):
    material = original.copy()
    material.name = 'StockingWhite_GeometricC_' + name
    nodes, links = material.node_tree.nodes, material.node_tree.links
    coordinates = nodes.new('ShaderNodeTexCoord')
    coordinates.name = 'Geometric C: frozen camera Window coordinates'
    c = nodes.new('ShaderNodeTexImage')
    c.name = 'Geometric C: projected yarn coverage'
    c.image = c_image
    c.interpolation = 'Closest'
    c.extension = 'EXTEND'
    mapped = nodes.new('ShaderNodeTexImage')
    mapped.name = 'Geometric C: material domain support'
    mapped.image = mapped_image
    mapped.interpolation = 'Closest'
    mapped.extension = 'EXTEND'
    links.new(coordinates.outputs['Window'], c.inputs['Vector'])
    links.new(coordinates.outputs['Window'], mapped.inputs['Vector'])
    mask = nodes.new('ShaderNodeMath')
    mask.name = 'Geometric C: preserve SkinMask'
    mask.operation = 'SUBTRACT'
    mask.use_clamp = True
    links.new(c.outputs[0], mask.inputs[0])
    links.new(nodes['Math.016'].outputs[0], mask.inputs[1])
    fallback = nodes.new('ShaderNodeMix')
    fallback.name = 'Geometric C: preserve unmapped authored domain'
    fallback.data_type = 'FLOAT'
    fallback.clamp_factor = True
    links.new(mapped.outputs[0], fallback.inputs[0])
    links.new(nodes['Math.011'].outputs[0], fallback.inputs[2])
    links.new(mask.outputs[0], fallback.inputs[3])
    for link in list(nodes['Mix.001'].inputs[2].links):
        links.remove(link)
    links.new(fallback.outputs[0], nodes['Mix.001'].inputs[2])
    coordinates.location = (-1500, 1100)
    c.location = (-1200, 1200)
    mapped.location = (-1200, 1000)
    mask.location = (-900, 1200)
    fallback.location = (-650, 1100)
    for n in (coordinates, c, mapped, mask, fallback):
        n.label = n.name
    material['coverage_definition'] = 'camera-dependent opaque yarn projected union fraction'
    material['coverage_scope'] = 'frozen CameraMain frame 996 pixel field; authored fallback outside mapped domain'
    material['alpha_policy'] = 'original authored alpha graph retained; C only changes skin/yarn mixture'
    return material


def audit_material(original_snapshot, original, modified):
    if node_snapshot(original.node_tree) != original_snapshot:
        raise AssertionError('Original material was modified')
    actual = node_snapshot(modified.node_tree)
    changed = []
    for name, old in original_snapshot.items():
        new = actual[name]
        if name == 'Mix.001':
            permitted = copy.deepcopy(new)
            permitted['inputs'][2]['links'] = old['inputs'][2]['links']
            if permitted != old:
                raise AssertionError('Authored Mix.001 changed beyond input A link')
            if new != old:
                changed.append('Mix.001.inputs[2].links')
        elif new != old:
            raise AssertionError(f'Authored node changed: {name}')
    expected_new = {
        'Geometric C: projected yarn coverage', 'Geometric C: material domain support',
        'Geometric C: preserve SkinMask', 'Geometric C: preserve unmapped authored domain',
        'Geometric C: frozen camera Window coordinates',
    }
    if set(actual) - set(original_snapshot) != expected_new:
        raise AssertionError('Unexpected added integration nodes')
    if actual['Mix.001']['inputs'][2]['links'] != [('Geometric C: preserve unmapped authored domain', 0)]:
        raise AssertionError('Geometric C is not connected to the intended mixture input')
    for node in ('Geometric C: projected yarn coverage', 'Geometric C: material domain support'):
        if actual[node]['inputs'][0]['links'] != [('Geometric C: frozen camera Window coordinates', 5)]:
            raise AssertionError('Coverage texture is not using the frozen Window coordinates')
    return {
        'passed': True, 'changed_original_inputs': changed,
        'new_nodes': sorted(expected_new), 'original_material_unchanged': True,
        'original_bsdf_inputs_unchanged': True, 'original_alpha_graph_unchanged': True,
        'skin_mask_unchanged': True, 'stocking_mask_unchanged': True,
        'leg_transparent_unchanged': True,
    }


def render(source, output, render_dir, report, percentage, samples):
    start = time.perf_counter()
    output, render_dir = Path(output).resolve(), Path(render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output / 'inputs' / 'manifest.json'
    inputs = json.loads(inputs_path.read_text(encoding='utf-8'))
    if inputs['source_sha256'] != sha(source):
        raise AssertionError('Extracted input source differs from render source')
    camera_manifest_path = output / 'camera-coverage-manifest.json'
    camera_manifest = json.loads(camera_manifest_path.read_text(encoding='utf-8'))
    camera_manifest_sha = sha(camera_manifest_path)
    coverage_verification_path = ROOT / 'build-support/yarn-shader-integration-v1/camera-coverage-verification.json'
    coverage_verification = json.loads(coverage_verification_path.read_text(encoding='utf-8'))
    if not coverage_verification.get('passed'):
        raise AssertionError('Independent camera coverage verification did not pass')
    if coverage_verification['coverage_manifest_sha256'] != camera_manifest_sha:
        raise AssertionError('Coverage verification belongs to a different camera manifest')
    mapping_path = output / camera_manifest['mapping_file']
    if sha(mapping_path) != camera_manifest['mapping_sha256'] or coverage_verification['mapping_manifest_sha256'] != sha(mapping_path):
        raise AssertionError('Coverage data/verification belongs to a different source mapping')
    verification_sha = sha(coverage_verification_path)
    scene = bpy.context.scene
    immutable_settings = settings_snapshot(scene)
    original_render_settings = render_settings(scene)
    original = bpy.data.materials['StockingWhite']
    original_snapshot = node_snapshot(original.node_tree)
    dg = bpy.context.evaluated_depsgraph_get()
    frozen = {}
    modified_materials = {}
    coverage_records = {}
    # Create every evaluated mesh before hiding source objects, because the
    # toe object evaluates Shrinkwrap against body_base.
    for name in OBJECT_NAMES:
        obj = scene.objects[name]
        evaluated = obj.evaluated_get(dg)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=dg)
        mesh.name = name + '_FrozenCGeometry'
        arrays = mesh_arrays(mesh, evaluated.matrix_world)
        if geometry_signature(arrays) != inputs['objects'][name]['geometry_signatures']:
            raise AssertionError('Frozen evaluated geometry differs from extracted input: ' + name)
        original_mesh_path = Path(inputs['objects'][name]['path'])
        if sha(original_mesh_path) != inputs['objects'][name]['sha256']:
            raise AssertionError('Extracted source mesh file changed: ' + name)
        with np.load(original_mesh_path, allow_pickle=False) as source_mesh:
            source_shading = {key: source_mesh[key] for key in source_mesh.files
                              if key.startswith('uv_layer_') or key == 'vertex_normals_world'}
        if shading_geometry_signature(arrays) != shading_geometry_signature(source_shading):
            raise AssertionError('Frozen normals/UV coordinates differ from source evaluation')
        path = output / (name + '-coverage.npz')
        camera_record = next(item for item in camera_manifest['objects'] if item['object'] == name)
        verified_record = next(item for item in coverage_verification['objects'] if item['object'] == name)
        if sha(path) != camera_record['sha256'] or sha(path) != verified_record['sha256'] or not verified_record['passed']:
            raise AssertionError('Rendered coverage NPZ does not match independently verified data')
        shape = (scene.render.resolution_y, scene.render.resolution_x)
        c, mapped = load_coverage(path, shape)
        filtered_c, filtered_mapped = render_footprint(c, mapped, percentage)
        full_c_image = coverage_image(name + ' full camera geometric C', c,
                                      output / 'camera-c-maps' / (name + '-C.exr'))
        full_mapped_image = coverage_image(name + ' full camera mapped support', mapped.astype(np.float32),
                                           output / 'camera-c-maps' / (name + '-mapped.exr'))
        if percentage == 100:
            c_image, mapped_image = full_c_image, full_mapped_image
        else:
            c_image = coverage_image(name + ' preview camera geometric C', filtered_c,
                                     output / 'camera-c-maps' / (name + '-C-preview.exr'))
            mapped_image = coverage_image(name + ' preview camera mapped support', filtered_mapped,
                                          output / 'camera-c-maps' / (name + '-mapped-preview.exr'))
        modified = geometric_material(original, name, c_image, mapped_image)
        modified_materials[name] = modified
        audit_material(original_snapshot, original, modified)
        # Keep per-object render visibility and light-linking settings while
        # clearing deformation: mesh is already the evaluated frame-996 result.
        new_obj = obj.copy()
        new_obj.name = name + '_GeometricCPreview'
        new_obj.data = mesh
        new_obj.animation_data_clear()
        new_obj.modifiers.clear()
        for constraint in list(new_obj.constraints):
            new_obj.constraints.remove(constraint)
        new_obj.parent = None
        scene.collection.objects.link(new_obj)
        new_obj.matrix_world = evaluated.matrix_world.copy()
        frozen[name] = new_obj
        coverage_records[name] = {
            'input_file': str(path), 'input_sha256': sha(path), 'vertices': len(mesh.vertices),
            'camera_pixel_shape': list(c.shape), 'render_texture_pixel_shape': list(filtered_c.shape),
            'mapped_pixels': int(np.count_nonzero(mapped)),
            'fully_mapped_pixels': int(np.count_nonzero(mapped == 1)),
            'mapped_sample_pixel_equivalent': float(mapped.sum(dtype=np.float64)),
            'mapped_fraction': float(mapped.mean()),
            'mapped_mean_c': float((c * mapped).sum(dtype=np.float64) / mapped.sum(dtype=np.float64)) if np.any(mapped) else None,
            'coverage_sha256': hashlib.sha256(c.tobytes()).hexdigest(),
            'mapped_sha256': hashlib.sha256(mapped.tobytes()).hexdigest(),
            'render_coverage_sha256': hashlib.sha256(filtered_c.tobytes()).hexdigest(),
            'render_mapped_sha256': hashlib.sha256(filtered_mapped.tobytes()).hexdigest(),
            'full_c_image': full_c_image.name, 'full_mapped_image': full_mapped_image.name,
            'render_c_image': c_image.name, 'render_mapped_image': mapped_image.name,
            'material': modified.name,
            'geometry_signatures': geometry_signature(arrays), 'exr_readback_exact': True,
            'shading_geometry_signatures': shading_geometry_signature(arrays),
            'filter': 'native pixel field' if percentage == 100 else '2x2 box filter; C averaged conditionally on mapped support',
        }
    for name in OBJECT_NAMES:
        scene.objects[name].hide_render = True
    scene.render.resolution_percentage = percentage
    eevee = getattr(scene, 'eevee', None)
    if hasattr(eevee, 'taa_render_samples'):
        eevee.taa_render_samples = samples
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.use_file_extension = True
    effective_render = render_settings(scene)
    variants = []
    for label in ('01_original', '02_geometric-C'):
        for name, obj in frozen.items():
            # Baseline is the actual unmodified animated source geometry.
            # Only the replacement variant renders the exact evaluated copy.
            scene.objects[name].hide_render = label != '01_original'
            obj.hide_render = label == '01_original'
            material = original if label == '01_original' else modified_materials[name]
            for slot in obj.material_slots:
                if slot.material == original or slot.material in modified_materials.values():
                    slot.material = material
        if settings_snapshot(scene) != immutable_settings:
            raise AssertionError('Camera/lighting/world/compositor changed between variants')
        if render_settings(scene) != effective_render:
            raise AssertionError('Render settings changed between comparison variants')
        scene.render.filepath = str(render_dir / (label + '.png'))
        print('ORIGINAL_SCENE_C_RENDER', label, effective_render, flush=True)
        tick = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        variants.append({'variant': label, 'path': scene.render.filepath,
                         'geometry': 'original animated source' if label == '01_original' else 'frozen evaluated source mesh',
                         'sha256': sha(scene.render.filepath), 'seconds': time.perf_counter() - tick})
    if settings_snapshot(scene) != immutable_settings:
        raise AssertionError('Scene settings changed during renders')
    material_check = {name: audit_material(original_snapshot, original, mat)
                      for name, mat in modified_materials.items()}
    scene['geometric_C_integration'] = 'Only Mix.001 A branch replaced; original Alpha and StockingMask retained'
    scene['source_blend_sha256'] = sha(source)
    scene['geometric_C_scope'] = 'fixed CameraMain frame 996 pixel field; authored fallback outside mapped domain'
    bpy.ops.file.pack_all()
    blend_path = output / 'original-geometric-C.blend'
    if blend_path.exists():
        raise FileExistsError('Refusing to overwrite an existing saved preview scene')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
    # Check the actual portable saved scene, rather than only in-memory nodes.
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    reopened = bpy.context.scene
    if settings_snapshot(reopened) != immutable_settings:
        raise AssertionError('Saved scene changed camera/lighting/world/compositor')
    reopened_check = {name: audit_material(original_snapshot, bpy.data.materials['StockingWhite'],
                                          bpy.data.materials[record['material']])
                      for name, record in coverage_records.items()}
    for name, record in coverage_records.items():
        obj = reopened.objects[name + '_GeometricCPreview']
        arrays = mesh_arrays(obj.data, obj.matrix_world)
        if geometry_signature(arrays) != record['geometry_signatures']:
            raise AssertionError('Saved geometry differs from extracted mesh')
        if shading_geometry_signature(arrays) != record['shading_geometry_signatures']:
            raise AssertionError('Saved normals/UV coordinates changed')
        for key, digest_key in (('full_c_image', 'coverage_sha256'),
                                ('full_mapped_image', 'mapped_sha256'),
                                ('render_c_image', 'render_coverage_sha256'),
                                ('render_mapped_image', 'render_mapped_sha256')):
            image = bpy.data.images[record[key]]
            if not image.packed_file or image.colorspace_settings.name != 'Non-Color':
                raise AssertionError('Saved numeric coverage texture must be packed Non-Color')
            values = image_values(image)
            if hashlib.sha256(values.tobytes()).hexdigest() != record[digest_key]:
                raise AssertionError('Saved C texture changed')
        if not reopened.objects[name].hide_render:
            raise AssertionError('Animated original mesh must be hidden in saved comparison')
        if any(slot.material and slot.material.name == 'StockingWhite' for slot in obj.material_slots):
            raise AssertionError('Saved scene reverted to original coverage material')
    if sha(source) != SOURCE_SHA:
        raise AssertionError('Original source file was modified')
    if sha(camera_manifest_path) != camera_manifest_sha or sha(coverage_verification_path) != verification_sha:
        raise AssertionError('Coverage manifest/verification changed during rendering')
    for record in coverage_records.values():
        if sha(record['input_file']) != record['input_sha256']:
            raise AssertionError('Coverage array file changed during rendering')
    result = {
        'schema': 'original-yarn-shader-integration-verification-v1', 'passed': True,
        'source_file': str(source), 'source_sha256': sha(source), 'source_unchanged': True,
        'extraction_manifest': str(inputs_path), 'extraction_manifest_sha256': sha(inputs_path),
        'camera_coverage_manifest': str(camera_manifest_path), 'camera_coverage_manifest_sha256': camera_manifest_sha,
        'camera_coverage_verification': str(coverage_verification_path), 'camera_coverage_verification_sha256': verification_sha,
        'camera_coverage_independently_verified': True,
        'settings_unchanged': True, 'settings_signature': signature(immutable_settings),
        'original_render_settings': original_render_settings, 'comparison_render_settings': effective_render,
        'material_check': material_check, 'saved_scene_material_check': reopened_check,
        'objects': coverage_records, 'renders': variants,
        'saved_scene': str(blend_path), 'saved_scene_sha256': sha(blend_path), 'saved_scene_reopened': True,
        'geometry_preserved': True, 'numeric_c_preserved': True, 'exr_readback_exact': True,
        'normals_uv_preserved': True, 'original_unfrozen_baseline_rendered': True,
        'all_coverage_textures_packed': True, 'camera_specific_coverage': True,
        'stocking_mask_forced_white_preserved': True, 'alpha_mode': 'authored alpha unchanged',
        'scope': 'fixed CameraMain frame 996 pixel field; source authored C outside mapped domain',
        'elapsed_seconds': time.perf_counter() - start,
    }
    json_write(report, result)
    print('ORIGINAL_SCENE_C_INTEGRATION_COMPLETE', report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--extract', action='store_true')
    stage.add_argument('--render', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / 'WhiteStocking_MIZI_Bl45.blend')
    parser.add_argument('--output', type=Path, default=ROOT / 'stocking/assets/original-yarn-shader-v1')
    parser.add_argument('--render-dir', type=Path, default=ROOT / 'render/original-yarn-shader-v1')
    parser.add_argument('--report', type=Path, default=ROOT / 'build-support/yarn-shader-integration-v1/integration-verification.json')
    parser.add_argument('--percentage', type=int, default=100)
    parser.add_argument('--samples', type=int, default=512)
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    args = parser.parse_args(argv)
    if not 1 <= args.percentage <= 100 or args.samples < 1:
        parser.error('Percentage must be 1..100 and samples must be positive')
    source = ensure_source(args.source)
    if args.extract:
        extract(source, args.output)
    else:
        render(source, args.output, args.render_dir, args.report, args.percentage, args.samples)


if __name__ == '__main__':
    main()
