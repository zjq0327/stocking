"""Export the existing scene's stocking target without changing the source file.

Run Blender -b WhiteStocking_MIZI_Bl45.blend --python this_file.
"""
import hashlib
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

code = Path(__file__).resolve().parent
root = code.parents[1]
sys.path.insert(0, str(code))
sys.pycache_prefix = str(root / 'build-support/stocking-material/pycache')
from stocking_material.authored_target import extract_authored_snapshots, clip_source, mesh_topology, join_authored_patches


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def intersections(positions, triangles, component):
    tree = BVHTree.FromPolygons([Vector(p) for p in positions], triangles.tolist(), all_triangles=True)
    sets = [set(row) for row in triangles]
    pairs = [(int(a), int(b)) for a,b in tree.overlap(tree) if a < b and sets[a].isdisjoint(sets[b])]
    same = [(a,b) for a,b in pairs if component[triangles[a,0]] == component[triangles[b,0]]]
    return {'nonadjacent_self_intersection_pairs': len(same), 'cross_leg_intersection_pairs': len(pairs)-len(same),
            'first_30_pairs': pairs[:30], 'scope':'Blender BVHTree triangle overlap, shared-vertex pairs excluded; floating-point diagnostic'}


def save_obj(path, target):
    with path.open('w', encoding='utf8') as stream:
        stream.write('# Authored worn target. Coordinates in millimetres. Not an unworn X0.\n')
        for p in target['positions_mm']:
            stream.write('v %.12g %.12g %.12g\n' % tuple(p))
        for uv in target['face_uv'].reshape(-1,2):
            stream.write('vt %.12g %.12g\n' % tuple(uv))
        for index, triangle in enumerate(target['triangles']):
            stream.write('f ' + ' '.join(f'{int(v)+1}/{3*index+i+1}' for i,v in enumerate(triangle)) + '\n')


def main():
    source_path = Path(bpy.data.filepath).resolve()
    if not source_path.is_file():
        raise ValueError('Open the original model before extraction')
    source_hash = sha256(source_path)
    scene = bpy.context.scene
    output = root / 'stocking/assets/authored-target-v1'
    output.mkdir(parents=True, exist_ok=True)
    report_dir = root / 'build-support/authored-stocking-target'
    report_dir.mkdir(parents=True, exist_ok=True)
    sources = extract_authored_snapshots(scene)
    # Read the source's actual fade parameters; never cut in posed world Z.
    group = bpy.data.node_groups['LegTransparent']
    fade_start = float(group.nodes['Math.001'].inputs[1].default_value)
    fade_scale = float(group.nodes['Map Range'].inputs['To Max'].default_value)
    if fade_scale <= 0 or group.nodes['Math.001'].operation != 'SUBTRACT':
        raise ValueError('Unsupported upper fade graph')
    fade_end = fade_start + 1/fade_scale
    patches = []
    for source in sources:
        if source['name'] == 'body_base':
            mask = .5-source['attributes']['SkinMask']
            mask[np.abs(mask) < 1e-5] = 0
            inequalities = [('SkinMask_mid_transition', mask),
                            ('fully_transparent_upper_end', fade_end-source['attributes']['PreSkinHeight'])]
        else:
            mask = source['attributes']['StockingMask']-.5
            mask[np.abs(mask) < 1e-5] = 0
            inequalities = [('StockingMask_mid_transition',mask)]
        patch = clip_source(source, inequalities, material_names={'StockingWhite'})
        patches.append(patch)
        np.savez_compressed(output / (source['name']+'-patch.npz'), **patch)
        source_arrays = {key: source[key] for key in ('positions_mm','unposed_positions_mm','triangles','face_uv','material_indices','face_source_polygon')}
        source_arrays.update(source['attributes'])
        np.savez_compressed(output / (source['name']+'-source.npz'), **source_arrays)
    decoration = clip_source(sources[1], [], material_names={'LineWhite'})
    np.savez_compressed(output/'toe-decoration.npz', **decoration)
    target, connectors = join_authored_patches(*patches)
    reports = {}
    for key in ('positions_mm','unposed_positions_mm'):
        topology, loops, component = mesh_topology(target[key],target['triangles'])
        reports[key] = {'topology':topology, 'intersections':intersections(target[key],target['triangles'],component),
                        'cuff_loops':[loop['vertices'] for loop in loops]}
    # Independently reconstruct every retained source vertex in both poses.
    errors = {}
    for key in ('positions_mm','unposed_positions_mm'):
        maximum = 0.
        for index, source in enumerate(sources):
            selected = target['vertex_source_object'] == index
            parents = source['triangles'][target['vertex_source_triangle'][selected]]
            reconstructed = np.einsum('ni,nij->nj',target['vertex_source_barycentric'][selected],source[key][parents])
            maximum = max(maximum,float(np.max(np.linalg.norm(reconstructed-target[key][selected],axis=1))))
        errors[key] = maximum
        if maximum > 1e-9:
            raise ValueError('Source reconstruction failed')
    target['face_region'] = np.where(target['face_source_object'] < 0, 2, target['face_source_object']).astype(np.int32)
    target['source_face_provenance_valid'] = target['face_source_object'] >= 0
    np.savez_compressed(output/'authored_target.npz', **target)
    save_obj(output/'authored_target_current_mm.obj',target)
    original = target['positions_mm']
    target['positions_mm'] = target['unposed_positions_mm']
    save_obj(output/'authored_target_unposed_mm.obj',target)
    target['positions_mm'] = original
    metadata = {'schema':'stocking.authored_target.v1', 'source_file':str(source_path), 'source_sha256':source_hash,
                'source_frame':scene.frame_current,'length_unit':'mm','mm_per_blender_unit':scene.unit_settings.scale_length*1000,
                'objects':['body_base','toe_stocking'], 'evaluated_geometry':True,
                'triangulation':'fixed unposed evaluated source polygons; current pose reuses the same vertex identities',
                'surface_role':'authored worn macro target; thickness convention not yet calibrated',
                'unposed_role':'same worn design before armature deformation; NOT garment X0 or stress-free state',
                'material_X0_correspondence_available':False,'dressing_equilibrium_solved':False,
                'source_uv_role':'original texture UV, plus separate connector charts; NOT verified knitting coordinates',
                'selection':{'body':'SkinMask <= 0.5','foot':'StockingMask >= 0.5 AND material StockingWhite',
                             'mask_snap_tolerance':1e-5,'upper':'PreSkinHeight <= fade_end; transparent fade retained below it',
                             'fade_start_BU':fade_start,'fade_end_BU':fade_end,'fade_scale':fade_scale,
                             'selection_role':'explicit geometry policy through authored transition masks, not exact shader inversion'},
                'decoration':'LineWhite saved separately; not part of the computational target shell',
                'connector_role':'added geometric transition strips at authored ankle boundary; no source vertex moved; not physical seams',
                'connectors':connectors,'validation':reports,'source_reconstruction_max_error_mm':errors,
                'query_target_extracted':True,'physical_solver_ready':False}
    metadata['source_file_unchanged'] = sha256(source_path) == source_hash
    if not metadata['source_file_unchanged']:
        raise ValueError('Source file changed during extraction')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    (report_dir/'extraction-validation.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    from stocking_material.authored_target_preview import create_preview
    create_preview(target, decoration, output, root/'render/authored-target-v1')
    print('AUTHORED_TARGET_EXPORTED',json.dumps({'output':str(output),'vertices':len(target['positions_mm']),
          'triangles':len(target['triangles']),'validation':{key: {'topology':value['topology'],'intersections':value['intersections']} for key,value in reports.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
