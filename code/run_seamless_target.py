"""Build a geometrically joined stocking target from the audited source patches."""
import json
import sys
from pathlib import Path
import numpy as np
import bpy
code=Path(__file__).resolve().parent
root=code.parents[1]
sys.path.insert(0,str(code))
sys.pycache_prefix=str(root/'build-support/stocking-material/pycache')
from stocking_material.seamless_target import make_seamless_target
from stocking_material.authored_target_preview import create_preview
from run_authored_target import sha256,save_obj,intersections


def main():
    previous=root/'stocking/assets/authored-target-v1'
    output=root/'stocking/assets/authored-target-v2'
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((previous/'metadata.json').read_text(encoding='utf8'))
    if sha256(Path(metadata['source_file']))!=metadata['source_sha256']:
        raise ValueError('Original scene changed since extraction')
    with np.load(previous/'authored_target.npz',allow_pickle=False) as archive:
        original=dict(archive)
    with np.load(previous/'toe-decoration.npz',allow_pickle=False) as archive:
        decoration=dict(archive)
    target,repair=make_seamless_target(original,metadata['connectors'],blend_width_mm=15.)
    for key,report in repair['poses'].items():
        report['intersections']=intersections(target[key],target['triangles'],target['component'])
    unposed=repair['poses']['unposed_positions_mm']
    if unposed['intersections']['nonadjacent_self_intersection_pairs'] or unposed['intersections']['cross_leg_intersection_pairs']:
        raise ValueError('Unposed welded target contains intersections')
    if max(row['seam']['maximum_dihedral_deg'] for row in repair['poses'].values())>15:
        raise ValueError('Ankle still contains an excessive normal discontinuity')
    np.savez_compressed(output/'authored_target.npz',**target)
    np.savez_compressed(output/'toe-decoration.npz',**decoration)
    for mode,key in (('current','positions_mm'),('unposed','unposed_positions_mm')):
        copy=dict(target)
        copy['positions_mm']=target[key]
        save_obj(output/f'authored_target_{mode}_mm.obj',copy)
    metadata.update(schema='stocking.authored_target.v2',parent_asset=str(previous),
                    parent_npz_sha256=sha256(previous/'authored_target.npz'),
                    surface_role='authored worn target with local ankle geometry repair; not a calibrated cloth mid-surface',
                    connector_role='v1 connector strips removed; source ankle boundaries now share vertex indices',
                    source_geometry_preserved_everywhere=False,
                    source_uv_role='original texture UV charts retained across a geometric weld; not a continuous knitting chart',
                    provenance='source face and barycentric binding plus explicitly saved face-corner geometric adjustments',
                    repair=repair,validation=repair['poses'])
    metadata.pop('connectors',None)
    metadata.pop('source_reconstruction_max_error_mm',None)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    report_dir=root/'build-support/authored-stocking-target'
    (report_dir/'seamless-repair.json').write_text(json.dumps(repair,indent=2),encoding='utf8')
    create_preview(target,decoration,output,root/'render/authored-target-v2',seamless=True)
    print('SEAMLESS_TARGET_EXPORTED',json.dumps({'output':str(output),'vertices':len(target['positions_mm']),
          'triangles':len(target['triangles']),'pose_metrics':{k:{x:v for x,v in row.items() if x not in ('topology','cuff_loops')} for k,row in repair['poses'].items()}}))


if __name__=='__main__':
    main()
