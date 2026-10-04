"""Generate camera-specific phase-averaged yarn C for both authored socks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parent
ROOT=CODE.parent.parent
sys.path.insert(0,str(CODE))
from stocking_material.asset_io import load_garment_pair
from stocking_material.periodic_yarn_coverage import build_native_kernel,face_geometry,periodic_coverage_batch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def diff(a,b):
    d=abs(a.astype(float)-b.astype(float))
    return {'mean_absolute':float(d.mean()),'p95_absolute':float(np.quantile(d,.95)),'maximum_absolute':float(d.max()),'different_faces':int(np.count_nonzero(d))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'stocking/assets/yarn-shader-integration-v1')
    parser.add_argument('--camera-audit',type=Path,default=ROOT/'build-support/yarn-shader-integration-v1/original-shader-audit.json')
    parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--mask-directory',type=Path)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    manifestpath=output/'face-coverage-manifest.json'
    if manifestpath.exists():raise ValueError('Existing generated face manifest; use a new version')
    audit=json.loads(args.camera_audit.read_text(encoding='utf8'))
    camera=np.asarray(audit['camera']['matrix_world'],dtype=float)[:3,3]*1000
    library=build_native_kernel()
    manifest={'schema':'stocking.authored_whole_phase_coverage.v1','source_scene_sha256':audit['source_sha256'],'source_frame':audit['frame'],'camera_position_mm':camera.tolist(),'camera_name':audit['camera']['name'],
        'grid':args.grid,'phase_averaged':True,'exact_curved_yarn_pixel_queries':False,'local_v5_tiling_used':False,'original_material_replaced':False,'native_library_identity':library.parent.name,'sides':[]}
    start=time.perf_counter()
    for side in ('left','right'):
        pairpath=ROOT/'stocking/assets/authored-fit-v1'/side/'garment_pair.npz'
        pairhash=sha(pairpath);pair=load_garment_pair(pairpath)
        J,normal,view,phase=face_geometry(pair,camera)
        maskpath=None if args.mask_directory is None else args.mask_directory.resolve()/(side+'-phase-mask-'+str(args.grid)+'.bin')
        coverage,diagnostics=periodic_coverage_batch(J,normal,view,grid=args.grid,packed_mask_path=maskpath)
        filename=side+'-face-coverage.npz';np.savez_compressed(output/filename,coverage=coverage,J=J,normal=normal,view=view,face_centers_mm=pair.current_positions_mm[pair.triangles].mean(axis=1))
        # Stratified face identities plus most grazing states test the actual
        # distribution without hiding difficult faces behind random selection.
        nview=view/np.linalg.norm(view,axis=1)[:,None]
        grazing=np.argsort(abs(np.einsum('fi,fi->f',normal,nview)))[:16]
        selected=np.unique(np.r_[np.linspace(0,len(J)-1,128,dtype=int),grazing])
        fine,fined=periodic_coverage_batch(J[selected],normal[selected],view[selected],grid=args.grid*2)
        twice,twiced=periodic_coverage_batch(J[selected],normal[selected],view[selected],grid=args.grid*2,nodes=384)
        guard,guardd=periodic_coverage_batch(J[selected],normal[selected],view[selected],grid=args.grid,guard=1)
        convergence={'selected_material_faces':selected.tolist(),'coarse_to_fine':diff(coverage[selected],fine),'nodes192_to384_on_fine_grid':diff(fine,twice),'support_extra_period_guard':diff(coverage[selected],guard),
            'fine_diagnostics':fined,'twice_nodes_diagnostics':twiced,'guard_diagnostics':guardd}
        convergencefile=side+'-convergence.npz'
        np.savez_compressed(output/convergencefile,face_indices=selected,coverage_coarse=coverage[selected],coverage_fine=fine,coverage_fine_nodes384=twice,coverage_coarse_extra_guard=guard)
        if sha(pairpath)!=pairhash:raise ValueError('Frozen pair was modified')
        record={'side':side,'pair_file':str(pairpath),'pair_sha256':pairhash,'file':filename,'sha256':sha(output/filename),'faces':len(coverage),'diagnostics':diagnostics,'convergence':convergence,'convergence_file':convergencefile,'convergence_sha256':sha(output/convergencefile),
            'coverage_quantiles':np.quantile(coverage,[0,.01,.1,.5,.9,.99,1]).tolist(),'coverage_mean':float(coverage.mean()),'coverage_area_weighted_mean':float(np.average(coverage,weights=np.linalg.norm(np.cross(pair.current_positions_mm[pair.triangles[:,1]]-pair.current_positions_mm[pair.triangles[:,0]],pair.current_positions_mm[pair.triangles[:,2]]-pair.current_positions_mm[pair.triangles[:,0]]),axis=1))),
            'course_repeat_count':phase.course_repeat_count,'toe_seam_face_count':int(phase.toe_seam_mask.sum()),'toe_scope':'same assumed plain-knit periodic layout; synthetic closure seam not separately modeled by this C field'}
        manifest['sides'].append(record)
        print('FULL_YARN_C_SIDE',json.dumps({'side':side,'seconds':diagnostics['elapsed_seconds'],'faces':len(coverage),'mean_C':record['coverage_mean'],'convergence':convergence['coarse_to_fine'],'nodes':convergence['nodes192_to384_on_fine_grid'],'guard':convergence['support_extra_period_guard']}),flush=True)
    manifest['elapsed_seconds']=time.perf_counter()-start
    manifestpath.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    if args.mask_directory is not None:
        index={'schema':'stocking.phase_mask_index.v1','grid':args.grid,'camera_position_mm':camera.tolist(),'source_frame':audit['frame'],
            'period_mm':manifest['sides'][0]['diagnostics']['period_mm'],'bitorder':'little','sides':[]}
        for side in manifest['sides']:
            record=dict(side['diagnostics']['packed_phase_mask']);record.update(side=side['side'],pair_sha256=side['pair_sha256'],sha256=sha(record['file']))
            index['sides'].append(record)
        (args.mask_directory/'phase-mask-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    print('FULL_YARN_C_COMPLETE',str(manifestpath),flush=True)

if __name__=='__main__':main()
