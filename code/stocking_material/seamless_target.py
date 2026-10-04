"""Remove the authored ankle step with a local displacement and an index weld.

This is a recorded geometric repair of the worn target, not yarn mechanics.
The same unposed spatial weights transport the correction into the posed mesh.
"""
from __future__ import annotations
import numpy as np
from .authored_target import mesh_topology


def nearest_loop_parameters(points, loop_points):
    starts = np.asarray(loop_points, dtype=float)
    vectors = np.roll(starts, -1, axis=0)-starts
    lengths2 = np.einsum('ij,ij->i',vectors,vectors)
    if np.any(lengths2 <= 1e-20):
        raise ValueError('Zero-length ankle loop edge')
    distances, edges, fractions = [], [], []
    for first in range(0,len(points),1024):
        difference = points[first:first+1024,None,:]-starts[None,:,:]
        t = np.clip(np.einsum('nki,ki->nk',difference,vectors)/lengths2,0,1)
        offset = difference-t[:,:,None]*vectors
        d2 = np.einsum('nki,nki->nk',offset,offset)
        chosen = np.argmin(d2,axis=1)
        rows = np.arange(len(chosen))
        distances.append(np.sqrt(d2[rows,chosen]))
        edges.append(chosen)
        fractions.append(t[rows,chosen])
    return np.concatenate(distances),np.concatenate(edges),np.concatenate(fractions)


def seam_normal_report(positions,triangles,face_regions):
    p=positions[triangles]
    normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    edges={}
    for face,(a,b,c) in enumerate(triangles):
        for x,y in ((a,b),(b,c),(c,a)):
            edges.setdefault((min(int(x),int(y)),max(int(x),int(y))),[]).append(face)
    angles=[]
    seam_edges=[]
    for edge,faces in edges.items():
        if len(faces)==2 and face_regions[faces[0]]!=face_regions[faces[1]]:
            angles.append(np.degrees(np.arccos(np.clip(normal[faces[0]]@normal[faces[1]],-1,1))))
            seam_edges.append(edge)
    if not angles:
        raise ValueError('No body/foot seam edges found')
    return {'shared_seam_edges':len(angles),'median_dihedral_deg':float(np.median(angles)),
            'p95_dihedral_deg':float(np.percentile(angles,95)),
            'maximum_dihedral_deg':float(np.max(angles))},np.array(seam_edges,dtype=np.int32)


def make_seamless_target(original,connectors,*,blend_width_mm=15.):
    if not np.isfinite(blend_width_mm) or blend_width_mm<=0:
        raise ValueError('Blend width must be finite and positive')
    coordinates=('positions_mm','unposed_positions_mm')
    corrected={key:original[key].copy() for key in coordinates}
    mapping=np.arange(len(original['positions_mm']),dtype=np.int32)
    foot=np.flatnonzero(original['vertex_source_object']==1)
    distance=np.full(len(mapping),np.inf)
    influence=np.zeros(len(mapping))
    assignments=np.full(len(mapping),-1,dtype=np.int32)
    loop_parameters=[]
    for loop_index,connector in enumerate(connectors):
        body_ids=np.asarray(connector['body_boundary_vertices'],dtype=np.int32)
        foot_ids=np.asarray(connector['foot_boundary_vertices'],dtype=np.int32)
        if len(body_ids)!=len(foot_ids) or not np.all(original['vertex_source_object'][body_ids]==0) or not np.all(original['vertex_source_object'][foot_ids]==1):
            raise ValueError('Invalid source ankle pairing')
        d,edge,t=nearest_loop_parameters(original['unposed_positions_mm'][foot],original['unposed_positions_mm'][foot_ids])
        chosen=d<blend_width_mm
        selected=foot[chosen]
        if np.any(assignments[selected]>=0):
            raise ValueError('Ankle correction bands overlap')
        s=np.clip(d[chosen]/blend_width_mm,0,1)
        weight=1-(6*s**5-15*s**4+10*s**3)
        distance[selected]=d[chosen]
        influence[selected]=weight
        assignments[selected]=loop_index
        for key in coordinates:
            deltas=original[key][body_ids]-original[key][foot_ids]
            i=edge[chosen]
            interpolated=(1-t[chosen,None])*deltas[i]+t[chosen,None]*deltas[(i+1)%len(deltas)]
            corrected[key][selected]+=weight[:,None]*interpolated
            # Exact coincidence before the index weld; no tolerance-based merge.
            corrected[key][foot_ids]=original[key][body_ids]
        mapping[foot_ids]=body_ids
        loop_parameters.append({'body':body_ids,'foot':foot_ids,'affected_foot_vertices':int(chosen.sum())})
    keep_faces=original['face_source_object']>=0
    old_faces=original['triangles'][keep_faces]
    mapped_faces=mapping[old_faces]
    if np.any((mapped_faces[:,0]==mapped_faces[:,1])|(mapped_faces[:,1]==mapped_faces[:,2])|(mapped_faces[:,2]==mapped_faces[:,0])):
        raise ValueError('Welding collapsed a retained source face')
    retained=np.unique(mapped_faces)
    compact=np.full(len(mapping),-1,dtype=np.int32)
    compact[retained]=np.arange(len(retained),dtype=np.int32)
    triangles=compact[mapped_faces]
    result={'triangles':triangles,'vertex_previous_index':retained,
            'previous_vertex_to_current':compact[mapping],
            'face_previous_index':np.flatnonzero(keep_faces),
            'correction_weight':influence[retained],
            'correction_loop_index':assignments[retained]}
    for key in ('vertex_source_triangle','vertex_source_barycentric','vertex_source_object'):
        result[key]=original[key][retained]
    for key in ('face_uv','face_source_triangle','face_source_barycentric',
                'face_material_index','face_source_object','face_chart','face_region'):
        result[key]=original[key][keep_faces]
    report={'method':'shared-index ankle weld; nearest loop-segment displacement interpolated linearly around each ring with quintic distance falloff',
            'blend_width_mm':blend_width_mm,'removed_connector_triangles':int((~keep_faces).sum()),
            'welded_vertex_pairs':int(np.sum(mapping!=np.arange(len(mapping)))),
            'band_selection':'Euclidean distance to unposed foot rim; not a geodesic distance or physical material coordinate',
            'poses':{}}
    for key in coordinates:
        result[key]=corrected[key][retained]
        result['source_'+key]=original[key][retained]
        result['adjustment_'+key]=result[key]-result['source_'+key]
        # Per-corner deltas preserve the alternative source on welded foot edges.
        result['face_corner_adjustment_'+key]=result[key][triangles]-original[key][old_faces]
        topology,loops,component=mesh_topology(result[key],triangles)
        if (len(topology['components'])!=2 or topology['boundary_loops']!=2 or topology['closed_boundary_loops']!=2
                or topology['nonmanifold_edges'] or topology['inconsistent_winding_edges']
                or topology['duplicate_faces'] or topology['degenerate_faces']):
            raise ValueError(f'Welded topology invalid: {topology}')
        result['component']=component
        normals,seam_edges=seam_normal_report(result[key],triangles,result['face_region'])
        if normals['shared_seam_edges']!=184:
            raise ValueError('Expected 184 closed seam edges')
        result['welded_seam_edges']=seam_edges
        movement=np.linalg.norm(corrected[key]-original[key],axis=1)
        a=original[key][old_faces]
        b=result[key][triangles]
        na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
        nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
        cosine=np.einsum('ij,ij->i',na,nb)/(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1))
        area_ratio=np.linalg.norm(nb,axis=1)/np.linalg.norm(na,axis=1)
        if np.min(cosine)<=0:
            raise ValueError('Correction inverted a source face')
        report['poses'][key]={'topology':topology,'seam':normals,
                'maximum_source_vertex_move_mm':float(movement.max()),
                'moved_source_vertices':int(np.sum(movement>1e-10)),
                'maximum_body_vertex_move_mm':float(movement[original['vertex_source_object']==0].max()),
                'minimum_face_area_ratio':float(area_ratio.min()),'maximum_face_area_ratio':float(area_ratio.max()),
                'maximum_face_normal_change_deg':float(np.degrees(np.arccos(np.clip(cosine.min(),-1,1)))),
                'cuff_loops':[loop['vertices'] for loop in loops]}
    change=np.linalg.norm(result['face_corner_adjustment_unposed_positions_mm'],axis=-1)
    result['source_face_provenance_valid']=np.max(change,axis=1)<=1e-10
    result['source_face_binding_valid']=np.ones(len(triangles),dtype=bool)
    return result,report
