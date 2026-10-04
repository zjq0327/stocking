"""Phase averaged geometric coverage of a locally affine periodic yarn.

The material chart derivative J is applied once to q(t); the yarn normal
height and circular capsule radius retain their geometric values. Orthographic
projection onto the view plane turns each 3D capsule into a 2D capsule. All
periodic images that can hit the unit-cell samples are included using exact
inverse-projection AABB bounds. This is a macro-scale local affine average,
not an exact camera pixel query against the curved whole garment or skin.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .parameters import Parameters

_LIBRARIES = {}

def native_paths():
    source=Path(__file__).with_suffix('.cpp')
    root=source.parents[3]
    identity=hashlib.sha256(source.read_bytes()+b'periodic-coverage-v1-msvc-O2-strict').hexdigest()[:20]
    directory=root/'build-support/stocking-material/periodic-coverage-native'/identity
    return source,directory,directory/'periodic_yarn_coverage.dll'

def build_native_kernel():
    source,directory,library=native_paths()
    if library.is_file():return library
    root=Path('C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC')
    versions=sorted((p for p in root.glob('*') if (p/'bin/Hostx64/x64/cl.exe').is_file()),key=lambda p:tuple(map(int,p.name.split('.'))))
    if not versions:raise RuntimeError('No existing MSVC compiler found')
    msvc=versions[-1];kits=Path('C:/Program Files (x86)/Windows Kits/10')
    sdks=sorted((p for p in (kits/'Include').glob('*') if (p/'ucrt/stdlib.h').is_file()),key=lambda p:tuple(map(int,p.name.split('.'))))
    if not sdks:raise RuntimeError('No existing Windows SDK found')
    sdk=sdks[-1];directory.mkdir(parents=True,exist_ok=True)
    command=[str(msvc/'bin/Hostx64/x64/cl.exe'),'/nologo','/LD','/O2','/fp:strict','/EHsc','/std:c++17','/MT',
        '/I'+str(msvc/'include'),'/I'+str(sdk/'ucrt'),'/I'+str(sdk/'shared'),'/I'+str(sdk/'um'),str(source),
        '/Fo'+str(directory/'periodic_yarn_coverage.obj'),'/Fd'+str(directory/'periodic_yarn_coverage.pdb'),
        '/link','/OUT:'+str(library),'/LIBPATH:'+str(msvc/'lib/x64'),
        '/LIBPATH:'+str(kits/'Lib'/sdk.name/'ucrt/x64'),'/LIBPATH:'+str(kits/'Lib'/sdk.name/'um/x64')]
    result=subprocess.run(command,cwd=directory,capture_output=True,text=True,encoding='utf8',errors='replace')
    (directory/'build.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode or not library.is_file():raise RuntimeError('Native coverage build failed: '+str(directory/'build.log'))
    return library

def _load():
    _,_,path=native_paths()
    if not path.is_file():raise RuntimeError('Explicitly call build_native_kernel before coverage generation')
    key=str(path)
    if key not in _LIBRARIES:
        lib=ctypes.CDLL(key);d=ctypes.POINTER(ctypes.c_double);i=ctypes.POINTER(ctypes.c_int);b=ctypes.POINTER(ctypes.c_ubyte)
        lib.stocking_periodic_coverage.argtypes=[ctypes.c_int,d,d,d,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,i,d,b,b]
        lib.stocking_periodic_coverage.restype=ctypes.c_int;_LIBRARIES[key]=lib
    return _LIBRARIES[key]

def projection_data(J,normal,view):
    J=np.asarray(J,dtype=np.float64);normal=np.asarray(normal,dtype=np.float64);view=np.asarray(view,dtype=np.float64)
    if J.ndim==2:J=J[None]
    if normal.ndim==1:normal=normal[None]
    if view.ndim==1:view=view[None]
    if J.shape[1:]!=(3,2) or normal.shape!=(len(J),3) or view.shape!=(len(J),3):raise ValueError('Expected J[M,3,2], normal[M,3], view[M,3]')
    if not all(np.isfinite(a).all() for a in (J,normal,view)):raise ValueError('Nonfinite coverage geometry')
    normal=normal/np.linalg.norm(normal,axis=1)[:,None];view=view/np.linalg.norm(view,axis=1)[:,None]
    axes=np.eye(3)[np.argmin(abs(view),axis=1)]
    right=np.cross(axes,view);right/=np.linalg.norm(right,axis=1)[:,None];up=np.cross(view,right)
    P=np.stack((right,up),axis=1)
    E=np.ascontiguousarray(P@J);H=np.ascontiguousarray(np.einsum('fij,fj->fi',P,normal))
    if not np.isfinite(E).all() or not np.isfinite(H).all():raise ValueError('Zero view or normal')
    return E,H,P

def periodic_coverage_batch(J,normal,view,parameters=None,*,grid=64,nodes=192,guard=0,workers=None,return_mask=False,packed_mask_path=None):
    p=Parameters() if parameters is None else parameters;p.validate()
    if not isinstance(grid,int) or not 4<=grid<=1024 or not isinstance(nodes,int) or nodes<16:raise ValueError('Invalid grid or yarn nodes')
    E,H,P=projection_data(J,normal,view)
    if return_mask and len(E)!=1:raise ValueError('Mask readback supports exactly one state')
    t=np.linspace(0,2*np.pi,nodes+1)
    q=np.ascontiguousarray(np.column_stack((p.scale_mm*(t+p.a*np.sin(2*t)),p.h*p.scale_mm*np.cos(t),p.d*p.scale_mm*np.cos(2*t))))
    counts=np.empty(len(E),dtype=np.int32);stats=np.zeros((len(E),3),dtype=np.float64)
    mask=np.empty((grid,grid),dtype=np.uint8) if return_mask else None
    packed=None
    if packed_mask_path is not None:
        packed_mask_path=Path(packed_mask_path).resolve()
        packed_mask_path.parent.mkdir(parents=True,exist_ok=True)
        if packed_mask_path.exists():raise ValueError('Packed mask destination already exists')
        packed=np.memmap(packed_mask_path,mode='w+',dtype=np.uint8,shape=(len(E),(grid*grid+7)//8))
    dp=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    worker_count=min(16,os.cpu_count() or 1) if workers is None else int(workers)
    started=time.perf_counter()
    success=_load().stocking_periodic_coverage(len(E),dp(E),dp(H),dp(q),nodes,*p.period_mm,p.R*p.scale_mm,grid,guard,worker_count,
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),dp(stats),None if mask is None else mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        None if packed is None else packed.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    if not success or np.any(counts<0):raise ValueError('Singular projection or excessive periodic support: '+str(np.flatnonzero(counts<0).tolist()[:20]))
    coverage=counts.astype(np.float32)/np.float32(grid*grid)
    diagnostics={'schema':'stocking.periodic_yarn_coverage.v1','faces':len(E),'grid_per_axis':grid,'samples_per_face':grid*grid,'centerline_segments_per_period':nodes,
        'radius_mm':p.R*p.scale_mm,'period_mm':list(p.period_mm),'parameters':p.to_dict(),'backend':'source-addressed MSVC C++ float64 capsule-union scan',
        'native_kernel_identity':native_paths()[1].name,'elapsed_seconds':time.perf_counter()-started,'period_image_aabbs_visited':int(stats[:,0].sum()),
        'uncovered_sample_distance_tests':int(stats[:,1].sum()),'all_samples_covered_early_faces':int(stats[:,2].sum()),'support_guard_extra_periods':guard,
        'state_quantization_used':False,'deformation_applied_once':True,'periodic_source_bound':'all inverse-projection radius-expanded segment AABBs intersecting the unit cell',
        'approximation':'local affine surface with constant normal and unit-cell phase averaged C','curved_whole_garment_exact':False,'real_skin_occlusion_used':False,
        'fiber_transmission_solved':False,'camera_direction_scope':'per-face view; regenerate for changed camera or pose','minimum_absolute_projected_determinant':float(np.min(abs(np.linalg.det(E))))}
    if mask is not None:diagnostics['sample_mask']=mask;diagnostics['projected_matrix']=E[0];diagnostics['height_projection']=H[0];diagnostics['projection_basis']=P[0]
    if packed is not None:
        packed.flush()
        diagnostics['packed_phase_mask']={'file':str(packed_mask_path),'shape':[len(E),(grid*grid+7)//8],'dtype':'uint8','bitorder':'little',
            'flatten_order':'flat=y_wale*grid+x_course; byte=flat//8, bit=flat%8','samples':'q=((x+.5)*period_course/grid,(y+.5)*period_wale/grid)',
            'row_direction':'+material q wale','face_order':'same as pair.triangles','grid':grid,'bytes':packed.nbytes}
    return coverage,diagnostics

def face_geometry(pair,camera_position_mm,parameters=None):
    from .authored_yarn import build_authored_phase
    from .garment_data import triangle_edges
    phase=build_authored_phase(pair,parameters)
    xyz=pair.current_positions_mm[pair.triangles]
    J=triangle_edges(xyz)@np.linalg.inv(triangle_edges(phase.face_q_mm))
    normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    view=np.asarray(camera_position_mm,dtype=np.float64)-xyz.mean(axis=1)
    return J,normal,view,phase

def compute_face_coverage(pair,camera_position_mm,*,grid=64,nodes=192,guard=0,workers=None):
    J,n,v,phase=face_geometry(pair,camera_position_mm)
    coverage,diagnostics=periodic_coverage_batch(J,n,v,grid=grid,nodes=nodes,guard=guard,workers=workers)
    diagnostics.update(course_repeat_count=phase.course_repeat_count,material_chart=phase.metadata['material_chart'])
    return coverage,diagnostics
