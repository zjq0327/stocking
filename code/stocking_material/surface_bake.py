"""Two-sided chart sampling with local P offsets, never legacy absolute P.

The shell frame is a right-handed orthonormal orientation only. Stretch has
already entered the yarn geometry; this module does not apply F again.
BVH intersections use Blender float32; NPZ retains the signed sampled data.
"""
from pathlib import Path
import json
import numpy as np

from .bake import make_bvh
from .export import _write_exr, _write_png, _read_exr

SCHEMA = "stocking.surface_features"
VERSION = 1
SIDES = ("front", "back")
FIELDS = ("P_offset_mm", "N_local", "T_local")


def validate_surface_frames(surface_mm, frames):
    s, b = np.asarray(surface_mm, dtype=float), np.asarray(frames, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3 or b.shape != (len(s),3,3) or not np.isfinite(s).all() or not np.isfinite(b).all():
        raise ValueError("Finite surface[N,3] and frame[N,3,3] arrays required")
    if not np.allclose(np.einsum("nji,njk->nik", b,b), np.eye(3), atol=1e-8, rtol=0) or not np.allclose(np.linalg.det(b),1,atol=1e-8,rtol=0):
        raise ValueError("Surface frames must be orthonormal and right-handed")
    return s,b


def sample_surface_points(geometry, bvh, surface_mm, frames, side, ray_offset_mm):
    if side not in SIDES:
        raise ValueError("side must be front or back")
    s,b = validate_surface_frames(surface_mm, frames)
    if isinstance(ray_offset_mm,(bool,np.bool_)) or not np.isfinite(ray_offset_mm) or ray_offset_mm <= 0:
        raise ValueError("ray_offset_mm must be finite and positive")
    sign = 1 if side == "front" else -1
    shell_n = b[:,:,2]
    origins = s+sign*ray_offset_mm*shell_n
    directions = -sign*shell_n
    indices = np.full(len(s),-1,dtype=np.int32)
    hit = np.zeros_like(s)
    for i in range(len(s)):
        point, _normal, index, _distance = bvh.ray_cast(tuple(origins[i]),tuple(directions[i]),2*float(ray_offset_mm))
        if index is not None:
            indices[i],hit[i] = index,point
    valid = indices >= 0
    result = {"valid":valid, **{k:np.zeros((len(s),3),dtype=np.float32) for k in FIELDS}}
    if valid.any():
        tri = geometry.triangles[indices[valid]]
        corners = geometry.vertices[tri]
        e0,e1 = corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]
        q = hit[valid]-corners[:,0]
        dot = lambda a,c: np.einsum("ij,ij->i",a,c)
        d00,d01,d11 = dot(e0,e0),dot(e0,e1),dot(e1,e1)
        denom = d00*d11-d01*d01
        if np.any(denom<=0):
            raise ValueError("Degenerate tube triangle at ray hit")
        w1 = (d11*dot(q,e0)-d01*dot(q,e1))/denom
        w2 = (d00*dot(q,e1)-d01*dot(q,e0))/denom
        w = np.clip(np.column_stack((1-w1-w2,w1,w2)),0,1)
        w /= w.sum(axis=1,keepdims=True)
        n = np.einsum("ij,ijk->ik",w,geometry.normals[tri])
        norm = np.linalg.norm(n,axis=1,keepdims=True)
        if np.any(norm<=1e-12):
            raise ValueError("Undefined interpolated tube normal")
        n /= norm
        t = np.einsum("ij,ijk->ik",w,geometry.tangents[tri])
        t -= dot(n,t)[:,None]*n
        norm = np.linalg.norm(t,axis=1,keepdims=True)
        if np.any(norm<=1e-12):
            raise ValueError("Undefined interpolated yarn tangent")
        t /= norm
        bt = b[valid]
        result["P_offset_mm"][valid] = np.einsum("nji,nj->ni",bt,hit[valid]-s[valid])
        result["N_local"][valid] = np.einsum("nji,nj->ni",bt,n)
        result["T_local"][valid] = np.einsum("nji,nj->ni",bt,t)
    return result


def bake_surface_geometry(geometry, chart, width, height, ray_offset_mm, progress=None):
    for v in (width,height):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v < 1:
            raise ValueError("Bake dimensions must be positive integers")
    period = np.asarray(chart.reference_period_mm,dtype=float)
    x = (np.arange(width)+.5)*period[0]/width
    y = (np.arange(height)+.5)*period[1]/height
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    s,b = chart.evaluate(xy)
    if hasattr(chart,"radius_mm") and ray_offset_mm >= chart.radius_mm:
        raise ValueError("Curved sampling rays must remain inside the near-surface radial layer")
    bvh = make_bvh(geometry)
    features = {}
    for side in SIDES:
        if progress:
            progress(f"Sampling {side}: {width} x {height}")
        data = sample_surface_points(geometry,bvh,s,b,side,ray_offset_mm)
        features[side] = {k:v.reshape((height,width)+(() if k=="valid" else (3,))) for k,v in data.items()}
    return features, xy.reshape(height,width,2), s.reshape(height,width,3), b.reshape(height,width,3,3)


def validate_surface_features(features):
    shape = None
    stats = {}
    for side in SIDES:
        values = features[side]
        valid = np.asarray(values["valid"])
        if valid.dtype != np.bool_ or valid.ndim != 2 or min(valid.shape)<1 or (shape is not None and valid.shape!=shape):
            raise ValueError("Matching bool[H,W] valid arrays required")
        shape = valid.shape
        for key in FIELDS:
            a = np.asarray(values[key])
            if a.dtype != np.float32 or a.shape != (*shape,3) or not np.isfinite(a).all() or np.any(a[~valid]!=0):
                raise ValueError(f"{side}.{key}: finite float32[H,W,3], zero at holes required")
        n,t = values["N_local"][valid],values["T_local"][valid]
        ne,te = np.abs(np.linalg.norm(n,axis=1)-1),np.abs(np.linalg.norm(t,axis=1)-1)
        ortho = np.abs(np.einsum("ij,ij->i",n,t))
        maximum = lambda a: float(np.max(a)) if len(a) else 0.0
        if max(maximum(ne),maximum(te),maximum(ortho))>2e-5:
            raise ValueError("Invalid unit or orthogonal N/T")
        stats[side] = {"hit_fraction":float(valid.mean()),"normal_unit_error":maximum(ne),"tangent_unit_error":maximum(te),"orthogonality_error":maximum(ortho)}
    return stats


def _metadata(metadata):
    required = {"schema":SCHEMA,"schema_version":VERSION,"length_unit":"mm",
                "position_semantics":"offset_from_current_shell_in_current_orthonormal_frame",
                "coordinate_space":"current_shell_local","macro_strain_already_applied":True}
    for key,value in required.items():
        if key in metadata and (type(metadata[key]) is not type(value) or metadata[key]!=value):
            raise ValueError(f"Incompatible feature metadata {key}")
    result = {**metadata,**required}
    for key in ("reference_version","material_version","geometry_state","ray_definition","chart"):
        if key not in result or not result[key]:
            raise ValueError(f"Missing feature metadata {key}")
    json.dumps(result,allow_nan=False)
    return result


def export_surface_features(features, asset_dir, metadata, *, sampling=None, preview_dir=None, write_exr=True):
    stats = validate_surface_features(features)
    meta = _metadata(metadata)
    meta["validation"] = stats
    meta["id_exported"] = False
    meta["valid_semantics"] = "first hit of the specified surface-normal ray; not arbitrary-view optical opacity"
    meta["image_color_space"] = "Non-Color; signed raw float32 RGB, EXR alpha is constant one"
    meta["array_row_order"] = "row 0 is lowest material y; column increases material x"
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True,exist_ok=True)
    arrays = {f"{side}_{key}":features[side][key] for side in SIDES for key in ("valid",*FIELDS)}
    if sampling:
        shape = features["front"]["valid"].shape
        expected_shapes = {"material_xy_mm":(*shape,2),"surface_mm":(*shape,3),"frames":(*shape,3,3)}
        for key in ("material_xy_mm","surface_mm","frames"):
            arrays[key] = np.asarray(sampling[key],dtype=np.float64)
            if arrays[key].shape != expected_shapes[key] or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid surface sampling field {key}")
        validate_surface_frames(arrays["surface_mm"].reshape(-1,3),arrays["frames"].reshape(-1,3,3))
    arrays["metadata_json"] = np.array(json.dumps(meta,ensure_ascii=False,allow_nan=False))
    path = asset_dir/"surface_features.npz"
    np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(arrays) or not all(np.array_equal(data[k],v) for k,v in arrays.items()):
            raise RuntimeError("Surface feature NPZ roundtrip failed")
    paths = {"npz":str(path)}
    if write_exr:
        for side in SIDES:
            for key in FIELDS:
                p = asset_dir/f"{side}_{key}.exr"
                _write_exr(p,features[side][key])
                if not np.array_equal(_read_exr(p)[...,:3],features[side][key]):
                    raise RuntimeError(f"Signed EXR roundtrip mismatch: {p}")
                paths[f"{side}_{key}"] = str(p)
            p = asset_dir/f"{side}_valid.exr"
            _write_exr(p,np.repeat(features[side]["valid"][...,None],3,axis=2).astype(np.float32))
            if not np.array_equal(_read_exr(p)[...,0],features[side]["valid"].astype(np.float32)):
                raise RuntimeError(f"Valid EXR roundtrip mismatch: {p}")
            paths[f"{side}_valid"] = str(p)
    (asset_dir/"surface_features.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True,exist_ok=True)
        for side in SIDES:
            valid = features[side]["valid"]
            _write_png(preview_dir/f"{side}_valid.png",np.repeat(valid[...,None],3,axis=2).astype(np.uint8)*255)
            for key in ("N_local","T_local"):
                rgb = np.clip((features[side][key]+1)*.5,0,1)
                rgb[~valid] = 0
                _write_png(preview_dir/f"{side}_{key}.png",np.rint(rgb*255).astype(np.uint8))
    return paths


def load_surface_features(asset_dir):
    with np.load(Path(asset_dir)/"surface_features.npz",allow_pickle=False) as data:
        meta = _metadata(json.loads(str(data["metadata_json"])))
        features = {side:{k:data[f"{side}_{k}"].copy() for k in ("valid",*FIELDS)} for side in SIDES}
    validate_surface_features(features)
    return features,meta
