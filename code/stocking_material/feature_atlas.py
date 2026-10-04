"""Pack already solved/baked states without interpolating different geometries.

This initial atlas uses nearest sample lookup. Edge padding prevents adjacent
blocks contaminating one another; it is not a physical coverage/mipmap filter.
"""
from pathlib import Path
import json
import numpy as np
from .surface_bake import validate_surface_features, _metadata, SIDES, FIELDS
from .export import _write_exr, _read_exr


def build_feature_atlas(regions, asset_dir, *, region_metadata, gutter=2):
    if not regions or isinstance(gutter,bool) or not isinstance(gutter,int) or gutter < 1:
        raise ValueError("Named regions and positive integer gutter required")
    shapes = []
    if set(region_metadata)!=set(regions):
        raise ValueError("Every named region requires its own feature contract")
    contracts = {name:_metadata(region_metadata[name]) for name in regions}
    identities = {(m["reference_version"],m["material_version"]) for m in contracts.values()}
    if len(identities)!=1:
        raise ValueError("A unified atlas cannot mix unrelated reference/material versions")
    for name, features in regions.items():
        if not isinstance(name,str) or not name:
            raise ValueError("Region names must be nonempty strings")
        validate_surface_features(features)
        shapes.append(features["front"]["valid"].shape)
    if len(set(shapes))!=1:
        raise ValueError("Initial atlas requires equally sized blocks")
    h,w = shapes[0]
    stride_x,stride_y = w+2*gutter,h+2*gutter
    columns = int(np.ceil(np.sqrt(len(regions))))
    rows = int(np.ceil(len(regions)/columns))
    ah,aw = rows*stride_y,columns*stride_x
    atlas = {side:{"valid":np.zeros((ah,aw),dtype=bool),**{key:np.zeros((ah,aw,3),dtype=np.float32) for key in FIELDS}} for side in SIDES}
    index = {}
    for i,(name,features) in enumerate(regions.items()):
        x0,y0 = (i%columns)*stride_x,(i//columns)*stride_y
        index[name] = {"pixel_origin":[x0+gutter,y0+gutter],"pixel_size":[w,h],
                       "uv_min":[(x0+gutter)/aw,(y0+gutter)/ah],"uv_max":[(x0+gutter+w)/aw,(y0+gutter+h)/ah],
                       "feature_metadata":contracts[name]}
        for side in SIDES:
            for key in ("valid",*FIELDS):
                value = features[side][key]
                padding = ((gutter,gutter),(gutter,gutter))+(((0,0),) if value.ndim==3 else ())
                atlas[side][key][y0:y0+stride_y,x0:x0+stride_x] = np.pad(value,padding,mode="edge")
    validate_surface_features(atlas)
    directory = Path(asset_dir)
    directory.mkdir(parents=True,exist_ok=True)
    arrays = {f"{side}_{key}":atlas[side][key] for side in SIDES for key in ("valid",*FIELDS)}
    meta = {"schema":"stocking.feature_atlas","schema_version":1,"size":[aw,ah],"regions":index,
            "reference_version":next(iter(identities))[0],"material_version":next(iter(identities))[1],
            "position_semantics":"offset_from_current_shell_in_current_orthonormal_frame","length_unit":"mm",
            "filter":"nearest; no cross-state interpolation or mipmap coverage", "gutter_pixels":gutter,
            "usage":"use region UV and that region's current shell frame; no additional macro deformation"}
    meta["image_paths"] = {f"{side}_{key}":str(directory/f"{side}_{key}.exr") for side in SIDES for key in (*FIELDS,"valid")}
    arrays["metadata_json"] = np.array(json.dumps(meta,allow_nan=False))
    np.savez_compressed(directory/"feature_atlas.npz",**arrays)
    paths = {}
    for side in SIDES:
        for key in FIELDS:
            p = directory/f"{side}_{key}.exr"
            _write_exr(p,atlas[side][key])
            if not np.array_equal(_read_exr(p)[...,:3],atlas[side][key]):
                raise RuntimeError("Atlas EXR roundtrip mismatch")
            paths[f"{side}_{key}"] = str(p)
        p = directory/f"{side}_valid.exr"
        _write_exr(p,np.repeat(atlas[side]["valid"][...,None],3,axis=2).astype(np.float32))
        if not np.array_equal(_read_exr(p)[...,0],atlas[side]["valid"].astype(np.float32)):
            raise RuntimeError("Atlas valid EXR roundtrip mismatch")
        paths[f"{side}_valid"] = str(p)
    meta["image_paths"] = paths
    (directory/"feature_atlas.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with np.load(directory/"feature_atlas.npz",allow_pickle=False) as restored:
        if not all(np.array_equal(restored[k],v) for k,v in arrays.items()):
            raise RuntimeError("Atlas roundtrip mismatch")
    return atlas,meta


def sample_atlas_nearest(atlas, metadata, region, phase_xy):
    if region not in metadata["regions"]:
        raise ValueError("Unknown region; atlas lookup cannot choose a nearby state")
    phase = np.asarray(phase_xy,dtype=float)
    if phase.ndim!=2 or phase.shape[1]!=2 or not np.isfinite(phase).all() or np.any(phase<0) or np.any(phase>1):
        raise ValueError("phase_xy must be finite [N,2] within the named region [0,1]")
    block = metadata["regions"][region]
    size = np.asarray(block["pixel_size"])
    pixels = np.minimum((phase*size).astype(int),size-1)+np.array(block["pixel_origin"])
    return {side:{key:atlas[side][key][pixels[:,1],pixels[:,0]] for key in ("valid",*FIELDS)} for side in SIDES}
