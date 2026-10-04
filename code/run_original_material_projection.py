"""Transfer the frozen authored material chart to original evaluated render geometry.

Source-face provenance selects the correct sock before nearest-surface
interpolation. This is an explicit geometric transfer, not a new material
correspondence, and records its residuals for the fixed-camera bake.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")
from stocking_material.asset_io import load_garment_pair
from stocking_material.authored_yarn import build_authored_phase
from stocking_material.parameters import Parameters


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closest_chart(tree, pair, phase, point):
    result = tree.find_nearest(Vector(point))
    if result is None:
        raise ValueError("no material face found")
    position, normal, face, distance = result
    xyz = pair.current_positions_mm[pair.triangles[face]]
    e = np.stack((xyz[1]-xyz[0], xyz[2]-xyz[0]), axis=1)
    uv = np.linalg.solve(e.T@e, e.T@(np.asarray(position)-xyz[0]))
    bary = np.asarray((1-uv.sum(), uv[0], uv[1]))
    if bary.min() < -2e-4 or bary.max() > 1.0002:
        raise ValueError("nearest surface barycentric coordinates escaped triangle")
    return phase.face_q_mm[face].T@bary, int(face), float(distance), bary


def main():
    started = time.perf_counter()
    asset = ROOT / "stocking/assets/original-yarn-shader-v1"
    audit = json.loads((ROOT / "build-support/yarn-shader-integration-v1/original-shader-audit.json").read_text(encoding="utf8"))
    target_path = ROOT / "stocking/assets/authored-target-v2/authored_target.npz"
    with np.load(target_path, allow_pickle=False) as z:
        target = {key:z[key].copy() for key in z.files}
    models = []
    for side in ("left", "right"):
        pair_path = ROOT / "stocking/assets/authored-fit-v1" / side / "garment_pair.npz"
        pair = load_garment_pair(pair_path)
        phase = build_authored_phase(pair, Parameters(samples_per_loop=192))
        tree = BVHTree.FromPolygons(pair.current_positions_mm.tolist(), pair.triangles.tolist(), all_triangles=True)
        models.append((pair, phase, tree, sha(pair_path)))
    face_component = target["component"][target["triangles"][:, 0]]
    assert np.all(target["component"][target["triangles"]] == face_component[:, None])
    report = {"schema":"stocking.original_material_projection.v1", "source_sha256":audit["source_sha256"],
              "frame":996, "target_sha256":sha(target_path), "pair_sha256":[m[3] for m in models],
              "method":"source provenance selects sock; nearest authored material surface transfers q and face state",
              "exact_inverse_material_binding":False, "local_affine_periodic_yarn_approximation":True,
              "objects":[]}
    for component, name in enumerate(("body_base", "toe_stocking")):
        source_path = asset / "inputs" / (name+".npz")
        with np.load(source_path, allow_pickle=False) as z:
            positions, triangles = z["positions_world_mm"].copy(), z["triangles"].copy()
            source_polygon = z["triangle_polygon_index"].copy()
        historical_path = ROOT / "stocking/assets/authored-target-v1" / (name+"-source.npz")
        with np.load(historical_path,allow_pickle=False) as z:
            fixed_source_polygons = z["face_source_polygon"].copy()
            original_positions = z["positions_mm"].copy()
        source_position_error = float(np.linalg.norm(positions-original_positions,axis=1).max())
        if source_position_error > 1e-4:
            raise ValueError("original render vertices differ from the frozen authored source")
        polygon_votes = np.zeros((int(source_polygon.max())+1,2),dtype=np.int32)
        selected = (target["face_source_object"] == component) & target["source_face_provenance_valid"]
        source_face = target["face_source_triangle"][selected]
        socks = face_component[selected]
        assert np.all((source_face >= 0) & (source_face < len(fixed_source_polygons)))
        # Source triangle IDs use fixed unposed tessellation. Current Blender
        # can flip a quad diagonal under pose, so retain polygon identity.
        np.add.at(polygon_votes, (fixed_source_polygons[source_face],socks), 1)
        face_votes = polygon_votes[source_polygon]
        touched_faces = np.any(face_votes > 0, axis=1)
        if np.any(np.all(face_votes > 0, axis=1)):
            raise ValueError("a source face is claimed by both socks")
        face_sock = face_votes.argmax(axis=1).astype(np.int32)
        face_sock[~touched_faces] = -1
        vertex_votes = np.zeros((len(positions),2), dtype=np.int32)
        for side in (0,1):
            np.add.at(vertex_votes[:,side], triangles[face_sock == side].reshape(-1), 1)
        touched_vertices = np.any(vertex_votes > 0,axis=1)
        if np.any(np.all(vertex_votes > 0,axis=1)):
            raise ValueError("source vertex crosses sock components")
        vertex_sock = vertex_votes.argmax(axis=1)
        q = np.zeros((len(positions),2),dtype=np.float64)
        residual = np.zeros(len(positions),dtype=np.float64)
        for index in np.flatnonzero(touched_vertices):
            pair, phase, tree, _ = models[vertex_sock[index]]
            q[index], _, residual[index], _ = closest_chart(tree,pair,phase,positions[index])
        center = positions[triangles].mean(axis=1)
        material_faces = np.full(len(triangles),-1,dtype=np.int32)
        center_residual = np.zeros(len(triangles),dtype=np.float64)
        for index in np.flatnonzero(touched_faces):
            pair, phase, tree, _ = models[face_sock[index]]
            _, material_faces[index], center_residual[index], _ = closest_chart(tree,pair,phase,center[index])
        triangle_q = q[triangles]
        # Each triangle uses one continuous course chart across the cyclic seam.
        for side in (0,1):
            mask = face_sock == side
            width = models[side][1].course_width_mm
            x = triangle_q[mask,:,0]
            x = x[:,0:1]+(x-x[:,0:1]+width/2)%width-width/2
            triangle_q[mask,:,0] = x
        # The original shader domain extends above the authored cuff. Do not
        # stretch that boundary's material chart into unbound original skin.
        # Reject distant transfers and let the unchanged shader branch handle
        # those pixels, instead of relaxing the material correspondence gate.
        transfer_limit_mm = 1.0
        rejected = touched_faces & ((center_residual > transfer_limit_mm) |
                    np.any(residual[triangles] > transfer_limit_mm,axis=1))
        mapped_faces = touched_faces & ~rejected
        out = asset / (name+"-material-projection.npz")
        np.savez_compressed(out, positions_world_mm=positions, triangles=triangles,
                            triangle_q_mm=triangle_q, triangle_material_face_index=material_faces,
                            triangle_mapped=mapped_faces.astype(np.uint8), triangle_sock_index=face_sock,
                            triangle_component=np.full(len(triangles),component,dtype=np.int32),
                            vertex_q_mm=q, vertex_mapped=touched_vertices.astype(np.uint8),
                            vertex_transfer_distance_mm=residual, triangle_transfer_distance_mm=center_residual)
        distances = residual[touched_vertices]
        record = {"name":name,"file":out.name,"sha256":sha(out),"source_geometry_sha256":sha(source_path),
                  "vertices":len(positions),"triangles":len(triangles),
                  "mapped_vertices":int(touched_vertices.sum()),"candidate_triangles":int(touched_faces.sum()),
                  "mapped_triangles":int(mapped_faces.sum()),"rejected_transfer_triangles":int(rejected.sum()),
                  "transfer_limit_mm":transfer_limit_mm,
                  "vertex_transfer_mm":{"median":float(np.median(distances)),"p95":float(np.quantile(distances,.95)),"max":float(distances.max())},
                  "triangle_transfer_max_mm":float(center_residual.max()),"sock_provenance_ambiguities":0,
                  "source_polygon_identity_used":True,"source_vertex_position_max_error_mm":source_position_error,
                  "fixed_source_npz_sha256":sha(historical_path),
                  "unmapped_behavior":"keep original empirical C via shader fallback"}
        report["objects"].append(record)
        print("MATERIAL_PROJECTION_OBJECT",json.dumps(record,ensure_ascii=True),flush=True)
    cam = np.asarray(audit["camera"]["matrix_world"],dtype=np.float64)
    projection = np.asarray(audit["camera"]["projection_matrix"],dtype=np.float64)
    scale = np.diag([.001,.001,.001,1.])
    report.update(world_to_clip=(projection@np.linalg.inv(cam)@scale).tolist(), width=1080,height=1920,
                  period_mm=Parameters().period_mm, elapsed_seconds=time.perf_counter()-started)
    (asset/"mapping-manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("MATERIAL_PROJECTION_COMPLETE",report["elapsed_seconds"],flush=True)


if __name__ == "__main__":
    main()
