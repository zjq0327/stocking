"""Bounded-memory queries of independent per-triangle yarn features.

Nearest local barycentric-grid sampling preserves holes; no interpolation
between yarn surfaces, holes, states or differently oriented local frames.
Shared-edge queries use their canonical edge grid.  World P=S+B*offset and
world N/T=B*local; no additional macro deformation is applied.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .triangle_features import (TriangleSampling, CONFIGURATIONS, SIDES, FIELDS,
    EDGES, face_sample_barycentric, sample_domain, _sha256)


def nearest_face_sample(sampling, face, weights, configuration="current"):
    """A bounded-distance local grid sample; not a global Euclidean search."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < -1e-12) or abs(w.sum() - 1) > 1e-12:
        raise ValueError("invalid barycentric query")
    w = np.maximum(w, 0)
    w /= w.sum()
    for corner in range(3):
        if w[corner] >= 1 - 1e-12:
            return corner
    divisions = sampling.arrays["face_edge_divisions"][face]
    edge_bases = np.r_[3, 3 + np.cumsum(divisions[:-1] - 1)]
    candidates = set((0, 1, 2))
    for edge_index, (a, b) in enumerate(EDGES):
        d = int(divisions[edge_index])
        vertices = sampling.arrays["triangles"][face]
        # Round ties in a common physical-edge orientation, even when the
        # neighboring face enumerates that edge in reverse order.
        forward = vertices[a] < vertices[b]
        t = w[b if forward else a] / max(w[a] + w[b], 1e-30)
        canonical_k = int(np.clip(np.floor(t * d + .5), 0, d))
        k = canonical_k if forward else d - canonical_k
        chosen = a if k == 0 else b if k == d else int(edge_bases[edge_index] + k - 1)
        if w[3 - a - b] <= 1e-12:
            return chosen
        candidates.add(chosen)
    n = int(sampling.arrays["face_subdivisions"][face])
    base = int(divisions.sum())
    for i in range(int(np.floor(w[1] * n)) - 1, int(np.floor(w[1] * n)) + 3):
        for j in range(int(np.floor(w[2] * n)) - 1, int(np.floor(w[2] * n)) + 3):
            if i > 0 and j > 0 and i + j < n:
                row = i - 1
                before = row * (2 * n - 3 - row) // 2
                candidates.add(base + before + j - 1)
    choices = np.array(sorted(candidates), dtype=np.int64)
    bary = face_sample_barycentric(sampling, face, choices)
    vertices = sampling.arrays[configuration + "_positions_mm"][sampling.arrays["triangles"][face]]
    distances = np.linalg.norm((bary - w) @ vertices, axis=1)
    return int(choices[np.argmin(distances)])


class TriangleFeatureReader:
    def __init__(self, directory, *, verify_hashes=True):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "triangle_features.json").read_text(encoding="utf-8"))
        path = self.directory / self.metadata["domain_file"]
        if verify_hashes and _sha256(path) != self.metadata["domain_sha256"]:
            raise ValueError("sampling domain hash differs from manifest")
        with np.load(path, allow_pickle=False) as raw:
            arrays = {key: raw[key].copy() for key in raw.files if key != "metadata_json"}
            domain_metadata = json.loads(str(raw["metadata_json"]))
        self.sampling = TriangleSampling(arrays, domain_metadata)
        if domain_metadata["pair_fingerprint"] != self.metadata["pair_fingerprint"]:
            raise ValueError("feature domain identity differs from manifest")
        self.verify_hashes = verify_hashes
        # Addresses increase by physical face then face-local index; gaps are
        # unrequested/ungenerated faces and remain geometry_available=false.
        offsets = arrays["face_sample_offsets"]
        self._starts = np.array([offsets[c["first_address"][0]] + c["first_address"][1]
                                for c in self.metadata["chunks"]], dtype=np.int64)
        self._ends = np.array([offsets[c["last_address"][0]] + c["last_address"][1]
                              for c in self.metadata["chunks"]], dtype=np.int64)
        self._cache_index, self._cache = None, None

    def _chunk(self, index):
        if index != self._cache_index:
            record = self.metadata["chunks"][index]
            path = self.directory / record["file"]
            if self.verify_hashes and _sha256(path) != record["sha256"]:
                raise ValueError("feature chunk hash differs from manifest")
            with np.load(path, allow_pickle=False) as raw:
                self._cache = {key: raw[key].copy() for key in raw.files}
            self._cache_index = index
        return self._cache

    def query(self, face_indices, barycentric, *, configuration="current"):
        if configuration not in CONFIGURATIONS:
            raise ValueError("unknown configuration")
        faces = np.asarray(face_indices)
        bary = np.asarray(barycentric, dtype=np.float64)
        if (faces.ndim != 1 or faces.dtype.kind not in "iu" or bary.shape != (len(faces), 3)
                or np.any(faces < 0) or np.any(faces >= len(self.sampling.arrays["triangles"]))):
            raise ValueError("query needs in-range integer faces and corresponding barycentric weights")
        # Validate all weights and retain exact requested surface addresses.
        requested_surface, _ = sample_domain(self.sampling, faces, bary, configuration)
        local = np.array([nearest_face_sample(self.sampling, int(f), w, configuration)
                          for f, w in zip(faces, bary)], dtype=np.int64)
        sampled_bary = np.array([face_sample_barycentric(self.sampling, int(f), np.array([k]))[0]
                                for f, k in zip(faces, local)], dtype=np.float64).reshape(-1, 3)
        surface, frames = sample_domain(self.sampling, faces, sampled_bary, configuration)
        addresses = self.sampling.sample_offsets[faces] + local
        selected_chunks = np.searchsorted(self._starts, addresses, side="right") - 1
        output = {"geometry_available": np.zeros(len(faces), dtype=bool),
                  "requested_surface_mm": requested_surface, "sample_surface_mm": surface,
                  "sample_barycentric": sampled_bary, "face_local_sample_index": local,
                  "sampling_distance_mm": np.linalg.norm(surface - requested_surface, axis=1),
                  "configuration": configuration,
                  **{side: {"valid": np.zeros(len(faces), dtype=bool),
                            **{key: np.zeros((len(faces), 3), dtype=np.float64) for key in ("P_mm", "N", "T")}}
                     for side in SIDES}}
        for chunk_index in np.unique(selected_chunks[selected_chunks >= 0]):
            positions = np.flatnonzero((selected_chunks == chunk_index) & (addresses <= self._ends[chunk_index]))
            data = self._chunk(int(chunk_index))
            chunk_addresses = self.sampling.sample_offsets[data["face_indices"]] + data["face_local_indices"]
            index = np.searchsorted(chunk_addresses, addresses[positions])
            # Selected-face chunks may span gaps in the global address space.
            safe = index < len(chunk_addresses)
            safe[safe] &= chunk_addresses[index[safe]] == addresses[positions[safe]]
            positions, index = positions[safe], index[safe]
            output["geometry_available"][positions] = True
            for side in SIDES:
                prefix = configuration + "_" + side + "_"
                hit = data[prefix + "valid"][index]
                target, source = positions[hit], index[hit]
                output[side]["valid"][target] = True
                output[side]["P_mm"][target] = surface[target] + np.einsum(
                    "nij,nj->ni", frames[target], data[prefix + "P_offset_mm"][source])
                for key, source_key in (("N", "N_local"), ("T", "T_local")):
                    output[side][key][target] = np.einsum("nij,nj->ni", frames[target], data[prefix + source_key][source])
        return output
