"""Strict, pickle-free storage for physical garment mesh correspondence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .garment_data import GarmentMeshPair, validate_garment_metadata


_REQUIRED = frozenset(("reference_positions_mm", "current_positions_mm", "triangles",
                       "material_uv", "material_point_ids"))
_OPTIONAL = frozenset(("face_material_uv", "current_face_directors"))


def _paths(path):
    source = Path(path)
    if source.suffix.lower() in (".npz", ".json"):
        source = source.with_suffix("")
    return Path(str(source) + ".npz"), Path(str(source) + ".json")


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_garment_pair(pair: GarmentMeshPair, path):
    """Write `<path>.npz` and `<path>.json`, with explicit contract and hash.

    Runtime source data must already be mm. Formal caller-created assets belong
    under stocking/assets; callers choose the garment-specific directory.
    """
    pair.validate()
    npz_path, json_path = _paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: getattr(pair, key) for key in sorted(_REQUIRED)}
    arrays.update({key: getattr(pair, key) for key in sorted(_OPTIONAL)
                   if getattr(pair, key) is not None})
    np.savez_compressed(npz_path, **arrays)
    payload = {"metadata": pair.metadata, "array_file": npz_path.name,
               "array_sha256": _digest(npz_path),
               "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in arrays.items()}}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"npz": str(npz_path), "json": str(json_path)}


def load_garment_pair(path, *, expected_reference_version=None, expected_material_version=None):
    """Load without pickle; reject unknown schema, unit, hash, or correspondence.

    Optional expected versions prevent mixing an asset with an unrelated yarn
    reference/material library. A supported schema alone is not that identity.
    """
    npz_path, json_path = _paths(path)
    def reject_constant(token):
        raise ValueError(f"JSON contains nonfinite constant {token}")
    payload = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "array_file", "array_sha256", "arrays"}:
        raise ValueError("garment asset envelope has unsupported fields")
    metadata = payload["metadata"]
    validate_garment_metadata(metadata)
    for expected, key in ((expected_reference_version, "reference_version"),
                           (expected_material_version, "material_version")):
        if expected is not None and metadata[key] != expected:
            raise ValueError(f"garment {key} does not match the requested version")
    if payload["array_file"] != npz_path.name:
        raise ValueError("garment array_file must name its own companion NPZ")
    if not isinstance(payload["array_sha256"], str) or payload["array_sha256"] != _digest(npz_path):
        raise ValueError("garment NPZ hash does not match metadata")
    with np.load(npz_path, allow_pickle=False) as source:
        keys = set(source.files)
        if not _REQUIRED <= keys or not keys <= _REQUIRED | _OPTIONAL or len(keys) != len(source.files):
            raise ValueError("garment NPZ has missing, duplicate, or unsupported arrays")
        if not isinstance(payload["arrays"], dict) or keys != set(payload["arrays"]):
            raise ValueError("garment array manifest does not match NPZ")
        arrays = {}
        for key in keys:
            value = source[key]
            description = payload["arrays"][key]
            if description != {"shape": list(value.shape), "dtype": str(value.dtype)}:
                raise ValueError(f"garment array manifest mismatch for {key}")
            arrays[key] = value.copy()
    return GarmentMeshPair(metadata=metadata, **arrays)
