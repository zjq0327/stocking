"""Export signed cloth feature data without applying display color management.

NPZ is authoritative. EXRs are linear, float32 exchange files; PNGs are explicitly
visualizations. All input arrays use row zero at the lowest local Y coordinate.
Only EXR I/O needs Blender. Nothing in this module changes a scene or renders it.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import numpy as np


EXPORT_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"
SIDES = ("front", "back")
VECTORS = ("P", "N", "T")
NPZ_NAME = "features.npz"


def _flatten_features(features):
    """Validate the data contract without changing or normalizing any values."""
    flattened = {}
    shape = None
    for side in SIDES:
        values = features[side]
        ids, valid = np.asarray(values["id"]), np.asarray(values["valid"])
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype != np.uint32:
            raise ValueError(f"{side}.id must be a nonempty uint32[H,W] array")
        if valid.shape != ids.shape or valid.dtype != np.bool_:
            raise ValueError(f"{side}.valid must be bool[H,W]")
        if not np.array_equal(ids != 0, valid):
            raise ValueError(f"{side}: ID zero must correspond exactly to invalid pixels")
        if shape is not None and ids.shape != shape:
            raise ValueError("Front and back must have identical array dimensions")
        shape = ids.shape
        flattened[f"{side}_id"] = ids
        flattened[f"{side}_valid"] = valid
        for name in VECTORS:
            array = np.asarray(values[name])
            if array.shape != (*shape, 3) or array.dtype != np.float32:
                raise ValueError(f"{side}.{name} must be float32[H,W,3]")
            if not np.isfinite(array).all():
                raise ValueError(f"{side}.{name} contains nonfinite values")
            if np.any(array[~valid] != 0):
                raise ValueError(f"{side}.{name} must be zero at invalid pixels")
            flattened[f"{side}_{name}"] = array
    return flattened


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path, data):
    path.write_text(json.dumps(_json_value(data), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _exr_channel_types(path):
    """Read the EXR header to distinguish FLOAT (2) from HALF (1) storage."""
    def cstring(stream):
        result = bytearray()
        while True:
            char = stream.read(1)
            if not char:
                raise ValueError("Truncated OpenEXR header")
            if char == b"\0":
                return result.decode("ascii")
            result.extend(char)
            if len(result) > 4096:
                raise ValueError("Invalid OpenEXR header string")

    with Path(path).open("rb") as stream:
        if stream.read(4) != struct.pack("<I", 20000630):
            raise ValueError(f"Not an OpenEXR file: {path}")
        if len(stream.read(4)) != 4:
            raise ValueError("Truncated OpenEXR version")
        while True:
            name = cstring(stream)
            if not name:
                break
            kind = cstring(stream)
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError("Truncated OpenEXR attribute")
            length = struct.unpack("<I", length_bytes)[0]
            if length > 16 * 1024 * 1024:
                raise ValueError("Unexpectedly large OpenEXR header attribute")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError("Truncated OpenEXR attribute payload")
            if name == "channels" and kind == "chlist":
                channels, offset = {}, 0
                while offset < len(payload) and payload[offset] != 0:
                    end = payload.index(0, offset)
                    channel = payload[offset:end].decode("ascii")
                    if end + 17 > len(payload):
                        raise ValueError("Truncated OpenEXR channel")
                    channels[channel] = struct.unpack_from("<i", payload, end + 1)[0]
                    offset = end + 17
                return channels
    raise ValueError("OpenEXR header contains no channel list")


def _write_exr(path, vector):
    import bpy

    height, width = vector.shape[:2]
    image = bpy.data.images.new("StockingFeatureExport", width=width, height=height,
                                alpha=True, float_buffer=True)
    try:
        image.colorspace_settings.name = "Non-Color"
        image.alpha_mode = "STRAIGHT"
        if hasattr(image, "use_half_precision"):
            image.use_half_precision = False
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[..., :3] = vector
        # Blender pixels are bottom-to-top, exactly matching our array contract.
        image.pixels.foreach_set(rgba.reshape(-1))
        image.update()
        image.file_format = "OPEN_EXR"
        image.filepath_raw = str(path)
        image.save()  # Deliberately NOT save_render(): no view/display transform.
        channels = _exr_channel_types(path)
        if any(channels.get(c) != 2 for c in ("R", "G", "B", "A")):
            raise RuntimeError(f"Expected float32 RGBA EXR, got {channels}: {path}")
    finally:
        bpy.data.images.remove(image)


def _read_exr(path):
    import bpy

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        image.alpha_mode = "STRAIGHT"
        width, height = image.size
        data = np.empty(height * width * 4, dtype=np.float32)
        image.pixels.foreach_get(data)
        return data.reshape(height, width, 4)
    finally:
        bpy.data.images.remove(image)


def _write_png(path, rgb, *, flip_y=True):
    """Minimal lossless RGB8 PNG writer, independent of Pillow and bpy."""
    pixels = np.asarray(rgb, dtype=np.uint8)
    if flip_y:
        pixels = pixels[::-1]
    height, width = pixels.shape[:2]

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    rows = b"".join(b"\0" + row.tobytes() for row in pixels)
    data = (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    Path(path).write_bytes(data)


# Compact bitmap labels keep the contact sheet self-contained on Blender Python.
_FONT = {
    "A": [14,17,17,31,17,17,17], "B": [30,17,17,30,17,17,30],
    "C": [14,17,16,16,16,17,14], "D": [30,17,17,17,17,17,30],
    "E": [31,16,16,30,16,16,31], "F": [31,16,16,30,16,16,16],
    "G": [14,17,16,23,17,17,15], "H": [17,17,17,31,17,17,17],
    "I": [14,4,4,4,4,4,14], "J": [7,2,2,2,18,18,12],
    "K": [17,18,20,24,20,18,17], "L": [16,16,16,16,16,16,31],
    "M": [17,27,21,21,17,17,17], "N": [17,25,21,19,17,17,17],
    "O": [14,17,17,17,17,17,14], "P": [30,17,17,30,16,16,16],
    "Q": [14,17,17,17,21,18,13], "R": [30,17,17,30,20,18,17],
    "S": [15,16,16,14,1,1,30], "T": [31,4,4,4,4,4,4],
    "U": [17,17,17,17,17,17,14], "V": [17,17,17,17,17,10,4],
    "W": [17,17,17,21,21,21,10], "X": [17,17,10,4,10,17,17],
    "Y": [17,17,10,4,4,4,4], "Z": [31,1,2,4,8,16,31],
    "0": [14,17,19,21,25,17,14], "1": [4,12,4,4,4,4,14],
    "2": [14,17,1,2,4,8,31], "3": [30,1,1,14,1,1,30],
    "4": [2,6,10,18,31,2,2], "5": [31,16,16,30,1,1,30],
    "6": [14,16,16,30,17,17,14], "7": [31,1,2,4,8,8,8],
    "8": [14,17,17,14,17,17,14], "9": [14,17,17,15,1,1,14],
    "-": [0,0,0,31,0,0,0], ".": [0,0,0,0,0,12,12],
    ":": [0,12,12,0,12,12,0], "/": [1,2,2,4,8,8,16],
    "(": [2,4,8,8,8,4,2], ")": [8,4,2,2,2,4,8],
    "+": [0,4,4,31,4,4,0], "=": [0,0,31,0,31,0,0],
    " ": [0,0,0,0,0,0,0],
}


def _label(canvas, text, x, y, scale=2):
    for char in text.upper():
        glyph = _FONT.get(char, _FONT[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    canvas[y + row*scale:y + (row+1)*scale,
                           x + col*scale:x + (col+1)*scale] = (224, 231, 239)
        x += 6 * scale


def _preview_images(features):
    heights = [features[s]["P"][features[s]["valid"], 2] for s in SIDES]
    available = [a for a in heights if a.size]
    low = min(float(a.min()) for a in available) if available else 0.0
    high = max(float(a.max()) for a in available) if available else 0.0
    images = {}
    palette = np.array([[0.10,0.24,0.55], [0.12,0.67,0.69], [1.0,0.77,0.32]])
    for side in SIDES:
        valid = features[side]["valid"]
        ids = features[side]["id"].astype(np.uint64)
        color = np.stack([64 + ((ids * a + b) % 176)
                          for a, b in ((97,31),(57,89),(131,7))], axis=-1).astype(np.uint8)
        color[~valid] = 0
        images[f"{side}_id"] = color
        z = features[side]["P"][..., 2].astype(np.float64)
        normalized = (np.clip((z-low)/(high-low), 0, 1) if high > low
                      else np.full_like(z, 0.5))
        rgb = np.stack([np.interp(normalized, [0,0.5,1], palette[:,i])
                        for i in range(3)], axis=-1)
        color = np.round(rgb * 255).astype(np.uint8)
        color[~valid] = 0
        images[f"{side}_P_height"] = color
        for name in ("N", "T"):
            color = np.round(np.clip((features[side][name]+1)*0.5, 0, 1)*255).astype(np.uint8)
            color[~valid] = 0
            images[f"{side}_{name}"] = color
    return images, [low, high]


def _contact_sheet(images, z_range):
    height, width = images["front_id"].shape[:2]
    scale = min(512 / max(height, width), max(1, 256 / min(height, width)))
    panel_w, panel_h = max(1, round(width*scale)), max(1, round(height*scale))
    # Minimum width leaves room for labels even for narrow rectangular maps.
    cell_w = max(256, panel_w)
    margin, heading = 16, 44
    canvas = np.full((2*(panel_h+heading)+3*margin+68, 4*cell_w+5*margin, 3),
                     (20,27,38), dtype=np.uint8)
    for row, side in enumerate(SIDES):
        for col, (key, title) in enumerate((("id","ID"),("P_height","P HEIGHT"),
                                          ("N","NORMAL XYZ"),("T","TANGENT XYZ"))):
            x, y = margin+col*(cell_w+margin), margin+row*(panel_h+heading+margin)
            _label(canvas, f"{side} / {title}", x+6, y+10)
            image = images[f"{side}_{key}"][::-1]
            yi = np.minimum((np.arange(panel_h)*height/panel_h).astype(int), height-1)
            xi = np.minimum((np.arange(panel_w)*width/panel_w).astype(int), width-1)
            canvas[y+heading:y+heading+panel_h, x:x+panel_w] = image[yi[:,None], xi]
    footer_y = canvas.shape[0]-58
    _label(canvas, f"SHARED Z RANGE (MM): {z_range[0]:.6g} .. {z_range[1]:.6g}",
           margin, footer_y)
    _label(canvas, "PREVIEW ONLY - RAW SIGNED DATA IN NPZ / EXR - BLACK = EMPTY",
           margin, footer_y+28)
    return canvas


def export_features(features, p, asset_dir, preview_dir, extra_metadata=None):
    """Export the feature arrays and return absolute output paths.

    Extra metadata is retained at top level, but generated contract/version/file
    fields take precedence so callers cannot accidentally mislabel raw arrays.
    """
    flat = _flatten_features(features)
    asset_dir, preview_dir = Path(asset_dir).resolve(), Path(preview_dir).resolve()
    # Fail before writing partial assets when called outside Blender.
    import bpy

    asset_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    files = {"npz": str(asset_dir/NPZ_NAME),
             "parameters": str(asset_dir/"parameters.json"),
             "metadata": str(asset_dir/"metadata.json"), "exr": {}, "previews": {}}
    np.savez_compressed(files["npz"], **flat)
    _write_json(Path(files["parameters"]), p.to_dict())
    for side in SIDES:
        for name in VECTORS:
            key = f"{side}_{name}"
            path = asset_dir/f"{key}.exr"
            _write_exr(path, flat[key])
            files["exr"][key] = str(path)
    images, z_range = _preview_images(features)
    for key, pixels in images.items():
        path = preview_dir/f"{key}.png"
        _write_png(path, pixels)
        files["previews"][key] = str(path)
    contact = preview_dir/"features_contact_sheet.png"
    _write_png(contact, _contact_sheet(images, z_range), flip_y=False)
    files["contact_sheet"] = str(contact)
    height, width = flat["front_id"].shape
    metadata = dict(extra_metadata or {})
    metadata.update({
        "schema_version": SCHEMA_VERSION, "exporter": "stocking_material.export",
        "exporter_version": EXPORT_VERSION, "blender_version": bpy.app.version_string,
        "period_mm": _json_value(p.period_mm), "resolution": {"width": width, "height": height},
        "representation": "Periodic ply-surface features, orthographic first intersection",
        "representation_reference": {
            "title": "A Realistic Surface-based Cloth Rendering Model (2023), section 4.1",
            "url": "https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig23cloth.pdf",
            "scope": "Geometric feature representation only; not the complete paper BSDF",
        },
        "coordinates": {
            "unit": "millimeter", "position": "P is local XYZ in millimeters",
            "array_order": "H,W,channel; row 0 is lowest local Y; columns increase with X",
            "front_rays": "origins above +Z, direction (0,0,-1)",
            "back_rays": "origins below -Z, direction (0,0,+1)",
            "sides": "same XY sample grid and same local vector basis; back is NOT mirrored",
            "N": "outward unit surface normal; signed XYZ, each component in [-1,1]",
            "T": "unit longitudinal ply tangent; signed XYZ, consistent yarn orientation",
        },
        "data_contract": {
            "npz": "features.npz is authoritative; no pickle/object arrays",
            "id": "uint32[H,W]; 0=empty; positive labels are categorical, never interpolated",
            "valid": "bool[H,W]; center-ray hit, NOT subpixel fractional coverage or directional transparency",
            "P_N_T": "float32[H,W,3]; invalid pixels contain zero, identified using valid",
            "exr": "RGBA FLOAT32; RGB=raw signed vector, A=1 everywhere, including empty pixels",
            "exr_color_management": "Non-Color; Image.save, no view transform or normalization",
        },
        "preview_encoding": {
            "id": "deterministic categorical colors; black is empty; raw IDs are only in NPZ",
            "P_height": "Z-only false color with a shared front/back range in millimeters",
            "P_height_range_mm": z_range,
            "N_T": "RGB=(XYZ+1)/2; black at invalid pixels",
            "rows": "PNG top row is highest local Y; all panels have the same orientation",
            "sampling": "nearest-neighbor contact sheet; no categorical or vector averaging",
        },
        "files": {"npz": NPZ_NAME, "parameters": "parameters.json",
                  "exr": {key: Path(path).name for key,path in files["exr"].items()},
                  "preview_directory": str(preview_dir),
                  "contact_sheet": contact.name},
    })
    _write_json(Path(files["metadata"]), metadata)
    return files


def verify_roundtrip(features, asset_dir):
    """Read every NPZ/EXR pixel back and report exactness, errors and sign retention."""
    expected = _flatten_features(features)
    asset_dir = Path(asset_dir).resolve()
    metrics = {"passed": True, "npz": {}, "exr": {}, "errors": []}
    with np.load(asset_dir/NPZ_NAME, allow_pickle=False) as data:
        if set(data.files) != set(expected):
            metrics["errors"].append("NPZ field set does not match the feature contract")
        for key, reference in expected.items():
            actual = data[key] if key in data else None
            same = (actual is not None and actual.dtype == reference.dtype and
                    actual.shape == reference.shape and np.array_equal(actual, reference))
            metrics["npz"][key] = {"exact": bool(same), "dtype": str(reference.dtype)}
            if not same:
                metrics["errors"].append(f"NPZ mismatch: {key}")
    for side in SIDES:
        for name in VECTORS:
            key = f"{side}_{name}"
            path = asset_dir/f"{key}.exr"
            channel_types = _exr_channel_types(path)
            actual = _read_exr(path)
            reference = expected[key]
            shape_ok = actual.shape == (*reference.shape[:2], 4)
            values = actual[..., :3]
            float32_storage = all(channel_types.get(c) == 2 for c in ("R","G","B","A"))
            exact = shape_ok and np.array_equal(values, reference)
            error = float(np.max(np.abs(values.astype(np.float64)-reference))) if shape_ok else None
            negative_count = int(np.count_nonzero(reference < 0))
            signs_ok = shape_ok and np.array_equal(values < 0, reference < 0)
            alpha_one = bool(np.all(actual[..., 3] == 1))
            passed = bool(float32_storage and exact and signs_ok and alpha_one)
            metrics["exr"][key] = {"passed": passed, "exact": bool(exact),
                "max_abs_error": error, "float32_storage": float32_storage,
                "negative_components": negative_count, "negative_signs_preserved": bool(signs_ok),
                "alpha_is_one": alpha_one, "channel_types": channel_types}
            if not passed:
                metrics["errors"].append(f"EXR round-trip mismatch: {key}")
    metrics["passed"] = not metrics["errors"]
    return metrics
