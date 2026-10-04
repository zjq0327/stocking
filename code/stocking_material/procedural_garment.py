"""Adjustable unworn stocking GEOMETRY INITIALIZATION, in millimetres.

A planar bent centerline carries an elliptical shell with a single open cuff.
Front/back retain distinct noncoincident material points.  The toe closes by
joining the two halves to a distributed seam chain, never to one star pole.
This is not a knitting-pattern/short-row generator, stress-free reference,
dressing solver, or material calibration.  It is an independent starting sock
shell, not a shrinkwrap of the worn leg.  Physical IDs/UV stay fixed when only
geometric parameters change; changing sampling resolution changes topology.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .garment_data import GarmentMeshPair, default_garment_metadata


@dataclass
class ProceduralGarmentParameters:
    leg_length_mm: float = 520.0
    leg_width_mm: float = 90.0
    ankle_width_mm: float = 48.0
    foot_length_mm: float = 210.0
    foot_width_mm: float = 64.0
    flattening: float = .16
    heel_bend_radius_mm: float = 55.0
    toe_length_mm: float = 35.0
    toe_taper: float = .15
    toe_rounding_ratio: float = .25
    circumference_segments: int = 32
    leg_segments: int = 64
    heel_segments: int = 24
    foot_segments: int = 24
    toe_segments: int = 12

    def __post_init__(self):
        self.validate()

    def validate(self):
        integer_names = ("circumference_segments", "leg_segments", "heel_segments", "foot_segments", "toe_segments")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{field.name} must be an integer")
                minimum = 12 if field.name == "circumference_segments" else 4
                if not minimum <= value <= 512:
                    raise ValueError(f"{field.name} must be between {minimum} and 512")
            elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite real number")
        if self.circumference_segments % 4:
            raise ValueError("circumference_segments must be divisible by four for paired toe halves")
        for name in ("leg_length_mm", "leg_width_mm", "ankle_width_mm", "foot_length_mm",
                     "foot_width_mm", "heel_bend_radius_mm", "toe_length_mm"):
            if not 1e-3 <= getattr(self, name) <= 1e5:
                raise ValueError(f"{name} must be in the supported 0.001..100000 mm range")
        if not .08 <= self.flattening <= .8:
            raise ValueError("flattening must be .08..0.8; front/back cannot coincide")
        if not 0 <= self.toe_taper <= .4:
            raise ValueError("toe_taper must be 0..0.4")
        if not 0 <= self.toe_rounding_ratio <= .4:
            raise ValueError("toe_rounding_ratio must be 0..0.4 to keep the cap moving forward")
        radius = self.heel_bend_radius_mm
        if radius <= .575 * max(self.ankle_width_mm, self.foot_width_mm):
            raise ValueError("heel bend radius must exceed 1.15 times the largest local half-width")
        if self.leg_length_mm <= radius + self.leg_width_mm:
            raise ValueError("leg_length_mm must leave a straight leg longer than leg_width_mm above the heel")
        if self.foot_length_mm <= radius + self.toe_length_mm + .5 * self.foot_width_mm:
            raise ValueError("foot_length_mm must leave a positive straight foot before the toe")
        return self

    def to_dict(self):
        self.validate()
        return {key: int(value) if "segments" in key else float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("garment parameters must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown garment parameters: " + ", ".join(sorted(unknown)))
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def topology_summary(triangles, vertex_count):
    """Check the generated shell is one orientable disk with only its cuff open."""
    edges = {}
    for a, b, c in triangles:
        for i, j in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(i), int(j))))
            edges[key] = edges.get(key, 0) + 1
    if any(count not in (1, 2) for count in edges.values()):
        raise ValueError("generated shell contains a nonmanifold edge")
    boundary = [edge for edge, count in edges.items() if count == 1]
    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if not neighbors or any(len(connected) != 2 for connected in neighbors.values()):
        raise ValueError("generated cuff boundary must be a closed vertex loop")
    remaining = set(neighbors)
    loops = 0
    while remaining:
        loops += 1
        pending = [remaining.pop()]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    euler = vertex_count - len(edges) + len(triangles)
    if loops != 1 or euler != 1:
        raise ValueError("generated stocking must have disk topology with exactly one open cuff")
    return dict(vertex_count=vertex_count, triangle_count=len(triangles), edge_count=len(edges),
                euler_characteristic=euler, boundary_loops=loops, boundary_edges=len(boundary),
                manifold_edges=True)


def generate_reference_garment(params=None):
    """Generate complete flattened X0; current=X0 until an explicit later solve.

    The broad stocking plane is XZ; Y is the front/back thickness.  Width is
    the full ellipse major diameter perpendicular to the local centerline in
    XZ. ``foot_length_mm`` is centerline reach from leg axis to toe center,
    not the complete projected bounding-box length.  These are freely
    adjustable illustrative dimensions, not measured sizing values.
    """
    p = ProceduralGarmentParameters() if params is None else params
    p.validate()
    circumference = p.circumference_segments
    angle = 2 * math.pi * np.arange(circumference) / circumference
    cosine, sine = np.cos(angle), np.sin(angle)
    sine[[0, circumference // 2]] = 0
    arc_radius = p.heel_bend_radius_mm
    rings, arc_positions, regions = [], [], []
    straight_leg = p.leg_length_mm - arc_radius
    arc_length = math.pi * arc_radius / 2
    straight_foot = p.foot_length_mm - p.toe_length_mm - arc_radius
    total = straight_leg + arc_length + straight_foot + p.toe_length_mm

    def add_ring(center, normal, width, distance, region, toe_fraction=0.):
        major = width / 2
        points = np.asarray(center)[None, :] + major * cosine[:, None] * np.asarray(normal)[None, :]
        thickness = major * p.flattening * math.cos(toe_fraction * math.pi / 2)
        points[:, 1] += thickness * sine
        points[:, 0] -= p.toe_length_mm * p.toe_rounding_ratio * toe_fraction**2 * cosine**2
        rings.append(points)
        arc_positions.append(distance / total)
        regions.append(region)

    for index in range(p.leg_segments + 1):
        t = index / p.leg_segments
        width = p.leg_width_mm + (p.ankle_width_mm - p.leg_width_mm) * _smoothstep(t)
        add_ring((0, 0, p.leg_length_mm - t * straight_leg), (1, 0, 0), width,
                 t * straight_leg, "leg")
    for index in range(1, p.heel_segments + 1):
        t = index / p.heel_segments
        theta = t * math.pi / 2
        center = (arc_radius * (1 - math.cos(theta)), 0, arc_radius * (1 - math.sin(theta)))
        normal = (math.cos(theta), 0, math.sin(theta))
        width = p.ankle_width_mm + (p.foot_width_mm - p.ankle_width_mm) * _smoothstep(t)
        add_ring(center, normal, width, straight_leg + t * arc_length, "heel")
    for index in range(1, p.foot_segments + 1):
        t = index / p.foot_segments
        add_ring((arc_radius + t * straight_foot, 0, 0), (0, 0, 1), p.foot_width_mm,
                 straight_leg + arc_length + t * straight_foot, "foot")
    for index in range(1, p.toe_segments):
        t = index / p.toe_segments
        add_ring((p.foot_length_mm - p.toe_length_mm + t * p.toe_length_mm, 0, 0), (0, 0, 1),
                 p.foot_width_mm * (1 - p.toe_taper * _smoothstep(t)),
                 straight_leg + arc_length + straight_foot + t * p.toe_length_mm, "toe", t)

    # Exactly C/2+1 distributed physical seam points; each joins two cap halves.
    seam_count = circumference // 2 + 1
    seam_cosine = cosine[:seam_count]
    seam = np.column_stack((p.foot_length_mm - p.toe_length_mm * p.toe_rounding_ratio * seam_cosine**2,
                            np.zeros(seam_count), p.foot_width_mm * (1 - p.toe_taper) / 2 * seam_cosine))
    ring_count = len(rings)
    seam_start = ring_count * circumference
    positions = np.vstack((*rings, seam))
    material_uv = np.vstack((np.array([(k / circumference, v) for v in arc_positions for k in range(circumference)]),
                             np.column_stack((np.arange(seam_count) / circumference, np.ones(seam_count)))))
    triangles, charts = [], []
    distances = arc_positions + [1.]
    for row in range(ring_count):
        for k in range(circumference):
            following = (k + 1) % circumference
            a, b = row * circumference + k, row * circumference + following
            if row + 1 < ring_count:
                d, c = (row + 1) * circumference + k, (row + 1) * circumference + following
            else:
                d = seam_start + min(k, circumference - k)
                c = seam_start + min(following, circumference - following)
            u0, u1, v0, v1 = k / circumference, (k + 1) / circumference, distances[row], distances[row + 1]
            # Ring angle increases as viewed DOWN the cuff.  These faces point
            # out of the shell, including both sides of the final toe seam.
            triangles.extend(((a, b, c), (a, c, d)))
            charts.extend((((u0, v0), (u1, v0), (u1, v1)),
                           ((u0, v0), (u1, v1), (u0, v1))))
    triangles = np.asarray(triangles, dtype=np.int64)
    # Our chosen +XZ normal, +Y binormal frame has inward initial ordering.
    triangles = triangles[:, [0, 2, 1]]
    charts = np.asarray(charts, dtype=float)[:, [0, 2, 1]]
    summary = topology_summary(triangles, len(positions))
    parameter_snapshot = p.to_dict()
    parameter_sha256 = hashlib.sha256(json.dumps(parameter_snapshot, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode("utf-8")).hexdigest()
    metadata = default_garment_metadata(
        "procedural-reference-v1-" + parameter_sha256[:12], "synthetic-uncalibrated-v1",
        reference_role="generated_initial", generator_version="procedural-reference-v1",
        reference_parameter_sha256=parameter_sha256,
        synthetic_uncalibrated=True, stress_free_assumed=False, dressing_equilibrium_solved=False,
        generator="L-shaped centerline elliptical shell sweep with distributed toe seam",
        initial_geometry_only=True, knitting_short_rows_implemented=False,
        parameters=parameter_snapshot, topology_summary=summary,
        cuff_material_point_ids=list(range(circumference)),
        toe_seam_material_point_ids=list(range(seam_start, len(positions))),
        ring_regions=regions, longitudinal_material_coordinate="reference centerline arc fraction",
        material_uv_contract="per-face circumferential seam chart; UV fixed after generation",
        material_point_id_contract="stable at fixed sampling resolution; dimensions never reorder vertices",
        coordinates="XYZ mm; flattened broad plane XZ; separate front/back +/-Y",
        foot_regions="heel bend; +Z instep, -Z sole along straight foot; closed toe seam",
        self_intersection_scope="analytic sweep reach/forward toe constraints; no cloth self-contact solve",
        method_context=[
            {"title": "Stitch Meshes for Modeling Knitted Clothing (2012)",
             "url": "https://www.cs.cornell.edu/projects/stitchmeshes/",
             "role": "independent shell then stitch/yarn construction; not implemented knitting topology"},
            {"title": "Knittable Stitch Meshes (2019)",
             "url": "https://kuiwuchn.github.io/stitchmodeling/",
             "role": "short-row/shift shaping is later work; current sweep is a geometry initialization"},
        ])
    return GarmentMeshPair(positions, positions.copy(), triangles, material_uv,
                           np.arange(len(positions), dtype=np.int64), metadata,
                           face_material_uv=charts)


def export_reference_obj(pair, path):
    """Export physical topology plus independent per-face UV seam corners."""
    pair.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Stocking generated_initial; XYZ units mm; synthetic_uncalibrated\n")
        output.write("# Physical vertex order follows garment_reference.npz material_point_ids\n")
        for point in pair.reference_positions_mm:
            output.write("v " + " ".join(f"{value:.12g}" for value in point) + "\n")
        for uv in pair.triangle_material_uv.reshape(-1, 2):
            output.write("vt " + " ".join(f"{value:.12g}" for value in uv) + "\n")
        for index, face in enumerate(pair.triangles):
            output.write("f " + " ".join(f"{int(vertex)+1}/{index*3+corner+1}" for corner, vertex in enumerate(face)) + "\n")
    return str(path)


def create_reference_scene(pair, params, *, scene_name="Stocking | Generated unworn reference"):
    """Make a separate Blender scene without modifying an existing scene.

    The saved source mesh keeps physical IDs and authoritative per-loop UV.
    Shader wire rendering is a topology preview, not a knitted optical model.
    """
    import bpy
    from mathutils import Vector
    pair.validate()
    scene = bpy.data.scenes.new(scene_name)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = .001
    scene.unit_settings.length_unit = "MILLIMETERS"
    collection = bpy.data.collections.new("Stocking | Procedural reference")
    scene.collection.children.link(collection)
    mesh = bpy.data.meshes.new("Stocking | Unworn X0 shell")
    mesh.from_pydata(pair.reference_positions_mm.tolist(), [], pair.triangles.tolist())
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="MaterialUV")
    uv_layer.data.foreach_set("uv", pair.triangle_material_uv.ravel())
    attribute = mesh.attributes.new("material_point_id", "INT", "POINT")
    attribute.data.foreach_set("value", pair.material_point_ids)
    obj = bpy.data.objects.new("Stocking | Unworn X0", mesh)
    collection.objects.link(obj)
    obj["reference_role"] = "generated_initial"
    obj["synthetic_uncalibrated"] = True
    obj["length_unit"] = "mm"
    obj["parameters"] = json.dumps(params.to_dict())
    obj["garment_metadata"] = json.dumps(pair.metadata)
    mat = bpy.data.materials.new("Stocking | Shell topology preview")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = .7
    wire = mat.node_tree.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = .55
    mix = mat.node_tree.nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = (.52, .70, .76, 1.)
    mix.inputs[2].default_value = (.045, .095, .11, 1.)
    mat.node_tree.links.new(wire.outputs[0], mix.inputs[0])
    mat.node_tree.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)
    low, high = pair.reference_positions_mm.min(axis=0), pair.reference_positions_mm.max(axis=0)
    center = Vector(((low + high) / 2).tolist())
    span = float(max(high - low))
    camera_data = bpy.data.cameras.new("Stocking | Reference camera")
    camera = bpy.data.objects.new("Stocking | Reference camera", camera_data)
    collection.objects.link(camera)
    camera.location = center + Vector((-.22 * span, -1.6 * span, .45 * span))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.16
    camera_data.clip_start, camera_data.clip_end = .01, span * 20
    scene.camera = camera
    for label, offset, power in (("Key", (-.7, -1., 1.), 12.), ("Fill", (.8, -.4, -.3), 6.)):
        data = bpy.data.lights.new("Stocking | " + label, "AREA")
        data.energy, data.size = power * span**2, .8 * span
        light = bpy.data.objects.new("Stocking | " + label, data)
        collection.objects.link(light)
        light.location = center + Vector(offset) * span
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("Stocking | Reference world")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs[0].default_value = (.07, .08, .10, 1.)
    world.node_tree.nodes.get("Background").inputs[1].default_value = .4
    scene.world = world
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 900, 1100
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    return scene, obj
