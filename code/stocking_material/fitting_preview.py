"""Actual Blender previews of corresponding X0/X1 sock shells.

These renders show geometric placement, fixed material UV and the resulting
kinematic strain. They do not claim cloth equilibrium or yarn/P/N/T shading.
Input coordinates are millimetres; every preview scene uses metres internally.
No source scene, object, material, or external file is modified here.
"""
from pathlib import Path
import json

import numpy as np

from .deformation_field import compute_deformation_field


def _mesh_object(scene, name, positions_mm, triangles, *, pair=None, material=None):
    import bpy
    points = np.asarray(positions_mm, dtype=float) / 1000.
    faces = np.asarray(triangles, dtype=np.int64)
    mesh = bpy.data.meshes.new(name + " | geometry")
    mesh.from_pydata(points.tolist(), [], faces.tolist())
    mesh.update()
    for face in mesh.polygons:
        face.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj["source_length_unit"] = "mm"
    obj["preview_method"] = "geometric shell fitting; no yarn optical material"
    if pair is not None:
        uv = mesh.uv_layers.new(name="MaterialUV")
        uv.data.foreach_set("uv", pair.triangle_material_uv.reshape(-1))
        ids = mesh.attributes.new("material_point_id", "INT", "POINT")
        ids.data.foreach_set("value", pair.material_point_ids)
        obj["garment_metadata"] = json.dumps(pair.metadata, ensure_ascii=False)
        obj["reference_version"] = pair.metadata["reference_version"]
        obj["dressing_equilibrium_solved"] = bool(pair.metadata.get("dressing_equilibrium_solved", False))
    if material is not None:
        mesh.materials.append(material)
    return obj


def _plain_material(name, color, roughness=.7, *, emission=False):
    import bpy
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        emitter = tree.nodes.new("ShaderNodeEmission")
        emitter.inputs["Color"].default_value = (*color[:3], 1.)
        emitter.inputs["Strength"].default_value = 1.
        tree.links.new(emitter.outputs["Emission"], tree.nodes.get("Material Output").inputs["Surface"])
    return material


def _shell_material():
    """One fixed material-chart grid shared by both worn shells and X0."""
    import bpy
    material = _plain_material("Fit preview | fixed material UV grid", (.66, .82, .88))
    tree, nodes = material.node_tree, material.node_tree.nodes
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "MaterialUV"
    separate = nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(uv.outputs["UV"], separate.inputs["Vector"])
    lines = []
    for axis, repeats in (("X", 24.), ("Y", 100.)):
        mult = nodes.new("ShaderNodeMath")
        mult.operation = "MULTIPLY"
        mult.inputs[1].default_value = repeats
        tree.links.new(separate.outputs[axis], mult.inputs[0])
        fraction = nodes.new("ShaderNodeMath")
        fraction.operation = "FRACT"
        tree.links.new(mult.outputs[0], fraction.inputs[0])
        threshold = nodes.new("ShaderNodeMath")
        threshold.operation = "LESS_THAN"
        threshold.inputs[1].default_value = .04
        tree.links.new(fraction.outputs[0], threshold.inputs[0])
        lines.append(threshold)
    combine = nodes.new("ShaderNodeMath")
    combine.operation = "MAXIMUM"
    for index, line in enumerate(lines):
        tree.links.new(line.outputs[0], combine.inputs[index])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = (.72, .86, .91, 1.)
    mix.inputs[2].default_value = (.055, .22, .29, 1.)
    tree.links.new(combine.outputs[0], mix.inputs[0])
    tree.links.new(mix.outputs[0], nodes.get("Principled BSDF").inputs["Base Color"])
    return material


def _setup_scene(name):
    import bpy
    scene = bpy.data.scenes.new(name)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.
    scene.unit_settings.length_unit = "METERS"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new(name + " | world")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs[0].default_value = (.027, .038, .052, 1.)
    world.node_tree.nodes.get("Background").inputs[1].default_value = .5
    scene.world = world
    camera_data = bpy.data.cameras.new(name + " | camera")
    camera_data.type = "ORTHO"
    # Keep ortho_scale the vertical view span for portrait and landscape.
    # Blender AUTO otherwise switches the span to width for landscape renders.
    camera_data.sensor_fit = "VERTICAL"
    camera_data.clip_start, camera_data.clip_end = .001, 100.
    camera = bpy.data.objects.new(name + " | camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene["preview_scope"] = "Independent generated X0 -> geometric X1; material UV remains fixed; no calibrated dressing equilibrium or yarn P/N/T."
    return scene


def _projected_frame(points_m, direction, aspect, margin=1.18):
    from mathutils import Vector
    points = np.asarray(points_m, dtype=float)
    direction = Vector(direction).normalized()
    rotation = (-direction).to_track_quat("-Z", "Y")
    basis = np.array(rotation.to_matrix())
    projected = points @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2.)
    scale = max(float(high[1] - low[1]), float(high[0] - low[0]) / aspect) * margin
    distance = max(float(np.linalg.norm(high - low)) * 3., .3)
    return center, rotation, max(scale, .01), distance, basis


def _set_camera(scene, points_mm, direction=(.65, 1.5, .6), size=(1000, 1200), margin=1.2):
    from mathutils import Vector
    aspect = size[0] / size[1]
    center, rotation, scale, distance, basis = _projected_frame(np.asarray(points_mm) / 1000., direction, aspect, margin)
    camera = scene.camera
    camera.rotation_euler = rotation.to_euler()
    camera.location = Vector(center) + Vector(direction).normalized() * distance
    camera.data.ortho_scale = scale
    scene.render.resolution_x, scene.render.resolution_y = size
    return center, scale, basis


def _lights(scene, points_mm):
    import bpy
    from mathutils import Vector
    points = np.asarray(points_mm) / 1000.
    center = (points.min(axis=0) + points.max(axis=0)) / 2.
    span = max(float(np.ptp(points, axis=0).max()), .15)
    for label, offset, energy in (("Key", (1., 1., 1.5), 150.),
                                  ("Fill", (-1.5, .5, .7), 85.),
                                  ("Rim", (.6, -1., .7), 110.),
                                  ("Sole fill", (.3, .8, -1.2), 65.)):
        data = bpy.data.lights.new(scene.name + " | " + label, "AREA")
        data.energy, data.size = energy * span**2, span
        light = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(light)
        light.location = Vector(center) + Vector(offset) * span
        light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()


def _caption(scene, title, subtitle):
    """Camera-facing physical text, kept in its own removable collection."""
    import bpy
    from mathutils import Vector
    old = bpy.data.collections.get(scene.name + " | captions")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new(scene.name + " | captions")
    scene.collection.children.link(collection)
    camera = scene.camera
    rotation = camera.rotation_euler.to_quaternion()
    height = camera.data.ortho_scale
    width = height * scene.render.resolution_x / scene.render.resolution_y
    text_material = _plain_material("Fit preview | caption emission", (.72, .86, .96), emission=True)
    band_material = _plain_material("Fit preview | caption backdrop", (.006, .011, .018), emission=True)
    for label, y0, y1 in (("Top band", .43 * height, .5 * height),
                           ("Bottom band", -.5 * height, -.44 * height)):
        mesh = bpy.data.meshes.new(scene.name + " | " + label)
        mesh.from_pydata([(-width / 2, y0, 0), (width / 2, y0, 0),
                          (width / 2, y1, 0), (-width / 2, y1, 0)], [], [(0, 1, 2, 3)])
        mesh.materials.append(band_material)
        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)
        obj.rotation_euler = camera.rotation_euler
        obj.location = camera.location + rotation @ Vector((0., 0., -.012))
        obj.visible_shadow = False
    for name, content, size, y in (("Title", title, .026 * height, .465 * height),
                                   ("Subtitle", subtitle, .015 * height, -.473 * height)):
        data = bpy.data.curves.new(scene.name + " | " + name, "FONT")
        data.body, data.size = content, size
        data.align_x, data.align_y = "LEFT", "CENTER"
        data.materials.append(text_material)
        obj = bpy.data.objects.new(data.name, data)
        collection.objects.link(obj)
        obj.rotation_euler = camera.rotation_euler
        obj.location = camera.location + rotation @ Vector((-.46 * width, y, -.01))
        obj.visible_shadow = False


def _pose_caption(pairs):
    poses = {pair.metadata.get("target_pose", "unspecified") for pair in pairs.values()}
    if poses == {"rest"}:
        return ("REFERENCE POSE", "REFERENCE POSE | Fitting verification",
                "Same model, reference pose | Synthetic X0 | Fixed UV | Geometric fitting, uncalibrated")
    if poses == {"current"}:
        return ("ORIGINAL POSE", "ORIGINAL POSE | Contact conflicts remain",
                "Original body has self / cross-leg intersections | Geometric candidate | Contact conflicts unresolved")
    return ("GEOMETRIC FIT", "X1 | Independent stockings fitted to your model",
            "Fixed material UV grid | Geometric verification | Synthetic X0, uncalibrated")


def create_fit_preview(body_positions_mm, body_triangles, pairs_by_side, *, scene_name="Stocking | Geometric fit verification"):
    """Build a fresh scene and return scene plus body/left/right objects.

    ``objects_dict['_pairs']`` retains the input contracts for subsequent
    previews; ``objects_dict['_body_arrays']`` retains world-space mm input.
    Only source arrays are read; all Blender meshes and materials are new.
    """
    if not pairs_by_side:
        raise ValueError("at least one fitted stocking is required")
    for side, pair in pairs_by_side.items():
        if side not in {"left", "right"}:
            raise ValueError("stocking side must be left or right")
        pair.validate()
    scene = _setup_scene(scene_name)
    skin = _plain_material("Fit preview | plain skin collision body", (.48, .32, .25), .78)
    shell = _shell_material()
    objects = {"body": _mesh_object(scene, "Fit | collision body", body_positions_mm, body_triangles, material=skin)}
    for side, pair in pairs_by_side.items():
        objects[side] = _mesh_object(scene, "Fit | " + side + " X1 stocking", pair.current_positions_mm,
                                     pair.triangles, pair=pair, material=shell)
    points = np.vstack([pair.current_positions_mm for pair in pairs_by_side.values()])
    _set_camera(scene, points)
    _lights(scene, points)
    pose_label, title, subtitle = _pose_caption(pairs_by_side)
    scene["target_pose"] = pose_label
    _caption(scene, title, subtitle)
    objects["_pairs"] = dict(pairs_by_side)
    objects["_body_arrays"] = (np.asarray(body_positions_mm), np.asarray(body_triangles))
    objects["_preview_scenes"] = {}
    return scene, objects


def _foot_points(pair):
    regions = pair.metadata.get("ring_regions", [])
    circumference = pair.metadata.get("parameters", {}).get("circumference_segments", 0)
    if regions and circumference and len(regions) * circumference <= len(pair.material_uv):
        mask = np.repeat(np.array([region != "leg" for region in regions]), circumference)
        mask = np.r_[mask, np.ones(len(pair.material_uv) - len(mask), dtype=bool)]
    else:
        mask = pair.material_uv[:, 1] >= .72
    return pair.current_positions_mm[mask]


def _foot_view_directions(pair):
    """Use the corresponding foot rings to find instep and sole directions.

    A fixed world-space 'below' view would be wrong for the raised, rotated
    foot. This derives the local axis and original U=0.5 sole side instead.
    """
    regions = pair.metadata.get("ring_regions", [])
    circumference = pair.metadata.get("parameters", {}).get("circumference_segments", 0)
    rows = np.flatnonzero(np.asarray(regions) == "foot")
    if not circumference or len(rows) < 2:
        return ((.85, 1.3, .65), (.8, 1.2, -1.))
    rings = pair.current_positions_mm[:len(regions) * circumference].reshape(-1, circumference, 3)
    centers = rings[rows].mean(axis=1)
    forward = centers[-1] - centers[0]
    forward /= max(np.linalg.norm(forward), 1e-12)
    sole = (rings[rows, circumference // 2] - centers).mean(axis=0)
    sole -= forward * np.dot(sole, forward)
    sole /= max(np.linalg.norm(sole), 1e-12)
    side = np.cross(forward, sole)
    if np.dot(side, (1., 1., 0.)) < 0:
        side = -side
    instep_view = side * .9 - sole * .6 + forward * .2
    sole_view = side * .8 + sole * .85 - forward * .15
    return tuple(instep_view), tuple(sole_view)


def _strain_material():
    import bpy
    material = _plain_material("Fit preview | measured course stretch", (.3, .5, .8))
    attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "course_stretch_color"
    material.node_tree.links.new(attribute.outputs["Color"], material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    return material


def _strain_color(values, minimum, maximum):
    fraction = np.clip((values - minimum) / max(maximum - minimum, 1e-9), 0., 1.)
    stops = np.array([[.04, .15, .65], [.02, .63, .80], [.18, .76, .42], [.96, .78, .05], [.94, .14, .05]])
    place = fraction * (len(stops) - 1)
    index = np.minimum(place.astype(int), len(stops) - 2)
    weight = place - index
    color = stops[index] * (1 - weight[:, None]) + stops[index + 1] * weight[:, None]
    return np.c_[color, np.ones(len(color))]


def _comparison_scene(objects):
    """Same world scale and fixed UV on displayed X0 and X1."""
    from mathutils import Vector
    scene = _setup_scene("Stocking | X0 to X1 comparison")
    pairs = objects["_pairs"]
    representative = next(iter(pairs.values()))
    direction = (.3, 1.6, .5)
    rotation = (-Vector(direction)).to_track_quat("-Z", "Y")
    basis = np.array(rotation.to_matrix())
    right = basis[:, 0]
    up = basis[:, 1]
    ref = representative.reference_positions_mm.copy()
    worn = np.vstack([pair.current_positions_mm for pair in pairs.values()])
    ref_extent = np.ptp(ref @ basis, axis=0)
    worn_extent = np.ptp(worn @ basis, axis=0)
    separation = .5 * (ref_extent[0] + worn_extent[0]) + 160.
    ref_center = ((ref @ basis).min(axis=0) + (ref @ basis).max(axis=0)) / 2.
    worn_center = ((worn @ basis).min(axis=0) + (worn @ basis).max(axis=0)) / 2.
    shift_ref = right * (-separation / 2. - ref_center[0]) - up * ref_center[1]
    shift_worn = right * (separation / 2. - worn_center[0]) - up * worn_center[1]
    material = objects[next(iter(pairs))].data.materials[0]
    _mesh_object(scene, "Compare | X0 generated reference", ref + shift_ref, representative.triangles,
                 pair=representative, material=material)
    for side, pair in pairs.items():
        _mesh_object(scene, "Compare | " + side + " X1", pair.current_positions_mm + shift_worn,
                     pair.triangles, pair=pair, material=material)
    body_points, body_faces = objects["_body_arrays"]
    _mesh_object(scene, "Compare | body", body_points + shift_worn, body_faces,
                 material=objects["body"].data.materials[0])
    all_points = np.vstack((ref + shift_ref, worn + shift_worn))
    _set_camera(scene, all_points, direction, size=(1600, 1000), margin=1.27)
    _lights(scene, all_points)
    pose_label, _, _ = _pose_caption(pairs)
    comparison_title = "X0 -> X1 | " + pose_label + " | Same physical scale"
    comparison_subtitle = "Left: procedural X0. Right: same material points / triangles / UV on your body."
    if pose_label == "ORIGINAL POSE":
        comparison_subtitle = "Original pose candidate | Body self / cross-leg intersections remain | Geometric mapping only"
    _caption(scene, comparison_title, comparison_subtitle)
    return scene


def _render(scene, path):
    import bpy
    scene.render.filepath = str(path)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True, scene=scene.name)
    return str(path)


def render_fit_previews(scene, objects_dict, output_dir):
    """Render complete limbs, each foot from two views, X0/X1 and strain.

    The returned paths are real PNG renders, not explanatory illustrations.
    Main scene camera/material/visibility is restored after detail/strain.
    Detail body copies are clipped only for presentation to remove the
    opposite limb; they are never used in fitting/contact verification.
    """
    import bpy
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    objects = objects_dict
    pairs = objects["_pairs"]
    points = np.vstack([pair.current_positions_mm for pair in pairs.values()])
    result = {}
    _set_camera(scene, points)
    pose_label, title, subtitle = _pose_caption(pairs)
    _caption(scene, title, subtitle)
    result["overall"] = _render(scene, output / "overall.png")
    visibility = {key: objects[key].hide_render for key in ("body", *pairs)}
    body_points, body_faces = objects["_body_arrays"]
    for side, pair in pairs.items():
        foot = _foot_points(pair)
        low, high = foot.min(axis=0), foot.max(axis=0)
        padding = max(float(np.ptp(foot, axis=0).max()) * .06, 4.)
        centroids = body_points[body_faces].mean(axis=1)
        keep = ((centroids >= low - padding) & (centroids <= high + padding)).all(axis=1)
        body_vertex_side = objects.get("_body_vertex_side")
        if body_vertex_side is not None:
            side_index = 0 if side == "left" else 1
            keep &= np.all(np.asarray(body_vertex_side)[body_faces] == side_index, axis=1)
        local_body = _mesh_object(scene, "Detail | " + side + " presentation body", body_points,
                                  body_faces[keep], material=objects["body"].data.materials[0])
        objects["body"].hide_render = True
        for other_side in pairs:
            objects[other_side].hide_render = other_side != side
        instep_direction, sole_direction = _foot_view_directions(pair)
        for view, direction in (("instep", instep_direction), ("sole-heel", sole_direction)):
            _set_camera(scene, foot, direction, size=(1300, 1000), margin=1.25)
            foot_subtitle = "Independent closed toe + heel shell | Fixed material UV | Geometric fit preview"
            if pose_label == "ORIGINAL POSE":
                foot_subtitle = "Clipped body view | Body self / cross-leg intersections remain | Geometric candidate only"
            _caption(scene, pose_label + " | " + side.upper() + " FOOT | " + view.upper(), foot_subtitle)
            result[side + "_foot_" + view.replace("-", "_")] = _render(scene, output / (side + "-foot-" + view + ".png"))
        bpy.data.objects.remove(local_body, do_unlink=True)
    for key, value in visibility.items():
        objects[key].hide_render = value
    _set_camera(scene, points)
    strain = {side: compute_deformation_field(pair).course_stretch for side, pair in pairs.items()}
    all_stretches = np.concatenate(list(strain.values()))
    minimum, maximum = np.percentile(all_stretches, (2., 98.))
    saved_materials = {}
    strain_material = _strain_material()
    for side, values in strain.items():
        mesh = objects[side].data
        attribute = mesh.color_attributes.get("course_stretch_color")
        if attribute is None:
            attribute = mesh.color_attributes.new(name="course_stretch_color", type="FLOAT_COLOR", domain="CORNER")
        attribute.data.foreach_set("color", np.repeat(_strain_color(values, minimum, maximum), 3, axis=0).reshape(-1))
        scalar = mesh.attributes.get("course_stretch")
        if scalar is None:
            scalar = mesh.attributes.new("course_stretch", "FLOAT", "FACE")
        scalar.data.foreach_set("value", values)
        saved_materials[side] = mesh.materials[0]
        mesh.materials[0] = strain_material
    strain_subtitle = f"Blue {minimum:.2f}x -> red {maximum:.2f}x | 2-98% range, tails clamped | Kinematics only"
    if pose_label == "ORIGINAL POSE":
        strain_subtitle += " | Unresolved contacts"
    _caption(scene, pose_label + " | Per-triangle course stretch", strain_subtitle)
    result["strain"] = _render(scene, output / "strain.png")
    result["strain_color_range"] = {"minimum_stretch": float(minimum), "maximum_stretch": float(maximum),
                                    "percentiles": [2, 98], "tails_clamped": True,
                                    "actual_minimum": float(all_stretches.min()), "actual_maximum": float(all_stretches.max())}
    for side, material in saved_materials.items():
        objects[side].data.materials[0] = material
    contact_samples = objects.get("_contact_samples", {})
    if contact_samples:
        contact_material = _plain_material("Fit preview | sampled body contact conflicts", (.4, .6, .7))
        attribute_node = contact_material.node_tree.nodes.new("ShaderNodeAttribute")
        attribute_node.attribute_name = "contact_conflict_color"
        contact_material.node_tree.links.new(attribute_node.outputs["Color"],
                                              contact_material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
        conflict_counts = {}
        for side, pair in pairs.items():
            samples = contact_samples.get(side)
            if samples is None:
                continue
            kind = samples["sample_kind"]
            indices = samples["sample_source_index"]
            bad = ((samples["signed_distance_mm"] < -.01)
                   | (samples["other_leg_signed_distance_mm"] < -.01))
            vertex_bad = np.zeros(len(pair.material_point_ids), dtype=bool)
            vertex_bad[indices[kind == 0]] = bad[kind == 0]
            face_bad = vertex_bad[pair.triangles].any(axis=1)
            face_bad[indices[kind == 2]] |= bad[kind == 2]
            colors = np.tile((.24, .5, .62, 1.), (len(pair.triangles), 1))
            colors[face_bad] = (.52, .004, .002, 1.)
            mesh = objects[side].data
            attribute = mesh.color_attributes.get("contact_conflict_color")
            if attribute is None:
                attribute = mesh.color_attributes.new(name="contact_conflict_color", type="FLOAT_COLOR", domain="CORNER")
            attribute.data.foreach_set("color", np.repeat(colors, 3, axis=0).reshape(-1))
            mesh.materials[0] = contact_material
            conflict_counts[side] = int(face_bad.sum())
        _set_camera(scene, points, direction=(-.85, 1.3, .4))
        _caption(scene, pose_label + " | Sampled body contact conflicts",
                 "Red: own / other-leg distance < -0.01 mm at vertices or face centers | Continuous collision not tested")
        result["contact_conflicts"] = _render(scene, output / "contact-conflicts.png")
        result["contact_conflict_marked_faces"] = conflict_counts
        for side, material in saved_materials.items():
            objects[side].data.materials[0] = material
    comparison = _comparison_scene(objects)
    objects["_preview_scenes"]["reference_to_worn"] = comparison
    result["reference_to_worn"] = _render(comparison, output / "reference-to-worn.png")
    _set_camera(scene, points)
    _caption(scene, title, subtitle)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    return result
