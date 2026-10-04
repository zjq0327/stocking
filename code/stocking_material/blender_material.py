"""Single atlas material for surface feature orientation/coverage inspection.

Principled approximation, not the paper BSDF. P offsets remain exported data;
the thin preview shell is not displaced. N/T are transformed by the current
orthonormal chart frame, with no second application of the strain matrix.
"""
from pathlib import Path


def create_surface_feature_material(atlas_metadata):
    import bpy
    material = bpy.data.materials.new("Stocking | Surface feature atlas (approximation)")
    material.use_nodes = True
    material["stocking_scope"] = "M1 geometry/coverage preview; not calibrated optics; P offset not displaced"
    nodes,links = material.node_tree.nodes,material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (.72,.84,.92,1)
    bsdf.inputs["Roughness"].default_value = .36
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(transparent.outputs[0],mix.inputs[1])
    links.new(bsdf.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],output.inputs[0])
    geo = nodes.new("ShaderNodeNewGeometry")
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "stocking_feature_uv"

    def image_output(side,key):
        node = nodes.new("ShaderNodeTexImage")
        node.label = f"{side} {key}"
        node.image = bpy.data.images.load(str(Path(atlas_metadata["image_paths"][f"{side}_{key}"])),check_existing=True)
        node.image.colorspace_settings.name = "Non-Color"
        node.interpolation = "Closest"
        node.extension = "EXTEND"
        links.new(uv.outputs[0],node.inputs[0])
        return node.outputs["Color"]

    def two_sided(key):
        node = nodes.new("ShaderNodeMixRGB")
        node.blend_type = "MIX"
        node.use_clamp = False
        links.new(geo.outputs["Backfacing"],node.inputs[0])
        links.new(image_output("front",key),node.inputs[1])
        links.new(image_output("back",key),node.inputs[2])
        return node.outputs[0]

    links.new(two_sided("valid"),mix.inputs[0])
    frames = []
    for name in ("stocking_course","stocking_wale","stocking_shell_normal"):
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = name
        frames.append(attr.outputs["Vector"])

    def orient(local):
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(local,split.inputs[0])
        terms = []
        for i,axis in enumerate(frames):
            scale = nodes.new("ShaderNodeVectorMath")
            scale.operation = "SCALE"
            links.new(axis,scale.inputs[0])
            links.new(split.outputs[i],scale.inputs["Scale"])
            terms.append(scale.outputs[0])
        for term in terms[1:]:
            add = nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            links.new(terms[0],add.inputs[0])
            links.new(term,add.inputs[1])
            terms[0] = add.outputs[0]
        normalize = nodes.new("ShaderNodeVectorMath")
        normalize.operation = "NORMALIZE"
        transform = nodes.new("ShaderNodeVectorTransform")
        transform.vector_type = "VECTOR"
        transform.convert_from = "OBJECT"
        transform.convert_to = "WORLD"
        # Blender 5.3 adds a hidden LightIndex before Vector. Numeric socket
        # positions are not a stable API; binding index zero loses the frame.
        links.new(terms[0],transform.inputs["Vector"])
        links.new(transform.outputs[0],normalize.inputs[0])
        return normalize.outputs[0]

    world_n = orient(two_sided("N_local"))
    world_t = orient(two_sided("T_local"))
    links.new(world_n,bsdf.inputs["Normal"])
    if "Tangent" in bsdf.inputs:
        links.new(world_t,bsdf.inputs["Tangent"])
        if "Anisotropic IOR Level" in bsdf.inputs:
            bsdf.inputs["Anisotropic IOR Level"].default_value = .25
    material["macro_strain_already_applied"] = True
    material["feature_position_semantics"] = atlas_metadata["position_semantics"]
    material["pose_contract"] = "B1 attributes in object coordinates; rigid or uniform unit-scale object transforms only; new deformation requires rebake"
    return material
