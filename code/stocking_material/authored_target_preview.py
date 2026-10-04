"""Render actual extracted target meshes and save a separate review scene."""
from pathlib import Path
import numpy as np


def create_preview(target, decoration, output, render_dir, *, seamless=False):
    import bpy
    from .fitting_preview import _setup_scene, _set_camera, _lights, _plain_material, _caption
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    # All source arrays have already been exported. This resets only the
    # background Blender process and never writes the loaded source .blend.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    materials = [_plain_material('Authored body target', (.22,.47,.66)),
                 _plain_material('Authored foot target', (.19,.62,.53)),
                 _plain_material('Added ankle connector', (.95,.42,.10)),
                 _plain_material('Separate toe decoration', (.72,.76,.80))]
    if seamless:
        for material in materials[:3]:
            bsdf=material.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Base Color'].default_value=(.26,.35,.42,1.)
            bsdf.inputs['Roughness'].default_value=.8
    scenes = []
    for mode, key in (('unposed','unposed_positions_mm'), ('current','positions_mm')):
        scene = _setup_scene('Authored target | '+mode)
        scene['preview_scope'] = 'Existing authored worn target extraction; not X0, not yarn or dressing equilibrium.'
        mesh = bpy.data.meshes.new('Target | '+mode)
        mesh.from_pydata((target[key]/1000).tolist(), [], target['triangles'].tolist())
        mesh.update()
        for mat in materials[:3]:
            mesh.materials.append(mat)
        mesh.polygons.foreach_set('material_index', target['face_region'])
        mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=bool))
        uv = mesh.uv_layers.new(name='SourceTextureUV_NotMaterialCoordinates')
        uv.data.foreach_set('uv',target['face_uv'].reshape(-1))
        for name, domain, values in [('source_object_index','POINT',target['vertex_source_object']),
                                     ('source_triangle_index','POINT',target['vertex_source_triangle']),
                                     ('target_component','POINT',target['component']),
                                     ('source_object_index_face','FACE',target['face_source_object']),
                                     ('target_region','FACE',target['face_region'])]:
            attr = mesh.attributes.new(name,'INT',domain)
            attr.data.foreach_set('value',values)
        attr = mesh.attributes.new('source_barycentric','FLOAT_VECTOR','POINT')
        attr.data.foreach_set('vector',target['vertex_source_barycentric'].reshape(-1))
        obj = bpy.data.objects.new('Stocking target | '+mode, mesh)
        obj['is_unworn_X0'] = False
        obj['source_unit'] = 'mm; scene vertices converted to metres'
        obj['material_correspondence_to_X0'] = 'not established'
        if seamless:
            obj['ankle_shared_vertex_weld']=True
            obj['local_geometry_repair_band_mm']=15.
            attr=mesh.attributes.new('ankle_repair_weight','FLOAT','POINT')
            attr.data.foreach_set('value',target['correction_weight'])
        scene.collection.objects.link(obj)
        decor_mesh = bpy.data.meshes.new('Separate toe decoration | '+mode)
        decor_mesh.from_pydata((decoration[key]/1000).tolist(),[],decoration['triangles'].tolist())
        decor_mesh.materials.append(materials[3])
        decor_mesh.polygons.foreach_set('use_smooth',np.ones(len(decor_mesh.polygons),dtype=bool))
        decor_obj = bpy.data.objects.new('Decoration excluded from target | '+mode,decor_mesh)
        decor_obj['part_of_computational_target'] = False
        scene.collection.objects.link(decor_obj)
        _lights(scene,target[key])
        if seamless:
            scene.view_settings.exposure=-1.
        direction = (.65,-1.8,.5) if mode == 'unposed' else (-1.2,1.7,.55)
        _set_camera(scene,target[key],direction=direction,size=(1000,1100),margin=1.28)
        _caption(scene,('SEAMLESS TARGET | ' if seamless else 'AUTHORED TARGET | ')+mode.upper(),
                 'SHARED ANKLE VERTICES | LOCAL GEOMETRY REPAIR | NO YARN SOLVE' if seamless
                 else 'BLUE: BODY   GREEN: FOOT   ORANGE: ADDED JOIN | NO YARN SOLVE')
        scene.render.filepath = str(render_dir/(mode+'-overview.png'))
        bpy.ops.render.render(write_still=True,scene=scene.name)
        if mode == 'unposed':
            selected = (target[key][:,0]>0) & (target[key][:,2]<160)
            _set_camera(scene,target[key][selected],direction=(1.0,-1.6,.9),size=(1100,850),margin=1.4)
            _caption(scene,'CONTINUOUS FOOT + ANKLE' if seamless else 'AUTHORED FOOT + ANKLE JOIN',
                     'ANKLE STEP REMOVED | UNIFORM MATERIAL | TOE DETAIL IS SEPARATE' if seamless
                     else 'SOURCE VERTICES PRESERVED | GREY TOE DETAIL IS A SEPARATE OBJECT')
            scene.render.filepath = str(render_dir/'foot-and-ankle.png')
            bpy.ops.render.render(write_still=True,scene=scene.name)
            if seamless:
                center=np.array([64.84,43.13,114.24])
                points=np.array([center+[x,y,z] for x in (-32,32) for y in (-32,32) for z in (-18,18)])
                _set_camera(scene,points,direction=(.5,-1,.2),size=(1100,750),margin=1.2)
                _caption(scene,'ANKLE | CONTINUOUS TARGET','SAME MATERIAL AND LIGHTING | SHARED VERTICES | LOCAL GEOMETRY REPAIR')
                scene.render.filepath=str(render_dir/'ankle-uniform-diagnostic.png')
                bpy.ops.render.render(write_still=True,scene=scene.name)
            _set_camera(scene,target[key],direction=direction,size=(1000,1100),margin=1.28)
            _caption(scene,('SEAMLESS TARGET' if seamless else 'AUTHORED TARGET')+' | UNPOSED',
                     'SAME WORN DESIGN BEFORE ARMATURE | THIS IS NOT UNWORN X0')
        scenes.append(scene)
    if bpy.context.window:
        bpy.context.window.scene = scenes[0]
    # An orthographic overview is the initial saved camera view.
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(output)/'authored_target.blend'))
