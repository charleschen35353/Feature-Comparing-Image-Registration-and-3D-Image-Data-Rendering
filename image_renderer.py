import sys
import csv
import bpy
import bmesh
import random
import math
from mathutils import Vector, Euler
import os
import string
import inspect
import glob
from bpy_extras.object_utils import world_to_camera_view

save_blend_file=True




def reset_blend():
    #bpy.ops.wm.read_factory_settings()
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.preferences.addon_enable(module='io_import_images_as_planes')
    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.meshes,
            bpy.data.lights,
            bpy.data.images,
            bpy.data.materials
            ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data, do_unlink=True)


def isVisible(mesh, cam):    
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(mesh.data) 
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    cam_pos = cam.location
    # print(cam_direction)
    mat_world = mesh.matrix_world
    ct1 = 0
    ct2 = 0
    for v in bm.verts:
        co_ndc = world_to_camera_view(bpy.context.scene, cam, mat_world @ v.co)
        nm_ndc = cam_direction.angle(v.normal)
        # v1 = v.co - cam_pos
        # nm_ndc = v1.angle(v.normal)
        if (co_ndc.x < 0.00 or co_ndc.x > 1.00 or co_ndc.y < 0.00 or co_ndc.y > 1.00):
            bm.free()
            print('out of view')
            return False
        # normal may be in two directions
        if nm_ndc < math.radians(120):
            ct1 += 1
        if nm_ndc > math.radians(60):
            ct2 += 1
    if False and min(ct1, ct2) / 10000. > 0.03:
        bm.free()
        print('ct1: {}, ct2: {}\n'.format(ct1, ct2))
        return False
    bm.free()
    return True

def select_object(ob):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    ob.select_set(state = True)
    bpy.context.view_layer.objects.active = ob


def prepare_scene():
    reset_blend()
    scene=bpy.data.scenes['Scene']
    scene.render.engine='CYCLES'
    scene.cycles.samples=128
    scene.cycles.use_square_samples=False    
    scene.display_settings.display_device='sRGB'
    if random.random() > 0.5:
        bpy.data.scenes['Scene'].view_settings.view_transform='Filmic'
    else:
        bpy.data.scenes['Scene'].view_settings.view_transform='Standard'


def prepare_rendersettings():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.scenes['Scene'].cycles.device='GPU'
    bpy.data.scenes['Scene'].render.resolution_x=1600	
    bpy.data.scenes['Scene'].render.resolution_y=1920
    bpy.data.scenes['Scene'].render.resolution_percentage=100
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].cycles.use_denoising = True
    
def position_object(mesh_name):
    mesh=bpy.data.objects[mesh_name]
    select_object(mesh)
    mesh.rotation_euler=[0.0,0.0,0.0]
    return mesh

def plane_to_obj(mesh, thickness):
    select_object(mesh)
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = thickness
    return mesh

def random_deform(mesh):
    select_object(mesh)
    bpy.ops.object.editmode_toggle()
    for _ in range(8):
        bpy.ops.mesh.subdivide()
    bpy.ops.object.editmode_toggle()
    bpy.ops.texture.new()
    bpy.data.textures[-1].type = 'CLOUDS'
    bpy.data.textures[-1].name = "cloud"
    modifier = mesh.modifiers.new(name="Displace", type='DISPLACE')
    modifier.texture = bpy.data.textures['cloud']
    modifier.strength = random.uniform(-0.05,0.05)
    bpy.ops.object.modifier_add(type='SMOOTH')
    bpy.context.object.modifiers["Smooth"].factor = 2

    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
    if random.random() > 0.5:
        bpy.context.object.modifiers["SimpleDeform"].angle = random.random()/3
    else:
        bpy.context.object.modifiers["SimpleDeform"].angle = -random.random()/3
    for i in range(2):
        bpy.context.object.modifiers["SimpleDeform"].limits[i] = i + random.random()/3

    return mesh


def add_lighting(hdr = None):
    world=bpy.data.worlds['World']
    world.use_nodes = True
    wnodes=world.node_tree.nodes
    wlinks=world.node_tree.links
    bg_node=wnodes['Background']
    # hdr lighting
    # remove old node
    for node in wnodes:
        if node.type in ['OUTPUT_WORLD', 'BACKGROUND']:
            continue
        else:
            wnodes.remove(node)

    # hdr world lighting
    if hdr is not None:
        texcoord = wnodes.new(type='ShaderNodeTexCoord')
        mapping = wnodes.new(type='ShaderNodeMapping')
        print(inspect.getmembers(bpy, predicate=inspect.ismethod))
        #mapping.rotation_set( 0,0,random.uniform(0, 6.28) )
        wlinks.new(texcoord.outputs[0], mapping.inputs[0])
        envnode=wnodes.new(type='ShaderNodeTexEnvironment')
        wlinks.new(mapping.outputs[0], envnode.inputs[0])
        envnode.image = bpy.data.images.load(hdr)
        bg_node.inputs[1].default_value=random.uniform(0.8 , 1.0)
        wlinks.new(envnode.outputs[0], bg_node.inputs[0])
    else:
        # point light
        bg_node.inputs[1].default_value=1
        d = random.uniform(1, 3)
        w = random.uniform(1, 3)
        h = random.uniform(1, 3)
        litpos = Vector((w, d, h))
        eul = Euler((0, 0, 0), 'XYZ')
        eul.rotate_axis('Z', random.uniform(0, 3.1415))
        eul.rotate_axis('X', random.uniform(math.radians(45), math.radians(135)))
        litpos.rotate(eul)

        bpy.ops.object.add(type='LIGHT', location=litpos)
        lamp = bpy.data.lights[0]
        #lamp.type = "AREA"
        lamp.use_nodes = True
        nodes=lamp.node_tree.nodes
        links=lamp.node_tree.links
        for node in nodes:
            if node.type=='OUTPUT':
                output_node=node
            elif node.type=='EMISSION':
                lamp_node=node
        strngth=random.randint(10,70)
        lamp_node.inputs[1].default_value=strngth
        #Change warmness of light to simulate more natural lighting
        bbody=nodes.new(type='ShaderNodeBlackbody')
        color_temp=random.uniform(2700,9000)
        bbody.inputs[0].default_value=color_temp
        links.new(bbody.outputs[0],lamp_node.inputs[0])


def reset_camera(mesh):
    bpy.ops.object.select_all(action='DESELECT')
    camera=bpy.data.objects['Camera']

    # sample camera config until find a valid one
    id = 0
    vid = False
    
    while not vid:
        # focal length
        bpy.data.cameras['Camera'].lens = random.randint(25, 35)
        # cam position
        d = random.uniform(0.80, 0.95)
        campos = Vector((0, d, 0))
        eul = Euler((0, 0, 0), 'XYZ')
        eul.rotate_axis('Z', random.uniform(0, 3.1415))
        eul.rotate_axis('X', random.uniform(math.radians(60), math.radians(120)))
    
        campos.rotate(eul)
        camera.location=campos

        # look at pos

        st = (d - 0.8) / 1.0 * 0.2 +0.12
        lookat = Vector((random.uniform(-st, st), random.uniform(-st, st), 0))
        eul = Euler((0, 0., 0), 'XYZ')
        
        eul.rotate_axis('X', math.atan2(lookat.y - campos.y, campos.z))
        eul.rotate_axis('Y', math.atan2(campos.x - lookat.x, campos.z))
        st = (d - 0.8) / 1.0 * 15 + 2.
        eul.rotate_axis('Z', random.uniform(math.radians(-st), math.radians(st)))
        
        camera.rotation_euler = eul
        bpy.context.view_layer.update()
    
        if isVisible(mesh, camera):
            vid = True

def page_texturing(mesh_name, texpath):
    bpy.data.objects[mesh_name]
    select_object(mesh_name)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.material_slot_add()
    bpy.data.materials.new('Material.001')
    mesh.material_slots[0].material=bpy.data.materials['Material.001']
    mat = bpy.data.materials['Material.001']
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # clear default nodes
    for n in nodes:
        nodes.remove(n)
    out_node=nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node=nodes.new(type='ShaderNodeBsdfDiffuse')
    texture_node=nodes.new(type='ShaderNodeTexImage')
    
    texture_node.image=bpy.data.images.load(texpath)

    links=mat.node_tree.links
    links.new(bsdf_node.outputs[0],out_node.inputs[0])
    links.new(texture_node.outputs[0],bsdf_node.inputs[0])

    bsdf_node.inputs[0].show_expanded=True
    texture_node.extension='EXTEND'
    texturecoord_node=nodes.new(type='ShaderNodeTexCoord')
    links.new(texture_node.inputs[0],texturecoord_node.outputs[2])
            

def color_wc_material(obj,mat_name):
    # Remove lamp
    for lamp in bpy.data.lights:
        bpy.data.lights.remove(lamp, do_unlink=True)

    select_object(obj)
    # Add a new material
    bpy.data.materials.new(mat_name)
    obj.material_slots[0].material=bpy.data.materials[mat_name]
    mat=bpy.data.materials[mat_name]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Add an material output node
    mat_node=nodes.new(type='ShaderNodeOutputMaterial')
    # Add an emission node
    em_node=nodes.new(type='ShaderNodeEmission')
    # Add a geometry node
    geo_node=nodes.new(type='ShaderNodeNewGeometry')
    
    # Connect each other
    tree=mat.node_tree
    links=tree.links
    links.new(geo_node.outputs[0],em_node.inputs[0])
    links.new(em_node.outputs[0],mat_node.inputs[0])


def get_worldcoord_img(img_name):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    file_output_node_0 = tree.nodes.new("CompositorNodeOutputFile")
    file_output_node_0.format.file_format = 'OPEN_EXR'
    file_output_node_0.base_path = wc_output_path
    file_output_node_0.file_slots[0].path = img_name

    links.new(render_layers.outputs[0], file_output_node_0.inputs[0])

def prepare_no_env_render():
    # Remove lamp
    for lamp in bpy.data.lights:
        bpy.data.lights.remove(lamp, do_unlink=True)

    world=bpy.data.worlds['World']
    world.use_nodes = True
    links = world.node_tree.links
    # clear default nodes
    for l in links:
        links.remove(l)

    scene=bpy.data.scenes['Scene']
    scene.cycles.samples=1
    scene.cycles.use_square_samples=True
    scene.view_settings.view_transform='Standard'


def render_pass(obj, objpath, texpath):
    # change output image name to obj file name + texture name + random three
    # characters (upper lower alphabet and digits)
    # save_blend_file


    fn = objpath.split('/')[-1][:-4] + '-' + texpath.split('/')[-1][:-4] + '-' + \
        ''.join(random.sample(string.ascii_letters + string.digits, 3))

    if save_blend_file:
        bpy.ops.wm.save_mainfile(filepath=blends_output_path+fn+'.blend')

    scene=bpy.data.scenes['Scene']
    bpy.context.view_layer.use_pass_uv = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    file_output_node_img = tree.nodes.new('CompositorNodeOutputFile')
    file_output_node_img.format.file_format = 'PNG'
    file_output_node_img.base_path = img_output_path
    file_output_node_img.file_slots[0].path = fn
    imglk = links.new(render_layers.outputs[0], file_output_node_img.inputs[0])
    scene.cycles.samples=128
    bpy.ops.render.render(write_still=False)

    # prepare to render without environment
    prepare_no_env_render()

    # remove img link
    links.remove(imglk)

    # render 
    file_output_node_uv = tree.nodes.new('CompositorNodeOutputFile')
    file_output_node_uv.format.file_format = 'OPEN_EXR'
    file_output_node_uv.base_path = uv_output_path
    file_output_node_uv.file_slots[0].path = fn
    uvlk = links.new(render_layers.outputs[4], file_output_node_uv.inputs[0])
    scene.cycles.samples = 1
    bpy.ops.render.render(write_still=False)

    # render world coordinates
    color_wc_material(obj,'wcColor')
    get_worldcoord_img(fn)
    bpy.ops.render.render(write_still=False)

    return fn

def render_img(img_path, tex_dir, hdr_dir):
    prepare_scene()
    prepare_rendersettings()
    img_name = img_path.split("/")[-1]
    img_dir = img_path[:len(img_path) - len(img_name)-1]
    bpy.ops.import_image.to_plane(files=[{"name":img_name, "name":img_name}], directory = img_dir , relative=False)
    doc_name=bpy.data.meshes[0].name
    doc = position_object(doc_name)
    doc = plane_to_obj(doc,0.0001)
    doc = random_deform(doc)
    
    tex = []
    tex_path = os.path.join(tex_dir, random.choice(os.listdir(tex_dir)))
    for _, _, f in os.walk(tex_path):
        tex = f
    tex_name = ""
    tex_norm = ""
    tex_rough = ""
    tex_disp = ""
    for t in tex:
        if tex_name == "" and "diff" in t:
            tex_name = t 
            break
        elif tex_norm == "" and "nor" in t:
            tex_norm = t 
        elif tex_rough == "" and "rou" in t:
            tex_rough = t 
        elif tex_disp == "" and "disp" in t:
            tex_disp = t     
    bpy.ops.import_image.to_plane(files=[{"name":tex_name, "name":tex_name}], directory = tex_path , relative=False)
    #mat = bpy.data.materials.new('mat')
    #mat.use_nodes = True
    #matnodes = mat.node_tree.nodes
    #bpy.data.images.load(filepath=os.path.join(tex_path, tex_norm))
    table_name = bpy.data.meshes[1].name
    table = position_object(table_name)
    select_object(table)
    bpy.ops.transform.resize(value=(4,4,4), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.transform.translate(value=(-0, -0,-0.15), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    
    hdr = hdr_dir + "/" + random.choice(os.listdir(hdr_dir))
    add_lighting(hdr = hdr)
    v = reset_camera(doc)
    if not TEST:
        render_pass(doc, img_path, tex_dir +"/" + tex_name)

tex_dir = "/home/charles/astri/image_generation/texs"
hdr_dir = "/home/charles/astri/image_generation/hdrs"


TEST = False
if TEST:
    img = "/home/charles/astri/image_generation/sns/00040534.png"
    render_img(img, tex_dir, hdr_dir) 

else:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    img_input_path = argv[0]
    output_folder = argv[1]

    img_output_path ='./img/{}/'.format(output_folder)
    uv_output_path = './uv/{}/'.format(output_folder)
    wc_output_path = './wc/{}/'.format(output_folder)
    if save_blend_file:
        blends_output_path ='./bld/{}/'.format(output_folder)

    for fd in [img_output_path, uv_output_path, wc_output_path, blends_output_path]:
        if not os.path.exists(fd):
            os.makedirs(fd)
    count = 1
    for img in sorted(glob.glob(img_input_path)):
        print("Rendering in progress......{} out of {}".format(count, len(sorted(glob.glob(img_input_path)))))
        render_img(img, tex_dir, hdr_dir)   
        count += 1




