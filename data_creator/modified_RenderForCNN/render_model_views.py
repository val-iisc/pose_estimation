#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
RENDER_MODEL_VIEWS.py
brief:
	render projections of a 3D model from viewpoints specified by an input parameter file
usage:
	blender blank.blend --background --python render_model_views.py -- <shape_obj_filename> <shape_category_synset> <shape_model_md5> <shape_view_param_file> <syn_img_output_folder>

inputs:
       <shape_obj_filename>: .obj file of the 3D shape model
       <shape_category_synset>: synset string like '03001627' (chairs)
       <shape_model_md5>: md5 (as an ID) of the 3D shape model
       <shape_view_params_file>: txt file - each line is '<azimith angle> <elevation angle> <in-plane rotation angle> <distance>'
       <syn_img_output_folder>: output folder path for rendered images of this model

author: hao su, charles r. qi, yangyan li

modified by: Aditya Ganeshan
'''

import os
import bpy
import sys
import math
import random
import numpy as np

# Load rendering light parameters
sys.path.insert(0,'/home/babu/126/pose_estimation/data_creator/modified_RenderForCNN')
from global_variables import *


def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def get_view_params(number_images):
    azimuth = np.random.uniform(low = 0.0,high = 360.0,size = (number_images))
    elevation = np.random.normal(10,10,size = (number_images))
    tilt = np.random.normal(0,5, size = (number_images))
    rad = np.ones(shape = (number_images))*3
    view_params = np.stack([azimuth,elevation,tilt,rad],1)
    return view_params


# Input parameters
shape_file = sys.argv[-5]
shape_synset = sys.argv[-4]
shape_md5 = sys.argv[-3]
number_images = int(sys.argv[-2])
syn_images_folder = sys.argv[-1]


if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)

view_params = get_view_params(number_images)

if not os.path.exists(syn_images_folder):
    os.makedirs(syn_images_folder)
    
bpy.ops.import_scene.obj(filepath=shape_file) 

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
bpy.context.scene.render.use_shadows = False
#bpy.context.scene.render.use_raytrace = False

bpy.data.objects['Lamp'].data.energy = 1

#m.subsurface_scattering.use = True

camObj = bpy.data.objects['Camera']

# set lights
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True # remove default light
bpy.ops.object.delete()

########################################################################################################
# Set up rendering of depth map:
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')

map = tree.nodes.new(type="CompositorNodeMapValue")
# Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.size = [1]
map.use_min = True
map.min = [0]
map.use_max = True
map.max = [20]
links.new(rl.outputs[2], map.inputs[0])

invert = tree.nodes.new(type="CompositorNodeInvert")
links.new(map.outputs[0], invert.inputs[1])

# The viewer can come in handy for inspecting the results in the GUI
depthViewer = tree.nodes.new(type="CompositorNodeViewer")
links.new(invert.outputs[0], depthViewer.inputs[0])
# Use alpha from input.
links.new(rl.outputs[1], depthViewer.inputs[1])
fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
fileOutput.format.file_format = "OPEN_EXR"

fileOutput.base_path = os.path.join(g_data_folder,"depth")
                                    
#links.new(invert.outputs[0], fileOutput.inputs[0])
links.new(map.outputs[0], fileOutput.inputs[0])

########################################################################################################

# clear default lights
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete(use_global=False)

# set environment lighting
#bpy.context.space_data.context = 'WORLD'
bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = 100
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'


light_azimuth_deg = [0,120,240,60,180,300]
light_elevation_deg  = [0,0,0,45,45,45]
light_dist = 4
for i in range(6):
    lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg[i], light_elevation_deg[i])
    bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
    bpy.data.objects['Point'].data.energy = 2
    print(dir(bpy.data.objects['Point'].data))
        
for iter in range(view_params.shape[0]):
    azimuth_deg = view_params[iter,0]
    elevation_deg = view_params[iter,1]
    theta_deg = view_params[iter,2]
    rho = view_params[iter,3]


    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    camObj.location[0] = cx
    camObj.location[1] = cy 
    camObj.location[2] = cz
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    # ** multiply tilt by -1 to match pascal3d annotations **
    theta_deg = (-1*theta_deg)%360
    syn_image_file = '%s_%s_a%3.3f_e%3.3f_t%3.3f_d%3.3f' % (shape_synset, shape_md5, azimuth_deg, elevation_deg, theta_deg, rho)
    syn_image_file = syn_image_file.replace('.','@')
    syn_image_file = syn_image_file + '.png'
    bpy.data.scenes['Scene'].render.filepath = os.path.join(syn_images_folder, syn_image_file)
    
    ########################################################################################################
    # Set up rendering of depth map name:
    fileOutput.file_slots[0].path = syn_image_file + '#'
    ########################################################################################################
    bpy.ops.render.render( write_still=True )
