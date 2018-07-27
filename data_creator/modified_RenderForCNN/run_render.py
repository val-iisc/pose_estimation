#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
RENDER_ALL_SHAPES
@brief:
    render all shapes of PASCAL3D 12 rigid object classes
'''

import os
import sys
import socket
from global_variables import *

def load_one_category_shape_list(shape_synset):
    shape_md5 = templates[shape_synset]
    view_num = g_syn_images_num_per_category
    shape_list = [(shape_synset, shape_md5, os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'model.obj'), view_num)]
    return shape_list


def render_one_category_model_views(shape_list):
    
    for shape_synset, shape_md5, shape_file, view_num in shape_list:
        command = '%s %s --background --python %s -- %s %s %s %s %s ' % (g_blender_executable_path, g_blank_blend_file_path, os.path.join(g_render4cnn_root_folder, 'render_model_views.py'), shape_file, shape_synset, shape_md5, view_num, os.path.join(g_syn_images_folder, shape_synset, shape_md5))
        os.system(command)

    

if __name__ == '__main__':
    if not os.path.exists(g_syn_images_folder):
        os.mkdir(g_syn_images_folder) 
    
    for synset in g_shape_synsets:
        synset_dir = os.path.join(g_syn_images_folder,synset)
        if not os.path.exists(synset_dir):
            os.mkdir(synset_dir)
        shape_list = load_one_category_shape_list(synset)
        render_one_category_model_views(shape_list)
        
