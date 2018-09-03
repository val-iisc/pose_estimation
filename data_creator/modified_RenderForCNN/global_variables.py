#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import socket
g_root_folder = '/home/babu/126/aditya/pose_estimation/data_creator'
# g_root_folder = '/data1/aditya/pose_estimation/data_creator'

g_render4cnn_root_folder = os.path.abspath(os.path.join(g_root_folder,'modified_RenderForCNN'))
# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
g_blender_executable_path = 'blender' #!! MODIFY if necessary
g_matlab_executable_path = 'matlab' # !! MODIFY if necessary

g_data_folder = os.path.abspath(os.path.join(g_root_folder, '../data','synthetic_data'))
g_datasets_folder = os.path.abspath(os.path.join(g_root_folder,'../data/', 'datasets'))
# g_datasets_folder = '/media/jogendra/data1/jogendra_dataset'
g_truncation_view_folder = os.path.join(g_root_folder,'data/pascal_samples')
g_shapenet_root_folder = os.path.join(g_datasets_folder, 'ShapeNetCore/ShapeNetCore.v1')
# ------------------------------------------------------------
# RENDER FOR CNN PIPELINE
# ------------------------------------------------------------
g_shape_synset_name_pairs = [#('02691156', 'aeroplane'),
                             #('02834778', 'bicycle'),
                             #('02858304', 'boat'),
                             #('02876657', 'bottle'),
                             ('02818832', 'bed'),
                             #('02924116', 'bus'),
                             # ('02958343', 'car'),
                             # ('03001627', 'chair'),
                             ('04379243', 'table'),
                            # ('04379243', 'diningtable'),
                             #('03790512', 'motorbike'),
                             ('04256520', 'sofa'),
                             #('04468005', 'train'),
                             #('03211117', 'tvmonitor')
                                ]
templates = {'03001627':'13fdf00cde077f562f6f52615fb75fca',
             '04256520':'1fd45c57ab27cb6cea65c47b660136e7',
             '02818832':'7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f',
             '04379243':'e41da371550711697062f2d72cde5c95',
             '02958343': '1aeee7288b89ec1cc805dfe4ca9f2fdb'}
g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]
g_syn_images_folder = os.path.join(g_data_folder,'render')
g_save_file_location_final = os.path.join(g_data_folder,'render_final')
g_save_alpha_location_final = os.path.join(g_data_folder,'alpha')
g_image_size = 224
g_syn_cluttered_bkg_ratio = 1.0

g_blank_blend_file_path = os.path.join(g_root_folder,'modified_RenderForCNN/blank.blend') 
g_syn_images_num_per_category = 8000
