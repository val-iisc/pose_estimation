import os
from config import exp_name

#path where logs are to be saved
path_to_log = os.path.join('./', 'summaries', exp_name)

#path for save checkpoints
path_to_saved_model = os.path.join('./', 'saved_models', exp_name)

#path to rendered synthetic images
render_loc = os.path.join('../data/synthetic_data/render_final/')

#path to alpha maps of synthetic images
alph_loc = os.path.join('../data/synthetic_data/alpha/')


loc_map = {'chair': '03001627',
           'sofa': '04256520',
           'bed': '02818832',
           'diningtable': '04379243'}

loc_map_with_model = {'chair': '03001627/13fdf00cde077f562f6f52615fb75fca',
           'sofa': '04256520',
           'bed': '02818832',
           'diningtable': '04379243'}

#path to saved disparity maps
FLOW_PATH = '../data/synthetic_data/disparity/'

#path to saved depth maps
DEPTH_LOCATION = '../data/synthetic_data/depth/'

class_map = {'chair': 10, 'sofa': 14, 'bed': 10, 'diningtable': 8}

#path to installed caffe
caffe_path = '/home/jogendra/project_rendering/caffe_combined/caffe_combined/python'

#path to ucn libs
lib_path = '../ucn_code/lib/'