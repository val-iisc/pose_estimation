import os
import numpy as np
WORLD_MATRIX = np.array([[1,0,0,0],
                        [0,0,-1,0],
                        [0,1,0,0],
                        [0,0,0,1]])
INTRINSIC_CAMERA_MATRIX = np.array([[1050,    0,  480],
                                    [   0, 1050,  270],
                                    [   0,    0,    1]])
# Not used
depth_location = '/media/jogendra/data1/project_rendering/clean_content/data/data_depth/syn_images_depth'
# Used
corr_location = '/media/jogendra/data1/jogendra_dataset/SIGGRAPH/Corr/Chair'
# not used
render_file_location = '/media/jogendra/data1/project_rendering/clean_content/data/data_chair_2/syn_images_cropped_bkg_overlaid'
# used
render_corr_location = '/media/jogendra/data1/project_rendering/clean_content/data/Corr_files/new_exp_train/'
