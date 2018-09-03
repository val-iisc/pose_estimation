import os
import numpy as np
import pickle
import cv2  
import itertools
import pickle
import argparse
import utils
import global_variables as gv
import make_dense_pointset as mdp
import sys
import utils
sys.path.insert(0,'render_wt_pt_proj')
import run_proj

def make_dense_keypoints(class_name):
    template_model,model_synset = gv.syn_template[class_name]
    model_location = os.path.join(gv.g_shapenet_root_folder,model_synset,template_model,'model.obj')    
    model_coords = np.loadtxt(os.path.join(gv.anchor_loc,template_model+'.txt'),delimiter=' ').transpose()    
    model_images_loc = gv.g_save_file_location_final
    depth_location = gv.depth_loc    
    save_loc = gv.syn_dense_keypoint_loc
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    
    # now we need the skeleton samples
    model_coords = utils.get_additional_coords_keypoint_5(model_coords,class_name,dim = 3)
    
    # now the skeleton
    skeleton,skele_info,seat_info = gv.skeleton_map[class_name]

    #k = K[class_name]
    all_coords = np.zeros([3,0])
    for iter,(i,j) in enumerate(skeleton):
        print(i,j)
        if skele_info[iter] =='O':
            all_coords = np.concatenate([all_coords,np.zeros((3,10))],1)
        else:
            new_coords = utils.get_3d_keypoints(model_coords[:,i],model_coords[:,j],10)
            all_coords = np.concatenate([all_coords,new_coords],1)
    print(all_coords.shape)
    # for i in range(model_coords.shape[1]):
    #     new_coords = model_coords[:,i:i+1]
    #     #  print new_coords.shape
    #     all_coords = np.concatenate([all_coords,new_coords],1)
    # Now, we need to project these 3D keypoints for each render.
    # For this, we use functions from render_wt_pt_proj
    image_list = os.listdir(model_images_loc)
    image_list = [x for x in image_list if template_model in x]
    #temp = ['04256520_1fd45c57ab27cb6cea65c47b660136e7_a90@179_e0@236_t0@000_d3@000_199_343_385_575_.jpg']
    for img in image_list:
        file_dict = utils.get_params_from_syn_file(img)
        print(file_dict)
        points_array = np.copy(all_coords.transpose())
        # now project the key_3d based on the params
        params = [file_dict['azimuth'], file_dict['elevation'], file_dict['tilt'], file_dict['rho']]
        key_2d = run_proj.get_3d_to_2d(points_array, params)
        print('2D projections of the points created.')
        print('No of keypoints',key_2d.shape)
        print('Pruning the projected points based on depth and cropping.')
        depth_file = os.path.join(depth_location,utils.img_to_depth_file(img))
        crop_params = file_dict['crop_params']
        truth, key_2d = run_proj.prune_pts(key_2d, depth_file,crop = True, crop_params = crop_params)

        for i in range(key_2d.shape[1]):
            if not truth[i]: key_2d[:,i] = np.zeros((3))
        np.save(os.path.join(save_loc,img.split('.')[0]),key_2d[:2,:])
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    
    args = parser.parse_args()
    make_dense_keypoints(args.class_name)

if __name__ == '__main__':
    main()
    
