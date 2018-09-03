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
        

def make_disparity(class_name):
    template_model,model_synset = gv.syn_template[class_name]
    model_file = os.path.join(gv.g_shapenet_root_folder,model_synset,template_model,'model.obj')     
    model_images_loc = gv.g_save_file_location_final
    depth_location = gv.depth_loc    
    save_loc = gv.syn_disparity_loc
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    
    # now we need the skeleton samples
    print(model_file)
    try:
        key_3d = run_proj.get_keypoints(model_file)
    except:
        # asdf
        key_3d = np.load('/data1/aditya/clean_content/data/sample_points/'+model_synset+'_'+template_model+'.npy')
    
    image_list = os.listdir(model_images_loc)
    image_list = [x for x in image_list if template_model in x]
    #temp = ['04256520_1fd45c57ab27cb6cea65c47b660136e7_a90@179_e0@236_t0@000_d3@000_199_343_385_575_.jpg']
    for img in image_list:
        file_dict = utils.get_params_from_syn_file(img)
        depth_file = os.path.join(depth_location,utils.img_to_depth_file(img))
        # now get the numpy with the view params
        params = [file_dict['azimuth'], file_dict['elevation'], file_dict['tilt'], file_dict['rho']]
        crop_params = file_dict['crop_params']
        
        key_2d_norm = run_proj.get_3d_to_2d(key_3d, params)
        
        truth_1, key_2d = run_proj.prune_pts(key_2d_norm,depth_file,crop = True, crop_params = crop_params)

        # now get the numpy with the view params + 10
        params[0] +=10
        params[1] +=10
        
        key_2d_more = run_proj.get_3d_to_2d(key_3d, params)
        truth_2, key_2d = run_proj.prune_pts(key_2d_more,depth_file,crop = True, crop_params = crop_params)

        truth = truth_1.astype('bool')
        print(len(truth),key_2d_norm.shape) 
        key_2d_more = np.copy(key_2d_more[:,truth])
        key_2d_norm = np.copy(key_2d_norm[:,truth])
        # Now for proxy optical-flow
        del_azi = key_2d_more -key_2d_norm
        flow = np.zeros((224,224,2))
        # pixel flow spread size
        k = 2
        print(len(truth),key_2d_norm.shape) 
        for i in range(key_2d_norm.shape[1]):
            pt = key_2d_norm[:,i][:2].astype(int)
            flow[pt[1]-k:pt[1]+k,pt[0]-k:pt[0]+k,0] = del_azi[:,i][0]
            flow[pt[1]-k:pt[1]+k,pt[0]-k:pt[0]+k,1] = del_azi[:,i][1]
        print(flow.shape,'Done!')
        np.save(os.path.join(save_loc,img.split('.')[0]),flow)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    
    args = parser.parse_args()
    make_disparity(args.class_name)

if __name__ == '__main__':
    main()
    
