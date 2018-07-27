import os
import numpy as np
import pickle
import cv2  
import itertools
import pickle
import argparse
import utils
import global_variables as gv

def setup(dataset,class_name,subset = 'pascal'):
    data_dir = os.path.join(gv.processed_image_loc,dataset)
    if dataset=='pascal3D':
        image_loc = os.path.join(data_dir,class_name+'_'+subset,'images')
        sparse_keypoint_loc = os.path.join(data_dir,class_name+'_'+subset,'sparse_keypoints')
        dense_keypoint_loc = os.path.join(data_dir,class_name+'_'+subset,'dense_keypoints')
    else:
        image_loc = os.path.join(data_dir,class_name,'images')
        sparse_keypoint_loc = os.path.join(data_dir,class_name,'sparse_keypoints')
        dense_keypoint_loc = os.path.join(data_dir,class_name,'dense_keypoints')
    if dataset=='keypoint-5':
        keypoint_extender = utils.get_additional_coords_keypoint_5
    else:
        keypoint_extender = utils.get_additional_coords_pascal
    img_list = os.listdir(image_loc)
    return data_dir, image_loc, img_list, sparse_keypoint_loc, dense_keypoint_loc, keypoint_extender
        
def make_dense_keypoints(dataset,class_name,subset):
    info = setup(dataset,class_name,subset)
    data_dir, image_loc, img_list, sparse_keypoint_loc, dense_keypoint_loc, keypoint_extender = info
    
    
    leg_range = gv.leg_range[class_name]
    skeleton,skele_info,seat_info = gv.skeleton_map[class_name]
    #id_x = img_list.index('00000042_left_backward_.jpg')
    save_name = '_'.join([dataset,class_name,subset])
    info_file = open(os.path.join(gv.data_dir,'image_lists','dense_keypoints',save_name+'.txt'),'w')
    for iter,img_name in enumerate(img_list):
        any_skeleton=False
        image = cv2.imread(os.path.join(image_loc,img_name))
        key_name = img_name = img_name.split('.')[0]+'.npy'
        print(image.shape)
        img_coords = np.load(os.path.join(sparse_keypoint_loc,key_name) )
        img_coords = np.clip(img_coords, 0,223).astype(int).T
        
        print(img_coords.shape)
        
        print(img_coords.shape,'the point set')
        img_coords = keypoint_extender(img_coords, class_name)
        print(img_coords.shape,'the point set')
        
        # not all keypoints given in pascal, objectnet.
        if not dataset == 'keypoint-5':
            skele_existence = [0,]*len(skele_info)
            truth_val = img_coords[2,:]
            act_skele_info =np.copy(skele_info)
            for i in range(len(skele_existence)):
                bone = skeleton[i]
                #print(bone,'the bone',bone[0],bone[1])
                if truth_val[bone[0]] == 1 and truth_val[bone[1]]==1:
                    act_skele_info[i] = skele_info[i]
                else:
                    act_skele_info[i] = 'O'
        else:
            act_skele_info = np.copy(skele_info)
                
                
                
        all_coords = np.zeros((2,0))
        for iter,(i,j) in enumerate(skeleton):
            print(i,j)
            if act_skele_info[iter] =='O':
                all_coords = np.concatenate([all_coords,np.zeros((2,10))],1)
            else:
                any_skeleton = True
                new_coords = utils.get_2d_keypoints(img_coords[:,i],img_coords[:,j],10)
                all_coords = np.concatenate([all_coords,new_coords],1)
        print(img_coords.astype(int))
        print('This is it')
        print(act_skele_info)
        print(np.min(all_coords),np.max(all_coords))
        # adjustment:
        #all_coords = adjust(all_coords,class_name)
        ## Pruning Functions
        
        # Now, we need to prune!
        # seat pruning
        # for all legs, remove points inside seat
        # seat 
        
        if any_skeleton:
            all_coords = utils.surface_pruned(all_coords,img_coords,seat_info,leg_range)
            all_coords = utils.dir_pruned(all_coords, img_coords,skeleton,seat_info, skele_info)
            if class_name in ['bed','sofa']:
                all_coords = utils.surface_dir_prune(all_coords, img_coords, skeleton,skele_info,seat_info)
            print('Done')
            #print(all_coords)
            info_file.write(key_name.split('.')[0]+'\n')
            np.save(os.path.join(dense_keypoint_loc,key_name),all_coords)
        else:
            print('No keypoints!')
    info_file.close()
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='keypoint-5', help='The dataset for processing.')
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    parser.add_argument('--subset', default='pascal', help='Only for Pascal Dataset.')
    
    args = parser.parse_args()
    make_dense_keypoints(args.dataset, args.class_name,args.subset)

if __name__ == '__main__':
    main()
    