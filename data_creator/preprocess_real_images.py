'''
Script for cropping images and getting the matched keypoints in various datasets.
'''
import cv2
import scipy.io as sio
import os
import numpy as np
import itertools
import pickle
import argparse
import global_variables as gv
import utils
import random

    
def setup(dataset,class_name,subset = 'pascal'):
    if dataset=='keypoint-5':
        data_dir = gv.keypoint_5_dir
        img_loc = os.path.join(data_dir,class_name,'images')
        save_folder = class_name
        image_list = os.listdir(img_loc)
        get_bbox, get_coords = utils.keypoint_5_fetcher(data_dir,class_name)
        mat_check = False
    elif dataset == 'pascal3D':
        data_dir = gv.pascal3D_dir
        img_loc = os.path.join(data_dir,'Images',class_name+'_'+subset)
        save_folder = class_name+'_'+subset
        image_list = os.listdir(img_loc)
        get_bbox, get_coords = utils.pascal_fetcher(dataset, data_dir,class_name+'_'+subset)
        mat_check = False
    elif dataset == 'ObjectNet3D':
        data_dir = gv.objectnet3D_dir
        img_loc = os.path.join(data_dir,'Images')
        save_folder = class_name
        image_list = os.listdir(img_loc)
        # remove the images whichhave old mat version
        image_list = [x for x in image_list if x not in gv.objnet_poor_mat]
        get_bbox, get_coords = utils.pascal_fetcher(dataset, data_dir,class_name)
        mat_check = True
    img_save_dir = os.path.join(gv.data_dir,'real_data',dataset,save_folder,'images')
    keypoint_save_dir = os.path.join(gv.data_dir,'real_data',dataset,save_folder,'sparse_keypoints')

    return data_dir,img_loc,img_save_dir, keypoint_save_dir,image_list,get_bbox,get_coords,mat_check

def make_crops(dataset,class_name,subset):
    data_dir, img_loc, img_save_dir,keypoint_save_dir,img_list, get_bbox, get_coords, mat_check = setup(dataset,class_name,subset)
    # random.shuffle(img_list)
    # check existence of save_dir
    # use setup.sh make_save_dir
    save_name = '_'.join([dataset,class_name,subset])
    info_file = open(os.path.join(gv.data_dir,'image_lists','bbox',save_name+'.txt'),'w')
    # already = os.listdir('../data/preprocessed_data/ObjectNet3D/bed/images/')
    # already = set(['_'.join(x.split('_')[:2]) for x in already])

    for iter,img_name in enumerate(img_list):
        if iter%100 == 0:
            print('cur iter',iter,'of',len(img_list))
        bboxes = get_bbox(img_name)
        coords = get_coords(img_name)
        img = cv2.imread(os.path.join(img_loc,img_name))
        for jter,bbox in enumerate(bboxes):
            cur_coords = np.copy(coords[jter])
            bbox[1] = np.clip(bbox[1],0,img.shape[0])
            bbox[3] = np.clip(bbox[3],0,img.shape[0])
            bbox[0] = np.clip(bbox[0],0,img.shape[1])
            bbox[2] = np.clip(bbox[2],0,img.shape[1])
            new_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            cur_coords[:,1] -= bbox[1] #+10
            cur_coords[:,0] -= bbox[0]#+10

            new_h,new_w,_ = new_img.shape
            if new_h>new_w:
                ratio = 224/float(new_h)
            else:
                ratio = 224/float(new_w)
            # Now cv2 resize according to ratio #for coords too
            # print(ratio,new_h,new_w)
            # print(bbox)
            new_img = cv2.resize(new_img, (int(ratio*new_w), int(ratio*new_h)), interpolation = cv2.INTER_CUBIC)
            # Resize the coords too
            cur_coords[:,:2] = ratio*cur_coords[:,:2]
            # Now pad the image
            new_h,new_w,_ = new_img.shape
            pad_h = (224-new_h)
            pad_h_start = pad_h//2
            pad_h_stop = pad_h - pad_h_start
            pad_w = (224-new_w)
            pad_w_start = pad_w//2
            pad_w_stop = pad_w - pad_w_start
            new_img= cv2.copyMakeBorder(new_img,pad_h_start,pad_h_stop,pad_w_start,pad_w_stop,cv2.BORDER_CONSTANT,value=0)
            # add the effect of pad on the coords
            cur_coords[:,1] += pad_h_start
            cur_coords[:,0] +=pad_w_start
            save_name = img_name.split('.')[0] + '_' + str(jter)

            # save all
            info_file.write(save_name+'\n')
            np.save(os.path.join(keypoint_save_dir, save_name + '.npy'),cur_coords)
            cv2.imwrite(os.path.join(img_save_dir, save_name + '.png'),new_img)
    info_file.close()
    
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='keypoint-5', help='The dataset for processing.')
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    parser.add_argument('--subset', default='pascal', help='Only for Pascal Dataset.')
    
    args = parser.parse_args()
    make_crops(args.dataset, args.class_name,args.subset)

if __name__ == '__main__':
    main()
    
