'''
Script for making the train,test and validation subsets of each dataset.
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
    if dataset == 'keypoint-5':
        data_dir = gv.keypoint_5_dir
        get_bbox, get_pose, get_attrs = utils.keypoint_5_fetcher(data_dir,class_name,pose=True)
    elif dataset == 'pascal3D':
        data_dir = gv.pascal3D_dir
        get_bbox, get_pose, get_attrs = utils.pascal_fetcher(dataset, data_dir,class_name+'_'+subset,pose=True)
    elif dataset == 'ObjectNet3D':
        data_dir = gv.objectnet3D_dir
        get_bbox, get_pose, get_attrs = utils.pascal_fetcher(dataset, data_dir,class_name,pose=True)
    save_name = '_'.join([dataset,class_name,subset])
    pose_image_list = open(os.path.join(gv.data_dir,'image_lists','bbox',save_name+'.txt')).readlines()
    pose_image_list = [x.strip() for x in pose_image_list]
    correspondence_image_list = open(os.path.join(gv.data_dir,'image_lists','dense_keypoints',save_name+'.txt')).readlines()
    correspondence_image_list = set([x.strip() for x in correspondence_image_list])

    return data_dir,pose_image_list, correspondence_image_list, get_bbox,get_pose, get_attrs

def get_subsets(dataset,class_name,subset,data_dir):
    save_name = '_'.join([dataset,class_name,subset])
    keys = open(os.path.join(gv.data_dir,'image_lists','bbox',save_name+'.txt')).readlines()
    keys = [x[:-3] for x in keys]

    if dataset == 'keypoint-5':
        test_list = open(os.path.join(data_dir,class_name,'test.txt')).readlines()
        test_list = [x.split('/')[-1].strip() for x in test_list]
    elif dataset == 'pascal3D':
        if subset == 'pascal':
            test_list = open(os.path.join(data_dir,'PASCAL/VOCdevkit/VOC2012/ImageSets/Main/'+ class_name+'_val.txt')).readlines()
            x  = {y[:11]:y.strip()[-2:] for y in test_list}
            test_list = set([y for y in x.keys() if x[y] ==' 1'])
        elif subset == 'imagenet':
            test_list = open(os.path.join(data_dir, 'Image_sets',class_name+'_'+subset+'_val.txt')).readlines()
            test_list = [x.strip() for x in test_list]
    elif dataset == 'ObjectNet3D':
        test_list = open(os.path.join(data_dir,'Image_sets','test.txt'))
        test_list = set([x.strip() for x in test_list])
        
    train_val = [key for key in keys if key not in test_list]
    val_len = min(int(0.15*len(train_val)),100)
    val_list = set(train_val[-val_len:])
    train_list = set(train_val[:-val_len])
    return train_list, val_list, test_list
    
    
def make_subsets(dataset, class_name, subset):
    data_dir, pose_image_list, correspondence_image_list, get_bbox, get_pose, get_attrs = setup(dataset,class_name,subset)
    save_name = '_'.join([dataset,class_name,subset])
    info_file_list = {}
    for im_list_type in ['train','test','val']:
        for im_subset in ['pose','correspondence']:
            info_file_list[im_subset+'_'+im_list_type] = open(os.path.join(gv.data_dir,'image_lists',im_subset,save_name+'_'+im_list_type+'.txt'),'w')
    if not dataset == 'keypoint-5':
        info_file_list_easy = {}
        for im_list_type in ['train_easy','test_easy','val_easy']:
            for im_subset in ['pose','correspondence']:
                info_file_list_easy[im_subset+'_'+im_list_type] = open(os.path.join(gv.data_dir,'image_lists',im_subset,save_name+'_'+im_list_type+'.txt'),'w')
    train_list, val_list, test_list = get_subsets(dataset,class_name,subset,data_dir) 
    print(len(train_list), len(val_list), len(test_list))

    for iter,img_name in enumerate(pose_image_list):
        if iter%100 == 0:
            print('cur iter',iter,'of',len(pose_image_list))
        real_img_name = '_'.join(img_name.split('_')[:-1]) + '.jpg'
        bboxes = get_bbox(real_img_name)
        poses = get_pose(real_img_name)
        attrs = get_attrs(real_img_name)
        img_key = real_img_name.split('.')[0]
        if img_key in test_list:
            img_type = 'test'
        elif img_key in val_list:
            img_type = 'val'
        elif img_key in train_list:
            img_type = 'train'
       
        out_file_correspondence = info_file_list['correspondence_'+img_type]
        
        
        if not dataset == 'keypoint-5':
            out_file_correspondence_easy = info_file_list_easy['correspondence_'+img_type+'_easy']
            out_file_pose = info_file_list['pose_'+img_type]
            out_file_pose_easy = info_file_list_easy['pose_'+img_type+'_easy']

        jter = int(img_name.split('_')[-1])
        print(poses[jter])
        print(attrs[jter])
        pose = poses[jter]
        attr = attrs[jter]
        #save_name = img_name.split('.')[0] + '_' + str(jter)
        print "img_name", pose
        pose_str = ' '.join([img_name, pose[0],pose[1],pose[2]])
        
        # for pose:
        if not dataset == 'keypoint-5':
            out_file_pose.write(pose_str+'\n')
            if attr[0] ==0 and attr[1] ==0 and attr[2] ==0 and not dataset=='keypoint-5':
                out_file_pose_easy.write(pose_str+'\n') 
        # for correspondence:
        if img_name in correspondence_image_list:
            out_file_correspondence.write(pose_str+'\n')
            if not dataset == 'keypoint-5':
                if attr[0] ==0 and attr[1] ==0 and attr[2] ==0:
                    out_file_correspondence_easy.write(pose_str+'\n') 
            
                
    for info_file in info_file_list.values():
        info_file.close()
    if not dataset == 'keypoint-5':
        for info_file in info_file_list_easy.values():
            info_file.close()
    
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pascal3D', help='The dataset for processing.')
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    parser.add_argument('--subset', default='pascal', help='Only for Pascal Dataset.')
    
    args = parser.parse_args()
    make_subsets(args.dataset, args.class_name,args.subset)


if __name__ == '__main__':
    main()
    
