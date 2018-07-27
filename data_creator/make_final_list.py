'''
Script for making the train,test and validation subsets for training.
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

def get_img_loc(cur_file):
    info = cur_file.split('_')
    if info[0] == 'pascal3D':
        img_loc = os.path.join(gv.processed_image_loc,info[0],info[1]+'_'+info[2])
    else:
        img_loc = os.path.join(gv.processed_image_loc,info[0],info[1])
    file_type = info[3]
    return img_loc,file_type

def get_datasets(dataset,task):
    if dataset == 'all' and task == 'pose':
        return ['pascal3D','ObjectNet3D']
    elif dataset == 'all' and task == 'correspondence':
        return ['pascal3D','ObjectNet3D','keypoint-5']
    else:
        return [dataset]
    
def make_final_subset(task,dataset, class_name,easy):
    datasets = get_datasets(dataset,task)
    all_files = os.listdir(os.path.join(gv.data_dir,'image_lists',task))
    all_files = [x for x in all_files if class_name in x]
    all_files = [x for x in all_files if x.split('_')[0] in datasets]
    if easy:
        easy_str = 'easy'
        all_files = [x for x in all_files if 'easy' in x]
    else:
        easy_str = 'hard'
    final_files = {}
    for x in ['train','test','val']:
        final_files[x] = open(os.path.join(gv.data_dir,'image_lists','final_files','_'.join([class_name,task,dataset,easy_str,x])+'.txt'),'w')
    for cur_file in all_files:
        img_loc,file_type = get_img_loc(cur_file)
        info = open(os.path.join(gv.data_dir, 'image_lists', task, cur_file)).readlines()
        for line in info:
            img_name = line.split(' ')[0]
            pose = ' '.join(line.split(' ')[1:])
            save_str = os.path.join(img_loc, img_name) + ' '+pose
            final_files[file_type].write(save_str)
# Now, it may be useful to split the dataset in a different way. For example, you may use the entirity of Imagenet subset of pascal3D. 
# in such a situation, combine the files as shown above, but with a different train/test/val split.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='pose', help='The dataset for processing.')
    parser.add_argument('--datasets', default='all', help='What datasets to consider.')
    parser.add_argument('--class_name', default='chair', help='The dataset the class to processed')
    parser.add_argument('--easy', default='True', help='The dataset the class to processed')
    
    args = parser.parse_args()
    make_final_subset(args.task,args.datasets, args.class_name,bool(args.easy))

if __name__ == '__main__':
    main()
    