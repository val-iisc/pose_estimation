'''
process_images.py

File for cropping the rendered images and overlaying some background on them.
'''

import os 
import sys
import cv2
import random
import numpy as np
from global_variables import *


if __name__ == '__main__':
    if not os.path.exists(g_save_file_location_final):
        os.mkdir(g_save_file_location_final)
    if not os.path.exists(g_save_alpha_location_final):
        os.mkdir(g_save_alpha_location_final)
    # for all the files in g_save_file_location, crop, then overlay, then save in g_save_file_location_final
    for synset in g_shape_synsets:
        render_loc = os.path.join(g_syn_images_folder,synset,templates[synset]) 
        all_img = os.listdir(render_loc)
        save_loc = os.path.join(g_save_file_location_final,synset,templates[synset])
        new_bg = np.zeros((g_image_size,g_image_size,3))
        # g_save_file_location_synset = os.path.join(g_save_file_location_final,synset)
        # if not os.path.exists(g_save_file_location_synset):
        #     os.mkdir(g_save_file_location_synset)
        
        #all_img = [os.path.join(g_save_file_location,x) for x in all_img]
        for iter,img_name in enumerate(all_img):
            print(img_name)
            if iter%100 ==0: print('cur iter', iter,'from', len(all_img) )
            img = cv2.imread(os.path.join(render_loc,img_name) , cv2.IMREAD_UNCHANGED) # to read 4 channel
            im = img[:,:,3] # now its 0,1
            im_0 = np.sum(im, axis = 1)
            im_0 = np.cumsum(im_0)
            im_1 = np.sum(im,axis = 0)
            im_1 = np.cumsum(im_1)
            ax_0_start = np.max(np.where(im_0 == im_0[0]))
            ax_0_stop = np.min(np.where(im_0 == im_0[-1]))
            ax_1_start = np.max(np.where(im_1 == im_1[0]))
            ax_1_stop = np.min(np.where(im_1 == im_1[-1]))

            leftnew = np.clip(ax_1_start,0,im.shape[1])
            rightnew = np.clip(ax_1_stop,0,im.shape[1])
            if leftnew > rightnew: leftnew,rightnew = ax_1_start,ax_1_stop

            topnew = np.clip(ax_0_start,0,im.shape[0])
            bottomnew = np.clip(ax_0_stop,0,im.shape[0])
            if topnew > bottomnew: topnew,bottomnew = ax_0_start,ax_0_stop

            left = int(leftnew)
            right = int(rightnew)
            top = int(topnew)
            bottom = int(bottomnew)
            new_img = img[top:bottom, left:right, :]
            new_h,new_w,_ = new_img.shape


            alpha = np.ceil(new_img[:,:,3]/255.0)
            alpha = np.stack([alpha,]*3,2) 
            seg_map = np.copy(alpha)

            if new_h>new_w:
                ratio = g_image_size/float(new_h)
            else:
                ratio = g_image_size/float(new_w)
            # Now cv2 resize according to ratio #for coords too
            new_img = cv2.resize(new_img, (int(ratio*new_w), int(ratio*new_h)), interpolation = cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (int(ratio*new_w), int(ratio*new_h)), interpolation = cv2.INTER_NEAREST)
            # Now pad the image
            new_h,new_w,_ = new_img.shape
            pad_h = (g_image_size-new_h)
            pad_h_start = pad_h//2
            pad_h_stop = pad_h - pad_h_start
            pad_w = (g_image_size-new_w) 
            pad_w_start = pad_w//2
            pad_w_stop = pad_w - pad_w_start
            new_img= cv2.copyMakeBorder(new_img,pad_h_start,pad_h_stop,pad_w_start,pad_w_stop,cv2.BORDER_CONSTANT,value=0)
            alpha= cv2.copyMakeBorder(alpha,pad_h_start,pad_h_stop,pad_w_start,pad_w_stop,cv2.BORDER_CONSTANT,value=0)
            # now, we need to overlay background

            # we should overlay background berfore resize and padding ? No

            new_h,new_w,_ = new_img.shape
            new_img = new_img[:,:,:3]
            new_img  = new_img*alpha + new_bg*(1-alpha)
            final_image_name = '_'.join([img_name.split('.')[0],str(top),str(bottom),str(left),str(right),'.'+img_name.split('.')[1]])


            # now we can save the image
            # print(final_image_name)
            alpha = alpha[:,:,0]
            cv2.imwrite(os.path.join(g_save_file_location_final,final_image_name), new_img)
            np.save(os.path.join(g_save_alpha_location_final,final_image_name.split('.')[0]), alpha)
        