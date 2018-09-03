import numpy as np

import cv2

"""
Function to find hard negatives from skeleton
"""
def find_hard_negative_from_positive_samples(keypoints_1, keypoints_2):
    
    #first find hard negatives for the legs
    
    keypoints_1_to_append = np.reshape(np.repeat([keypoints_1[0:10,:]],3,axis=0),[30,2])
    
    #first get the hard negative keypoints separately
    keypoints_2_hard_negative = np.zeros([4,10,2])

    for i in range(4):
        keypoints_2_hard_negative[i,:,:] = keypoints_2[i*10:i*10+10]

    for i in range(1,4):
        keypoints_1_to_append = np.append(keypoints_1_to_append,\
                                          np.reshape(np.repeat([keypoints_1[i*10:i*10+10,:]],3,axis=0),[30,2]),\
                                          axis=0)
        
    array_list = np.arange(4)
    indices_hard_neg = array_list != 0
    keypoints_2_to_append = np.reshape(keypoints_2_hard_negative[indices_hard_neg,:,:],[30,2])
    
    #using the saved hard negative keypoints permute them to the actaul set
    for i in range(1,4):
        indices_hard_neg = array_list != i
        keypoints_2_to_append = np.append(keypoints_2_to_append,\
                                          np.reshape(keypoints_2_hard_negative[indices_hard_neg,:,:],[30,2]),\
                                          axis=0) 
        
    keypoints_1 = np.append(keypoints_1, keypoints_1_to_append, axis=0)
    keypoints_2 = np.append(keypoints_2, keypoints_2_to_append, axis=0)
    
    #find hard negatives for the seat sides
    
    keypoints_1_to_append = np.append(keypoints_1[50:60,:],\
                                      keypoints_1[70:80,:],\
                                      axis=0)

    keypoints_2_to_append = np.append(keypoints_2[70:80,:],\
                                      keypoints_2[50:60,:],\
                                      axis=0)

    keypoints_1 = np.append(keypoints_1, keypoints_1_to_append, axis=0)
    keypoints_2 = np.append(keypoints_2, keypoints_2_to_append, axis=0)                                                  
                                                  
    
    #find hard negatives for the sides of the back rest
    
    keypoints_1_to_append = np.append(keypoints_1[80:90,:],\
                                      keypoints_1[100:110,:],\
                                      axis=0)

    keypoints_2_to_append = np.append(keypoints_2[100:110,:],\
                                      keypoints_2[80:90,:],\
                                      axis=0)
                                                  
    keypoints_1 = np.append(keypoints_1, keypoints_1_to_append, axis=0)
    keypoints_2 = np.append(keypoints_2, keypoints_2_to_append, axis=0)
                                                  
    similarities = np.append(np.ones(130), np.zeros(160))                                              
    return keypoints_1, keypoints_2, similarities  

"""
Function to add random flips to image
"""
def flip_image(image):
    prob = np.random.uniform(0,1)
    success = True
    if prob > 0.5:
        flipped_image = cv2.flip(img, 1).astype('uint8')
        return flipped_image, success
    else:
        return image, success
 

"""
Function to add rgb jitter to image
"""
    
def rgb_jitter(image):
    prob = np.random.uniform(0,1)
    if prob > 0.6:
        probabilities = np.random.uniform(0.8,1.2,3)
        jittered_image = image[:,:,:] * probabilities
        jittered_image = np.clip(jittered_image, 0, 255)
        return jittered_image
    else:
        return image
    
    












