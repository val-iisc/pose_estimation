import numpy as np

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
    
    












