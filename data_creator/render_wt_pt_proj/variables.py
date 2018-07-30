import os
import numpy as np
WORLD_MATRIX = np.array([[1,0,0,0],
                        [0,0,-1,0],
                        [0,1,0,0],
                        [0,0,0,1]])
INTRINSIC_CAMERA_MATRIX = np.array([[1050,    0,  480],
                                    [   0, 1050,  270],
                                    [   0,    0,    1]])
blender_executable_path = 'blender' #!! MODIFY if necessary
