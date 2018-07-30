# Functions for projecting sampled points to rendered image.
import numpy as np

# function to map the points to the rendered locations
# given the world matrix and camera matrix
def get_render_location(pt_array,camera_matrix):
    pt = np.insert(np.transpose(pt_array), 3, 1, axis=0)
    pt_render = np.matmul(camera_matrix,pt)
    z = np.copy(pt_render[2,:])
    pt_render = pt_render/pt_render[2,:]
    pt_render = np.round(pt_render)
    pt_render[2,:] = z
    return pt_render

# Function for cropping visibility
# crop parameter and occulsion based.
# crop_params are [top,bottom,left,right]    
def crop_visibility_array(pt_render,crop_params,resize=True):
    truth_y = (pt_render[1,:]>crop_params[0])*(pt_render[1,:] <crop_params[1])
    truth_x = (pt_render[0,:]>crop_params[2])*(pt_render[0,:] <crop_params[3])
    truth_global = truth_y*truth_x
    # pt_render = pt_render[:,truth_global]
    pt_render[:2,:] = pt_render[:2,:] - np.array([[crop_params[2]],[crop_params[0]]])
    if resize:
        y_len = crop_params[1]-crop_params[0]
        x_len = crop_params[3]-crop_params[2]
        if y_len>=x_len:
            ratio = 224/float(y_len)
            pt_render[:2,:] = pt_render[:2,:]*ratio
            pad_x = (224-x_len*ratio)/2
            pt_render[0,:] +=pad_x
        else:
            ratio = 224/float(x_len)
            pt_render[:2,:] = pt_render[:2,:]*ratio
            pad_y = (224-y_len*ratio)/2
            pt_render[1,:] +=pad_y
        pt_render[:2,:] = np.round(pt_render[:2,:])
    return truth_global, pt_render


# function for depth visibility
# get depth map from io_utils.get_depthmap_from_exr(filename)
def depth_visibility_array(pt_render,depth_map,error_relaxation=0.1):
    truth = np.array([0,]*pt_render.shape[1],dtype=float)
    for i in range(pt_render.shape[1]):
        y,x = (int(pt_render[1,i]),int(pt_render[0,i]))
        if (y>0 and y<540) and (x>0 and x<950):
            # print(pt_render[2,i],depth_map[y-1:y+2,x-1:x+2])
            #print(pt_render.shape,depth_map.shape)
            truth[i] = abs(pt_render[2,i] -depth_map[y,x])<=error_relaxation
        else:
            truth[i] = False
    truth = truth.astype('bool')
    #pt_render = pt_render[:,truth]
    return truth,pt_render
###########################################################################################################
# function for rotation matrix by yaw, pitch,roll
def rot_three_angles(yaw,pitch,roll):
    ### for matching blender
    pitch=-pitch
    ###
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    rot_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    rot_pitch = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    rot_roll = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    ## yaw then pitch then roll
    rot = np.matmul(rot_yaw,np.matmul(rot_pitch,rot_roll))
    ### for matching blender
    rot = np.transpose(rot)
    rot= np.roll(rot,axis = 0, shift= -1)
    return rot

# function for 3x4 camera matrix 
def ex_camera_matrix(yaw,pitch,roll,rho):
    rot_matrix = rot_three_angles(yaw,pitch,roll)
    translation = -np.transpose(np.array([0,0,rho]))
    matrix = np.zeros((3,4))
    matrix[:,0:3] = rot_matrix
    matrix[:,3] = translation
    ### for matching blender
    matrix[1:,:] = -matrix[1:,:]
    return matrix

def get_camera_matrix(intrinsic,extrinsic,world_matrix):
    return np.matmul(np.matmul(intrinsic,extrinsic),world_matrix)



# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location
    print(T_world2bcam)
    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K*RT, K, RT

# function to get the world matrix in blender
def get_world_matrix():
    obj_name = bpy.data.objects.keys()[0]
    obj = bpy.data.objects[obj_name]
    return np.array(obj.matrix_world)

############################################################################################

    
    
    
