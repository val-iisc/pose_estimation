
import numpy as np
import OpenEXR
import Imath
import pt2render
import variables
import os
import math

def get_depthmap_from_exr(file_name):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(file_name)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depthmap = np.fromstring(redstr, dtype = np.float32)
    depthmap.shape = (size[1], size[0]) # Numpy arrays are (row, col)
    return depthmap

def get_params_from_file(file_name):
    file_name = file_name.split('/')[-1]
    list_params = file_name.split('_')
    dict_params = {'model_type': list_params[0],
                  'model_md5': list_params[1],
                  'azimuth':float(list_params[2][1:].replace('@','.')), 
                  'elevation':float(list_params[3][1:].replace('@','.')), 
                  'tilt':-float(list_params[4][1:].replace('@','.')),
                  'rho':float(list_params[5][1:].replace('@','.')),
                  'crop_params': [int(list_params[6]),int(list_params[7]),
                                 int(list_params[8]),int(list_params[9])]}
    return dict_params


def get_point_arrays(file_name,model_same):
    file = np.loadtxt(file_name)
    # print('this is file',file)
    if not model_same:
        model_1 = file[:,:3].astype('float')
        model_2 = file[:,3:].astype('float')
        
    else:
        model_1 = file[:,:3].astype('float')
        model_2 = file[:,:3].astype('float')
    return model_1,model_2

    
def find_corr_file(dict_1,dict_2,location = variables.corr_location):
    #print(location)
    all_files = os.listdir(location)
    name_1 = dict_1['model_md5']
    name_2 = dict_2['model_md5']
    if name_1==name_2:
        model_same = True
        possible_list = [s for s in all_files if name_1+"___corr___" in s]
        if len(possible_list)==0:
            possible_list = [s for s in all_files if "___corr___"+name_1 in s]
    else:
        model_same = False
        possible_list = [s for s in all_files if (name_1+"___corr___"+ name_2 + '.pts' in s)]# or
                                                 #name_2+"___corr___"+ name_1 + '.pts' in s )]
    try:
        file_name = possible_list[0]
        return os.path.join(location,file_name),model_same
    except:
        return None,None

def make_render_corr_file(dict_1,dict_2,corr_file,file_name,location=variables.render_corr_location,
                          render_file_names=None,depth_location=[None,None],model_same=False):
    y = get_point_arrays(corr_file,model_same)
    model_1_pts,model_2_pts = y
    truth_1,crop_pt_array_1 =  get_render_array(dict_1,model_1_pts,depth_location[0])
    truth_2,crop_pt_array_2 =  get_render_array(dict_2,model_2_pts,depth_location[1])
    
    global_truth = truth_1*truth_2
    final_file = open(os.path.join(location,file_name),'w')
    #write file_location
    if render_file_names==None:
        final_file.write(get_final_file_name(dict_1)+'\n')
        final_file.write(get_final_file_name(dict_2)+'\n')
    else:
        final_file.write(render_file_names[0]+'\n')
        final_file.write(render_file_names[1]+'\n')
    # write difference 
    diff_list = get_difference(dict_1,dict_2)
    print(diff_list)
    diff_list = map(str, diff_list)
    final_file.write(' '.join(diff_list)+'\n')
    # write pt to pt correspondence
    final_corr_pt_1 = crop_pt_array_1[:,global_truth]
    final_corr_pt_2 = crop_pt_array_2[:,global_truth]
    for i in range(final_corr_pt_1.shape[1]):
        pt_1 = ' '.join(map(str,final_corr_pt_1[:,i]))
        pt_2 = ' '.join(map(str,final_corr_pt_2[:,i]))
        final_file.write(pt_1 + ' ' + pt_2 + '\n')
    final_file.close()
    
def get_render_array(file_dict,model_pts,depth_location):
    #camera matrix
    extrinsic_camera_matrix =  pt2render.ex_camera_matrix(file_dict['azimuth'],file_dict['elevation'],
                                                          file_dict['tilt'],file_dict['rho'])
    camera_matrix = pt2render.get_camera_matrix(variables.INTRINSIC_CAMERA_MATRIX,extrinsic_camera_matrix,
                                      variables.WORLD_MATRIX)
    #render
    pt_array = pt2render.get_render_location(model_pts,camera_matrix)
    
    # depth Map
    depth_map_name = get_depth_map_name(file_dict,depth_location)    
    depth_map = get_depthmap_from_exr(depth_map_name)
    depth_truth,depth_pt_array = pt2render.depth_visibility_array(pt_array,depth_map)
    #print(depth_map_name)
    #print(depth_pt_array.shape)
    #crop 
    crop_truth,crop_pt_array = pt2render.crop_visibility_array(pt_array,file_dict['crop_params'])
    #print(crop_pt_array.shape)
    #global_return
    global_truth = crop_truth*depth_truth
    return global_truth,crop_pt_array

def get_pt_depth_map(render_array,size = (540,960),radius = 2):
    depth_map = np.zeros(size)
    for i in range(render_array.shape[1]):
        depth_map[int(render_array[1,i])-radius:int(render_array[1,i])+radius,
          int(render_array[0,i])-radius:int(render_array[0,i])+radius] =  render_array[2,i]
    return depth_map

def get_depth_map_name(file_dict,location=variables.depth_location):
    depth_map_file_name = file_dict['model_type']+'_'+ file_dict['model_md5']+'_a'+ name_version(file_dict['azimuth'])+\
    '_e'+ name_version(file_dict['elevation'])+'_t'+ name_version(-file_dict['tilt'])+ '_d'+\
    name_version(file_dict['rho'])+'.png1.exr'
    return os.path.join(location,depth_map_file_name)

def get_final_file_name(file_dict,location= variables.render_file_location):
    file_name =  file_dict['model_type']+'_'+ file_dict['model_md5']+'_a'+ name_version(file_dict['azimuth'])+\
    '_e'+ name_version(file_dict['elevation'])+'_t'+ name_version(-file_dict['tilt'])+ '_d'+ name_version(file_dict['rho'])+\
    '_'+str(file_dict['crop_params'][0])+'_'+str(file_dict['crop_params'][1])+'_'+str(file_dict['crop_params'][2])+\
    '_'+str(file_dict['crop_params'][3])+'_.jpg'
    file_name = os.path.join(file_dict['model_type'],file_dict['model_md5'],file_name)
    return os.path.join(location,file_name)
    
def get_difference(dict_1,dict_2):
    azi = dict_1['azimuth'] -dict_2['azimuth']
    ele = dict_1['elevation'] -dict_2['elevation']
    tilt = dict_1['tilt'] -dict_2['tilt']
    rho = dict_1['rho'] -dict_2['rho']
    return azi,ele,tilt,rho

def name_version(strng):
    return ('%3.3f'%strng).replace('.','@')


def render_random_direction_depth_map(pt_array,size=(540,960)):
    azi = np.rad2deg(2*math.pi*np.random.uniform())
    z = np.random.uniform()-0.001
    ele = np.rad2deg(np.arctan(z/math.sqrt(1-z*z)))
    tilt = np.rad2deg(-math.pi/8.0 + np.random.uniform()*math.pi/4.0)
    rho = 3 + np.random.uniform()
    extrinsic_camera_matrix =  pt2render.ex_camera_matrix(azi,ele,tilt,rho)
    camera_matrix = pt2render.get_camera_matrix(variables.INTRINSIC_CAMERA_MATRIX,extrinsic_camera_matrix,
                                      variables.WORLD_MATRIX)
    pt_array = pt2render.get_render_location(pt_array,camera_matrix)
    depth_map = get_pt_depth_map(pt_array,size)
    return depth_map
    
def get_render_array_no_prune(file_dict,model_pts):
    #camera matrix
    extrinsic_camera_matrix =  pt2render.ex_camera_matrix(file_dict['azimuth'],file_dict['elevation'],
                                                          file_dict['tilt'],file_dict['rho'])
    camera_matrix = pt2render.get_camera_matrix(variables.INTRINSIC_CAMERA_MATRIX,extrinsic_camera_matrix,
                                      variables.WORLD_MATRIX)
    #render
    pt_array = pt2render.get_render_location(model_pts,camera_matrix)
    return pt_array
    
        
        
