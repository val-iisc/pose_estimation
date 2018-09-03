'''
script for the projection of 3D keypoints to 2D perspective projection.

Usage: python run_proj.py --shape_file --file <shape-file> --azimuth <azimuth> --elevation <elevation> --tilt <tilt> --distance <distance> 
'''
import trimesh
import pt2render
import argparse
import numpy as np
import variables
import Imath
import OpenEXR 


def get_keypoints(shape_file,k = 100000):
    mesh = trimesh.load(shape_file)
    if type(mesh) == list:
        for j in range(1,len(mesh)):
            mesh[0] = mesh[0].union(mesh[j])
        mesh = mesh[0]
    all_pts = mesh.sample(int(k), 0)
    return all_pts

def get_3d_to_2d(model_pts, params):
    extrinsic_camera_matrix =  pt2render.ex_camera_matrix(params[0], params[1], params[2], params[3])
    camera_matrix = pt2render.get_camera_matrix(variables.INTRINSIC_CAMERA_MATRIX,extrinsic_camera_matrix,
                                      variables.WORLD_MATRIX)
    #render
    pt_array = pt2render.get_render_location(model_pts,camera_matrix)
    
    return pt_array

def prune_pts(pt_array, depth_map_name, crop=False, crop_params=None):
    # depth Map  
    depth_map = get_depthmap_from_exr(depth_map_name)
    depth_truth,depth_pt_array = pt2render.depth_visibility_array(pt_array,depth_map)
    #print(depth_map_name)
    #crop 
    if crop:
        crop_truth,crop_pt_array = pt2render.crop_visibility_array(pt_array, crop_params)
    else:
        crop_truth = np.ones(depth_truth.shape)
        crop_pt_array = depth_pt_array
    global_truth = crop_truth*depth_truth

    return global_truth,crop_pt_array


def get_depthmap_from_exr(file_name):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(file_name)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depthmap = np.fromstring(redstr, dtype = np.float32)
    depthmap.shape = (size[1], size[0]) # Numpy arrays are (row, col)
    return depthmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projection of 3D points to 2D perspective projection.')
    parser.add_argument('--keypoints',
                        help='if set, numpy file containing 3D keypoints is expected in --file',   action='store_true')
    parser.add_argument('--shape_file',
                        help='if set, .obj file containing a 3D object is expected in --file ',   action='store_true')
    parser.add_argument('--file',               type=str,
                        help='file name',         required=True)
    parser.add_argument('--depth_prune',
                        help='if set, depth based pruning is done, and depth_map expected in  --depth-file ',   action='store_true')
    parser.add_argument('--depth_file',               type=str,
                        help='depth file-name')
    parser.add_argument('--azimuth', type=float,
                        help='the azimuth angle of object in render',      required=True)
    parser.add_argument('--elevation', type=float,
                        help='the elevation angle of object in render',      required=True)
    parser.add_argument('--tilt', type=float,
                        help='the tilt angle of object in render',      required=True)
    parser.add_argument('--distance', type=float,
                        help='the distance of object in render',      required=True)

    args = parser.parse_args()
    # An example use of the functions:
    # Given a render param, you can get the pixel coordinates of the 3D points
    if args.shape_file:
        key_3d = get_keypoints(args.file) # get 1000 random points on the object
    else:
        key_3d = np.load(args.file)
    print('Got the 3D keypoints!')
    print('No of keypoints',key_3d.shape)
    # now project the key_3d based on the params
    params = [args.azimuth, args.elevation, args.tilt, args.distance]
    key_2d = get_3d_to_2d(key_3d, params)
    print('2D projections of the points created.')
    print('No of keypoints',key_2d.shape)
    # if pruning required:
    if args.depth_prune:
        print('Pruning the projected points based on depth.')
        truth, key_2d = prune_pts(key_2d,args.depth_file)
        key_2d = np.copy(key_2d[:,truth.astype('bool')])
        print('No of keypoints after pruning',key_2d.shape)
        # Additionally pruning based on crop params also provided
    
    np.save('temp.npy',key_2d)
    print('2D projections saved in temp.npy.')







       
