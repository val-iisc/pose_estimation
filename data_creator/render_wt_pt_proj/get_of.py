'''
Script for demo of proxy Optical Flow maps.

'''
import argparse
import variables as v
import os
from run_proj import *



if __name__ == '__main__':
    # First Render with run_render

    parser = argparse.ArgumentParser(description='Projection of 3D points to 2D perspective projection.')
    parser.add_argument('--shape_file',               type=str,
                        help='file name',         required=True)
    parser.add_argument('--azimuth', type=str,
                        help='the azimuth angle of object in render',      required=True)
    parser.add_argument('--elevation', type=str,
                        help='the elevation angle of object in render',      required=True)
    parser.add_argument('--tilt', type=str,
                        help='the tilt angle of object in render',      required=True)
    parser.add_argument('--distance', type=str,
                        help='the distance of object in render',      required=True)

    args = parser.parse_args()

    params = [int(float(args.azimuth)), int(float(args.elevation)), int(float(args.tilt)), int(float(args.distance))]
    command = ' '.join([v.blender_executable_path,'blank.blend','--background', '--python','run_render.py',args.shape_file,
     args.azimuth, args.elevation, args.tilt, args.distance])
    os.system(command)
    #print('Done!')

    depth_file = 'z_map/temp0001.exr'
    key_3d = get_keypoints(args.shape_file)

    # now get the numpy with the view params
    key_2d_norm = get_3d_to_2d(key_3d, params)
    truth_1, key_2d = prune_pts(key_2d_norm,depth_file)

    # now get the numpy with the view params + 10
    params[0] +=10
    params[1] +=0
    key_2d_more = get_3d_to_2d(key_3d, params)
    truth_2, key_2d = prune_pts(key_2d_more,depth_file)

    truth = truth_1.astype('bool')&truth_2.astype('bool')
    key_2d_more = np.copy(key_2d_more[:,truth])
    key_2d_norm = np.copy(key_2d_norm[:,truth])

    # Now for proxy optical-flow
    del_azi = key_2d_more -key_2d_norm
    flow = np.zeros((540,960,2))
    # pixel flow spread size
    k = 2
    for i in range(key_2d_norm.shape[1]):
        pt = key_2d_norm[:,i][:2].astype(int)
        flow[pt[1]-k:pt[1]+k,pt[0]-k:pt[0]+k,0] = del_azi[:,i][0]
        flow[pt[1]-k:pt[1]+k,pt[0]-k:pt[0]+k,1] = del_azi[:,i][1]
    print(flow.shape,'Done!')
    np.save('temp_flow.npy',flow)


    