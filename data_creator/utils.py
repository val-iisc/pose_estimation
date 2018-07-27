import scipy.io as sio
import numpy as np
import os
import cv2
import pickle
import global_variables as gv

def keypoint_5_fetcher(data_dir,class_name,pose=False):
    #if class_name == 'table':
    #    coords = os.path.join(data_dir,class_name,'new_coords_adjusted.pkl')
    #    coords = pickle.load(open(coords))
    #    print(coords.keys())        
    #else:
    coords = sio.loadmat(os.path.join(data_dir,class_name,'coords.mat'))['coords']
    coords = np.copy(coords[:,:gv.skeleton_size[class_name],:,:])
    _,_,seat = gv.skeleton_map[class_name]
    dir_to_angle = {'right_forward': (45,0,0),'right_backward': (135,0,0),
                    'left_forward': (315,0,0), 'left_backward': (225,0,0)}
    def get_bbox(img_name,coords=coords):
        cur_id = int(img_name.split('.')[0])
        cur_coords = np.copy(coords[:,:,:3,cur_id-1])
        # till 3 only as bed.4 is not annotated (others donot have 4)
        min_y = max(0,int(np.nanmin(cur_coords[1,:,:])))
        max_y = max(0,int(np.nanmax(cur_coords[1,:,:])))
        min_x = max(0,int(np.nanmin(cur_coords[0,:,:])))
        max_x = max(0,int(np.nanmax(cur_coords[0,:,:])))
        bbox = [np.array([min_x,min_y,max_x,max_y])]
        
        return bbox

    def get_coords(img_name,coords=coords):
        cur_id = int(img_name.split('.')[0])
        cur_coords = np.copy(coords[:,:,:3,cur_id-1])
        cur_coords = np.nanmedian(cur_coords,2)
        cur_coords = np.clip(cur_coords,0,np.max(cur_coords))
        # till 3 only as bed.4 is not annotated (others donot have 4)
        out = np.ones((cur_coords.shape[1],3))
        out[:,:2] = cur_coords.T
        out = [out]
        return out
    
    def get_dummy_attrs(img_name):
        # dummy function:
        return (1,1,1)
    
    def get_approx_pose(img_name,coords=coords,seat=seat,dir_to_angle=dir_to_angle):
        cur_id = int(img_name.split('.')[0])
        cur_coords = np.copy(coords[:,:,:3,cur_id-1])
        cur_coords = np.nanmedian(cur_coords,2)
        cur_coords = np.clip(cur_coords,0,np.max(cur_coords))
        # getting dirction for classification
        front_ = cur_coords[:,seat[0]]+ cur_coords[:,seat[1]]
        back_ = cur_coords[:,seat[2]]+ cur_coords[:,seat[3]]
        dir_ = front_-back_
        if dir_[0] >=0:
            # Facing right
            direction = 'right'
            if dir_[1]>=0:
                # facing forward
                direction += '_forward'
            elif dir_[1]<0:
                # facing backward
                direction += '_backward'
        elif dir_[0]<0:
            # facing left
            direction = 'left'
            if dir_[1]>=0:
                # facing forward
                direction += '_forward'
            elif dir_[1]<0:
                # facingbackward
                direction += '_backward'
        poser = [dir_to_angle[direction]]
        return poser
    if pose:
        return get_bbox, get_approx_pose,get_dummy_attrs
    else:
        return get_bbox,get_coords


def pascal_fetcher(dataset, data_dir, class_name,pose=False):
    # pascal Imageinet
    if dataset == 'pascal3D':
        ann_loc = os.path.join(data_dir,'Annotations',class_name)
    elif dataset == 'ObjectNet3D':
        ann_loc = os.path.join(data_dir,'Annotations')
    ann_name = lambda x:x+'.mat'
    img_name = lambda x:x+ '.JPEG'
    class_name = class_name.split('_')[0]
    names = gv.pascal_keypoint_names[class_name]
    num_coords = len(names)

    def get_bbox(img_name,class_name = class_name):
        img_name = img_name.split('.')[0]
        class_info,bbox_params_all = load_pascal_mat(os.path.join(ann_loc,ann_name(img_name)),'bbox')
        bbox_params_all = bbox_params_all[class_info == class_name]#.astype(int)
        final_bboxes = []
        for bbox_params in bbox_params_all:
            final_bboxes.append(list(bbox_params[0].astype(int)))
        return final_bboxes

    def get_coords(img_name,class_name = class_name, names = names, num_coords = num_coords):
        img_name = img_name.split('.')[0]
        class_info,location_gt = load_pascal_mat(os.path.join(ann_loc,ann_name(img_name)),'anchors')
            
        location_gt = location_gt[class_info == class_name]
        
        final_coords = []
        for i in range(location_gt.shape[0]):
            loc_rec = np.zeros((num_coords,3))        
            if location_gt[i].shape[0] >0:
                for j in range(len(names)):                                             
                    if not names[j] is None:
                        if location_gt[i][names[j]][0][0]['location'][0][0].shape[0] ==1:      
                            loc_rec[j,:2] = location_gt[i][names[j]][0][0]['location'][0][0][0]
                            loc_rec[j,2] = 1 #location_gt[i][names[j]][0][0]['status'][0][0][0]
                        else: loc_rec[j,2] = 0
                    else: loc_rec[j,2] = 0
            final_coords.append(loc_rec)
        return final_coords
    
    def get_pose_pascal(img_name,class_name = class_name, names = names, num_coords = num_coords):
        img_name = img_name.split('.')[0]
        class_info,pose_gt = load_pascal_mat(os.path.join(ann_loc,ann_name(img_name)),'viewpoint')
        pose_gt = pose_gt[class_info==class_name]
        pose = []
        for poser in pose_gt:
            print(poser)
            if poser['distance'][0][0][0][0] ==0:
                azi = poser['azimuth_coarse'][0][0][0][0]
                ele = poser['elevation_coarse'][0][0][0][0]
                tilt = 360-poser['theta'][0][0][0][0]
                print(azi,ele)
            else:
                azi = poser['azimuth'][0][0][0][0]
                ele = poser['elevation'][0][0][0][0]
                tilt = 360-poser['theta'][0][0][0][0]
            azi_str = "{0:.3f}".format(azi)
            ele_str = "{0:.3f}".format(ele)
            tilt_str = "{0:.3f}".format(tilt)
            pose.append([azi_str,ele_str,tilt_str])
        return pose
    def get_pose_objectnet(img_name,class_name = class_name, names = names, num_coords = num_coords):
        img_name = img_name.split('.')[0]
        class_info,pose_gt = load_pascal_mat(os.path.join(ann_loc,ann_name(img_name)),'viewpoint')
        pose_gt = pose_gt[class_info==class_name]
        pose = []
        for poser in pose_gt:
            print(poser)
            try:
                azi = poser['azimuth'][0][0][0][0]
                ele = poser['elevation'][0][0][0][0]
                tilt = 360-poser['theta'][0][0][0][0]
            except:
                azi = poser['azimuth_coarse'][0][0][0][0]
                ele = poser['elevation_coarse'][0][0][0][0]
                tilt = 360-poser['theta'][0][0][0][0]
            print(azi,ele)
            azi_str = "{0:.3f}".format(azi)
            ele_str = "{0:.3f}".format(ele)
            tilt_str = "{0:.3f}".format(tilt)
            pose.append([azi_str,ele_str,tilt_str])
        return pose
    
    def get_attrs(img_name,class_name = class_name, names = names, num_coords = num_coords):
        img_name = img_name.split('.')[0]
        mat_info = sio.loadmat(os.path.join(ann_loc,ann_name(img_name)))
        class_info = mat_info['record'][0]['objects'][0]['class'][0]
        difficult_info = mat_info['record'][0]['objects'][0]['difficult'][0][class_info==class_name]
        occluded_info = mat_info['record'][0]['objects'][0]['occluded'][0][class_info==class_name]
        truncated_info = mat_info['record'][0]['objects'][0]['truncated'][0][class_info==class_name]
        attrs = []
        for iter in range(len(difficult_info)):
            info = [difficult_info[iter][0][0],occluded_info[iter][0][0],truncated_info[iter][0][0]]
            attrs.append(info)
        return attrs

    if pose:
        if dataset == 'pascal3D':
            return get_bbox, get_pose_pascal,get_attrs
        else:
            return get_bbox, get_pose_objectnet,get_attrs
    else:
        return get_bbox, get_coords

def load_pascal_mat(path,seek_info):
    mat_info = sio.loadmat(path)
    class_info = mat_info['record'][0]['objects'][0]['class'][0]
    other_info = mat_info['record'][0]['objects'][0][seek_info][0]
    return class_info,other_info

def get_2d_keypoints(pt_1,pt_2,n):
    keypoints_2d = np.zeros((2,n))
    keypoints_2d[0,:] = np.linspace(pt_1[0],pt_2[0],n)
    keypoints_2d[1,:] = np.linspace(pt_1[1],pt_2[1],n)
    return keypoints_2d

        
def get_additional_coords_keypoint_5(img_coords, class_name,dim = 3):
    # additional points and the locations
    if class_name == 'chair':    
        add_img_coords = np.zeros((dim,4))
        add_img_coords[:,0] = np.nanmean(img_coords[:,4:6],1)
        add_img_coords[:,1] = np.nanmean(img_coords[:,5:7],1)
        add_img_coords[:,2] = np.nanmean(img_coords[:,6:8],1)
        additional = np.concatenate([img_coords[:,7:8],img_coords[:,4:5]],1)
        #print(additional)
        add_img_coords[:,3] = np.nanmean(additional,1)
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'swivelchair':
        add_img_coords = np.zeros((dim,4))
        add_img_coords[:,0] = np.nanmean(img_coords[:,7:9],1)
        add_img_coords[:,1] = np.nanmean(img_coords[:,8:10],1)
        add_img_coords[:,2] = np.nanmean(img_coords[:,9:11],1)
        additional = np.concatenate([img_coords[:,10:11],img_coords[:,7:8]],1)
        #print(additional)
        add_img_coords[:,3] = np.nanmean(additional,1)
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'table':
        add_img_coords = np.zeros((dim,4))
        add_img_coords[:,0] = np.nanmean(img_coords[:,0:2],1)
        add_img_coords[:,1] = np.nanmean(img_coords[:,1:3],1)
        add_img_coords[:,2] = np.nanmean(img_coords[:,2:4],1)
        additional = np.concatenate([img_coords[:,0:1],img_coords[:,3:4]],1)
        add_img_coords[:,3] = np.nanmean(additional,1)
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'bed':
        add_img_coords = np.zeros((dim,7))
        additional = np.concatenate([img_coords[:,3:4],img_coords[:,8:9]],1)
        add_img_coords[:,0] = np.copy(np.nanmean(additional,1))
        add_img_coords[:,1] = np.nanmean(img_coords[:,7:9],1)
        additional = np.concatenate([img_coords[:,7:8],img_coords[:,2:3]],1)
        add_img_coords[:,2] = np.nanmean(additional,1)
        add_img_coords[:,3] = np.nanmean(img_coords[:,2:4],1)
        add_img_coords[:,4] = np.nanmean(img_coords[:,0:2],1)
        additional = np.concatenate([img_coords[:,1:2],img_coords[:,6:7]],1)
        add_img_coords[:,5] = np.nanmean(additional,1)
        add_img_coords[:,6] = np.nanmean(img_coords[:,5:7],1)
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'sofa':
        add_img_coords = np.zeros((dim,4))
        additional = np.concatenate([img_coords[:,1:2],img_coords[:,8:9]],1)
        add_img_coords[:,0] = np.nanmean(additional,1)
        additional = np.concatenate([img_coords[:,3:4],img_coords[:,10:11]],1)
        add_img_coords[:,1] = np.nanmean(additional,1)
        additional = np.concatenate([img_coords[:,2:3],img_coords[:,9:10]],1)
        add_img_coords[:,2] = np.nanmean(additional,1)
        additional = np.concatenate([img_coords[:,6:7],img_coords[:,13:14]],1)
        add_img_coords[:,3] = np.nanmean(additional,1)
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    return img_coords
       
def get_additional_coords_pascal(img_coords, class_name,dim = 3):
    # additional points and the locations
    if class_name == 'chair':    
        add_img_coords = np.zeros((dim,4))
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'swivelchair':
        add_img_coords = np.zeros((dim,4))
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'diningtable':
        add_img_coords = np.zeros((dim,4))
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'bed':
        add_img_coords = np.zeros((dim,7))
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    if class_name == 'sofa':
        add_img_coords = np.zeros((dim,4))
        img_coords = np.concatenate([img_coords,add_img_coords],1)
    return img_coords

# Pruning functions

def surface_pruned(all_coords,img_coords,seat_info,k=range(40)):
    points = [ img_coords[:2,seat_info[0]],img_coords[:2,seat_info[1]],img_coords[:2,seat_info[2]],img_coords[:2,seat_info[3]] ]
    points = np.array(points,dtype=np.int32)
    print('points',points)
    src = np.zeros(gv.img_size,np.uint8)
    
    cv2.polylines(src,[points],True,255,1)
    
    _, contours,hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]  
    for m in k:
        (j,i) = all_coords[:,m]
        if cv2.pointPolygonTest(cnt,(j,i),False)>-1.0:
            print(m,i,j)
            all_coords[:,m] = np.zeros((2))

    return all_coords
    
def dir_pruned(all_coords, img_coords,skeleton,seat_info, skele_info):
    center_front = (img_coords[:2,seat_info[0]]+img_coords[:2,seat_info[1]])/2
    center_back = (img_coords[:2,seat_info[2]]+img_coords[:2,seat_info[3]])/2
    print(center_front,center_back)
    dir_vector = center_front - center_back
    # now check if its left of right

    src = np.zeros(gv.img_size,np.uint8)
    if dir_vector[0]>0:
        print('facing right!')
        prune_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'L' in skele_info[i]]
        test_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'R' in skele_info[i] or 'C' in skele_info[i]]
        # Then weightage to the right side of the model,
    elif dir_vector[0]<0:
        print('facing left!')
        prune_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'R' in skele_info[i]]
        test_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'L' in skele_info[i] or 'C' in skele_info[i]]
    else:
        prune_skeleton = []
        test_skeleton = []
    #for the pruners
    for (i,j) in prune_skeleton:
        cv2.line(src,tuple(img_coords[:2,i].astype(int)),tuple(img_coords[:2,j].astype(int)),(255),5)
    for i in test_skeleton:
        for m in range(8):
            i_index = skeleton.index(i)
            test_pt_ind = i_index*10 +1+ m
            test_pt = all_coords[:,test_pt_ind].astype(int)
            # check if red in src
            #print(all_coords)
            if src[test_pt[1],test_pt[0]] == 255:
                print('point is in ',test_pt)
                all_coords[:,test_pt_ind] = np.zeros((2))
    return all_coords

def surface_dir_prune(all_coords, img_coords, skeleton,skele_info,seat_info):
    center_front = (img_coords[:2,seat_info[0]]+img_coords[:2,seat_info[1]])/2
    center_back = (img_coords[:2,seat_info[2]]+img_coords[:2,seat_info[3]])/2
    print(center_front,center_back)
    dir_vector = center_front - center_back
    if dir_vector[0]>0:
        print('facing right!')
        prune_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'RR' in skele_info[i]]
        # Then weightage to the right side of the model,
    elif dir_vector[0]<0:
        print('facing left!')
        prune_skeleton = [skeleton[i] for i in range(len(skeleton)) if 'LL' in skele_info[i]]
    else:
        prune_skeleton = []
    for i in prune_skeleton:
        for m in range(10):
            i_index = skeleton.index(i)
            test_pt_ind = i_index*10+ m
            test_pt = all_coords[:,test_pt_ind].astype(int)
            # check if red in src
            all_coords[:,test_pt_ind] = np.zeros((2))
    return all_coords
    
# for synthetic keypoints

def get_3d_keypoints(pt_1,pt_2,n):
    keypoints_2d = np.zeros((3,n))
    keypoints_2d[0,:] = np.linspace(pt_1[0],pt_2[0],n)
    keypoints_2d[1,:] = np.linspace(pt_1[1],pt_2[1],n)
    keypoints_2d[2,:] = np.linspace(pt_1[2],pt_2[2],n)
    return keypoints_2d

def get_params_from_syn_file(file_name):
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

def img_to_depth_file(img):
    x = '_'.join(img.split('_')[:-5]) + '.png1.exr'
    return x