'''
This script contains the paths and links to the various datasets.
'''
import os

BASE_DIR = os.getcwd()
data_dir = os.path.join(BASE_DIR, '../data')
keypoint_5_dir = os.path.join(data_dir,'keypoint-5')
pascal3D_dir = os.path.join(data_dir, 'pascal3D')
objectnet3D_dir = os.path.join(data_dir, 'ObjectNet3D')
processed_image_loc = os.path.join(data_dir, 'real_data')
synthetic_data_dir = os.path.join(data_dir,'synthetic_data')
anchor_loc = os.path.join(synthetic_data_dir,'anchors')
depth_loc = os.path.join(synthetic_data_dir,'depth')
syn_disparity_loc = os.path.join(synthetic_data_dir,'disparity')
g_save_file_location_final = os.path.join(synthetic_data_dir,'render_final')

syn_dense_keypoint_loc =os.path.join(synthetic_data_dir,'dense_keypoints') 

g_datasets_folder = os.path.abspath(os.path.join(data_dir, 'datasets'))
g_shapenet_root_folder = os.path.join(g_datasets_folder, 'ShapeNetCore/ShapeNetCore.v1')


pascal_keypoint_names = {'chair' : ['leg_lower_left','leg_lower_right','leg_upper_right',
         'leg_upper_left', 'seat_lower_left','seat_lower_right','seat_upper_right',
         'seat_upper_left','back_upper_left','back_upper_right'],
         'sofa': ['left_bottom_back','front_bottom_left','seat_top_left','seat_bottom_left', 
         None,None ,'top_left_corner', 'right_bottom_back','front_bottom_right',
         'seat_top_right','seat_bottom_right',None, None,'top_right_corner'],
         'diningtable': ['top_lower_left','top_upper_left','top_upper_right','top_lower_right',
         'leg_lower_left','leg_upper_left','leg_upper_right','leg_lower_right'],
         'bed':['frame_back1','frame_front1','mattress4','mattress1','bedpost_top4',
         'frame_back2','frame_front2','mattress3','mattress2','bedpost_top3'] }

# will be solved soon
objnet_poor_mat = ['n00000000_10.JPEG','n00000000_127.JPEG','n03761084_13462.JPEG',
            'n03761084_13568.JPEG','n04146614_5299.JPEG','n03180011_7424.JPEG',
            'n04004475_17984.JPEG','n03761084_13592.JPEG','n04576002_1671.JPEG',
            'n03761084_13637.JPEG','n03483316_11733.JPEG','n03483316_21011.JPEG',
            'n03483316_12227.JPEG','n02871439_4427.JPEG','n02871439_4521.JPEG' ]
skeleton_size = {'chair':10,'swivelchair':13,'table':8,'sofa':14,'bed':10 }


skeleton_map = {'chair': ([[4,0],[5,1],[6,2],[7,3],[4,5],[5,6],[6,7],[7,4],[7,8],[8,9],[9,6],[10,12],[11,13]],
                          ['L','R','R','L','C','R','C','L','L','C','R','C','C'],[4,5,6,7]),
               'swivelchair': ([[4,0],[5,1],[6,2],[7,3],[7,8],[8,9],[9,10],[10,7],[10,11],[11,12],[12,9],[13,15],[14,16]],
                          ['O','O','O','O','C','R','C','L','L','C','R','C','C'],[7,8,9,10]),     
               'bed':([[3,1],[8,6],[7,5],[2,0],[3,8],[8,7],[7,2],[2,3],[2,4],[4,9],[9,7],[13,14],[10,15],[11,16],[0,1],[1,6],[6,5],[5,0],[10,12],[11,13]],
                      ['L','R','RR','LL','C','R','C','L','L','C','R','LL','C','RR','LL','C','RR','O','C','C'],[3,8,7,2]),
               'table': ([[0,4],[3,7],[2,6],[1,5],[0,3],[3,2],[2,1],[1,0],[8,10],[9,11]],
                         ['L','R','R','L','C','R','C','L','C','C'],[0,3,2,1]),
                'diningtable': ([[0,4],[3,7],[2,6],[1,5],[0,3],[3,2],[2,1],[1,0],[8,10],[9,11]],
                         ['L','R','R','L','C','R','C','L','C','C'],[0,3,2,1]),
               'sofa' : ([[3,1],[10,8],[11,7],[4,0],[3,10],[10,9],[9,2],[2,3],[2,6],[6,13],[13,9],[15,14],[6,4],[4,5],[5,3],[13,11],[11,12],[12,10],[0,1],[1,8],[8,7],[7,0],[15,16],[16,17]],
                        ['L','R','RR','LL','C','R','C','L','L','C','R','C','L','L','L','R','R','R','LL','C','RR','O','C','C'],[3,10,9,2])}

leg_range = {'chair':range(40), 'swivelchair':range(40), 'bed':list(range(40))+list(range(110,140)), 'table':range(40), 'diningtable':range(40), 'sofa':list(range(40))+list(range(110,120))}
img_size = (224,224)

syn_template = {'chair': ('13fdf00cde077f562f6f52615fb75fca','03001627'),
            'swivelchair': ('1e92f53f3a191323d42b9650f19dd425','03001627'),
            'sofa':('1fd45c57ab27cb6cea65c47b660136e7','04256520'),
            'bed':('7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f','02818832'),
            'table':('e41da371550711697062f2d72cde5c95','04379243')
            }