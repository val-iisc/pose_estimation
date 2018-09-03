import os

#class to evaluate
class_name = 'chair'

#path to trained model
model_to_load = os.path.join('./pretrained_weights', 'multi_view_models/classifiers', 'pascal_' + class_name, 'all_data/pose_estimator.pth')

batch_size = 8

#gpu-d for testing
gpu = 2

#num views to be used for keypoint correspondence
num_views = 3

