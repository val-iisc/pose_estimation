import os

#class to evaluate
class_name = 'chair'

#number of iterations per image
num_iters_iterative_val = 10

#path to trained Pose Estimator Model
model_to_load = os.path.join('./pretrained_weights/', 'ispa_net_models/classifiers', 'pascal/' + class_name, 'pose_estimator.pth')

batch_size = 8

#gpu-id for testing
gpu = 2