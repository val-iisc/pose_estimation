#class to train
class_name = 'chair'

#gpu-id for training
gpu = 2

batch_size = 8

save_checkpoint_after = 1000

val_iters_after = 200

num_val_samples = 50

train_script = 'multi_view_train.py'

#num views for keypoint correspondence
num_views = 3

start_iteration = 0