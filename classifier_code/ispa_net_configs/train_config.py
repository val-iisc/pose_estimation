#class to tarin
class_name = 'chair'

#gpu-id for training
gpu = 2

#number of bins of Viewpoint Classifier Network
n_bins_classifier = 16

batch_size = 8

save_checkpoint_after = 1000

val_iters_after = 200

num_val_samples = 50

#Script to be used for training
train_script = 'iterative_train.py'

start_iteration = 0


