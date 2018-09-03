import os
import numpy as np
import glob
import lib.config as config


def get_keypoint_path(path, data_type="real"):
    if data_type == 'syn':
        return '../data/synthetic_data/dense_keypoints/' + path.split('/')[-1].strip('\n').split('.')[0] + '.npy'
    else:
        keyp_path = path.split('/')
        keyp_path = '/'.join(keyp_path[:-2]) + '/dense_keypoints/' + keyp_path[-1].split('.')[0] + '.npy'
        return keyp_path


def get_real_image_paths(file_paths):
    image_paths = []

    for x in file_paths:
        for kk in open(x).readlines():
            st = kk.split(' ')
            image_paths.append(
                ('/'.join(st[0].split('/')[0:-1]) + '/images/' + st[0].split('/')[-1] + '.png', int(float(st[1]))))



    return image_paths


def get_all_image_paths(class_name='chair'):
    if config.CLASSIFIER:
        task = 'pose'
    else:
        task = 'correspondence'

    train_files_paths = glob.glob(
        '../data/image_lists/final_files/' + class_name + '*' + task + '*' + 'train' + '.txt')

    train_image_paths = get_real_image_paths(train_files_paths)

    val_files_paths = glob.glob(
        '../data/image_lists/final_files/' + class_name + '*' + task + '*' + 'val' + '.txt')

    val_image_paths = get_real_image_paths(val_files_paths)

    test_files_paths = glob.glob(
        '../data/image_lists/final_files/' + class_name + '*' + task + '*' + 'test' + '.txt')

    test_image_paths = get_real_image_paths(test_files_paths)

    if not config.CLASSIFIER:
        train_image_paths_keypoint5 = [(x, 0.0) for x in
                                       glob.glob('../data/real_data/keypoint-5/' + class_name + '/images/*.png')]
        train_image_paths += train_image_paths_keypoint5

    syn_image_paths = glob.glob(
        os.path.join('../data/synthetic_data/render_final/', config.models[class_name] + '*.png'))

    syn_image_paths = [(x, int(float(x.split('@')[-5].split('_')[-1][1:]))) for x in syn_image_paths]

    num_syn_images = len(syn_image_paths)



    if config.CLASSIFIER:
        inds = np.arange(len(train_image_paths))
        sampled_training_images = np.random.choice(inds, num_syn_images)

        train_image_paths = list(np.array(train_image_paths)[sampled_training_images]) + syn_image_paths

    return train_image_paths, val_image_paths, test_image_paths, syn_image_paths


def get_keypoints(real_image_path, syn_image_path):

    if config.CLASSIFIER:
        keyp_real = np.ones([config.BATCH_SIZE, 130]).T
        keyp_syn = np.ones([config.BATCH_SIZE, 130]).T
        return keyp_real, keyp_syn

    keyp_real_path = get_keypoint_path(real_image_path)

    keyp_real = np.load(keyp_real_path)
    keyp_syn_path = get_keypoint_path(syn_image_path, 'syn')
    keyp_syn = np.load(keyp_syn_path)

    keyp_real = keyp_real.transpose(1, 0)
    keyp_syn = keyp_syn.transpose(1, 0)

    if keyp_real.shape[0] > keyp_syn.shape[0]:
        keyp_syn = np.append(keyp_syn, np.zeros([keyp_real.shape[0] - keyp_syn.shape[0], 2]), axis=0)

    if keyp_real.shape[0] < keyp_syn.shape[0]:
        keyp_real = np.append(keyp_real, np.zeros([keyp_syn.shape[0] - keyp_real.shape[0], 2]), axis=0)

    keypoints_1_invalid = (keyp_real[:, 0] == 0) & (keyp_real[:, 1] == 0)
    keypoints_2_invalid = (keyp_syn[:, 0] == 0) & (keyp_syn[:, 1] == 0)
    keypoints_invalid = keypoints_1_invalid | keypoints_2_invalid
    valid_indices = np.where(np.logical_not(keypoints_invalid))

    keyp_real = keyp_real[valid_indices]
    keyp_syn = keyp_syn[valid_indices]

    return keyp_real, keyp_syn
