import glob
import numpy as np
import os
import scipy.misc
import random
import cv2
import pdb

from configs import constants

FLOW_PATH = '../data/synthetic_data/disparity'
PIXEL_MEANS = constants.PIXEL_MEANS


def get_image_path(x):
    st = x.strip('\n').split(' ')[0]
    st = st.split('/')
    initial_path = '/'.join(st[:-1])
    path = initial_path + '/images/' + st[-1] + '.png'

    return path

def get_all_paths(file_name):
    file_id = open(file_name)
    all_files = [(get_image_path(x),
      (float(x.strip('\n').split(' ')[-3]), float(x.strip('\n').split(' ')[-2]),
       float(x.strip('\n').split(' ')[-1])))
     for x in file_id.readlines()]
    file_id.close()
    return all_files


def all_images_list(train=True, pascal=True, objectnet=False, class_name='chair'):
    files_train = []
    files_val = []
    files_test = []

    if pascal:
        paths_train = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*pascal3D*train*.txt')
        files_train += glob.glob(paths_train)

        paths_val = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*pascal3D*val*.txt')
        files_val += glob.glob(paths_val)

        paths_test = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*pascal3D*test*.txt')
        files_test += glob.glob(paths_test)

    if objectnet:
        paths_train = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*ObjectNet3D*train*.txt')
        files_train += glob.glob(paths_train)

        paths_val = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*ObjectNet3D*val*.txt')
        files_val += glob.glob(paths_val)

        paths_test = os.path.join('../data/image_lists/final_files/', class_name + '_pose' + '*ObjectNet3D*test*.txt')
        files_test += glob.glob(paths_test)

    images_train = []
    images_val = []
    images_test = []

    for file_name in files_train:
        images_train += get_all_paths(file_name)

    for file_name in files_val:
        images_val += get_all_paths(file_name)

    for file_name in files_test:
        f = open(file_name)
        images_test += get_all_paths(file_name)

    if train:
        return images_train, images_val
    else:
        return images_test


def augmenter(img1, img_1_angle):
    x = np.random.uniform(size=(4))
    if x[1] > 0.6:
        probabilities = np.random.uniform(0.8, 1.2, 3)
        img1 = img1[:, :, :] * probabilities
        img1 = np.clip(img1, 0, 255)
    if x[2] > 0.6:
        if x[3] > 0.5:
            img1 = cv2.resize(img1, (56, 56))
        else:
            img1 = cv2.resize(img1, (112, 112))
        img1 = cv2.resize(img1, (224, 224))
    return img1, img_1_angle


def get_canonical_angle(view_params):
    angles = np.array(view_params)
    angles[2] = angles[2] % 360
    if angles[2] > 180:
        angles[2] = -(360 - angles[2])
    elif angles[2] < -180:
        angles[2] = 360 + angles[2]
    return angles


def batch_data_processor_real(img_list, batch_size, augment=True):
    iter = 0
    while True:
        if iter % len(img_list) == 0: random.shuffle(img_list)
        cur_batch = max(0, iter % len(img_list) - batch_size)
        img_batch = []
        img_angles = []
        for img_name, view_params in img_list[cur_batch:cur_batch + batch_size]:
            img_1 = scipy.misc.imread(img_name).astype('float')
            img_1_angle = get_canonical_angle(view_params)
            if augment: img_1, img_1_angle = augmenter(img_1, img_1_angle)
            # for caffe
            img_1 = (img_1 - PIXEL_MEANS).transpose(2, 0, 1)
            img_batch.append(img_1)
            img_angles.append(img_1_angle)
            iter += 1

        yield img_batch, img_angles


def batch_data_processor_real_unique_all(img_list, batch_size, augment=False, name=False):
    for iter in range(len(img_list)):
        cur_batch = iter * batch_size

        img_batch = []
        img_angles = []

        for img_name, view_params in img_list[cur_batch:cur_batch + batch_size]:

            img_1 = scipy.misc.imread(img_name).astype(float)
            img_1_angle = get_canonical_angle(view_params)
            if augment: img_1, img_1_angle = augmenter(img_1, img_1_angle)

            # for caffe
            img_1 -= np.array([[[102.9801, 115.9465, 122.7717]]])
            img_1 = img_1.transpose(2, 0, 1)

            img_batch.append(img_1)
            img_angles.append(img_1_angle)

        if name:
            yield img_batch, img_angles, img_list[cur_batch]
        else:
            yield img_batch, img_angles


def main():
    pdb.set_trace()


if __name__ == '__main__':
    main()
