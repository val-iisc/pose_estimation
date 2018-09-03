import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import logm
from torch.autograd import Variable
from caffe.proto import caffe_pb2
import google.protobuf.text_format


def create_prototxt(batch_size, val_set_size = -1):
    net = caffe_pb2.NetParameter()
    f = open('./protos/googlenet_ucn_classifier_enhanced_conv_stn_classifier_batch8_classifier.prototxt', 'r')
    net1 = google.protobuf.text_format.Merge(str(f.read()), net)
    f.close()

    net1.layer[0].input_param.shape[0].dim[0] = batch_size
    f = open('./protos/train_test.prototxt', 'w')
    f.write(str(net1))
    f.close()
    if val_set_size != -1:
        net1.layer[0].input_param.shape[0].dim[0] = val_set_size % batch_size

        f = open('./protos/train_test_rem.prototxt', 'w')
        f.write(str(net1))
        f.close()


def get_adjusted_pred_uniform_mulaw(pred_angle, gt_real, mu=20, in_range=45, out_range=180, rand_ele=True):
    # from a 180 class to 45 conversion.
    diff = np.zeros(pred_angle.shape)
    diff[:, 0] = mu_inv(np.random.uniform(-in_range, in_range, size=diff.shape[0]))

    if rand_ele:
        diff[:, 1] = np.random.uniform(-5, 25, size=(gt_real.shape[0]))
        diff[:, 2] = np.random.uniform(-10, 10, size=(gt_real.shape[0]))

    pred_angle = np.copy((gt_real + diff) % 360)

    diff = get_actual_diff(gt_real, pred_angle)

    return pred_angle, diff


def mu_inv(inp, mu=20.0, in_range=45.0, out_range=180.0):
    inp = inp / in_range
    out = np.sign(inp) / mu * (np.power(1 + mu, np.abs(inp)) - 1) * out_range
    return out


def mu_law(inp, mu=20.0, in_range=180.0, out_range=45.0):
    inp = inp / in_range
    out = np.sign(inp) * (np.log(1 + mu * np.abs(inp))) / np.log(1 + mu) * out_range
    return out


def preprocess(caffe_model, input_list):
    syn_img = np.array(input_list[0])
    feature_i = caffe_model.get_feature_blob(syn_img)
    feature_i = Variable(torch.from_numpy(feature_i).cuda())

    mask = np.array(input_list[1]).transpose(0, 3, 1, 2).astype('float32')
    mask = Variable(torch.from_numpy(mask).cuda())
    mask = F.max_pool2d(input=mask, kernel_size=4, stride=4, ceil_mode=True).cuda()

    optical_flow = np.array(input_list[2])
    optical_flow = Variable(torch.FloatTensor(optical_flow).cuda())

    gt_syn = np.array(input_list[3])
    return feature_i, mask, optical_flow, gt_syn, syn_img


def preprocess_multi_view(caffe_model, input_list):
    syn_img = np.array(input_list[0])
    feature_i = caffe_model.get_feature_blob(syn_img)
    # feature_i = Variable(torch.from_numpy(feature_i).cuda())

    mask = np.array(input_list[1]).transpose(0, 3, 1, 2).astype('float32')
    mask = Variable(torch.from_numpy(mask).cuda())
    mask = F.max_pool2d(input=mask, kernel_size=4, stride=4, ceil_mode=True).cuda()

    optical_flow = np.array(input_list[2])
    optical_flow = Variable(torch.FloatTensor(optical_flow).cuda())

    gt_syn = np.array(input_list[3])
    # print "mask", mask
    return feature_i, mask, optical_flow, gt_syn, syn_img

def get_actual_diff(angle_1, angle_2):
    angle_1 = angle_1 % 360
    angle_2 = angle_2 % 360
    diff = angle_1 - angle_2
    diff[diff[:, 0] > 180, 0] = -(360 - diff[diff[:, 0] > 180, 0])
    diff[diff[:, 0] < -180, 0] = 360 + diff[diff[:, 0] < -180, 0]
    diff[diff[:, 2] > 180, 2] = -(360 - diff[diff[:, 2] > 180, 2])
    diff[diff[:, 2] < -180, 2] = 360 + diff[diff[:, 2] < -180, 2]
    diff[diff[:, 1] > 180, 1] = -(360 - diff[diff[:, 1] > 180, 1])
    diff[diff[:, 1] < -180, 1] = 360 + diff[diff[:, 1] < -180, 1]

    return diff


def get_rotmat(angles):
    angles = angles[:, [1, 0, 2]]
    angles = angles * math.pi / 180.

    coses = np.cos(angles)
    sines = np.sin(angles)
    cos_2 = coses[:, 2]
    cos_0 = coses[:, 0]
    cos_1 = coses[:, 1]
    sin_0 = sines[:, 0]
    sin_2 = sines[:, 2]
    sin_1 = sines[:, 1]

    a11 = cos_0 * cos_1
    a12 = sin_2 * sin_0 * cos_1 - cos_2 * sin_1
    a13 = cos_2 * sin_0 * cos_1 + sin_2 * sin_1
    a21 = cos_0 * sin_1
    a22 = sin_2 * sin_0 * sin_1 + cos_2 * cos_1
    a23 = cos_2 * sin_0 * sin_1 - sin_2 * cos_1
    a31 = - sin_0
    a32 = sin_2 * cos_0
    a33 = cos_2 * cos_0

    a1 = np.stack([a11, a12, a13], 1)
    a2 = np.stack([a21, a22, a23], 1)
    a3 = np.stack([a31, a32, a33], 1)
    out = np.stack([a1, a2, a3], 1)
    # this will be batch_size,3,3
    return out


def get_distance_rotmat_render_for_cnn(rotmat_1, rotmat_2):
    dist = []
    batch_size = rotmat_1.shape[0]
    for batch in range(batch_size):
        mult = logm(np.matmul(rotmat_1[batch, :].T, rotmat_2[batch, :]))
        val = np.linalg.norm(mult, ord=2) / np.sqrt(2)
        dist.append(val)
    dist = np.array(dist) * 180 / math.pi
    return dist


def dist_rotmat(pred_angle, gt_real):
    rotmat_1 = get_rotmat(pred_angle)
    rotmat_2 = get_rotmat(gt_real)
    dist = get_distance_rotmat_render_for_cnn(rotmat_1, rotmat_2)
    return np.abs(dist)


def dist_abs(pred_angle, gt_real, angle_type='azimuth'):
    print "gt_real", gt_real.shape
    batch_size = gt_real.shape[0]
    if angle_type == 'azimuth':
        error_azi = np.abs([(gt_real[i, 0] - pred_angle[i][0]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    elif angle_type == 'elevation':
        error_azi = np.abs([(gt_real[i, 1] - pred_angle[i][1]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    elif angle_type == 'tilt':
        error_azi = np.abs([(gt_real[i, 2] - pred_angle[i][2]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    return error_azi


def mean(x, key):
    if key == 0:
        return np.clip(x, -45, 45) * np.abs(np.clip(x, -45, 45)) / 45.0
    elif key in [1, 2]:
        return np.clip(x, -15, 15) * np.abs(np.clip(x, -15, 15)) / 15.0


def sigma(x, k=10):
    return 15 + k * (1.5 - np.cos(2 * x / 180. * math.pi) - (1 - abs(x / 180.)) * 2)


def get_knn_visibility(pt_list):
    return np.array(pt_list)[:, 0, :, :2], np.array(pt_list)[:, 0, :, :2]


def tester():
    x = np.random.uniform(0, 360, size=(1, 3))


if __name__ == '__main__':
    tester()
