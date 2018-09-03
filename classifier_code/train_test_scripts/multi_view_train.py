import __init__paths
import caffe
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable

import utils.data_loader as dl
import utils.proxy_renderer as pr
import utils.utils as utils
from arch.classifier_single import Classifier
from configs import paths

from train_test_scripts.utils import summary_writer_common

import arch.eccv_net.eccv_net as nnew
from multi_view_configs import train_config, ucn_config

batch_size = train_config.batch_size

angles = np.arange(0, 360, 15)[::24 / train_config.num_views]


def main():
    caffe.set_mode_gpu()
    caffe.set_device(train_config.gpu)
    torch.cuda.set_device(train_config.gpu)

    utils.create_prototxt(batch_size)
    caffe_model = Classifier('./protos/train_test.prototxt', ucn_config.dict_models[train_config.class_name], train_config.gpu)

    lr = 0.001

    py_net = nnew.Net(class_name=train_config.class_name)

    py_net.cuda()

    optimizer = optim.Adam(params=py_net.parameters(), lr=lr, weight_decay=0.0001)

    train_list, val_list = dl.all_images_list(train=True, pascal=True, objectnet=True,
                                              class_name=train_config.class_name)

    add_info = pr.get_all_rendered_images(train_config.class_name)

    trainval(py_net, caffe_model, train_list, val_list, optimizer, add_info)


def trainval(py_net, caffe_model, train_list, val_list, optimizer, add_info):
    input_fetcher = dl.batch_data_processor_real(train_list, batch_size, augment=True)

    best_val_error_rotmat = 10000

    for act_iter, input_fetch in enumerate(input_fetcher):
        print "act_iter", act_iter
        loss_azi, loss_ele, loss_tilt, error_rotmat = get_losses_errors(input_fetch,
                                                                        caffe_model,
                                                                        add_info,
                                                                        py_net,
                                                                        )
        total_loss = loss_azi + 0.5 * loss_ele + 0.5 * loss_tilt
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log_losses(loss_azi, loss_ele, loss_tilt)
        log_errors(error_rotmat)

        if act_iter % train_config.val_iters_after == 0:
            rotmat_error = test(py_net, caffe_model, val_list, add_info)
            best_val_error_rotmat = log_best_error(py_net, optimizer, rotmat_error, best_val_error_rotmat, 'rotmat',
                                                   act_iter)

            py_net.cuda()


def test(py_net, caffe_model, test_list, add_info):
    py_net.eval()
    for param in py_net.parameters():
        param.volatile = True
        param.requires_grad = False

    num_test_samples = train_config.num_val_samples

    input_fetcher = dl.batch_data_processor_real(test_list, batch_size, augment=False)

    loss_azi = 0
    loss_ele = 0
    loss_tilt = 0

    all_error_rotmat = []

    for act_iter, input_fetch in enumerate(input_fetcher):
        print "validation act_iter", act_iter

        loss_a, loss_e, loss_t, error_rotmat = get_losses_errors(input_fetch,
                                                                 caffe_model,
                                                                 add_info, py_net)

        loss_azi += loss_a.data.cpu().numpy()
        loss_ele += loss_e.data.cpu().numpy()
        loss_tilt += loss_t.data.cpu().numpy()

        all_error_rotmat.append(error_rotmat)

        if act_iter == num_test_samples - 1:
            break

    log_losses(loss_azi / num_test_samples, loss_ele / num_test_samples, loss_tilt / num_test_samples, True)
    error_rotmat = np.median(all_error_rotmat)

    py_net.train()
    for param in py_net.parameters():
        param.volatile = False
        param.requires_grad = True

    return error_rotmat



def get_losses_errors(input_fetch, caffe_model, add_info, py_net):
    pred_angle = np.zeros([batch_size, 3])
    real_img = np.array(input_fetch[0])
    gt_real = np.array(input_fetch[1])
    gt = gt_real.copy()

    bin_img, real_features = caffe_model.classify(real_img)
    for x in range(train_config.num_views - 1):
        if x == 0:
            real_features_concat = np.stack((real_features, real_features), 1)
        else:
            real_features_concat = np.concatenate((real_features_concat, np.expand_dims(real_features, 1)), 1)
    feature_i = torch.autograd.Variable(torch.FloatTensor(real_features_concat).cuda())
    ct = 0
    global knn_inds
    global visibility

    knn_inds = []
    visibility = []
    for x in angles:

        pred_angle = np.array([[x, 10, 0] for i in range(real_img.shape[0])])

        input_list = pr.get_syn_images_per_batch_wt_name_and_pt(pred_angle, add_info)
        knn_inds_vis = np.array(input_list[-1])[0][0]
        knn_inds_tmp = knn_inds_vis[:, :2]
        visibility_tmp = knn_inds_vis[:, 2]
        knn_inds.append(knn_inds_tmp)
        visibility.append(visibility_tmp)
        feature_j_lp, mask, optical_flow, gt_syn, syn_img = utils.preprocess_multi_view(caffe_model, input_list)
        mask = mask.data.cpu().numpy()
        if ct == 0:
            feature_j = feature_j_lp
            mask_j = mask
        elif ct == 1:
            mask_j = np.stack((mask_j, mask), 1)
            feature_j = np.stack((feature_j, feature_j_lp), 1)

        else:
            mask_j = np.concatenate((mask_j, np.expand_dims(mask, 1)), 1)
            feature_j = np.concatenate((feature_j, np.expand_dims(feature_j_lp, 1)), 1)

        ct += 1
    knn_inds = np.clip(np.stack(knn_inds, 0), 0, 223)
    visibility = np.stack(visibility, 0)

    mask = Variable(torch.from_numpy(mask_j).cuda())
    feature_j = torch.autograd.Variable(torch.FloatTensor(feature_j).cuda())

    diff = utils.get_actual_diff(gt_real, gt_syn)
    gt_real[:, 1] = np.clip(gt_real[:, 1], -45, 44) + 45
    gt_real[:, 2] = np.clip(gt_real[:, 2], -25, 24) + 25
    labels = np.stack([gt_real, gt_syn, diff], 2)
    labels = Variable(torch.Tensor(labels).cuda())

    predicted_azi, predicted_ele, predicted_tilt = py_net.forward(feature_i, feature_j, mask, optical_flow,
                                                                  knn_inds / (224.0 / 28), visibility)

    pred_angle[:, 0] = np.squeeze(np.argmax(predicted_azi.data.cpu().numpy(), 2))
    pred_angle[:, 1] = np.squeeze(np.argmax(predicted_ele.data.cpu().numpy(), 2) - 45)
    pred_angle[:, 2] = np.squeeze(np.argmax(predicted_tilt.data.cpu().numpy(), 2) - 25)

    loss_azi, loss_ele, loss_tilt = py_net.loss_all(predicted_azi, predicted_ele, predicted_tilt, labels)
    error_rotmat = np.median(utils.dist_rotmat(pred_angle, gt))

    return loss_azi, loss_ele, loss_tilt, error_rotmat



def log_errors(error_rotmat):
    summary_writer_common.summary_writer.add_scalar_summary('error_median_rotmat', error_rotmat)


def log_losses(loss_azi, loss_ele, loss_tilt, val=False):
    summary_writer_common.summary_writer.add_scalar_summary('loss_azi', loss_azi, val)
    summary_writer_common.summary_writer.add_scalar_summary('loss_ele', loss_ele, val)
    summary_writer_common.summary_writer.add_scalar_summary('loss_tilt', loss_tilt, val)

def log_best_error(py_net, optimizer, val_error, best_val_error, name, act_iter):
    if val_error < best_val_error:
        best_val_error = val_error
        dict_to_save = {'net_parameters': py_net.cpu().state_dict(),
                        'optimizer_state': optimizer.state_dict()}

        torch.save(dict_to_save, os.path.join(paths.path_to_saved_model, 'train_best_valid_' + name + '.pth'))

        best_val_iter = act_iter

if __name__ == '__main__':
    main()