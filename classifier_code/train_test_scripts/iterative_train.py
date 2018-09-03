import __init__paths
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from configs import paths

from ispa_net_configs import train_config, ucn_config

import utils.utils as utils
from arch.classifier_single import Classifier

import utils.data_loader as dl
import utils.proxy_renderer as pr

from train_test_scripts.utils import summary_writer_common

import arch.acmm_net.net_final_old as nnew

batch_size = train_config.batch_size
best_val_error_rotmat = 100000

import caffe


def main():
    caffe.set_mode_gpu()
    caffe.set_device(train_config.gpu)
    torch.cuda.set_device(train_config.gpu)

    utils.create_prototxt(batch_size)
    caffe_model = Classifier('./protos/train_test.prototxt', ucn_config.dict_models[train_config.class_name], train_config.gpu)

    lr = 0.001

    py_net = nnew.Net(20)

    py_net.cuda()

    optimizer = optim.Adam(params=py_net.parameters(), lr=lr, weight_decay=0.0001)

    train_list, val_list = dl.all_images_list(train=True, pascal=True, objectnet=True,
                                              class_name=train_config.class_name)

    add_info = pr.get_all_rendered_images(train_config.class_name)

    trainval(py_net, caffe_model, train_list, val_list, optimizer, add_info)


def trainval(py_net, caffe_model, train_list, val_list, optimizer, add_info):
    input_fetcher = dl.batch_data_processor_real(train_list, batch_size, augment=True)
    global best_val_error_rotmat

    for act_iter, input_fetch in enumerate(input_fetcher):
        print "act_iter", act_iter
        real_img = np.array(input_fetch[0])
        gt_real = np.array(input_fetch[1])
        bin_img, real_features = caffe_model.classify(real_img)
        feature_i = torch.autograd.Variable(torch.FloatTensor(real_features).cuda())

        pred_angle = np.array([[float(bin_img[i] * 22.5 + 22.5 / 2), 10, 0] for i in range(batch_size)])

        pred_angle, diff_new = utils.get_adjusted_pred_uniform_mulaw(pred_angle, gt_real)

        input_list = pr.get_syn_images_per_batch(pred_angle, add_info)
        feature_j, mask, optical_flow, gt_syn, syn_img = utils.preprocess(caffe_model, input_list)
        pred_angle = np.copy(gt_syn)


        diff = utils.get_actual_diff(gt_real, gt_syn)

        labels = np.stack([gt_real, gt_syn, diff], 2)
        labels = Variable(torch.Tensor(labels).cuda())

        predicted1, predicted2, predicted3 = py_net.forward(feature_i, feature_j, mask, optical_flow)
        loss1, loss2, loss3, loss4 = py_net.loss_all(predicted1, predicted2, predicted3, labels)
        total_loss = loss1 + 0.5 * loss2 + 0.5 * loss3 + 0.5 * loss4
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        pred = predicted1.data.cpu().numpy()
        pred = (np.argmax(pred, 2) - 45).astype(float)
        pred = utils.mu_inv(pred)
        pred_angle[:, 0] = (pred_angle[:, 0] + pred[:, 0]) % 360
        pred2 = predicted2.data.cpu().numpy()
        pred2 = (np.argmax(pred2, 2) - 10).astype(float)

        pred2 = utils.mu_inv(pred2, mu=10.0, in_range=10.0, out_range=45.0)

        pred_angle[:, 1:] = (pred_angle[:, 1:] + pred2[:, 0:]) % 360

        pred3 = predicted3.data.cpu().numpy()
        pred3 = (np.argmax(pred3, 2)).astype(float)

        error_aux = np.abs([(gt_real[i, 0] - pred3[i][0]) % 360 for i in range(batch_size)])
        error_aux[error_aux > 180] = 360 - error_aux[error_aux > 180]

        loss_azi = loss1.data.cpu().numpy()
        loss_ele = loss2.data.cpu().numpy()
        loss_tilt = loss3.data.cpu().numpy()
        loss_aux = loss4.data.cpu().numpy()
        loss_tot = total_loss.data.cpu().numpy()

        print("total_loss", loss_tot[0])

        summary_writer_common.summary_writer.add_scalar_summary('loss_azi', loss_azi)
        summary_writer_common.summary_writer.add_scalar_summary('loss_ele', loss_ele)
        summary_writer_common.summary_writer.add_scalar_summary('loss_aux', loss_aux)
        summary_writer_common.summary_writer.add_scalar_summary('loss_tilt', loss_tilt)
        summary_writer_common.summary_writer.add_scalar_summary('loss_total', loss_tot)

        dist = utils.dist_rotmat(pred_angle, gt_real)

        summary_writer_common.summary_writer.add_scalar_summary('error_median_rotmat', np.median(dist))

        if act_iter % train_config.val_iters_after == 0:

            val_error = test(py_net, caffe_model, val_list, add_info)

            if val_error < best_val_error_rotmat:
                best_val_error_rotmat = val_error
                dict_to_save = {'net_parameters': py_net.cpu().state_dict(),
                                'optimizer_state': optimizer.state_dict()}

                torch.save(dict_to_save, os.path.join(paths.path_to_saved_model, 'train_best_valid_rotmat' + '.pth'))

                best_val_iter = train_config.start_iteration + act_iter
                file = open(os.path.join(paths.path_to_log, 'best_valid_rotmat.txt'), "w")
                file.write("best iter " + str(best_val_iter) + '\n')
                file.write("best val_error " + str(best_val_error_rotmat) + '\n')
                file.close()

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
    loss_tot = 0
    loss_aux = 0
    all_error_rotmat = 0

    for act_iter, input_fetch in enumerate(input_fetcher):

        print "validation act_iter", act_iter
        real_img = np.array(input_fetch[0])
        gt_real = np.array(input_fetch[1])
        bin_img, real_features = caffe_model.classify(real_img)

        feature_i = torch.autograd.Variable(torch.FloatTensor(real_features).cuda())

        pred_angle = np.array([[float(bin_img[i] * 22.5 + 22.5 / 2), 10, 0] for i in range(batch_size)])

        input_list = pr.get_syn_images_per_batch(pred_angle, add_info)
        feature_j, mask, optical_flow, gt_syn, syn_img = utils.preprocess(caffe_model, input_list)

        diff = utils.get_actual_diff(gt_real, gt_syn)
        labels = np.stack([gt_real, gt_syn, diff], 2)
        labels = Variable(torch.Tensor(labels).cuda())

        predicted1, predicted2, predicted3 = py_net.forward(feature_i, feature_j, mask, optical_flow)
        loss1, loss2, loss3, loss4 = py_net.loss_all(predicted1, predicted2, predicted3, labels)
        total_loss = loss1 + 0.5 * loss2 + 0.5 * loss3 + 0.5 * loss4

        pred = predicted1.data.cpu().numpy()
        pred = (np.argmax(pred, 2) - 45).astype(float)
        pred = utils.mu_inv(pred)
        pred_angle[:, 0] = (pred_angle[:, 0] + pred[:, 0]) % 360
        pred2 = predicted2.data.cpu().numpy()
        pred2 = (np.argmax(pred2, 2) - 10).astype(float)
        pred2 = utils.mu_inv(pred2, mu=10.0, in_range=10.0, out_range=45.0)

        pred_angle[:, 1:] = (pred_angle[:, 1:] + pred2[:, 0:]) % 360

        dist = utils.dist_rotmat(pred_angle, gt_real)

        loss_azi += loss1.data.cpu().numpy()
        loss_ele += loss2.data.cpu().numpy()
        loss_tilt += loss3.data.cpu().numpy()
        loss_aux += loss4.data.cpu().numpy()
        loss_tot += total_loss.data.cpu().numpy()

        all_error_rotmat += np.mean(dist)

        if act_iter == num_test_samples - 1:
            break

    summary_writer_common.summary_writer.add_scalar_summary('error_median_rotmat', np.mean(all_error_rotmat),
                                                            val=True)
    summary_writer_common.summary_writer.add_scalar_summary('loss_total', loss_tot / num_test_samples, val=True)
    summary_writer_common.summary_writer.add_scalar_summary('loss_azi', loss_azi / num_test_samples, val=True)
    summary_writer_common.summary_writer.add_scalar_summary('loss_tilt', loss_tilt / num_test_samples, val=True)
    summary_writer_common.summary_writer.add_scalar_summary('loss_aux', loss_aux / num_test_samples, val=True)
    summary_writer_common.summary_writer.add_scalar_summary('loss_ele', loss_ele / num_test_samples, val=True)

    py_net.train()
    for param in py_net.parameters():
        param.volatile = False
        param.requires_grad = True

    return all_error_rotmat / num_test_samples


if __name__ == '__main__':
    main()
