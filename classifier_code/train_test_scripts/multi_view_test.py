import __init__paths

import numpy as np
import torch
from torch.autograd import Variable

import arch.eccv_net.eccv_net as nnew
import utils.data_loader as dl
import utils.proxy_renderer as pr
import utils.utils as utils
from arch.classifier_single import Classifier
from multi_view_configs import test_config, ucn_config

batch_size = test_config.batch_size

angles = np.arange(0, 360, 15)[::24 / test_config.num_views]
import caffe

def main():
    caffe.set_mode_gpu()
    caffe.set_device(test_config.gpu)
    torch.cuda.set_device(test_config.gpu)

    py_net = nnew.Net(class_name=test_config.class_name)

    file_name = test_config.model_to_load
    dict_to_load = torch.load(file_name, map_location=lambda storage, loc: storage)

    py_net.load_state_dict(dict_to_load['net_parameters'])

    py_net.cuda()

    test_list = dl.all_images_list(train=False, pascal=True, objectnet=False,
                                   class_name=test_config.class_name)

    add_info = pr.get_all_rendered_images(test_config.class_name)

    test_list_size = len(test_list)

    rem = test_list_size % batch_size

    utils.create_prototxt(batch_size, rem)
    caffe_model = Classifier('./protos/train_test.prototxt', ucn_config.dict_models[test_config.class_name], test_config.gpu)
    caffe_model_rem = Classifier('./protos/train_test_rem.prototxt', ucn_config.dict_models[test_config.class_name], test_config.gpu)

    test(py_net, caffe_model, caffe_model_rem, test_list, add_info)


def test(py_net, caffe_model, caffe_model_rem, test_list, add_info):
    py_net.eval()

    for param in py_net.parameters():
        param.volatile = True
        param.requires_grad = False

    num_test_samples = int(len(test_list) * 1. / batch_size + 1)

    input_fetcher = dl.batch_data_processor_real_unique_all(test_list, batch_size, augment=False)

    all_error_rotmat = []

    for act_iter, input_fetch in enumerate(input_fetcher):
        print "act_iter", act_iter

        if len(input_fetch[0]) < batch_size:
            if len(input_fetch[0]) == 0:
                break
            print "now for remaining"
            caffe_model = caffe_model_rem

        loss_a, loss_e, loss_t, error_rotmat = get_losses_errors(input_fetch,
                                                                 caffe_model,
                                                                 add_info,
                                                                 py_net)

        all_error_rotmat.append(error_rotmat)

        if act_iter == num_test_samples - 1:
            break

    all_error_rotmat = np.concatenate(all_error_rotmat, 0).flatten()
    print "median_error", np.median(all_error_rotmat)
    print "acc pi/6_rotmat", np.mean(all_error_rotmat < 30)
    print "acc pi/8_rotmat", np.mean(all_error_rotmat < 22.5)
    print "acc pi/12_rotmat", np.mean(all_error_rotmat < 15)


def get_losses_errors(input_fetch, caffe_model, add_info, py_net):
    pred_angle = np.zeros([batch_size, 3])
    real_img = np.array(input_fetch[0])
    gt_real = np.array(input_fetch[1])
    gt = gt_real.copy()

    bin_img, real_features = caffe_model.classify(real_img)
    for x in range(test_config.num_views - 1):
        if x == 0:
            real_features_concat = np.stack((real_features, real_features), 1)
        else:
            real_features_concat = np.concatenate((real_features_concat, np.expand_dims(real_features, 1)), 1)
    feature_i = torch.autograd.Variable(torch.FloatTensor(real_features_concat).cuda())
    ct = 0

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
    error_rotmat = utils.dist_rotmat(pred_angle, gt)

    return loss_azi, loss_ele, loss_tilt, error_rotmat


if __name__ == '__main__':
    main()
