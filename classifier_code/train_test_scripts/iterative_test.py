import __init__paths
import numpy as np
import torch

from ispa_net_configs import test_config, ucn_config
import utils.utils as utils
from arch.classifier_single import Classifier

import arch.acmm_net.net_final_old as nnew
import utils.data_loader as dl
import utils.proxy_renderer as pr

batch_size = test_config.batch_size

a = []
for x in range(10):
    a.append(np.exp(-x))


def main():
    torch.cuda.set_device(test_config.gpu)

    py_net = nnew.Net(20)

    file_name = test_config.model_to_load
    dict_to_load = torch.load(file_name, map_location=lambda storage, loc: storage)

    py_net.load_state_dict(dict_to_load['net_parameters'])

    py_net.cuda()

    add_info = pr.get_all_rendered_images(test_config.class_name)

    test_list = dl.all_images_list(train=False, pascal=True, objectnet=False, class_name=test_config.class_name)

    test_list_size = len(test_list)

    rem = test_list_size % batch_size

    utils.create_prototxt(batch_size, rem)
    caffe_model = Classifier('./protos/train_test.prototxt', ucn_config.dict_models[test_config.class_name], test_config.gpu)
    caffe_model_rem = Classifier('./protos/train_test_rem.prototxt', ucn_config.dict_models[test_config.class_name], test_config.gpu)

    test(py_net, caffe_model, test_list, add_info, caffe_model_rem=caffe_model_rem)


def test(py_net, caffe_model, test_list, add_info, caffe_model_rem=None):
    py_net.eval()
    for param in py_net.parameters():
        param.volatile = True
        param.requires_grad = False
    global batch_size
    input_fetcher = dl.batch_data_processor_real_unique_all(test_list, batch_size, augment=False)

    all_error = np.zeros([0, test_config.num_iters_iterative_val + 1])


    for act_iter, input_fetch in enumerate(input_fetcher):

        if len(input_fetch[0]) < batch_size:
            if len(input_fetch[0]) == 0:
                break
            print "now for remaining"
            caffe_model = caffe_model_rem
            batch_size = len(input_fetch[0])
        print "act_iter", act_iter

        real_img = np.array(input_fetch[0])

        gt_real = np.array(input_fetch[1])

        bin_img, real_features = caffe_model.classify(real_img)


        feature_i = torch.autograd.Variable(torch.FloatTensor(real_features).cuda())

        pred_angle = np.array([[float(bin_img[i] * 22.5 + 22.5 / 2), 10, 0] for i in range(batch_size)])

        error = np.abs([(gt_real[i, 0] - pred_angle[i][0]) % 360 for i in range(batch_size)])
        error[error > 180] = 360 - error[error > 180]

        error_iter = np.reshape(error, [batch_size, 1])

        for iter in range(test_config.num_iters_iterative_val):
            input_list = pr.get_syn_images_per_batch(pred_angle, add_info)
            feature_j, mask, optical_flow, gt_syn, syn_img = utils.preprocess(caffe_model, input_list)

            exit(0)

            predicted1, predicted2, predicted3 = py_net.forward(feature_i, feature_j, mask, optical_flow)

            pred = predicted1.data.cpu().numpy()
            pred = (np.argmax(pred, 2) - 45).astype(float)
            pred = utils.mu_inv(pred)

            pred_angle[:, 0] = (pred_angle[:, 0] + pred[:, 0]) % 360

            pred2 = predicted2.data.cpu().numpy()

            pred2 = (np.argmax(pred2, 2) - 10).astype(float)

            pred2 = utils.mu_inv(pred2, mu=10.0, in_range=10.0, out_range=45.0)

            pred_angle[:, 1] = (pred_angle[:, 1] + a[iter] * pred2[:, 0]) % 360

            pred_angle[:, 2] = (pred_angle[:, 2] + a[iter] * pred2[:, 1]) % 360

            dist = utils.dist_rotmat(pred_angle, gt_real)

            error = np.abs(dist)
            error_iter = np.concatenate((error_iter, np.reshape(error, [batch_size, 1])), axis=1)

        all_error = np.concatenate((all_error, error_iter), axis=0)

        if act_iter >= len(test_list) / batch_size:
            break

    all_error = np.array(all_error)

    print "median over iters", np.median(all_error, axis=0)

    print "acc pi/6", np.mean(all_error < 30, axis=0)

    print "acc pi/8", np.mean(all_error < 22.5, axis=0)

    print "acc pi/12", np.mean(all_error < 15, axis=0)

if __name__ == '__main__':
    main()