import caffe

caffe.set_mode_gpu()

import numpy as np


class Classifier():

    def __init__(self, proto, ucn_model_path, gpu_id):
        caffe.set_device(gpu_id)

        self.weights_classifier = ucn_model_path

        self.caffe_net_classifier = caffe.Net(proto, self.weights_classifier, caffe.TEST)

    """
    img: shape batch_size,224,224,3
    """

    def classify(self, img):
        self.caffe_net_classifier.blobs['image'].data[...] = img
        self.caffe_net_classifier.forward()
        bin_img = np.argmax(self.caffe_net_classifier.blobs['fc_2'].data, axis=1)
        feature = self.get_feature_blob(img)

        return bin_img, feature

    """
    img: shape batch_size,224,224,3
    """

    def get_feature_blob(self, img):
        self.caffe_net_classifier.blobs['image'].data[...] = img
        self.caffe_net_classifier.forward()
        feature = self.caffe_net_classifier.blobs['feature1'].data

        return feature


if __name__ == '__main__':
    x = 5
