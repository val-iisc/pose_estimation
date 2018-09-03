import numpy as np
import yaml
import caffe
import cv2
import sys

import lib.config as cfg

#it = 0

class CorrespondenceContrastiveLosslayer(caffe.Layer):
    """
    Given activations from a bottom layer and feature pair coordinates, compute
    the L1, L2, norms and return the loss.

    Currently L2 norm
    """
    def _transform_coordinate(self, x):
        # return (x + 0.5) / self._downsampling_factor - 0.5
        return x / self._downsampling_factor

    def setup(self, bottom, top):
        if len(bottom) < 3:
            raise Exception("Need two inputs to compute distance.")

        # Save pair coordinates for backward
        self._pair_coords = []
        layer_params = yaml.load(self.param_str)

        if layer_params.has_key('down_sampling_factor'):
            self._downsampling_factor = layer_params['down_sampling_factor']


        self._margin = layer_params['margin']


        self._pck_radii = cfg.PCK_RADII
        self._neg_mdist_sq = cfg.MARGIN_DIST_SQ
        if layer_params.has_key('bilinear_interpolation'):
            self._bilinear_interpolation = layer_params['bilinear_interpolation']
        else:
            self._bilinear_interpolation = cfg.BILINEAR_INTERPOLATION

        # L loss
        if layer_params.has_key('LN'):
            self._LN = layer_params['LN']
        else:
            self._LN = 2

        self._loss = 0
        self._loss_positive = 0
        self._loss_negative = 0
        #-1 if this computation is not required
        self._loss_only_positive = -1
        self.bf = cv2.BFMatcher()
        # loss output is scalar
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
        top[4].reshape(1)

    def reshape(self, bottom, top):
        # check input dimensions match
        pass
        # if bottom[0].count != bottom[1].count:
        #     raise Exception("Inputs must have the same dimension.")

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        #global it
        # Activation from the bottom layers
        features_i = bottom[0].data
        features_j = bottom[1].data
        batch_size, channels, height1, width1 = features_i.shape

        # consists of image index, h,w coordinate and corresponding feature
        # point image index and h,w
        pair_coords = bottom[2].data
        only_pos_corr = bottom[3].data
        # print ('in correspondance layer forward')
        # print ('features_i',features_i.shape)
        # print ('features_j',features_j.shape)
        # print ('pair_coords',pair_coords.shape)
        # print (pair_coords)

        loss = 0
        loss_positive = 0
        loss_negative = 0
        loss_only_positive = 0
        self._N = 0
        self._N_pos = 0

        for batch_ind in xrange(bottom[0].num):
            similarities = pair_coords[batch_ind, :, 4]
            valid_pair = similarities >= 0

            w_i_s, h_i_s, w_j_s, h_j_s, _ = \
                self._transform_coordinate(pair_coords[batch_ind, valid_pair]).T

            diff = self.bilinear_interpolate(features_i[batch_ind], w_i_s, h_i_s) \
                - self.bilinear_interpolate(features_j[batch_ind], w_j_s, h_j_s)

            dist_sq = np.sum(diff ** 2, axis=0)
            self._N += np.sum(valid_pair)
            # print ('batch: ', batch_ind, 'dist sq:', type(dist_sq), dist_sq.shape, dist_sq)
            # print ('batch: ', batch_ind, 'self._N: ', type(self._N), self._N.shape, self._N)
            similar_pair = similarities[valid_pair] == 1
            # print ('similar pairs', np.sum(similarities[valid_pair] == 1))
            dissimilar_pair = np.logical_not(similar_pair)

            loss += np.sum(dist_sq[similar_pair])
            loss_positive += np.sum(dist_sq[similar_pair])
            if self._neg_mdist_sq:
                temp = np.sum(np.maximum(self._margin - np.sqrt(dist_sq[dissimilar_pair]), 0) ** 2)
                loss += temp
                loss_negative += temp
            else:
                temp = np.sum(np.maximum(self._margin - dist_sq[dissimilar_pair], 0))
                loss += temp
                loss_negative += temp


                
        self._loss = loss / self._N / 2.
        self._loss_positive = loss_positive / self._N / 2.
        self._loss_negative = loss_negative / self._N / 2.

        top[0].data[0] = self._loss
        top[1].data[0] = self._N
        top[2].data[0] = self._loss_positive
        top[3].data[0] = self._loss_negative
        top[4].data[0] = 0

    def backward(self, top, propagate_down, bottom):
        # Activation from the bottom layers
        features_i = bottom[0].data
        features_j = bottom[1].data
        batch_size, channels, height1, width1 = features_i.shape

        # batch_size, n_channel, height, width = features_i.shape
        pair_coords = bottom[2].data
        
        only_pos_corr = bottom[3].data
        
        for i in range(2):
            # Initialize all the gradients to 0. We will accumulate gradient
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

        # Scale gradient using top diff (loss_weight)
        alphas = [(1 if i == 0 else -1) * top[0].diff[0] / self._N for i in range(2)]
        

        for batch_ind in xrange(bottom[0].num):
            similarities = pair_coords[batch_ind, :, 4]
            valid_pair = similarities >= 0
            w_i_s, h_i_s, w_j_s, h_j_s, _ = \
                self._transform_coordinate(pair_coords[batch_ind, valid_pair]).T

            diff = self.bilinear_interpolate(features_i[batch_ind], w_i_s, h_i_s) \
                - self.bilinear_interpolate(features_j[batch_ind], w_j_s, h_j_s)

            dist_sq = np.sum(diff ** 2, axis=0)
            self._N += np.sum(valid_pair)
            similar_pair = similarities[valid_pair] == 1
            bottom[0].diff[batch_ind] = self.add_by_bilinear_interpolation(bottom[0].diff[batch_ind], w_i_s[similar_pair], h_i_s[similar_pair], alphas[0] * diff[:, similar_pair])
            bottom[1].diff[batch_ind] = self.add_by_bilinear_interpolation(bottom[1].diff[batch_ind], w_j_s[similar_pair], h_j_s[similar_pair], alphas[1] * diff[:, similar_pair])


            dist = np.sqrt(dist_sq)
            mdist = self._margin - dist
            dissimilar_pair = np.logical_not(similar_pair)

            neg_pair = np.logical_and(mdist > 0, dissimilar_pair)
            if self._neg_mdist_sq:
                beta = -alphas[0] * mdist[neg_pair] / (dist[neg_pair] + 1e-8)
            else:
                beta = -alphas[0]

            bottom[0].diff[batch_ind] = self.add_by_bilinear_interpolation(bottom[0].diff[batch_ind], w_i_s[neg_pair], h_i_s[neg_pair], beta * diff[:, neg_pair])
            bottom[1].diff[batch_ind] = self.add_by_bilinear_interpolation(bottom[1].diff[batch_ind], w_j_s[neg_pair], h_j_s[neg_pair], - beta * diff[:, neg_pair])

    def bilinear_weights(self, width, height, x, y):
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        wx0 = x1 - x
        wx1 = x - x0
        wy0 = y1 - y
        wy1 = y - y0

        # If both coordinates are the same, they are both outside the image and
        # wx0 and wx1 will both be 0. Set one of them to be 1.
        x_same = x1 == x0
        wx0[x_same] = 1
        wx1[x_same] = 0
        y_same = y1 == y0
        wy0[y_same] = 1
        wy1[y_same] = 0

        return x0, x1, y0, y1, wx0, wx1, wy0, wy1

    def bilinear_interpolate(self, im, x, y):
        """For c x h x w representation, use bilinear interpolation to extract values"""
        x0, x1, y0, y1, wx0, wx1, wy0, wy1 = \
            self.bilinear_weights(im.shape[2], im.shape[1], x, y)

        I00 = im[:, y0, x0]
        I10 = im[:, y1, x0]
        I01 = im[:, y0, x1]
        I11 = im[:, y1, x1]

        # return wa*Ia + wb*Ib + wc*Ic + wd*Id
        return wy0 * wx0 * I00 + wy1 * wx0 * I10 + wy0 * wx1 * I01 + wy1 * wx1 * I11

    def add_by_bilinear_interpolation(self, im, x, y, feat):
        x0, x1, y0, y1, wx0, wx1, wy0, wy1 = \
            self.bilinear_weights(im.shape[2], im.shape[1], x, y)

        f00 = wy0 * wx0 * feat
        f10 = wy1 * wx0 * feat
        f01 = wy0 * wx1 * feat
        f11 = wy1 * wx1 * feat

        for i in xrange(len(x)):
            im[:, y0[i], x0[i]] += f00[:, i]
            im[:, y1[i], x0[i]] += f10[:, i]
            im[:, y0[i], x1[i]] += f01[:, i]
            im[:, y1[i], x1[i]] += f11[:, i]

        return im
