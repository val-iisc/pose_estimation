# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Christopher B. Choy
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    'float32', order='C')
    for i in xrange(num_images):
        im = ims[i]
        # TODO no indexing required
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def gray_im_list_to_blob(ims):
    """Convert a list of images into a network input.
    """
    blob = np.zeros((len(ims), 1, ims[0].shape[0], ims[0].shape[0]),
                    'float32', order='C')
    blob[:, 0, : :] = np.array(ims)

    return blob

def blob_to_im_list(blob):
    ims = []
    for datum in blob:
        ims.append(np.clip(datum.transpose(1,2,0) +
                    cfg.PIXEL_MEANS, 0, 255).astype(np.uint8)[:,:,::-1])
    return ims

def label_list_to_blob(labels):
    max_shape = np.array([label.shape for label in labels]).max(axis=0)
    num_images = len(labels)
    blob = np.zeros((num_images, 1, max_shape[0], max_shape[1]),
                    'float32', order='C')
    for i in xrange(num_images):
        label = labels[i]
        # TODO no indexing required
        blob[i, 0, 0:label.shape[0], 0:label.shape[1]] = label

    # Axis order: (batch elem, channel, height, width)
    return blob

def similarity_to_blob(similarities):
    batch_size = len(similarities)
    blob = np.zeros((batch_size, 1, 1, 1),
                    'float32', order='C')
    blob[:, 0, 0, 0] = similarities

    # Axis order: (batch elem, channel, height, width)
    return blob

def single_coord_list_to_blob(coords):
    max_corrs = max([len(coord) for coord in coords])
    if max_corrs == 0:
        print 'Empty correspondence blob detected'

    num_batch = len(coords)
    blob = np.zeros((num_batch, max_corrs, coords[0].shape[1]), 'float32', order='C')
    for batch_ind in xrange(num_batch):
        num_pairs = len(coords[batch_ind])
        if num_pairs > 0:
            blob[batch_ind, :num_pairs] = coords[batch_ind]
    return blob

def coord_list_to_blob(coords1, coords2, similarities=None):
    # Assume that coords have the same number if pairs
    # for batch_ind in xrange(len(coords1)):
    #     assert(len(coords1[batch_ind]) == len(coords2[batch_ind]))
    max_corrs = max([len(coord1) for coord1 in coords1])
    lengths = [len(coord1) for coord1 in coords1]
    #print ('no. of corr points', lengths)
    if max_corrs == 0:
        print 'Empty correspondence blob detected'

    num_batch = len(coords1)
    blob = np.zeros((num_batch, max_corrs, 5), 'float32', order='C')
    num_coord_blob = np.zeros((num_batch, 1), 'float32', order='C')
    for batch_ind in xrange(num_batch):
        num_pairs = len(coords1[batch_ind])
        num_coord_blob[batch_ind] = num_pairs
        if num_pairs > 0:
            blob[batch_ind, 0:num_pairs, 0:2] = coords1[batch_ind]
            blob[batch_ind, 0:num_pairs, 2:4] = coords2[batch_ind]
            if len(similarities[batch_ind]) > 0 :
                blob[batch_ind, 0:num_pairs, 4] = similarities[batch_ind]

            # Set the similarity to be -1 to indicate that the rest is empty
            blob[batch_ind, num_pairs:, 4] = -1
    return blob, num_coord_blob

def coord_list_to_blob_orig(coords1, coords2, similarities, coords2_orig):
    # Assume that coords have the same number if pairs
    # for batch_ind in xrange(len(coords1)):
    #     assert(len(coords1[batch_ind]) == len(coords2[batch_ind]))
    max_corrs = max([len(sim) for sim in similarities])
    if max_corrs == 0:
        print 'Empty correspondence blob detected'

    num_batch = len(coords1)
    blob = np.zeros((num_batch, max_corrs, 7), 'float32', order='C')
    num_coord_blob = np.zeros((num_batch, 1), 'float32', order='C')
    for batch_ind in xrange(num_batch):
        num_pairs = len(coords1[batch_ind])
        num_coord_blob[batch_ind] = num_pairs
        if num_pairs > 0:
            blob[batch_ind, 0:num_pairs, 0:2] = coords1[batch_ind]
            blob[batch_ind, 0:num_pairs, 2:4] = coords2[batch_ind]
            blob[batch_ind, 0:num_pairs, 4] = similarities[batch_ind]
            blob[batch_ind, 0:num_pairs, 5:7] = coords2_orig[batch_ind]

        # Set the similarity to be -1 to indicate that the rest is empty
        blob[batch_ind, num_pairs:, 4] = -1

    return blob, num_coord_blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def blob_to_feature(feature_blob):
    # data x channel x height x width to data x height x width x channel
    channel_swap = (0, 2, 3, 1)
    return feature_blob.transpose(channel_swap)
