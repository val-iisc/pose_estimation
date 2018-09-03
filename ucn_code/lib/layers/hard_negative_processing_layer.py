import numpy as np
import yaml
import caffe

import lib.config as cfg
from lib.utils.timer import Timer
from lib.utils.blob import single_coord_list_to_blob

#it = 0

class HardNegativeEnum:
    NO_PROCESSING = 0  # Only used for evaluating PCK
    HARD_NEG_AND_POS = 1
    HARD_NEG_AND_RAND_NEG_AND_POS = 2
    HARD_NEG_TRIPLET = 3
    HARD_NEG_RETAIN_POS = 4


HardNegativeStr2Enum = {
    'NO_PROCESSING': HardNegativeEnum.NO_PROCESSING,
    'HARD_NEG_AND_POS': HardNegativeEnum.HARD_NEG_AND_POS,
    'HARD_NEG_AND_RAND_NEG_AND_POS': HardNegativeEnum.HARD_NEG_AND_RAND_NEG_AND_POS,
    'HARD_NEG_TRIPLET': HardNegativeEnum.HARD_NEG_TRIPLET,
    'HARD_NEG_RETAIN_POS' : HardNegativeEnum.HARD_NEG_RETAIN_POS
}


class HardNegativeProcessingLayer(caffe.Layer):
    """
    Given ground truth correspondence, 1-NN index from the KNN layer,
    process whether the pair should be a negative or positive.
    """
    def _transform_coord_im2feat(self, x):
        # print ('in hdp trans')
        """ image coord to feature coord"""
        # return (x + 0.5) / self._downsampling_factor - 0.5
        return x / self._downsampling_factor

    def _transform_coord_feat2im(self, x):
        """ transform feature coord to image coord """
        # print ('in hdp feat2im')
        return x * self._downsampling_factor

    def _ind2coord(self, ind, width):
        """ takse in num_batch x num_coords x k ind and genertes num_batch x
        num_coords x 2"""
        # print ('in hdp ind2cord')
        num_batch, num_coord, k = ind.shape

        # print (num_batch, num_coord, ind.shape)
        # print ('after ind ', ind)
        # k-th NN index: ind[:, k, :]
        x = ind % width
        y = np.floor(ind / width)
        # print (ind[0,0,0], x[0,0,0], y[0,0,0], width)
        xy_coords = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=3)
        # print ('xy ',xy_coords.shape, xy_coords)
        return xy_coords

    def _get_random_negatives(self, correspondences, num_coord_blob):


        num_batch = correspondences.shape[0]
        max_num_coord = correspondences.shape[1]
        new_correspondences = np.zeros((num_batch, max_num_coord*2, 5)).astype(np.float32)
        for index in range(num_batch):
            curr_num_coord = int(num_coord_blob[index][0])
            corr_pos = correspondences[index,:,:].copy()
            corr_neg = correspondences[index,:,:].copy()

            gt = corr_neg[:,2:4]

            rand_set = np.arange(curr_num_coord)
            self._permute(rand_set)
            
            #if the number of random negative is to be capped then these invalid inds can be changed
            invalid_inds = np.arange(curr_num_coord, max_num_coord)
            rand_set = np.append(rand_set, invalid_inds)
            gt_perm = gt[rand_set,:]

            test_x_same = gt_perm[:,0]==gt[:,0]
            test_y_same = gt_perm[:,1]==gt[:,1]

            invalids = np.append(np.where(test_x_same&test_y_same), invalid_inds)

            corr_neg[:,4] = 0
            corr_neg[:,2:4] = gt_perm
            corr_neg[invalids,4] = -1
            
            # corr_pos[invalids,4] = -1

            new_correspondences[index,:,:] = np.concatenate((corr_pos, corr_neg), axis = 0)

        return new_correspondences




   #fischer yates shuffle

    def _swap(self, xs, a, b):
        xs[a], xs[b] = xs[b], xs[a]
        
    def _permute(self, xs):
        for a in xrange(len(xs)-1):
            #print a
            b = np.random.choice(xrange(a+1, len(xs)))
            self._swap(xs, a, b)  
            
            
    def setup(self, bottom, top):
        # print ('in hdp setup')
        if len(bottom) != 4:
            raise Exception("Requires four blobs: original coordinate, KNN index, blob that was used for KNN reference (second image blob), and the blob for number of coords")

        # Save pair coordinates for backward
        self._pair_coords = []
        layer_params = yaml.load(self.param_str)

        self._downsampling_factor = layer_params['down_sampling_factor']
        self.negative_type = HardNegativeStr2Enum[layer_params['negative_type']]  # ['

        self._pck_radii = cfg.PCK_RADII

        # Generate random negatives
        # self._sample = False
        self._timer = Timer()

        # Set the correspondence coord output
        top[0].reshape(1)
        # Set the PCK output
        top[1].reshape(len(self._pck_radii))
        top[2].reshape(1)
        top[3].reshape(1)
        top[4].reshape(1)
        self._return_all_pcks = False
        #if len(top) > 4:
         #   self._return_all_pcks = True
          #  top[4].reshape(4)

    def reshape(self, bottom, top):
        # print ('in hdp reshape')
        correspondences = bottom[0].data
        
        #np.save('corr.npy', correspondences)
        

        coord_shape = correspondences.shape

        num_batch = coord_shape[0]
        max_num_coord = coord_shape[1]

        if self.negative_type == HardNegativeEnum.HARD_NEG_TRIPLET:
            # Triplet x_r, y_r, x_pos, y_pos, x_neg, y_neg, valid
            #top[0].reshape(num_batch, max_num_coord, 7)
            top[0].reshape(num_batch, max_num_coord*2, 5)
        elif self.negative_type == HardNegativeEnum.HARD_NEG_RETAIN_POS:
            top[0].reshape(num_batch, max_num_coord*2, 5)  
        elif self.negative_type == HardNegativeEnum.HARD_NEG_AND_RAND_NEG_AND_POS:
            top[0].reshape(num_batch, max_num_coord*2, 5)
        else:
            top[0].reshape(num_batch, max_num_coord, 5)
        
        top[4].reshape(num_batch, max_num_coord, 5)      
        #top[2].reshape(len(self._pck_radii), num_batch)

    def forward(self, bottom, top):
        # print ('in hdp forward')
        #global it
        """Get blobs and copy them into this layer's top blob vector."""
        # Activation from the bottom layers
        self._timer.tic()

        correspondences = bottom[0].data
        knn_inds = bottom[1].data
        features = bottom[2].data
        num_coord_blob = bottom[3].data

        coord_shape = correspondences.shape
        num_batch = coord_shape[0]
        max_num_coord = coord_shape[1]

        knn_inds_shape = knn_inds.shape
        knn = knn_inds_shape[1]
        _, _, height, width = features.shape


        feat2_xy_coord_knn = self._ind2coord(knn_inds - 1, width)
        img2_xy_coord_knn = self._transform_coord_feat2im(feat2_xy_coord_knn)
        img2_xy_coord_gt = correspondences[:, :, 2:4]

        feat2_xy_coord_gt = self._transform_coord_im2feat(img2_xy_coord_gt)

        if self.negative_type == HardNegativeEnum.HARD_NEG_AND_RAND_NEG_AND_POS:

            correspondences = self._get_random_negatives(correspondences, num_coord_blob)
            
        # copy the correspondences from the original blob so we can overwrite
        if self.negative_type == HardNegativeEnum.HARD_NEG_TRIPLET:
            hard_neg_corr_temp = np.ones((num_batch, max_num_coord, 7)).astype(np.float32)
            hard_neg_corr_temp[:, :, :4] = correspondences[:, :, :4]
            hard_neg_corr = np.zeros((num_batch, max_num_coord*2, 5)).astype(np.float32)
        elif self.negative_type == HardNegativeEnum.HARD_NEG_RETAIN_POS:
            hard_neg_corr = np.zeros((num_batch, max_num_coord*2, 5)).astype(np.float32) 
        else:
            hard_neg_corr = correspondences.copy()
            
        pos_corr = np.zeros((num_batch, max_num_coord, 5)).astype(np.float32)  
        num_hard_negative_for_random_neg = 0
        # If distance from the ground truth is small, consider they
        # are the true positive
        pcks = np.zeros((len(self._pck_radii), num_batch))

        eval_num_coords = []



        for n in xrange(num_batch):
            curr_num_coord = int(num_coord_blob[n][0])

            # Evaluate using ground truth correspondences
            where_ones = correspondences[n, :, 4] == 1
            where_minus_ones = correspondences[n, :, 4] == -1

            eval_coord_indicator = where_ones|where_minus_ones
            eval_num_coords.append(np.sum(where_ones))

            where_ones = where_ones[:max_num_coord]
            im_dist = np.sqrt(np.sum((img2_xy_coord_knn[n, 0, where_ones]
                - img2_xy_coord_gt[n, where_ones])**2, axis=1))
           

            for i, pck_radius in enumerate(self._pck_radii):
                pcks[i, n] = np.sum(im_dist < pck_radius)

            

                

            if self.negative_type == HardNegativeEnum.HARD_NEG_AND_RAND_NEG_AND_POS:

                eval_coord_inds = np.where(where_ones)[0]
                neg_coord_inds = np.where(np.logical_not(eval_coord_indicator))[0]

                # From the positive correspondences (eval_coord_inds), find hard negatives
                feat_dist = np.sqrt(np.sum(
                    (feat2_xy_coord_knn[n, 0, where_ones] - feat2_xy_coord_gt[n, where_ones])**2,
                    axis=1))

                hard_neg_inds = feat_dist > cfg.NEGATIVE_MINING_PIXEL_THRESH

                if len(hard_neg_inds) > 0:
                    # Put those hard negatives to the random negative pairs
                    hard_neg_coord_inds = eval_coord_inds[hard_neg_inds]
                    
                    max_possible_overwrite_inds = min(len(neg_coord_inds), len(hard_neg_coord_inds))
                    
                    hard_neg_coord_inds = hard_neg_coord_inds[:max_possible_overwrite_inds]

                    if len(neg_coord_inds) != 0 and len(hard_neg_coord_inds) !=0:
                        overwrite_coord_inds = np.random.choice(neg_coord_inds, len(hard_neg_coord_inds), replace=False)

                        # Overwrite the negatives
                        hard_neg_corr[n, overwrite_coord_inds] = hard_neg_corr[n, hard_neg_coord_inds]
                        hard_neg_corr[n, overwrite_coord_inds, 2:4] = img2_xy_coord_knn[n, 0, hard_neg_coord_inds]
                        hard_neg_corr[n, overwrite_coord_inds, 4] = 0

                    # Set the other correspondences to be -1
                    #hard_neg_corr[n, curr_num_coord:, 4] = -1
                    num_hard_negative_for_random_neg += len(hard_neg_coord_inds)

            positive_coords = correspondences[n,:max_num_coord,:].copy()
                
            positive_coords[curr_num_coord:, 4] = -1
                
            pos_corr[n,:,:] = positive_coords   

        top[0].data[...] = hard_neg_corr.astype(np.float32)
        top[1].data[...] = np.sum(pcks, axis=1) / np.sum(eval_num_coords)

        
       
        top[2].data[...] = np.sum(hard_neg_corr[:,:,4]==1)*1.0/num_batch
        top[3].data[...] = np.sum(hard_neg_corr[:,:,4]==0)*1.0/num_batch
        
        if self.negative_type == HardNegativeEnum.HARD_NEG_AND_RAND_NEG_AND_POS:
            top[3].data[...] = num_hard_negative_for_random_neg*1.0/num_batch
        else:
            top[3].data[...] = np.sum(hard_neg_corr[:,:,4]==0)*1.0/num_batch
        top[4].data[...] = pos_corr.astype(np.float32)

        self._timer.toc()

    def backward(self, top, propagate_down, bottom):
        """This layer does not backpropagate"""
        pass

    @property
    def average_time(self):
        # print ('in hdp avg')
        return self._timer.average_time
