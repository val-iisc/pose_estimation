import caffe
import lib.config as config
import numpy as np
import yaml
import scipy.misc

from lib.utils.blob import im_list_to_blob, coord_list_to_blob
from lib.datasets.keypoint5_pascal3d import KITTIFlowDB

from augmenter import ImageCorrespondenceTransformer


class IntraClassCorrespondenceDataLayer(caffe.Layer):

    def has_next(self):
        self._corrdb.has_next(self._training_or_deploy)

    def setup(self, bottom, top):
        # print 'in data layer setup'
        """Setup the ImageDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        # Top layer mapping
        self._name_to_top_map = {
            'image_1': 0,
            'image_2': 1,
            'correspondence': 2,
            'num_coord': 3,
            'img_size': 4,
            'image_1_azimuth': 5,
            'image_2_azimuth': 6
        }

        # TODO get layer params directly. No python param for phase
        layer_params = yaml.load(self.param_str)

        self._training_or_deploy = layer_params['phase'].lower()
        self.batch_size = config.BATCH_SIZE
        # default is True
        self.repeat = layer_params.get('repeat', True)

        im_shape = config.IM_SHAPE

        # data blob: holds a batch of N images, each with 3 channels

        top[0].reshape(self.batch_size, 3, im_shape[0], im_shape[1])
        top[1].reshape(self.batch_size, 3, im_shape[0], im_shape[1])
        top[2].reshape(self.batch_size, 12, 5)

        # num correspondence layer
        top[3].reshape(self.batch_size)
        top[4].reshape(self.batch_size, 2)
        top[5].reshape(self.batch_size, 1)
        top[6].reshape(self.batch_size, 1)

        self.corrdb = KITTIFlowDB(self._training_or_deploy)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        global it
        # print 'just before next batch in forward'
        if self._training_or_deploy == 'train':
            if self.phase == 0:
                self.actual_phase = 'train'
            else:
                self.actual_phase = 'val'
        else:
            self.actual_phase = 'test'

        blobs = self.get_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]

            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob

    def get_minibatch(self):

        imgt = ImageCorrespondenceTransformer(self.actual_phase)
        ims1 = []
        ims2 = []
        img1_azimuths = []
        img2_azimuths = []
        coords1 = []
        coords2 = []
        classes = []
        similarities = []
        pair_gen = self.corrdb.get_set(self.actual_phase.lower(), repeat=self.repeat)

        for (imf1, img1_azimuth, (kp1, mask1)), (imf2, img2_azimuth, (kp2, mask2)) in pair_gen:

            im1 = scipy.misc.imread(imf1).astype(float)
            im2 = scipy.misc.imread(imf2).astype(float)

            if len(mask1) == 0 or len(mask2) == 0:  # Some may lack keypoint labels.
                continue
            mask = (mask1 & mask2).astype(bool)

            if not any(mask):  # Must have at least one overlapping keypoint.
                continue

            kp1, kp2 = kp1[:, mask].T, kp2[:, mask].T
            im1, im2, kp1, kp2, sim = imgt.transform(im1, im2, kp1, kp2, None, True)

            ims1.append(im1)
            ims2.append(im2)

            img1_azimuths.append(int((img1_azimuth%360)/(360.0/16)))
            img2_azimuths.append(int((img2_azimuth%360)/(360.0/16)))
            coords1.append(kp1)
            coords2.append(kp2)
            similarities.append(sim)

            if len(ims1) >= self.batch_size:
                break



        coord, num_coord = coord_list_to_blob(coords1, coords2, similarities)

        blobs = {'image_1': im_list_to_blob(ims1),
                 'image_2': im_list_to_blob(ims2),
                 'correspondence': coord,
                 'num_coord': num_coord,
                 'img_size': np.array([img.shape[:2] for img in ims2],
                                      dtype='float32', order='C'),
                 'image_1_azimuth': np.array(img1_azimuths),
                 'image_2_azimuth': np.array(img2_azimuths)
                 }

        return blobs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
