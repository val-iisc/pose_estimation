import lib.config as config
import numpy as np
import cv2
from lib.utils import image_utils

class ImageCorrespondenceTransformer(object):
    """
    Using the same small crop size for correspondence might yield too few
    correspondence
    """

    def __init__(self, phase):
        # Randomly perturb correspondences

        self.rng_crop = np.random.RandomState(1)
        self.rng_mirror = np.random.RandomState(2)
        self.rng_sample = np.random.RandomState(3)
        self.rng_blur = np.random.RandomState(4)
        self.rng_blur_56 = np.random.RandomState(5)
        self.crop_size = config.IM_SHAPE

        if phase == 'TRAIN':
            self.random_crop = True
            self.mirror = config.USE_FLIPPED
        else:
            self.random_crop = False
            self.mirror = False

    def cropped_coord_inds_set(self, coord, w_crop, h_crop):
        pos_inds, = np.where(np.all(coord > 0, axis=1))  # coordinate in real number not integer
        inside_h_inds, = np.where(np.round(coord[:, 0]) < w_crop)
        inside_w_inds, = np.where(np.round(coord[:, 1]) < h_crop)

        return set(pos_inds) & set(inside_h_inds) & set(inside_w_inds)

    def transform(self, img1, img2, coord1, coord2, sim=None, blur=False, crop_size=None):

        if crop_size is None:
            hcrop, wcrop = self.crop_size
        else:
            hcrop, wcrop = crop_size

        img1 = np.array(img1, dtype=np.float32)
        img2 = np.array(img2, dtype=np.float32)

        coord1 = np.array(coord1, dtype=np.float32)
        coord2 = np.array(coord2, dtype=np.float32)

        if coord1.shape[1] > 2:
            coord1 = coord1[:, :2]
            coord2 = coord2[:, :2]

        # CROP
        if self.crop_size is not None:
            h, w = img1.shape[:2]  # assumes that images have the same dims
            if h != hcrop or w != wcrop:
                if self.random_crop:
                    hoff = self.rng_crop.randint(0, h - hcrop + 1)
                    woff = self.rng_crop.randint(0, w - wcrop + 1)
                else:
                    hoff = (h - hcrop) / 2
                    woff = (w - wcrop) / 2
                img1 = img1[hoff:hoff + hcrop, woff:woff + wcrop]
                img2 = img2[hoff:hoff + hcrop, woff:woff + wcrop]

                coord1 -= [woff, hoff]
                coord2 -= [woff, hoff]

        # MIRROR
        if self.mirror:
            if self.rng_mirror.randint(0, 1):
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]

                if len(coord1) > 0:
                    coord1[:, 0] = wcrop - coord1[:, 0]
                    coord2[:, 0] = wcrop - coord2[:, 0]
        if blur:
            if self.rng_blur.randint(0, 1):
                if self.rng_blur_56.randint(0, 1):
                    img1 = cv2.resize(img1, (56, 56))
                else:
                    img1 = cv2.resize(img1, (112, 112))
                img1 = cv2.resize(img1, (224, 224))

        if config.ADD_RGB_JITTER:
            img1 = image_utils.rgb_jitter(img1)

            # SUBTRACT
        img1 -= config.PIXEL_MEANS
        img2 -= config.PIXEL_MEANS

        # SCALE INTENSITY
        if config.IMG_SCALE > 0:
            img1 *= config.IMG_SCALE
            img2 *= config.IMG_SCALE

        # Remove correspondences that fall outside the image.
        inds1 = self.cropped_coord_inds_set(coord1, wcrop - 1, hcrop - 1)
        inds2 = self.cropped_coord_inds_set(coord2, wcrop - 1, hcrop - 1)

        # Find common indices survived the cropping.
        common_inds = inds1 & inds2
        coord1 = np.array([coord1[c_i] for c_i in common_inds])
        coord2 = np.array([coord2[c_i] for c_i in common_inds])

        sample_ids = self.rng_sample.permutation(np.arange(len(coord1)))[:min(
            config.MAX_NUM_CORRESPONDENCE, len(coord1))]

        coord1 = coord1[sample_ids]
        coord2 = coord2[sample_ids]

        if sim is None:
            sim = np.ones(coord1.shape[0])
        else:
            sim = sim[sample_ids]

        return img1, img2, coord1, coord2, sim