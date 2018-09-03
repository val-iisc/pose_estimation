from tensorboardX import SummaryWriter
import os
import torch
import numpy as np


class Summary():

    def __init__(self, start_iteration, path_to_log, pixel_means, val_iters_after):

        self.val_iters_after = val_iters_after

        self.start_iteration = start_iteration

        self.path_to_log = path_to_log

        self.pixel_means = pixel_means

        train_writer_path = os.path.join(self.path_to_log, 'train')

        val_writer_path = os.path.join(self.path_to_log, 'val')

        if not os.path.exists(train_writer_path):
            os.system('mkdir ' + train_writer_path)

        if not os.path.exists(val_writer_path):
            os.system('mkdir ' + val_writer_path)

        self._train_writer = SummaryWriter(train_writer_path)

        self._val_writer = SummaryWriter(val_writer_path)

        self._dict = {}

        self._dict_val = {}

    '''
    pass a single channel tensor of hxw or a 3 channel image of hxwx3 or a 1 channel image of hxw
    if all 3 bools are set to false, then it is like directly passing the image
    '''

    def add_image_summary(self, img, var_name, tensor=False, pixel_means=False, stack=False, val=False):

        if val:
            dict_to_use = self._dict_val
        else:
            dict_to_use = self._dict

        if var_name not in dict_to_use.keys():
            if val == False:
                dict_to_use[var_name] = self.start_iteration
            else:
                dict_to_use[var_name] = self.start_iteration * 1.0 / self.val_iters_after

        if tensor:
            img = torch.stack((img, img, img), dim=2).float()
            img = img.transpose(2, 0).transpose(1, 2)
        else:
            if stack:
                img = np.stack((img, img, img), axis=-1)
            else:
                if pixel_means:
                    img = (img.transpose(1, 2, 0) + self.pixel_means).astype('uint8')
                else:
                    img = img.transpose(1, 2, 0).astype('uint8')

        if val == True:
            iter_no = int(dict_to_use[var_name] * self.val_iters_after)
            self._val_writer.add_image(var_name, img, iter_no)
        else:
            iter_no = dict_to_use[var_name]
            self._train_writer.add_image(var_name, img, iter_no)

        dict_to_use[var_name] += 1

    '''
    pass the scalar value and the variable name
    '''

    def add_scalar_summary(self, var_name, scalar, val=False):

        if val:
            dict_to_use = self._dict_val
        else:
            dict_to_use = self._dict

        if var_name not in dict_to_use.keys():
            if val == False:
                dict_to_use[var_name] = self.start_iteration
            else:
                dict_to_use[var_name] = self.start_iteration * 1.0 / self.val_iters_after

        # print "name", var_name
        # print "count", self._dict[var_name]
        if val == True:
            iter_no = int(dict_to_use[var_name] * self.val_iters_after)

            self._val_writer.add_scalar(var_name, scalar, iter_no)
        else:
            iter_no = dict_to_use[var_name]

            self._train_writer.add_scalar(var_name, scalar, iter_no)

        dict_to_use[var_name] += 1
