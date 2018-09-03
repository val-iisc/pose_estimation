from copy import deepcopy
import numpy as np
import lib.config as config
import dataset_utils

dict_models = config.dict_models
model_map = config.model_map


class KITTIFlowDB:
    set_class = []
    set_distributions = {}

    def __init__(self, phase):

        # new loader
        self.set_data = {'train': deepcopy(dict_models), 'train_indices': deepcopy(dict_models),
                         'length_classes': deepcopy(dict_models), 'test': deepcopy(dict_models),
                         'val': deepcopy(dict_models), 'syn': deepcopy(dict_models)}
        self.num_classes = len(dict_models.keys())


        for x in dict_models.keys():
            train_image_paths, val_image_paths, test_image_paths, syn_image_paths = dataset_utils.get_all_image_paths(
                class_name=x)

            self.set_data['train'][x].extend(train_image_paths)
            self.set_data['length_classes'][x] = len(train_image_paths)
            self.set_data['train_indices'][x].extend(self.get_random_indices(self.set_data['length_classes'][x]))
            self.set_data['val'][x].extend(val_image_paths)
            self.set_data['test'][x].extend(test_image_paths)
            self.set_data['syn'][x].extend(syn_image_paths)





    def get_random_indices(self, length):
        return np.random.choice(np.arange(length), length)

    def has_next(self, phase, class_name):
        return len(self.set_data[phase][class_name]) > 0

    def get_set(self, phase, repeat=False):
        """Data pair generator. Generates a pair of data from a same set.
        Guarantees to iterate over all combinations of pairs of all set.
        If repeat is True, indefinitely iterates over all pairs.
        Yields ((data1_fn, data1_meta), (data2_fn, data2_meta), set_class).
        """

        c_name = 0

        while self.has_next(phase.lower(), config.model_map[c_name]) or repeat:
            class_name = config.model_map[c_name]
            # If all combinations of all set has been seen, stop or initialize.
            if len(self.set_data['train_indices'][class_name]) == 0:
                self._initialize_iterations(class_name)

            # Randomly choose a set.
            if phase == 'train':
                real_image_idx = self.set_data['train_indices'][class_name][0]
                self.set_data['train_indices'][class_name].pop(0)
            elif phase == 'val':
                real_image_idx = np.random.choice(range(len(self.set_data['val'][class_name])))
            else:
                real_image_idx = len(self.set_data['test'][class_name]) - 1

            real_image_path = self.set_data[phase][class_name][real_image_idx][0]
            real_image_azimuth = int(self.set_data[phase][class_name][real_image_idx][1])

            syn_image_idx = np.random.choice(range(len(self.set_data['syn'][class_name])))
            syn_image_path = self.set_data['syn'][class_name][syn_image_idx][0]
            syn_image_azimuth = self.set_data['syn'][class_name][syn_image_idx][1]



            coord1, coord2 = dataset_utils.get_keypoints(real_image_path, syn_image_path)

            coord1 = coord1.T
            coord2 = coord2.T

            if len(coord1) != 0:
                keypts_status1 = np.ones(len(coord1[0])).astype(bool)
                keypts_status2 = np.ones(len(coord2[0])).astype(bool)
            else:
                keypts_status1 = -1
                keypts_status2 = -1

            c_name = (c_name + 1) % self.num_classes

            yield [(real_image_path, real_image_azimuth, (coord1, keypts_status1)),
                   (syn_image_path, syn_image_azimuth, (coord2, keypts_status2))]

    def _initialize_iterations(self, class_name):

        for x in dict_models.keys():
            self.set_data['train_indices'][x].extend(self.get_random_indices(self.set_data['length_classes'][x]))
