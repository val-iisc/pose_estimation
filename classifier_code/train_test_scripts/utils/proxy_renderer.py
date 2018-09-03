import os
import numpy as np
import scipy.misc

from configs import constants
import io_utils
from configs.paths import render_loc, alph_loc, loc_map, FLOW_PATH, DEPTH_LOCATION, class_map, loc_map_with_model


def get_optical_flow_file(path):
    file = os.path.join(FLOW_PATH, path)
    file = file.split('/')
    file = '/'.join(file[:-1]) + '/' + file[-1].split('.')[0] + '.npy'

    return file


def get_param_list(img_name):
    l = img_name.split('_')[2:5]
    l = [l[1:].replace('@', '.') for l in l]
    l = [float(l) for l in l]
    return l


def get_all_rendered_images(class_name, avoid_big_e_t=True, lim=20.0):
    loc = os.path.join(render_loc)

    alpha_loc = os.path.join(alph_loc)
    img_list = os.listdir(loc)
    img_list = [x for x in img_list if loc_map[class_name] in x]

    def cond(x, lim=lim):
        a = get_param_list(x)
        if abs(a[1]) > lim or abs(a[2]) > lim:
            return False
        else:
            return True

    if avoid_big_e_t:
        param_list = [get_param_list(x) for x in img_list if cond(x)]
        param_dict = {tuple(get_param_list(x)): x for x in img_list if cond(x)}
    else:
        param_list = [get_param_list(x) for x in img_list]
        param_dict = {tuple(get_param_list(x)): x for x in img_list}
    param_np = np.array(param_list)

    return param_np, param_dict, loc, alpha_loc, class_name


def get_pt(file_name, class_name):
    all_coords = np.loadtxt(os.path.join('anchors', loc_map_with_model[class_name], 'anchors.txt'), delimiter=' ').transpose()
    final_array = np.zeros((1, class_map[class_name], 3))
    file_dict = io_utils.get_params_from_file(file_name)
    points_array = all_coords.transpose()
    truth, crop_pt_array = io_utils.get_render_array(file_dict, points_array, DEPTH_LOCATION)
    final_array[0, :, :2] = crop_pt_array[:2, :].transpose()
    final_array[0, :, 2] = truth
    return final_array


def get_syn_images_per_batch_wt_name_and_pt(view_params, add_info):
    img_list = []
    alpha_list = []
    optical_list = []
    actual_angle_list = []
    file_names = []
    pt_list = []
    for batch in range(view_params.shape[0]):
        params = view_params[batch]
        rbg_flipped_img, alpha_map, optical_map, actual_angle, filer = proxy_render(params, add_info[0], add_info[1],
                                                                                    add_info[2], add_info[3], True)
        file_names.append(filer)
        pt = get_pt(filer, add_info[-1])
        img_list.append(rbg_flipped_img)
        alpha_list.append(alpha_map)
        optical_list.append(optical_map)
        actual_angle_list.append(actual_angle)
        pt_list.append(pt)

    return img_list, alpha_list, optical_list, actual_angle_list, file_names, pt_list


def get_syn_images_per_batch(view_params, add_info):
    img_list = []
    alpha_list = []
    optical_list = []
    actual_angle_list = []
    for batch in range(view_params.shape[0]):
        params = view_params[batch]
        rbg_flipped_img, alpha_map, optical_map, actual_angle = proxy_render(params, add_info[0], add_info[1],
                                                                             add_info[2], add_info[3])
        img_list.append(rbg_flipped_img)
        alpha_list.append(alpha_map)
        optical_list.append(optical_map)
        actual_angle_list.append(actual_angle)

    return img_list, alpha_list, optical_list, actual_angle_list


def proxy_render(view_params, param_np, param_dict, loc, alpha_loc, name=False):
    diff_np = param_np - np.array(view_params)
    diff_np = normalize(diff_np)
    # give priority to Azi
    diff_np[:, 1:] /= 8.0
    truth_list = np.linalg.norm(diff_np, axis=1)
    angle = param_np[np.argmin(truth_list)]

    file_name = param_dict[tuple(angle)]

    img = (scipy.misc.imread(os.path.join(loc, file_name)).astype('float') - constants.PIXEL_MEANS).transpose(2, 0, 1)

    alpha = np.expand_dims(np.load(alpha_loc + '/' + file_name.split('.')[0] + '.npy'), 2)

    opt = get_optical_flow_file(file_name)

    opt = np.load(os.path.join(opt)).transpose(2, 0, 1)

    angle = get_canonical_angle_syn(file_name)
    if name:
        return img, alpha, opt, angle, file_name
    else:
        return img, alpha, opt, angle


def get_canonical_angle_syn(img_name):
    angles = img_name.split('_')[2:5]
    angles = [float(a[1:].replace('@', '.')) for a in angles]

    angles = [float(a) for a in angles]
    angles[2] = angles[2] % 360
    if angles[2] > 180:
        angles[2] = -(360 - angles[2])
    elif angles[2] < -180:
        angles[2] = 360 + angles[2]
    return angles


def normalize(delta):
    delta_azi_ind = delta[:, 0]
    delta[delta_azi_ind >= 180, 0] -= 360
    delta[delta_azi_ind < -180, 0] += 360

    delta_ele_ind = delta[:, 1]
    delta[delta_ele_ind >= 90, 1] -= 180
    delta[delta_ele_ind < -90, 1] += 180

    delta_til_ind = delta[:, 2]
    delta[delta_til_ind >= 180, 2] -= 360
    delta[delta_til_ind < -180, 2] += 360

    return delta


def main():
    name = 'chair'


if __name__ == '__main__':
    main()
