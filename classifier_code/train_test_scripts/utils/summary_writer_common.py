from train_test_scripts.utils.summary_iteration_folders import Summary
from configs.config import ispa_net
from configs import paths
from configs import constants
if ispa_net:
    from ispa_net_configs import train_config
else:
    from multi_view_configs import train_config
summary_writer = Summary(train_config.start_iteration, paths.path_to_log, constants.PIXEL_MEANS, train_config.val_iters_after)