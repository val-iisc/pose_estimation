import __init__paths
from configs.config import ispa_net
import configs.paths as paths

if ispa_net:
    from ispa_net_configs import train_config
else:
    from multi_view_configs import train_config
import os
import sys

if not os.path.exists(paths.path_to_saved_model):
    os.system('mkdir ' + paths.path_to_saved_model)

print "path_to_log:", paths.path_to_log

if os.path.exists(paths.path_to_log):
    choice = raw_input("Path exists Do you want to overwrite the summaries?y/n\n")
    if choice == 'y':
        os.system('rm -rf ' + paths.path_to_log)
    else:
        sys.exit(0)

os.system('mkdir ' + paths.path_to_log)

os.system('cp -r ./train_test_scripts ' + paths.path_to_log + '/code_status/')
os.system('cp ./train_test_scripts/' + train_config.train_script + ' ' + os.path.join(paths.path_to_log,
                                                                                'train_script_used.py'))

if ispa_net:
    os.system('cp ./ispa_net_configs/train_config.py' + ' ' + paths.path_to_log)
else:
    os.system('cp ./multi_view_configs/train_config.py' + ' ' + paths.path_to_log)

print "running script:" + train_config.train_script
os.system('python ./train_test_scripts/' + train_config.train_script)
