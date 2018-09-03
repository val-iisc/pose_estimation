import __init__paths
import os
from lib.config import LOG_FILE
from lib.config import EXP_PATH
if __name__ == '__main__':

    os.system('rm -rf ' + EXP_PATH)

    os.system('mkdir ' + EXP_PATH)

    os.system('python main.py 2>&1 | tee ' + LOG_FILE)
