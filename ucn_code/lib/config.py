import numpy as np

#prototxt to use for training
NET_PROTOTXT = './nets/train_classifier.prototxt'

#model to be used for initializing the net
PRETRAINED_MODEL = './nets/bvlc_googlenet/bvlc_googlenet.caffemodel'

#set this to True for training the Viewpoint Classifier Network
CLASSIFIER = True

#GPU to be used
GPU_ID = 3

#name of the experiment, a folder is automatically created by this name, to sveave the summaries
EXP_NAME = 'test_classifier'

#directory where models are saved
EXP_FOLDERS_PATH = './saved_models/'

EXP_PATH = EXP_FOLDERS_PATH + EXP_NAME

LOG_FILE_NAME = 'googlenet.log'

CLASS_NAME = 'chair'

dict_models = {CLASS_NAME: []}
model_map = {0: CLASS_NAME}

models = {'chair': '03001627', 'sofa': '04256520', 'diningtable': '04379243', 'bed': '02818832'}

NUM_CLASSES = len(dict_models.keys())

# same string is used for snapshot prefix as well
LOG_FILE = EXP_PATH + '/' + LOG_FILE_NAME

# set this to -1 if pruning is not required for RETAIN_POS only
PRUNE_NEGATIVE = 500

NUMBER_OF_TRAINING_ITERATIONS = 100000

SOLVER_PARAMS = {
    'test_iter': 3,
    'test_interval': 100,
    'base_lr': 0.00009,
    'display': 20,
    'lr_policy': "step",
    'gamma': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'stepsize': 20000,
    'snapshot': 2000,
    'snapshot_prefix': EXP_PATH + '/snapshots/' + LOG_FILE_NAME.split('.')[0],
    'net': NET_PROTOTXT,
    'solver_type': 'ADAM',
    'solver_mode': 'GPU',
}

ADD_RGB_JITTER = True

MAX_NUM_TRAIN = 100000

CONFIG_PATH = './lib/config.py'

SOLVER_PROTOTXT = EXP_PATH + '/solver.prototxt'
# where to save output at the end
EXPORT_DIR = EXP_PATH + '/snapshots'

IM_SHAPE = (224, 224)

BATCH_SIZE = 12

# Limit the number of correspondence pairs per training.
MAX_NUM_CORRESPONDENCE = 1000

#radius used to compute pck
PCK_RADII = [10]

# Distance to be used for Hard negative mining
NEGATIVE_MINING_PIXEL_THRESH = 4

# (margin - dist)^2 as negative loss
MARGIN_DIST_SQ = True

# Use bilinear interpolation when evaluating
BILINEAR_INTERPOLATION = False

# Use horizontally-flipped images during training?
USE_FLIPPED = True

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# A small number that's used many times
EPS = 1e-14

IMG_SCALE = 1