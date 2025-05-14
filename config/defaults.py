from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DEVICE = "cuda"
_C.SEED = 0
_C.DATA_TYPE = 'float'  # or double

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 10

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 32
# Size of the image during test
_C.INPUT.SIZE_TEST = 32
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.1307, ]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.3081, ]

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.NEED_CHRCKPOINT = False
_C.TRAIN.DATA_PATH = r''

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 1
_C.INFERENCE.MODEL_PATH = r'output\model.pth'
_C.INFERENCE.DATA_PATH = r''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = "outputs"

# -----------------------------------------------------------------------------
# LOG
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.DIR = "./log"
_C.LOG.ITER_INTERVAL = 1
_C.LOG.EPOCH_INTERVAL = 10
_C.LOG.OUTPUT_TO_FILE = True # 是否输出到文件，默认输出到控制台
_C.LOG.PREFIX = "default" # 输出到文件的命名前缀