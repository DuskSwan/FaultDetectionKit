# encoding: utf-8

import time
import os
import random

from loguru import logger
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed=31415):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def initiate_cfg(cfg,merge_file = ''):
    '''
    Initiate the cfg object with the default config file and the extra config file. 
    The cfg will be frozen after initiation.
    '''
    if(merge_file): 
        logger.info("Try to merge from {}.".format(merge_file))
        extra_cfg = Path(merge_file)
        if extra_cfg.exists() and extra_cfg.suffix == '.yml':
            cfg.merge_from_file(extra_cfg)
            logger.info("Merge successfully.")
        else:
            logger.info("Wrong file path or file type of extra config file.")
    else: 
        logger.info("No extra config file to merge.")

    cfg.freeze()
