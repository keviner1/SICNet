#!/usr/local/bin/python
import sys
import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
#                              Server settings
# -----------------------------------------------------------------------------
_C.SERVER = CN()
_C.SERVER.gpus = 1

_C.SERVER.TRAIN_DATA = 'Your Path'
_C.SERVER.VAL_DATA = 'Your Path'
_C.SERVER.TEST_DATA = 'images\\in'
_C.SERVER.OUTPUT = 'output\\job1'  


# ------------------------------------------------------------cd-----------------
#                            Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.channels = 48
# -----------------------------------------------------------------------------
#                           Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.PRINT_FREQ = 500
_C.TRAIN.RESUME = ""

_C.TRAIN.lamda1 = 0.2
_C.TRAIN.lamda1_step = 10
_C.TRAIN.lamda2 = 0.1
_C.TRAIN.lamda2_step = 10


_C.TRAIN.Teacher = "model\\Teacher\\chk.pth"
_C.TRAIN.pretrain = ""


# ----------------------------------LR scheduler-------------------------------
_C.TRAIN.LR_MODE = "step"
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.LR_DECAY = 0.5
_C.TRAIN.LR_STEP = 15
_C.TRAIN.MILESTONES = [3, 6, 10, 15, 25, 45, 65, 85, 105, 125, 145, 175, 215, 260, 290, 330]
# -----------------------------------Optimizer-----------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


def get_config():
    config = _C.clone()
    return config


