#!/usr/bin/env python
import sys
import lib.config as cfg
from lib.solver import Solver
import os
import lib.create_solver as cs
import caffe

if __name__ == '__main__':



    cs.create(cfg.SOLVER_PARAMS, cfg.SOLVER_PROTOTXT)

    os.system('mkdir ' + cfg.EXP_PATH + '/snapshots')

    os.system('cp ' + cfg.NET_PROTOTXT + ' ' + cfg.EXP_PATH + '/')

    os.system('cp ' + cfg.CONFIG_PATH + ' ' + cfg.EXP_PATH + '/')

    caffe.set_device(cfg.GPU_ID)
    caffe.set_mode_gpu()

    sw = Solver(cfg.SOLVER_PROTOTXT, cfg.EXPORT_DIR, pretrained_model=cfg.PRETRAINED_MODEL)

    sw.train(cfg.NUMBER_OF_TRAINING_ITERATIONS)


