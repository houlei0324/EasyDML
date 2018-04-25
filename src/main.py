#!/usr/bin/python3

__author__ = 'houlei'
__date__ = '04/24/2018'

import gflags
import pickle
import mpi4py.MPI as MPI
import numpy as np
import edml_log as log

logger = log.Logger('../log.txt').init_logger()
