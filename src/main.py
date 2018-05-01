#!/usr/bin/python3

__author__ = 'houlei'
__date__ = '04/24/2018'

import time
import gflags
import pickle
import mpi4py.MPI as MPI
import numpy as np
import edml_log as log
import dataloader as loader

class MachineLearning:
    def __init__(self):
        self.logger = log.Logger('../log.txt').init_logger()
        self.loader = loader.Loader(self.logger)
        # instance for invoking MPI related functions
        self.comm = MPI.COMM_WORLD
        # the node rank in the whole community
        self.comm_rank = self.comm.Get_rank()
        # the size of the whole community (the total number of working nodes)
        self.comm_size = self.comm.Get_size()
        self.data = []
        self.load_time = 0
        self.run_time = 0
        self.superstep = 0
        self.iter_finished = false

    def loadData(self, iofrom, datadir):
        if self.comm_rank == 0:
            self.load_time = time.time()
        #===============================
        if iofrom == 'local':
            self.loader.fromLocal(self.comm_rank, self.comm_size,
                                  self.data, datadir)
        elif iofrom == 'http':
            self.loader.fromHttp(self.comm_rank, self.comm_size,
                                  self.data, datadir)
        elif iofrm == 'hdfs':
            self.loader.formHdfs(self.comm_rank, self.comm_size,
                                  self.data, datadir)
        else:
            self.logger.info('Unsupported data source.')
        self.comm.barrier()
        #===============================
        if self.comm_rank == 0:
            self.load_time = time.time() - self.load_time

    def initEval(self):
        pass

    def iterEval(self):
        pass

    def assumbleEval(self):
        pass

    def metirs(self):
        if self.comm_rank == 0:
            self.logger.info('[EasyDML] Data loading time: ' +
                            str(self.load_time) + 's')
            self.logger.info('[EasyDML] Run time: ' + str(self.run_time) + 's')

    def run(self):
        if comm_rank == 0:
            self.run_time = tme.time()
            #========= InitEval =============
            self.logger.info('[InitEval] Start ...')
            self.comm.barrier()
            self.logger.info('[InitEval] Finished!')
            #========= IterEval =============
            while self.iter_finished == false:
                self.logger.info('[IterEval %d] Start ...' %(self.superstep))
                self.comm.barrier()
                self.logger.info('[IterEval %d] Finished!' %(self.superstep))
            #========= AssumbleEval ==========
            self.logger.info('[AssumbleEval] Start ...')
            self.comm.barrier()
            self.logger.info('[AssumbleEval] Finished!')
        if comm_rank > 0:
            #========= InitEval =============
            initEval()
            self.comm.barrier()
            #========= IterEval =============
            while self.iter_finished == false:
                iterEval()
                self.comm.barrier()
            #========= AssumbleEval ==========
            assumbleEval();
            self.comm.barrier()
