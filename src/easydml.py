#!/usr/bin/python3

__author__ = 'houlei'
__date__ = '04/24/2018'

import time
import gflags
import pickle
import mpi4py.MPI as MPI
import numpy as np

import src.edml_log as log
import src.dataloader as loader
import src.deliverer as deliverer

'''
The class shuold be inherited when programming DML algorithm.
Users are support to rewrite three functions in the subclass,
InitEval(), IterEval(), and AssumbleEval(), then invoke the function
run() to run your algorithm easily.
'''

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
        self.deliverer = deliverer.Deliverer(self.comm)
        self.data = []
        self.results = []
        self.load_time = 0
        self.run_time = 0
        self.superstep = 0
        self.data_size = 0
        self.data_dim = 0
        self.iter_finished = False

    def loadData(self, datadir):
        if self.comm_rank == 0:
            self.load_time = time.time()
            self.logger.info('[EasyDML] Start to load data ...')
        else:
            if datadir[:3] == 'http':
                self.data = self.loader.fromHttp(self.comm_rank,
                            self.comm_size, datadir)
            elif datadir[:3] == 'hdfs':
                self.data = self.loader.fromHdfs(self.comm_rank,
                            self.comm_size, datadir)
            else:
                self.data = self.loader.fromLocal(self.comm_rank,
                            self.comm_size, datadir)
            self.data_dim = np.shape(self.data)[1]
            self.data_size = np.shape(self.data)[0]
        tmp_dim = np.mean(self.deliverer.message_gather(self.data_dim)[1:])
        tmp_size = self.deliverer.message_reduce_sum(self.data_size)
        self.comm.barrier()
        if self.comm_rank == 0:
            self.data_dim = tmp_dim
            self.data_size = tmp_size
            self.load_time = time.time() - self.load_time
            self.logger.info('[EasyDML] Finished loading data.')

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
            initEval()
            self.comm.barrier()
            self.logger.info('[InitEval] Finished!')
            #========= IterEval =============
            while self.iter_finished == False:
                self.logger.info('[IterEval %d] Start ...' %(self.superstep))
                iterEval()
                self.comm.barrier()
                self.logger.info('[IterEval %d] Finished!' %(self.superstep))
            #========= AssumbleEval ==========
            self.logger.info('[AssumbleEval] Start ...')
            assumbleEval()
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
