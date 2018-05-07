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

class MachineLearning:
    """Machine Learning Class

    The class shuold be inherited when programming DML algorithm.
    Users are support to rewrite three functions in the subclass,
    InitEval(), IterEval(), and AssumbleEval(), then invoke the function
    run() to run your algorithm easily.

    Attributes:
        logger: an object from the class Logger in edml_log.py to output logs
        loader: an object from the class Loader in dataloader.py to input data
        deliverer: an object form the class Deliverer to package the operations of messages
        comm: the communicator form MPI to manage the communcation around the workers
        comm_rank: the id of the communication processor
        comm_size: the size of the communication world(including all processors)
        data: the training data loading from files
        data_size: the size of training data
        data_dim: the dimensions of training data
        results: the parameters of the finial model
        load_time: the time of loading training data
        run_time: the time of training the model
        superstep: the step of iteration
        iter_finished: the flag to stop the iteration
    """

    def __init__(self):
        """ Inits MachineLearning """
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
        self.data_size = 0
        self.data_dim = 0
        self.load_time = 0
        self.run_time = 0
        self.superstep = 0
        self.iter_finished = False

    def loadData(self, datadir):
        """ Load training data from given datadir.

        Load training data from given datadir, which support laoding from
        local, laoding from http and loading from hdfs.

        Args:
            datadir: the dir of the given dataset.
        """
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
        tmp_dim = self.deliverer.message_gather(self.data_dim)
        tmp_size = self.deliverer.message_reduce_sum(self.data_size)
        self.comm.barrier()
        if self.comm_rank == 0:
            self.data_dim = tmp_dim[1]
            self.data_size = tmp_size
            self.load_time = time.time() - self.load_time
            self.logger.info('[EasyDML] Finished loading data.')

    def initEval(self):
        """ The first main API: InitEval

        [When users create a subclass of DML, they must rewrite this API.]
        Inits the necessary parameters of the models and invoke the deliverer
        to synchronize those parameters.
        """
        pass

    def iterEval(self):
        """ The second main API: IterEval

        [When users create a subclass of DML, they must rewrite this API.]
        Run the main iteration of your algorithm to update parameters, users
        can define the termination condititons and invoke different communication
        methods.
        """
        pass

    def assumbleEval(self):
        """ The third main API: AssumbleEval

        [When users create a subclass of DML, they must rewrite this API.]
        Assumble the results of iteration and send them to the coodinator, then
        testing operation can be define and run.
        """
        pass

    def metirs(self):
        """ To output the statistics of loding and running

        Now this function supports the output of loading time and running time,
        more information can be meature.
        """
        if self.comm_rank == 0:
            self.logger.info('[EasyDML] Data loading time: ' +
                            str(self.load_time) + 's')
            self.logger.info('[EasyDML] Run time: ' + str(self.run_time) + 's')

    def run(self):
        """ The main function to run EasyDMl

        This function combines functions above and define the model of the
        whole algorithm, as long as users finish the rewrite of the IIA functions,
        this function will be invoked and distribute your algorithm automatically.
        """
        if self.comm_rank == 0:
            self.run_time = time.time()
            #========= InitEval =============
            self.logger.info('[InitEval] Start ...')
            self.initEval()
            self.comm.barrier()
            self.logger.info('[InitEval] Finished!')
            #========= IterEval =============
            while self.iter_finished == False:
                self.logger.info('[IterEval %d] Start ...' %(self.superstep))
                self.iterEval()
                self.comm.barrier()
                self.logger.info('[IterEval %d] Finished!' %(self.superstep - 1))
            #========= AssumbleEval ==========
            self.logger.info('[AssumbleEval] Start ...')
            self.assumbleEval()
            self.comm.barrier()
            self.logger.info('[AssumbleEval] Finished!')
            self.run_time = time.time() - self.run_time
        else:
            #========= InitEval =============
            self.initEval()
            self.comm.barrier()
            #========= IterEval =============
            while self.iter_finished == False:
                self.iterEval()
                self.comm.barrier()
            #========= AssumbleEval ==========
            self.assumbleEval();
            self.comm.barrier()
