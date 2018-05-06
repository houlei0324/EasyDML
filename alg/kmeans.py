__author__ = 'houlei'
__date__ = '05/01/2018'

import sys
import numpy as np
import gflags

sys.path.insert(0, sys.path[0]+'/..')
from src.easydml import MachineLearning

'''
K-means algorithm is a basic clustering algorithm.
Here we rewrite three function to achieve easy distribute programming
'''

FLAGS = gflags.FLAGS

#gflags.DEFINE_string('dataset', '.\data\iris_norm.csv', 'the input dataset')
gflags.DEFINE_string('dataset', '../data/iris_norm.csv', 'the input dataset')
gflags.DEFINE_integer('k', 3, 'the number of clusters')
gflags.DEFINE_integer('max_iteration', 100, 'the max iteration to run')
gflags.DEFINE_float('tolerance', 0.0, 'the tolerance to stop')

class Kmeans(MachineLearning):
    # the constructor of your algorithm
    # use gflags to set params
    # k            the number of clusters
    # iteration    the max iteration to stop
    # loss         the threshold of loss to stop
    def __init__(self, k, iteration, tolerance):
        super(Kmeans, self).__init__()
        self.k = k
        self.loss = 0
        self.max_iteration = iteration
        self.tolerance = tolerance
        self.centers = []

    # Coordinater
    # Init the centers of k cluster and send them to each worker
    def initEval(self):
        if self.comm_rank == 0:
            self.centers = np.random.rand(self.k, self.data_dim)
        else:
            self.lable = np.ones((self.data_size, 1))
            self.new_centers = np.zeros((self.k, self.data_dim))
        self.centers = self.deliverer.message_bcast(self.centers)

    def iterEval(self):
        if self.comm_rank == 0:
            new_centers = np.zeros((self.k, self.data_dim))
            tmp_recv = []
            last_loss = self.loss
            self.loss = 0
            for i in range(1, self.comm_size):
                tmp_recv = self.deliverer.message_recv(i)
                new_centers += tmp_recv[0]
                self.loss += tmp_recv[1]
            self.centers = new_centers / self.data_size
            self.superstep += 1
            self.logger.info('Loss: '+str(self.loss))
            # to decide whether to finish iteration
            if (self.superstep == self.max_iteration) or (abs(self.loss -
                                            last_loss) < self.tolerance):
                self.iter_finished = True
        else:
            loss = 0.0
            i = 0
            self.new_centers = np.zeros((self.k, self.data_dim))
            for vec in self.data:
                dist = []
                for center in self.centers:
                    dist.append(np.linalg.norm(vec - center))
                dist = np.array(dist)
                #self.logger.info('\n'+str(dist))
                self.lable[i] = np.argmin(dist)
                self.new_centers[np.argmin(dist)] += vec
                loss += np.min(dist)
                i += 1
            message = []
            message.append(self.new_centers)
            message.append(loss)
            self.deliverer.message_send(message, 0)
        #========================================
        self.comm.barrier()
        self.centers = self.deliverer.message_bcast(self.centers)
        self.superstep = self.deliverer.message_bcast(self.superstep)
        self.iter_finished = self.deliverer.message_bcast(self.iter_finished)

    def AssumbleEval(self):
        pass

if __name__ == '__main__':
    FLAGS(sys.argv)
    kmeans = Kmeans(FLAGS.k, FLAGS.max_iteration, FLAGS.tolerance)
    kmeans.loadData(FLAGS.dataset)
    kmeans.run()
    kmeans.metirs()
