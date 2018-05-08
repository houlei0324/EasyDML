__author__ = 'wudh'
__date__ = '05/07/2018'

import sys
import numpy as np
import gflags

#sys.path.insert(0, sys.path[0]+'\\..')
sys.path.insert(0, sys.path[0]+'/..')
from src.easydml import MachineLearning

'''
NaiveBayes algorithm is a basic clustering algorithm.
Here we rewrite three function to achieve easy distribute programming
'''

FLAGS = gflags.FLAGS

#gflags.DEFINE_string('datadir', '.\data\Chess_Data_Set(28056_6_cate)/krkopt.data', 'the input dataset')
gflags.DEFINE_string('datadir', '../data/Chess_Data_Set(28056_6_cate)/krkopt.data', 'the input dataset')
gflags.DEFINE_string('separator', ',', 'the data separator')

class NaiveBayes(MachineLearning):
    # the constructor of your algorithm
    # use gflags to set params
    # d            the dimension of data
    def __init__(self):
        super(NaiveBayes, self).__init__()

    # Coordinater
    # Init the centers of k cluster and send them to each worker
    def initEval(self):
        if self.comm_rank == 0:
            self.F = {}
        else:
            self.features = [1,2,1,3,4,2]
            data_vec = np.array(self.data)
            data_train = data_vec[:,0:data_vec.shape[1] - 1]
            data_label = data_vec[:,data_vec.shape[1] - 1]
            labels = list(data_label)
            P_y = {}
            for label in labels:
                P_y[label] = labels.count(label)/float(len(labels))
            P_xy = {}
            for y in P_y.keys():
                y_index = [i for i, label in enumerate(labels) if label == y]
                for j in range(len(self.features)):
                    x_index = [i for i, feature in enumerate(data_train[:,j]) if feature == self.features[j]]
                    xy_count = len(set(x_index) & set(y_index))
                    pkey = str(self.features[j]) + '*' + str(y)
                    P_xy[pkey] = xy_count / float(len(labels))
            P = {}
            for y in P_y.keys():
                for x in self.features:
                    pkey = str(x) + '|' + str(y)
                    P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])
            F = {}
            for y in P_y:
                F[y] = P_y[y]
                for x in self.features:
                    F[y] = F[y]*P[str(x)+'|'+str(y)]
            message = []
            message.append(F)
            self.deliverer.message_send(message, 0)

    def iterEval(self):
        if self.comm_rank == 0:
            for i in range(1, self.comm_size):
                tmp_recv = self.deliverer.message_recv(i)
                for k, v in tmp_recv[0].items():
                    if k in self.F.keys():
                        self.F[k] += v
                    else:
                        self.F[k] = v
            features_label = max(self.F, key=self.F.get)
            self.logger.info('Belong to %s'%(features_label))
            self.iter_finished = True
        #========================================
        self.comm.barrier()
        self.iter_finished = self.deliverer.message_bcast(self.iter_finished)
    def AssumbleEval(self):
        pass

if __name__ == '__main__':
    FLAGS(sys.argv)
    NaiveBayes = NaiveBayes()
    NaiveBayes.loadData(FLAGS.datadir, FLAGS.separator)
    NaiveBayes.run()
    NaiveBayes.metirs()
