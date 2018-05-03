__author__ = 'houlei'
__date__ = '04/29/2018'

import re
import numpy as np
import src.edml_log as log

'''
The class used to load input data according to the comm rank,
which means that except comm_rank == 0, for other comm_rank i,
the loader read data from different input(such as local, http and hdfs)
whose id is the hash of comm_rank i.
'''

class Loader:
    def __init__(self, logger):
        self.logger = logger

    def fromLocal(self, comm_rank, comm_size, datadir):
        input = open(datadir)
        num = 0
        load_num = 0
        data = []
        for line in input:
            # to decide whether this line of data should be loaded
            # according to an samole hash
            if num % (comm_size - 1) == comm_rank - 1:
                data.append(line.split())
                load_num = load_num + 1
            num = num + 1
        self.logger.info(
            '[Processor %d] loads %d lines of data from the local file.'
             %(comm_rank, load_num))
        return np.array(data).astype(np.float64)

    def fromHttp(self, comm_rank, comm_size, data, datadir):
        pass

    def fromHdfs(self, comm_rank, comm_size, data, datadir):
        pass