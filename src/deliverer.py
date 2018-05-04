__author__ = 'houlei'
__date__ = '05/01/2018'

'''
This is the deliverer for the send and receive of messages between workers,
which further packaged the API of the pagkage mpi4py, and support the
requirement of different kinds of communitation in the communiting world.
'''

import mpi4py.MPI as MPI

class Deliverer():
    def __init__(self, comm):
        self.comm = comm

    def message_send(self, message, dst):
        self.comm.send(message, dest=dst)

    def message_recv(self, src):
        message = self.comm.recv(source=src)
        return message

    def message_bcast(self, message):
        return self.comm.bcast(message if self.comm.Get_rank() == 0 else None, root=0)

    def message_gather(self, message):
        return self.comm.gather(message, root=0)

    def message_reduce_sum(self, message):
        return self.comm.reduce(message, root=0, op=MPI.SUM)

