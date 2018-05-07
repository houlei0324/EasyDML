__author__ = 'houlei'
__date__ = '05/01/2018'

import mpi4py.MPI as MPI

class Deliverer():
    """ The Deliverer of messages using MPI

    This is the deliverer for the send and receive of messages between workers,
    which further packaged the API of the pagkage mpi4py, and support the
    requirement of different kinds of communitation in the communiting world.

    Attributes:
        comm: the communicator form MPI to manage the communcation around the workers
              given by the ML object
    """

    def __init__(self, comm):
        """ Inits the deliverer with the MPI comm

        Args:
            comm: the communicator form MPI to manage the communcation around the workers
                  given by the ML object.
        """
        self.comm = comm

    def message_send(self, message, dst):
        """ Point to point message send

        Package MPI_send as a method of Deliverer.

        Args:
            message: the message to be sent.
            dst: the object processor to receive the message.
        """
        self.comm.send(message, dest=dst)

    def message_recv(self, src):
        """ Point to point message receive

        Package MPI_send as a method of Deliverer.

        Args:
            src: the source processor where the message send from.
        Return:
            the message received.
        """
        message = self.comm.recv(source=src)
        return message

    def message_bcast(self, message):
        """ Message Bocast in the whole world

        Package MPI_Bocast as a method of Deliverer.

        Args:
            message: the message coodinator want to bocast.
        Return:
            the message workers received.
        """
        return self.comm.bcast(message if self.comm.Get_rank() == 0 else None, root=0)

    def message_gather(self, message):
        """ Message gather in the whole world

        Package MPI_Gather as a method of Deliverer, which should be used when
        users want to gather message to coodinator from workers.

        Args:
            message: the message coodinator want to gather.
        Return:
            the message from all workers.
        """
        return self.comm.gather(message, root=0)

    def message_reduce_sum(self, message):
        """ Message Reduce in the whole world

        Package MPI_Reduce as a method of Deliverer, which should be used when
        users want to calculate the sum of the same object from all workers.

        Args:
            message: the message coodinator want to reduce.
        Return:
            the sum of messages from workers.
        """
        return self.comm.reduce(message, root=0, op=MPI.SUM)

