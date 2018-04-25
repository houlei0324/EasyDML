__author__ = 'houlei'
__date__ = '04/25/2018'

import logging

# A logger class

class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # create a handler for log files
        filehandler = logging.FileHandler(filename, mode='w')
        filehandler.setLevel(logging.INFO)
        # create a handler for log console
        conshandler = logging.StreamHandler()
        conshandler.setLevel(logging.INFO)
        # the format of logs
        formatter = logging.Formatter('%(asctime)s %(filename)s '\
                            '[line:%(lineno)d] %(levelname)s: %(message)s')
        filehandler.setFormatter(formatter)
        conshandler.setFormatter(formatter)
        # add handlers
        self.logger.addHandler(filehandler)
        self.logger.addHandler(conshandler)

    def init_logger(self):
        return self.logger
