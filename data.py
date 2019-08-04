import os
import sys
from os.path import join as pjoin
import fire
import h5py
import numpy as np
from misc import get_logger


class Data:

    def __init__(self, path):
        self.path = path
        self.logger = get_logger()

    def load(self):
        stock_data = {}
        with h5py.File(self.path, "r") as fin:
            for stockcode in fin.keys():
                self.logger.info(stockcode)
                stock_data[stockcode] = {
                    "price" : fin[stockcode]["prices"][:],
                    "volume" : fin[stockcode]["volumes"][:]
                }
        self.logger.info("Get Stock Data : {} images".format(len(stock_data.keys())))
        return stock_data

