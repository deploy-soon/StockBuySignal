import h5py
import random
import numpy as np
from misc import get_logger
import pprint

class Data:

    def __init__(self, path, window, stride, minute_after, bandwidth, batch_size, train_ratio, verbose=False):
        """
        :param path: data path
        :param window: length of windows in each frame
        :param stride: stride of frame
        :param minute_after: compare with the last of frame close and minute_after close
        :param bandwidth: height of frame
        :param batch_size: train batch
        :param train_ratio:
        :param verbose:
        """
        self.logger = get_logger()
        self.path = path
        self.window = window
        self.stride = stride
        self.minute_after = minute_after
        self.bandwidth = bandwidth

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose

        self.threshold = 0.01

    def load(self):
        """
        :data structure:
        {
            "stockcode": {
                "dates": list,
                "minutes": list,
                "opens": list,
                "highs": list,
                "lows": list,
                "closes": list,
                "volumes": list
            },
            ...
        }
        """
        stock_data, answer = [], []
        with h5py.File(self.path, "r") as fin:
            for stockcode in fin.keys():
                if self.verbose:
                    self.logger.info(stockcode)
                pivots = set(fin[stockcode]["dates"][:])
                for pivot in pivots:
                    opens = [i for i, p in zip(fin[stockcode]["opens"][:], fin[stockcode]["dates"][:]) if p == pivot]
                    highs = [i for i, p in zip(fin[stockcode]["highs"][:], fin[stockcode]["dates"][:]) if p == pivot]
                    lows = [i for i, p in zip(fin[stockcode]["lows"][:], fin[stockcode]["dates"][:]) if p == pivot]
                    closes = [i for i, p in zip(fin[stockcode]["closes"][:], fin[stockcode]["dates"][:]) if p == pivot]
                    volumes = [i for i, p in zip(fin[stockcode]["volumes"][:], fin[stockcode]["dates"][:]) if p == pivot]
                    minutes = [i for i, p in zip(fin[stockcode]["minutes"][:], fin[stockcode]["dates"][:]) if
                               p == pivot]
                    min_price, max_price = min(opens), max(opens)
                    max_volume = float(max(volumes))
                    if min_price * 1.05 > max_price:
                        max_price = min_price * 1.05
                    tab = (max_price - min_price) / float(self.bandwidth)
                    price_zone = [min_price + i * tab for i in range(self.bandwidth + 1)]
                    frames, y = [], []
                    for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
                        # frame = [0 if (h < s and h < e) or (s < l and e < l) else 1
                        #          for s, e in zip(price_zone, price_zone[1:])]
                        frame = [0 if (h < s and h < e) or (s < l and e < l) else v/max_volume
                                 for s, e in zip(price_zone, price_zone[1:])]
                        frames.append(frame)
                    for i in range(len(closes) - self.window - self.minute_after):
                        # _y = (max(closes[i+self.window:i+self.window+self.minute_after+1]) - closes[i+self.window]) / \
                        #      float(closes[i+self.window])
                        _y = (closes[i + self.window + self.minute_after] - closes[i + self.window]) / \
                             float(closes[i + self.window])
                        y.append(_y)
                    frames = np.array(frames)
                    if frames.shape[0] <= self.window + self.minute_after:
                        continue
                    shape = ((frames.shape[0] - self.window) // self.stride + 1, self.window, frames.shape[-1])
                    strides = (frames.strides[0], frames.strides[0], frames.strides[-1])
                    images = np.lib.stride_tricks.as_strided(frames, shape=shape, strides=strides)
                    y = y[::self.stride]
                    images = images[:len(y)]
                    assert len(y) == len(images), "y: {} images: {}".format(len(y), len(images))
                    stock_data += images.tolist()
                    answer += y

        self.logger.info("Get Stock Data : {} images".format(len(stock_data)))
        return stock_data, answer

    def launch(self):
        X, y = self.load()
        perm = list(range(len(X)))
        random.shuffle(perm)
        X = [X[idx] for idx in perm]
        y = [y[idx] for idx in perm]
        y_binary = [1 if _y >= self.threshold else 0 for _y in y]
        self.logger.info("one : {}, zero : {}".format(sum(y), len(y) - sum(y)))
        self.train_num = int(len(X) * self.train_ratio)
        self.vali_num = len(X) - self.train_num
        self.logger.info("train num : {} validation num : {}".format(self.train_num, self.vali_num))
        self.train_X, self.train_y = X[:self.train_num], y_binary[:self.train_num]
        self.vali_X, self.vali_y = X[self.train_num:], y_binary[self.train_num:]
        self.test_y = y[self.train_num:]

    def get_train_batch_num(self):
        return self.train_num // self.batch_size

    def get_vali_batch_num(self):
        return self.vali_num // self.batch_size

    def get_test_batch_num(self):
        return self.vali_num // self.batch_size

    def gen_train(self):
        train_num = self.train_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, train_num, self.batch_size):
                limit = min(train_num, offset + self.batch_size)
                yield np.array(self.train_X[offset:limit]).reshape((-1, self.window, self.bandwidth, 1)), \
                      np.array(self.train_y[offset:limit])

    def gen_vali(self):
        vali_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, vali_num, self.batch_size):
                limit = min(vali_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.window, self.bandwidth, 1)), \
                      np.array(self.vali_y[offset:limit])

    def gen_test(self):
        test_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, test_num, self.batch_size):
                limit = min(test_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.window, self.bandwidth, 1)), \
                      np.array(self.test_y[offset:limit])

