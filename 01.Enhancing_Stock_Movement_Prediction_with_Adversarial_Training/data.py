import os
import csv
import numpy as np
from os.path import join as pjoin
import torch
from tqdm import tqdm
from torch.utils.data import Subset, Dataset, DataLoader, random_split


def get_close_data(data, closes):
    assert len(data) == len(closes)
    return [o / c - 1. for o, c in zip(data, closes)]

def get_rel_data(data):
    res = [0.]
    for i in range(1, len(data)):
        res.append(data[i] / data[i-1] - 1.)
    return res

def get_moving_data(data, n=5):
    """t-th data = \frac{\sum_{i=0}^{n-1}data[t-i]/n}{data[t]} - 1
    """
    res = [sum(data[max(0, i-n): i])/n/data[i] - 1. for i in range(len(data))]
    return res


def binary_labeler(x):
    if x >= 0.0055:
        return 1.
    elif x <= -0.005:
        return 0.
    else:
        return None


class TrainDataset(Dataset):

    def __init__(self, lags=7, is_regression=False, todate=20190101):
        self.lags = lags
        self.todate = todate
        self.start_num = 30
        self.rolling = 1

        if is_regression:
            self.labeler = lambda x: x
        else:
            self.labeler = binary_labeler

        self.data_dir = "/data/date"

        x, y = self.load()
        print("data num: ", len(x))
        self.x = x
        self.y = y


    def load(self):
        Xs, ys = [], []
        for stock_code in tqdm(os.listdir(self.data_dir)):
            X, y = self.load_stock_code(stock_code)
            assert len(X) == len(y)
            Xs += X
            ys += y
        print("LOAD {} data".format(len(ys)))
        Xs = np.array(Xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        return Xs, ys

    def load_stock_code(self, stock_code):
        Xs, ys = [], []

        dates, opens, highs, lows, closes = [], [], [], [], []
        with open(pjoin(self.data_dir, stock_code)) as fout:
            reader = csv.reader(fout, delimiter=',')
            for row in reader:
                d, o, h, l, c, v = map(int, row)
                dates.append(d)
                opens.append(o)
                highs.append(h)
                lows.append(l)
                closes.append(c)
        data_len = len(opens)
        if data_len < 300:
            return [], []

        c_open = get_close_data(opens, closes)
        c_high = get_close_data(highs, closes)
        c_low = get_close_data(lows, closes)
        n_close = get_rel_data(closes)
        day_5 = get_moving_data(closes, 5)
        day_10 = get_moving_data(closes, 10)
        day_15 = get_moving_data(closes, 15)
        day_20 = get_moving_data(closes, 20)
        day_25 = get_moving_data(closes, 25)
        day_30 = get_moving_data(closes, 30)

        X = list(zip(c_open, c_high, c_low, n_close,
                     day_5, day_10, day_15, day_20, day_25, day_30))

        y = []
        for i in range(len(opens)-self.rolling):
            y.append(closes[i+1]/closes[i] - 1.)

            ###################################################################
            #    O    O    O    O    O    O    O    O    O    O    O
            #    pivot-self.lags               pivot
            #    <----------train--------->    <answer>
            ###################################################################

        for pivot in range(self.start_num + self.lags, data_len - self.rolling):
            if dates[pivot] > self.todate:
                continue
            label = self.labeler(y[pivot-1])
            if label is None:
                continue
            ys.append(label)
            Xs.append(X[pivot-self.lags:pivot])
        return Xs, ys

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

