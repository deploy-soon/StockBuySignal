import csv
import json
import tqdm
import time
import random
import pathlib
import numpy as np
from os.path import join as pjoin

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from misc import get_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ma(opens, N):
    mas = []
    for i in range(len(opens)):
        ma = sum(opens[max(0,i-N+1):i+1]) / N
        mas.append(ma)
    return mas

def get_ema(opens, N):
    ema = [opens[0]]
    smoothing = 2.0 / (N + 1)
    for _open in opens[1:]:
        value = ema[-1] * smoothing + _open * (1 - smoothing)
        ema.append(value)
    return ema

def get_macd(opens):
    ema_12 = get_ema(opens, 12)
    ema_26 = get_ema(opens, 26)
    macd = [(one - two) / o for one, two, o in zip(ema_12, ema_26, opens)]
    return macd

def get_cci(highs, lows, closes):
    typical_prices = [(high + low + close) / 3 for high, low, close in zip(highs, lows, closes)]
    sma = typical_prices[:20]
    cci = [0.0] * 20
    for step in range(len(typical_prices) - 20):
        average = sum(typical_prices[step: step+20]) / 20.0
        deviation = [abs(t - average) for t in typical_prices[step:step+20]]
        mean_deviation = sum(deviation) / 20.0
        cci.append((typical_prices[step+20] - average) / 0.015 * mean_deviation)
    return cci

def get_atr(highs, lows, closes, N = 14):
    true_ranges = [0.0]
    for prev_close, low, high in zip(closes, lows[1:], high[1:]):
        true_ranges.append(max(high-low, abs(high-prev_close), abs(low-prev_close)))
    atr_0 = sum(true_ranges[:N]) / N
    atrs = [atr_0]
    for tr in true_ranges[1:]:
        atr = (atrs[-1] * (N-1) + tr) / N
        atrs.append(atr)
    return atrs

def get_bollinger(opens, N = 20):
    bands = []
    for i in range(len(opens)):
        bands.append(np.std(opens[max(0, i-N+1):i+1]))
    return bands

def get_roc(closes, N = 12):
    rocs = [0.0] * N
    for i in range(N, len(closes)):
        roc = (closes[i] - closes[i-N]) / closes[i-N]
        rocs.append(roc)
    return rocs

def get_wvad(opens, highs, lows, closes, volumes):
    # william's varaible accumulation distribution
    raws = [(c-o)/(h-l+1e-8)*v for o, h, l, c, v in zip(opens, highs, lows, closes, volumes)]
    ones = []
    for i in range(len(opens)):
        ones.append(sum(raws[max(0, i-5+1):i+1])/5.0)
    twos = []
    for i in range(len(opens)):
        twos.append(sum(raws[max(0, i-10+1):i+1])/10.0)
    return [o - t for o, t in zip(ones, twos)]


def get_ibs(opens, highs, lows, closes):
    """
    Internal Bar Strength (IBS) is an idea that has been around for some time
    :param o: open price
    :param h: high price
    :param l: low price
    :param c: close price
    :return: IBS value in [0, 1]
    """
    return [(c-l)/(h-l+1e-8) for o, h, l, c in zip(opens, highs, lows, closes)]

def get_rsi(opens, N=14):
    rsis = [0.0] * N
    for i in range(N, len(opens)):
        returns = [(prices[j+1]-prices[j]) / prices[j] for j in range(i-N, i)]
        u = [max(r, 0) for r in returns]
        d = [-min(r, 0) for r in returns]
        au, ad = sum(u), sum(d)
        _rsi = au / (au + ad + 1e-10)
        rsis.append(_rsi)
    return rsis

def mfi(highs, lows, closes, volumes):
    typical_prices = [(high + low + close) / 3 for high, low, close in zip(highs, lows, closes)]
    MF = [volume * tp for volume, tp in zip(volumes, typical_prices)]
    PMF = [MF[i+1] for i in range(len(typical_prices)-1) if typical_prices[i] < typical_prices[i+1]]
    NMF = [MF[i+1] for i in range(len(typical_prices)-1) if typical_prices[i] > typical_prices[i+1]]
    MR = sum(PMF) / (sum(NMF) + 1e-10)
    MFI = MR / (1 + MR)
    return MFI


kospi_price_interval = [
    (500000, 1000),
    (100000, 500),
    (50000, 100),
    (10000, 50),
    (5000, 10),
    (0, 5)
]

kosdaq_price_interval = [
    (50000, 100),
    (10000, 50),
    (5000, 10),
    (1000, 5),
    (0, 1)
]


class Data(Dataset):

    def __init__(self, stock_code, marketkind=1, verbose=False):

        self.logger = get_logger()
        self.verbose = verbose

        self.barrier = 8
        self.window = 12
        assert self.barrier > 0, "target inference"

        self.transaction_fee = 0.0035
        self.threshold = 0.0

        stock_code = str(stock_code)
        if not stock_code.startswith("A"):
            stock_code = "A" + stock_code
        self.stock_code = stock_code
        self.logger.info("target stock code: {}".format(self.stock_code))
        self.marketkind = marketkind

        self.parse()

    def get_slippage(self, price):
        """
        get slippage within stock price
        :param price: stock price
        :param market: 1-kospi, 2-kosdaq
        :return: amount of slippage in price
        """
        assert self.marketkind in [1, 2], "set market 1 or 2"
        interval = kospi_price_interval if self.marketkind == 1 else kosdaq_price_interval
        for p, slip in interval:
            if price > p:
                return slip
        return 0

    def load(self):
        minute_rows = []
        with open(pjoin("minute_data", self.stock_code), "r", newline='') as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                minute_rows.append(row)
        self.logger.info("load rows: {}".format(len(minute_rows)))
        return minute_rows

    def parse(self):
        minute_rows = self.load()
        time_len = len(minute_rows)
        frames, preds, labels = [], [], []

        dates = [row["date"] for row in minute_rows]
        opens = [float(row["open"]) for row in minute_rows]
        highs = [float(row["high"]) for row in minute_rows]
        lows = [float(row["low"]) for row in minute_rows]
        closes = [float(row["close"]) for row in minute_rows]
        volumes = [float(row["volume"]) for row in minute_rows]

        macds = get_macd(opens)
        ccis = get_cci(highs, lows, closes)
        atrs = get_atr(highs, lows, closes)
        start_point = 26
        assert time_len == len(macds) == len(ccis) == len(atrs)
        windows = list(zip(macds, ccis, atrs))

        for step in range(start_point, time_len - self.window - self.barrier):
            window_dates = dates[step: step + self.window + self.barrier]
            if len(set(window_dates)) > 1:
                continue

            frames.append(windows[step:step + self.window])

            step = step + self.window
            _open = opens[step]
            _close = closes[step + self.barrier - 1]
            buy_slippage = self.get_slippage(_open)
            sell_slippage = self.get_slippage(_close)
            _y = (_close - _open - buy_slippage - sell_slippage) / _open
            _y = _y - self.transaction_fee
            preds.append(_y)

        assert len(frames) == len(preds), "not equal {}, {}".format(len(frames), len(preds))
        labels = [1 if _y > self.threshold else 0 for _y in preds]

        self.logger.info("Get Stock Data : {} patterns".format(len(frames)))
        self.logger.info("class 1: {}, class 0: {}".format(sum(labels), len(labels) - sum(labels)))
        self.data_len = len(frames)
        self.features = len(frames[0][0])

        self.frames = torch.tensor(frames, dtype=torch.float)
        self.preds = torch.tensor(preds, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)
        print(self.frames[0])
        print(self.preds[0])

    def _meta_normalize(self, meta_info):
        keys = list(meta_info[0].keys())
        value_list = {}
        for key in keys:
            values = [v[key] for v in meta_info]
            _min, _max = min(values), max(values)
            _mean, _std = sum(values) / len(values), np.std(values)
            self.logger.info("data {} range: {:.4f} ~ {:.4f}, mean: {:.4f} std: {:.4f}"
                             .format(key, _min, _max, _mean, _std))

            values = [(v-_mean) / (_std + 1e-10) for v in values]
            value_list[key] = values
        res = [list(v) for v in zip(*(value_list.values()))]
        return res

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        item = {
            "frame": self.stock_data[index],
            "meta": self.metas[index],
            "pred": self.pred[index],
            "label": self.label[index]
        }
        return item


def recall(y_true, y_pred):
    assert len(y_true) == len(y_pred), "invalid length {} vs {}".format(len(y_true), len(y_pred))
    ones, match = 0, 0
    for t, p in zip(y_true, y_pred):
        if t > 0.5:
            ones = ones + 1
            if p > 0.5:
                match = match + 1
    return match / (ones + 1e-10)

def precision(y_true, y_pred):
    assert len(y_true) == len(y_pred), "invalid length {} vs {}".format(len(y_true), len(y_pred))
    pred_ones, match = 0, 0
    for t, p in zip(y_true, y_pred):
        if p > 0.5:
            pred_ones = pred_ones + 1
            if t > 0.5:
                match = match + 1
    return match / (pred_ones + 1e-10)


def f1_score(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    return 2 * ((_precision * _recall) / (_precision + _recall + 1e-10))


class Encoder(nn.Module):

    def __init__(self, features, hid_dim, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(features, hid_dim, layers, dropout = dropout)

    def forward(self, frame):
        outputs, (hidden, cell) = self.rnn(frame)
        return hidden, cell

class Network(nn.Module):

    def __init__(self, encoder, num_meta, enc_hid_dim, hid_dim, device):
        super().__init__()

        self.encoder = encoder
        #self.embedding = nn.Linear(num_meta, hid_dim)
        #self.fc1 = nn.Linear(hid_dim + e, nc_hid_dim, 256)
        self.fc1 = nn.Linear(enc_hid_dim, 256)
        self.fc2 = nn.Linear(256, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.device = device


    def forward(self, frame, meta):
        hidden, _ = self.encoder(frame)
        #emb = F.relu(self.embedding(meta))
        #x = F.relu(self.fc1(torch.cat((hidden[1], emb), dim=1)))
        x = F.relu(self.fc1(hidden[1]))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        return output


class Train:

    def __init__(self, epochs=10, batch_size=128):
        self.logger = get_logger()
        self.data = Data()
        data_len = len(self.data)
        train_num = int(data_len * 0.8)
        valid_num = int(data_len * 0.1)
        test_num = data_len - train_num - valid_num
        train, valid, test = random_split(self.data, [train_num, valid_num, test_num])
        self.train_iter = DataLoader(train, batch_size = batch_size, shuffle=True, num_workers=4)
        self.valid_iter = DataLoader(valid, batch_size = batch_size, shuffle=True, num_workers=4)
        self.test_iter = DataLoader(test, batch_size = batch_size, shuffle=True, num_workers=4)
        self.encoder = Encoder(features=self.data.features,
                               hid_dim = 256,
                               layers=2,
                               dropout=0.5)
        self.network = Network(encoder=self.encoder,
                               num_meta=self.data.num_meta,
                               enc_hid_dim=256,
                               hid_dim=64,
                               device=device).to(device)
        print(self.network)
        self.epochs = epochs
        self.batch_size = batch_size

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, iterator, optimizer, criterion, batch_size):
        self.network.train()

        epoch_loss = 0

        with tqdm.tqdm(total=len(iterator)) as t:
            for i, batch in enumerate(iterator):
                batch_size = batch["frame"].shape[0]
                frame = batch["frame"].view(-1, batch_size, self.data.features).to(device)
                meta = batch["meta"].to(device)
                label = batch["label"].view(-1, 1).to(device)

                output = self.network(frame, meta)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                optimizer.step()
                epoch_loss += loss.item()

                t.set_postfix(loss='{:05.3f}'.format(epoch_loss / (i+1)))
                t.update()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator, criterion, is_validate=True):
        self.network.eval()
        epoch_loss = 0

        predict = []
        real = []
        returns = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):

                batch_size = batch["frame"].shape[0]
                frame = batch["frame"].view(-1, batch_size, self.data.features).to(device)
                meta = batch["meta"].to(device)
                label = batch["label"].to(device)
                if not is_validate:
                    _returns = batch["pred"]
                    returns.extend(_returns.tolist())

                real.extend(label.tolist())

                output = self.network(frame, meta)
                output = output.squeeze()
                pred = [1 if o > 0.18 else 0 for o in output]
                predict.extend(pred)
                loss = criterion(output, label)
                epoch_loss += loss.item()
                #if i % 200 == 199:
                #    print(output)

        if not is_validate:
            self.logger.info("buy {} cases among {}".format(sum(predict), len(predict) - sum(predict)))
            pred_returns = [_return for buy, _return in zip(predict, returns) if buy > 0.18]
            initial_return = 1.0
            for _returns in pred_returns:
                initial_return *= (1+_returns)
            self.logger.info("Test returns: {:.4f}, overall: {:.4f}"
                             .format(sum(pred_returns) / len(pred_returns), initial_return))
        else:
            self.logger.info("Validation. Prec: {:.4f}, recall: {:.4f}, f1: {:.4f}"
                             .format(precision(real, predict),
                                     recall(real, predict),
                                     f1_score(real, predict)))

        return epoch_loss / len(iterator)

    def run(self):
        self.logger.info("Model trainable parameters: {}".format(self.count_parameters(self.network)))
        self.network.apply(self.init_weights)
        optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            train_loss = self.train(self.train_iter, optimizer, criterion, self.batch_size)
            vali_loss = self.evaluate(self.valid_iter, criterion)
        _ = self.evaluate(self.test_iter, criterion, is_validate=False)


if __name__ == "__main__":
    d = Data("005930")
    #print(d[0])
    #t = Train(epochs=50)
    #t.run()

