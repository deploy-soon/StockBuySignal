import csv
import h5py
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

def ibs(o, h, l, c):
    """
    Internal Bar Strength (IBS) is an idea that has been around for some time
    :param o: open price
    :param h: high price
    :param l: low price
    :param c: close price
    :return: IBS value in [0, 1]
    """
    return (c - l) / (h - l + 1e-8)

def rsi(prices):
    # N in rsi is dynamically set with len(prices)
    returns = [(prices[i+1]-prices[i]) / prices[i] for i in range(len(prices)-1)]
    u = [max(r, 0) for r in returns]
    d = [-min(r, 0) for r in returns]
    au, ad = sum(u), sum(d)
    _rsi = au / (au + ad + 1e-10)
    return _rsi

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


class StockMetaData:

    def __init__(self):
        with open("data/fnguide_report.csv", "r") as fin:
            reader = csv.DictReader(fin)
            self.rows = list(reader)
        self.preprocess()

    def _get_stock_code(self, stockcode):
        if stockcode.startswith("A"):
            return stockcode[1:]
        return stockcode

    def _parse(self, row):
        if not row.get("Total Assets") or not row.get("Profit")\
                or not row.get("Gross Profit"):
            return None
        return {
            "ta": np.log(1 + float(row.get("Total Assets"))),
            "roa": float(row.get("Profit")) / float(row.get("Total Assets")),
            "gpa": float(row.get("Gross Profit")) / float(row.get("Total Assets")),
        }

    def preprocess(self):
        stock_info = {}
        raw_data = {}
        for row in self.rows:
            #code = self._get_stock_code(row.get("Code"))
            code = row.get("Code")
            if not code:
                continue
            stock_info.setdefault(code, {}).update({
                row.get("period"): self._parse(row),
            })
            raw_data.setdefault(code, []).append(row)
        self.stock_info = stock_info
        self.raw_data = raw_data

    def get_market(self, stockcode):
        #never mind time constraint
        if stockcode not in self.raw_data or not self.raw_data[stockcode]:
            return None
        row = self.raw_data[stockcode][0]
        return 1 if row.get("Market") == "KS" else 2


    def _get_ym(self, ymd):
        y = int(ymd[:4])
        m = int(ymd[4:6])
        m = m - 1
        if m <= 2:
            m = 12
            y = y - 1
        m = int(m / 3) * 3
        return "{}{:02d}".format(y, m)

    def get(self, stockcode, ymd, price=0.0):
        ym = self._get_ym(ymd)
        if stockcode not in self.stock_info or ym not in self.stock_info[stockcode]:
            return None
        if ym not in self.stock_info[stockcode] or not self.stock_info[stockcode][ym]:
            return None

        return self.stock_info[stockcode][ym]



class Data(Dataset):

    def __init__(self, path="data/ninetoten.h5", verbose=False):

        self.logger = get_logger()
        self.path = path
        self.verbose = verbose

        self.barrier = 4
        self.window = 7
        assert self.barrier > 0, "target inference"

        self.transaction_fee = 0.0035
        self.threshold = 0.0001

        self.stock_meta = StockMetaData()

        self.load()

    def get_slippage(self, price, market):
        """
        get slippage within stock price
        :param price: stock price
        :param market: 1-kospi, 2-kosdaq
        :return: amount of slippage in price
        """
        assert market in [1, 2], "set market 1 or 2"
        interval = kospi_price_interval if market == 1 else kosdaq_price_interval
        for p, slip in interval:
            if price > p:
                return slip
        return 0

    def _sort_time(self, target_list, time_list):
        merge_list = list(zip(target_list, time_list))
        merge_list = sorted(merge_list, key=lambda x: x[1])
        merge_list = [m[0] for m in merge_list]
        return merge_list

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
        stock_data, metas, pred, label = [], [], [], []
        self.logger.info("Start Load {}".format(self.path))
        with h5py.File(self.path, "r") as fin:
            for stockcode in tqdm.tqdm(fin.keys()):
                if self.verbose:
                    self.logger.info(stockcode)

                datetimes = set(fin[stockcode]["dates"][:])
                for _datetime in datetimes:

                    meta_info = self.stock_meta.get(stockcode, str(_datetime))
                    if meta_info is None:
                        continue

                    opens = [i for i, p in zip(fin[stockcode]["opens"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    highs = [i for i, p in zip(fin[stockcode]["highs"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    lows = [i for i, p in zip(fin[stockcode]["lows"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    closes = [i for i, p in zip(fin[stockcode]["closes"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    volumes = [i for i, p in zip(fin[stockcode]["volumes"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    minutes = [i for i, p in zip(fin[stockcode]["minutes"][:], fin[stockcode]["dates"][:]) if
                               p == _datetime]

                    opens = self._sort_time(opens, minutes)
                    highs = self._sort_time(highs, minutes)
                    lows = self._sort_time(lows, minutes)
                    closes = self._sort_time(closes, minutes)
                    volumes = self._sort_time(volumes, minutes)

                    ibss = []
                    for o, h, l, c in zip(opens, highs, lows, closes):
                        ibss.append(ibs(o, h, l, c))
                    time_len = len(opens)
                    frames, meta, y = [], [], []


                    feature_window = 9
                    rsis = []
                    mfis = []
                    for step in range(time_len - feature_window):
                        _opens = opens[step:step+feature_window]
                        _highs = highs[step:step+feature_window]
                        _lows = lows[step:step+feature_window]
                        _closes = closes[step:step+feature_window]
                        _volumes = volumes[step:step+feature_window]
                        rsis.append(rsi(_opens))
                        mfis.append(mfi(_highs, _lows, _closes, _volumes))

                    # for X
                    for step in range(time_len - self.window - self.barrier - feature_window):
                        _rsis = rsis[step:step+self.window]
                        _mfis = mfis[step:step+self.window]
                        step = step + feature_window
                        _opens = opens[step:step+self.window]
                        _highs = highs[step:step+self.window]
                        _lows = lows[step:step+self.window]
                        _closes = closes[step:step+self.window]
                        _volumes = volumes[step:step+self.window]
                        _ibss = ibss[step:step+self.window]

                        fo = _opens[0] # first open
                        _opens = [(o - fo) / fo for o in _opens]
                        _highs = [(h - fo) / fo for h in _highs]
                        _lows = [(l - fo) / fo for l in _lows]
                        _closes = [(c - fo) / fo for c in _closes]
                        _max, _min = max(_volumes), min(_volumes)
                        _volumes = [(v-_min)/(_max-_min) for v in _volumes]

                        frame = [[o, h, l, c, v, i, r, m] for o, h, l, c, v, i, r, m in
                                 zip(_opens, _highs, _lows, _closes, _volumes, _ibss, _rsis, _mfis)]
                        frames.append(frame)
                        meta.append(meta_info)

                    marketkind = self.stock_meta.get_market(stockcode)
                    # for y
                    for step in range(self.window, time_len - self.barrier - feature_window):
                        step = step + feature_window
                        buy_slippage = self.get_slippage(opens[step], marketkind)
                        sell_slippage = self.get_slippage(closes[step + self.barrier - 1], marketkind)
                        _y = (closes[step + self.barrier - 1] - opens[step] - buy_slippage - sell_slippage) / opens[step]
                        _y = _y - self.transaction_fee
                        y.append(_y)

                    assert len(frames) == len(y) == len(meta)
                    stock_data += frames
                    metas += meta
                    pred += y
                    label += [1 if _y > self.threshold else 0 for _y in y]
                #break
        self.logger.info("Get Stock Data : {} patterns".format(len(stock_data)))
        self.logger.info("class 1: {}, class 0: {}".format(sum(label), len(label) - sum(label)))
        self.data_len = len(stock_data)
        self.features = len(stock_data[0][0])
        self.num_meta = len(metas[0])

        metas = self._meta_normalize(metas)
        self.stock_data = torch.tensor(stock_data, dtype=torch.float)
        self.metas = torch.tensor(metas, dtype=torch.float)
        self.pred = torch.tensor(pred, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.float)
        print(self.stock_data[0])
        print(self.metas[0])
        print(self.pred[0])

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
    #d = Data()
    #print(d[0])
    t = Train(epochs=50)
    t.run()

