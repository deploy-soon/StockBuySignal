import h5py
import json
import random
import pathlib
import numpy as np
from os.path import join as pjoin
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, LeakyReLU

from misc import get_logger

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
    PMF = [MF[i+1] for i in range(len(typical_prices)) if typical_prices[i] < typical_prices[i+1]]
    NMF = [MF[i+1] for i in range(len(typical_prices)) if typical_prices[i] > typical_prices[i+1]]
    MR = PMF / (NMF + 1e-10)
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



class Data:

    def __init__(self, path="data/ninetoten.h5", batch_size=128, train_ratio=0.8, verbose=False):

        self.logger = get_logger()
        self.path = path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose

        self.barrier = 4
        self.window = 7
        assert self.barrier > 0, "target inference"

        with open("data/stock_meta.json", "r") as fin:
            self.stock_meta = json.load(fin)
            self.logger.info("stock meta info loaded {}".format(len(self.stock_meta)))

        self.transaction_fee = 0.0035
        self.threshold = 0.003

    def get_market_info(self, stock_code):
        stock_info = self.stock_meta.get(stock_code, {})
        return stock_info.get("marketkind", None)

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
                if stockcode not in self.stock_meta:
                    continue

                datetimes = set(fin[stockcode]["dates"][:])
                for _datetime in datetimes:
                    opens = [i for i, p in zip(fin[stockcode]["opens"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    highs = [i for i, p in zip(fin[stockcode]["highs"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    lows = [i for i, p in zip(fin[stockcode]["lows"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    closes = [i for i, p in zip(fin[stockcode]["closes"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    volumes = [i for i, p in zip(fin[stockcode]["volumes"][:], fin[stockcode]["dates"][:]) if p == _datetime]
                    minutes = [i for i, p in zip(fin[stockcode]["minutes"][:], fin[stockcode]["dates"][:]) if
                               p == _datetime]

                    if len(minutes) < 60:
                        continue

                    ibss = []
                    for o, h, l, c in zip(opens, highs, lows, closes):
                        ibss.append(ibs(o, h, l, c))
                    time_len = len(opens)
                    frames, y = [], []

                    # for X
                    for step in range(time_len - self.window - self.barrier):
                        _opens = opens[step:step+self.window]
                        _highs = highs[step:step+self.window]
                        _lows = lows[step:step+self.window]
                        _closes = closes[step:step+self.window]
                        _volumes = volumes[step:step+self.window]

                        frame = ibss[step:step + self.window]
                        frame.append(rsi(opens[step:step + self.window]))
                        frames.append(frame)

                    marketkind = self.get_market_info(stockcode)
                    # for y
                    for step in range(self.window, time_len - self.barrier):
                        buy_slippage = self.get_slippage(opens[step], marketkind)
                        sell_slippage = self.get_slippage(closes[step + self.barrier - 1], marketkind)
                        _y = (closes[step + self.barrier - 1] - opens[step] - buy_slippage - sell_slippage) / opens[step]
                        _y = _y - self.transaction_fee
                        y.append(_y)

                    assert len(frames) == len(y)
                    stock_data += frames
                    answer += y

        self.logger.info("Get Stock Data : {} patterns".format(len(stock_data)))
        self.features = len(stock_data[0])
        return stock_data, answer

    def test(self):
        X, y = self.load()
        print(X[:5], y[:5])
        print(len(y))

    def launch(self):
        X, y = self.load()
        perm = list(range(len(X)))
        random.shuffle(perm)
        X = [X[idx] for idx in perm]
        y = [y[idx] for idx in perm]
        y_binary = [1 if _y >= self.threshold else 0 for _y in y]
        self.logger.info("one : {}, zero : {}".format(sum(y_binary), len(y) - sum(y_binary)))
        self.train_num = int(len(X) * self.train_ratio)
        self.vali_num = len(X) - self.train_num
        self.logger.info("train num : {} | validation num : {}".format(self.train_num, self.vali_num))
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
                yield np.array(self.train_X[offset:limit]).reshape((-1, self.features)), \
                      np.array(self.train_y[offset:limit])

    def gen_vali(self):
        vali_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, vali_num, self.batch_size):
                limit = min(vali_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.features)), \
                      np.array(self.vali_y[offset:limit])

    def gen_test(self):
        test_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, test_num, self.batch_size):
                limit = min(test_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.features)), \
                      np.array(self.test_y[offset:limit])



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class PatternModel:

    def __init__(self, path="data/ninetoten.h5", batch_size=128, train_ratio=0.8, epochs=30, verbose=False):
        self.logger = get_logger()
        self.path = path

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose

        self.log_path = "tmp/tensorboard"
        self.checkpoint_path = "tmp/checkpoint"
        self.epochs = epochs

    def build_model(self, with_compile=True):
        pattern = Input(shape=(self.features, ))

        # fc layer
        x = Dense(128)(pattern)
        x = LeakyReLU()(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        output = Dense(1, activation="sigmoid", name="logits")(x)
        model = Model(pattern, output)
        model.summary()
        if with_compile:
            optimizer = Adam(lr=0.0001)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=["accuracy", f1_m, recall_m, precision_m])
        self._feed_forward = K.function([model.input, K.learning_phase()],
                                        [model.get_layer("logits").output])

        return model

    def feed_forward(self, x):
        embd = self._feed_forward([x.reshape(1, self.features, 1), 0])
        return embd.reshape(-1) # flatten

    def feed_forward_batch(self, x):
        embd = self._feed_forward([x, 0])
        return embd

    def test(self, iterator):
        test_batch = iterator.get_test_batch_num()
        test_iterator = iterator.gen_test()
        return_log = []
        total_test_num = 0
        for b in range(test_batch):
            test_X, test_y = next(test_iterator)
            test_pred = self.feed_forward_batch(test_X)[0]
            return_log += [_y for _y, _pred in zip(test_y, test_pred) if _pred > 0.5]
            total_test_num += len(test_y)
        self.logger.info("select {} signal among {} cases".format(len(return_log), total_test_num))
        avg_return = sum(return_log) / (len(return_log) + 1e-8)
        all_return = 1.0
        for ret in return_log:
            all_return *= (1+ret)
        self.logger.info("average return : {} all_return : {}".format(avg_return, all_return))

    def train(self):
        self.logger.info("start training")
        data_iter = Data(path=self.path,
                         batch_size=self.batch_size,
                         train_ratio=self.train_ratio)
        data_iter.launch()
        self.features = data_iter.features
        net = self.build_model()
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
        # checkpointer = ModelCheckpoint(filepath=self.checkpoint_path+"{epoch:02d}-{val_loss:.4f}.hdf5",
        #                                verbose=True, monitor="val_loss", save_best_only=True)
        # tensor_board = TensorBoard(log_dir=self.log_path, histogram_freq=0,
        #                            write_graph=True, write_images=True, batch_size=self.batch_size)
        # callbacks = [checkpointer, tensor_board]
        net.fit_generator(data_iter.gen_train(),
                          steps_per_epoch=data_iter.get_train_batch_num(),
                          epochs=self.epochs,
                          verbose=True,
                          # callbacks=callbacks,
                          validation_data=data_iter.gen_vali(),
                          validation_steps=data_iter.get_vali_batch_num(),
                          initial_epoch=0)

        self.test(data_iter)

        net.save(pjoin("res", "pattern.h5"))



if __name__ == "__main__":
    # d = Data()
    # d.test()
    m = PatternModel()
    m.train()