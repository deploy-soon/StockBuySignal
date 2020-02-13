import fire
import json
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from misc import get_logger
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, LeakyReLU



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

def get_one_zero(list):
    if not list:
        return []
    _max = max(list)
    _min = min(list)
    _divider = _max - _min + 1e-8
    return [(v - _min) / _divider for v in list]

def get_normalize_list(list):
    if not list:
        return []
    _mean = float(sum(list)) / len(list)
    _std = np.std(list)
    return [(v - _mean) / _std for v in list], _mean, _std

class Data:

    def __init__(self, report, feat_len, batch_size, train_ratio, verbose=False):
        """
        :param path: data path
        :param report: report type string
        :param batch_size: train batch
        :param train_ratio:
        :param verbose:
        """
        self.logger = get_logger()
        self.report = report
        self.feature_len = feat_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose
        self.feature_norm = []

        self.threshold = 0.006

    def _get_danil(self, report):
        prices = report["prices"]
        if len(prices) < 5:
            return None, None
        prices = sorted(prices, key=lambda x: x["minute"])
        prices = sorted(prices, key=lambda x: x["date"])
        y = float(prices[-1]["close"] - prices[0]["open"]) / float(prices[0]["open"])
        volume = (report["prices"][0]["volume"] + report["prices"][1]["volume"]) / 2 * report["prices"][0]["open"]
        features = [
            np.log(1 + int(report["prices"][0].get("marketcap"))) if report["prices"][0].get("marketcap") else None,
            report["prices"][0].get("foreign"),
            report.get("profit_ratio"),
            np.log(1 + volume)
        ]
        for feature in features:
            if not feature:
                return None, None
        else:
            features = [float(f) for f in features]
            return features, y

    def _get_usang(self, report):
        prices = report["prices"]
        if len(prices) < 5:
            return None, None
        prices = sorted(prices, key=lambda x: x["minute"])
        prices = sorted(prices, key=lambda x: x["date"])
        y = float(prices[-1]["close"] - prices[0]["open"]) / float(prices[0]["open"])
        volume = (report["prices"][0]["volume"] + report["prices"][1]["volume"]) / 2 * report["prices"][0]["volume"]
        facility_fund = report.get("facility_fund")
        operation_fund = report.get("operation_fund")
        acquisition_fund = report.get("operation_fund")
        guitar_fund = report.get("operation_fund")
        funds = int(facility_fund) if facility_fund else 0
        funds += int(operation_fund) if operation_fund else 0
        funds += int(acquisition_fund) if acquisition_fund else 0
        funds += int(guitar_fund) if guitar_fund else 0
        features = [
            np.log(1 + int(report["prices"][0].get("marketcap"))) if report["prices"][0].get("marketcap") else None,
            report["prices"][0].get("foreign"),
            np.log(1 + funds) if funds else None,
            np.log(1 + volume)
        ]
        for feature in features:
            if not feature:
                return None, None
        else:
            features = [float(f) for f in features]
            return features, y

    def _get_cb(self, report):
        prices = report["prices"]
        if len(prices) < 5:
            return None, None
        prices = sorted(prices, key=lambda x: x["minute"])
        prices = sorted(prices, key=lambda x: x["date"])
        y = float(prices[-1]["close"] - prices[0]["open"]) / float(prices[0]["open"])
        volume = (report["prices"][0]["volume"] + report["prices"][1]["volume"]) / 2 * report["prices"][0]["open"]
        cb_amount = report.get("cb_amount")
        features = [
            np.log(1 + int(report["prices"][0].get("marketcap"))) if report["prices"][0].get("marketcap") else None,
            report["prices"][0].get("foreign"),
            np.log(1 + int(cb_amount)) if cb_amount else None,
            np.log(1 + volume)
        ]
        for feature in features:
            if not feature:
                return None, None
        else:
            features = [float(f) for f in features]
            return features, y

    def get_raw(self, report):
        if self.report == "danil":
            return self._get_danil(report)
        if self.report == "usang":
            return self._get_usang(report)
        if self.report == "cb":
            return self._get_cb(report)
        else:
            raise AttributeError("report type error")

    def load(self):
        _stock_data, answer = [], []
        self.feature_norm = []
        with open(pjoin("data", "dart_report.json"), "r", newline='', encoding='utf-8') as fin:
            dart_reports = json.load(fin)
            assert self.report in dart_reports
            reports = dart_reports[self.report]
            for report in reports:
                if "prices" not in report:
                    continue
                parse_data, y = self.get_raw(report)
                if not parse_data or not y:
                    continue
                _stock_data.append(parse_data)
                answer.append(y)
        stock_data = []
        data_transfer = []
        for i in range(self.feature_len):
            _feature = [v[i] for v in _stock_data]
            _feature, _mean, _std = get_normalize_list(_feature)
            self.feature_norm.append((_mean, _std))
            data_transfer.append(_feature)
        for i in range(len(_stock_data)):
            _data = []
            for j in range(self.feature_len):
                _data.append(data_transfer[j][i])
            stock_data.append(_data)
        self.logger.info("Get Stock Data : {} rows".format(len(stock_data)))
        self.logger.info("Average return value: {}".format(sum(answer) / (len(answer) + 1e-8)))
        return stock_data, answer

    def launch(self):
        X, y = self.load()
        perm = list(range(len(X)))
        random.shuffle(perm)
        X = [X[idx] for idx in perm]
        y = [y[idx] for idx in perm]
        y_binary = [1 if _y >= self.threshold else 0 for _y in y]
        self.logger.info("one : {}, zero : {}".format(sum(y_binary), len(y_binary) - sum(y_binary)))
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
                yield np.array(self.train_X[offset:limit]).reshape((-1, self.feature_len)), \
                      np.array(self.train_y[offset:limit])

    def gen_vali(self):
        vali_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, vali_num, self.batch_size):
                limit = min(vali_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.feature_len)), \
                      np.array(self.vali_y[offset:limit])

    def gen_test(self):
        test_num = self.vali_num // self.batch_size * self.batch_size
        while True:
            for offset in range(0, test_num, self.batch_size):
                limit = min(test_num, offset + self.batch_size)
                yield np.array(self.vali_X[offset:limit]).reshape((-1, self.feature_len)), \
                      np.array(self.test_y[offset:limit])


class Classifier:

    def __init__(self, report, feat_len, batch_size, train_ratio, epochs, verbose=False):
        self.logger = get_logger()
        self.report = report
        self.feat_len = feat_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose

        self.data_iter = Data(report=self.report,
                              feat_len=self.feat_len,
                              batch_size=self.batch_size,
                              train_ratio=self.train_ratio)
        self.data_iter.launch()

        self.log_path = "tmp/tensorboard"
        self.checkpoint_path = "tmp/checkpoint"
        self.epochs = epochs

    def build_model(self, with_compile=True):
        input = Input(shape=(self.feat_len, ))
        x = Dense(128)(input)
        x = LeakyReLU()(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        output = Dense(1, activation="sigmoid", name="logits")(x)
        model = Model(input, output)
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
        assert len(x) == len(self.data_iter.feature_norm), "feature number {}, {}".format(len(x), len(self.data_iter.feature_norm))
        _x = [(d - _mean) / _std for d, (_mean, _std) in zip(x, self.data_iter.feature_norm)]
        _x = np.array(_x)
        embd = self._feed_forward([_x.reshape(-1, self.feat_len), 0])
        return embd

    def feed_forward_batch(self, x):
        embd = self._feed_forward([x, 0])
        return embd

    def test(self):
        test_batch = self.data_iter.get_test_batch_num()
        test_iterator = self.data_iter.gen_test()
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
        plot_data = []
        for ret in return_log:
            all_return *= (1+ret - 0.007)
            plot_data.append(all_return)
        self.logger.info("average return : {} all_return : {}".format(avg_return, all_return))
        # plt.plot(plot_data)
        # plt.show()

    def train(self):
        self.logger.info("start training")

        net = self.build_model()
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
        # checkpointer = ModelCheckpoint(filepath=self.checkpoint_path+"{epoch:02d}-{val_loss:.4f}.hdf5",
        #                                verbose=True, monitor="val_loss", save_best_only=True)
        # tensor_board = TensorBoard(log_dir=self.log_path, histogram_freq=0,
        #                            write_graph=True, write_images=True, batch_size=self.batch_size)
        # callbacks = [checkpointer, tensor_board]
        net.fit_generator(self.data_iter.gen_train(),
                          steps_per_epoch=self.data_iter.get_train_batch_num(),
                          epochs=self.epochs,
                          verbose=False,
                          # callbacks=callbacks,
                          validation_data=self.data_iter.gen_vali(),
                          validation_steps=self.data_iter.get_vali_batch_num(),
                          initial_epoch=0)
        print(self.data_iter.feature_norm)
        # self.feed_forward([20, 0.5, 0.5, 8])
        self.test()

        net.save(pjoin("res", "dart.h5"))

if __name__ == "__main__":
    fire.Fire(Classifier)
