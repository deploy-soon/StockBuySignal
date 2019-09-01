import fire
import pathlib
import numpy as np
from os.path import join as pjoin
from data import Data
from misc import get_logger
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Conv2D, BatchNormalization, \
    MaxPooling2D, GlobalMaxPooling2D, LeakyReLU


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

class CNN:

    def __init__(self, path, window, stride, minute_after, bandwidth,
                 batch_size, train_ratio, epochs, verbose=False):
        self.logger = get_logger()
        self.path = path
        self.window = window
        self.stride = stride
        self.minute_after = minute_after
        self.bandwidth = bandwidth

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.verbose = verbose

        self.log_path = "tmp/tensorboard"
        self.checkpoint_path = "tmp/checkpoint"
        self.epochs = epochs

    def build_model(self, with_compile=True):
        chart = Input(shape=(self.window, self.bandwidth, 1))
        x = Conv2D(32,
                   kernel_size=(3, 6),
                   padding='valid')(chart)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(64,
                   kernel_size=(2, 4),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        for _ in range(1):
            x = Conv2D(128,
                       kernel_size=(2, 4),
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=None)(x)

        # pattern recognition
        x = Conv2D(256,
                   kernel_size=(2, 4),
                   padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = GlobalMaxPooling2D()(x)

        # fc layer
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        output = Dense(1, activation="sigmoid", name="logits")(x)
        model = Model(chart, output)
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
        embd = self._feed_forward([x.reshape(1, self.window, self.bandwidth, 1), 0])
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
        data_iter = Data(path = self.path,
                         window = self.window,
                         stride = self.stride,
                         minute_after = self.minute_after,
                         bandwidth = self.bandwidth,
                         batch_size=self.batch_size,
                         train_ratio=self.train_ratio)
        data_iter.launch()
        net = self.build_model()
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
        checkpointer = ModelCheckpoint(filepath=self.checkpoint_path+"{epoch:02d}-{val_loss:.4f}.hdf5",
                                       verbose=True, monitor="val_loss", save_best_only=True)
        tensor_board = TensorBoard(log_dir=self.log_path, histogram_freq=0,
                                   write_graph=True, write_images=True, batch_size=self.batch_size)
        callbacks = [checkpointer, tensor_board]
        net.fit_generator(data_iter.gen_train(),
                          steps_per_epoch=data_iter.get_train_batch_num(),
                          epochs=self.epochs,
                          verbose=True,
                          callbacks=callbacks,
                          validation_data=data_iter.gen_vali(),
                          validation_steps=data_iter.get_vali_batch_num(),
                          initial_epoch=0)

        self.test(data_iter)

        net.save(pjoin("res", "CNN.h5"))

if __name__ == "__main__":
    fire.Fire(CNN)

