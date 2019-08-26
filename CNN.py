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
        self.checkpoint_path = "tmp"
        self.epochs = epochs

    def build_model(self, with_compile=True):
        chart = Input(shape=(self.window, self.bandwidth, 1))
        x = Conv2D(32,
                   kernel_size=(3, 3),
                   padding='valid')(chart)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(64,
                   kernel_size=(2, 2),
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        for _ in range(1):
            x = Conv2D(128,
                       kernel_size=(2, 2),
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2,2), strides=None)(x)

        # pattern recognition
        x = Conv2D(256,
                   kernel_size=(2, 2),
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
                          metrics=["accuracy"])
        self._feed_forward = K.function([model.input, K.learning_phase()],
                                        [model.get_layer("logits").output])

        return model

    def feed_forward(self, x):
        embd = self._feed_forward([x.reshape(1, self.window, self.bandwidth, 1), 0])
        return embd.reshape(-1) # flatten

    def feed_forward_batch(self, x):
        embd = self._feed_forward([x, 0])
        return embd

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

        net.save(pjoin("res", "CNN.h5"))

if __name__ == "__main__":
    fire.Fire(CNN)

