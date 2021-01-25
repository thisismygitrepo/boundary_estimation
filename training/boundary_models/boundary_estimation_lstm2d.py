"""
Using a single LSTMConv2D which performs 2D conv on 16x16x2 *image* that is changing across "time".
This did not work very well. No norming was used, this was intentional to maintain Sii dominance.
"""

import alexlib.toolbox as tb
import alexlib.deeplearning as dl
import resources.s_params as stb
import os
from tensorflow.keras.layers import Dense
import tensorflow as tf
import training.boundary_models.boundary_prediction_lstm as ref
from measurements import PA


class HParams(dl.HyperParam):
    def __init__(self):
        # ================  env ========================
        super().__init__()
        self.exp_name = 'lstm2d'  # 'gru_vs_rnn_vs_lstm'
        self.root = 'tmp/lstm2d/' + self.exp_name
        self.device_name = dl.Device.gpu0
        self.array = ['threeC', 'logp'][0]
        os.makedirs(self.save_dir, exist_ok=True)
        # ================= data ====================
        self.s_select = stb.SSelect.all_
        self.dtype = stb.Dtype.rect
        self.shape = stb.Shape.eye
        self.ignore_last50 = False
        self.sort_freq = False
        # =============== Model =========================
        self.num_points = int(360 / 10)
        self.filter_size = 3
        self.filters = 7
        self.l1 = 0.0005
        self.milk = [False, True][0]
        self.nsl = [False, True][0]
        # ================ Train ======================
        self.split = 0.1
        self.lr = 0.0005
        self.batch_size = 32
        self.num_epochs = 30


class LSTM2D(ref.BoundaryTFModels):
    def __init__(self, hyperp, data_):
        super().__init__(self)
        self.data = data_
        self.hp = hyperp

        def create_lstm_model():
            model_ = tf.keras.models.Sequential()
            model_.add(tf.keras.layers.ConvLSTM2D(filters=self.hp.filters,
                                                  kernel_size=(self.hp.filter_size, self.hp.filter_size)))
            model_.add(tf.keras.layers.Reshape((-1,)))
            model_.add(Dense(units=self.hp.num_points, activation=tf.keras.activations.tanh,
                             kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1)))
            model_.add(Dense(units=self.hp.num_points, kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1)))
            model_.compile(optimizer=tf.keras.optimizers.Adam(lr=self.hp.lr),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[tf.keras.metrics.mean_absolute_error, dl.get_mean_max_error(tf)()])
            return model_

        self.model = create_lstm_model()
        print(self.model(self.data.s[:self.hp.batch_size]).shape)

    def fit(self, **kwargs):
        self.history = self.model.fit(x=self.data.split.x_train, y=self.data.split.y_train,
                                      validation_data=(self.data.split.x_test, self.data.split.y_test),
                                      batch_size=self.hp.batch_size,
                                      epochs=self.hp.epochs,
                                      verbose=1, shuffle=True, callbacks=[])
        self.history = self.history.history.copy()


if __name__ == '__main__':
    hp = HParams()
    d = ref.AnthDataReader(hp)
    model = LSTM2D(hp, d)
    model.evaluate()  # visualize results
    pa = PA()  # Apply on clinical S params
