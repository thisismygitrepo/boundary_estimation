"""We want to estimate the normal distance from the antenna, rather than radii from the origin as done in all other
files. The performance in previous cases was uneven across antennas. In physics based model, antenna 1 was suffering
while other antennas were doing good. because estimation radii of boundary was unnatural (geometrically) to antenna1.
The GIF in experimental/physics explains.
"""

import alexlib.toolbox as tb
import alexlib.deeplearning as dl
import numpy as np
import tensorflow as tf
import training.boundary_models.boundary_prediction_lstm as ref
import measurements as ms
import resources.s_params as stb
import pywt


class HParams(dl.HyperParam):
    def __init__(self):
        # =================== Enviroment ========================
        super().__init__()
        self.root = 'tmp/normal_estimation'
        self.exp_name = 'polar_vs_rect_rect'
        self.device_name = dl.Device.gpu0
        self.array = ['threeC', 'threeB', 'threeB+threeC', 'logp', 'SkinSkullBucket3C'][-1]
        self.pkg_name = tf.__name__
        # ===================== DATA ============================
        self.diags = 1
        self.s_select = stb.SSelect.get_diagonals_indices(16, self.diags)
        self.dtype = stb.Dtype.rect
        self.shape = stb.Shape.callable(lambda x: x[...])  # add 1 axis (ip_size) if dtype is mangitude.
        self.cal = [False, True][0]
        self.freq_select = None  # np.arange(0.5 + 0.004 * 7, 2.0, 0.004)  # s_select specific 368 frequencies
        self.norm_factor = None
        self.sort_freq = [False, True][0]
        self.dmd = [False, True][0]
        # =================== Model =============================
        self.hidden_size = 32
        self.rnn_layers = 2
        self.l1 = 0.0  # L1 regularization
        self.beta = 0.0  # uniformity regularization
        self.symmetry = list(range(16))  # )) + list(range(8))
        self.bi = [False, True][0]  # bidirectional LSTM or not.
        self.merge_mode = ['sum', 'mul', 'concat', 'ave', None][0]  # Type of merge in Bidirectional LSTM
        # ===================== Training ========================
        self.split = 0.4
        self.lr = 0.001
        self.batch_size = 32
        self.epochs = 700
        self.shuffle = True
        self.nsl = [False, True][0]
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.01
        self.adv_grad_norm = 'infinity'
        self.save_code()


class DistanceDataReader(ref.AnthDataReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, process_s=True, process_labels=True, split_data=True, **kwargs)
        self.ip_size = ((self.hp.diags - 1) * 2 + 1) * 2
        self.ip_size = int(self.ip_size / 2) if self.hp.dtype == stb.Dtype.magnitude else self.ip_size
        self.data_specs['antenna_ax'] = 2


class ScaloGramReader(ref.AnthDataReader):
    def __init__(self, *args, generate=True, **kwargs):
        super().__init__(*args, process_s=False, process_labels=True, split_data=False, **kwargs)
        self.scalo_path = self.ssb3c.parent / "ssb_scalogram.npy"
        if generate:
            self.generate()

    def generate(self):
        load = False
        if load:
            self.s = np.linalg.norm(np.load(self.scalo_path) / 4, axis=-1)[..., None]
        else:
            self.generate_scalograms()
        self.split_now()
        self.data_specs['antenna_ax'] = 1

    def generate_scalograms(self, save=False):
        # generate images out of 1D curves using scaleograms
        import pywt
        q = self.s_raw[:, :, range(16), range(16)]  # complex values of main diagonal
        q = tb.Manipulator.merge_axes(q.astype(np.complex64), 0, -1)
        q = tb.List(q)
        q = q.apply(lambda x: pywt.cwt(x, np.arange(1, 256), "cgau2")[0], jobs=28).np
        q = tb.Manipulator.expand_axis(q, 0, 16).__abs__()
        self.s = q
        if save:
            np.save(self.scalo_path, q.numpy())
        # tb.ImShow(w.__abs__())


class SingeModelCreator(ref.BoundaryTFModels):
    """
    """
    def __init__(self, hp__, data__, build=True, *kwargs):
        super().__init__(hp__, data__, *kwargs)
        self.pa = None
        if build:
            self.model = self.create_lstm_model(name='MLP')
            self.build(shape=(self.data.seq_len, self.data.ip_size))
            self.summary()
            self.compile(optimizer=self.Op.Adam(lr=self.hp.lr),
                         loss=self.Lo.MeanSquaredError(),
                         metrics=[self.Me.MeanAbsoluteError(), dl.Norm.get_mean_max_error(tf)()])

    def create_lstm_model(self, name=None):
        s = self
        model_ = s.M.Sequential(name=name)
        # model_.add(s.L.Reshape((self.data.seq_len, self.data.ip_size)))  # ip_size is usually (2) * related_s's
        for ii in range(self.hp.rnn_layers):
            rnn = s.L.SimpleRNN(units=self.hp.hidden_size,
                                return_sequences=True if (self.hp.rnn_layers - 1) > ii else False)
            if self.hp.bi:
                rnn = s.L.Bidirectional(rnn, merge_mode=self.hp.merge_mode)
            model_.add(rnn),
        model_.add(s.L.Dense(units=32, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        model_.add(s.L.LeakyReLU(alpha=0.3))
        model_.add(s.L.Dense(units=1, kernel_regularizer=s.R.l1(l=self.hp.l1), activation=None))
        return model_

    def fit(self, plot=True, **kwargs):
        fit_kwargs = dict(x=self.reshape(self.data.split.x_train), y=self.data.split.y_train.reshape((-1,)),
                          validation_data=(self.reshape(self.data.split.x_test),
                                           self.data.split.y_test.reshape((-1,))))
        super().fit(plot=plot, update_default=True, fit_kwargs=fit_kwargs, **kwargs)

    def reshape(self, s):
        op = tb.Manipulator.merge_axes(s, 0, self.data.antenna_ax)
        # Merge Batch axis with Antenna axis in Decoupled predictions Models.
        return op

    def infer(self, s):  # usually assumes a preprocessed input, just like the data in dataset.
        output = self.reshape(s)  # reshape the method defined by the model
        output = self.model.predict(output)
        output = output.reshape((s.shape[0], -1))  # reshape, the numpy method
        return output

    @tb.batcher('method')
    def postprocess(self, distances, num_pts=360, interp=stb.Interp.spline, legend="NN Prediction", **kwargs):
        distances_ = distances * self.data.std + self.data.mean
        landing_pts = stb.get_xy_from_normals(distances_, self.data.antenna_positions)
        xy = stb.interpolate(landing_pts, interp=interp, num_pts=num_pts)
        xy = stb.allign_boundary(xy, num_pts=num_pts)
        obj = stb.BoundaryDescriptor(xy=xy, distances=distances_, landing_pts=landing_pts, legend=legend,
                                     **kwargs)
        obj.r = np.linalg.norm(xy, axis=-1)
        return obj

    def predict_from_da_pos(self, pos):
        prediction = self.model.predict(pos.da)  # 16 x 10 ==> 16 x 1
        postprocessed = self.postprocess(prediction.T)[0]
        obj = tb.Struct(prep=pos.da, pred=prediction, postp=postprocessed)
        pos.predictions.append(obj)

    def act_on_pa(self, pos=None, da=False, filters=None):
        if filters is None:
            def filters(x):
                return x.em_boundary is not None

        if pos is None:
            if self.pa is None:
                self.pa = ms.PA()
            pos = self.pa.positions.filter(filters)
            self.pa.pos = pos

        class MyArtist(stb.PlotBoundaryArtist):
            def __init__(self, other):
                super(MyArtist, self).__init__()
                self.s = other

            def plot(self, *positions, **kwargs):
                self.get_axes()
                if da:
                    self.s.predict_from_da_pos(positions[0])
                else:
                    self.s.predict_from_position(positions[0], viz=False)
                positions[0].em_boundary.legend = f"Ali's prediction"
                positions[0].predictions[-1].postprocessed.legend = f"Alex's prediction"
                positions[0].em_boundary.plot(positions[0].predictions[-1].postprocessed, ax=self.ax[0])

        self.plotter = tb.VisibilityViewerAuto(data=[pos], artist=MyArtist(self))

    def test_outcome(self):
        self.tmp = self.predict(self.data.s[:10])
        tb.VisibilityViewerAuto(data=[self.tmp], artist=stb.PlotBoundaryArtist())

    def test_outcome_pa(self):
        self.pa.pos.apply(lambda x: self.predict_from_position(x, viz=False))
        tb.VisibilityViewerAuto(data=[self.pa.pos.postprocessed], artist=stb.PlotBoundaryArtist())


class PerAntennaModelCreator(SingeModelCreator):
    """Programming point: Composition VS inheritence:
    At this point, instead of inheriting and overriding methods, it is much better to use composition.
    That is, use instances of parent class as attributes, instead of inheriting from the parent.
    """
    def __init__(self, hp__, data__, model_class, data_class):
        super().__init__(hp__, data__, build=False)
        self.model_class = model_class
        self.models = tb.List()
        self.data_class = data_class
        for antenna in range(self.data.ports):
            datum = self.data_class(self.hp)
            datum.s = datum.s[..., antenna][..., None]
            datum.labels = datum.labels[:, antenna]
            datum.split_my_data_now()
            model_ = self.model_class(hp__=self.hp, data__=datum)
            self.models.append(model_)

    def fit(self, **kwargs):
        self.models.fit()

    def infer(self, s):
        op = np.zeros((s.shape[0], self.data.ports), dtype='float32')
        for index, amodel in enumerate(self.models):
            selection = stb.SSelect.get_indices_per_antenna(idx=index, num_diag=self.hp.diags, size=self.data.ports)
            distances = amodel.infer(s[:, :, selection])
            op[:, index] = distances.squeeze()
        return op


#%%


class OneDConvModel(SingeModelCreator):
    def __init__(self, hp__, data__):
        super(OneDConvModel, self).__init__(hp__, data__, build=False)
        self.model = self.create_1d_conv_model()
        self.build(shape=(self.data.seq_len, self.data.ip_size))
        self.summary()
        _ = self.evaluate()
        # this gives a good test to whether the model is flexible enough and is producing random prediction.
        self.compile()

    def create_1d_conv_model(self):
        model_ = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=10, kernel_size=(15,), strides=(10,), padding='valid',
                                   activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Conv1D(filters=2, kernel_size=(15,), strides=(5,), padding='same',
                                   activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Conv1D(filters=5, kernel_size=(5,), strides=(3,), padding='valid',
            #                        activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv1D(filters=5, kernel_size=(5,), strides=(3,), padding='valid',
            #                        activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Conv1D(filters=2, kernel_size=(3,), strides=(1,), padding='valid',
            #                        activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=8, kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1),
                                  activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(units=8, kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1),
                                  activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(units=8, kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1),
                                  activation=None), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(units=1, activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1)),
        ])
        return model_


class ConvModel(SingeModelCreator):
    def __init__(self, hp__, data__):
        super(ConvModel, self).__init__(hp__, data__, build=False)
        self.model = self.create_conv_model()
        self.build(shape=(255, 751, 1))
        self.summary()
        _ = self.evaluate()
        self.compile()

    def create_conv_model(self):
        model_ = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=5, kernel_size=(7, 10), strides=(4, 6), padding='valid',
                                   activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Conv2D(filters=5, kernel_size=(6, 8), strides=(3, 4), padding='same',
                                   activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Conv2D(filters=5, kernel_size=(4, 6), strides=(2, 3), padding='same',
                                   activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Conv2D(filters=2, kernel_size=(4, 4), strides=(2, 2), padding='valid',
                                   activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=5, kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1),
                                  activation=None,), tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(units=1, activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l1(l=self.hp.l1)),
        ])
        return model_

    def preprocess(self, *args, **kwargs):
        """
        :param args: Raw S params (complex valued)
        :param kwargs:
        :return: should generate data that look like training set.
        """
        q = args[0].s[:, :, range(16), range(16)]  # complex values of main diagonal
        q = tb.Manipulator.merge_axes(q, 0, -1)  # merge batch with antennas
        q = tb.List(q)
        q = q.apply(lambda x: pywt.cwt(x, np.arange(1, 256), "cgau2")[0]).np
        q = tb.Manipulator.expand_axis(q, 0, 16)
        q = stb.Norm.complex_to_real(q / 4, stb.Dtype.magnitude, axis=-1)[..., None]
        return q


#%% main


if __name__ == '__main__':
    hp = HParams()
    d = ScaloGramReader(hp)
    m = ConvModel(hp__=hp, data__=d)
    # m.fit()
