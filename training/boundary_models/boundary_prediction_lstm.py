""" Methods to do convolutions

* 1x1 conv kernel: this one is equivalent to taking linear combination of channels. Is used to reduce/increase
  number of channels (number of filters) without really doing any spatial processing.

* Kernel with image size filter. This filter cannot move spatially, and therefore, acts like taking a linear
  combination of channels, except that the combination is per pixel.

* DepthWise Spatial convolution, does convolution per channel. Results can be combined by further 1x1 convolution.

* 3D convolution, channels depth of filter may not necessarily be equal to number of channels in input data.

* Data can be transposed so that channels are made a spatial axis and what was applicable on spatial axes is now
  applicable to channels and vice versa.


Methods to handle sequential data in deep learning:
===================================================

* Feed the signal as is to RNNs.
* Use a technique to reduce the signal before RNN, E.g. wavelet scattering transform.
* Convert the signal to image then use CNN. Conversion techniques: Spectrogram, Scalogram, Constant-Q Transform.

"""

from abc import ABC
import alexlib.toolbox as tb
import resources.s_params as stb
import alexlib.deeplearning as dl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import resources.source_of_truth as st
# import neural_structured_learning as nsl
# import measurements


class HParams(dl.HyperParam):
    def __init__(self):
        # =================== Enviroment ========================
        super().__init__()
        self.exp_name = 'trained_on_data_orig_nocal'  # 'gru_vs_rnn_vs_lstm'
        self.root = 'tmp/lstm'
        self.pkg_name = tf.__name__
        self.device_name = dl.Device.gpu1
        self.array = ['threeC', 'threeB', 'threeB+threeC', 'logp', 'SkinSkullBucket3C'][-1]
        self.ignore_last50 = False
        # ===================== DATA ============================
        self.s_select = stb.SSelect.get_diagonals_indices(16, 1)
        self.cal = [False, True][0]
        self.dtype = stb.Dtype.rect
        self.shape = stb.Shape.eye
        self.freq_select = None  # np.arange(0.5 + 0.004 * 7, 2.0, 0.004)  # s_select specific 368 frequencies
        self.norm_factor = 1.0  # tb.get_nf(self.freq_select, factor=10000, decay=0.0015, mag=16 if self.cal else 8)
        self.sort_freq = [False, True][0]
        self.dmd = [False, True][0]
        self.seed = 234
        self.shuffle = True
        self.dmd = False
        # =================== Model =============================
        self.num_points = 1  # int(360 / 10)
        self.hidden_size = 36
        self.rnn_layers = 1
        self.bi = [False, True][0]  # bidirectional LSTM or not.
        self.merge_mode = ['sum', 'mul', 'concat', 'ave', None][0]  # Type of merge in Bidirectional LSTM
        # ===================== Training ========================
        self.split = 0.2
        self.lr = 0.0005
        self.batch_size = 32
        self.epochs = 30
        self.l1 = 0.0005
        self.nsl = [False, True][0]
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.01
        self.adv_grad_norm = 'infinity'


class AnthDataReader(dl.DataReader):
    def __init__(self, hyp=None, split_data=True, process_labels=True, process_s=True, load=False):
        """
        1. array: to tell the reader what to read
        2. ignore_last50: to tell what to ignore
        """
        super().__init__(hyp)
        self.plotter = False
        base = st.d / r'Boundary/AnthonyBoundaryMeasurements'
        self.logp_path = base / 'logp_array'
        self.logp_experiments = tb.List(['Cal1', 'Container1', 'Container1Target1', 'Container2', 'Container2Target1',
                                         'Container3', 'Container3Target1', 'PhantomH', 'PhantomUH2'])
        self.threeC_path = base / 'threeC_array'
        self.threeC_experiments = tb.List(['Cal1', 'Container1', 'Container1Target1', 'Container2', 'Container2Target1',
                                           'PhantomH', 'PhantomUH2'])
        self.threeB_path = base / 'threeB_array'
        self.threeB_experiments = tb.List(['PhantomHSlice1', 'PhantomHSlice2', 'PhantomUH2Slice1',
                                           'PhantomUH2Slice1-Wig1', 'PhantomUH2Slice2'])
        self.ssb3c = base / 'SkinSkullBucket3C'
        self.data_dict = {'threeC': self.threeC_experiments.apply(lambda x: self.threeC_path / x + '.npy'),
                          'threeB': self.threeB_experiments.apply(lambda x: self.threeB_path / x + '.npy'),
                          'logp': self.logp_experiments.apply(lambda x: self.logp_path / x + '.npy'),
                          'threeB+threeC': self.threeC_path.search('*')[:5] + self.threeB_path.search('*'),
                          'SkinSkullBucket3C': self.ssb3c.search('*')}

        if load:
            outputs = self.read_data(self.data_dict[self.hp.array])
            self.s_raw, self.xy_labels, self.names, self.freq, antenna_positions, self.experiment = outputs
            self.data_specs['antenna_positions'] = antenna_positions.to_numpy()
            self.data_specs['ports'] = len(self.antenna_positions)
            self.data_specs['size'] = len(self.s_raw)
            self.sim = None

            self.labels = None
            if process_labels:
                self.process_labels()

            base = st.g / 'emvdata/MeasurementData/20200309-AS-CNC-Boundary-Movement'
            self.cals_available = dict(threeC=base / 'EMV-3C/Cal1/CalibratedData/IF10k-CNC-CentreExp.s16p',
                                       logp=base / 'LogPeriodic/Cal1/Data/IF10k-CNC-CentreExp.s16p')
            self.cals_available['threeB'] = self.cals_available['threeC']
            self.cals_available['threeB+threeC'] = self.cals_available['threeC']
            self.cals_available['SkinSkullBucket3C'] = None
            self.cal_ip = self.cals_available[self.hp.array]
            self.cal_dict = None
            self.s = None
            if process_s:
                self.process_s()

            self.split = None
            if split_data:
                self.split_now()

    def split_now(self):
        self.data_split(self.s, self.labels, self.names, self.experiment, np.arange(0, self.size),
                        strings=['x', 'y', 'names', 'experiment', 'index'])

    @staticmethod
    def read_data(paths):
        s_, xy_labels_, names_, freq_, antenna_positions_, experiment_ = [], [], [], [], [], []
        from tqdm import tqdm
        for i, afolder in tqdm(enumerate(paths)):
            datum = np.load(afolder, allow_pickle=True).all()
            s_.append(datum['s'])
            names_ += [afolder.parent.name + '_' + afolder.name.split('.')[0] + '_' + k for k in datum['names']]
            experiment_ += [i for _ in range(len(datum['names']))]
            xy_labels_.append(datum['positions'])
            antenna_positions_ = datum[
                'antenna_positions']  # grab those from any experiment folder belonging to same array
            freq_ = datum['freq']  # around 751, instead of 1000 like in the case of simulations
        s_ = np.concatenate(s_, axis=0)
        xy_labels_ = np.concatenate(xy_labels_, axis=0)
        return s_, xy_labels_, names_, freq_, antenna_positions_, experiment_

    def process_labels(self):
        """ Converts xy labels to distances from antennas. Assumes that xy labels are prealigned (start from zero).
        :return:
        """
        self.labels = np.array([stb.get_normal_distances(item, self.antenna_positions)[0] for item in self.xy_labels])
        self.data_specs['mean'], self.data_specs['std'] = 11, 2.5
        self.labels = ((self.labels - self.mean) / self.std).astype('float32')

    def process_s(self):
        # Processing S values:
        if self.hp.cal:
            self.cal_dict = stb.S(self.cal_ip)
        else:
            self.cal_dict = None
        if self.hp.dmd:
            s = self.dmd_processing(self.s_raw)
        else:
            s = stb.Preprocessing(self.hp, self.s_raw, self.cal_dict).preprocess()
        # BatchSize x LengthSeq x SizeInput  Length of sequence is 368 while size of input is 32 (16 * 2)
        self.s = s

        self.data_specs['size'], self.data_specs['seq_len'], self.data_specs['ip_size'] = \
            s.shape[0], s.shape[1], s.shape[2]
        # _, func, S, n_ch = data.s.shape  # n_ch = num_features, ip_size = num_freq
        if len(s.shape) == 2:
            self.data_specs['n_ch'] = s.shape[2]
        else:
            self.data_specs['n_ch'] = None

    def dmd_processing(self, measurement_dict):
        import pydmd
        self.hp.shape = None
        self.hp.dtype = stb.Dtype.comp
        self.hp.s_select = stb.SSelect.all
        s_raw = stb.Preprocessing(measurement_dict, None).preprocess()
        s = np.zeros((s_raw.shape[0], s_raw.shape[1], 16), dtype='complex128')
        for i, an_s in enumerate(s_raw):
            dmd = pydmd.DMD(svd_rank=16)
            dmd.fit(an_s)
            s[i] = dmd.dynamics.T
        s = stb.Norm.complex_to_real(s, dtype=stb.Dtype.rectangular)
        return s

    def read_sim(self):
        class SIM:
            def __init__(self, hps):
                self.cal_dict = stb.S(path=st.g / 'cals/Results/cals/base_model_with_CalPhantom1.s16p')
                self.s_raw = np.load(st.g / 'datasets/s_image_pairs/s_phong.npy')
                self.ims = np.load(st.g / 'datasets/s_image_pairs/images_phong.npy')
                data_dict = {'data': self.s_raw, 'freq': self.cal_dict['freq']}
                self.s = stb.Preprocessing(hps, data_dict, self.cal_dict if hps.cal else None).preprocess()

        self.sim = SIM(self.hp)

    def check_area(self):
        """Finds area of shapes used in the dataset, look for abnomalities.

        :return:
        """
        from alexlib.miscellaneous import polygon_area
        result = np.array([polygon_area(aboundary) for aboundary in self.xy_labels])
        fig = plt.figure('Area Comparison')
        ax = fig.subplots()
        ax.plot(result, linewidth=5)
        ax.set_ylabel('Area of object in squared millimeters')
        ax.set_xlabel('Experiment index')
        ymin, ymax = ax.get_ylim()

        acc = 0
        flag = True
        index = 0
        while flag:
            tmp = self.experiment.count(index)
            if tmp == 0:
                flag = False
            else:
                acc += tmp
                plt.vlines(acc, ymin, ymax)
                plt.text(x=acc, y=ymax - np.random.random() * ymax * 0.02,
                         s=self.threeC_experiments[index], color='red')
                index += 1
        return result


class AnthonyBoundaryMeasurementReader:
    # Lab Measurements Readers
    """ Caveats on the dataset. README file provided by Anothny explains a lot. Besides those ones, those are some
    other key points: * In case of: 3C array, epxeriments of Cal and Phantoms: the CNC machine was obstructed from
    going to the intended direction The reason is the membrane and the coupling medium. It never happened in the case
    of other objects because they are relatively small The result of this is: the labels (positions) reported are the
    intended to go to ones, not the real ones that the CNC managed to achieve Anthony will try next time to report
    whatever positions the CNC ended up in, instead of what order was sent to it. You can check what cases suffered
    from this scenario by seeing the photos, there is a flashing blue light telling that CNC was obstructed. Notice
    that absence of light doesn't mean the scenario did not happen, it is just that camera shot did not capture it. *

    In 3B campaing everything is okay.
    """

    def __init__(self):
        data_path = st.d
        self.result_path = tb.P(data_path / r'Boundary/AnthonyBoundaryMeasurements')
        self.root = st.g / r'emvdata/MeasurementData/20200309-AS-CNC-Boundary-Movement'
        self.threeC_dict = self.root / r"EMV-3C"
        self.logp_dict = self.root / r"LogPeriodic"
        self.threeB_dict = st.g / r'emvdata/MeasurementData/0200501-AS-CNC-Boundary-Movement/EMV-3B'
        self.ssb_3c = st.g / r"emvdata/MeasurementData/20200804-AS-MM-M9800A-CNC/EMV-SSB"
        self.data_logp = self.data_3c = self.data_3b = self.data_ssb_3c = None

    def read_data(self, folders, x, y, save_folder, antenna_positions):
        data_dict = {}
        from tqdm import tqdm
        for a_folder in tqdm(folders):
            try:
                data_path_ = a_folder.joinpath(x)
                ground_truth_path = a_folder.joinpath(y)
                ground_truth_files = list(ground_truth_path.glob('*.csv'))
                pos = np.zeros((len(ground_truth_files), 720, 2))
                for idx, a_file in enumerate(ground_truth_files):
                    pos[idx] = tb.pd.read_csv(a_file, header=None)
                ss, freq, names = stb.S.get_dataset_from_paths(data_path_)
                folder_dict = {'s': ss, 'freq': freq, 'names': names, 'positions': pos,
                               'antenna_positions': antenna_positions}
                np.save(self.result_path.joinpath(save_folder, a_folder.stem + '.npy'), folder_dict)
                data_dict[a_folder] = folder_dict
            except FileNotFoundError:
                print(f'"{a_folder}" is not a valid folder, moving on to the next one ... ')
        return data_dict

    def read_logp(self, ):
        folders = list(self.logp_dict.glob('*'))
        a_p = tb.pd.read_csv(next(self.logp_dict.glob('*.csv')))
        self.data_logp = self.read_data([self.logp_dict[i] for i in folders], x='Data', y='GroundTruth',
                                        save_folder='logp_array',
                                        antenna_positions=a_p)
        # Folder is called Data instead of Calibrated Data because LogPeriodic array has no calibration PCBs.
        # once you do a set of measurements, they are never repeatable.

    def read_3c(self):
        folders = list(self.threeC_dict.glob('*'))
        a_p = tb.pd.read_csv(next(self.threeC_dict.glob('*.csv')))
        self.data_3c = self.read_data(folders, x='CalibratedData', y='GroundTruth', save_folder='threeC_array',
                                      antenna_positions=a_p)

    def read_3b(self):
        folders = tb.P(self.threeB_dict).search('*', list_=True, files=False)
        csv = r'emvdata/MeasurementData/20200501-AS-CNC-Boundary-Movement/EMV-3B/AntennaPositions.csv'
        a_p = tb.pd.read_csv(st.g / csv)
        self.data_3b = self.read_data(folders, x='CalibratedData', y='GroundTruth', save_folder='threeB_array',
                                      antenna_positions=a_p)

    def read_skin_skull_bucket(self):
        cal = ['*-cal.s16p', '*-calp.s16p'][0]
        folders = tb.List(["Empty", "DeepTarget", "ShallowTarget"]).apply(lambda x: self.ssb_3c / x)

        def read_folder(path):
            s_paths = (path / "CalibratedData").search(cal)
            result = stb.S.get_dataset_from_paths(s_paths)
            gt_paths = (path / "GroundTruth").search("*.csv")
            result.positions = gt_paths.apply(lambda x: tb.pd.read_csv(x, header=None)).apply(lambda x: x.to_numpy()).np
            for s_path, gt_path in zip(s_paths, gt_paths):
                assert gt_path.stem in str(s_path), "Mismatch in parsing S params and ground truths"
            ap = tb.pd.read_csv(st.g /
                                r"emvdata/MeasurementData/20200804-AS-MM-M9800A-CNC/EMV-SSB/AntennaPositions.csv")
            result.antenna_positions = ap
            return result

        combine = False
        if combine:
            dicts = [read_folder(item) for item in folders]
            s_ = np.concatenate([adict['s'] for adict in dicts], axis=0)
            pos_ = np.concatenate([adict['positions'] for adict in dicts], axis=0)
            names_ = np.concatenate([adict['s_names'] for adict in dicts])
            my_dict = {'freq': dicts[0]['freq'], 's': s_, 'positions': pos_, 'names': names_}
            np.save(self.result_path / "total.npy", my_dict)
        else:
            for afolder in folders:
                np.save(self.result_path / "SkinSkullBucket3C" / (afolder.stem + ".npy"), read_folder(afolder))


class BoundaryTFModels(dl.BaseModel, ABC):
    def __init__(self, *args):
        self.L = tf.keras.layers
        self.A = tf.keras.activations
        self.R = tf.keras.regularizers
        self.Op = tf.keras.optimizers
        self.Me = tf.keras.metrics
        self.Lo = tf.keras.losses
        self.M = tf.keras.models
        self.plotter = None
        super().__init__(*args)

    def viz(self, *boundary_descriptors, new_fig=False,
            show_template=False, plot_diff=False, plot_normals=True, **kwargs):
        """ Plots boundary and saves results. Like all model methods, it only works on batches
        In other wods, this is just a wrapper around plot_boundary that works on single input.

        :param boundary_descriptors: list of predictions, and potentially a list of ground truths.
                                     so, it looks like this bd = [list1, list2] or [list1]
        :param new_fig:
        :param plot_diff:
        :param show_template:
        :param plot_normals:

        Rules for default args: in each function, you put what it needs, and in args and kwargs
        what can possibly be relayed to subfunctinon Resolve contradiction between kwargs contents and
        defaults required by main fuction calling subone if any. if k not in kwargs: update kwargs.
        """
        artist = stb.PlotBoundaryArtist(plot_diff=plot_diff, plot_normals=plot_normals,
                                        show_template=show_template, new_fig=new_fig, **kwargs)
        self.plotter = tb.VisibilityViewerAuto(data=boundary_descriptors, artist=artist)

    def save_results(self, prediction_batch, save_name=None):
        save_dir = self.hp.save_dir.joinpath(f'results')
        save_dir.mkdir(exist_ok=True)

        for item in prediction_batch:
            path_name = save_dir.joinpath(f"xy_boundary_{save_name}_{tb.get_time_stamp()}_.mat")
            from scipy.io import savemat
            savemat(path_name, {'boundary_xy': item.xy, 'normals': item.distances})
            plt.pause(0.01)

    @tb.batcher('method')
    def postprocess(self, inferred, interp=stb.Interp.spline, num_pts=360,
                    smoothness_sampler=1, **kwargs):
        r = inferred * self.data.std + self.data.mean
        xy = stb.get_xy_from_radii(r)
        xy = stb.interpolate(xy, interp=interp, num_pts=num_pts,
                             smoothness_sampler=smoothness_sampler)
        distances, landing_pts = stb.get_normal_distances(xy, self.data.antenna_positions)
        obj = stb.BoundaryDescriptor(xy=xy, distances=distances, landing_pts=landing_pts, legend="NN solution",
                                     **kwargs)
        obj.r = r
        return obj


class LSTM(BoundaryTFModels):
    def __init__(self, hp__, data__, name='lstm_boundary'):
        super().__init__()
        self.data = data__
        self.hp = hp__
        s = self

        def create_lstm_model(name_):
            model_ = s.M.Sequential(name=name_)
            model_.add(tf.keras.Input(shape=(self.data.seq_len, self.data.ip_size), name='s')),
            rnn = s.L.GRU(units=self.hp.hidden_size, return_sequences=True if self.hp.rnn_layers > 1 else False)
            if self.hp.bi:
                rnn = s.L.Bidirectional(rnn, merge_mode=self.hp.merge_mode)
            model_.add(rnn),
            for ii in range(self.hp.rnn_layers - 1):
                model_.add(tf.keras.layers.GRU(units=self.hp.hidden_size))
            # model_.add(s.L.Dense(units=self.hp.num_points, activation=s.A.tanh,
            #                      kernel_regularizer=s.R.l1(l=self.hp.l1)))
            model_.add(s.L.Dense(units=self.hp.num_points, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
            return model_

        self.model = create_lstm_model(name)
        self.compile(optimizer=s.Op.Adam(lr=self.hp.lr), loss=s.Lo.MeanSquaredError(),
                     metrics=[s.Me.RootMeanSquaredError(), s.Me.MeanAbsoluteError(),
                              dl.get_mean_max_error(tf)()])

    def plot_loss(self):
        history = tb.Struct.concat_dicts_(*self.history)  # unpack the list of dicts.
        plt.figure(num='Training results')
        for key in ['loss', 'val_loss']:
            name = self.compiler.loss.name
            plt.plot(np.array(history[key]) * self.data.std, label=key + '_' + name)
        plt.legend(), plt.grid('on')
        plt.ylabel(f'Square Error in mms * {self.data.std}')
        plt.xlabel('Training epochs')

        plt.figure(num='Training results metrics')
        for a_metric in self.compiler.metrics:
            if a_metric is not None:
                name = a_metric.name
                plt.plot(np.array(history[name]) * self.data.std, label=name)
                plt.plot(np.array(history['val_' + name]) * self.data.std, label=key + '_' + name)
        plt.legend(), plt.grid('on'), plt.ylabel('Error in mms'), plt.xlabel('Training epochs')

    def fit_adv(self):
        import neural_structured_learning as nsl
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=self.hp.adv_multiplier,
            adv_step_size=self.hp.adv_step_size,
            adv_grad_norm=self.hp.adv_grad_norm)
        adv_model = nsl.keras.AdversarialRegularization(
            base_model=self.model,
            label_keys=['boundary'],
            adv_config=adv_config)

        adv_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.hp.lr),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.mean_absolute_error, dl.get_mean_max_error(tf)])

        def create_ds(feature, label):
            ds = tf.data.Dataset.from_tensor_slices((feature, label))

            def convert_to_dictionaries(image, label_):
                return {'s': image, 'boundary': label_}

            # or: tf.data.Dataset.from_tensor_slices({'s': x_train, 'boundary': y_train})
            return ds.batch(self.hp.batch_size).map(convert_to_dictionaries)

        ds_train = create_ds(self.data.split.x_train, self.data.split.y_train)
        ds_test = create_ds(self.data.split.x_test, self.data.split.y_test)
        adv_model.fit(ds_train, validation_data=ds_test, epochs=self.hp.epochs,
                      verbose=1, shuffle=True, callbacks=[])

        self.model.save(self.hp.save_dir)


class ResidualRNN(BoundaryTFModels):
    """
    The intuition behind this design is to all the rnn to return a sequence, and as such, compute the loss for every
    sample of the output. The goal of this is to make all results reasonable, and hopefully it gets better towards
    end of the sequence. In particular, we want each instance of the sequenc to predict the residual of the previous
    output.
    Summary: did not work well, training failed.
    """

    class C:
        pass

    obj = C()
    obj.v1 = None
    obj.v2 = None

    def __init__(self, hyp_p, data_, *args):
        super().__init__(*args)
        self.hp = hyp_p
        self.data = data_

        def create_lstm_model_residual():
            model_ = tf.keras.models.Sequential()
            model_.add(tf.keras.Input(shape=(self.data.seq_len, self.data.ip_size), name='s')),
            rnn = tf.keras.layers.GRU(units=self.hp.hidden_size, return_sequences=True)
            model_.add(rnn)
            model_.add(tf.keras.layers.GRU(units=self.hp.num_points, return_sequences=True, activation=None))
            model_.compile(optimizer=tf.keras.optimizers.Adam(lr=self.hp.lr),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[tf.keras.metrics.mean_absolute_error, dl.get_mean_max_error(tf)()])
            return model_

        self.model = create_lstm_model_residual()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.data.split.x_train, self.data.split.y_train)). \
            shuffle(1000).batch(self.hp.batch_size, drop_remainder=True)
        self.test_ds = tf.data.Dataset.from_tensor_slices((self.data.split.x_test, self.data.split.y_test)). \
            shuffle(100).batch(self.hp.batch_size, drop_remainder=True)

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            pred = self.model(inputs, training=True)
            if self.obj.v1 is None:
                self.obj.v1 = tf.Variable(tf.zeros_like(pred))  # temporary storage that is mutable and assignable
                self.obj.v2 = tf.Variable(tf.zeros_like(pred))  # temporary storage that is mutable and assignable

            # preparing the labels on the fly
            self.obj.v1.assign(tf.roll(pred, shift=1, axis=1))
            self.obj.v1[:, 0:1, :].assign(tf.zeros_like(pred[:, 0:1, :]))
            self.obj.v2.assign(self.obj.v1)
            # for _ in range(368-1):
            #     self.obj.v1.assign(tf.roll(self.obj.v1, shift=1, axis=1))
            #     self.obj.v1[:, 0, :].assign(tf.zeros_like(pred[:, 0, :]))
            #     self.obj.v2.assign_add(self.obj.v1)
            # [y-0 y-pf0 y-pf0-pf1 y-pf0-pf1-pf2 ...]
            # [y-0 y-pf0 y-pf1 y-pf2 y-pf3]
            target = tf.expand_dims(labels, axis=1) - self.obj.v2
            loss = self.loss_object(target, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def test_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        t_loss = self.loss_object(labels, 2 * tf.reduce_mean(predictions, axis=1))
        self.test_loss(t_loss)

    def fit(self, **kwargs):
        for epoch in range(self.hp.epochs):
            self.train_loss.reset_states()
            for batch_idx, (xx, yy) in enumerate(self.train_ds):
                self.train_step(xx, yy)
                if batch_idx % 10 == 0:
                    print(f'Train Loss = {self.train_loss.result()}', end='\r')

            self.test_loss.reset_states()
            for xx, yy in self.test_ds:
                self.test_step(xx, yy)
            print(f'Epoch:{epoch:2}/{self.hp.epochs:2}, TrainLoss {self.train_loss.result():1.3f}. TestLoss ='
                  f' {self.test_loss.result():1.3f}')

    def predict(self, xx, **kwargs):
        y = self.model.predict(xx)  # batch x freq x num_pionts
        return y[:, 0::2].sum(axis=1) + y[:, 1::2].mean(axis=1)


if __name__ == '__main__':
    hp = HParams()  # HPs come first
    d = AnthDataReader(hp)  # next is data
    m = LSTM(hp, d)  # lastly comes the model
    # pa = measurements.PA()
