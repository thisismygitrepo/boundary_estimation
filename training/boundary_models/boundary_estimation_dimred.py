"""
Uses PCA on signal to reduce its dimensionality.

Training using SkinSkullBucket data, and validating on PA data.
Problems:

* Data do not validate
* PCA needs to be domain adapted.

Caveat:
ascertain that the same cal type (cal/calp) is used in Bucket measurements and PA measurements.
Try using calibration.

Advantages:
* both source data and target data are measurements.
* Additional advantage for coming from the same domain, is that both have similar frequecies, no need to subsample.
    For Ali's boundary technique which is based on frequency shift. It was determiend that frequency resolution is a
    critical factor.
* the only differences are
    ** PA data come from patients while source data come from buckets.
    ** different array is used (3C in lab vs 3B in hospital)

Ideas:
* one can quantify uncertainty type when moving from measurements to reality. This can also be verified via
reconsruction quality of dimenaionality reduction used in the preprocessing step. The PCA was trained on measurements.
Thus, a test on real data could reveal how different they are.

"""


import alexlib.toolbox as tb
import resources.s_params as stb
import alexlib.deeplearning as dl
import training.boundary_models.boundary_prediction_lstm as data_base
import training.boundary_models.boundary_prediction_normal_estimation as model_base
import measurements as ms
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


class HParams(dl.HyperParam):
    def __init__(self, opt=None):
        _ = opt
        # =================== Enviroment ========================
        super().__init__()
        self.root = 'tmp/reduced_freq'
        self.exp_name = 'pca_dim_red'
        self.pkg_name = tf.__name__
        self.device_name = dl.Device.gpu1
        self.array = ['threeC', 'threeB', 'threeB+threeC', 'logp', 'SkinSkullBucket3C'][-1]
        self.version = "2.5"
        # ===================== DATA ============================
        self.diags = 1
        self.s_select = stb.SSelect.get_diagonals_indices(16, self.diags)
        self.shape = stb.Shape.eye
        # np.pad(x, [(0, 0), (0, 0), (1, 1), (0, 0)], mode='wrap')
        self.cal = [False, True][0]
        self.dtype = stb.Dtype.magnitude
        self.freq_select = stb.skrf.frequency.Frequency(start=700, stop=1602, unit='MHz', npoints=452)
        # np.arange(0.7, 1.602, 0.002)
        # no need to subsample as both source and target has similar frequencies.
        self.norm_factor = 1.0
        self.sort_freq = [False, True][0]
        self.dmd = [False, True][0]
        self.noise = None  # noise added to reduced input data
        self.normalize_pca = False
        # =================== Model =============================
        self.d = 10
        if opt is None:
            self.l1 = 0.01  # L1 regularization
        else:
            self.l1 = opt
        self.symmetry = list(range(16))  # )) + list(range(8))
        # ===================== Training ========================
        self.split = 0.2  # to be used by split method of DataReader class
        self.shuffle = True  # to be passed to `fit` method of Model class.
        self.seed = 234  # to be passed to split method.
        self.lr = 0.001  # Learning rate
        self.batch_size = 32
        self.epochs = 100
        self.save_code()


class DimRedDataReader(data_base.AnthDataReader):
    def __init__(self, *args, append_pa=True, load=False, **kwargs):
        super().__init__(*args, process_s=False, process_labels=True, split_data=False, load=load, **kwargs)
        target = None
        if load:
            if append_pa:  # append data from PA to fit data for PCA.
                self.pa = ms.PA()
                self.pos = tb.List(self.pa.positions[:4] + self.pa.positions[19:62])  # avoiding bad measurements where
                # antennas where faulty. as of the time of writing, this goes up to case 30
                # (equivalent to total of roughly 62 positions)
                target = self.pos.read().data_m.s.np
                self.s_raw = np.concatenate([self.s_raw, target], axis=0)

            # Process the input data, and get data for PCA. process_s() consumes .s_raw ans spits out .s
            self.process_s()  # shape = N x Freq x 16 (Magnitude only) or possible x 2
            s_per_ant = tb.Manipulator.merge_axes(self.s, 0, 2)  # shape = N * 16 x Freq (x 2)
            s_per_ant = s_per_ant[5000:]  # this allows resonable proportions of PA data and SSB data to be mixed.
            self.s_per_ant = s_per_ant
            if append_pa:
                self.s = self.s[:-len(target)]  # exclude PA data from training.

            # Dim Red of processed S
            from sklearn.decomposition import PCA
            if self.hp.dtype == stb.Dtype.polar:
                self.pca1 = PCA(n_components=self.hp.d)
                self.pca2 = PCA(n_components=self.hp.d)
                mag = s_per_ant[..., 0]
                phase = np.unwrap(s_per_ant[..., -1] * 3, axis=-1)  # work on frequency axis
                self.pca1.fit(mag)
                self.pca2.fit(phase)
                self.data_specs['pca1'] = self.pca1
                self.data_specs['pca2'] = self.pca2
            else:
                self.pca = PCA(n_components=self.hp.d)
                self.pca.fit(s_per_ant)
                self.data_specs['pca'] = self.pca

            self.split_my_data_now()

    def split_my_data_now(self, seed=None):
        if seed is not None:
            self.hp.seed = seed
        self.data_split(self.s, self.labels, self.names, np.arange(0, len(self.names)),
                        strings=['x', 'y', 'names', 'index'])

    def convert_labels_to_ali_classes(self):
        # because all labels fall into one category, there's no instances of other categories, hence, cannot train.
        thresh = (np.array([18, 30]) - self.mean) / self.std
        tmp = np.ones_like(self.labels).astype('uint8')
        case, ant = self.labels.shape
        for i in range(case):
            for j in range(ant):
                dist = self.labels[i, j]
                if dist < thresh[0]:
                    clas = 0
                elif dist < thresh[1]:
                    clas = 1
                else:
                    clas = 2
                tmp[i, j] = clas
        self.labels = tmp
        self.split_my_data_now()

    def test_pca_visually(self, ex=None):
        if ex is None:
            ex = self.s[0, :, :1].T
        self.plotter = stb.PCAArtist.evaluate_pca(pca=self.pca, signals=ex)

    def get_performance_stats(self, data=None):
        if data is None:
            data = tb.Manipulator.merge_axes(self.data.s, 0, 2)
        reduced = self.pca.transform(data)
        recons = self.pca.inverse_transform(reduced)
        error = np.linalg.norm(recons - data, axis=-1)
        print(f"Average reconstruction error = {error.mean()}")
        # pos.data_m.get_ii().T.__abs__()
        return error

    def get_pca_from_measurements(self):
        files = tb.P(ms.HHT3().path).search('*.s16p') + tb.P(ms.PA().path).search('*.s16p')
        dataset = stb.S.get_dataset_from_paths(files)
        s = stb.Preprocessing(self.hp, dataset).preprocess()
        s = s.transpose((0, 2, 1)).reshape(-1, len(self.hp.freq_select))  # N * 16 x Freq
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.hp.d)
        pca.fit(s)
        # np.save('tmp.npy', pca, allow_pickle=True)

    def preprocess(self, args):
        s_obj = args
        res = stb.Preprocessing(self.hp, s_obj).preprocess()
        return res


class HPAdjacent(HParams):
    def __init__(self):
        super(HPAdjacent, self).__init__()
        i, j = stb.SSelect.split_to_per_ant(1, 16)
        i = i[2::3]
        j = j[2::3]
        self.s_select = (i, j)


class DimRedDataReaderAdjacent(DimRedDataReader):
    def __init__(self, hyp):
        super(DimRedDataReaderAdjacent, self).__init__(hyp)
        # =============== Get the new 'antenna positions' and their normals, and build new labels from xy boundary =====
        tmp = stb.BoundaryDescriptor()
        points = stb.interpolate(tmp.antenna_positions, num_pts=1600)
        # tmp.plot()
        # plt.scatter(*points[50::100].T)  # ascretain that points are inbetween.
        antenna_positions = points[50::100]
        norm_angles = []
        for point, idx in zip(antenna_positions, range(50, len(points), 100)):
            diff_vec = points[idx - 1] - points[idx + 1]
            diff_vec = diff_vec / np.linalg.norm(diff_vec)
            normal = np.array([diff_vec[1], diff_vec[0] * -1])
            norm_angles.append(np.rad2deg(np.arctan2(normal[1:], normal[0:1])))
            # tmp.plotter.fig.axes[0].arrow(point[0], point[1], normal[0]*10, normal[1]*10)
        antenna_positions = np.concatenate([antenna_positions, np.array(norm_angles)], axis=1)
        self.antenna_positions = antenna_positions

        class BoundaryDescripter(stb.BoundaryDescriptor):
            @property
            def antenna_positions(self):
                return points[50::100]

        self.boundary_descriptor_class = BoundaryDescripter
        self.data_specs['antenna_positions'] = self.antenna_positions
        self.process_labels()  # again because the first was wrong.

    @staticmethod
    def merge_predictions(pos, m_sii, m_adj):
        m_sii.predict_from_position(pos, viz=False)
        pred1 = pos.predictions[-1].postprocessed
        m_adj.predict_from_position(pos, viz=False)
        pred2 = pos.predictions[-1].postprocessed
        dist2 = pos.predictions[-1].postprocessed.distances
        points = np.dstack([pred1.landing_pts, pred2.landing_pts]).transpose((0, 2, 1)).reshape(
            pred1.landing_pts.shape[0] * 2, pred1.landing_pts.shape[1])
        distances = np.random.random(len(pred1.distances) * 2)
        distances[::2] = pred1.distances
        distances[1::2] = dist2
        xy = stb.interpolate(points, interp=stb.Interp.spline, num_pts=360)
        xy = stb.allign_boundary(xy, num_pts=360)
        obj = pos.predictions[0].postprocessed.copy()
        obj.xy = xy
        obj.distances = distances
        obj.landing_pts = points
        obj.sii = pred1
        obj.sij = pred2
        obj.legend = "Combined"
        # path = tb.P(r"saved_models\boundary_models\reduced_freq").relativity_transform("deephead")
        # model = MLP.from_class_weights(path)  # model for boundary estimation
        # q = merge_predictions(d.pa.positions[-20], model, m)
        # # q.plot(q.sii, plot_normals=False)
        # # q.plotter.bdry_axis.plot(*q.sij.xy.T)
        # q.sii.plot(q.sij)
        return obj


class MLP(model_base.SingeModelCreator):
    """Inherit from that class to get for free:
    * infer
    * postprocess
    * fit
    * evaluate
    * viz
    """

    def __init__(self, hp__, data__, build=True, evaluate=False):
        super().__init__(hp__=hp__, data__=data__, build=False)

        if build:
            if self.hp.dtype == stb.Dtype.polar:
                self.model = self.create_mlp_2ch(name=self.hp.exp_name.__str__() + '_MLP2CH')
                self.build(shape=(self.hp.d, 2))
            else:
                self.model = self.create_mlp(name=self.hp.exp_name.__str__() + '_MLP')
                self.build((self.hp.d,))
            self.compile(optimizer=self.Op.Adam(lr=self.hp.lr),
                         loss=self.Lo.MeanSquaredError(),
                         metrics=[self.Me.mean_absolute_error])
            self.model.summary()
            if evaluate:
                self.evaluate()
        self.pa = self.tmp = self.plotter = None

    def create_mlp(self, name=None):
        s = self
        model_ = s.M.Sequential(name=name)
        model_.add(s.L.Dense(units=self.hp.d, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        model_.add(s.L.LeakyReLU(alpha=0.3))
        model_.add(s.L.Dense(units=10, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        model_.add(s.L.LeakyReLU(alpha=0.3))

        model_.add(s.L.Dense(units=10, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        model_.add(s.L.LeakyReLU(alpha=0.3))
        # model_.add(s.L.Dense(units=7, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        # model_.add(s.L.LeakyReLU(alpha=0.3))

        model_.add(s.L.Dense(units=4, activation=None, kernel_regularizer=s.R.l1(l=self.hp.l1)))
        model_.add(s.L.LeakyReLU(alpha=0.3))
        model_.add(s.L.Dense(units=1, kernel_regularizer=None, activation=None))
        return model_

    def create_mlp_2ch(self, name=None):
        s = self
        ip = tf.keras.Input(shape=(self.hp.d, 2))
        ch1, ch2 = s.L.Lambda(lambda x: (x[..., 0], x[..., 1]))(ip)

        op1 = s.L.Dense(units=self.hp.d, activation=None)(ch1)
        op1 = s.L.LeakyReLU(alpha=0.3)(op1)
        op1 = s.L.Dense(units=7, activation=None)(op1)
        op1 = s.L.LeakyReLU(alpha=0.3)(op1)
        op1 = s.L.Dense(units=4, activation=None)(op1)
        op1 = s.L.LeakyReLU(alpha=0.3)(op1)

        op2 = s.L.Dense(units=self.hp.d, activation=None)(ch2)
        op2 = s.L.LeakyReLU(alpha=0.3)(op2)
        op2 = s.L.Dense(units=7, activation=None)(op2)
        op2 = s.L.LeakyReLU(alpha=0.3)(op2)
        op2 = s.L.Dense(units=4, activation=None)(op2)
        op2 = s.L.LeakyReLU(alpha=0.3)(op2)

        op = tf.concat([op1, op2], axis=-1)
        op = s.L.Dense(units=4, activation=None)(op)
        op = s.L.LeakyReLU(alpha=0.3)(op)
        op = s.L.Dense(units=1)(op)
        model_ = s.M.Model(ip, op, name=name)
        return model_

    def reshape(self, s):
        """To be used by infer method, which is designed to process a whole set of *preprocessed* S params
        i.e. Takes in N x Freq x 16, then makes it:
        """
        op = tb.Manipulator.merge_axes(s, 0, 2)  # Merge Batch axis with Antenna axis in Decoupled predictions Models.
        # N*16 x seq_len x (2)  if dtype=polar
        if self.hp.dtype == stb.Dtype.polar:
            op1 = self.data.pca1.transform(op[..., 0])
            op2 = self.data.pca2.transform(op[..., 1])
            if self.hp.normalize_pca:
                op1 /= np.sqrt(self.data.pca1.explained_variance_)
                op2 /= np.sqrt(self.data.pca2.explained_variance_)
            op = np.dstack([op1, op2])  # concat along a new dim, put it in the end ==> equiv: stack(~, axis=-1)
        else:
            op = self.data.pca.transform(op)
            if self.hp.normalize_pca:
                op /= np.sqrt(self.data.pca.explained_variance_)
            if self.hp.noise is not None:
                op += np.random.randn(*op.shape) * self.hp.noise  # adding noise.
            # N*16 x d  # ready to be consumed by a NN.
        return op


class PerAntennaMLP(model_base.PerAntennaModelCreator):
    """Inheriting from MLP to get crate_mlp method for models.
    Inheriting from PerAntennaModelCreator to get `fit`, `fit_one_model` and `infer`
    """

    def __init__(self, hp__, data__):
        super().__init__(hp__, data__, model_class=MLP, data_class=DimRedDataReader)

    def infer(self, s_):
        op = np.zeros((s_.shape[0], self.data.ports), dtype='float32')
        for index, amodel in enumerate(self.models):
            op[:, index] = amodel.infer(s_[..., index][..., None]).squeeze()
        return op


def generate_ensemble():
    hp = HParams()
    d = DimRedDataReader(hp)
    ens = dl.Ensemble(HParams, DimRedDataReader, MLP, n=30)
    ens.fit(tf.keras.callbacks.LearningRateScheduler(lambda ep, lr: lr if ep < 200 else lr * 0.3))

    path_ = tb.tmp() / "one_hund_models"
    ens = dl.Ensemble.from_saved_weights(path_, MLP)
    pa = ms.PA().positions.filter(lambda x: x.em_boundary is not None)
    pos = pa[-11]

    def process_pos(ensemble, apos):
        result = ensemble.predict_from_position(apos)
        result.plot(apos.em_boundary)
        recons = d.get_performance_stats(apos.data_m.get_ii().__abs__().T)
        df = tb.pd.DataFrame.from_dict(dict(std=result.std.np, recons=recons))
        print(df)
    process_pos(ens, pos)

    m1 = MLP.from_class_weights(r"C:\Users\s4551072\OneDrive - The University of "
                                r"Queensland\Pycharm\deephead\tmp\boundaries\best_single_model\pca_dim_red")
    ens1 = dl.Ensemble.from_saved_weights(r"C:\Users\s4551072\OneDrive - The University of "
                                          r"Queensland\Pycharm\deephead\tmp\boundaries\one_hund_models", MLP)
    ens2 = dl.Ensemble.from_saved_weights(r"C:\Users\s4551072\OneDrive - The University of "
                                          r"Queensland\Pycharm\deephead\tmp\boundaries\one_hund_models_50epochs", MLP)
    return ens1, ens2, m1


def do_da():
    """The difference between doing TCA here and in S2Code is that here we have much less data. We only use the main
    diagonals. Additonally, our source is the SSBucket which only have 455 readings instead of Wiener Simulations
    which has 933. So it is more managable from computational perspective.
    :return:
    """
    kind = 'sa'

    hp = HParams()
    d = DimRedDataReader(hp)
    m = MLP(hp__=hp, data__=d)
    m.evaluate()

    # target domain
    pa = ms.PA()
    pos = tb.List(pa.positions[:4] + pa.positions[19:])  # avoiding bad measurements
    # TODO: at the time of training there was 62 positions form PA added to data.
    # excluding measurements with faulty antennas. Read S position, get s params of shape 22 x 751 x 16 x 16 complex
    target = pos.read().data_m.s.np
    target = m.preprocess({'data': target, 'freq': pa.positions[0].data_m.f / 1000})  # do preprocessing as in model
    # result is of shape 444 x 751 x 16
    target = target.transpose((0, 2, 1)).reshape(-1, d.seq_len)  # new shape is 22*16 x 751

    # source domain
    source = d.s.transpose((0, 2, 1)).reshape(-1, d.seq_len)  # new shape is 444*16 x 751

    # Domain adaptation
    import training.deephead_models.s_to_code as s2c
    if kind == 'tca':
        gammas = np.linspace(0.1, 10, 20)
        lambdas = np.linspace(0.1, 10, 20)
        # good results with gamma = 20, lambda = 0.1 ==> resonable std in outcome.
        results = []
        for i, gamma in enumerate(gammas):
            per_lambda = []
            for j, lamb in enumerate(lambdas):
                tca = s2c.TCA(kernel_type='rbf', dim=hp.d, lamb=lamb, gamma=gamma)
                new_source, new_taget = tca.fit(source[:1000], target)
                std, mean = new_source.__abs__().std(axis=0), new_source.__abs__().mean(axis=0)
                per_lambda.append((mean, std))
                print(f"gamma={gamma}, lambda={lamb}. mean: \n {mean}, \n\n STD: {std} \n\n", "=" * 100)
            results.append(per_lambda)
        # Training on new reduced dimesionality source, then testing on new domain adapted data.
        # m_ = MLP(hp, new_source)

    elif kind == 'sa':
        import training.boundary_models.suba as suba
        tca = suba.SubspaceAlignedClassifier(subspace_dim=hp.d)
        a, b, c = tca.subspace_alignment(source, target, hp.d)
        x_new, t_new = tca.align_data(source, target, b, c, a)
        x_mean = x_new.mean(axis=0)
        x_new = x_new - x_mean
        t_new = t_new - x_mean
        t_new = t_new.reshape(len(pos), 16, hp.d)
        pos.modify("x.da=y", t_new)
        m.model.fit(x=x_new, y=d.labels.reshape(-1), epochs=100)
        # choosing the ones for which Ali made a prediction
        # m.act_on_pa(pos.filter(lambda x: x.em_boundary is not None), da=True)


def use_adjacent_antenna_data():
    # attempt to build an auxiliary model that uses adjacent antenna data.
    hp1 = HParams()
    hp1.s_select = stb.SSelect.split_to_per_ant(1, 16)
    d1 = DimRedDataReader(hp1)
    hp1 = HParams()
    hp1.s_select = stb.SSelect.split_to_per_ant(1, 16)
    hp1.shape = stb.Shape.callable(lambda x: x.reshape(..., ()))
    result = stb.Preprocessing(hp1, d1.s_raw).preprocess()
    print(result.shape)


def main():
    hp = HParams()
    d = DimRedDataReader(hp)
    m = MLP(hp__=hp, data__=d)
    return hp, d, m


def main_adjacent():
    hp = HPAdjacent()
    d = DimRedDataReaderAdjacent(hp)
    m = MLP(hp, d)
    return m


class HPT(dl.HPTuning):
    def __init__(self):
        super(HPT, self).__init__()
        self.hp = HParams()
        self.data = DimRedDataReader(self.hp)
        self.dir = tb.tmp("tuning")
        self.reg = self.hpt.HParam('regu', self.hpt.Discrete([0.005, 0.02]))
        self.params = tb.List([self.reg])
        self.acc_metric = "lossy"
        self.metrics = [self.hpt.Metric('accc', display_name="Accuracyy")]

    def run(self, param_dict):
        hp = HParams(param_dict[self.reg])
        model = MLP(hp, self.data)
        model.fit()
        return model.history[-1]['loss'][-1]


if __name__ == '__main__':
    path = tb.tmp()
