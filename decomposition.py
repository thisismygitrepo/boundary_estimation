import pydmd
import tensorly as tl
import resources.toolbox as tb
import resources.s_params as stb
import resources.deeplearning as dl
import numpy as np
import measurements as m
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import tsfel
import pandas as pd
import matplotlib.pyplot as plt
tb.DisplayData.set_display()


"""
decompositions attempted are:

* Tensor Decomposition (Tensorly): Higher order extension of SVD.
* Dynamic Mode Decompsotion (PyDMD): Inuition is combinging PCA and FT to handle spatiotemporal data.
* NonLinear Fourier Transform (FNFTpy): Decompose signal into frequency changing waves.
* Scattering Wavelet Transform (Kymatio): Decompose signal into nonlinearly interacting waves.

Accordning to `Ali`, the following points are threshold distances sepraratig different trends of resonance dynamics
* [0 to 18]
* [18 to 30]
* [30 to 40]

Files required for this to run: Anthony's CNC Data and Ali's Bank of data. Both are on D drive.
"""


class AnthonyDataHandler:
    def __init__(self):
        import training.boundary_models.boundary_prediction_lstm as lstm

        class HParams(dl.HyperParam):
            def __init__(self):
                # =================== Enviroment ========================
                super().__init__()
                self.exp_name = 'trained_on_data_orig_nocal'  # 'gru_vs_rnn_vs_lstm'
                self.root = 'tmp/lstm'
                self.array = ['threeC', 'logp', 'threeB'][-1]
                self.seed = 224
                self.shuffle = True
                self.split = 0.2
                self.save_code()

        self.hp = HParams()
        self.data = lstm.AnthDataReader(self.hp, process_s=False, process_labels=False, split_data=False)
        self.plotter = None
        self.dmd = None

    def plot(self, idx, **kwargs):
        self.dmd = Decomposition(stb.S.from_numpy(self.data.s_raw[idx], self.data.freq))
        self.dmd.dynamic_mode_decomposition(**kwargs)
        label = self.data.xy_labels[idx]
        desc = stb.BoundaryDescriptor(xy=label)
        self.plotter = stb.BoundaryPlotter(desc, antenna_positions=self.dmd.antenna_positions)


class Decomposition:
    """Generates DMD plots"""
    def __init__(self, s_obj, **kwargs):
        self.f = s_obj.f
        self.s = s_obj.s
        self.s_prime = self.s
        self.name = "Boundary Simulation Bank " + s_obj.path.stem
        self.s_indices = np.arange(0, 256, 16)
        self.dmd = None
        self.peak_indices = self.peaks = self.features = self.flattest_mode = self.delta = None
        self.antenna_positions = np.load(tb.P(r'gen_dataset\antenna_positions.npy').relativity_transform())
        self.result = self.normal_distances = None
        self.fig = self.ax = self.txt = self.plotter = None
        self.svd_rank = 2
        self.tmp = None
        self.kwargs = kwargs

    @classmethod
    def from_position(cls, position, **kwargs):
        position.read()
        obj = cls(position.data_m, **kwargs)
        obj.postion = position
        return obj

    def run(self, **kwargs):
        self.dynamic_mode_decomposition(**kwargs)
        self.predict_distance()

    @staticmethod
    def cp(s):  # canonical paravac decomp
        return tl.partial_svd(s)

    def dynamic_mode_decomposition(self, svd_rank=2, plot=True, mode=None, peak_finder='blah',
                                   decomp=pydmd.DMD, **kwargs):
        self.svd_rank = svd_rank  # 2 gives decent reconstruction error.
        self.dmd = decomp(svd_rank=svd_rank, **kwargs)
        self.dmd.fit(self.s_prime)
        # DMD expects columns to be snapshots if you feed a 2D array. Otherwise, you can feed an iterable that contains
        # measurements of the system at each instance and everything will be handled internally including vectorizing.
        # dmd.plot_eigs(show_axes=True, show_unit_circle=True)
        self.peaks = []
        self.peak_indices = []
        for i in range(svd_rank):
            if peak_finder == 'index':
                self.peaks.append(self.dmd.modes.__abs__()[:, i][self.s_indices])
                self.peak_indices.append(self.s_indices)
            else:
                signal = np.append([0, 0, 0], np.abs(self.dmd.modes[:, i]))
                signal = np.append(signal, [0, 0, 0])
                peaks = find_peaks(signal, height=0.02, distance=10)
                self.peaks.append(peaks[1]['peak_heights'])
                self.peak_indices.append(peaks[0] - 3)
        error = np.abs(self.s_prime.reshape(self.s_prime.shape[0], -1) - self.dmd.reconstructed_data.T).mean()
        print(f"Recons Error = {error}")

        delta = []
        for amode in range(self.svd_rank):
            delta.append(self.dmd.dynamics[amode, 0].__abs__() - self.dmd.dynamics[amode, -1].__abs__())
        self.flattest_mode = np.array(delta).__abs__().argmin()
        self.delta = delta[self.flattest_mode]
        self.tmp = self.dmd.modes[:, self.flattest_mode].__abs__()
        tmp2 = np.zeros_like(self.tmp[:8])
        self.features = np.concatenate([tmp2, self.tmp, tmp2])
        if plot:
            self.plot(mode)

    def plot(self, mode=None, figure=None, suffix='_'):
        if mode is None:
            mode = self.flattest_mode
        if figure is None:
            self.fig = plt.figure(num='Decomposition Plot' + suffix)
        else:
            self.fig = figure
        self.ax = self.fig.subplots(1, 2)
        if self.svd_rank > 4:
            tmp = mode
        else:
            tmp = Ellipsis
        self.ax[0].plot(np.abs(self.dmd.modes[:, tmp]))
        self.ax[0].set_title('Modes')
        self.ax[0].legend(np.arange(self.svd_rank))
        self.ax[0].scatter(self.peak_indices[mode], self.peaks[mode], color='red')
        self.ax[0].set_ylim([0, 0.5])
        self.ax[0].grid(b=True, which='minor')
        self.ax[0].grid(b=True, which='major')
        self.ax[1].plot(np.abs(self.dmd.dynamics[tmp, :].T))
        self.ax[1].set_title('Modes evolution')
        self.ax[1].legend(np.arange(self.svd_rank))
        self.ax[1].grid(b=True, which='minor')
        self.ax[1].grid(b=True, which='major')
        self.ax[1].set_ylim([0, 2.5])
        self.ax[1].set_xlabel(f"Delta in mode {mode} = {round(self.delta, 3)}")
        self.txt = [self.fig.text(x=0.5, y=0.98, s=self.name, horizontalalignment='center')]

    @staticmethod
    def estimate_period(signal):
        pt = signal[0]
        period = np.argmin(np.abs(signal - pt))
        return period

    def plot_peaks(self):
        nrows, ncols = tb.ImShow.get_nrows_ncols(self.svd_rank)
        fig_dyn_abs, ax_dyn_abs = plt.subplots(nrows, ncols)
        fig_dyn_abs.subplots_adjust(hspace=0.4)
        ax_dyn_abs = ax_dyn_abs.ravel()
        self.fig, self.ax = plt.subplots(nrows, ncols)
        self.fig.subplots_adjust(hspace=0.4)
        self.ax = self.ax.ravel()
        for i in range(self.svd_rank):
            self.ax[i].stem(np.abs(self.dmd.modes)[:, i], use_line_collection=True)
            self.ax[i].set_title(f"rank={i}")
            ax_dyn_abs[i].plot(np.abs(self.dmd.dynamics[i, :]))
            ax_dyn_abs[i].plot(np.angle(self.dmd.dynamics[i, :]) / 3)

    def predict_distance(self, alpha=2, beta=0.8):
        num = alpha * np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self.normal_distances = num / self.peaks[1] ** beta

    def plot_from_normals(self):
        xy = stb.get_xy_from_normals(self.normal_distances, self.antenna_positions)
        xy = stb.interpolate(xy)
        self.result = stb.BoundaryDescriptor(xy=xy)
        self.plotter = stb.BoundaryPlotter(self.result, antenna_positions=self.antenna_positions, name=self.name)

    def close(self):
        plt.close(self.plotter.fig)
        plt.close(self.fig)


class AliDatasetHandler:
    def __init__(self):
        self.ali_path = tb.P(r"D:\RESULTS\Boundary\boundary_esimation_ali_5tissuemodel")
        tmp = self.ali_path.myglob("*S*.s*", win_order=True)
        self.s = tmp[:-1].apply(stb.S)
        assert self.s.path.get_num().list == list(range(1, 41)), "Files did not load in the correct order"
        self.calibrate = False
        self.peaks_ds = None
        self.modes_ds = None
        self.cal_path = tmp[-1]
        self.cal = stb.S(self.cal_path)
        self.tmp = None
        self.obj = None
        self.real = None
        self.mode = None
        self.plotter = None
        self.per_ant_ds = self.labels = self.classifier = None

    def generate_ali_ds(self, mode=None, save_dir=None, **kwargs):
        if save_dir is None:
            save_dir = tb.P.tmp()
        self.mode = mode
        self.plotter = tb.VisibilityViewer()
        self.peaks_ds = []
        self.modes_ds = []
        saver = tb.SaveType.GIFFileBased(watch_figs=["Decomposition Plot (Ali)"], delay=10, save_dir=save_dir,
                                         save_name="dmd_coeff_animation_bank_data")
        for s_obj in self.s:
            artist = Decomposition(s_obj)
            artist.dynamic_mode_decomposition(peak_finder='blah', plot=False, **kwargs)
            artist.plot(mode=mode, suffix=' (Ali)')
            self.peaks_ds.append(artist.peaks[artist.flattest_mode])
            self.modes_ds.append(artist.features)
            saver.add()
            plt.pause(0.1)
            self.plotter.add(artist=artist)
        saver.finish()
        self.plotter.finish()
        self.peaks_ds = np.array(self.peaks_ds)
        self.modes_ds = np.array(self.modes_ds)
        tmp = Decomposition(stb.S(self.cal_path))
        tmp.dynamic_mode_decomposition(peak_finder='blah', plot=False, **kwargs)
        self.cal = tmp.features
        my_dict = {'cal': self.cal, 'ds': self.modes_ds}
        np.save(self.ali_path / 'decompostion_ds.npy', my_dict)

    def load_ali_ds(self):
        my_dict = (self.ali_path / 'decompostion_ds.npy').readit()
        self.modes_ds = my_dict['ds']
        self.cal = my_dict['cal']

    def vizualize_ali_ds(self):
        assert self.peaks_ds is not None, f"Build or load the dataset first"
        figure, axis = plt.subplots()
        figure.canvas.manager.full_screen_toggle()
        saver = tb.SaveType.GIFFileBased(watch_figs=[figure], delay=10, save_dir=tb.P.tmp())
        for idx, item in enumerate(self.peaks_ds):
            axis.stem(item, use_line_collection=True)
            axis.set_ylim(0, 0.3)
            axis.set_title(self.s[idx].path.stem)
            axis.grid()
            saver.add()
            axis.cla()
        saver.finish()

    def estimate_distance(self):
        distance = []
        # indices = self.obj.peak_indices[self.mode] + 8
        indices = np.arange(0, 256, 16)
        for index, item in enumerate(indices):
            feature = self.obj.features[item + index: item + index + 16]
            diff = feature - self.modes_ds[:, item + index: item + 16 + index]
            min_index = np.argmin(np.linalg.norm(diff, axis=-1)) + 1
            distance.append(min_index)
        self.obj.normal_distances = np.array(distance)
        self.obj.plot_from_normals()

    @staticmethod
    @tb.batcher()
    def get_modes_per_antenna(modes):
        output = []
        indices = np.arange(0, 256, 16)
        for index, item in enumerate(indices):
            output.append(modes[item + index: item + index + 16])
        return np.array(output)

    def build_classifier(self):
        ds = self.get_modes_per_antenna(self.modes_ds)
        labels = np.arange(1, 41).repeat(16)
        ds = ds.reshape(-1, ds.shape[-1])
        self.per_ant_ds = ds
        self.labels = labels

        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(self.per_ant_ds, np.where(self.labels <= 18, 0, 1))
        self.classifier = classifier

    def classify(self, features):
        return self.classifier.predict(self.get_modes_per_antenna(features))

    def estimate_boundary_from_position(self, pos):
        self.obj = Decomposition.from_position(pos)
        self.obj.dynamic_mode_decomposition()
        self.estimate_distance()


class PADataHandler:
    def __init__(self, **kwargs):
        self.pa = m.PA()
        self.ds = []
        self.plotter = tb.VisibilityViewer()
        for ii, case in enumerate(self.pa.cases):
            tmp = Decomposition.from_position(case.positions[0])
            tmp.dynamic_mode_decomposition(plot=False, **kwargs)
            self.ds.append(tmp.features)
            tmp.plot(suffix=' (PA)')
            self.plotter.add(artist=tmp)
        self.plotter.finish()
        self.ds = np.array(self.ds)
        ds = AliDatasetHandler.get_modes_per_antenna(self.ds)
        self.per_ant_ds = ds.reshape(-1, ds.shape[-1])


class Mode:
    def __init__(self, s_list_obj):
        obj = s_list_obj

        class ModePlotter(tb.Artist):
            def accessorize(self, *args, title=None, legends=None, **kwargs):
                self.ax[0].stem(*args, use_line_collection=True)
                self.ax[0].set_ylim([0, 1.2])
                self.suptitle(title)

        results = []
        for s in obj.list:
            tmp = pydmd.DMD(svd_rank=1)
            tmp.fit(s.get_entries_per_antenna(ant_idx=2, num_diag=1, exclude_main=True).T)
            results.append(tmp.modes.__abs__().squeeze())

        self.viz = tb.VisibilityViewerAuto(data=[results], titles=obj.name.list,
                                           artist=ModePlotter(create_new_axes=True),
                                           legends=[[f"S {i, i - 1}", f"S {i, i + 1}"] for i in range(16)] * len(obj))
        self.obj = obj
        results = np.array(results)
        self.results = results
        x, y = results[:, 0], results[:, 1]
        fig, ax = tb.plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(obj.nw.name.list):
            ax.annotate(txt, (x[i], y[i]))


class FeatureExtractor:
    def __init__(self, s, start_freq=600, end_freq=1800, **kwargs):
        if type(s) is tb.List:
            self.s = s
        else:
            self.s = tb.List([s])
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.s.get_common_freq(end_freq=end_freq, start_freq=start_freq, overwrite=True)
        self.data = self.s.get_all_entries_per_antenna(num_diag=2, exclude_main=False). \
            np.reshape(-1, self.s[0].f.__len__(), 3)
        self.repo = self.titles = self.desc = self.cfg = None
        self.my_dict = None
        self.extract_features(**kwargs)

    def extract_features(self, ant_st=0, ant_fi=16, verbose=False):
        cfg = tsfel.get_features_by_domain()
        desc = {key: list(val.keys()) for key, val in cfg.items()}
        if verbose:
            for main_key, main_values in desc.items():
                print(main_key.center(50, '='))
                for i, item in enumerate(main_values):
                    print(f"{i:2}- {item:30s} {cfg[main_key][item]['description']:100s}"
                          f" {cfg[main_key][item]['parameters']}")
        cfg['statistical']['ECDF']['parameters']['d'] = 1
        cfg['statistical']['Histogram']['parameters']['nbins'] = 1
        cfg['spectral']['FFT mean coefficient']['parameters']['nfreq'] = 2  # this is the minimum. 1 Causes error.
        cfg['spectral']['LPCC']['parameters']['n_coeff'] = 1
        cfg['spectral']['MFCC']['parameters']['num_ceps'] = 1
        cfg['spectral']['Wavelet absolute mean']['parameters']['widths'] = 'np.arange(1, 2)'
        cfg['spectral']['Wavelet energy']['parameters']['widths'] = 'np.arange(1, 2)'
        cfg['spectral']['Wavelet variance']['parameters']['widths'] = 'np.arange(1, 2)'
        cfg['spectral']['Wavelet standard deviation']['parameters']['widths'] = 'np.arange(1, 2)'
        # res = tsfel.time_series_features_extractor(cfg, a_d[:10, :, 0])
        # tb.List(res.columns).print()
        # plane to generate 12 plots. 3 x 4. Showing features applies to single antenna
        # Each row concern Si,i, Si,i+1, Si, i-1. Cols: features applied on real, imag, abs, ang
        repo = []
        real = None
        for antenna in range(ant_st, ant_fi):
            data = self.data[antenna::16, :, :]  # select same antenna, from all the data set (possibly one)
            # all frequencies, sii plus its adjacent elements.
            results = []
            for signal in data.transpose(-1, 0, 1):  # put signal axis first so that it is iterated over.
                real = tsfel.time_series_features_extractor(cfg, signal.real)
                imag = tsfel.time_series_features_extractor(cfg, signal.imag)
                magitude = tsfel.time_series_features_extractor(cfg, np.abs(signal))
                angle = tsfel.time_series_features_extractor(cfg, list(np.unwrap(np.angle(signal))))
                results.append([real.to_numpy(), imag.to_numpy(), magitude.to_numpy(), angle.to_numpy()])
            results = np.array(results)
            results = results.transpose((-1, 0, 1, 2))  # put the animation axis first. The AutoVis requires that.
            repo.append(results)

        self.repo = np.array(repo)
        self.titles = [i[2:] for i in list(real.columns)]  # get rid of 0_
        self.desc = desc
        self.cfg = cfg
        self.my_dict = {'repo': self.repo, 'cfg': self.cfg,
                        'titles': self.titles, 'description': self.desc}

    def save_my_dict(self):
        np.save(rf"D:\RESULTS\Boundary\boundary_esimation_ali_5tissuemodel\feature_bank_ali_start_freq_"
                rf"{self.start_freq}_end_freq_{self.end_freq}.npy", self.my_dict)

    @staticmethod
    def generate_and_save_bank(ali_dataset, start_freq=600, end_freq=1800):
        """
        :return:
        """
        temp1 = FeatureExtractor(ali_dataset.s, start_freq=start_freq, end_freq=end_freq)
        temp2 = FeatureExtractor(ali_dataset.cal, start_freq=start_freq, end_freq=end_freq)
        temp1.my_dict.update({'cal1': temp2.repo})
        temp1.save_my_dict()


class FeaturePlotter(tb.Artist):
    def __init__(self, antenna, figname='FeaturePlot'):
        self.row_ids = [f'S[{antenna},{antenna - ii}]' for ii in [1, 0, -1]]
        self.col_ids = ['Real Part', 'Imaginary Part', 'Magnitude', 'Angle']
        super(FeaturePlotter, self).__init__(create_new_axes=True, figname=figname, figsize=(13, 8))
        # self.maximize_fig()
        self.plotter = None
        self.create_new_axes = True

    def plot(self, dat, title=None, **kwargs):  # recieves an array of shape 3 x 4, and feature id
        self.get_axes()
        self.fig.suptitle(f"Feature Name = {title}", size="xx-large", color="red")
        for ii, row in enumerate(dat):  # 3 signals
            for jj, col in enumerate(row):  # real, imag, angle, magnitude
                self.ax[ii, jj].plot(col)
                self.ax[ii, jj].set_title(f"feature applied on {self.col_ids[jj]} of {self.row_ids[ii]}", size='small')
        plt.pause(0.01)  # allow plots to take final shape before proceeding cause next steps depend on that.
        self.grid(self.ax.ravel())

    def get_axes(self):
        if self.create_new_axes:
            axis = self.fig.subplots(nrows=3, ncols=4)
            self.fig.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.9, wspace=0.2, hspace=0.35)
            self.ax = axis
        else:  # use same old axes
            pass

    @staticmethod
    def from_dict(my_dict, antenna=0):
        data_ = my_dict['repo'][antenna]
        titles_ = my_dict['titles']
        return tb.VisibilityViewerAuto(data=[data_], titles=titles_, artist=FeaturePlotter(antenna),
                                       save_type=tb.SaveType.Null, pause=True,
                                       save_dir=tb.P('tmp').relativity_transform())

    @staticmethod
    def help():
        my_dict = tb.Read.read(r"D:\RESULTS\datasets\raw\feature_bank_ali_start_freq600_end_freq_1800.npy")
        plotter = FeaturePlotter.from_dict(my_dict)
        return my_dict, plotter


class Search:
    def __init__(self, start_freq=600, end_freq=1200, cal_type='div'):
        repo_path = fr"D:\RESULTS\Boundary\boundary_esimation_ali_5tissuemodel\feature_bank_ali_start_freq_" \
                    fr"{start_freq}_end_freq_{end_freq}.npy"
        my_dict = tb.Read.read(repo_path)  # Antenna(16) x Feature(64) x Signal(3) x Part(4) x Distances(40)
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.repo = my_dict['repo']
        self.titles = my_dict['titles']
        self.titles = [title + f"  (index={idx})" for idx, title in enumerate(self.titles)]
        self.sim_cal1_repo = my_dict['cal1']
        self.features_used = np.arange(64)
        self.data_features = self.data_features_raw = None
        self.ant_dfs = tb.List()
        self.s_obj = None
        self.plotter = None
        self.data_cal1 = self.bias_repo = None
        self.cal_type = cal_type

    def infer_from_position(self, pos_obj):
        pos_obj.read_cal1()
        self.data_cal1 = FeatureExtractor(pos_obj.data_cal1, start_freq=self.start_freq, end_freq=self.end_freq)
        if self.cal_type == 'div':
            self.bias_repo = self.sim_cal1_repo / self.data_cal1.repo
        else:
            self.bias_repo = self.sim_cal1_repo - self.data_cal1.repo
        self.infer(pos_obj.read().data_m, bias_repo=self.bias_repo)

    def infer(self, s_obj, bias_repo=None):
        self.s_obj = s_obj
        self.data_features_raw = FeatureExtractor(s_obj, start_freq=self.start_freq, end_freq=self.end_freq)
        from copy import copy
        self.data_features = copy(self.data_features_raw)
        if bias_repo is not None:  # update the respository of data features by adding the bias
            if self.cal_type == 'div':
                self.data_features.repo = self.data_features.repo * bias_repo
            else:
                self.data_features.repo = self.data_features.repo + bias_repo
        for ant in range(self.s_obj.nports):
            res_feat = []
            for feature_idx in self.features_used:
                res_feat.append(self.classify_one_feature(ant, feature_idx))
            # Build a dataframe out of the results.
            # 64Features x 3Parts x (3+4)things
            my_dict = {'feature_idx': self.features_used, 'feature_name': [self.titles[i] for i in self.features_used],
                       'ant_idx': [ant for _ in self.features_used]}
            columns = ['prev_distance', 'sii_distance', 'next_distance',
                       'dist_left', 'class_left', 'dist_right', 'class_right']
            all_columns = ['real_' + i for i in columns] + ['imag_' + i for i in columns] + \
                          ['abs__' + i for i in columns] + ['angl_' + i for i in columns]
            for index, a_column in enumerate(all_columns):
                part_idx = {'real': 0, 'imag': 1, 'abs_': 2, 'angl': 3}[a_column[:4]]
                subt = part_idx * 7
                tmp = [feat[part_idx][index - subt] for feat in res_feat]
                my_dict.update({a_column: tmp})
            df = pd.DataFrame(my_dict)
            self.ant_dfs.append(df)

    def classify_one_feature(self, ant, feature_idx):
        result = []
        for part_idx in range(4):
            result.append(self.classify_one_part(ant, feature_idx, part_idx))
        return result

    def classify_one_part(self, ant, feature_idx, part_idx):
        feature_values = self.data_features.repo[ant, feature_idx, :, part_idx, 0]  # 1 value for each signal (3)
        signals = self.repo[ant, feature_idx, :, part_idx, :]  # 40 values for each signal (3)
        results = Search.search_adjacent(signals, feature_values)
        return results

    @staticmethod
    def search_adjacent(signals, values):  # 3 x signals and 3 values
        results = []
        for signal, value in zip(signals, values):
            results.append(Search.get_intersection(signal, value, limit=3))  # 3 intersections.
        ref = results[1]  # notice that those indices are distances as well.
        dist_left, class_left = Search.match_results(ref, results[0])
        dist_right, class_right = Search.match_results(ref, results[-1])
        points = [(a_result, a_value) for a_result, a_value in zip(results, values)]
        return points + [dist_left, class_left, dist_right, class_right]

    @staticmethod
    def match_results(one, two, verbose=False):
        if len(one) == 0 or len(two) == 0:
            return None, None
        matrix = abs(np.array(one) - np.array(two)[:, None])
        row, col = np.unravel_index(matrix.argmin(), matrix.shape)
        if verbose:
            print(f"Closest match is {one[row]} and {two[col]} @ {row, col} respectively")
        if matrix.min() <= 1:  # error is less than or equal one millimeter
            return one[col], col  # return distance + index of this distances among potential results
        else:  # This feature failed
            return None, None

    @staticmethod
    def get_intersection(signal, value, limit=3, verbose=False):
        # use scipy peak finder with abs(diff) and ensure distance between results.
        distances = np.argwhere(np.diff(np.sign(signal - value))).squeeze()
        if distances.shape == ():
            distances = distances[None, ...]
        distances = distances[:limit]
        final_distances = []
        for distance in distances:
            other_candidates = [distance]  # 0
            if distance != 0:
                other_candidates.append(distance - 1)  # 1
            if distance != len(signal) - 1:
                other_candidates.append(distance + 1)  # 2
            other_candidates = np.array(other_candidates).squeeze()
            nearest = np.argmin(abs(signal[other_candidates] - value))
            nearest_distance = other_candidates[nearest]
            final_distances.append(nearest_distance)
            if verbose:
                print(f"Diff = {signal[nearest_distance] - value}")
        return final_distances

    def plot(self, ant=10, feature_idx=1):
        """Uses plot_all_features but also adds an extra horizontal line to the plot showing the level of
        every feature for a certian realistic input."""
        self.plotter = tb.VisibilityViewerAuto(data=[self.repo[ant]], titles=self.titles, artist=FeaturePlotter(ant),
                                               save_type=tb.SaveType.Null, pause=True,
                                               save_dir=tb.P('tmp').relativity_transform())
        self.plotter.index = feature_idx
        self.plotter.animate()
        for ii, sig_ax in enumerate(self.plotter.artist.ax):
            for jj, part_ax in enumerate(sig_ax[:4]):
                dist, val = self.ant_dfs[ant].query(f"feature_idx=={feature_idx}").iloc[0][3 + ii + jj * 7]
                if val:
                    part_ax.axhline(val, color='red')
                for a_dist in dist:
                    part_ax.axvline(a_dist, color='black')

    def save_dfs(self, root=None):
        if root is None:
            root = tb.P.tmp('dfs')
        for idx, df in enumerate(self.ant_dfs):
            df.to_csv(root / f"ant_{idx}.csv")

    def plot_all_features(self, ant=10, save_name='results', save_dir=None):
        """Generates a pdf with 62 features (one feature is examined per page)
        In each page, the feature is applied on {real, imag, phase and mag} for sii, sii-1 and sii+1 (4x3 subplots)
        In each figure the feature is applied to 40 simulations, so the x-axis foes from 1 to 40 in each plot.
        Notice that cal_type and freq range are fixed from constructor initialization.
        """
        saver = tb.SaveType.PDF(save_name=save_name, save_dir=save_dir)
        for feat in self.features_used[:]:
            self.plot(ant=ant, feature_idx=feat)
            if plt.get_backend() == 'TkAgg':
                plt.pause(0.75)
            saver.add(fig_names=['FeaturePlot'])
            plt.close('all')
        saver.finish(open_result=False)

    @staticmethod
    def gen_data(index=0):
        """
        """
        hht3 = m.HHT3()
        case = hht3.cases[index]
        case.positions[0].read()
        cases = [case.positions[0]]
        cal_types = ['div', 'sub']
        freqs = [1200, 1800]
        for case in cases:
            for cal_type in cal_types:
                for freq in freqs:
                    temp = Search(cal_type=cal_type, end_freq=freq)
                    temp.infer_from_position(case)
                    folder = tb.P.tmp(f"{case.name}_freq_{freq}_cal_{cal_type}")
                    temp.save_dfs(root=folder)
                    for ant in range(16):
                        desc = f"{case.name}_freq_{freq}_cal_{cal_type}_ant_{ant}"
                        temp.plot_all_features(ant=ant, save_name=desc, save_dir=folder)


def findings():
    """Test whether adjacent S params are informative for distance estimation
    """
    # pa_d = pa.s.get_all_entries_per_antenna(num_diag=1, exclude_main=True).np.reshape(-1, 600, 2)
    # make sure to get_common_freq before this step.
    obj = AliDatasetHandler()
    a_d = obj.s.get_all_entries_per_antenna(num_diag=2, exclude_main=True).np.reshape(-1, 600, 2)
    legs = [[f"S {i, i - 1}", f"S {i, i + 1}"] for i in range(16)]
    _ = tb.VisibilityViewerAuto(data=[abs(a_d)], figname='Ali data', titles=np.arange(1, 41).repeat(16),
                                legends=legs * len(a_d))
    # _ = tb.VisibilityViewerAuto(data=[abs(pa_d)], figname='PA data', titles=pa.s.name.np.repeat(16),
    #                             legends=legs * len(pa.s))

    # This gives the same plot that splits the data into three categories.
    plt.plot(tb.List(a_d[..., 0]).apply(np.angle).apply(tsfel.features.kurtosis).np[::16])

    classif = KMeans(n_clusters=3)
    classif.fit(a_d.sum(axis=(1, 2))[..., None])
    classif.cluster_centers_ = np.array([37., 77., 90.])[..., None]
    # _ = classif.predict(pa_d.sum(axis=(1, 2))[..., None])

    # cfg = tsfel.get_features_by_domain()
    # features = tsfel.time_series_features_extractor(cfg, pa_d)
    # classif = KMeans(n_clusters=3)
    # classif.fit(features)
    # _ = classif.predict(features)


class NsevArtist(tb.Artist):
    """NLFT & Scattering Transform feature extraction"""
    def __init__(self, transf='NLFT'):
        super(NsevArtist, self).__init__(create_new_axes=True)
        self.transf = transf

    def plot(self, *data, mm=256, xi1=-5, xi2=5, title=None, **kwargs):
        """
        :param title:
        :param mm: Samples of the continous spectrum
        :param xi1:
        :param xi2:
        :param kwargs:
        :return:
        """
        self.get_axes()
        if self.transf == 'NLFT':
            from FNFTpy import nsev
            res = nsev(*data, np.linspace(0, 10, len(data[0])), M=mm, Xi1=-5, Xi2=5)
            for i, (key, val) in enumerate(res.items()):
                if key == 'options':
                    print('Options'.center(50, '='))
                    print('\n'.join(res['options'].split(',')))
                elif i > 2:
                    print(key, val.shape)
                else:
                    print(key, val)
            # self.ax.plot(abs(res['disc_norm']), label='disc_norm')
            self.ax[0].plot(abs(res['cont_ref']), label='cont_norm')
            self.ax[0].legend()
            self.ax[0].set_title(title)
        else:
            from kymatio.numpy import Scattering1D  # HarmonicScattering3D
            res = Scattering1D(4, Q=10, shape=len(data[0]))(*data)
            self.ax[0].plot(abs(res[0]), label='Scattering Coefficients')
            self.ax[0].legend()
            self.ax[0].set_title(title)

    @staticmethod
    def go():
        aa = AliDatasetHandler()
        return tb.VisibilityViewerAuto(data=[aa.s.get_ii().np[..., 0]], artist=NsevArtist(transf='SWT'))


if __name__ == '__main__':
    # pa = m.PA()
    a = AliDatasetHandler()
    # w = Search()
    # w.infer_from_position(pa.positions[0])
    # w.plot_all_features()
    # # w.infer_from_position(pa.cases[0].positions[0])
