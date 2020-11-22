
"""
This is a Python Implementation of Resonance Shift technique by Ali Zamani.
"""
import measurements as m
import resources.all_in_one as tb
import gen_dataset.toolbox as tbd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)


class Resonance:
    def __init__(self, s_params, freq, name=None, path=None):
        self.name = name
        self.path = path
        self.n = s_params.shape[-1]  # Num of antennas
        self.resonances = np.zeros((self.n, 3), dtype='float32')
        self.freq = freq
        self.s = s_params[:, range(self.n), range(self.n)]  # pick main diagonal
        self.s = np.abs(self.s).T  # now it is: n x #num_freq
        for i, signal in enumerate(self.s):
            minima, _ = self.find_resonance_per_s(signal)
            self.resonances[i, :len(minima)] = minima

    @classmethod
    def from_s_path(cls, path):
        s, freq, name = tb.S(path).get_data_dict().values
        s = s[0]
        return cls(s, freq, name, path)

    def find_resonance_per_s(self, signal):
        fmin = np.searchsorted(self.freq, 0.7)
        fmax = np.searchsorted(self.freq, 1.65)
        freq_ = self.freq[fmin:fmax]
        s_params_ = signal[fmin:fmax]
        ss = -s_params_  # make valleys peaks
        # difference between resonances is at least 300 MHz, we'll go with 200MHz
        # Translating this to index distance:
        distance = 200 * 0.001 / (freq_[1] - freq_[0])
        width = 50 * 0.001 / (freq_[1] - freq_[0])
        indices = find_peaks(ss, distance=distance, width=width)[0]
        minima_freq = freq_[indices]
        minima_vals = -ss[indices]
        return minima_freq, minima_vals

    def plot(self):
        fig = plt.figure('Minima')
        ax = fig.subplots(4, 4)
        fig.tight_layout()
        ax = ax.ravel()
        for i, an_ax in enumerate(ax):
            an_ax.plot(self.freq, self.s[i], picker=True)
            an_ax.set_title(str(i))
            picks = np.zeros_like(self.freq).astype('bool')
            for afreq in self.resonances[i]:
                picks += np.isclose(self.freq, np.round(afreq, 6))
            an_ax.plot(self.freq, picks)

        def onpick1(event):
            if isinstance(event.artist, plt.Line2D):
                thisline = event.artist
                xdata = thisline.get_xdata()
                ydata = thisline.get_ydata()
                ind = event.ind
                print(xdata, ydata, ind)
        fig.canvas.mpl_connect('pick_event', onpick1)


class FrequencyShift:
    def __init__(self, cals_path=None):
        self.result_cal1 = self.result_cal2 = None
        self.plotter = None
        # Read Simulation Cals and save them.
        if cals_path is None:
            cals_path = r'D:\RESULTS\WienerSims\cals\Results\cals'
        self.cals_path = tb.P(cals_path).myglob('*cal*')
        self.sim_cal1_res = Resonance.from_s_path(self.cals_path[0])
        self.sim_cal2_res = Resonance.from_s_path(self.cals_path[0])
        if len(self.cals_path) > 1:
            self.sim_cal2_res = Resonance.from_s_path(self.cals_path[1])
        self.measured_cal1_res = None
        self.measured_cal2_res = None
        self.measured_case_res = None
        self.diff_cal1 = self.diff_cal2 = None
        self.s = None  # do not read the S dataset.
        self.s_names = None  # names of s files for double checking.
        self.heads = None  # do not load this by default.
        self.names = np.load(r'D:\RESULTS\datasets\raw\names.npy')
        self.source = np.load(r'D:\RESULTS\datasets\raw\source.npy')
        self.boundaries = np.load(r'D:\RESULTS\datasets\s_image\boundaries.npy')
        self.resonance_bank = np.load(r'D:\RESULTS\datasets\raw\resonances.npy')
        self.antenna_positions = np.load(tb.P(r'gen_dataset\antenna_positions.npy').relativity_transform())
        self.normals_bank = None
        self.use_normals_bank = False  # as opposed to boundaries bank.
        self.n_resonances = 1
        self.fig = None
        self.ax = None
        self.campaign_results = None

    def load_s_and_heads(self):
        self.heads = np.load(r'D:\RESULTS\datasets\raw\heads.npy', allow_pickle=True)
        data = np.load(r'D:\RESULTS\datasets\raw\s.npy', allow_pickle=True).all()
        self.s = data['s']
        self.s_names = data['names']

    def generate_resonance_bank(self, data=None, save_name='resonances'):
        if data is None:
            data_path = r'D:\RESULTS\datasets\raw\s.npy'
            data = np.load(data_path, allow_pickle=True).all()
        self.s = data['s']
        self.s_names = data['names']
        self.resonance_bank = np.zeros((self.s.shape[0], self.s.shape[-1], 3))
        for i, s in tqdm(enumerate(self.s)):
            self.resonance_bank[i] = Resonance(s, self.sim_cal1_res.freq).resonances
        np.save(rf'D:\RESULTS\datasets\raw\{save_name}.npy', self.resonance_bank)

    def estimate_boundary(self, position, plot=True):
        self.measured_cal1_res = Resonance.from_s_path(position.cal1)
        self.measured_cal2_res = Resonance.from_s_path(position.cal2)
        self.measured_case_res = Resonance.from_s_path(position.path[0])
        self.diff_cal1 = self.measured_cal1_res.resonances - self.sim_cal1_res.resonances
        self.diff_cal2 = self.measured_cal2_res.resonances - self.sim_cal2_res.resonances

        boundaries, match_indices = self.match_to_bank(self.measured_case_res.resonances - self.diff_cal1)
        self.result_cal1 = tb.BoundaryDescriptor(xy=boundaries, match_indices=match_indices, position=position)
        boundaries, match_indices = self.match_to_bank(self.measured_case_res.resonances - self.diff_cal2)
        self.result_cal2 = tb.BoundaryDescriptor(xy=boundaries, match_indices=match_indices, position=position)
        if plot:
            self.plot()

    def match_to_bank(self, resonance):
        normal_distances = []
        match_indices = []
        for i, antenna_res in enumerate(resonance):
            diff = self.resonance_bank[:, i, :] - antenna_res
            diff = np.linalg.norm(diff[:, :self.n_resonances], axis=-1)
            idx = np.argmin(diff)
            error = self.resonance_bank[idx, i, :] - antenna_res
            match_indices.append([idx, self.names[idx], self.source[idx], antenna_res.round(3),
                                  self.resonance_bank[idx, i, :].round(3), (error * 1000).round(3)])
            if not self.use_normals_bank:
                normals, _ = tb.get_normal_distances(self.boundaries[idx], self.antenna_positions)
                # This returns normals for all antennas.
            else:  # In Ali's bank the distance is the same for all antennas
                normals = [self.normals_bank[idx] for _ in range(16)]
            normal_distances.append(normals[i])
        xy = tb.get_xy_from_normals(normal_distances, self.antenna_positions)
        boundaries = tb.interpolate(xy)
        match_indices = pd.DataFrame(match_indices,
                                     columns=['Match Index', 'Name', 'Source', 'Measured Resonance',
                                              'Bank Match Resonance', 'Error in MHz'],
                                     index=range(len(resonance)))
        # match_indices.rename_axis('Antenna', inplace=True)
        return boundaries, match_indices

    def process_campaign(self, handle):
        results = []
        names = []
        for position in tqdm(handle.root.positions):
            try:
                self.estimate_boundary(position, plot=False)
                results.append((self.result_cal1, self.result_cal2))
                names.append(position.name)
            except FileNotFoundError as e:
                _ = e
                pass  # there's no cal for this case. "No such a file called None"
        self.campaign_results = results
        self.play()

    def play(self, save_type=tb.SaveType.Null):
        saver = save_type()
        self.fig = plt.figure()
        self.ax = self.fig.subplots(1, 1)
        for aresult in self.campaign_results:
            self.result_cal1 = aresult[0]
            self.result_cal2 = aresult[1]
            self.plotter = tb.BoundaryPlotter(self.result_cal1, antenna_positions=d.antenna_positions,
                                              ax=self.ax, name=self.result_cal1.position.name)
            saver.add()
            self.ax.cla()
        saver.finish()

    def plot(self, **kwargs):
        self.fig = plt.figure()
        self.ax = self.fig.subplots(1, 1)
        self.plotter = tb.BoundaryPlotter(self.result_cal1,
                                          antenna_positions=d.antenna_positions, ax=self.ax, **kwargs)
        self.plotter = tb.BoundaryPlotter(self.result_cal2, name=self.result_cal1.position.name, ax=self.ax)

    def show(self, idx):
        return tbd.Plotter(self.heads[idx]).show_slice()


def base_on_ali_bank():
    dd = FrequencyShift(cals_path=r'D:\RESULTS\Boundary\boundary_esimation_ali_5tissuemodel')
    dd.use_normals_bank = True
    dd.normals_bank = np.arange(1, 41, 1)
    generate = False
    if generate:
        s, freq, names = tb.S(r'D:\RESULTS\Boundary\boundary_esimation_ali_5tissuemodel').get_data_dict().values
        dd.generate_resonance_bank(data={'s': s, 'names': names}, save_name='resonances_ali_bank')
    else:
        dd.resonance_bank = np.load(r"D:\RESULTS\datasets\raw\resonances_ali_bank.npy")
    dd.estimate_boundary(m.HHT3().positions[0])


if __name__ == '__main__':
    d = FrequencyShift()
    d.estimate_boundary(m.HHT3().positions[0])
    _ = tbd
