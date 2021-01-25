
import enum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import alexlib.toolbox as tb
import cv2 as cv
import skrf
import alexlib.miscellaneous as ms
from resources import source_of_truth as st


#%% ========================== S Param Kit ==============================================


class Dtype(enum.Enum):
    polar = 'polar'
    rect = 'rectangular'
    real = 'real'
    imag = 'imaginary'
    magnitude = 'magitude'
    phase = 'phase'
    complex = 'complex'


class S(tb.Base):
    """
    As per separation of concerns principle, this should never worry about calibration or other details.
    This is only concerned with single S param, no method to handle anything else beyond this.
    """
    # ================================== Constructors ==============================
    def __init__(self, path=None, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if path is not None:
            self.nw = skrf.network.Network(str(path))
            self.path = tb.P(path)
            self.s = self.nw.s[None, ...]  # add extra dimension to indicate 1 set of measurements only.
            self.f = self.nw.frequency
            self.nports = self.s.shape[-1]
            if name is None:
                name = list(self.path.parts[-3:])
                name[-1] = self.path.stem
                self.bname = ' | '.join(name)
                name = '__'.join(name)
                self.name = name
            else:
                self.bname = name
            self.name = name
        else:
            self.nw = self.path = self.s = self.f = self.nports = self.name = self.bname = None
        # ===============================================
        # self.alpha = self.beta = None
        self.fig = None
        self.ax = None

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        _ = memodict
        obj = self.__class__()
        shallow = self.__dict__.copy()
        shallow.update(dict(fig=None, ax=None))
        obj.__dict__.update(tb.copy.deepcopy(shallow))
        return obj

    @staticmethod
    def get_dataset_from_paths(paths_list):  # returns a dataset with: tensor of S, list of names, frequency
        paths_list = tb.List(paths_list)
        if len(paths_list) == 0:
            return S()
        results = paths_list.apply(S, verbose=True)  # A List
        sample = results[0]  # get an S object.
        sample.s = results.s.np.squeeze()  # should be N x Freq x ports x ports.
        sample.name = paths_list.stem
        sample.path = paths_list
        return sample

    def __repr__(self):
        return f"S object. Instance details are:\n" \
               f"Name = {self.bname}\nS shape = {self.s.shape}\n{str(self.f)}.\nStep = {self.f.step:,} Hz"

    # ============================================== OPS =====================================================
    def select(self, index=None):
        """Select a measurement from self.S withwhich self.nw.s will be populated, and thus, plotted later."""
        if index:
            self.nw.s = self.s[index]
        return self

    def __sub__(self, other):
        from copy import copy
        obj = copy(self)
        if self.nw is not None:
            nw_obj = self.nw - other.nw
            obj.nw = nw_obj
        else:
            obj.s = obj.s - other.s
        return obj

    def __len__(self):
        return self.s.shape[0]

    def __add__(self, other):
        """add two s params to make dataset that includes both"""
        self.s = np.concatenate([self.s, other.s], axis=0)
        self.name = self.name + other.name
        self.path = self.path + other.path
        return self

    def merge(self, other, clone=False):
        return self + other.get_common_freq(self, clone=clone)

    def __getitem__(self, item=None):
        """Helps splitting one object from a larger one."""
        if item is None:
            item = np.random.choice(len(self.s))
        result_dict = self.__dict__.copy()  # shallow copy
        result_dict.update(dict(s=self.s[item][None, ...], name=self.name[item]))
        result = self.__class__()
        result.__dict__.update(result_dict)
        return result

    def __mul__(self, scalar):
        result = self.__deepcopy__()
        result.s *= scalar
        return result

    # ================================ Plotting methods =======================================
    def plot_sii(self, idx=None, **kwargs):
        """Plot all Sii's
        """
        self.select(idx)
        if self.nw:
            for i in range(0, self.nports):
                self.nw.plot_s_db(m=i, n=i, **kwargs)
        tb.FigureManager.grid(plt.gcf().axes[0])

    def plot_sii_separately(self, idx=None):
        """
        """
        self.select(idx)
        s = self.nw.s
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30, 30))
        i, j = np.mgrid[0:4, 0:4]
        counter = 0
        for x, y in tqdm(zip(i.ravel(), j.ravel())):
            ax[x, y].plot(self.f.f, 20 * np.log10(abs(s[:, counter, counter])))
            ax[x, y].grid()
            ax[x, y].set_title(str(counter + 1))
            counter += 1

    def plot_all(self, idx=None, **kwargs):
        """
        If we want to ignore repeated S params:
        """
        self.select(idx)
        x, y = np.meshgrid(range(0, self.nports), range(0, self.nports))
        ix, iy = np.triu_indices(self.nports)
        x, y = x[ix, iy], y[ix, iy]
        for i, j in tqdm(zip(x, y)):
            plt.plot(20 * np.log10(abs(self.nw.s[..., i, j])), **kwargs)

    def animate_signals(self, string="self.s[0, :, 0, :].__abs__()", **kwargs):
        data = self.evalstr(string)
        return tb.VisibilityViewerAuto(data.T[None, ...], **kwargs)

    def plot_row(self, idx=0, row_idx=0, **kwargs):
        """ Plot a row of S matrix, i.e. signals that relate to a single antenna
        The args you need to pass are the same as the ones for plotcomp, except for the column.
        """
        return self.plot(string=f"self.s[{idx}, :, {row_idx}, :].__abs__()", **kwargs)

    def plot(self, string="self.s[0, :, 0, :].__abs__()", scale='linear'):
        def scaler(x):
            return x
        if scale == 'linear':
            pass
        elif scale == 'log':
            scaler = Norm.db
        data = self.evalstr(string)
        data = scaler(data)
        plt.plot(data)

    # ================== Picking entries methods ==============================================
    def get_adjacent(self, diff=0, verbose=False):
        """Returns all S params [i, j] that are ``diff`` away from each other. I.e. ``| i - j | = diff``
        .. note:: * if ``diff=0``, it returns ``ii`` main diagonal elements.
                  * This acts as giving a single off diagonal + extra elements required to make its length = ``nports``.

        """
        if diff == int(self.nports / 2):
            i = np.arange(int(self.nports / 2))
        else:
            i = np.arange(self.nports)

        j = i + diff
        j = np.where(j >= self.nports, j - self.nports, j)
        if verbose:
            print(np.vstack([i, j]))
        return self.s[..., i, j]

    def get_ii(self):
        return self.get_adjacent(diff=0)

    def get_entries_per_antenna(self, ant_idx, num_diag=1, exclude_main=False, verbose=False):
        """Returns entries of S matrix that are related to one antenna, up to certain adjacency.
        This is different from picking a row in that the antenna of interest is mdae always in the middle.
        """
        bef = ant_idx - np.arange(num_diag)
        aft = ant_idx + np.arange(num_diag)
        bef = np.flip(bef)[:-1]
        aft = aft[1:] if exclude_main else aft
        tot = np.concatenate([bef, aft])
        for idx, elem in enumerate(tot):
            if elem >= self.nports:
                tot[idx] = elem - self.nports
        if verbose:
            print(f"Indices selected are:\ni = {ant_idx} \nj = {tot}")
        return self.s[..., ant_idx, tot]

    def get_all_entries_per_antenna(self, num_diag=1, exclude_main=False):
        results = [self.get_entries_per_antenna(i, num_diag=num_diag, exclude_main=exclude_main)
                   for i in range(self.nports)]
        return np.array(results)

    # ======================= Reshaping methods ===============================
    def vectorize(self):
        return self.s.reshape(self.f.npoints, np.prod((self.nports, self.nports)))

    # ======================= Selecting frequency methods =========================
    def get_common_freq(self, other_f=None, clone=False):
        if other_f is None:
            return self

        indices = np.searchsorted(self.f.f, other_f.f)
        assert len(indices) == other_f.npoints, 'Something wrong happened here'
        new_freq = skrf.Frequency.from_f(self.f.f[indices], unit='Hz')
        if not clone:
            obj = self
        else:
            obj = self.__deepcopy__()

        obj.s = self.s[:, indices, ...]
        obj.f = new_freq
        obj.nw.s = obj.s
        obj.nw.f = obj.f
        return obj

    def get_approximate_common_freq(self, other_f, verbose=False, clone=False):
        indices = []
        for freq in other_f.f:
            indices.append(np.argmin(np.abs(self.f.f - freq)))
        if verbose:
            print(self.f[indices] - other_f)
        if clone:
            self.s = self.s[:, indices, ...]
            self.f.f = self.f.f[indices]
            return self
        else:
            obj = tb.copy.deepcopy(self)
            obj.s = self.s[:, indices, ...]
            self.f.f = self.f.f[indices]
            obj.indices = np.array(indices)
            return obj

    def interpolate_the_overlap(self, other_f):
        """Get common frequencies and interpolate as required by other_f
        Caveat: works only per S object"""
        # use freq.overlap and nw.interpolate
        pass

    # =========================== Stability Analysis ===============================
    def check_symmetry(self):
        return Norm.db(self.s[0] - self.s[0].transpose(0, 2, 1))

    def differ_to(self, other):
        diff = self.s[0] - other.s[0]
        res = Norm.db(diff)
        return res

    @staticmethod
    def get_stability_matrix(meas, plot=True):
        mat = np.zeros(shape=(len(meas), len(meas)))
        for i in range(len(meas)):
            for j in range(i + 1):
                mat[i, j] = np.max(meas[i].differ_to(meas[j]))
                mat[j, i] = mat[i, j]
        mat = mat.clip(min=-120)
        if plot:
            fm = tb.ImShow([mat])
            plt.colorbar(fm.fig.axes[0].images[0])
        return mat


class Calibrator(tb.Base):
    """To close the gap between simulations and reality, this class uses cals in both domains to come up with a
    formula that can later be used to convert other S parameters from one domain to the other."""
    def __init__(self, s2t='r2s', kind=["mul", "add", "linear", "eye"][2], dtype=Dtype.complex, **kwargs):
        """Options available:
        * What is the source and what is the target domain? e.g. source is reality and target is simulation ``r2s``.
        * Object to be feed for calibration? Array or S object.
        * Kind of calibration:
            * add: 1 factor calibration (alpha)
            * mul: 1 factor calibration (alpha)
            * linear: two factor calibration (alpha + beta)
            * eye: identity calibration.
        """
        super().__init__(**kwargs)
        self.s2t = s2t
        self.kind = kind
        self.dtype = dtype
        self.alpha = self.beta = self.cals = None

    @staticmethod
    def solve_two_cals_based_calibration_equations():
        """source (s) could be reality and target (t) could be simulations or vice versa."""
        from sympy import symbols, Eq, solve
        cal1_s, cal1_t, cal2_s, cal2_t, alpha, beta = symbols('cal1_s cal1_t cal2_s cal2_t alpha beta')
        eq1 = Eq(alpha * cal1_s + beta, cal1_t)
        eq2 = Eq(alpha * cal2_s + beta, cal2_t)
        print(solve([eq1, eq2])[0])

    def load_up_cal_s_objects(self, pos=None):
        """Loads up the cals from a ``Position`` object, in addition to Simulation Cals.
        :param pos: A position object
        """
        cal1_sim = S(st.d / "datasets/raw/base_model_with_CalPhantom1.s16p")
        cal2_sim = S(st.d / "datasets/raw/base_model_with_CalPhantom2.s16p")
        base_model = S(st.d / "datasets/raw/base_model.s16p")
        if pos:
            pos.read_cals()
        else:
            pos = tb.Struct(data_cal1=None, data_cal2=None)
        if self.s2t == 'r2s':
            self.cals = tb.Struct(cal1_s=pos.data_cal1, cal2_s=pos.data_cal2, base_model=base_model,
                                  cal1_t=cal1_sim, cal2_t=cal2_sim)
        elif self.s2t == 's2r':
            self.cals = tb.Struct(cal1_s=cal1_sim, cal2_s=cal2_sim, base_model=base_model,
                                  cal1_t=pos.data_cal1, cal2_t=pos.data_cal2)
        return self

    def get_s_params_from_s_objects(self, clone=False):
        """Takes care of frequency matching detail amount source and target domains then returns properly named arrays.

        :param clone:
        :return: A structure with numerical arrays named cal1_s, cal2_s, cal1_t, cal2_t
        """
        cals = self.cals
        num_struct = None
        if self.kind == "linear":
            cal1_s = cals.cal1_s.get_common_freq(cals.cal1_t.f, clone=clone)
            cal2_s = cals.cal1_s.get_common_freq(cals.cal1_t.f, clone=clone)
            num_struct = tb.Struct(cal1_s=cal1_s.s, cal2_s=cal2_s.s, cal1_t=cals.cal1_t.s, cal2_t=cals.cal2_t.s)
        elif self.kind in {"mul", "add"}:
            cal1_s = cals.cal1_s.get_common_freq(cals.cal1_t.f, clone=clone)
            num_struct = tb.Struct(cal1_s=cal1_s.s, cal1_t=cals.cal1_t.s)
        elif self.kind == "eye":
            pass
        else:
            raise KeyError(f"Unknown calibration kind {self.kind}")

        if self.dtype == Dtype.complex:
            pass  # leave num_struct as is, which is complex valued.
            assert "complex" in str(num_struct.dtype), "This should be complex valued"
        else:
            num_struct.apply(lambda x: Norm.complex_to_real(x, dtype=self.dtype))

        return num_struct

    def generate_factors(self, num_struct):
        """
        :param num_struct: A structure that has strictly numpy arrays as attributes.
        :return:
        """

        if self.kind == "linear":
            """populates alpha and beta attributes."""
            self.alpha = (num_struct.cal1_t - num_struct.cal2_t) / (num_struct.cal1_s - num_struct.cal2_s)
            self.beta = (num_struct.cal1_s * num_struct.cal2_t - num_struct.cal1_t * num_struct.cal2_s) / \
                        (num_struct.cal1_s - num_struct.cal2_s)
        elif self.kind == 'mul':
            """only populate alpha"""
            self.alpha = num_struct.cal1_t / num_struct.cal1_s
        elif self.kind == 'add':
            self.alpha = num_struct.cal1_t - num_struct.cal1_s
        elif self.kind == "eye":
            self.alpha = 1.0
        else:
            print(f"Unknown kind {self.kind}".center(100, "!"))

    def calibrate(self, data, clone=True):
        """ Use this only after figuring out alpha and beta values.
        :param data: A numpy array with the same dimensions as self.alpha and self.beta, or broadcast should apply.
        :param clone:
        """

        if clone:
            tmp = data.copy()
        else:
            tmp = data

        if self.kind == "mul":
            calibrated = self.alpha * tmp
        elif self.kind == "add":
            calibrated = self.alpha + tmp
        elif self.kind == "linear":
            calibrated = self.alpha * tmp + self.beta
        elif self.kind == "eye":
            calibrated = tmp
        else:
            print(f"Unknown kind of calibration {self.kind}".center(100, "!"))
            calibrated = None
        return calibrated

    def calibrate_s_object(self, s_obj, clone=True):
        """This is the same as ``calibrate`` method except that it accepts s_objects instead of numpy array.
        :param s_obj:
        :param clone:
        :return:
        """
        calibrated = s_obj.get_common_freq(self.cals.cal1_t.f, clone=clone)
        calibrated.s = self.calibrate(calibrated.s)
        return calibrated

    def load_cals_generate_factors_calibrate_s_object(self, pos, clone=True):
        self.load_up_cal_s_objects(pos)
        num_struct = self.get_s_params_from_s_objects()
        self.generate_factors(num_struct)
        return self.calibrate_s_object(pos.read_measurement().data_m, clone=clone)


class SSelect:
    """
    Those static methods are either selected (unexecuted) or in some cases evaulated and returns indices.
    """

    @staticmethod
    def all(x):
        return x

    @staticmethod
    def ii(x):
        """pick main diagonals. Don't do it with slices s = s[:, :, :16, :16]
        """
        n = x.shape[-1]
        return x[..., range(n), range(n)]

    @staticmethod
    def ij(x):
        """Selects off diagonal entries.
        """
        n = x.shape[-1]
        # beware of ambiguous Numpy behaviour when selecting index in first dimension result[3, :, [2, 4]]
        # which happens only when mixing indexing styles.
        return x[..., np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]]

    @staticmethod
    def ii_plus_ij(x):
        """Eleminate repeated values from the symmetric S matrix.
        """
        n = x.shape[-1]
        return x[..., np.triu_indices(n, 0)[0], np.triu_indices(n, 0)[1]]

    @staticmethod
    def get_diagonals_indices(size, num_diag):
        """Main diag + Next diagonal. The arragement goes like: First, indices of main diagonal, then indices of
        first off diagonal (less impact) and so on.
        Warning: this works only for symmetric matrices.
        :param size: size of the matrix
        :param num_diag: number of diagonals starting from the main (1)
        :return: tupe of indices of type int32
        """
        x = list(range(size)) * num_diag
        y = list(range(size))  # Main diagonal covered, moving on to next diagonals.
        for i in range(1, num_diag):
            # By default this stops at k-1. We start from one beceause we already finished the main D.
            y += list(np.roll(range(size), -i))
        return x, y

    @staticmethod
    def get_indices_per_antenna(idx, num_diag, size=16):
        """Following the use of get_diagonal_indices, entries will be extracted and will be vetorized.
         In some cases, one might want to extract indices pertinent to single antenna from the aforementioned result.
            """
        s_fake = np.zeros(shape=(size, size), dtype='uint8')  # construct a fake S martrix of zeros.
        for value in range(num_diag):  # put nonzero values in entries of interest.
            s_fake[idx, idx + value] = 1  # next to the antenna CCW
            s_fake[idx - value, idx] = 1  # the neighbour from the otherside, but transposed.
        x, y = SSelect.get_diagonals_indices(size=size, num_diag=num_diag)  # apply transformation
        selected_entries = s_fake[x, y]
        return selected_entries.nonzero()[0]

    @staticmethod
    def split_to_per_ant(num_diag=2, size=16):
        """Similar to `get_diagonal_indices` but the arrangement is different. It returns the result per antenna.
        So you get: ant0-(main diag, off-diag1, off-diag2 ...), then the same for the next antennas.
        :param num_diag:
        :param size:
        :return: x, y indices of required entries, to be then applied, and reshaped by to (size, num_diag)
        """
        x, y = np.arange(size), np.array([])
        for ant in x:
            y = np.concatenate([y,
                                tb.Manipulator.slicer(x, slice(ant - num_diag, ant + num_diag + 1))], 0).astype('uint8')
        return np.arange(size).repeat(num_diag * 2 + 1), y

    @staticmethod
    def get_side_indices(ports=16, main=False):
        start, step, end = 0, 1, int(ports / 2)
        indices = []  # related to S param entries
        connec_index = []  # related to connection index (a connection is a fixed distance between antennas)
        for connec in range(0 if main else 1, int(ports / 2), 1):
            previoud_len = len(indices)
            for ant in range(start, end - connec, step):
                indices.append([ant, ant + connec])
            connec_index.append([slice(previoud_len, len(indices)), len(indices) - previoud_len])  # st, sp, len
        return tb.Struct(left=ports - 1 - np.array(indices), right=np.array(indices),
                         conn_slices=np.array(connec_index)[:, 0], connec_len=np.array(connec_index)[:, 1])

    @staticmethod
    def extract_side_differentials(x, main=False):
        # Notice that cal type affects the differentials. Magnitude per signal doesn't, but magnitude of result does.
        idx = SSelect.get_side_indices(ports=x.shape[-1], main=main)
        return x[..., idx.right[:, 0], idx.right[:, 1]] - x[..., idx.left[:, 0], idx.left[:, 1]], idx


class Shape:
    @staticmethod
    def vectorize(x):
        n = x.shape[0]
        num_freq = x.shape[1]
        return x.reshape(n, num_freq, np.prod(x.shape[2:]))

    @staticmethod
    def freq_cat(x):
        # concatenate the re_im values with the frequency axis, doubling its length.
        # result = result.reshape((N, 2 * func, n, n))
        """The trick is to bring the real_imag axis from last position to next to frequency dimension.
        This helps doing the desired reshaping. Proof by example: a = np.arange(18*2*2).reshape(2, 2, 3, 3, 2)
        a.transpose(0, 1, -1, 2, 3).reshape(2, 4, 3, 3)
        """
        # result = result.transpose(1, 2, 0)
        # take back s axis to the last cause it goes first when indexing not slicing
        raise NotImplementedError

    @staticmethod
    def sf_image():
        """ A way to visualize all of S params in a single image.
        The image will be Freq vs S's. The channels will be the parts of the complex numbers.
        """
        raise NotImplementedError

    @staticmethod
    def f_conv(x):
        # result = result[:, :, np.triu_indices(n)[0], np.triu_indices(n)[1]]
        # result = complex_to_real(result, mode='polar', axis=-1)  # Now we have extra dimension of size two in the end.
        # result = result.reshape(result.shape[:-2] + (-1,))
        # real valued, Freq x 277 # concatenating mag_ph into S axis.
        # result = result.transpose((0, 2, 1))  # make frequency spatial and make S channels of a 1-D signal.
        raise NotImplementedError

    @staticmethod
    def time_series(x):  # create extra S param curves (real and imag or abs and ang)
        x = x.reshape(x.shape[:-2] + (-1,))
        # real valued, Freq x 277 # concatenating mag_ph into S axis.
        x = np.flip(x, axis=1)  # we want to start with highest frequency then the lowest, so that its more
        # remembered.
        return x

    @staticmethod
    def eye(x):
        return x  # don't do anything.

    @staticmethod
    def callable(func):
        return Shape(func)

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Preprocessing:
    def __init__(self, hp, m_struct, cal_struct=None, clone=False):
        """ Consumes S objects, returns numerics. The class has the capacity to store intermediate prep results.
        Freq selection, Calibration, Normalization, Selection, Converting to real, Reshaping, Reversing order of freq:
        dtype:
        s_select: either a function that selects or a tuple of indices
        measurement: dictionary with two keys: 'data', 'freq'.
        cal: if dictionary is provided, it automatically means: CALIBRATE
        freq_select: list of frequency **values** in GHz. If provided: automatically means: select those frequencies
        norm_factor: if provided, there will be element-wise multiplication
        shape: selecting appropriate S params, putting them in right shape, treat complex values, and reshape it.
        :param clone: act on provided data or never mess with their values and create new results from them.
        :return: prepared data.
        """
        """Num of frequency channels: We will stack real and imaginary parts as extra channels. As such we will end up 
        with twice more channels than freq points. Num of freq points @ reduced sampling = 751 / 2 = 375. Thus, 
        we will be having a total of 750 channels. To do grouping in the first layer, we need num of in_channels and 
        out_channels be divisible by group number. Let group number = 8. out_channels = 256 ==> good combo (divisible) 
        Now we need num of channels to be divisble by 8, and also when divided, further divisible by 2 so that we don't 
        split real and imaginary values of a single frequency into different channels. 92 is divisble by 2 | 92 * 8 = 
        736. | 736 / 2 = 368 | so, instead of 375, we will pick 368 freq points only. When 751 are captured, 
        the band [0.5, 2.0] is spanned in a step of 2 MHz. When 375 points are captured, the step is 4 MHz. The cutoff 
        frequncy of the array is 0.6 GHz, so if we start from there, we want the following frequency points: np.arange(
        0.5 + 0.004*7, 2.0, 0.004). It starts @ 0.528 till 2.0, and has a length of 368 as required.
        """
        self.hp = hp
        self.m_struct = m_struct
        self.cal_struct = cal_struct
        self.copy = clone

    def preprocess(self):

        # FIRSTLY: select frequencies #
        m_struct = self.m_struct.get_common_freq(other_f=self.hp.freq_select, clone=self.copy)
        if self.cal_struct:
            cal_struct = self.cal_struct.get_common_freq(other_f=self.hp.freq_select, clone=self.copy)
        else:
            cal_struct = None

        # Get the data from those S objects
        raw_data = m_struct.s
        if self.cal_struct:
            cal_data = cal_struct.s
        else:
            cal_data = None

        # SECONDLY: calibrate while data is still complex, then, normalize it
        result = (raw_data - cal_data) if self.hp.cal else raw_data

        # Thirdly, normalize while the data is still in raw shape, meaning no need to reshape norm factor
        if self.hp.norm_factor is None:
            pass  # no normalization factor
        else:
            result *= self.hp.norm_factor

        # Fourthly, S param selection:
        if type(self.hp.s_select) is tuple:
            result = result[..., self.hp.s_select[0], self.hp.s_select[1]]
        else:  # it is a function
            result = self.hp.s_select(result)

        # Fifthly: RealComplex
        result = Norm.complex_to_real(result, dtype=self.hp.dtype, axis=-1)

        # Sixthly: reshaping
        result = self.hp.shape(result)
        return result


class Norm:
    class Standardrizer:
        """Has the edge over Sklearn of supporting complex numbers."""
        def __init__(self, mode=Dtype.rect, axis=None):
            self.mode = mode
            self.mean = None
            self.abs_std = None
            self.re_std = None
            self.im_std = None
            self.axis = axis

        def fit(self, data):
            self.mean = data.mean(axis=self.axis, keepdims=True)
            if self.mode is Dtype.magnitude:
                self.abs_std = data.__abs__().std(axis=self.axis, keepdims=True)
            elif self.mode is Dtype.rect:
                self.re_std = data.real.std(axis=self.axis, keepdims=True)
                self.im_std = data.imag.std(axis=self.axis, keepdims=True)
            return self

        def transform(self, data):
            if self.mode is Dtype.rect:
                normed = (data.real - self.mean.real) / self.re_std + 1j * (data.imag - self.mean.imag) / self.im_std
                return normed
            elif self.mode is Dtype.magnitude:
                normed = (data - self.mean) / self.abs_std
                return normed

    @staticmethod
    def get_nf(freq, spatial_growth=0.01, spectral_growth=0., scale=1, verbose=False):
        """
        To understand the work, play with this: plt.plot(get_nf(freq_select, factor=100, decay=0.15, mag=1)[:, 0, :])

        :param verbose:
        :param scale: simple scalar multiplication with the norm factor.
        :param spatial_growth: determines how much decay in the exponential used to create spatial norm (per S).
        :param freq: ``skrf.Frequency`` class instance.
        :param spectral_growth: determines how much decay in the exponential used to create norm across frequencies.
        :return: normalization factor.
        """
        # ====================================== Spatial Scale =========================================================
        # path = tb.P('gen_dataset/antenna_location.txt').absolute_from(reference='deephead')
        # df = tb.pd.read_csv(path, header=None)
        # antenna_positions = df.to_numpy()
        antenna_positions = tb.P('gen_dataset/antenna_positions.npy').absolute_from("deephead").readit()[:, :2]

        ports = antenna_positions.shape[0]
        spatial_scale = np.zeros((ports, ports))
        for i in range(ports):
            for j in range(ports):
                distance = np.linalg.norm(antenna_positions[i] - antenna_positions[j])
                spatial_scale[i, j] = np.exp(distance * spatial_growth)  # till now it solely depends on S location.

        # ============================ Spectral Scale ==========================================================
        spectral_scale = np.exp(spectral_growth * freq.f[:, None, None] / 1e9)
        # note: one can stuff spatial_scale into the expression above to make s-dependent spectral_scale.

        norm_factor = spatial_scale * spectral_scale * scale

        if verbose:
            print('Max spatial multiplier = ', np.max(spatial_scale))
            print('Max spectral multiplier = ', np.max(spectral_scale))
            plt.plot(norm_factor[:, 0, 0])
            plt.figure()
            plt.imshow(norm_factor[0])
        return norm_factor

    @staticmethod
    def complex_to_real(x, dtype: Dtype = Dtype.polar, axis=-1, verbose=False):
        """ This function converts complex128/64 to float32

        :param axis: relevant only if `Dtype` is `recangular` or `polar`. Determines where the extra axis will be.
        :param dtype: see :py:class:`.Dtype`
        :param x: compelx array of any shape s, rank r
        :param verbose:
        :return: real array of rank r+1
        """
        if x.dtype != 'complex64' and x.dtype != 'complex128':
            if verbose:
                print(f"Warning: Input is not complex-valued")
            return x
        y = None
        if dtype is Dtype.rect:
            y = np.stack([np.real(x), np.imag(x)], axis=axis)
        elif dtype is Dtype.polar:
            y = np.stack([np.abs(x), np.angle(x) / 3], axis=axis)
        elif dtype is Dtype.real:
            y = np.real(x)
        elif dtype is Dtype.imag:
            y = np.imag(x)
        elif dtype is Dtype.magnitude:
            y = np.abs(x)
        elif dtype is Dtype.phase:
            y = np.angle(x) / 3
        elif dtype is Dtype.complex:
            y = x
        return y

    @staticmethod
    def db(x, absolute=True, clip=None):
        result = 20 * np.log10(abs(x)) if absolute else 20 * np.log10(x)
        if clip is not None:
            result = result.clip(min=clip)
        return result

    @staticmethod
    def linear(x):
        return 10**(x / 20)

    @staticmethod
    def add_uniform_noise(sparam, level_db=-70):
        linear_level = 10 ** (level_db / 20)
        noise = np.random.uniform(low=-linear_level, high=linear_level, size=sparam.shape)
        return sparam + noise


class IFFT:
    """Converting the signal to the time domain, doubles the length of the signal as the spectrum needs to be
    2 sided before inverted, but comes with the benefit of obtaining a real valued signal.
    """
    def __init__(self, freq):
        self.freq = freq
        self.df = self.freq.df[0]
        # padding the missing frequency steps from 0Hz to freq[0]
        self.missing_freq = np.arange(0, freq.f[0], self.df)
        self.full_freq = np.concatenate([self.missing_freq, self.freq.f])
        self.full_signal_2sided = None

    def __call__(self, signal, debug=False, parity='odd'):
        # padding zeros from 0Hz till freq[0] by df steps to signal
        missing_vals = np.zeros_like(self.missing_freq, dtype=signal.dtype)
        full_signal = np.concatenate([missing_vals, signal])
        # getting negative frequencies replica. Ignore the DC component (no replica for it).
        # Apropos the highest frequency, take or leave it based on parity (odd or even)
        if parity == "odd":
            self.full_signal_2sided = np.concatenate([full_signal, np.conj(np.flip(full_signal[1:]))])
        else:
            self.full_signal_2sided = np.concatenate([full_signal, np.conj(np.flip(full_signal[1:-1]))])
        time_domain = np.fft.ifft(self.full_signal_2sided, norm="ortho")  # ortho = /sqrt(N)
        if debug:
            assert time_domain.imag.__abs__().sum() < 1e-10
            plt.plot(time_domain.real)
        return time_domain.real

    @staticmethod
    def naive_ifft(signal, axis=None):  # only pad one zero (real DC component is required)
        dc = np.zeros_like(signal[axis])[None]
        two_sided = np.concatenate([dc, signal, np.conj(np.flip(signal, axis=axis))], axis=axis)
        time_domain = np.fft.ifft(two_sided, axis=axis, norm="ortho")  # ortho = /sqrt(N)
        return time_domain.real


class PCAArtist(tb.Artist):
    """Use in conjuction with VisibilityViewerAuto"""
    def __init__(self, *args, pca=None, **kwargs):
        self.pca = pca
        super(PCAArtist, self).__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.get_axes()
        transf = self.pca.transform(args[0][None, ...])
        recons = self.pca.inverse_transform(transf)
        self.ax[0].plot(args[0], label='Original')
        self.ax[0].plot(recons[0], label='Reconstructed')
        self.ax[0].legend()
        self.ax[0].set_title(f"reconstruction loss = {np.linalg.norm(args[0] - recons[0], axis=-1)}")

    @classmethod
    def evaluate_pca(cls, pca, signals):
        artist = cls(create_new_axes=True, pca=pca)
        return tb.VisibilityViewerAuto(data=[signals], artist=artist)


#%% ========================== Boundary kit =============================================


class Interp(enum.Enum):
    linear = 'linear interpolation'
    spline = 'spline interpolation'
    bezier = 'bezier interpolation'


def interpolate(points, interp: Interp = Interp.spline, num_pts=500, smoothness_sampler=1):
    """
    :param smoothness_sampler: steps at which points are subsamples before interpolaying to increase smoothness
    :param num_pts: number of points of interpolation
    :param points: two coloums matrix containing 2D points.
    :param interp: type of interpolation. In bezzier curve there's no gaurantee of touching points.
    :return: x, y interpolation of points given.
    """
    points = points[::smoothness_sampler]
    x = y = None
    if interp is Interp.spline:
        from scipy.interpolate import CubicSpline
        num = len(points)
        angles = np.linspace(0, 360 - 360 / num, num)
        angles = np.append(angles, 360)
        points = np.concatenate([points, points[:1]])
        cs = CubicSpline(angles, points, bc_type='periodic')
        result = cs(np.linspace(0, 360, num_pts))
        x, y = result[:, 0], result[:, 1]
    elif interp is Interp.bezier:
        import bezier
        xy = points.T  # Fortran-like array
        curve = bezier.Curve.from_nodes(np.asfortranarray(xy))
        result = curve.evaluate_multi(np.linspace(0.0, 1.0, num_pts - 1))  # because next we will add another point.
        # Since no concatenation was done behforehad, we do it now to close the circle:
        # closing the loop beforehand causes Bezier library to fail.
        result = np.concatenate([result, result[:, :1]], axis=1)
        x, y = result[0, :], result[1, :]
    if x is None or y is None:
        raise Warning('No processing took place at interpolation step')
    return np.stack([x, y], axis=1)


def get_xy_from_radii(rs):
    """ Converts radii to x, y, and process them so that the can be plotted on top of the image.

    :param rs:
    :return:
    """
    x, y = [], []
    num = len(rs)
    angles = np.linspace(0, 360 - 360 / num, num)
    for an_angle, an_r in zip(angles, rs):
        x.append(an_r * np.cos(an_angle * np.pi / 180))
        y.append(an_r * np.sin(an_angle * np.pi / 180))
    x = np.array(x)
    y = np.array(y)
    return np.stack([x, y], axis=1)


def get_normal_distances(xy_boundary, antenna_positions):
    """Converts xy boundary points to normal distances to antennas.

    .. warning:: this function assumes that `xy_boundary` is pre-aligned. Additionally, results depend on resolution of
       input curve.

    :param antenna_positions:
    :param xy_boundary:
    :return: normal distance from array points to the head boundary.
    """
    xy_boundary = xy_boundary[:-1]  # last point is repeated.
    pts_per_ant = round(len(xy_boundary) / len(antenna_positions))

    def func(norm_, point_):
        """
        :param norm_: must be a normalized vector
        :param point_: a matrix of row vectors
        :return: The difference between the orthogonal projection onto norm_ and point_
        """
        return np.dot(point_, norm_)[..., None] * norm_ - point_

    result = []
    intersection = []
    theta_points = (np.arctan2(*xy_boundary.T[::-1]) + 2 * np.pi) % (2 * np.pi)
    theta_ant = (np.arctan2(*antenna_positions[:, :2].T[::-1]) + 2 * np.pi) % (2 * np.pi)
    for antenna in range(len(antenna_positions)):
        center_idx = np.searchsorted(theta_points, theta_ant[antenna])
        lower = center_idx - pts_per_ant
        upper = center_idx + pts_per_ant
        candidate_points = tb.Manipulator.slicer(xy_boundary, slice(lower, upper))
        x, y, norm_ang = antenna_positions[antenna]
        point = np.array([x, y])
        norm = np.array([np.cos(np.deg2rad(norm_ang)), np.sin(np.deg2rad(norm_ang))])
        shift_vector = func(norm, [point])
        normal_vectors = func(norm, candidate_points)
        idx = np.argmin(np.linalg.norm(normal_vectors - shift_vector, axis=1))
        boundary_point = candidate_points[idx]
        result.append(np.linalg.norm(point - boundary_point))
        intersection.append(boundary_point)
    result = np.array(result)
    intersection = np.array(intersection)
    return result, intersection


def get_xy_from_normals(normal_distances, antenna_positions):
    result = np.zeros((len(normal_distances), 2), dtype='float32')
    for index in range(len(antenna_positions)):
        x, y, norm_ang = antenna_positions[index]
        distnace = normal_distances[index]
        point = np.array([x, y])
        norm = -1 * np.array([np.cos(np.deg2rad(norm_ang)), np.sin(np.deg2rad(norm_ang))])
        result[index] = point + distnace * norm
    return result


def allign_boundary(xy, x0=0, y0=0, theta_deg=0, num_pts=360):
    """Finding contour point @ theta = 0. This starting point is important for matching with other curve.

    :param theta_deg: angle from which you'd like the curve to start from.
    :param xy:
    :param x0: x coordinates of the origin
    :param y0: y coordinate of the origin
    :param num_pts:
    :return:
    """
    closed_flag = False
    if np.all(xy[0] == xy[-1]):  # this is a closed loop curve.
        xy = xy[:-1]  # remove the repeated point for now.
        closed_flag = True  # so that later this is reversed.
    xx, yy = xy.T
    theta = np.arctan2(yy - y0, xx - x0)
    zero_idx = np.argmin(np.abs(theta - np.deg2rad(theta_deg)))
    xx = np.roll(xx, -zero_idx)
    yy = np.roll(yy, -zero_idx)
    # We also want to make sure that xx, yy are rotating CCW
    if theta[5] > theta[10]:
        xx = np.flip(xx)
        yy = np.flip(yy)
    # Next, interpolate to cover the length of the curve uniformly, parametrically from 0 to 360
    if num_pts != len(xx):
        xy = interpolate(np.vstack([xx, yy]).T, num_pts=num_pts)  # this internally repeats the last point.
    else:
        xy = np.vstack([xx, yy]).T
        if closed_flag:  # add back the repeated point.
            xy = np.concatenate([xy, [xy[0]]], axis=0)
    return xy


class BoundaryPlotter:
    def __init__(self, inferred, ground_truth=None, plot_normals=False, plot_diff=False, new_fig=True,
                 template_image=None, template_specs=None, run=True, ax=None, **kwargs):
        """ Shows boundary.
        :param inferred: two-coloums matrix carrying xy points for inferred boundary.
        :param ground_truth:
        :param str name: to be added to plots and also for saving purpose.
        :param antenna_positions: A two columns, 16 rows matrix of antenna centers.
        :return:
        """
        self.ground_truth = ground_truth
        self.inferred = inferred
        self.kwargs = kwargs
        self.template_image = template_image
        self.template_specs = template_specs
        self.overlay_specs = None
        self.run = run
        self.plot_normals = plot_normals
        self.plot_diff = plot_diff

        if not ax:
            if new_fig:
                tmp = 'HeadBoundary_' + tb.get_time_stamp()
            else:
                tmp = 'HeadBoundary'
            self.fig = plt.figure(num=tmp, figsize=(14, 9))
            if plot_diff is False:
                self.bdry_axis = self.fig.subplots()
            else:
                self.bdry_axis, self.diff_axis = self.fig.subplots(1, 2)
        else:
            self.fig = ax.figure
            self.bdry_axis = ax

        self.inf_bdry = self.bdry_axis.plot([], [], label=inferred.legend, color='blue')[0]
        if ground_truth is not None:
            # the ground truth can also be added to the original figure
            self.gt_bdry = self.bdry_axis.plot([], [], label=ground_truth.legend, color='red', linewidth=1.0)[0]
        if plot_diff:
            # An error signal can be plotted because ground truth is available.
            self.err = self.diff_axis.plot([], label='Difference')[0]
            self.diff_axis.set_ylabel('Difference in Mms'), self.diff_axis.set_xlabel('Angle in degrees')
            self.diff_axis.legend()

        tmp = inferred.antenna_positions
        self.bdry_axis.scatter(tmp[:, 0], tmp[:, 1], label='Antenna Positions', s=5, color='red')
        xy = interpolate(points=tmp, interp=Interp.spline)  # Don't user Bezier for that.
        self.bdry_axis.plot(*xy.T, label='Boundary of DOI', linewidth=1, color='black')
        self.fig.set_facecolor('white')
        for i, (x, y) in enumerate(tmp[:, :2]):
            theta = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2)
            #       [0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5]
            r = r + [3, 3, 3, 3, 4, 4, 5, 7, 8, 8, 9, 8, 8, 8, 8, 5][i]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.bdry_axis.text(x, y, str(i), size=10)

        if self.template_image is not None:
            self.bdry_axis.autoscale(False)
            self.inf_tmp = self.bdry_axis.imshow([[0, 1], [1, 0]], origin='lower')
            # Warning, the line above could rescale the axis to go from 0 to 1 unless anything previously
            # plotted, like antenna positions.
        if self.plot_normals:
            self.inf_norms = []
            for ii in range(len(inferred.distances)):
                if ii == 0:
                    self.inf_norms.append(self.bdry_axis.plot([], [], color='green', label="Normals")[0])
                else:
                    self.inf_norms.append(self.bdry_axis.plot([], [], color='green')[0])

        self.bdry_axis.set_xticks(ticks=np.arange(-100, 100, 5), minor=True)
        self.bdry_axis.set_yticks(ticks=np.arange(-120, 120, 5), minor=True)
        self.bdry_axis.set_xticks(ticks=np.arange(-100, 100, 20), minor=False)
        self.bdry_axis.set_yticks(ticks=np.arange(-120, 120, 20), minor=False)
        self.bdry_axis.grid(which='minor', axis='x', alpha=0.5)
        self.bdry_axis.grid(which='major', axis='x', alpha=1)
        self.bdry_axis.grid(which='minor', axis='y', alpha=0.5)
        self.bdry_axis.grid(which='major', axis='y', alpha=1)
        self.bdry_axis.set_ylabel('Mms'), self.bdry_axis.set_xlabel('Mms')
        self.bdry_axis.axis('square'), self.bdry_axis.legend(fontsize=8, loc='upper right')
        if self.run:
            _ = self.run_animate()

    def run_animate(self):
        return self.animate((self.inferred, self.ground_truth, self.inferred.name))

    def animate(self, data, **kwargs):
        """This method is prohibited from creating new axes or objects. It can only update data of existing objects.
        """
        _ = kwargs
        inferred, ground_truth, name = data
        self.bdry_axis.set_title(name)
        if inferred.xy is not None:
            self.inf_bdry.set_data((inferred.xy[:, 0], inferred.xy[:, 1]))
        if ground_truth is not None:
            self.gt_bdry.set_data((ground_truth.xy[:, 0], ground_truth.xy[:, 1]))
        if self.plot_diff:
            diff = inferred.r - ground_truth.r
            self.err.set_data((np.arange(len(diff)), diff))
            self.diff_axis.xaxis.set_major_locator(plt.MultipleLocator(45))
            self.diff_axis.xaxis.set_minor_locator(plt.MultipleLocator(5))
            self.diff_axis.yaxis.set_major_locator(plt.MultipleLocator(1.0))
            self.diff_axis.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
            self.diff_axis.grid(which='minor', axis='x', color='green', linewidth=0.5, alpha=0.5)
            self.diff_axis.grid(which='minor', axis='y', color='green', linewidth=0.5, alpha=0.5)
            self.diff_axis.grid(which='major', axis='x', color='red', linewidth=0.5, alpha=1)
            self.diff_axis.grid(which='major', axis='y', color='red', linewidth=0.5, alpha=1)
            self.diff_axis.relim()
            self.diff_axis.autoscale()
            self.diff_axis.set_title('Difference for case ' + name)
        if self.template_image is not None:
            self.overlay_template(inferred.xy)
        if self.plot_normals:
            # plot numbers of antennas and highlight landing points:
            # num = [str(i) for i in range(len(result))]
            # plt.scatter(*intersection.T)
            # for i in range(len(num)):
            #     plt.text(*intersection[i], s=num[i], size=7)
            for ii in range(len(inferred.distances)):
                self.inf_norms[ii].set_data(np.vstack([inferred.landing_pts[ii],
                                                       inferred.antenna_positions[ii, :2]]).T)
        return self.fig,

    def close(self):
        plt.close(self.fig)

    def overlay_template(self, prediction, cmap='viridis'):
        """
        :param cmap:
        :param prediction: xy data of the boundary
        :return: Superimposes an image on the currently activated plot
        """
        prediction_im = prediction[:-1]
        # Remove the last point because it is repeated and was intended to close the loop of the boundary when plotting.
        # If not removed, it created problems with TPS.
        from collections import namedtuple
        Pixel = namedtuple('Pixel', ['row', 'col'])
        im_size = Pixel(*self.template_image.shape)
        center_pix = Pixel(im_size.row / 2, im_size.col / 2)
        dx, dy = self.template_specs['pixel_width'], self.template_specs['pixel_height']
        # Convert mms to pixel space of reference model.
        prediction_im = prediction_im / np.array([dx, dy]) + np.array([center_pix.col, center_pix.row])
        # Send the processed points to warp method
        template = Warpper.warp(self.template_image, target_points=prediction_im, plot=False,
                                threshold=self.template_specs['threshold'],
                                maxval=self.template_specs['maxval'])
        mask = template > 0
        from matplotlib import cm
        # normalizing the template [0-255] cause it could be segmented image before trying to colorize it.
        tmp = ((template - template.min()) / template.max() * 255).astype('uint8')
        col_template = np.uint8(cm.get_cmap(cmap)(tmp) * 255)[..., :-1]  # make it color image
        mask = mask[..., None]  # one advantage for channel first is natural broadcasting
        col_template = (col_template * mask + 255) * (1 - mask) + mask * col_template
        # template[template == 0] = 255  # so that we can make the background white (works if the template is gray)
        # Figure out the physical size of the template returned, so that we can plot it on top of the boundaries.
        size_y_mm, size_x_mm = im_size.row * dy, im_size.col * dx  # that finishes the scale issue
        # Next, where to put it shuch that the center pixel goes to (0, 0) point on the plot?
        bottom_left_corner_mm = -center_pix.col * dx, -center_pix.row * dy
        top_right_corner_mm = bottom_left_corner_mm[0] + size_x_mm, bottom_left_corner_mm[1] + size_y_mm
        xmin, xmax = bottom_left_corner_mm[0], top_right_corner_mm[0]
        ymin, ymax = bottom_left_corner_mm[1], top_right_corner_mm[1]
        self.inf_tmp.set_data(col_template)
        self.inf_tmp.set_extent([xmin, xmax, ymin, ymax])
        self.bdry_axis.set_xlabel(str(self.template_specs))
        self.overlay_specs = tb.Struct(warpped_template=template, colored_template=col_template,
                                       extent=[xmin, xmax, ymin, ymax])


class PlotBoundaryArtist(tb.Artist):
    """A wrapper around PlotBoundary to make it behave as AutoViewer expects.
    """
    def __init__(self, **kwargs):
        super(PlotBoundaryArtist, self).__init__(create_new_axes=True, **kwargs)
        self.plotter = None
        self.kwargs = kwargs

    def plot(self, *boundary_descriptor, **kwargs):
        self.get_axes()
        self.plotter = BoundaryPlotter(*boundary_descriptor,
                                       ax=self.ax[0], **self.kwargs)


class Smoother:
    """
    Smooth the boundaries using KNN with dataset already built 1632.
    """
    def __init__(self, test=False):
        self.ip = self.op = self.match_index = self.match_boundary = self.plotter = None
        self.thresh = 5
        self.criterion_choice = 1

        if test:
            self.test_files = tb.P(r'experimental/Ali_Boundary_Clinical1').absolute_from("deephead").glob('*')
            xy_container = []
            normals_container = []
            self.names = []
            for file in self.test_files:
                xy = np.genfromtxt(file)
                xy_container.append(xy)
                self.names.append(file.stem)
                xy = allign_boundary(xy, num_pts=361)
                normals, _ = get_normal_distances(xy, self.antenna_positions)
                normals_container.append(normals)
            self.test_xy = np.array(xy_container)
            self.test_normals = np.array(normals_container)

    @property
    def boundary_bank(self):
        return np.load(st.d / r'datasets/s_image/boundaries.npy')

    @property
    def normals_bank(self):
        return np.load(st.d / r'datasets/s_image/normals.npy')

    @property
    def antenna_positions(self):
        return np.load(tb.P(r'gen_dataset/antenna_positions.npy').absolute_from("deephead"))

    @property
    def criterion(self):
        return [lambda x, y: (x + y)/2, lambda x, y: y][self.criterion_choice]

    def smooth(self, b_desc, external_flags=None, verbose=False):
        self.ip = b_desc
        if self.ip.distances is None:
            self.ip.distances, _ = get_normal_distances(self.ip.xy, self.antenna_positions)

        predicted = b_desc.distances.copy()
        diff = self.normals_bank - predicted
        self.match_index = np.argmin(np.linalg.norm(diff, axis=-1))
        best_fit = self.normals_bank[self.match_index]
        abs_diff_per_normal = np.abs(best_fit - predicted)
        if verbose:
            print(abs_diff_per_normal)

        for i, point in enumerate(abs_diff_per_normal):
            if external_flags is None and point > self.thresh:
                predicted[i] = self.criterion(predicted[i], best_fit[i])
            if external_flags is not None and external_flags[i]:
                predicted[i] = self.criterion(predicted[i], best_fit[i])
        landing_pts = get_xy_from_normals(predicted, self.antenna_positions)
        xy_boundary = interpolate(landing_pts)
        xy_boundary = allign_boundary(xy_boundary)
        self.op = BoundaryDescriptor(xy=xy_boundary, match_indices=self.match_index,
                                     legend="Alex's Smoothed version")
        self.match_boundary = BoundaryDescriptor(xy=self.boundary_bank[self.match_index],
                                                 legend="Matched item from bank")

    def plot_result(self, show_match=True):
        self.plotter = BoundaryPlotter(inferred=self.ip)
        self.plotter.bdry_axis.plot(*self.op.xy.T, color='red', label=self.op.legend)
        if show_match:
            self.plotter.bdry_axis.plot(*self.match_boundary.xy.T, color='green', label=self.match_boundary.legend)
        self.plotter.bdry_axis.legend()

    def test(self, idx=0, verbose=False):
        self.ip = BoundaryDescriptor(xy=self.test_xy[idx], match_indices=idx, legend="Ali's Prediction",
                                     name=self.names[idx], distances=self.test_normals[idx])
        self.smooth(self.ip, verbose=verbose)
        self.plot_result()

    def boundaries_to_normals(self):
        normals_bank = []
        for datum in self.boundary_bank:
            normals, _ = get_normal_distances(datum, self.antenna_positions)
            normals_bank.append(normals)
        normals_bank = np.array(normals_bank)
        np.save(st.d / r'datasets/s_image/normals.npy', normals_bank)

    @staticmethod
    def get_erroneous_estimations(distances):
        """ This is not based on the overall shape, but rather on comparing adjacent distances.
        :param distances: postprocessed!
        :return: indices of distances that are egregiously wrong.
        """
        thresh = 1.5 * 2.5
        prev, nxt = np.roll(distances, 1), np.roll(distances, -1)
        avg = (prev + nxt) / 2
        delta1 = abs(distances - prev) > thresh
        delta2 = abs(distances - nxt) > thresh
        # each distance is judged by its neighbours and gets two flags from above.
        result = np.logical_and(delta1, delta2)  # both neighbours indicate this is erroneuous
        corrected = distances.copy()
        corrected[result] = avg[result]
        return result, corrected

    def get_uncertain_estimations(self, distances, stds):
        _ = self
        thresh = 0.6
        indices = np.where(stds > thresh)
        return distances, indices


class BoundaryDescriptor(tb.Base):
    def __init__(self, xy=None, distances=None, landing_pts=None, position=None, match_indices=None, name='Slim Shady',
                 legend='Prediction', position_specs=None, *args, **kwargs):
        """
        :param xy: points representing head boundary
        :param distances: The normals form antenna pitch to the head boundary
        :param landing_pts: intersection points of the normals with the xy head boundary.
        :param position:
        :param match_indices:
        :param name:
        :param position_specs: yaw and height_mm
        """
        super().__init__(*args, **kwargs)
        self.xy = xy
        self.distances = distances
        self.landing_pts = landing_pts
        self.position = position
        self.match_indices = match_indices
        self.name = name  # case name
        self.legend = legend  # prediction name, or algorithm name, version name, etc.
        self.plotter = None
        # self.fig = None
        self.ref_model = None
        self.position_specs = position_specs
        self.rotation = self.yaw = None
        self.smoother = None
        self.antenna_positions = tb.P('gen_dataset/antenna_positions.npy').absolute_from("deephead").readit(np.load)
        self.flags = None

    def smooth(self, plot=True, external_flags=None, **kwargs):
        if self.smoother is None:
            self.smoother = Smoother()
        self.smoother.smooth(self, external_flags=external_flags)
        if plot:
            self.smoother.plot_result(**kwargs)

    def fix_erroneous_distances(self, verbose=True):
        b_desc = self.__deepcopy__()
        flags, corrected = Smoother.get_erroneous_estimations(self.distances)
        b_desc.flags = tb.List(flags).apply(bool).list
        b_desc.distances = corrected
        b_desc.landing_pts = get_xy_from_normals(b_desc.distances, b_desc.antenna_positions)  # update xy
        b_desc.xy = interpolate(b_desc.landing_pts, num_pts=361)
        b_desc.raw_xy = self.xy
        if verbose:
            # where = np.argwhere(flags == True)
            print(f"Estimations fixed in antennas {flags}")
        return b_desc

    def __deepcopy__(self, *args, **kwargs):
        self.plotter = None  # important before copying because Matplotlib objects cannot be copied.
        # the reason is because of cross reference between children and parents, creatinga  complex graph.
        return super(BoundaryDescriptor, self).__deepcopy__()

    def reflect(self, ref='right', plot=True):
        if self.distances is None:
            self.distances, self.landing_pts = get_normal_distances(self.xy, self.antenna_positions)
        b_desc = self.__deepcopy__()
        b_desc.legend = ref + " reflected"
        if ref == 'right':
            b_desc.distances[8:] = np.flip(self.distances[:8])
        else:
            b_desc.distances[:8] = np.flip(self.distances[8:])
        b_desc.landing_pts = get_xy_from_normals(b_desc.distances, b_desc.antenna_positions)  # update xy
        b_desc.xy = interpolate(b_desc.landing_pts, num_pts=361)
        if plot:
            b_desc.plot(self)
        return b_desc

    def plot(self, ground_truth=None, plot_normals=True, ax=None, legend=None, show_orientation=False,
             **kwargs):
        if self.distances is None and plot_normals is True:
            plot_normals = False
        self.plotter = BoundaryPlotter(inferred=self, ground_truth=ground_truth, plot_normals=plot_normals,
                                       ax=ax, legend=legend, **kwargs)
        if show_orientation:
            self.plotter.bdry_axis.scatter(*self.xy[0], color='r')
            self.plotter.bdry_axis.scatter(*self.xy[50], color='r')

    def estimate_rotation(self, plot=False):
        from skimage.measure import EllipseModel
        ell = EllipseModel()
        ell.estimate(self.xy)
        from collections import namedtuple
        Rot = namedtuple('EllipseFit', ['xc', 'yc', 'a', 'b', 'theta'])
        self.rotation = Rot(*ell.params)
        if self.rotation.a > self.rotation.b:
            self.yaw = np.rad2deg(-np.pi / 2 + self.rotation.theta)
            print(f"YAW = {self.yaw}")
        else:
            self.yaw = np.rad2deg(self.rotation.theta)
            print(f"yaw = {self.yaw}")
        if plot:
            xy = EllipseModel().predict_xy(t=np.linspace(0, 2 * np.pi, 100), params=ell.params)
            self.plot()
            self.plotter.bdry_axis.plot(*xy.T, label='Fit')

    def plot_plain_xy(self, path=tb.P.tmp() / "plain_boundary.png", color='red', debug=False, ax=None):
        """save plain bounary image (useful for superimposition with inkscape).
         We want to explicitly mention everything so that results are
        independent of screen resolution, especially for this graph.

        * figsize: determines PHYSICAL screen size in inches.
        * dpi: determines the resolution, how many dots / pixels per inch. Thus, depending on screen size
            and resolution, this means having different window size for fixed dpi.
        * default values for those params seem to rely on screen used, so its better to set it explicitly
        * large figsize with "square" axis could mean that padded margins on one axis is not reasonable.
        * that said, those can be accounted for at save time.
        * In fact, even dpi can be set to different value at save time.
        * we need to adjust those at save time in case we wanted the plot to reflect true mm scale
          of what is plotted inside which is data-dependent.
        * if two circles of radii 10 and 20 are plotted and printed, they would look same size. The reason is that
          figures has same sizes. Changing dpi would only change resolution of the circles but not sizes.
          Only figure size can change the real life size of the circle. Dpi is stored in the metadata
          along with resolution, they determine the real life size. Resolution alone (size in pixels) doesn't settle it.
        """
        if ax is None:
            fig, ax = tb.plt.subplots(figsize=(12, 14), dpi=150)
            self.plotter = fig
        xy = self.xy.copy()
        xy = xy - (xy.max(axis=0) + xy.min(axis=0)) / 2  # center the shape.
        ax.plot(*xy.T, color=color)
        ax.axis('square')
        ax.set_xlim([-100, 100])
        ax.set_ylim([-120, 120])
        if not debug:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        tb.FigureManager.set_ax_to_real_life_size(ax)
        ax.figure.savefig(path, transparent=True, dpi=150, bbox_inches="tight", pad_inches=0.0)
        if not debug:
            tb.plt.close(ax.figure)

    @staticmethod
    def save_grid(direc=None):
        if direc is None:
            direc = tb.P.tmp()
        fig, ax = plt.subplots(figsize=(15, 15), dpi=150)  # start with large figure size, otherwise small one
        ax.axis('square')  # create troubles with the grids.
        ax.set_xlim(-100, 100)
        ax.set_ylim(-120, 120)
        ax.set_xlabel("Mms")
        ax.set_ylabel("Mms")
        tb.FigureManager.grid(ax, factor=5, x_or_y='x')
        tb.FigureManager.grid(ax, factor=10, x_or_y='y')
        tb.FigureManager.set_ax_to_real_life_size(ax)
        fig.savefig(direc / "real_life_size_grid.png", transparent=True, dpi=150, bbox_inches="tight", pad_inches=0.0)
        return ax

    def evaluate(self, other_estimation):
        """
        The following are metrics that are invariant under rigid shape transformations.
        Area / length method: computes the difference in area between two estimations.
        These measures are not sound as there could be a case where the measure is zero yet, the shapes are different.
        self is assumed to be the ground truth and other_estimation is the prediction.
        """
        if other_estimation is None:
            return None
        decimals = 4

        other_area = ms.polygon_area(other_estimation.xy + (-10, 50))
        gt_area = ms.polygon_area(self.xy)
        area_diff = round(abs(other_area - gt_area) / gt_area, decimals)

        def arclength(xy):
            next_ = np.roll(xy, -1, axis=0)
            lengths = np.linalg.norm(xy - next_, axis=-1)
            return lengths.sum()

        other_length = arclength(other_estimation.xy + (-10, 50))  # shift do not affect the result.
        gt_length = arclength(self.xy)
        len_diff = round(abs(other_length - gt_length) / gt_length, decimals)

        # For matchShape to work, it needs contours (image space objects) not xy curves.
        self.plot_plain_xy(path=tb.P.tmp().joinpath("1.png"))
        other_estimation.plot_plain_xy(path=tb.P.tmp().joinpath("2.png"))
        im_a = cv.imread(tb.P.tmp().joinpath("1.png").string, cv.IMREAD_GRAYSCALE)
        im_b = cv.imread(tb.P.tmp().joinpath("2.png").string, cv.IMREAD_GRAYSCALE)
        tb.P.tmp().joinpath("1.png").delete(are_you_sure=True)
        tb.P.tmp().joinpath("2.png").delete(are_you_sure=True)
        xy_a = Warpper.get_boundary(255 - im_a, align=False)  # convert the image from being negative to positive.
        xy_b = Warpper.get_boundary(255 - im_b, align=False)  # convert the image from being negative to positive.
        hu_diff = cv.matchShapes(xy_a, xy_b, 1, 0)  # invariant to + (-10, 10)

        evaluation = tb.Struct(len=len_diff, area=area_diff, hu=round(hu_diff, decimals), name=self.name)

        # one could also estimate best RigidTransf then find square error difference.
        # m, _ = cv.estimateAffinePartial2D(xy_a, xy_b[:len(xy_a)])
        # q = np.dot(xy_a, m[:, :2].T) + m[np.newaxis, :, 2]
        # tb.plt.plot(*xy_a.T, color='blue', label="curve 1")
        # tb.plt.plot(*xy_b.T, color='red', label="curve 2")
        # tb.plt.plot(*q.T, color='orange', label="curve 1 matched to 2")
        # plt.axis('square')
        # plt.legend()
        # alternatively, template matching uses correlation to determine the (x, y) translation
        # rotation must be accounted for on top of it by manually rotating.
        # it requires that the template is much smaller than the target image.
        return evaluation


class Warpper:
    @staticmethod
    def stack_images(scale, img_array):
        """
        An auxiliary function to help view multiple images in one window.
        """
        rows = len(img_array)
        cols = len(img_array[0])
        rows_available = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rows_available:
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv.resize(img_array[x][y], (img_array[0][0].shape[1],
                                                                      img_array[0][0].shape[0]), None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
            image_blank = np.zeros((height, width, 3), np.uint8)
            hor = [image_blank] * rows
            # hor_con = [imageBlank ] *rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1],
                                                            img_array[0].shape[0]), None, scale, scale)
                if len(img_array[x].shape) == 2:
                    img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
            hor = np.hstack(img_array)
            ver = hor
        return ver

    def test_binarization_threshold(self, im_):
        def empty(*args):
            _ = args
            pass

        cv2 = cv
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 640, 240)
        cv2.createTrackbar("Threshold1", "Parameters", 0, 255, empty)
        cv2.createTrackbar("Threshold2", "Parameters", 0, 255, empty)

        while True:
            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")  # actual threshold
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")  # max value
            _, thresh = cv.threshold(im_, threshold1, threshold2, 0)  # threshold, max_val, replace_number
            # APPROX_NONE gives all the points. The other option gives only a certain number of points.

            # findContours function takes in a binary image. You can get this either by thresholding your gray scale
            # image (meaning if it was colored you need to gray it first). Or, you can go one step further and
            # provide an image of edges instead of any binary image. For that, you need to use: imgCanny = cv2.Canny(
            # imgGray, threshold1, threshold2)  # get edges kernel = np.ones((5, 5)) imgDil = cv2.dilate(imgCanny,
            # kernel, iterations=1)  # fix the edges, make them more definite by increasing width.

            contours_, hierarchy_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # RETR_EXTERANL only gives external contours. TREE version gives everything.
            # contours returned by the function is just a list of points. There are tools to handle those points.
            # for example:
            # gap_free_mask = cv.fillPoly(thresh, pts=contours_, color=1)
            # im_ *= gap_free_mask
            over_lay = cv.cvtColor(src=im_, code=cv.COLOR_GRAY2BGR)
            cv.drawContours(image=over_lay, contours=contours_, contourIdx=-1, color=(0, 0, 255), thickness=2)
            # cv.imshow(winname="Contour overlayying origional image", mat=over_lay)
            # But we are going to grab those points and use them in our own way.
            stacked = self.stack_images(0.9, ([im_, thresh, over_lay],))
            cv.imshow("Original Gray - Binarized Image - Original overlayed with red boundary from binarized", stacked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def get_boundary(image, theta=0, threshold=49, maxval=255, plot=False, num_pts=360, align=True):
        """Take a slice from self.data model and extract its boundary.
        :param maxval:
        :param theta: angle from which the curve starts.
        :param image:
        :param num_pts: number of points in the boundary returned.
        :param threshold:
        :param plot:
        :param align: returns a contour (image space object) or xy space vars with fixed length.
        :return:
        """
        ret, binarized = cv.threshold(image, threshold, maxval, 0)  # image, threshold, max_val, replace_number
        contours, hierarchy = cv.findContours(binarized.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        gap_free_mask = cv.fillPoly(binarized, pts=contours, color=1)
        image *= gap_free_mask
        from alexlib.miscellaneous import polygon_area
        idx = np.argmax([polygon_area(acontour.squeeze()) if len(acontour.squeeze().shape) > 1 else 0.
                         for acontour in contours])
        xy = contours[idx][:, 0, :]  # orignal shape was list[NumPoints x 1 x 2]

        if align:
            xmax, ymax = xy.max(axis=0)
            xmin, ymin = xy.min(axis=0)
            xy = allign_boundary(xy, x0=(xmax + xmin) / 2, y0=(ymax + ymin) / 2, theta_deg=theta, num_pts=num_pts)
        # deleted: x0=image.shape[1] / 2, y0=image.shape[0] / 2
        if plot:
            _, ax = plt.subplots()
            ax.imshow(image, origin='lower')
            ax.plot(*xy.T, color='r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(*xy[0])
            ax.scatter(*xy[50])
        return xy

    @staticmethod
    def warp(source_image, target_points, theta=0, plot=True, **kwargs):
        """
        If target_points are living inside (466, 394). Then, everything will work seamlessly.
        Otherwise, the output image is not going to show the brain in the area required or it will show it cropped.
        The reason is that the output image by default is the same as source image which is (466, 394).
        To fix this, one can enlarge the source image by padding it on sides to make it look like the output image.
        However, the source points must be adjusted properly to reflect the area of interest in the reshaped source.

        There are two ways to introduce rotations. Either by changing starting angle of the target points, or,
        when extracting the boundary from source image, changing the starting angle for that closed curve.
        The default is to keep them aligned and both start from theta=0.

        :param theta: starting angle applied to boundary extracted from source image. Can be used to apply rotations.
        :param source_image:
        :param target_points: pixels arranged in two rows matrix enclosing the area to warp to.
        :param plot:
        :return: an image with the brain warped into the area enclosed by target points sent.
        """
        source_points = Warpper.get_boundary(image=source_image, theta=theta, num_pts=len(target_points), **kwargs)
        # source_image = cv.copyMakeBorder(source_image, 500, 500, 500, 500, cv.BORDER_CONSTANT, 0)
        sshape = source_points[None, ...]
        tshape = target_points[None, ...]
        tps = cv.createThinPlateSplineShapeTransformer()
        matches = list()
        for i in range(len(source_points)):
            matches.append(cv.DMatch(i, i, i))
        tps.estimateTransformation(tshape, sshape, matches)
        out_img = tps.warpImage(source_image)
        if plot:
            plt.figure()
            plt.imshow(out_img, origin='lower')
        return out_img

    @staticmethod
    def test_warpping(im, theta_start=0):
        # Generate circle points in the middle of an image and pass it to warp method.
        _ = Warpper.get_boundary(im, theta=theta_start, plot=True, num_pts=360)
        target_image = np.zeros_like(im)
        radius = target_image.shape[0] / 4
        r0, c0 = round(target_image.shape[0] / 2), round(target_image.shape[1] / 2)
        pts = np.zeros((360, 2))
        for theta in range(360):
            x, y = radius * np.cos(np.deg2rad(theta)), radius * np.sin(np.deg2rad(theta))
            pts[theta] = [x + c0, y + r0]
        pts = np.roll(pts, int(theta_start), axis=0)
        plt.figure()
        plt.imshow(target_image, origin='lower')
        plt.scatter(*pts.T)
        plt.scatter(*pts[0], color='r')
        plt.scatter(*pts[50], color='r')
        plt.title('The head is going to be warped into this circle')
        _ = Warpper.warp(im, target_points=pts, plot=True)

    @staticmethod
    def extract_boundary(path, show=True, save=False):
        """Reads image from path and gets its boundaries.
        :param save:
        :param path:
        :param show:
        :return:
        """
        image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        xy = Warpper.get_boundary(image)
        # the boundary returned is [x, y] but, it assumes that y starts from top and goes down (works well with images)
        xy[:, 1] = image.shape[0] - xy[:, 1]  # convert y's to normal y axis that increases in the up direction.
        xy_center = np.flip(np.array(image.shape)) / 2  # image center coordinates are unknown!
        xy -= xy_center
        if save:
            np.save(path.parent / "raw_xy_boundary.npy", xy)
        if show:
            plotter = tb.Artist(*xy.T, color='red')
            extent = [-xy_center[0], xy_center[0], -xy_center[1], xy_center[1]]  # left right bottom top
            plotter.ax[0].imshow(image, extent=extent)
            plotter.ax[0].set_xticks([])
            plotter.ax[0].set_yticks([])
            plotter.fig.savefig(path.parent / "boundary_image.png", pad_inches=0.0, bbox_inches='tight', dpi=150)
            plt.close(plotter.fig)
        return xy, image


if __name__ == '__main__':
    pass
