

"""
This file regenerates all the plots in boundary_estimation paper.

Data required:
1- G drive Sparams (PA clinical data)
2- SSB Data summarized and stored on D:/ locally. Those come from Data that lives on G drive and can be regenerated
   using a script in gen_dataset module.
3- Neural net saved weights
4- MRIs of healthy volunteers.
"""

import numpy as np
import matplotlib.pyplot as plt
import resources.toolbox as tb
import resources.s_params as stb
import training.boundary_models.boundary_prediction_normal_estimation as ne
import training.boundary_models.boundary_estimation_dimred as dr
import training.boundary_models.decomposition as dec
import training.signal_nn.regression as reg
from training.boundary_models.boundary_estimation_dimred import HParams, MLP
import measurements as ms
import training.template_models.mricrogl as mricrogl

results_path = tb.P.tmp("paper_results")
data_path = tb.P.tmp().joinpath("pa_boundaries_estimated_by_nn")


#%% Latex Activation for paper plots (inefficient)
size = 20
tb.FigureManager.activate_latex(size)


#%% Shape to signal

"""Things to consider with this approach:
* How to handle frequencies and S params? it is 2D signal.
* Memory of RNN could caue problems as we do not expect Sparams to affect each other beyond 
  adjacent antennas. 
* This also makes it susciptible to mass-scale corruption of results because of a single bad measurement in
  one of the antennas.
"""
# generate a random 2D smooth curve
N = 100
theta = np.linspace(0, np.pi * 2, N)
gp = reg.GP(x=np.linspace(0, 1, N)[..., None], kernel=reg.k.RBF(0.1))
np.random.seed(53)
while True:
    z = gp.sample(plot=False)
    if abs(z[0] - z[-1]) < 0.05:
        break
r = z[:, 0] + 4  # so that the radius is always positive, otherwise the shape will resolve to multiple cardiacs
x = r * 5 * np.cos(theta) + len(theta) / 2
y = r * np.sin(theta)
fig, ax = plt.subplots(figsize=(10, 10.7))
ax.plot(x, y, linewidth=2, label="Boundary Geometry")  # shift the plot to the middle and
ax.plot(r - 20, linewidth=2, label="Signal-ized shape")  # plot it down the shape
ax.scatter(len(theta) / 2, 0, color='red', s=15, marker='s', label="Origin")

for idx in range(len(theta)):
    if idx % 3 == 0:
        xx, yy = x[idx], y[idx]  # base of arrow.
        # head of arrow @ (idx, r[idx] - 20) ==> dx, dy = blah blah
        ax.arrow(xx, yy, idx - xx, r[idx] - 20 - yy, linewidth=0.5)
        ax.arrow(xx, yy, len(theta) / 2 - xx, 0 - yy, linewidth=0.1)
        ax.scatter(xx, yy, color='black', s=6)
        ax.scatter(idx, r[idx] - 20, color='black', s=6)

tb.FigureManager.grid(ax)
ax.legend()
ax.set_yticks([])
ax.set_xlabel("Time index")
# ax.set_xticks([])
# ax.set_xticks()
fig.savefig(results_path / "signal.png", dpi=120, pad_inches=0.1, bbox_inches='tight')


#%% Generates radii.png

bd = stb.BoundaryDescriptor(name=None)
np.random.seed(53)
distances = np.random.random(16) * 80 - 20
bd.landing_pts = stb.get_xy_from_normals(distances, bd.antenna_positions)  # update xy
bd.xy = stb.interpolate(bd.landing_pts, num_pts=361)
bd.plot()

bd.plotter.bdry_axis.set_xlim(-10, 100)
bd.plotter.bdry_axis.set_ylim(-10, 120)
tb.List(bd.plotter.bdry_axis.texts[4:]).modify("x.remove()")
bd.plotter.bdry_axis.collections[0].set_sizes([10 for _ in range(16)])
tb.List(bd.plotter.bdry_axis.texts).modify(f"x.set_size({size})")

points = bd.xy[:90:7]
for i, point in enumerate(points):
    line = np.stack([point, [0, 0]])
    if i == 0:
        bd.plotter.bdry_axis.plot(*line.T, color='green', label="Radii")
    else:
        bd.plotter.bdry_axis.plot(*line.T, color='green')
bd.plotter.bdry_axis.legend()
bd.plotter.fig.savefig(results_path / "radii.png", dpi=120, layout='tight', pad_inches=0.1, bbox_inches='tight')


#%% normal.png:

bd = stb.BoundaryDescriptor(name=None)
np.random.seed(66)
distances = np.random.random(16) * 10 + 10
bd.distances = distances
bd.landing_pts = stb.get_xy_from_normals(distances, bd.antenna_positions)  # update xy
bd.xy = stb.interpolate(bd.landing_pts, num_pts=700)
bd.plot(plot_normals=False)

bd.plotter.bdry_axis.set_xlim(-10, 100)
bd.plotter.bdry_axis.set_ylim(-10, 120)
tb.List(bd.plotter.bdry_axis.texts[4:]).modify("x.remove()")
bd.plotter.bdry_axis.collections[0].set_sizes([10 for _ in range(16)])
tb.List(bd.plotter.bdry_axis.texts).modify(f"x.set_size({size})")

point = bd.xy[44]
line = np.stack([point, [0, 0]])
bd.plotter.bdry_axis.plot(*line.T, color='gray', label="Radii")
point = bd.xy[131]
line = np.stack([point, [0, 0]])
bd.plotter.bdry_axis.plot(*line.T, color='gray')

line = np.stack([bd.landing_pts[1], bd.antenna_positions[1][:2]])
bd.plotter.bdry_axis.plot(*line.T, color='green', label="Normal Distance")
line = np.stack([bd.landing_pts[3], bd.antenna_positions[3][:2]])
bd.plotter.bdry_axis.plot(*line.T, color='green')

bd.plotter.bdry_axis.scatter(*bd.landing_pts[1], color='black', label="Landing Point")
bd.plotter.bdry_axis.scatter(*bd.landing_pts[3], color='black')

plt.rc('legend', fontsize=size-10)
bd.plotter.bdry_axis.legend()
bd.plotter.fig.savefig(results_path / "normal.png", dpi=200, layout='tight', pad_inches=0.1, bbox_inches='tight',
                       transparent=True)


#%% scalo.png & scalo_animation.png

hp = ne.HParams()
d = ne.ScaloGramReader(hp)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].plot(d.s_raw[0, :, 0, 0].real, label=f"Real Part")
ax[0].plot(d.s_raw[0, :, 0, 0].imag, label=f"Imaginary Part")
ax[0].legend()
ax[0].grid()
ax[0].set_xlabel(f"(a) Reflection parameter over 751 frequencies")
ax[1].imshow(d.s[0, 0])
ax[1].set_xlabel(f"(b) Scalogram generated with Complex Gaussian kernel CWT")

cbar = fig.colorbar(ax[1].images[0], shrink=0.7, pad=0.05)
# tmp = ax[1].get_position().bounds
# shift = 0.2
# ax[1].set_position([tmp[0] + shift, tmp[1], tmp[2] + shift, tmp[3]])
# tmp = cbar.get_position().bounds
# cbar.set_position([tmp[0] + shift, tmp[1], tmp[2] + shift, tmp[3]])

fig.savefig(results_path / "scalo.png", dpi=120, layout='tight', pad_inches=0.1, bbox_inches='tight')

tmp = tb.List(d.names[::10]).apply(lambda qx, qy: qx + qy,
                                   lest=tb.List(d.labels[::10, 0]).apply(lambda qw: f". Distance = {qw:1.2f}"))
tb.ImShow(d.s[::10, 0, ...], sublabels=[tmp],
          save_dir=results_path, save_name="scalo_animation", save_type=tb.SaveType.GIFFileBasedAuto)
# TODO: replace this animation with one that includes S params as well.

#%% DMD animation, SWT & NLFT.

d = dec.AliDatasetHandler()
d.generate_ali_ds(save_dir=results_path)

p = tb.VisibilityViewerAuto(data=[d.s.get_ii().np[..., 0]], artist=dec.NsevArtist(transf='SWT'),
                            save_dir=results_path, save_name="swt", pause=False)
pp = tb.VisibilityViewerAuto(data=[d.s.get_ii().np[..., 0]], artist=dec.NsevArtist(),
                             save_dir=results_path, save_name="swt", pause=False)


#%% pca_recons.png

hp = dr.HParams()
hp.freq_select = None
d1 = dr.DimRedDataReader(hp, append_pa=False)  # augments SSB with PA data.
d2 = dr.DimRedDataReader(hp, append_pa=True)  # only loads SSB data
# In-sample signal coming form SSB training set
sparam0 = d1.s[0, :, :1].T
transf0 = d1.pca.transform(sparam0)
recons0 = d1.pca.inverse_transform(transf0)
loss0 = round(np.linalg.norm(sparam0[0] - recons0[0], axis=-1), 2)
# out of sample signal coming from PA
sparams1 = d2.pos[1].data_m.s[:, 4, 4][None, ...].__abs__()  # a known case for close object to antennas
transf1 = d1.pca.transform(sparams1)
recons1 = d1.pca.inverse_transform(transf1)
loss1 = round(np.linalg.norm(sparams1[0] - recons1[0], axis=-1), 2)
# out of sample signal coming from PA using the no padding PCA
transf2 = d2.pca.transform(sparams1)
recons2 = d2.pca.inverse_transform(transf2)
loss2 = round(np.linalg.norm(sparams1[0] - recons2[0], axis=-1), 2)
print(f"reconstruction errors {loss0}, {loss1}, {loss2}")

ssb_s = tb.Manipulator.merge_axes(d1.s, 0, 2)
loss_ssb1 = np.linalg.norm(d1.pca.inverse_transform(d1.pca.transform(ssb_s)) - ssb_s, axis=-1)
loss_ssb2 = np.linalg.norm(d2.pca.inverse_transform(d2.pca.transform(ssb_s)) - ssb_s, axis=-1)
pa_s = tb.Manipulator.merge_axes(d2.pos.data_m.get_ii().np.__abs__(), 0, 2)
loss_pa_1 = np.linalg.norm(d1.pca.inverse_transform(d1.pca.transform(pa_s)) - pa_s, axis=-1)
loss_pa_2 = np.linalg.norm(d2.pca.inverse_transform(d2.pca.transform(pa_s)) - pa_s, axis=-1)
print(f"Mean reconstructrion error {loss_pa_1.mean()}, {loss_pa_2.mean()}")

# ========================== Generating Plots ==========================================
tb.FigureManager.set_linestyles_and_markers_and_colors()
tb.FigureManager.activate_latex(size=25)
fig, ax = plt.subplots(2, 1, figsize=(12, 22))
# ax.plot(sparam0[0], label='In-sample signal', linestyle='dotted', color='blue', linewidth=0.5)
# ax.plot(recons0[0], label='In-sample reconstruction', linestyle='dashed', color='blue', linewidth=0.5)
ax[0].plot(d1.freq, sparams1[0], label='Realistic signal', marker='')
ax[0].plot(d1.freq, recons1[0], label='Reconstruction 1', markersize=1)
ax[0].plot(d1.freq, recons2[0], label='Reconstruction 2', markersize=1.5)
ax[0].legend()
ax[0].set_xlabel("Frequency in GHz")
ax[0].set_ylabel("Reflection Coefficient Value")
ax[1].boxplot([loss_ssb1, loss_ssb2, loss_pa_1, loss_pa_2], whis=10)
ax[1].set_ylabel("Square Reconstruction Error")
tb.FigureManager.grid(ax[0], factor=4)
tb.FigureManager.grid(ax[1], factor=4, x_or_y='y')
ax[1].set_xticklabels(["Phantom data error\n@ phantom data fit", "Phantom data error\n@ Phantom data + Real data fit",
                       "Real data error\n@ Phantom data fit", "Real data error\n@ Phantom data + Real data fit"],
                      rotation=70, size=25)

fig.savefig(results_path / "recons_pca.png", dpi=120, layout='tight', pad_inches=0.1, bbox_inches='tight')


#%% Load up the model

path = tb.P(r"saved_models\boundary_models\reduced_freq").relativity_transform("deephead")
_ = HParams
model = MLP.from_class_weights(path)  # model for boundary estimation


#%% compare_1.png and compare_2.png

src = ms.PA()  # PA data

cases = ["000006", "000008"]  # We publish those cases, for which there is result from Resonance shift.
names = ["compare_1.png", "compare_2.png"]
for case, name in zip(cases, names):
    pos = src.positions.find(case)
    model.predict_from_position(pos, viz=False)
    tb.FigureManager.activate_latex(15)
    boundaries = pos.predictions[-1].postprocessed.fix_erroneous_distances()
    boundaries.plot(pos.em_boundary)
    boundaries.plotter.fig.axes[0].set_title('')  # Paper should not include Patient details like session and what not.
    boundaries.plotter.fig.axes[0].legend(prop={'size': 12})
    boundaries.plotter.fig.savefig(results_path / name, dpi=120, pad_inches=0.1, bbox_inches='tight')
    result = boundaries.evaluate(pos.em_boundary)
    print(result.__dict__)
"""
{'length_diff': 9.69, 'area_diff': 6721, 'cv_diff': 0.3259,
 'name': 'Case 000006}
Estimations fixed in antennas []
{'length_diff': 9.98, 'area_diff': 8037, 'cv_diff': 0.360,
 'name': 'Case 000008}
# keep in mind, this is not comparison against ground truth.
"""


#%% Full case presented: Case 30.

pos = src.positions.find("000017")
pos.session.case.__delattr__("finished_flag")  # to allow repeating boundary extraction
model.predict_from_position(pos, viz=False)  # gettting model prediction
boundaries = pos.predictions[-1].postprocessed.fix_erroneous_distances()  # next the plots of the prediction:
res_path = results_path.joinpath(pos.path[0].split("output")[1].with_suffix("")).create()
boundaries.plot_plain_xy(res_path / f"_plain_boundary.png", color="red")  # for overlaying purpose.
ax = boundaries.save_grid(direc=res_path)
boundaries.plot_plain_xy(path=res_path / f"_plain_boundary_grid.png", debug=True, color="red", ax=ax)  # standalone
spt1 = mricrogl.MRICROGL.get_boundary_from_pos(pos=pos, res_path=results_path, color="blue")  # getting ground truth
diff_metrics = spt1.boundary.evaluate(boundaries)


#%% test perturbation

src = ms.HHT3P()  # HHT3 data
res_path = results_path.joinpath(f"{src.name}_boundaries_estimated_by_nn").create()

pos = src.positions[0]  # healthy volunteer 1
pos.session.case.slice_details = (0, [(0.4, 0, 65)])  # add the slices one by one to allow different colors.
spt1 = mricrogl.MRICROGL.get_boundary_from_pos(pos=pos, res_path=res_path, color="blue")
pos.session.case.slice_details = (0, [(0.4, 0, 70)])
pos.session.case.__delattr__("finished_flag")  # to allow repeating boundary extraction
spt2 = mricrogl.MRICROGL.get_boundary_from_pos(pos=pos, res_path=res_path, color="red")
result = spt1.boundary.evaluate(spt2.boundary)
"""
{'length_diff': 1.366,
 'area_diff': 122.007,
 'hu_diff': 0.002,
 'name': 'Slim Shady'}
"""

#%% Table.

path = data_path / "report.csv"
csv = tb.pd.read_csv(path)
# ignore the entries in which some estimations were fixed.
# ignore entries in which there is no ground truth, hence, no evaluation
# ignore the entries in which the ground truth is faulty, thus the evaluation is bad.
filtered = csv.sort_values(by="nn_hu")[(csv.num_fixed < 1) & tb.pd.notna(csv.nn_hu) & (csv.nn_hu < 1)]
# these cases still have repeated measurements from the same session.
# We only report results from at least different sessions
seen_session = set()
indices = []
for idx, name in enumerate(filtered.name):
    if tuple(name.split(" | ")[:2]) in seen_session:
        pass
    else:
        seen_session.add(tuple(name.split(" | ")[:2]))
        indices.append(idx)
filtered = filtered.iloc[indices]
# the cases before 6 have faulty antenna. Other cases have faulty ground truth. Those are excluded as follows:
table = filtered[:20][['nn_hu', 'nn_area', 'nn_len', 'sex']]
table['nn_hu'] = np.float32(100 * table['nn_hu'])
table['nn_area'] = np.float32(100 * table['nn_area'].to_numpy())
table['nn_len'] = np.float32(100 * table['nn_len'].to_numpy())

rename = {"sex": "Gender", "nn_hu": "100 * Hu-difference", "nn_area": "% Area Change",
          "nn_len": "% Length Change"}
table = table.sample(20).reset_index(drop=True).rename(columns=rename)  # shuffle the cases.
table.index.name = "Index"
table.loc['min'] = table.min()
table.loc['max'] = table.max()
table.loc['mean'] = table.mean()
table = round(table, 3)
table.describe()
"""
       1k * Hu-difference  % Area-difference  % Length-difference
count           20.000000        20.000000          20.000000
mean            11.600000      1207.300000          14.550000
std              9.201831      1181.828294          14.759386
min              0.000000        65.000000           0.000000
25%              5.000000       435.250000           3.750000
50%              7.000000       739.500000           8.500000
75%             17.750000      1391.000000          20.500000
max             32.000000      4471.000000          49.000000
"""


table.to_csv(results_path / "table.csv")


#%%  ranges of distances predicted by the model on real data as opposed to training set.

boundaries = data_path.myglob("*cal_boundary.npy", r=True).readit(reader=stb.BoundaryDescriptor.from_saved)
predictions = boundaries.distances.np

fig, ax = plt.subplots()
mngr = tb.FigureManager()

for idx, ant in enumerate(predictions.T):
    ax.scatter([idx + 0.3] * len(ant), ant, color=mngr.colors.next(), s=3)
    ax.boxplot(ant, positions=[idx], whis=10)
tb.FigureManager.grid(ax)


#%% Features that exhibit similar behaviour to resonance shift.

# import tsfel
# q = np.apply_along_axis(lambda x: tsfel.sum_abs_diff(x), axis=1, arr=a.s.s.np[:, :, 10, 10].__abs__())
# plt.plot(q, range(40))

