# %% [markdown]
# # Comprehensive Exam
#
# ## Coding Artifact
#
# Kalin Gibbons
#
# Nov 20, 2020

# ## Introduction
#
# ---
#
# Outcomes of total knee arthroplasty (TKA) are dependent on surgical
# technique, patient variability, and implant design. Poor surgical or design
# choices can lead to undesirable contact mechanics and joint kinematics,
# including poor joint alignment, instability, and reduced range of motion. Of
# these three factors, implant design and surgical alignment are within our
# control, and there is a need for robust implant designs that can accommodate
# variability within the patient population. One of the tools used to evaluate
# implant designs is simulation through finite element analysis (FEM), which
# offers considerable early design-stage speedups when compared to traditional
# prototyping and mechanical testing. Nevertheless, the usage of FEM predicates
# a considerable amount of software and engineering knowledge, and it can take
# a great deal of time, compute or otherwise, to generate and analyse results.
# Currently used hardware and software combinations can take anywhere from 4 to
# 24 hours to run a single simulation of one daily-living task, for a
# moderately complex knee model. A possible solution to this problem is to use
# the FEM results to train predictive machine learning regression models
# capable of quickly  iterating over a number of potential designs. Such models
# could be used to hone in on a subset of potential designs worthy of further
# investigation.

# ### Data Description
#
# This _training_ dataset is generated from simplified finite element models of
# a cruciate-sacrificing, post and cam driven knee implant performing a
# deep-knee-bend. The implant geometries and surgical alignments are
# parameterized by 13 predictor variables which were drawn using Latin
# hypercube sampling from a range of currently used manufacturer dimensions,
# and angles performed during successful surgeries. There were originally 15
# predictors for this dataset, but two were fixed at average values for this
# particular batch of simulations. For the test dataset, the same predictors
# were uniformly drawn across the ranges of potential values.


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

import artifact
from artifact.datasets import load_tkr


# %%
plt.rcParams['figure.figsize'] = (9, 5.5)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times New Roman'
final_run = False


# %% [markdown]
#
# ## Data Cleaning
#
# ---
#
# We can begin by loading the datasets from MATLAB binary files. Following that
# we can take a look at the predictor distributions within the training and
# test sets, and confirm that the Latin hypercube and uniform sampling
# successfully covered the design space.


# %%
# Load the test data
tkr_train = artifact.Results(load_fcn=load_tkr, subset='train')
tkr_test = artifact.Results(load_fcn=load_tkr, subset='test')

tkr_train.features.hist(figsize=(12, 11))
tkr_train.features.describe()


# %%
# Describe the test space
tkr_test.features.hist(figsize=(12, 11))
tkr_test.features.describe()


# %% [markdown]
#
# It looks like the LHS did a good job of covering all of the design space for
# our training set, but our histograms are looking less uniform for the testing
# set. The parameter for the offset of the trochlear groove in particular seem
# right-skewed. We can lean on domain knowledge to guess that a positive offset
# lead to more simulation failures, which we have already removed. This makes
# sense because your trochlear groove is a smooth asymmetrical surface that
# your patella (kneecap) rides in, and moving this in a way that compounds the
# effect of asymmetrical vasti muscle distribution would increase the
# likelihood of patellar dislocation during, making simulated patient fail to
# complete their deep-knee-bend. A negative offset most likely reduced the risk
# of these dislocations, while a positive offset had the opposite effect. We'll
# skip going back and checking the histograms before removing the empty rows;
# that's a problem for the FEA modeler.

# ### Ranking Feature Importances
#
# This dataset began with 15 features, but cam radius and femoral
# flexion-extension were removed after being found to be the least important.
# Let's check how the other features rank.


# %%
tkr_train.plot_feature_importances()
plt.show()


# %% [markdown]
#
# This dataset has been truncated to only include contact mechanics and joint
# loads response data, and we're seeing posterior femoral radius and tibial
# conformity ratios as being the most important, followed by internal-external
# alignment of the tibial insert. This makes sense because the majority of our
# contact mechanics response variables concern the center of pressure
# ordinates, and the area of contact between the femoral component and the
# plastic tibial spacer.
#
# Some of those features don't seem very important. Let's try running principal
# component analysis on this dataset to see how quickly we're capturing
# variance.


# %%
X = StandardScaler().fit_transform(tkr_train.features.values)
pca = PCA(n_components=tkr_train.features.columns.size)
pca.fit(X)
_, ax = artifact.pareto(pca.explained_variance_, cmap='viridis')
ax[0].set_ylabel('Variance Explained')
ax[0].set_xlabel('Principal Component')


# %%
# _, ax = artifact.pareto(imps[indices], cmap='magma', names=feats[indices])
# ax[0].set_ylabel('Importances')


# %% [markdown]
#
# Looks like each variable is contributing about the same amount to the overall
# variance of this feature data, but we've picked up about 80% of the
# importance by the sixth feature. We're performing regression instead of
# classification, so the importance plots holds more weight for our use case.
# We'll leave them in because I want to make a comparison to some earlier
# results from looking at this problem in 2017.

# ## Regression
#
# ---
#
# Now that we understand our data, let's see if we can predict the output
# values. We're going to skip splitting into a develop-validate-test set for
# this example project, again so I can compare to earlier results.


# %%
# Choose the response of interest
resp_name = 'lat_force_2'

# Get the data ready
X_train = tkr_train.features.to_numpy()
X_test = tkr_test.features.to_numpy()
y_train = tkr_train.collect_response(resp_name)
y_test = tkr_test.collect_response(resp_name)


# Scale the data
scaler = StandardScaler
transformer = scaler().fit(X_train)
X_train_scaled = transformer.transform(X_train)
X_test_scaled = transformer.transform(X_test)

# Select a model to investigate
names = ('grad_boost', 'rand_forest', 'adaboost_tree', 'adaboost_linear',
         'decision_tree', 'linear')
learners = (
    GradientBoostingRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=100),
    AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100),
    AdaBoostRegressor(LinearRegression(), n_estimators=100),
    DecisionTreeRegressor(),
    LinearRegression()
)
# errs = np.zeros_like(names, dtype=np.float)
# for idx, learner in enumerate(learners):
#     regr = MultiOutputRegressor(learner)
#     regr.fit(X_train_scaled, y_train)
#     y_pred = regr.predict(X_test_scaled)
#     errs[idx] = mean_squared_error(y_test, y_pred)

# results = pd.Series(errs, index=names)
# results


# %% [markdown]
#
# Looks like the ordinary linear regression had the smallest errors. Let's take
# a look at the plots and see how it does.


# %%
learner = LinearRegression()
regr = MultiOutputRegressor(learner)
regr.fit(X_train_scaled, y_train)
y_pred = regr.predict(X_test_scaled)


def smooth(array, window=15, poly=3):
    return signal.savgol_filter(array, window, poly)


# Plot the data from the training set, include 95% interval
def plot_response(x, *ys, ylabel=None, legend_labels=None, ax=plt.gca()):
    if legend_labels is None:
        legend_labels = [None] * len(ys)
    for y, lbl in zip(ys, legend_labels):
        ax.plot(x, smooth(y), label=lbl)

    ax.set_xlim((x.min(), x.max()))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=x.max()))
    ax.set_xlabel(r'% of Deep Knee Bend')
    ax.set_ylabel(resp_name.replace('_', ' ').title())


def plot_bounds(x, y_train, y_pred, ax=plt.gca()):
    avg = smooth(y_train.mean(axis=0))
    sd2 = 2 * y_train.std(axis=0)
    err = np.zeros(len(y_pred))
    ax.fill_between(
        x,
        smooth(avg - sd2),
        smooth(avg + sd2),
        color='r',
        alpha=0.3,
        label=r'$\pm 2\sigma$'
    )


# def create_plots(n_rows, n_cols, fig_dir):
#     n_plots = int(n_rows * n_cols)
#     splits = np.arange(0, len(y_test), n_plots)[1:]
#     fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 30))
#     combo = zip(np.array_split(y_test, splits), np.array_split(y_pred, splits))
#     fig_dir = Path.cwd().parent / 'models' / 'predictions'
#     fig_dir.mkdir(exist_ok=True)
#     pbar = tqdm(total=len(splits), desc='Plotting')
#     lbls = ['Simulated', 'Predicted']
#     ylbl = resp_name.replace('_', ' ').title()
#     for idx, (y_test_sp, y_pred_sp) in enumerate(combo):
#         [ax.clear() for ax in axs.ravel()]
#         for ax, y_t, y_p in zip(axs.ravel(), y_test_sp, y_pred_sp):
#             plot_bounds(tim, y_train, y_pred, ax=ax)
#             plot_response(tim, y_t, y_p, legend_labels=lbls, ylabel=ylbl, ax=ax)
#         ax.legend(loc='best')
#         resp_str = resp_name.replace('_', '-')
#         save_dir = fig_dir / resp_str
#         save_dir.mkdir(exist_ok=True)
#         save_path = save_dir / '-'.join((resp_str, str(idx)))
#         fig.savefig(save_path, bbox_inches='tight')
#         pbar.update(1)
#     plt.close(fig)


top_fig_dir = Path.cwd().parent / 'models' / 'predictions'
# create_plots(n_rows=3, n_cols=4, fig_dir=top_fig_dir)

tim = tkr_train.response['time'][0]
avg = smooth(y_train.mean(axis=0))
sd2 = 2 * y_train.std(axis=0)
err = np.zeros(len(y_pred))
n_cols = 3
n_rows = 4
n_plots = int(n_rows * n_cols)
splits = np.arange(0, len(y_test), n_plots)[1:]
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 30))
combo = zip(np.array_split(y_test, splits), np.array_split(y_pred, splits))
top_fig_dir = Path.cwd().parent / 'models' / 'predictions'
top_fig_dir.mkdir(exist_ok=True)
pbar = tqdm(total=len(splits))
lbls = ['Simulated', 'Predicted']
ylbl = resp_name.replace('_', ' ').title()
for idx, (y_test_sp, y_pred_sp) in enumerate(combo):
    [ax.clear() for ax in axs.ravel()]
    for ax, y_t, y_p in zip(axs.ravel(), y_test_sp, y_pred_sp):
        ax.clear()
        plot_bounds(tim, y_train, y_pred, ax=ax)
        plot_response(tim, y_t, y_p, legend_labels=lbls, ylabel=ylbl, ax=ax)
    ax.legend(loc='best')
    resp_str = resp_name.replace('_', '-')
    save_dir = top_fig_dir / resp_str
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / '-'.join((resp_str, str(idx)))
    fig.savefig(save_path, bbox_inches='tight')
    pbar.update(1)
plt.close(fig)
pbar.close()


# %%
view = artifact.plotting.ImageViewer(top_fig_dir)
view.show()
