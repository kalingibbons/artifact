# %%
import os
import sys
import math
import logging
from pathlib import Path

from IPython.display import display
import numpy as np
import scipy as sp
import scipy.io as spio
import sklearn
import statsmodels.api as sm
from statsmodels.formula.api import ols

import sklearn
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm

# !%load_ext autoreload
# !%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'

# import seaborn as sns
import pandas as pd

import artifact
from artifact.datasets import load_tkr, tkr_group_lut
from artifact.helpers import RegressionProfile, REGRESSION_PROFILE_PATH


# %%

plt.rcParams['figure.figsize'] = (9, 5.5)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times New Roman'

# sns.set_context("poster")
# sns.set(rc={'figure.figsize': (16, 9.)})
# sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]

# Next, we'll select a functional group to examine, and only load the necessary
# data.

# ### Functional group selection

# %%
func_groups = list(tkr_group_lut.keys())
func_groups

# %%
group = 'joint_loads'

# %% [markdown]
# ### Loading the data
#
# We'll load a subset of the data containing the responses making up the chosen
# functional group.

# %%
shared_kwargs = dict(load_fcn=load_tkr, functional_group=group)
tkr_train = artifact.Results(**shared_kwargs, subset='train')
tkr_test = artifact.Results(**shared_kwargs, subset='test')
display(tkr_train.response_names[1:])

reg_prof = RegressionProfile(load_path=REGRESSION_PROFILE_PATH)
reg_prof.summarize(group)


# %%
learner = LinearRegression()
lrn_name = type(learner).__name__
top_fig_dir = Path.cwd().parent / 'models' / 'predictions' / group / lrn_name
n_rows, n_cols = 4, 3
tim = tkr_train.response['time'][0]
scaler = StandardScaler()
regr = artifact.Regressor(tkr_train, tkr_test, learner, scaler=scaler)
# for resp_name in tkr_train.response_names:
#     if resp_name == 'time':
#         continue
#     artifact.create_plots(n_rows, n_cols, regr, resp_name, top_fig_dir)


# %%
view = artifact.plotting.ImageViewer(top_fig_dir)
view.show()
