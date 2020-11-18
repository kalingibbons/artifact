# %%
import os
import sys
import math
import logging
from pathlib import Path
from joblib.externals.cloudpickle.cloudpickle import cell_set

import numpy as np
import scipy as sp
import scipy.io as spio
import statsmodels.api as sm
from statsmodels.formula.api import ols

import sklearn
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from artifact.datasets import load_tkr, func_group_lut


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

# %%
func_groups = func_group_lut.keys()
func_groups

# %%
# tkr_train = artifact.Results(load_fcn=load_tkr, subset='test')
# tkr_test = artifact.Results(load_fcn=load_tkr, subset='train')

test_feat, test_resp = artifact.datasets.load_tkr(
    subset='test',
    functional_group='patella'
)
