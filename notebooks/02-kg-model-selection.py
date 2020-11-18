# %%
import os
import sys
import json
import pickle
import math
import logging
from pathlib import Path

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

# %%


class RegressionProfile:
    def __init__(self, load_path=None):
        self.error_dataframes = dict()
        if load_path:
            try:
                self.load(load_path)
            except FileNotFoundError:
                pass

    def add_results(self, name, error_dataframe):
        self.error_dataframes[name] = error_dataframe

    def save(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.error_dataframes, file, pickle.HIGHEST_PROTOCOL)

    def load(self, load_path):
        with open(load_path, 'rb') as file:
            self.error_dataframes = pickle.load(file)


# %% [markdown]

# %%
func_groups = list(tkr_group_lut.keys())
func_groups

# %%
group = 'ligaments'
force_search = False

shared_kwargs = dict(load_fcn=load_tkr, functional_group=group)
tkr_train = artifact.Results(**shared_kwargs, subset='train')
tkr_test = artifact.Results(**shared_kwargs, subset='test')

# %%
regr_profile_path = (
    Path.cwd().parent / 'models' / 'selection' / 'learner_profiles.pkl'
)
reg_prof = RegressionProfile(load_path=regr_profile_path)

# %%
learners = (
    GradientBoostingRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=100),
    AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100),
    AdaBoostRegressor(LinearRegression(), n_estimators=100),
    DecisionTreeRegressor(),
    LinearRegression()
)

saved_keys = reg_prof.error_dataframes.keys()
if (not force_search) and (group not in saved_keys):
    names = [x.__str__().replace('()', '') for x in learners]
    scaler = StandardScaler()
    regr = artifact.Regressor(tkr_train, tkr_test, learners[0], scaler=scaler)
    err_df = pd.DataFrame(index=names)
    for name in regr.train_results.response_names:
        if name == 'time':
            continue
        errs = np.zeros_like(names, dtype=np.float)
        for idx, lrn in enumerate(learners):
            regr.learner = MultiOutputRegressor(lrn)
            y_pred = regr.fit(name).predict()
            errs[idx] = regr.prediction_error
        err_df[name] = errs

    reg_prof.add_results(group, err_df)
    reg_prof.save(regr_profile_path)

# %%
err_df = reg_prof.error_dataframes[group]
err_df

# %%
best_learners = err_df.idxmin()
best_learners

# %%
best_learners.value_counts()

# %%
best_learners.sort_values()


# %%
err_df.drop('DecisionTreeRegressor').idxmin().sort_values()