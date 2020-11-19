# %% [markdown]
# # Model Selection
#
# Base selection of regressors is performed by fitting multiple regressors
# without performing any parameter tuning, then comparing the resulting errors
# across functional groups. Models with lower errors will be marked for
# parameter tuning investigations.

# %%
import os
import sys
import json
import pickle
import math
import logging
from pathlib import Path
from IPython.display import display

from colorama import Fore, Style
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

# %% [markdown]
# ## Class and function definitions

# %%


class RegressionProfile:
    def __init__(self, load_path=None):
        self.error_dataframes = dict()
        if load_path:
            try:
                self.load(load_path)
            except FileNotFoundError:
                pass

    def __repr__(self):
        keys = self.error_dataframes.keys()
        if keys is not None:
            return f'RegressionProfile object with keys {list(keys)}'
        else:
            return 'Uninitialized RegressionProfile object'

    def add_results(self, name, error_dataframe):
        self.error_dataframes[name] = error_dataframe

    def save(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.error_dataframes, file, pickle.HIGHEST_PROTOCOL)

    def load(self, load_path):
        with open(load_path, 'rb') as file:
            self.error_dataframes = pickle.load(file)

    def summarize(self, name):
        try:
            df = self.error_dataframes[name]
        except KeyError:
            print(f'No error summary with key {name} was found.')
            return
        best_learners = df.idxmin()
        print(1 * '\n')
        print(Fore.YELLOW + name + '\n' + len(name) * '-')
        # print(len(name) * '-')
        print(Style.RESET_ALL)
        print('Best learners total by response:')
        display(best_learners.value_counts(), best_learners.sort_values())
        print('\n\nSorted by median RMS error (smallest to largest):')
        display(df.T.describe().T.sort_values(by=['50%']))
        print('\n\nRMS Errors:')
        display(df)
        print(2 * '\n')


# %% [markdown]
# ## Profiling the regressors
#
# First, we'll choose potential regressors to investigate. Early choices are
# linear, decision trees, as well as boosting and forest ensemble methods.

# ### Learner Selection

# %%
learners = (
    GradientBoostingRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=100),
    AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100),
    AdaBoostRegressor(LinearRegression(), n_estimators=100),
    DecisionTreeRegressor(),
    LinearRegression()
)

# %% [markdown]

# Next, we'll select a functional group to examine, and only load the necessary
# data.

# ### Functional group selection

# %%
func_groups = list(tkr_group_lut.keys())
func_groups

# %%
group = 'patella'

# %% [markdown]
# ### Loading the data
#
# We'll load a subset of the data containing the responses making up the chosen
# functional group. We'll also use a `RegressionProfile` object to allow
# persistent results.

# %%
shared_kwargs = dict(load_fcn=load_tkr, functional_group=group)
tkr_train = artifact.Results(**shared_kwargs, subset='train')
tkr_test = artifact.Results(**shared_kwargs, subset='test')
display(tkr_train.response_names[1:])

regr_profile_path = (
    Path.cwd().parent / 'models' / 'selection' / 'learner_profiles.pkl'
)
reg_prof = RegressionProfile(load_path=regr_profile_path)

# %% [markdown]
# ### Fitting and profiling

# If the profiling results from the selected functional group have been loaded,
# then the `force_search` flag will need to be set to `True` to overwrite the
# previous profiling session.

# %%
force_search = False

# %%
learner_names = [x.__str__().replace('()', '') for x in learners]
scaler = StandardScaler()
regr = artifact.Regressor(tkr_train, tkr_test, learners[0], scaler=scaler)
err_df = pd.DataFrame(index=learner_names)

saved_keys = reg_prof.error_dataframes.keys()
if (force_search) or (group not in saved_keys):
    resp_pbar = tqdm(regr.train_results.response_names, desc='Processing...')
    for resp in resp_pbar:
        if resp == 'time':
            continue
        resp_pbar.set_description(f'Processing {resp}')
        errs = np.zeros_like(learner_names, dtype=np.float)
        lrn_pbar = tqdm(learners, desc='Fitting...', leave=False)
        for idx, lrn in enumerate(lrn_pbar):
            desc = f'{learner_names[idx].replace("base_estimator=", "")}'
            lrn_pbar.set_description(desc)
            regr.learner = MultiOutputRegressor(lrn)
            y_pred = regr.fit(resp).predict()
            errs[idx] = regr.prediction_error
        err_df[resp] = errs
        lrn_pbar.close()
    resp_pbar.close()

    reg_prof.add_results(group, err_df)
    reg_prof.save(regr_profile_path)

# %% [markdown]
# ## Results

# %%
# reg_prof.summarize(group)
for key in reg_prof.error_dataframes.keys():
    reg_prof.summarize(key)
