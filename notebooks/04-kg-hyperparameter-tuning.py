# %% [markdown]
#  # Comprehensive Exam
#
#  ## Coding Artifact
#
#  Kalin Gibbons
#
#  Nov 20, 2020
#
#  > Note: A hyperparameter is a numerical or other measurable factor
#  responsible for some aspect of training a machine learning model, whose value
#  cannot be estimated from the data, unlike regular parameters which represent
#  inherent properties of the natural processes which generated data.
#
#  ## Hyperparameter Optimization
#
#  There are several python packages with automatic hyperparameter selection
#  algorithms. A relatively recent contribution which I find particularly easy
#  to use is [optuna](https://optuna.org/), which is detailed in this
#  [2019 paper](https://arxiv.org/abs/1907.10902). Optuna allows the user to
#  suggest ranges of values for parameters of various types, then utilizes a
#  parameter sampling algorithms to find an optimal set of hyperparameters. Some
#  of the sampling schemes available are:
#
#  * Grid Search
#  * Random
#  * Bayesian
#  * Evolutionary
#
# While the parameter suggestion schemes available are:
#
#  * Integers
#    * Linear step
#    * Logarithmic step
#  * Floats
#    * Logarithmic
#    * Uniform
#  * Categorical
#    * List
#
#  This notebook uses Optuna to implement hyperparameter tuning on a number of
#  ensemble algorithms.
#
#  ## Imports

# %%
import os
import sys
import math
import logging
from pathlib import Path

from IPython.display import display, clear_output
from colorama import Fore, Style
import numpy as np
import scipy as sp
import scipy.io as spio
import sklearn
import statsmodels.api as sm
from statsmodels.formula.api import ols

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
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

import optuna
from optuna.visualization import plot_optimization_history

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
#  Next, we'll select a functional group to examine, and only load the necessary
#  data.
#  ### Functional group selection

# %%
func_groups = list(tkr_group_lut.keys())
func_groups


# %%
group = 'joint_loads'

# %% [markdown]
#  ### Loading the data
#
#  We'll load a subset of the data containing the responses making up the chosen
#  functional group.

# %%
shared_kwargs = dict(load_fcn=load_tkr, functional_group=group)
tkr_train = artifact.Results(**shared_kwargs, subset='train')
tkr_test = artifact.Results(**shared_kwargs, subset='test')
display(tkr_train.response_names[1:])

reg_prof = RegressionProfile(load_path=REGRESSION_PROFILE_PATH)
reg_prof.summarize(group)

# %% [markdown]
# ### Creating the optimization study
#
# First we must define an objective function, which suggests the ranges of
# hyperparameters to be sampled. We can use switch-cases to optimize the machine
# learning algorithm itself, in addition to the hyperparameters.

# %%
learners = (
    # GradientBoostingRegressor(),
    # RandomForestRegressor(),
    # AdaBoostRegressor(DecisionTreeRegressor()),
    # AdaBoostRegressor(LinearRegression()),
    # DecisionTreeRegressor(),
    Ridge(),
    # AdaBoostRegressor()
)


def objective(trial, train, test, regressors):
    reg_strs = [r.__repr__() for r in regressors]
    regressor_name = trial.suggest_categorical('classifier', reg_strs)

    if regressor_name == 'GradientBoostingRegressor()':
        # learner_obj = GradientBoostingRegressor()
        pass

    elif regressor_name == 'RandomForestRegressor()':
        pass

    elif regressor_name == 'AdaBoostRegressor(base_estimator=DecisionTreeRegressor())':
        criterion = trial.suggest_categorical('criterion', [
            'mse', 'friedman_mse', 'mae', 'poisson'
        ])
        splitter = trial.suggest_categorical('splitter', ['best', 'random'])
        max_depth = trial.suggest_categorical('max_depth', [
             3, 4, 5
        ])
        min_samples_split = trial.suggest_categorical('min_samples_split', [
            2,
        ])
        min_samples_leaf = trial.suggest_uniform('min_samples_leaf', 0, 0.5)
        estimator = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        loss = trial.suggest_categorical('loss', [
            'linear', 'square', 'exponential'
        ])
        n_estimators = trial.suggest_categorical('n_estimators', [100])
        learner_obj = AdaBoostRegressor(
            estimator,
            n_estimators=n_estimators,
            loss=loss
        )
        cv = 7

    elif regressor_name == 'AdaBoostRegressor(base_estimator=LinearRegression())':
        loss = trial.suggest_categorical('loss', [
            'linear', 'square', 'exponential'
        ])
        n_estimators = trial.suggest_categorical('n_estimators', [100])
        learner_obj = AdaBoostRegressor(
            LinearRegression(),
            n_estimators=n_estimators,
            loss=loss
        )
        cv = 7

    elif regressor_name == 'DecisionTreeRegressor()':
        criterion = trial.suggest_categorical('criterion', [
            'mse', 'friedman_mse', 'mae', 'poisson'
        ])
        splitter = trial.suggest_categorical('splitter', ['best', 'random'])
        max_depth = trial.suggest_categorical('max_depth', [
            3, 4, 5
        ])
        min_samples_split = trial.suggest_categorical('min_samples_split', [
            2,
        ])
        min_samples_leaf = trial.suggest_uniform('min_samples_leaf', 0, 0.5)
        learner_obj = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        cv = 7

    elif regressor_name == 'Ridge()':
        # alpha = trial.suggest_loguniform('alpha', 1e-5, 10)
        alpha = trial.suggest_uniform('alpha', 4, 6)
        learner_obj = Ridge(alpha=alpha)
        cv = 7

    elif regressor_name == 'AdaBoostRegressorj()':
        pass

    else:
        pass

    regressor = artifact.Regressor(train,
                                   test,
                                   learner_obj,
                                   scaler=StandardScaler())
    scores = regressor.cross_val_score(n_jobs=-1, cv=cv)

    return scores.mean() * 100

# %% [markdown]
# ### Running the optimization
#
# Optuna will sample the parameters automatically, for a maximum number of trials
# specified.

# %%
study = optuna.create_study(direction='minimize')
study.optimize(
    lambda t: objective(t, tkr_train, tkr_test, learners),
    n_trials=50
)

# %%
plot_optimization_history(study).show()
print(study.best_trial)
print(Fore.YELLOW
      + f'\nBest trial\n  RMSE% = {study.best_value} \n  {study.best_params}')
print(Style.RESET_ALL)

# %% [markdown]
# ### Plotting the results from the optimization
#
# We can assign the hyperparameters selected by optuna, and plot the resulting joint mechanics.

# %%
learner_strs = [lrn.__repr__() for lrn in learners]
learner_dict = dict(zip(learner_strs, learners))
learner_kwargs = study.best_params.copy()
learner = learner_dict[learner_kwargs['classifier']]
learner_kwargs.pop('classifier')
learner.set_params(**learner_kwargs)


# %%
lrn_name = type(learner).__name__
try:
    lrn_name = '-'.join((lrn_name, type(learner.base_estimator).__name__))
except AttributeError:
    pass

top_fig_dir = Path.cwd().parent / 'models' / 'predictions'
save_dir = top_fig_dir / group / lrn_name
n_rows, n_cols = 4, 3
tim = tkr_train.response['time'][0]
scaler = StandardScaler()
regr = artifact.Regressor(tkr_train, tkr_test, learner, scaler=scaler)
for resp_name in tkr_train.response_names:
    if resp_name == 'time':
        continue
    artifact.create_plots(n_rows, n_cols, regr, resp_name, save_dir)
    clear_output(wait=True)


# %%
view = artifact.plotting.ImageViewer(top_fig_dir)
view.show()
