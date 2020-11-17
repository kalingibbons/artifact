# %% [markdown]
# ## Data Description
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

# %%
import os
import sys
import math
import logging
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.io as spio
import sklearn
import statsmodels.api as sm
from statsmodels.formula.api import ols

# !%load_ext autoreload
# !%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'

import seaborn as sns
import pandas as pd

import artifact


# %%
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = (9, 5.5)

sns.set_context("poster")
sns.set(rc={'figure.figsize': (16, 9.)})
sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

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
# Load and describe the training data
drop_regex = [
    # r'^time$',
    r'(femfe|patthick|patml|patsi)',
    r'^\w{3}_[xyz]_\w{3,4}$',
    r'^post\w+',
    r'^v[ilm][2-6]_(disp|force)$',
    r'^v[lm]1_(disp|force)$',
    r'^vert_(disp|force)$',
    r'^flex_(force|rot)$',
    r'^ap_force$',
    r'^(vv|ie)_torque',
    r'^(ml|pcm|pcl|pol)_force$',  # Always zero
    r'^(lclp|lcl|pmc|lcla|mcla)_force$',  # Often zero and bad predict
    r'^(pom|alc|mcl|mclp)_force$'  # Often zero and fairly bad predict.
]


def import_matlab_data(matfilepath):
    data = spio.loadmat(matfilepath, squeeze_me=True)
    keys = list(data.keys())
    data = data[keys[-1]]
    columns = list(map(lambda x: x.lower(), data.dtype.names))
    old = ['femie', 'femvv', 'tibslope', 'tibie', 'tibvv', 'xn', 'ctf', 'ctm']
    new = ['fem_ie', 'fem_vv', 'tib_slope', 'tib_ie', 'tib_vv', 'cop_',
           'force_', 'torque_']
    for o, n in zip(old, new):
        columns = list(map(lambda x: x.replace(o, n), columns))

    data_df = pd.DataFrame(data)
    data_df.columns = columns
    return data_df


def drop_columns(data_df, regex_list):
    cols = data_df.columns
    needs_drop = np.any([cols.str.contains(x) for x in regex_list], axis=0)
    return data_df.drop(cols[needs_drop], axis='columns')


def remove_failed(response_series, df_list):
    try:
        len(df_list)
    except TypeError:
        df_list = [df_list]

    failed_idx = response_series.apply(lambda x: x.size == 0)
    new_df_list = np.full(len(df_list), np.nan, dtype='object')
    for idx, df in enumerate(df_list):
        new_df_list[idx] = df.loc[~failed_idx]

    if len(new_df_list) == 1:
        return new_df_list[0]
    else:
        return new_df_list


# Source paths
dirty_data_dir = Path.cwd().parent / 'data' / 'interim'
dirty_test_path = dirty_data_dir / 'test.mat'
dirty_train_path = dirty_data_dir / 'doe.mat'

# Import and clean data
# Test
dirty_test = import_matlab_data(dirty_test_path)
dirty_test = drop_columns(dirty_test, drop_regex)
clean_test = remove_failed(dirty_test.iloc[:, -1], [dirty_test])

# Train
dirty_train = import_matlab_data(dirty_test_path)
dirty_train = drop_columns(dirty_train, drop_regex)
clean_train = remove_failed(dirty_train.iloc[:, -1], [dirty_train])

# %%
# Destination paths
cleaned_dir = dirty_data_dir.parent / 'preprocessed'
cleaned_test_path = cleaned_dir / 'test.parquet'
cleaned_train_path = cleaned_dir / 'train.parquet'

# Save the cleaned data
clean_test.to_parquet(cleaned_test_path)
clean_train.to_parquet(cleaned_train_path)