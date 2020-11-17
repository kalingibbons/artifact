# %%
import os
import sys
import math
import logging
from pathlib import Path

import numpy as np
import scipy as sp
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
sns.set_context("poster")
sns.set(rc={'figure.figsize': (16, 9.)})
sns.set_style("whitegrid")

import pandas as pd
pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# %%
from artifact import *

# %% [markdown]
# **PLEASE** save this file right now using the following naming convention:
# `NUMBER_FOR_SORTING-YOUR_INITIALS-SHORT_DESCRIPTION`, e.g.
# `1.0-fw-initial-data-exploration`. Use the number to order the file within
# the directory according to its usage.

# %%


