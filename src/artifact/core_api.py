from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor

from artifact.plotting import pareto


def select_by_regex(data_df, regex_list, axis=0, negate=False):
    """Index a dataframe using regex label matching.

    Args:
        data_df (pandas.DataFrame): the dataframe to be indexed.
        regex_list ([str]): A list of regular expressions used for pattern
        matching the index or columns of data_df.
        axis (int, optional): The axis to match against. Defaults to 0.
        negate (bool, optional): Flag to select the inverse of the regex
            match. Defaults to False.

    Returns:
        pandas.DataFrame: A copy of data_df containing only the matched index
            or columns (or with the negated match removed)
    """
    if axis == 0:
        labels = data_df.index
    elif axis == 1:
        labels = data_df.columns
    has_match = np.any([labels.str.contains(x) for x in regex_list], axis=0)
    if negate:
        has_match = ~has_match

    selected_labels = labels[~has_match]
    return data_df.drop(selected_labels, axis=axis), selected_labels


class Results:
    def __init__(self, *load_args, load_fcn=None, **load_kwargs):
        self._importances = None
        self._importances_std = None
        self._importances_indices = None
        if load_fcn is not None:
            self.features, self.response = load_fcn(*load_args, **load_kwargs)
            self.feature_names = list(self.features.columns)
            self.response_names = list(self.response.columns)
        else:
            self.features = None
            self.response = None
            self.feature_names = None
            self.response_names = None

    def describe_features(self):
        display(self.features.describe())
        axs = self.features.hist(figsize=(12, 11))
        return axs

    def _calc_importances(self):
        X = StandardScaler().fit_transform(self.features.values)
        feats = self.features.columns

        imps = self._importances
        if imps is None:
            imps = np.zeros_like(feats, dtype=np.float)
            imps_std = imps.copy()
            for col_name, col in self.response.iteritems():
                y = self.collect_response(col_name)
                forest = RandomForestRegressor().fit(X, y)
                imps = imps + forest.feature_importances_
                imps_std = imps_std + np.std(
                    [tree.feature_importances_ for tree in forest.estimators_],
                    axis=0
                )
            self._importances = imps / self.response.shape[1]
            self._importances_std = imps_std / self.response.shape[1]
            self._importances_indices = np.argsort(imps)[::-1]

    def plot_feature_importances(self, use_pareto=False):
        X = StandardScaler().fit_transform(self.features.values)
        if self._importances is None:
            self._calc_importances()

        imps = self._importances
        imps_std = self._importances_std
        indices = self._importances_indices

        if not use_pareto:
            fig = plt.gcf()
            ax = fig.add_subplot(1, 1, 1)
            # ax.set_title('Feature importances')
            ax.bar(range(X.shape[1]), imps[indices], yerr=imps_std[indices],
                   color='DarkSeaGreen', align='center')
            ax.set_xticks(range(X.shape[1]))
            ax.set_xticklabels(self.features.columns[indices],
                               rotation=45,
                               ha='right')
            ax.set_xlim([-1, X.shape[1]])
            ylim = ax.get_ylim()
            ax.set_ylim([0, ylim[1]])
            ax.set_ylabel('Average Importances')
        else:  # use_pareto
            feats = self.features.columns
            fig, ax = pareto(imps[indices], cmap='magma', names=feats[indices])
            ax[0].set_ylabel('Importances')
        return fig, ax

    def plot_feature_pca(self):
        X = StandardScaler().fit_transform(self.features.to_numpy())
        pca = PCA(n_components=self.features.columns.size)
        pca.fit(X)
        fig, ax = pareto(pca.explained_variance_, cmap='viridis')
        ax[0].set_ylabel('Variance Explained')
        ax[0].set_xlabel('Principal Component')
        return fig, ax

    def collect_response(self, response_name):
        return np.vstack(self.response[response_name].ravel())


class Regressor:
    def __init__(self,
                 train_results,
                 test_results,
                 learner,
                 scaler=None):
        self.train_results = train_results
        self.test_results = test_results
        self.learner = MultiOutputRegressor(learner)
        self.scaler = scaler
        self.x_train = self.train_results.features.to_numpy()
        self.x_test = self.test_results.features.to_numpy()
        # self.response_dict = dict()

        if self.scaler is not None:
            self.scaler.fit(self.x_train)
            self.x_train = self.scaler.transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)

    def fit(self, response_name):
        y_train = self.train_results.collect_response(response_name)
        self.test_values = self.test_results.collect_response(response_name)
        self.learner.fit(self.x_train, y_train)
        self.current_response_name = response_name
        return self

    def predict(self):
        self.test_predictions = self.learner.predict(self.x_test)
        err = mean_squared_error(
            self.test_values,
            self.test_predictions
        )
        self.prediction_error = err
        return self.test_predictions
