import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class Results:
    def __init__(self, *load_args, load_fcn=None, **load_kwargs):
        self._importances = None
        self._importances_std = None
        self._importances_indices = None
        if load_fcn is not None:
            self.features, self.response = load_fcn(*load_args, **load_kwargs)
        else:
            self.features = None
            self.responses = None

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

    def plot_feature_importances(self):
        X = StandardScaler().fit_transform(self.features.values)
        if self._importances is None:
            self._calc_importances()

        imps = self._importances
        imps_std = self._importances_std
        indices = self._importances_indices

        plt.figure()
        plt.title('Feature importances')
        plt.bar(range(X.shape[1]), imps[indices], yerr=imps_std[indices],
                color='DarkSeaGreen', align='center')
        plt.xticks(range(X.shape[1]),
                   self.features.columns[indices],
                   rotation=45,
                   ha='right')
        plt.xlim([-1, X.shape[1]])
        plt.ylabel('Average Importances')

    def collect_response(self, response_name):
        return np.vstack(self.response[response_name].ravel())
