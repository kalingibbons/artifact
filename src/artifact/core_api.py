import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class Results:
    def __init__(self, load_fcn=None):
        self._importances = None
        if load_fcn is not None:
            data = load_fcn()
            self.train_features = data[0]
            self.train_response = data[1]
            self.test_features = data[2]
            self.test_response = data[3]
        else:
            self.train_features = None
            self.train_response = None
            self.test_features = None
            self.test_response = None

    def plot_feature_importances(self):
        train_feat_df = self.train_features
        train_resp_df = self.train_response
        X = StandardScaler().fit_transform(train_feat_df.values)
        feats = train_feat_df.columns

        imps = self._importances
        if imps is None:
            imps = np.zeros_like(feats, dtype=np.float)
            imps_std = imps.copy()
            for col_name, col in train_resp_df.iteritems():
                y = self._collect_response(col)
                forest = RandomForestRegressor().fit(X, y)
                imps = imps + forest.feature_importances_
                imps_std = imps_std + np.std(
                    [tree.feature_importances_ for tree in forest.estimators_],
                    axis=0
                )
            imps = imps / train_resp_df.shape[1]
            imps_std = imps_std / train_resp_df.shape[1]
            indices = np.argsort(imps)[::-1]
            self._importances = imps
            self._importances_std = imps_std
            self._importances_indices = indices

        imps = self._importances
        imps_std = self._importances_std
        indices = self._importances_indices

        plt.figure()
        plt.title('Feature importances')
        plt.bar(range(X.shape[1]), imps[indices], yerr=imps_std[indices],
                color='DarkSeaGreen', align='center')
        plt.xticks(range(X.shape[1]), feats[indices], rotation=45, ha='right')
        plt.xlim([-1, X.shape[1]])
        plt.ylabel('Average Importances')

    @staticmethod
    def _collect_response(response_series):
        return np.vstack(response_series.ravel())
