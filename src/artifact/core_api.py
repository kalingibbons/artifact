"""Core API for performing predictive statistical analysis of FEA results."""

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        is_match = np.any([labels.str.contains(x) for x in regex_list], axis=0)
    if negate:
        is_match = ~is_match

    selected_labels = labels[~is_match]
    return data_df.drop(selected_labels, axis=axis), selected_labels


class Results:
    """Load and describe Abaqus simulation results."""

    def __init__(self, *load_args, results_reader=None, **load_kwargs):
        """Load simulation results using a reader function.

        Args:
            load_fcn ([function_handle], optional): The reader function used
                for loading the results data. Should return a tuple of pandas
                DataFrame objects, separating feature columns from response
                columns. Defaults to None.
        """
        self._importances = None
        self._importances_std = None
        self._importances_indices = None
        if results_reader is not None:
            self.features, self.response = results_reader(
                *load_args, **load_kwargs
            )
            self.feature_names = list(self.features.columns)
            self.response_names = list(self.response.columns)
        else:
            self.features = None
            self.response = None
            self.feature_names = None
            self.response_names = None

    def describe_features(self):
        """Print summary statistics and histogram for features.

        The summary statistics are output from the DataFrame.describe() method.

        Returns:
            [list] Axes: Histograms produced by the DataFrame.hist() method.
        """
        display(self.features.describe())
        axs = self.features.hist(figsize=(12, 11))
        return axs

    def _calc_importances(self):
        """Calculate feature importances using random forest regression."""
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
        """Plot feature importances.

        Will calculate importances if they do not already exist. Can be plotted
        with standard deviation bars, or as a Pareto plot in order to display
        cumulative importances.

        Args:
            use_pareto (bool, optional): Set to True to use a pareto plot
                instead of the standard deviations. Defaults to False.

        Returns:
            (Figure, Axes): The figure and axes handles for the resulting
                plots.
        """
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
        """Pareto plot the principal components of dataset features.

        Principal components are axes of highest variance.

        Returns:
            (Figure, Axes): The figure and axes handles of the resulting Pareto
                plot.
        """
        X = StandardScaler().fit_transform(self.features.to_numpy())
        pca = PCA(n_components=self.features.columns.size)
        pca.fit(X)
        fig, ax = pareto(pca.explained_variance_, cmap='viridis')
        ax[0].set_ylabel('Variance Explained')
        ax[0].set_xlabel('Principal Component')
        return fig, ax

    def collect_response(self, response_name):
        """Extract single response column as numpy array.

        Args:
            response_name (str): The response name to be extracted.

        Returns:
            [NdArray]: A numpy array of response variables, with each entry
                corresponding to a single observation.
        """
        return np.vstack(self.response[response_name].ravel())


class Regressor:
    """Perform multi output regression on simulation results."""

    def __init__(self,
                 train_results,
                 test_results,
                 learner,
                 scaler=None):
        """Construct Regressor with optional data scaling.

        Args:
            train_results (DataFrame): Results from the training simulations.
            test_results (DataFrame): Results from the validation simulations.
            learner (Regressor): Single output regressor implementing standard
                scikit-learn API
            scaler (Scaler, optional): Data scaler object implementing
                scikit-lear API. Defaults to None.
        """
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
        """Fit multioutput regressor to training data.

        Args:
            response_name (str): The response name to be fit.

        Returns:
            MultiOutputRegressor: The fitted multioutput regressor object for
                method chaining.
        """
        y_train = self.train_results.collect_response(response_name)
        self.test_values = self.test_results.collect_response(response_name)
        self.learner.fit(self.x_train, y_train)
        self.current_response_name = response_name
        return self

    def predict(self):
        """Predict new outputs using the already fitted Regressor.

        Operates on the entire test dataset, and calculates root-mean-square
        prediction errors for the observations.

        Returns:
            NdArray: An array of predictions for each observation in the
                validation dataset.
        """
        self.test_predictions = self.learner.predict(self.x_test)
        err = mean_squared_error(
            self.test_values,
            self.test_predictions,
            squared=False
        )
        self.prediction_error = err
        return self.test_predictions

    def cross_val_score(self, response_names=None, n_jobs=1, cv=3):
        """Score a list of responses using cross validation.

        Calculates the root mean squared errors for the list of response names,
        then non dimensionalizes the errors by response range (RMSE / range).

        Args:
            response_names ([list](str), optional): A list of response names to
                perform cross validation on. Defaults to None.
            n_jobs (int, optional): The number of jobs to run in parallel.
                Defaults to 1.
            cv (int, optional): The number of folds in a stratified K-fold.
                Defaults to 3.

        Returns:
            float: An array of nondimensionalized RMSE ratios scoring the
                cross-validation.
        """
        if response_names is None:
            response_names = self.train_results.response_names
            if 'time' in response_names:
                response_names.pop(response_names.index('time'))

        elif isinstance(response_names, str):
            response_names = [response_names]

        rmse_ratio = np.full((len(response_names), cv), np.nan)
        for resp_idx, response in enumerate(response_names):
            y_train = self.train_results.collect_response(response)
            rmse = cross_val_score(
                self.learner,
                self.x_train,
                y_train,
                n_jobs=n_jobs,
                scoring=make_scorer(mean_squared_error, squared=False),
                cv=cv
            )
            rmse_ratio[resp_idx, :] = rmse / (y_train.max() - y_train.min())
        return rmse_ratio
