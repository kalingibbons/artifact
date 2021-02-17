"""Configures sample datasets and defines sample dataset loading functions."""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from artifact.core_api import select_by_regex

# warnings.filterwarnings('ignore', category=UserWarning, module=pd)


"""A dictionary of regular expressions for selecting functional groups."""
tkr_group_lut = dict(
    contact_mechanics=r'^(?!pat).+(_area|_press|_cop_\d)$',
    joint_loads=r'^(?!pat).+(_force_\d|_torque_\d)$',
    kinematics=r'^(?!pat).+(_lat|_ant|_inf|_valgus|_external)$',
    ligaments=r'^(?!ml|pl).+(_force|_disp)$',
    patella=r'(pl|pat).*',
)


def split_df(df, predictor_index):
    """Split dataframe into predictor and response columns.

    Uses the difference between two sets of indices to split, where the user
    passes the predictor columns as a function parameter.

    Args:
        df (DataFrame): The dataframe to be split.
        predictor_index ([list] int): A list of numeric indices for the
            predictor columns.

    Returns:
        [DataFrame]: The predictor and response dataframes.
    """
    every_index = np.arange(df.shape[1])
    response_index = np.setdiff1d(every_index, predictor_index)
    pred_df = df.iloc[:, predictor_index].drop(columns=['cam_rad'])  # constant
    resp_df = df.iloc[:, response_index]
    return pred_df.astype(np.float), resp_df


def drop_columns(data_df, regex_list, inplace=False):
    """Drop DataFrame columns by a list of regular expressions.

    Args:
        data_df (DataFrame): The dataframe with columns to be searched.
        regex_list ([list] str): A list of regular expressions for pattern
            matching.
        inplace (bool, Optional): Allow mutating of data_df. Defaults to False.

    Returns:
        DataFrame: A dataframe with any matching columns removed.
    """
    cols = data_df.columns
    needs_drop = np.any([cols.str.contains(x) for x in regex_list], axis=0)
    return data_df.drop(cols[needs_drop], axis='columns', inplace=inplace)


def load_tkr(functional_group=None, subset=None):
    """Reader function for the 2018 total knee replacement dataset.

    Able to load only a subset of the data (train, test), as well as only a
    subset of the variables using functional group names as the selector:

        * contact_mechanics - Tibiofemoral contact areas and pressures
        * joint_loads - Tibiofemoral muscle forces and moments
        * kinematics - Joint coordinate system tibiofemoral kinematics
        * ligaments - Tibiofemoral ligament elongations and developed forces
        * patella - All of the above for the patellofemoral joint

    Args:
        functional_group (str, optional): A functional group to load.
            Defaults to None.
        subset (str, optional): Either the train or test subset. Defaults to
            None.

    Returns:
        [DataFrame]: If a subset is selected, a pair of dataframes for the
            features or response variables. If no subset of passed, than a
            tuple of pairs of dataframes.
    """
    pred_idx = np.arange(0, 14)

    data_dir = Path.cwd().parent / 'data' / 'preprocessed'

    def select_group(df, functional_group):
        if functional_group in tkr_group_lut.keys():
            patterns = df.iloc[0, pred_idx].index.to_list()
            patterns.append(tkr_group_lut[functional_group])
            patterns.append('time')
            df, _ = select_by_regex(df, patterns, axis=1)
        return df

    if (subset is None) or (subset.lower() == 'test'):
        test_data = pd.read_parquet(data_dir / 'test.parquet')
        test_data = select_group(test_data, functional_group)
        test_feat, test_resp = split_df(test_data, pred_idx)

    if (subset is None) or (subset.lower() == 'train'):
        train_data = pd.read_parquet(data_dir / 'train.parquet')
        train_data = select_group(train_data, functional_group)
        train_feat, train_resp = split_df(train_data, pred_idx)

    if subset is None:
        return (train_feat, train_resp), (test_feat, test_resp)
    if subset.lower() == 'train':
        return train_feat, train_resp
    if subset.lower() == 'test':
        return test_feat, test_resp
