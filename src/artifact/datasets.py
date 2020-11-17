import re
from pathlib import Path
import numpy as np
import pandas as pd


def split_df(df, predictor_index):
    every_index = np.arange(df.shape[1])
    response_index = np.setdiff1d(every_index, predictor_index)
    pred_df = df.iloc[:, predictor_index].drop(columns=['cam_rad'])  # constant
    resp_df = df.iloc[:, response_index]
    return pred_df.astype(np.float), resp_df


def drop_columns(data_df, regex_list, inplace=False):
    cols = data_df.columns
    needs_drop = np.any([cols.str.contains(x) for x in regex_list], axis=0)
    return data_df.drop(cols[needs_drop], axis='columns', inplace=inplace)


def load_tkr(functional_group=None, subset=None):
    pred_idx = np.arange(0, 14)

    data_dir = Path.cwd().parent / 'data' / 'preprocessed'

    if (subset is not None) or (subset.lower() == 'train'):
        test_data = pd.read_parquet(data_dir / 'test.parquet')
        test_feat, test_resp = split_df(test_data, pred_idx)

    if (subset is not None) or (subset.lower() == 'test'):
        train_data = pd.read_parquet(data_dir / 'train.parquet')
        train_feat, train_resp = split_df(train_data, pred_idx)

    if subset is None:
        return (train_feat, train_resp), (test_feat, test_resp)
    if subset.lower() == 'train':
        return train_feat, train_resp
    if subset.lower() == 'test':
        return test_feat, test_resp
