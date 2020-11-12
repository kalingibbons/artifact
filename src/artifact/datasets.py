from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as spio


def import_data(matfilepath):
    # TODO: Dynamic import of different import functions?
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


def split_df(df, predictor_index):
    every_index = np.arange(df.shape[1])
    response_index = np.setdiff1d(every_index, predictor_index)
    pred_df = df.iloc[:, predictor_index].drop(columns=['cam_rad'])  # constant
    resp_df = df.iloc[:, response_index]
    return pred_df.astype(np.float), resp_df


def remove_failed(response_series, df_list):
    failed_idx = response_series.apply(lambda x: x.size == 0)
    new_df_list = np.full(len(df_list), np.nan, dtype='object')
    for idx, df in enumerate(df_list):
        new_df_list[idx] = df[~failed_idx]

    return new_df_list


def load_tkr(use_reduced=True, subset=None):
    pred_idx = np.arange(0, 14)

    test_name = 'test_reduced.mat' if use_reduced else 'test.mat'
    train_name = 'doe_reduced.mat' if use_reduced else 'doe.mat'
    data_dir = Path.cwd().parent / 'data' / 'preprocessed'

    test_feat, test_resp = split_df(
        import_data(data_dir / test_name),
        pred_idx
    )
    train_feat, train_resp = split_df(
        import_data(data_dir / train_name),
        pred_idx
    )

    # Failed simulations will have empty rows. Remove them.
    test_resp, test_feat = remove_failed(
        test_resp['time'], (test_resp, test_feat)
    )

    train_resp, train_feat = remove_failed(
        train_resp['time'], (train_resp, train_feat)
    )
    if subset is None:
        return (train_feat, train_resp), (test_feat, test_resp)
    if subset.lower() == 'train':
        return train_feat, train_resp
    if subset.lower() == 'test':
        return test_feat, test_resp
