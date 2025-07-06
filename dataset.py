import pandas as pd
import numpy as np


def extract_labels(df: pd.DataFrame, window: int):
    df = df.values.reshape(len(df) // window, window)
    return df[:, 1]


def extract_variable_train(df: pd.DataFrame, name: str, window: int):
    x = df[name]
    x = pd.DataFrame(x)
    x = x[x.columns[::2]]
    return x[name].values.reshape(len(df) // window, window, 1)


def processing(df: pd.DataFrame):
    y = df.pop('status')
    df.pop('status_label')
    df.pop('cik')
    df.pop('fyear')
    df.pop('company_name')
    df.pop('tic')
    return df, y


def shuffle_group(df: pd.DataFrame, k: int):
    len_group = k

    index_list = np.array(df.index)
    np.random.shuffle(np.reshape(index_list, (-1, len_group)))

    shuffled_df = df.loc[index_list, :]
    return shuffled_df


def undersample_dataset(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = shuffle_group(df, window)
    grp = df.groupby(['status', 'company_name'])
    postproc = pd.DataFrame()

    for key, _ in grp:
        df_grp = grp.get_group(key).sort_values("fyear")
        postproc = pd.concat([postproc, df_grp])

    to_cut = len(df[df.status_label == 'alive']) - len(df[df.status_label == "failed"])
    df = postproc.drop(index=postproc.index[:to_cut])
    return df

