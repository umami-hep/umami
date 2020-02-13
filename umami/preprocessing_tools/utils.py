import numpy as np
from dask.array.slicing import shuffle_slice
from sklearn.preprocessing import LabelBinarizer


def ShuffleDataFrame(df, seed=42, df_len=None, return_array=True):
    """ Shuffles dask DataFrame.
    Parameters
    ----------
    df: dask DataFrame to be shuffled
    seed:   int
            random seed, to maintain reproducability
    df_len: int
            length of DataFrame, if already known to speed up code
    return_array:   bool
                    if set to True (default) functin returns dask array
                    else dask DataFrame is returned
    Returns
    -------
    shuffled dask array:    if `return_array=True` (default)
    shuffled dask DataFrame:    if `return_array=False`
    """

    if df_len is None:
        df_len = len(df)
    d_arr = df.to_dask_array(True)
    np.random.seed(seed)
    index = np.random.choice(df_len, df_len, replace=False)
    print("suffling sample")
    d_arr = shuffle_slice(d_arr, index)
    if return_array:
        return d_arr
    return d_arr.to_dask_dataframe(df.columns)


def GetBinaryLabels(df, column='label'):
    """ Transforms labels to binary labels
    Parameters
    ----------
    df: dask DataFrame
    column: label name to be used to binarise

    Returns
    -------
    ndarray:    containing binary label with shape (len(df), n_classes)
    """
    lb = LabelBinarizer()
    labels = np.array(df[column].compute().values)
    return lb.fit_transform(labels)
