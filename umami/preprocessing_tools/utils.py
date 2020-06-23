import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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
    if type(df) is np.ndarray:
        return lb.fit_transform(df)

    labels = np.array(df[column].compute().values)
    return lb.fit_transform(labels)


def MakePlots(bjets, ujets, cjets, plot_name="plots/InfoPlot.pdf",
              binning={"pt_btagJes": np.linspace(10000, 2000000, 200),
                       "eta_btagJes": np.linspace(0, 2.5, 26)}):
    """ Plots pt and eta distribution.
    Parameters
    ----------
    TODO
    bjets: array of b-jets
    ujets: array of light jets

    Returns
    -------
    TODO
    """

    vars = ["pt_btagJes", "absEta_btagJes"]
    # print(pd.DataFrame(bjets)["pt_btagJes"])
    # print(bjets["eta_btagJes"])

    for i, var in enumerate(vars):
        plt.subplot(1, 2, i + 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3),
                             useMathText=True)
        plt.hist([ujets[var], cjets[var], bjets[var]], binning[var],
                 # weights=[arr_u['weight'], arr_c['weight'],
                 #          np.ones(len(arr_b))],
                 # color=['#4854C3', '#97BD8A', '#D20803'],
                 # color=['#2ca02c', '#1f77b4', '#d62728'],
                 color=['#2ca02c', '#ff7f0e', '#1f77b4'],
                 # color=['forestgreen', 'mediumblue', 'r'],alpha=0.8,
                 label=['ujets', 'cjets', 'bjets'], histtype='step',
                 stacked=False, fill=False)
        plt.yscale('log')
        plt.title(var)
        plt.legend()
    plt.tight_layout()
    if not os.path.exists(os.path.abspath("./plots")):
        os.makedirs(os.path.abspath("./plots"))
    plt.savefig(plot_name, transparent=True)
    plt.close()


def Plot_vars(bjets, cjets, ujets, plot_name="InfoPlot"):
    """Creates plots of all variables and saves them."""
    bjet = pd.DataFrame(bjets)
    cjet = pd.DataFrame(cjets)
    ujet = pd.DataFrame(ujets)
    variablelist = list(bjet.columns.values)
    print(variablelist)
    print(len(variablelist))
    variablelist.remove('label')
    variablelist.remove('weight')
    if 'category' in variablelist:
        variablelist.remove('category')

    plt.figure(figsize=(20, 60))
    for i, var in enumerate(variablelist):
        if "isDefaults" in var:
            nbins = 2
        else:
            nbins = 50
        plt.subplot(20, 5, i + 1)
        plt.hist([ujet[var], cjet[var], bjet[var]], nbins,  # normed=1,
                 weights=[ujet['weight'], cjet['weight'], bjet['weight']],
                 # color=['#4854C3', '#97BD8A', '#D20803'],
                 # color=['#2ca02c', '#1f77b4', '#d62728'],
                 color=['#2ca02c', '#ff7f0e', '#1f77b4'],
                 label=['ujets', 'cjets', 'bjets'], histtype='step',
                 stacked=False, fill=False)
        plt.yscale('log')
        plt.title(var)
        plt.legend()
    plt.tight_layout()
    plotname = "plots/%s_all_vars.pdf" % plot_name
    print("save plot as", plotname)
    plt.savefig(plotname, transparent=True)


def ScaleTracks(data, var_names, scale_dict=None, mask_value=0):
    '''
    Args:
    -----
        data: a numpy array of shape (nJets, nTrks, nFeatures)
        var_names: list of keys to be used for the model
        scale_dict: dict -- None for training, scaling dictionary for testing
                  it decides whether we want to fit on data to find mean and
                  std or if we want to use those stored in the scale dict
        mask_value: the value to mask when taking the avg and stdev

    Returns:
    --------
        modifies data in place, if scale_dict was specified
        scaling dictionary, if scale_dict was None

    Reference: https://github.com/mickypaganini/RNNIP/blob/master/dataprocessing.py#L235-L319  # noqa
    '''

    # Track variables
    # data has shape nJets,nTrks,nFeatures,so to sort out the mask,
    # we need to find where the value is masked for a track over
    # all it's features
    # mask has shape nJets,nTrks
    mask = ~ np.all(data == mask_value, axis=-1)

    if scale_dict is None:
        scale_dict = {}
        for v, name in enumerate(var_names):
            print(f'Scaling feature {v + 1} of {len(var_names)} ({name}).')
            f = data[:, :, v]
            slc = f[mask]
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {'shift': float(m), 'scale': float(s)}

        return scale_dict

    else:
        for v, name in enumerate(var_names):
            print(f'Scaling feature {v + 1} of {len(var_names)} ({name}).')
            f = data[:, :, v]
            slc = f[mask]
            m = scale_dict[name]['shift']
            s = scale_dict[name]['scale']
            slc -= m
            slc /= s
            data[:, :, v][mask] = slc.astype('float32')
