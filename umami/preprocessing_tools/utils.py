import os

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.array.slicing import shuffle_slice
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import global_config, logger
from umami.tools import applyATLASstyle, makeATLAStag


def ShuffleDataFrame(df, seed=42, df_len=None, return_array=True):
    """Shuffles dask DataFrame.
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


def GetBinaryLabels(df, column="label"):
    """Transforms labels to binary labels
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


def MakePresentationPlots(
    bjets,
    ujets,
    cjets,
    taujets=None,
    plots_path="plots/",
    binning={
        global_config.pTvariable: np.linspace(10000, 2000000, 200),
        global_config.etavariable: np.linspace(0, 2.5, 26),
    },
    Log=True,
):
    """Plots pt and eta distribution as nice plots for
    presentation.

    Parameters
    ----------
    bjets: array of b-jets
    ujets: array of light jets
    cjets: array of c-jets

    Returns
    -------
    Save nice plots of pt and eta to plots_path
    """

    var_list = [global_config.pTvariable, global_config.etavariable]

    for var in var_list:
        # Get number of flavours
        N_b = len(bjets[var])
        N_c = len(cjets[var])
        N_u = len(ujets[var])
        if taujets is not None:
            N_tau = len(taujets[var])

        divide_val = 1
        if var == global_config.pTvariable:
            divide_val = 1000
        # Calculate Binning and counts for plotting
        counts_b, Bins_b = np.histogram(
            bjets[var] / divide_val,
            bins=binning[var],
        )

        # Calculate Binning and counts for plotting
        counts_c, Bins_c = np.histogram(
            cjets[var] / divide_val,
            bins=binning[var],
        )

        # Calculate Binning and counts for plotting
        counts_u, Bins_u = np.histogram(
            ujets[var] / divide_val,
            bins=binning[var],
        )

        # Calculate Binning and counts for plotting
        if taujets is not None:
            counts_tau, Bins_tau = np.histogram(
                taujets[var] / divide_val,
                bins=binning[var],
            )

        # Calculate the bin centers
        bincentres = [
            (Bins_b[i] + Bins_b[i + 1]) / 2.0 for i in range(len(Bins_b) - 1)
        ]

        # Calculate poisson uncertainties and lower bands
        unc_b = np.sqrt(counts_b) / N_b
        band_lower_b = counts_b / N_b - unc_b

        # Calculate poisson uncertainties and lower bands
        unc_c = np.sqrt(counts_c) / N_c
        band_lower_c = counts_c / N_c - unc_c

        # Calculate poisson uncertainties and lower bands
        unc_u = np.sqrt(counts_u) / N_u
        band_lower_u = counts_u / N_u - unc_u

        # Calculate poisson uncertainties and lower bands
        if taujets is not None:
            unc_tau = np.sqrt(counts_tau) / N_tau
            band_lower_tau = counts_tau / N_tau - unc_tau

        applyATLASstyle(mtp)
        plt.figure()
        plt.hist(
            x=Bins_b[:-1],
            bins=Bins_b,
            weights=(counts_b / N_b),
            histtype="step",
            linewidth=1.0,
            color="#1f77b4",
            stacked=False,
            fill=False,
            label=r"$b$-jets",
        )

        plt.hist(
            x=bincentres,
            bins=Bins_b,
            bottom=band_lower_b,
            weights=unc_b * 2,
            fill=False,
            hatch="/////",
            linewidth=0,
            edgecolor="#666666",
        )

        plt.hist(
            x=Bins_c[:-1],
            bins=Bins_c,
            weights=(counts_c / N_c),
            histtype="step",
            linewidth=1.0,
            color="#ff7f0e",
            stacked=False,
            fill=False,
            label=r"$c$-jets",
        )

        plt.hist(
            x=bincentres,
            bins=Bins_c,
            bottom=band_lower_c,
            weights=unc_c * 2,
            fill=False,
            hatch="/////",
            linewidth=0,
            edgecolor="#666666",
        )

        plt.hist(
            x=Bins_u[:-1],
            bins=Bins_u,
            weights=(counts_u / N_u),
            histtype="step",
            linewidth=1.0,
            color="#2ca02c",
            stacked=False,
            fill=False,
            label=r"light-jets",
        )

        plt.hist(
            x=bincentres,
            bins=Bins_u,
            bottom=band_lower_u,
            weights=unc_u * 2,
            fill=False,
            hatch="/////",
            linewidth=0,
            edgecolor="#666666",
        )

        if taujets is not None:
            plt.hist(
                x=Bins_tau[:-1],
                bins=Bins_tau,
                weights=(counts_tau / N_tau),
                histtype="step",
                linewidth=1.0,
                color="#7c5295",
                stacked=False,
                fill=False,
                label=r"tau-jets",
            )

            plt.hist(
                x=bincentres,
                bins=Bins_tau,
                bottom=band_lower_tau,
                weights=unc_tau * 2,
                fill=False,
                hatch="/////",
                linewidth=0,
                edgecolor="#666666",
            )

        if Log is True:
            plt.yscale("log")
            ymin, ymax = plt.ylim()

            if var == global_config.pTvariable:
                plt.ylim(ymin=ymin, ymax=100 * ymax)

            else:
                plt.ylim(ymin=ymin, ymax=10 * ymax)

        elif Log is False:
            ymin, ymax = plt.ylim()
            plt.ylim(ymin=ymin, ymax=1.2 * ymax)

        if var == global_config.pTvariable:
            plt.xlabel(r"$p_T$ in GeV")
            plt.xlim(right=6500)

        elif var == global_config.etavariable:
            plt.xlabel(r"$\eta$")

        else:
            plt.xlabel(var)

        plt.ylabel(r"Number of Jets")
        plt.legend(loc="upper right")

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag="Internal Simulation",
            second_tag=r"$\sqrt{s}$ = 13 TeV, Combined $t\bar{t} + $ ext. $Z'$ PFlow Jets",
            ymax=0.9,
        )

        plt.tight_layout()
        if not os.path.exists(os.path.abspath("./plots")):
            os.makedirs(os.path.abspath("./plots"))
        plt.savefig(f"{plots_path}{var}.pdf")
        plt.close()
        plt.clf()


def MakePlots(
    bjets,
    ujets,
    cjets,
    taujets=None,
    plot_name="plots/InfoPlot.pdf",
    binning={
        global_config.pTvariable: np.linspace(10000, 2000000, 200),
        global_config.etavariable: np.linspace(0, 2.5, 26),
    },
):
    """Plots pt and eta distribution.
    Parameters
    ----------
    TODO
    bjets: array of b-jets
    ujets: array of light jets

    Returns
    -------
    TODO
    """

    vars = [global_config.pTvariable, global_config.etavariable]

    bool_plot_taujets = taujets is not None
    fig = plt.figure()
    heights = [3, 1]
    spec = fig.add_gridspec(ncols=2, nrows=2, height_ratios=heights)
    for i, var in enumerate(vars):
        # plt.subplot(2, 2, i + 1)
        ax = fig.add_subplot(spec[0, i])

        ax.ticklabel_format(
            style="sci", axis="x", scilimits=(0, 3), useMathText=True
        )
        if bool_plot_taujets:
            array_entries, bins, _ = ax.hist(
                [taujets[var], ujets[var], cjets[var], bjets[var]],
                binning[var],
                color=["#7c5295", "#2ca02c", "#ff7f0e", "#1f77b4"],
                label=["tau-jets", "ujets", "cjets", "bjets"],
                histtype="step",
                stacked=False,
                fill=False,
            )
        else:
            array_entries, bins, _ = ax.hist(
                [ujets[var], cjets[var], bjets[var]],
                binning[var],
                color=["#2ca02c", "#ff7f0e", "#1f77b4"],
                label=["ujets", "cjets", "bjets"],
                histtype="step",
                stacked=False,
                fill=False,
            )
        ax.set_yscale("log")
        ax.set_title(var)
        ax.legend()

        # Do ratios now:
        ax = fig.add_subplot(spec[1, i])
        ax.ticklabel_format(
            style="sci", axis="x", scilimits=(0, 3), useMathText=True
        )
        b_data = array_entries[-1]
        x_error = np.zeros((2, len(b_data)))
        x_error[1, :] = bins[1] - bins[0]

        if bool_plot_taujets:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_taub = array_entries[0].astype(float) / b_data
                ratio_ub = array_entries[1].astype(float) / b_data
                ratio_cb = array_entries[2].astype(float) / b_data
                ratio_ub[~np.isfinite(ratio_ub)] = 1000
                ratio_cb[~np.isfinite(ratio_cb)] = 1000
                ratio_taub[~np.isfinite(ratio_taub)] = 1000
            ax.errorbar(
                x=bins[:-1],
                y=ratio_taub,
                xerr=x_error,
                fmt="none",
                ecolor="#7c5295",
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_ub = array_entries[0].astype(float) / b_data
                ratio_cb = array_entries[1].astype(float) / b_data
                ratio_ub[~np.isfinite(ratio_ub)] = 1000
                ratio_cb[~np.isfinite(ratio_cb)] = 1000
        ax.errorbar(
            x=bins[:-1], y=ratio_ub, xerr=x_error, fmt="none", ecolor="#2ca02c"
        )
        ax.errorbar(
            x=bins[:-1], y=ratio_cb, xerr=x_error, fmt="none", ecolor="#ff7f0e"
        )
        ax.set_ylabel("Ratio to b")
        ax.set_ylim(bottom=-0.5, top=4.0)
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True)
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
    logger.info(f"variable list: {variablelist}")
    logger.info(f"#variable: {len(variablelist)}")
    variablelist.remove("label")
    variablelist.remove("weight")
    if "category" in variablelist:
        variablelist.remove("category")

    plt.figure(figsize=(20, 60))
    for i, var in enumerate(variablelist):
        if "isDefaults" in var:
            nbins = 2
        else:
            nbins = 50
        plt.subplot(20, 5, i + 1)
        plt.hist(
            [ujet[var], cjet[var], bjet[var]],
            nbins,  # normed=1,
            weights=[ujet["weight"], cjet["weight"], bjet["weight"]],
            # color=['#4854C3', '#97BD8A', '#D20803'],
            # color=['#2ca02c', '#1f77b4', '#d62728'],
            color=["#2ca02c", "#ff7f0e", "#1f77b4"],
            label=["ujets", "cjets", "bjets"],
            histtype="step",
            stacked=False,
            fill=False,
        )
        plt.yscale("log")
        plt.title(var)
        plt.legend()
    plt.tight_layout()
    plotname = "plots/%s_all_vars.pdf" % plot_name
    logger.info(f"Save plot as {plotname}")
    plt.savefig(plotname, transparent=True)


def ScaleTracks(data, var_names, scale_dict=None, mask_value=0):
    """
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
    """

    # Track variables
    # data has shape nJets,nTrks,nFeatures,so to sort out the mask,
    # we need to find where the value is masked for a track over
    # all it's features
    # mask has shape nJets,nTrks
    mask = ~np.all(data == mask_value, axis=-1)

    if scale_dict is None:
        scale_dict = {}
        for v, name in enumerate(var_names):
            logger.info(
                f"Scaling feature {v + 1} of {len(var_names)} ({name})."
            )
            f = data[:, :, v]
            slc = f[mask]
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {"shift": float(m), "scale": float(s)}

        return scale_dict

    else:
        for v, name in enumerate(var_names):
            logger.info(
                f"Scaling feature {v + 1} of {len(var_names)} ({name})."
            )
            f = data[:, :, v]
            slc = f[mask]
            m = scale_dict[name]["shift"]
            s = scale_dict[name]["scale"]
            slc -= m
            slc /= s
            data[:, :, v][mask] = slc.astype("float32")
