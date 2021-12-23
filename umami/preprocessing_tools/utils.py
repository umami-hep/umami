import os

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import global_config, logger
from umami.tools import applyATLASstyle, makeATLAStag


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
    if isinstance(df, np.ndarray):
        return lb.fit_transform(df)

    labels = np.array(df[column].values)
    return lb.fit_transform(labels)


def ResamplingPlots(
    concat_samples: dict,
    positions_x_y: list = [0, 1],
    variable_names: list = ["pT", "abseta"],
    plot_base_name: str = "plots/resampling-plot",
    binning: dict = {
        "pT": np.linspace(10000, 2000000, 200),
        "abseta": np.linspace(0, 2.5, 26),
    },
    Log: bool = True,
    after_sampling: bool = False,
    normalised: bool = False,
    use_weights: bool = False,
    hist_input: bool = False,
    second_tag: str = "",
    fileformat: str = "pdf",
):
    """
    Plots pt and eta distribution as nice plots for presentation.

    Parameters
    ----------
    concat_samples: dict
        dict with the format given in the Undersampling class by the class object
        `concat_samples` or the `x_y_after_sampling` depending on the `after_sampling`
        option
    positions_x_y:  list
        The position where the variables are stored the sub-dict `jets`
    variable_names: list
        The name of the 2 variables which will be plotted
    plot_base_name: str
        Folder and name of the plot w/o extension, this will be appened as well as
        the variable name
    binning: dict
        dict of the bin_edges used for plotting
    Log: bool
        boolean indicating if plot is in log scale or not (default True)
    after_sampling: bool
        If False (default) using the synthax of `concat_samples`
    normalised: bool
        Normalises the integral of the histogram to 1
    hist_input: bool
        If True the concat_samples is a dictionary of histograms and binning is
        already the full axes.
    second_tag: str
        Second tag which is inserted below the ATLAS tag (using the makeATLAStag
        function)
    fileformat: str
        Fileending of the plot. Default is "pdf"

    Returns
    -------
    Save plots of pt and eta to plot_base_name
    """
    applyATLASstyle(mtp)

    for varname, varpos in zip(variable_names, positions_x_y):
        # Loop over flavours
        plt.figure()
        for flav in concat_samples:
            if normalised:
                norm_factor = (
                    len(concat_samples[flav])
                    if after_sampling
                    else len(concat_samples[flav]["jets"])
                )
            else:
                norm_factor = 1.0

            scale_val = 1

            if varname in ["pT", "pt_btagJes"] or varname == global_config.pTvariable:
                scale_val = 1e3

            if hist_input:
                # Working directly on the x-D array
                direction_sum = tuple([i for i in positions_x_y if i != varpos])
                counts = np.sum(concat_samples[flav], axis=direction_sum)
                Bins = binning[varname] / scale_val

            else:
                # Calculate Binning and counts for plotting
                counts, Bins = np.histogram(
                    concat_samples[flav][:, varpos] / scale_val
                    if after_sampling
                    else concat_samples[flav]["jets"][:, varpos] / scale_val,
                    bins=binning[varname],
                    weights=concat_samples[flav]["weight"] if use_weights else None,
                )

            # Calculate the bin centers
            bincentres = [(Bins[i] + Bins[i + 1]) / 2.0 for i in range(len(Bins) - 1)]
            # Calculate poisson uncertainties and lower bands
            unc = np.sqrt(counts) / norm_factor
            band_lower = counts / norm_factor - unc

            plt.hist(
                x=Bins[:-1],
                bins=Bins,
                weights=(counts / norm_factor),
                histtype="step",
                linewidth=1.0,
                color=global_config.flavour_categories[flav]["colour"],
                stacked=False,
                fill=False,
                label=global_config.flavour_categories[flav]["legend_label"],
            )

            plt.hist(
                x=bincentres,
                bins=Bins,
                bottom=band_lower,
                weights=unc * 2,
                **global_config.hist_err_style,
            )

        if Log is True:
            plt.yscale("log")
            ymin, ymax = plt.ylim()

            if varname == "pT":
                plt.ylim(ymin=ymin, ymax=100 * ymax)

            else:
                plt.ylim(ymin=ymin, ymax=10 * ymax)

        elif Log is False:
            ymin, ymax = plt.ylim()
            plt.ylim(ymin=ymin, ymax=1.2 * ymax)

        if varname == global_config.pTvariable:
            plt.xlabel(r"$p_T$ in GeV")

        elif varname == global_config.etavariable:
            plt.xlabel(r"$\eta$")
        else:
            plt.xlabel(varname)

        plt.ylabel(r"Number of Jets")
        plt.legend(loc="upper right")

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag="Internal Simulation",
            second_tag=second_tag,
            ymax=0.9,
        )

        plt.tight_layout()
        if not os.path.exists(os.path.abspath("./plots")):
            os.makedirs(os.path.abspath("./plots"))

        plt.savefig(f"{plot_base_name}{varname}.{fileformat}")
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

        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 3), useMathText=True)
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
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 3), useMathText=True)
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
        ax.errorbar(x=bins[:-1], y=ratio_ub, xerr=x_error, fmt="none", ecolor="#2ca02c")
        ax.errorbar(x=bins[:-1], y=ratio_cb, xerr=x_error, fmt="none", ecolor="#ff7f0e")
        ax.set_ylabel("Ratio to b")
        ax.set_ylim(bottom=-0.1, top=2.1)
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


def generate_process_tag(preparation_ntuples_keys):
    """
    Builds a tag that contains the used processes (e.g. Z' and ttbar)
    which can then be used for plots.

    Parameters
    ----------
    preparation_ntuples_keys : dict_keys
        Dict keys from the preparation.ntuples section of the preprocessing
        config

    Returns
    -------
    second_tag_for_plot : str
        String which is used as 'second_tag' parameter  by the makeATLAStag
        function.
    """
    # Loop over the keys in the "preparation.ntuples" section of the
    # preprocessing config. For each process, try to extract the
    # corresponding plot label from the global config and add it to
    # the string that is inserted as "second_tag" in the plot
    processes = ""
    combined_sample = False
    logger.info("Looking for different processes in preprocessing config.")
    # for process in self.config.preparation["ntuples"].keys():
    for process in preparation_ntuples_keys:
        try:
            label = global_config.process_labels[process]["label"]
            if processes == "":
                processes += f"{label}"
            else:
                combined_sample = True
                processes += f" + {label}"
            logger.info(f"Found the process '{process}' with the label '{label}'")
        except KeyError:
            raise KeyError(
                f"Plot label for the process {process} was not"
                "found. Make sure your entries in the 'ntuples'"
                "section are valid entries that have a matching entry"
                "in the global config."
            )
    # Combine the string that contains the latex code for the processes
    # and the "sqrt(s)..." and "PFlow Jets" part
    if combined_sample is True:
        second_tag_for_plot = r"$\sqrt{s}$ = 13 TeV, Combined " + processes
    else:
        second_tag_for_plot = r"$\sqrt{s}$ = 13 TeV, " + processes

    second_tag_for_plot += " PFlow Jets"

    return second_tag_for_plot
