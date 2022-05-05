"""Collection of plotting function for ftag performance plots."""
# pylint: disable=consider-using-f-string, invalid-name
# TODO: switch to new plotting API with pep8 conform naming
from umami.configuration import global_config, logger  # isort:skip

from collections import OrderedDict

import matplotlib as mtp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.helper_tools import hist_ratio, hist_w_unc
from umami.metrics import eff_err
from umami.plotting import roc, roc_plot, var_vs_eff, var_vs_eff_plot
from umami.plotting.utils import translate_kwargs
from umami.tools import applyATLASstyle


def FlatEfficiencyPerBin(
    df, predictions, variable, var_bins, classes, target="bjets", wp=0.7
):
    """For each bin in var_bins of variable, cuts the score in predictions column to get
    the desired WP (working point) df must (at least) contain the following columns:
        - score
        - value of variable
        - labels (with the true labels)
    Creates a column 'tag' with the tagged (1/0) info in df.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe
    predictions : str
        predictions column name
    variable : string
        variable
    var_bins : list
        bins
    classes : list
        classes
    target : str, optional
        target flavour, by default "bjets"
    wp : float, optional
        working point, by default 0.7

    Returns
    -------
    np.ndarray
        tag indicating if jet is tagged (1/0)
    """
    target_index = classes.index(target)
    df["tag"] = 0
    for i in range(len(var_bins) - 1):
        index_jets_in_bin = (var_bins[i] <= df[variable]) & (
            df[variable] < var_bins[i + 1]
        )
        df_in_bin = df[index_jets_in_bin]
        if len(df_in_bin) == 0:
            continue
        scores_b_in_bin = df_in_bin[df_in_bin["labels"] == target_index][predictions]
        if len(scores_b_in_bin) == 0:
            continue
        cutvalue_in_bin = np.percentile(scores_b_in_bin, 100.0 * (1.0 - wp))
        df.loc[index_jets_in_bin, ["tag"]] = (
            df_in_bin[predictions] > cutvalue_in_bin
        ) * 1

    return df["tag"]


# TODO: move this to new python API
def plotEfficiencyVariable(
    df: pd.DataFrame,
    class_labels_list: list,
    main_class: str,
    variable: str,
    var_bins: np.ndarray,
    plot_name: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    ThirdTag: str = "DL1r",
    yAxisIncrease: float = 1.4,
    yAxisAtlasTag: float = 0.9,
    efficiency: float = 0.70,
    frac_values: dict = None,
    centralise_bins: bool = True,
    xticksval: list = None,
    xticks: list = None,
    minor_ticks_frequency: int = None,
    xlabel: str = None,
    logy: bool = False,
    ymin: float = None,
    ymax: float = None,
    dpi: int = 400,
):
    """
    For a given variable (string) in the panda dataframe df, plots
    the eff for each flavour as a function of variable
    (discretised in bins as indicated by var_bins input).

    Note: to get a flat efficiency plot, you need to produce a 'tag' column
    in df using FlatEfficiencyPerBin (defined above).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas.DataFrame with the columns needed are:
        - The efficiency is computed from the tag column of df.
        - variable (see eponymous parameter)
        - labels (as defined in the preprocessing, MUST MATCH class_labels_list)
    class_labels_list : list
        List indicating the class order
        as defined in the preprocessing!
        WARNING: wrong behaviour if order is different.
    main_class : str
        Name of the main class.
    variable : str
        String of the variable in the dataframe to plot against.
    var_bins : np.ndarray
        Array of the bins to use.
    plot_name : str
        Path, Name and format of the resulting plot file.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample"
    ThirdTag : str, optional
        Additional tag, by default "DL1r"
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag in, by default 1.4
    yAxisAtlasTag : float, optional
        Relative y axis position of the ATLAS Tag, by default 0.9
    efficiency : float, optional
        Working point value, by default 0.70
    frac_values : dict, optional
        Dict with the fraction values inside, by default None
    centralise_bins : bool, optional
        Decide to centralise point in the bins, by default True
    xticksval : list, optional
        List with the xticks values, by default None
    xticks : list, optional
        List with the xticks for the given values, by default None
    minor_ticks_frequency : int, optional
        If given, sets the frequency of minor ticks, by default None
    xlabel : str, optional
        x axis label, by default None
    logy : bool, optional
        Decide, to set the y axis to logarithmic, by default False
    ymin : float, optional
        Minimal y axis value, by default None
    ymax : float, optional
        Maximal y axis value, by default None
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    null_value = np.nan

    store_dict = {
        "store_eff_tagger": np.zeros((len(var_bins) - 1, len(class_labels_list))),
        "store_err_tagger": np.zeros((len(var_bins) - 1, len(class_labels_list))),
    }

    for i in range(len(var_bins) - 1):
        # all jets in bin
        df_in_bin = df[(var_bins[i] <= df[variable]) & (df[variable] < var_bins[i + 1])]
        total_in_bin = len(df_in_bin)
        # all tagged jets in bin. (note: can also work for other flavour)
        df_in_bin_tagged = df_in_bin.query("tag == 1")
        if total_in_bin != 0:
            index, counts = np.unique(
                df_in_bin_tagged["labels"].values, return_counts=True
            )
            count_dict = dict(zip(index, counts))

        else:
            count_dict = {}
        # store result for each flavour in store_dict
        for label_ind, label in enumerate(class_labels_list):
            label_in_bin = len(df_in_bin[df_in_bin["labels"] == label_ind])

            if label_ind in count_dict and label_in_bin != 0:
                eff = count_dict[label_ind] / label_in_bin
                err = eff_err(eff, label_in_bin)
            else:
                eff = null_value
                err = null_value

            store_dict["store_eff_tagger"][i, label_ind] = eff
            store_dict["store_err_tagger"][i, label_ind] = err

    if xlabel is None:
        xlabel = variable
        if variable == "actualInteractionsPerCrossing":
            xlabel = "Actual interactions per bunch crossing"
        elif variable == "pt":
            xlabel = r"$p_T$ [GeV]"

    if frac_values is None:
        ThirdTag = (
            ThirdTag + f"\n{efficiency}% {flav_cat[main_class]['legend_label']} WP"
        )
    else:
        frac_string = ""
        for frac in list(frac_values.keys()):
            frac_string += f"{frac}: {frac_values[frac]} | "
        frac_string = frac_string[:-3]
        ThirdTag = (
            ThirdTag
            + f"\n{efficiency}% {flav_cat[main_class]['legend_label']} WP,"
            + f"\nFractions: {frac_string}"
        )

    # Divide pT values to express in GeV, not MeV
    if variable == "pt":
        var_bins = var_bins / 1000

    x_value, x_err = [], []
    for i in range(len(var_bins[:-1])):
        if centralise_bins:
            x_value.append((var_bins[i] + var_bins[i + 1]) / 2)
        else:
            x_value.append(var_bins[i])
        x_err.append(abs(var_bins[i] - var_bins[i + 1]) / 2)

    x_label = var_bins
    if xticksval is not None:
        trimmed_label_val = xticksval
        trimmed_label = xticksval
        if xticksval is not None:
            trimmed_label = xticks
    else:
        trimmed_label = []
        trimmed_label_val = []
        selected_indices = [int((len(x_label) - 1) / 4) * i for i in range(5)]
        if len(x_label) > 5:
            for i, x_label_i in enumerate(x_label):
                if i in selected_indices:
                    trimmed_label.append(
                        np.format_float_scientific(x_label_i, precision=3)
                    )
                    trimmed_label_val.append(x_label_i)
        else:
            trimmed_label = x_label
            trimmed_label_val = x_label

    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 3), useMathText=True)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 3), useMathText=True)

    ax.axhline(y=1, color="#696969", linestyle="-")

    for label_ind, label in enumerate(class_labels_list):
        colour = flav_cat[label]["colour"]
        ax.errorbar(
            x=x_value,
            y=store_dict["store_eff_tagger"][:, label_ind],
            yerr=store_dict["store_err_tagger"][:, label_ind],
            xerr=x_err,
            label=flav_cat[label]["legend_label"],
            fmt="o",
            markersize=2.0,
            markeredgecolor=colour,
            markerfacecolor=colour,
            c=colour,
            alpha=1,
        )

    ax.set_xticks(trimmed_label_val)
    ax.set_xticklabels(trimmed_label)
    ax.set_xlim(trimmed_label_val[0], trimmed_label_val[-1])

    # Increase ymax so atlas tag don't cut plot
    if (ymin is None) and (ymax is None):
        plot_ymin, plot_ymax = ax.get_ylim()

    elif ymin is None:
        plot_ymin, _ = ax.get_ylim()

    elif ymax is None:
        _, plot_ymax = ax.get_ylim()

    # Increase the yaxis limit upper part by given factor to fit ATLAS Tag in
    if logy is True:
        ax.set_yscale("log")
        if plot_ymin <= 0:
            plot_ymin = 1e-4
        ax.set_ylim(
            plot_ymin,
            plot_ymax * np.log(plot_ymax / plot_ymin) * 10 * yAxisIncrease,
        )
    else:
        ax.set_ylim(bottom=plot_ymin, top=yAxisIncrease * plot_ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efficiency")
    if minor_ticks_frequency is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
    ax_r = ax.secondary_yaxis("right")
    ax_t = ax.secondary_xaxis("top")
    ax_r.set_yticklabels([])
    ax_t.set_xticks(trimmed_label_val)
    if minor_ticks_frequency is not None:
        ax_t.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
    ax_t.set_xticklabels([])
    if logy:
        ax_r.set_yscale("log")
    ax_r.tick_params(axis="y", direction="in", which="both")
    ax_t.tick_params(axis="x", direction="in", which="both")
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right")
    if UseAtlasTag:
        pas.makeATLAStag(
            ax, fig, AtlasTag, SecondTag + f"\n{ThirdTag}", ymax=yAxisAtlasTag
        )

    fig.tight_layout()
    plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


# TODO: move this to new python API
def plotEfficiencyVariableComparison(
    df_list: list,
    model_labels: list,
    tagger_list: list,
    class_labels_list: list,
    main_class: str,
    variable: str,
    var_bins: np.ndarray,
    plot_name: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    ThirdTag: str = "DL1r",
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    efficiency: float = 0.70,
    frac_values: dict = None,
    centralise_bins: bool = True,
    xticksval: list = None,
    xticks: list = None,
    minor_ticks_frequency: int = None,
    xlabel: str = None,
    logy: bool = False,
    colors: list = None,
    ymin: float = None,
    ymax: float = None,
    dpi: int = 400,
):
    """
    For a given variable (string) in the panda dataframe df, plots
    the eff of each flavour as a function of variable for several taggers
    (discretised in bins as indicated by var_bins input).

    Note: to get a flat efficiency plot, you need to produce a 'tag' column
    in df using FlatEfficiencyPerBin (defined above).

    Parameters
    ----------
    df_list : list
        List with the pd.DataFrame for the given models
    model_labels : list
        Legend labels for the given models
    tagger_list : list
        List with the names of the taggers
    class_labels_list : list
        List indicating the class order
        as defined in the preprocessing!
        WARNING: wrong behaviour if order is different.
    main_class : str
        Name of the main class.
    variable : str
        String of the variable in the dataframe to plot against.
    var_bins : np.ndarray
        Array of the bins to use.
    plot_name : str
        Path, Name and format of the resulting plot file.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample"
    ThirdTag : str, optional
        Additional tag, by default "DL1r"
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag in, by default 1.4
    yAxisAtlasTag : float, optional
        Relative y axis position of the ATLAS Tag, by default 0.9
    efficiency : float, optional
        Working point value, by default 0.70
    frac_values : dict, optional
        Dict with the fraction values inside, by default None
    centralise_bins : bool, optional
        Decide to centralise point in the bins, by default True
    xticksval : list, optional
        List with the xticks values, by default None
    xticks : list, optional
        List with the xticks for the given values, by default None
    minor_ticks_frequency : int, optional
        If given, sets the frequency of minor ticks, by default None
    xlabel : str, optional
        x axis label, by default None
    logy : bool, optional
        Decide, to set the y axis to logarithmic, by default False
    colors : list, optional
        List of costum colours, by default None
    ymin : float, optional
        Minimal y axis value, by default None
    ymax : float, optional
        Maximal y axis value, by default None
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    null_value = np.nan

    store_dict = {}
    for label in class_labels_list[0]:
        store_dict[label] = {
            "store_eff_tagger": np.zeros((len(var_bins) - 1, len(tagger_list))),
            "store_err_tagger": np.zeros((len(var_bins) - 1, len(tagger_list))),
        }

    for tagger_ind, (tagger_data, class_labels) in enumerate(
        zip(df_list, class_labels_list)
    ):
        for i in range(len(var_bins) - 1):
            # all jets in bin
            df_in_bin = tagger_data[
                (var_bins[i] <= tagger_data[variable])
                & (tagger_data[variable] < var_bins[i + 1])
            ]
            total_in_bin = len(df_in_bin)
            # all tagged jets in bin. (note: can also work for other flavour)
            df_in_bin_tagged = df_in_bin.query("tag == 1")
            if total_in_bin != 0:
                index, counts = np.unique(
                    df_in_bin_tagged["labels"].values, return_counts=True
                )
                count_dict = dict(zip(index, counts))

            else:
                count_dict = {}
            # store result for each flavour in store_dict
            for label_ind, label in enumerate(class_labels):
                label_in_bin = len(df_in_bin[df_in_bin["labels"] == label_ind])

                if label_ind in count_dict and label_in_bin != 0:
                    eff = count_dict[label_ind] / label_in_bin
                    err = eff_err(eff, label_in_bin)
                else:
                    eff = null_value
                    err = null_value

                store_dict[label]["store_eff_tagger"][i, tagger_ind] = eff
                store_dict[label]["store_err_tagger"][i, tagger_ind] = err

    extension = plot_name.split(".")[-1]
    plot_name = plot_name[
        : -(len(extension) + 1)
    ]  # remove ".{extension}" from plotname.

    if xlabel is None:
        xlabel = variable
        if variable == "actualInteractionsPerCrossing":
            xlabel = "Actual interactions per bunch crossing"
        elif variable == "pt":
            xlabel = r"$p_T$ [GeV]"

    if frac_values is None:
        ThirdTag = (
            ThirdTag + f"\n{efficiency}% {flav_cat[main_class]['legend_label']} WP"
        )
    else:
        frac_string = ""
        for frac in list(frac_values.keys()):
            frac_string += f"{frac}: {frac_values[frac]} | "
        frac_string = frac_string[:-3]
        ThirdTag = (
            ThirdTag
            + f"\n{efficiency}% {flav_cat[main_class]['legend_label']} WP,"
            + f"\nFractions: {frac_string}"
        )

    # Divide pT values to express in GeV, not MeV
    if variable == "pt":
        var_bins = var_bins / 1000

    x_value, x_err = [], []
    for i in range(len(var_bins[:-1])):
        if centralise_bins:
            x_value.append((var_bins[i] + var_bins[i + 1]) / 2)
        else:
            x_value.append(var_bins[i])
        x_err.append(abs(var_bins[i] - var_bins[i + 1]) / 2)

    x_label = var_bins
    if xticksval is not None:
        trimmed_label_val = xticksval
        trimmed_label = xticksval
        if xticksval is not None:
            trimmed_label = xticks
    else:
        trimmed_label = []
        trimmed_label_val = []
        selected_indices = [int((len(x_label) - 1) / 4) * i for i in range(5)]
        if len(x_label) > 5:
            for i, x_label_i in enumerate(x_label):
                if i in selected_indices:
                    trimmed_label.append(
                        np.format_float_scientific(x_label_i, precision=3)
                    )
                    trimmed_label_val.append(x_label_i)
        else:
            trimmed_label = x_label
            trimmed_label_val = x_label

    if colors is None:
        colors = [f"C{i}" for i in range(len(model_labels))]

    for label in class_labels_list[0]:
        fig, ax = plt.subplots()
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 3), useMathText=True)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 3), useMathText=True)
        if label == main_class:
            ax.axhline(y=1, color="#696969", linestyle="-")

        for tagger_ind, tagger in enumerate(model_labels):
            ax.errorbar(
                x=x_value,
                y=store_dict[label]["store_eff_tagger"][:, tagger_ind],
                yerr=store_dict[label]["store_err_tagger"][:, tagger_ind],
                xerr=x_err,
                label=tagger,
                fmt="o",
                markersize=2.0,
                markeredgecolor=colors[tagger_ind],
                markerfacecolor=colors[tagger_ind],
                c=colors[tagger_ind],
                alpha=1,
                # zorder=10,
            )

        ax.set_xticks(trimmed_label_val)
        ax.set_xticklabels(trimmed_label)
        ax.set_xlim(trimmed_label_val[0], trimmed_label_val[-1])

        # Increase ymax so atlas tag don't cut plot
        if (ymin is None) and (ymax is None):
            plot_ymin, plot_ymax = ax.get_ylim()

        elif ymin is None:
            plot_ymin, _ = ax.get_ylim()

        elif ymax is None:
            _, plot_ymax = ax.get_ylim()

        # Increase the yaxis limit upper part by given factor to fit ATLAS Tag in
        if logy is True:
            ax.set_yscale("log")
            if plot_ymin <= 0:
                plot_ymin = 1e-4
            ax.set_ylim(
                plot_ymin,
                plot_ymax * np.log(plot_ymax / plot_ymin) * 10 * yAxisIncrease,
            )
        else:
            ax.set_ylim(bottom=plot_ymin, top=yAxisIncrease * plot_ymax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Efficiency {flav_cat[label]['legend_label']}")
        if minor_ticks_frequency is not None:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
        ax_r = ax.secondary_yaxis("right")
        ax_t = ax.secondary_xaxis("top")
        ax_r.set_yticklabels([])
        ax_t.set_xticks(trimmed_label_val)
        if minor_ticks_frequency is not None:
            ax_t.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
        ax_t.set_xticklabels([])
        if logy:
            ax_r.set_yscale("log")
        ax_r.tick_params(axis="y", direction="in", which="both")
        ax_t.tick_params(axis="x", direction="in", which="both")
        ax.grid(color="grey", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right")
        if UseAtlasTag:
            pas.makeATLAStag(
                ax,
                fig,
                AtlasTag,
                SecondTag + f"\n{ThirdTag}",
                ymax=yAxisAtlasTag,
            )

        fig.tight_layout()
        plt.savefig(plot_name + f"_{label}.{extension}", transparent=True, dpi=dpi)
        plt.close()


def plot_pt_dependence(
    df_list: list,
    tagger_list: list,
    model_labels: list,
    plot_name: str,
    class_labels: list,
    main_class: str,
    flavour: str,
    working_point: float = 0.77,
    disc_cut: float = None,
    fixed_eff_bin: bool = False,
    bin_edges: list = None,
    wp_line: bool = False,
    grid: bool = False,
    colours: list = None,
    alpha: float = 0.8,
    trans: bool = True,
    linewidth: float = 1.6,
    **kwargs,
):
    """For a given list of models, plot the b-eff, l and c-rej as a function of jet pt.

    Parameters
    ----------
    df_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    model_labels : list
        List of the labels which are to be used in the plot.
    plot_name : str
        Path, Name and format of the resulting plot file.
    class_labels : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class (= signal class). For b-tagging obviously "bjets"
    flavour : str
        Class that is to be plotted. For all non-signal classes, this
        will be the rejection and for the signal class, this will be
        the efficiency.
    working_point : float, optional
        Working point which is to be used, by default 0.77.
    disc_cut : float, optional
        Set a discriminant cut value for all taggers/models.
    fixed_eff_bin : bool, optional
        Calculate the WP cut on the discriminant per bin, by default None.
    bin_edges : list, optional
        As the name says, the edges of the bins used. Will be set
        automatically, if None. By default None.
    wp_line : bool, optional
        Print a WP line in the upper plot, by default False.
    grid : bool, optional
        Use a grid in the plots, by default False
    colours : list, optional
        Custom colour list for the different models, by default None
    alpha : float, optional
        Value for visibility of the plot lines, by default 0.8
    trans : bool, optional
        saving figure with transparent background, by default True
    linewidth : float, optional
        Define the linewidth of the plotted lines, by default 1.6
    **kwargs : kwargs
        kwargs for `var_vs_eff_plot` function

    Raises
    ------
    ValueError
        If deprecated options are given.
    """
    if "colors" in kwargs:
        colours = kwargs["colors"]
        kwargs.pop("colors")
    if "WP" in kwargs:
        working_point = kwargs["WP"]
        kwargs.pop("WP")
    if "Disc_Cut_Value" in kwargs:
        disc_cut = kwargs["Disc_Cut_Value"]
        kwargs.pop("Disc_Cut_Value")
    if "Fixed_WP_Bin" in kwargs:
        fixed_eff_bin = kwargs["Fixed_WP_Bin"]
        kwargs.pop("Fixed_WP_Bin")
    if "Grid" in kwargs:
        grid = kwargs["Grid"]
        kwargs.pop("Grid")
    if "WP_Line" in kwargs:
        wp_line = kwargs["WP_Line"]
        kwargs.pop("WP_Line")
    kwargs = translate_kwargs(kwargs)

    deprecated = {
        "SWP_Comparison",
        "SWP_label_list",
        "Passed",
        "binomialErrors",
        "frameon",
        "Ratio_Cut",
    } & set(kwargs.keys())
    if deprecated:
        raise ValueError(
            f"The options {list(deprecated)} are deprecated. "
            "Please use the plotting python API."
        )
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = r"$p_T$ [GeV]"

    if bin_edges is None:
        bin_edges = [0, 20, 50, 90, 150, 300, 1000]
    # Get global config of the classes
    flav_cat = global_config.flavour_categories
    n_curves = len(tagger_list)
    if colours is None:
        colours = [f"C{i}" for i in range(n_curves)]

    # Get the indicies of the flavours
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

    # check length
    # TODO: change in python 3.10 -> add to zip() with strict=True argument
    if not all(
        len(elem) == n_curves
        for elem in [
            df_list,
            model_labels,
            tagger_list,
            colours,
        ]
    ):
        raise ValueError("Passed lists do not have same length.")

    mode = "bkg_rej"
    y_label = f'{flav_cat[flavour]["legend_label"]} rejection'
    if main_class == flavour:
        mode = "sig_eff"
        y_label = f'{flav_cat[flavour]["legend_label"]} efficiency'

    plot_pt = var_vs_eff_plot(
        mode=mode,
        ylabel=y_label,
        **kwargs,
    )
    # Redefine Second Tag with inclusive or fixed tag
    if fixed_eff_bin:
        plot_pt.atlas_second_tag = (
            f"{plot_pt.atlas_second_tag}\nConstant "
            rf"$\epsilon_b$ = {int(working_point * 100)}% per bin"
        )
    else:
        plot_pt.atlas_second_tag = (
            f"{plot_pt.atlas_second_tag}\nInclusive "
            rf"$\epsilon_b$ = {int(working_point * 100)}%"
        )

    for i, (df_results, model_label, tagger, colour) in enumerate(
        zip(df_list, model_labels, tagger_list, colours)
    ):
        # Get jet pts
        jetPts = df_results["pt"] / 1e3
        # Get truth labels
        is_signal = df_results["labels"] == index_dict[main_class]
        is_bkg = (
            df_results["labels"] == index_dict[flavour] if mode == "bkg_rej" else None
        )
        disc = df_results[f"disc_{tagger}"]
        plot_pt.add(
            var_vs_eff(
                x_var_sig=jetPts[is_signal],
                disc_sig=disc[is_signal],
                x_var_bkg=jetPts[is_bkg] if mode == "bkg_rej" else None,
                disc_bkg=disc[is_bkg] if mode == "bkg_rej" else None,
                bins=bin_edges,
                wp=working_point,
                disc_cut=disc_cut,
                fixed_eff_bin=fixed_eff_bin,
                label=model_label,
                colour=colour,
                alpha=alpha,
                linewidth=linewidth,
            ),
            reference=i == 0,
        )

    plot_pt.draw()
    # Set grid
    if grid is True:
        plot_pt.set_grid()
    # Set WP Line
    if wp_line is True:
        plot_pt.draw_hline(working_point)
        if main_class != flavour:
            logger.warning(
                "You set `wp_line` to True but you are not looking at the singal "
                "efficiency. It will probably not be visible on your plot."
            )
    plot_pt.savefig(plot_name, transparent=trans)


def plotROCRatio(
    df_results_list: list,
    tagger_list: list,
    rej_class_list: list,
    labels: list,
    main_class: str,
    plot_name: str,
    df_eff_key: str = "effs",
    draw_errors: bool = True,
    rlabel: str = "Ratio",
    labelpad: int = None,
    WorkingPoints: list = None,
    same_height_WP: bool = True,
    linestyles: list = None,
    colours: list = None,
    n_test=None,
    ratio_id=0,
    **kwargs,
):
    """Plotting the rejection curve for a given background class
    for the given models/taggers against the main_class efficiency.
    A ratio plot (first given model is the reference) is plotted in
    a subpanel underneath.

    Parameters
    ----------
    df_results_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    rej_class_list : list
        List of the class rejection which is to be plotted for each model.
    labels : list
        List of labels for the given models.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    plot_name : str
        Path, Name and format of the resulting plot file.
    df_eff_key : str, optional
        Dict key under which the efficiencies of the main class are saved,
        by default "effs"
    draw_errors : bool, optional
        Binominal errors on the lines, by default True
    rlabel : str, optional
        The label for the y-axis for the ratio panel, by default "Ratio"
    labelpad : int, optional
        Spacing in points from the axes bounding box including
        ticks and tick labels, by default None
    WorkingPoints : list, optional
        List of working points which are to be plotted as
        vertical lines in the plot, by default None
    same_height_WP : bool, optional
        Decide, if all working points lines have the same height or
        not, by default True
    linestyles : list, optional
        List of linestyles to use for the given models, by default None
    colours : list, optional
        List of linecolors to use for the given models, by default None
    n_test : [type], optional
        A list of the same length as class_rejections, with the number of
        events used to calculate the background efficiencies.
        We need this To calculate the binomial errors on the background
        rejection, using the formula given by
        http://home.fnal.gov/~paterno/images/effic.pdf, by default 0
    ratio_id : int, optional
        List to which given model the ratio is calulcated, by default 0
    **kwargs : kwargs
        kwargs passed to roc_plot


    Raises
    ------
    ValueError
        if n_test not int, float of given for each roc
    ValueError
        if lists don't have the same length
    """
    n_rocs = len(df_results_list)
    # maintain backwards compatibility
    if "nTest" in kwargs:
        if n_test is None:
            n_test = kwargs["nTest"]
        kwargs.pop("nTest")
    if "colors" in kwargs:
        if colours is None:
            colours = kwargs["colors"]
        kwargs.pop("colors")
    if "binomialErrors" in kwargs:
        if draw_errors is None:
            draw_errors = kwargs["binomialErrors"]
        kwargs.pop("binomialErrors")
    if "styles" in kwargs:
        if linestyles is None:
            linestyles = kwargs["styles"]
        kwargs.pop("styles")
    kwargs = translate_kwargs(kwargs)

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    if draw_errors is True:
        # Check if n_test is provided in all samples
        if n_test is None:
            n_test_in_file = ["N_test" in df_results for df_results in df_results_list]

            if not all(n_test_in_file):
                logger.error(
                    "Requested binomialErrors, but not all models have n_test. "
                    "Will NOT plot rej errors."
                )
                draw_errors = False

        elif isinstance(n_test, (int, float)):
            n_test = [n_test] * len(df_results_list)
        elif isinstance(n_test, list):
            if len(n_test) != len(df_results_list):
                raise ValueError(
                    "The provided `n_test` do not have the same length as the "
                    "`df_results_list`."
                )

    if linestyles is None:
        linestyles = ["-" for _ in labels]
    if colours is None:
        colours = [f"C{i}" for i in range(n_rocs)]

    # check length
    # TODO: change in python 3.10 -> add to zip() with strict=True argument
    if not all(
        len(elem) == n_rocs
        for elem in [
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
        ]
    ):
        raise ValueError("Passed lists do not have same length.")

    plot_roc = roc_plot(
        n_ratio_panels=1,
        ylabel=f'{flav_cat[rej_class_list[0]]["legend_label"]} Rejection',
        xlabel=f'{flav_cat[main_class]["legend_label"]} Efficiency',
        **kwargs,
    )
    plot_roc.set_ratio_class(ratio_panel=1, rej_class=rej_class_list[0], label=rlabel)
    if WorkingPoints is not None:
        plot_roc.draw_wps(WorkingPoints, same_height_WP)

    # Loop over the models with the different settings for each model
    for i, (df_results, tagger, rej_class, label, linestyle, colour, nte) in enumerate(
        zip(
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
            linestyles,
            colours,
            n_test,
        )
    ):
        roc_curve = roc(
            df_results[df_eff_key],
            df_results[f"{tagger}_{rej_class}_rej"],
            n_test=nte,
            rej_class=rej_class,
            signal_class=main_class,
            label=label,
            colour=colour,
            linestyle=linestyle,
        )
        plot_roc.add_roc(roc_curve, reference=i == ratio_id)

    plot_roc.draw(labelpad=labelpad)
    plot_roc.savefig(plot_name)


def plotROCRatioComparison(
    df_results_list: list,
    tagger_list: list,
    rej_class_list: list,
    labels: list,
    plot_name: str,
    main_class: str = "bjets",
    df_eff_key: str = "effs",
    draw_errors: bool = True,
    labelpad: int = None,
    WorkingPoints: list = None,
    same_height_WP: bool = True,
    linestyles: list = None,
    colours: list = None,
    n_test=None,
    reference_ratio: list = None,
    **kwargs,
):
    """Plotting the rejection curve for a given background class
    for the given models/taggers against the main_class efficiency.
    A ratio plot (first given model is the reference) is plotted in
    a subpanel underneath.

    Parameters
    ----------
    df_results_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    rej_class_list : list
        List of the class rejection which is to be plotted for each model.
    labels : list
        List of labels for the given models.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    plot_name : str
        Path, Name and format of the resulting plot file.
    df_eff_key : str, optional
        Dict key under which the efficiencies of the main class are saved,
        by default "effs"
    draw_errors : bool, optional
        Binominal errors on the lines, by default True
    labelpad : int, optional
        Spacing in points from the axes bounding box including
        ticks and tick labels, by default None
    WorkingPoints : list, optional
        List of working points which are to be plotted as
        vertical lines in the plot, by default None
    same_height_WP : bool, optional
        Decide, if all working points lines have the same height or
        not, by default True
    linestyles : list, optional
        List of linestyles to use for the given models, by default None
    colours : list, optional
        List of linecolors to use for the given models, by default None
    n_test : [type], optional
        A list of the same length as class_rejections, with the number of
        events used to calculate the background efficiencies.
        We need this To calculate the binomial errors on the background
        rejection, using the formula given by
        http://home.fnal.gov/~paterno/images/effic.pdf, by default 0
    reference_ratio : list, optional
        List of bools indicating which roc used as reference for ratio calculation,
        by default None
    **kwargs : kwargs
        kwargs passed to roc_plot

    Raises
    ------
    ValueError
        if n_test not int, floar of given for each roc
    ValueError
        if lists don't have the same length
    """
    n_rocs = len(df_results_list)
    # maintain backwards compatibility
    if "nTest" in kwargs:
        if n_test is None:
            n_test = kwargs["nTest"]
        kwargs.pop("nTest")
    if "colors" in kwargs:
        if colours is None:
            colours = kwargs["colors"]
            # remnant of old implementation passing empty list as default
            if kwargs["colors"] == []:
                colours = None
        kwargs.pop("colors")
    if "binomialErrors" in kwargs:
        if draw_errors is None:
            draw_errors = kwargs["binomialErrors"]
        kwargs.pop("binomialErrors")
    if "ratio_id" in kwargs:
        if reference_ratio is None and kwargs["ratio_id"] is not None:
            # if old keyword is used the syntax was also different
            # translating this now into the new syntax
            # the old syntax looked like
            # ratio_id = [0, 0, 1, 1]
            # rej_class_list = ['ujets', 'ujets', 'cjets', 'cjets']
            # tagger_list = ['RNNIP', 'DIPS', 'RNNIP', 'DIPS']
            # in that case the first entry was used for the upper ratio and the
            #  3rd entry for the 2nd ratio
            # in the new syntax this would mean
            # reference_ratio = [True, False, True, False]
            reference_ratio = []
            _tmp_ratio_id = []
            for elem in kwargs["ratio_id"]:
                reference_ratio.append(elem not in _tmp_ratio_id)
                _tmp_ratio_id.append(elem)
        kwargs.pop("ratio_id")
    kwargs = translate_kwargs(kwargs)

    # catching default value as in old implementation to maintain backwards
    # compatibility
    if reference_ratio is None:
        reference_ratio = []
        _tmp_ratio_id = []
        for elem in rej_class_list:
            reference_ratio.append(elem not in _tmp_ratio_id)
            _tmp_ratio_id.append(elem)

    # remnant of old implementation passing empty list as default
    if linestyles == []:
        linestyles = None

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Loop over the given rejection types and add them to a lists
    flav_list = list(OrderedDict.fromkeys(rej_class_list))
    if len(flav_list) > 2:
        raise ValueError("Can't plot more than 2 rejections!")

    # Append a linestyles for each model determined by the rejections
    # with solid lines or dashed dotted lines
    if linestyles is None:
        linestyles = [
            "-" if elem == flav_list[0] else (0, (3, 1, 1, 1))
            for elem in rej_class_list
        ]

    # Create list for the models
    model_list = list(OrderedDict.fromkeys(labels))

    # Fill in the colors for the models given
    if colours is None:
        model_colours = {model: f"C{i}" for i, model in enumerate(model_list)}
        colours = [model_colours[elem] for elem in labels]

    if draw_errors is True:
        # Check if n_test is provided in all samples
        if n_test is None:
            n_test_in_file = ["N_test" in df_results for df_results in df_results_list]

            if not all(n_test_in_file):
                logger.error(
                    "Requested binomialErrors, but not all models have n_test. "
                    "Will NOT plot rej errors."
                )
                draw_errors = False

        elif isinstance(n_test, (int, float)):
            n_test = [n_test] * len(df_results_list)
        elif isinstance(n_test, list):
            if len(n_test) != len(df_results_list):
                raise ValueError(
                    "The provided `n_test` do not have the same length as the "
                    "`df_results_list`."
                )

    # check length
    # TODO: change in python 3.10 -> add to zip() with strict=True argument
    if not all(
        len(elem) == n_rocs
        for elem in [
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
            linestyles,
            colours,
            n_test,
            reference_ratio,
        ]
    ):
        raise ValueError("Passed lists do not have same length.")

    plot_roc = roc_plot(
        n_ratio_panels=2,
        ylabel="Background rejection",
        xlabel=f'{flav_cat[main_class]["legend_label"]} efficiency',
        **kwargs,
    )
    plot_roc.set_ratio_class(
        ratio_panel=1,
        rej_class=flav_list[0],
        label=f'{flav_cat[flav_list[0]]["legend_label"]} ratio',
    )
    plot_roc.set_ratio_class(
        ratio_panel=2,
        rej_class=flav_list[1],
        label=f'{flav_cat[flav_list[1]]["legend_label"]} ratio',
    )
    if WorkingPoints is not None:
        plot_roc.draw_wps(WorkingPoints, same_height_WP)

    # Loop over the models with the different settings for each model
    for df_results, tagger, rej_class, label, linestyle, colour, nte, ratio_ref in zip(
        df_results_list,
        tagger_list,
        rej_class_list,
        labels,
        linestyles,
        colours,
        n_test,
        reference_ratio,
    ):
        roc_curve = roc(
            df_results[df_eff_key],
            df_results[f"{tagger}_{rej_class}_rej"],
            n_test=nte,
            rej_class=rej_class,
            signal_class=main_class,
            label=label,
            colour=colour,
            linestyle=linestyle,
        )
        plot_roc.add_roc(roc_curve, reference=ratio_ref)

    plot_roc.set_leg_rej_labels(
        flav_list[0],
        label=f'{flav_cat[flav_list[0]]["legend_label"]} rejection',
    )
    plot_roc.set_leg_rej_labels(
        flav_list[1],
        label=f'{flav_cat[flav_list[1]]["legend_label"]} rejection',
    )
    plot_roc.draw(
        labelpad=labelpad,
    )
    plot_roc.savefig(plot_name)


def plotSaliency(
    maps_dict: dict,
    plot_name: str,
    title: str,
    target_beff: float = 0.77,
    jet_flavour: str = "bjets",
    PassBool: bool = True,
    nFixedTrks: int = 8,
    fontsize: int = 14,
    xlabel: str = "Tracks sorted by $s_{d0}$",
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = r"$\sqrt{s}$ = 13 TeV, $t\bar{t}$ PFlow Jets",
    yAxisAtlasTag: float = 0.925,
    FlipAxis: bool = False,
    dpi: int = 400,
):
    """Plot the saliency map given in maps_dict.

    Parameters
    ----------
    maps_dict : dict
        Dict with the saliency values inside
    plot_name : str
        Path, Name and format of the resulting plot file.
    title : str
        Title of the plot
    target_beff : float, optional
        Working point to use, by default 0.77
    jet_flavour : str, optional
        Class which is to be plotted, by default "bjets"
    PassBool : bool, optional
        Decide, if the jets need to pass (True) or fail (False)
        the working point cut, by default True
    nFixedTrks : int, optional
        Decide, how many tracks the jets need to have, by default 8
    fontsize : int, optional
        Fontsize to use in the title, legend, etc, by default 14
    xlabel : str, optional
        x-label, by default "Tracks sorted by {d0}$"
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag, by default
        "$sqrt{s}$ = 13 TeV, $t bar{t}$ PFlow Jets"
    yAxisAtlasTag : float, optional
        y position of the ATLAS Tag, by default 0.925
    FlipAxis : bool, optional
        Decide, if the x- and y-axis are switched, by default False
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Transform to percent
    target_beff = 100 * target_beff

    # Little Workaround
    AtlasTag = " " + AtlasTag

    gradient_map = maps_dict[f"{int(target_beff)}_{jet_flavour}_{PassBool}"]

    colorScale = np.max(np.abs(gradient_map))
    cmaps = {
        "ujets": "RdBu",
        "cjets": "PuOr",
        "bjets": "PiYG",
    }

    nFeatures = gradient_map.shape[0]

    if FlipAxis is True:
        fig = plt.figure(figsize=(0.7 * nFeatures, 0.7 * nFixedTrks))
        gradient_map = np.swapaxes(gradient_map, 0, 1)

        plt.yticks(
            np.arange(nFixedTrks),
            np.arange(1, nFixedTrks + 1),
            fontsize=fontsize,
        )

        plt.ylabel(xlabel, fontsize=fontsize)
        plt.ylim(-0.5, nFixedTrks - 0.5)

        # ylabels. Order must be the same as in the Vardict
        xticklabels = [
            "$s_{d0}$",
            "$s_{z0}$",
            "PIX1 hits",
            "IBL hits",
            "shared IBL hits",
            "split IBL hits",
            "shared pixel hits",
            "split pixel hits",
            "shared SCT hits",
            "log" + r" $p_T^{frac}$",
            "log" + r" $\mathrm{\Delta} R$",
            "nPixHits",
            "nSCTHits",
            "$d_0$",
            r"$z_0 \sin \theta$",
        ]

        plt.xticks(
            np.arange(nFeatures),
            xticklabels[:nFeatures],
            rotation=45,
            fontsize=fontsize,
        )

    else:
        fig = plt.figure(figsize=(0.7 * nFixedTrks, 0.7 * nFeatures))

        plt.xticks(
            np.arange(nFixedTrks),
            np.arange(1, nFixedTrks + 1),
            fontsize=fontsize,
        )

        plt.xlabel(xlabel, fontsize=fontsize)
        plt.xlim(-0.5, nFixedTrks - 0.5)

        # ylabels. Order must be the same as in the Vardict
        yticklabels = [
            "$s_{d0}$",
            "$s_{z0}$",
            "PIX1 hits",
            "IBL hits",
            "shared IBL hits",
            "split IBL hits",
            "shared pixel hits",
            "split pixel hits",
            "shared SCT hits",
            "log" + r" $p_T^{frac}$",
            "log" + r" $\mathrm{\Delta} R$",
            "nPixHits",
            "nSCTHits",
            "$d_0$",
            r"$z_0 \sin \theta$",
        ]

        plt.yticks(np.arange(nFeatures), yticklabels[:nFeatures], fontsize=fontsize)

    im = plt.imshow(
        gradient_map,
        cmap=cmaps[jet_flavour],
        origin="lower",
        vmin=-colorScale,
        vmax=colorScale,
    )

    plt.title(title, fontsize=fontsize)

    ax = plt.gca()

    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax,
            fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Plot colorbar and set size to graph size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(im, cax=cax)
    colorbar.ax.set_title(
        r"$\frac{\mathrm{\partial} D_{b}}{\mathrm{\partial} x_{ik}}$",
        size=1.6 * fontsize,
    )

    # Set the fontsize of the colorbar yticks
    for t in colorbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    # Save the figure
    plt.savefig(plot_name, transparent=True, bbox_inches="tight", dpi=dpi)


def plot_score(
    df_results: pd.DataFrame,
    plot_name: str,
    tagger_name: str,
    class_labels: list,
    main_class: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    WorkingPoints: list = None,
    nBins: int = 50,
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    xlabel: str = None,
    WorkingPoints_Legend: bool = False,
    dpi: int = 400,
    xmin: float = None,
    xmax: float = None,
):
    """Plot the discriminant score for a given model

    Parameters
    ----------
    df_results : pd.DataFrame
        Pandas DataFrame with the discriminant scores of the jets inside.
    plot_name : str
        Path, Name and format of the resulting plot file.
    tagger_name : str
        Name of the tagger/model that is to be plotted.
    class_labels : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample"
    WorkingPoints : list, optional
        List of working points which are to be plotted as
        vertical lines in the plot, by default None
    nBins : int, optional
        Number of bins to use, by default 50
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag, by default 1.3
    yAxisAtlasTag : float, optional
        y position of the ATLAS tag, by default 0.9
    xlabel : str, optional
        x-label, by default None
    WorkingPoints_Legend : bool, optional
        Show the working points in the legend, by default False
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

    # Calculate the binning for all flavours
    _, Binning = np.histogram(
        df_results.query(f"labels=={index_dict[main_class]}")[f"disc_{tagger_name}"],
        bins=nBins,
    )

    # Clear the figure and init a new one
    plt.clf()
    fig = plt.figure()
    ax = fig.gca()

    for flavour in class_labels:
        # Select correct jets
        flav_tracks = df_results.query(f"labels=={index_dict[flavour]}")[
            f"disc_{tagger_name}"
        ]

        # Calculate bins
        bins, weights, unc, band = hist_w_unc(
            a=flav_tracks,
            bins=Binning,
        )

        plt.hist(
            x=bins[:-1],
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.0,
            color=flav_cat[flavour]["colour"],
            stacked=False,
            fill=False,
            label=flav_cat[flavour]["legend_label"],
        )

        # Check for last item so the label for the legend is
        # only printed once
        if flavour == class_labels[-1]:
            plt.hist(
                x=bins[:-1],
                bins=bins,
                bottom=band,
                weights=unc * 2,
                label="stat. unc.",
                **global_config.hist_err_style,
            )

        else:
            plt.hist(
                x=bins[:-1],
                bins=bins,
                bottom=band,
                weights=unc * 2,
                **global_config.hist_err_style,
            )

    # Increase ymax so atlas tag don't cut plot
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)

    # Define handles and labels
    handles, _ = plt.gca().get_legend_handles_labels()

    # Set WP vertical lines if given in config
    if WorkingPoints is not None:

        # Iterate over WPs
        for WP in WorkingPoints:

            # Calculate x value of WP line
            x_value = np.percentile(
                df_results.query(f"labels=={index_dict[main_class]}")[
                    f"disc_{tagger_name}"
                ],
                (1 - WP) * 100,
            )

            # Add WP cut values to legend if wanted
            if WorkingPoints_Legend is True:
                handles.extend(
                    [
                        mpatches.Patch(
                            color="w",
                            label=f"{int(WP * 100)}% WP cut value: {x_value:.2f}",
                        )
                    ]
                )

            # Draw WP line
            plt.vlines(
                x=x_value,
                ymin=ymin,
                ymax=WP * ymax,
                colors="#990000",
                linestyles="dashed",
                linewidth=2.0,
            )

            # Set the number above the line
            ax.annotate(
                f"{int(WP * 100)}%",
                xy=(x_value, WP * ymax),
                xytext=(x_value, WP * ymax),
                textcoords="offset points",
                ha="center",
                va="bottom",
                size=10,
            )

    # Init legend
    plt.legend(handles=handles)

    if xlabel:
        plt.xlabel(
            xlabel,
            horizontalalignment="right",
            x=1.0,
        )

    else:
        plt.xlabel(
            f'{flav_cat[main_class]["legend_label"]} discriminant',
            horizontalalignment="right",
            x=1.0,
        )
    plt.ylabel("Normalised number of jets", horizontalalignment="right", y=1.0)

    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.tight_layout()
    plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


def plot_score_comparison(
    df_list: list,
    model_labels: list,
    tagger_list: list,
    class_labels_list: list,
    main_class: str,
    plot_name: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    WorkingPoints: list = None,
    tagger_for_WP: str = None,
    nBins: int = 50,
    figsize: list = None,
    labelpad: int = None,
    labelFontSize: int = 10,
    legFontSize: int = 10,
    loc_legend: str = "best",
    ncol: int = 2,
    Ratio_Cut=None,
    which_axis="left",
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    ycolor: str = "black",
    title: str = None,
    dpi: int = 400,
):
    """Plot multiple discriminant scores from different models for the same
    jets in one plot with a ratio plot in the subpanel.

    Parameters
    ----------
    df_list : list
        List with the pandas DataFrames inside.
    model_labels : list
        List of labels for the given models
    tagger_list : list
        List of tagger names of the used taggers
    class_labels_list : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    plot_name : str
        Path, Name and format of the resulting plot file.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample"
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag, by default 1.3
    yAxisAtlasTag : float, optional
        y position of the ATLAS tag, by default 0.9
    WorkingPoints : list, optional
        List of working points which are to be plotted as
        vertical lines in the plot, by default None
    tagger_for_WP : str, optional
        Tagger which is used for working point calculation, by default None
    nBins : int, optional
        Number of bins to use, by default 50
    figsize : list, optional
        Size of the resulting figure as a list with two elements. First
        is the width, second the height. By default None.
    labelpad : int, optional
        Spacing in points from the axes bounding box including
        ticks and tick labels, by default None
    labelFontSize : int, optional
        Fontsize of the axis labels, by default 10
    legFontSize : int, optional
        Fontsize of the legend, by default 10
    loc_legend : str, optional
        Position of the legend in the plot, by default "best"
    ncol : int, optional
        Number of columns in the legend, by default 2
    Ratio_Cut : list, optional
        List of the lower and upper y-limit for the ratio plot,
        by default None
    which_axis : list, optional
        List which y-axis to use for the given models, by default "left"
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    ymin : float, optional
        Minimum value of the y-axis, by default None
    ymax : float, optional
        Maximum value of the y-axis, by default None
    ycolor : str, optional
        y-axis color, by default "black"
    title : str, optional
        Title over the plot, by default None
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels_list[0])}

    if not isinstance(which_axis, list):
        which_axis = [which_axis] * len(df_list)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    if figsize is None:
        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(8, 1, figure=fig)
    axis_dict["left"] = {}
    axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
    axis_dict["left"]["ratio"] = fig.add_subplot(
        gs[6:, 0], sharex=axis_dict["left"]["top"]
    )

    # Get binning for the plot
    _, Binning = np.histogram(
        df_list[0].query(f"labels=={index_dict[main_class]}")[f"disc_{tagger_list[0]}"],
        bins=nBins,
    )

    # Init bincout and unc dict for ratio calculation
    bincounts = {}
    bincounts_unc = {}

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    for i, (
        df_results,
        linestyle,
        which_a,
        model_label,
        tagger,
        class_labels,
    ) in enumerate(
        zip(
            df_list,
            linestyles,
            which_axis,
            model_labels,
            tagger_list,
            class_labels_list,
        )
    ):
        # Get index dict
        index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

        for flavour in class_labels:
            # Select correct jets
            flav_tracks = df_results.query(f"labels=={index_dict[flavour]}")[
                f"disc_{tagger}"
            ]

            bins, weights, unc, band = hist_w_unc(
                a=flav_tracks,
                bins=Binning,
            )

            hist_counts, _, _ = axis_dict[which_a]["top"].hist(
                x=bins[:-1],
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2.0,
                linestyle=linestyle,
                color=flav_cat[flavour]["colour"],
                stacked=False,
                fill=False,
                label=flav_cat[flavour]["legend_label"] + f" {model_label}",
            )

            if (flavour == class_labels[-1]) and i == 0:
                axis_dict[which_a]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    label="stat. unc.",
                    **global_config.hist_err_style,
                )

            else:
                axis_dict[which_a]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    **global_config.hist_err_style,
                )

            bincounts.update({f"{flavour}{i}": hist_counts})
            bincounts_unc.update({f"{flavour}{i}": unc})

        # Start ratio plot
        if i != 0:
            for flavour in class_labels:

                # Calculate the step and step_unc for ratio
                step, step_unc = hist_ratio(
                    numerator=bincounts[f"{flavour}{i}"],
                    denominator=bincounts[f"{flavour}0"],
                    numerator_unc=bincounts_unc[f"{flavour}{i}"],
                    denominator_unc=bincounts_unc[f"{flavour}0"],
                )

                axis_dict["left"]["ratio"].step(
                    x=Binning,
                    y=step,
                    color=flav_cat[flavour]["colour"],
                    linestyle=linestyles[i],
                )

                axis_dict["left"]["ratio"].fill_between(
                    x=Binning,
                    y1=step - step_unc,
                    y2=step + step_unc,
                    step="pre",
                    facecolor="none",
                    edgecolor=global_config.hist_err_style["edgecolor"],
                    linewidth=global_config.hist_err_style["linewidth"],
                    hatch=global_config.hist_err_style["hatch"],
                )

    # Add axes, titels and the legend
    axis_dict["left"]["top"].set_ylabel(
        "Normalised Number of Jets",
        fontsize=labelFontSize,
        horizontalalignment="right",
        y=1.0,
        color=ycolor,
    )
    if title is not None:
        axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["ratio"].set_xlabel(
        f'{flav_cat[main_class]["legend_label"]} discriminant',
        horizontalalignment="right",
        x=1.0,
    )

    axis_dict["left"]["ratio"].set_ylabel(
        "Ratio",
        labelpad=labelpad,
        fontsize=labelFontSize,
    )

    if Ratio_Cut is not None:
        axis_dict["left"]["ratio"].set_ylim(bottom=Ratio_Cut[0], top=Ratio_Cut[1])

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    if xmin is not None:
        axis_dict["left"]["top"].set_xlim(left=xmin)

    else:
        axis_dict["left"]["top"].set_xlim(left=Binning[0])

    if xmax is not None:
        axis_dict["left"]["top"].set_xlim(right=xmax)

    else:
        axis_dict["left"]["top"].set_xlim(right=Binning[-1])

    if ymin is not None:
        axis_dict["left"]["top"].set_ylim(bottom=ymin)

    if ymax is not None:
        axis_dict["left"]["top"].set_ylim(top=ymax)

    # Add black line at one
    axis_dict["left"]["ratio"].axhline(
        y=1,
        xmin=0,
        xmax=1,
        color="black",
        alpha=0.5,
    )

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    axis_dict["left"]["top"].set_ylim(
        left_y_limits[0], left_y_limits[1] * yAxisIncrease
    )

    axis_dict["left"]["top"].legend(
        loc=loc_legend,
        fontsize=legFontSize,
        ncol=ncol,
    )  # , title="DL1r")

    # Set WP vertical lines if given in config
    if WorkingPoints is not None:

        # Iterate over WPs
        for WP in WorkingPoints:

            # Get tagger for WP calculation
            tagger_name_WP = tagger_list[0] if tagger_for_WP is None else tagger_for_WP

            # Calculate x value of WP line
            x_value = np.percentile(
                df_list[0].query(f"labels=={index_dict[main_class]}")[
                    f"disc_{tagger_name_WP}"
                ],
                (1 - WP) * 100,
            )

            # Draw WP line
            axis_dict["left"]["top"].axvline(
                x=x_value,
                ymin=0,
                ymax=0.75 * WP,
                color="red",
                linestyle="dashed",
                linewidth=1.0,
            )

            # Draw WP line
            axis_dict["left"]["ratio"].axvline(
                x=x_value,
                ymin=0,
                ymax=1,
                color="red",
                linestyle="dashed",
                linewidth=1.0,
            )

            # Set the number above the line
            axis_dict["left"]["top"].annotate(
                f"{int(WP * 100)}%",
                xy=(
                    x_value,
                    0.75 * WP * axis_dict["left"]["top"].get_ylim()[1],
                ),
                xytext=(
                    x_value,
                    0.75 * WP * axis_dict["left"]["top"].get_ylim()[1],
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                size=10,
            )

    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.tight_layout()
    plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


def plotFractionScan(
    data,
    label,
    plot_name,
    x_val,
    y_val,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal",
    SecondTag: str = "$\\sqrt{s}=13$ TeV, PFlow Jets, $t\\bar{t}$",
    dpi: int = 400,
):
    """DEPRECATED. Plots a 2D heatmap of rej for a given eff
        (frac flavour X vs frac flavour Y).
    Data needs to be a list of numpy arrays with the flavour

    Parameters
    ----------
    data : _type_
        data
    label : _type_
        label
    plot_name : _type_
        plot name
    x_val : _type_
        x values
    y_val : _type_
        y values
    UseAtlasTag : bool, optional
        adding ATLAS tag, by default True
    AtlasTag : str, optional
        ATLAS tag, by default "Internal"
    SecondTag : str, optional
        second line of ATLAS tag
    dpi : int, optional
        figure dpi, by default 400
    """
    if label == "umami_crej":
        draw_label = r"$c$-Jet Rejection from b ($1/\epsilon_{c}$)"
    elif label == "umami_urej":
        draw_label = r"Light-Flavour Jet Rejection  from b ($1/\epsilon_{l}$)"
    elif label == "umami_taurej":
        draw_label = r"Tau-Jet Rejection  from b ($1/\epsilon_{\tau}$)"

    elif label == "umami_brejC":
        draw_label = r"$b$-Jet Rejection from c ($1/\epsilon_{b}$)"
    elif label == "umami_urejC":
        draw_label = r"Light-Jet Rejection from c ($1/\epsilon_{l}$)"
    elif label == "umami_taurejC":
        draw_label = r"Tau-Jet Rejection from c ($1/\epsilon_{\tau}$)"
    else:
        logger.error("Problem")
        # TODO: add something more specific here and raise an error??
    # Take data from relevant columns:
    x_data = data[x_val]
    y_data = data[y_val]
    z_data = data[label]

    x_compact = np.unique(x_data)
    y_compact = np.unique(y_data)
    size_x = len(x_compact)
    size_y = len(y_compact)
    z_data_table = z_data.values.reshape((size_y, size_x))
    for y_ind in range(size_y):
        for x_ind in range(size_x):
            if x_compact[x_ind] + y_compact[y_ind] > 1:
                z_data_table[y_ind, x_ind] = np.nan
    cmap = plt.get_cmap(name="RdBu")
    cmap.set_bad(color="#C5C9C7")

    fig, ax = plt.subplots()
    plot = ax.imshow(
        z_data_table,
        cmap=cmap,
        interpolation="bilinear",
        # extent = [x_compact[0], x_compact[-1], y_compact[-1], y_compact[0]],
        aspect=size_x / size_y,
    )
    cbar = ax.figure.colorbar(plot, ax=ax)
    cbar.ax.set_ylabel(draw_label, rotation=-90, va="bottom")

    take_x = list(range(size_x))
    take_y = list(range(size_y))
    if size_x > 6:
        frac = (size_x - 1) / 5
        take_x = [int(i * frac) for i in range(6)]
    if size_y > 6:
        frac = (size_y - 1) / 5
        take_y = [int(i * frac) for i in range(6)]
    y_label = y_val
    x_label = x_val
    if x_val == "fraction_taus":
        x_label = r"$f_\tau$"
    if y_val == "fraction_c":
        y_label = r"$f_c$"
    elif y_val == "fraction_b":
        y_label = r"$f_b$"
    ax.set_xticks(take_x)
    ax.set_yticks(take_y)
    ax.set_xticklabels(np.around(x_compact[take_x], 3))
    ax.set_yticklabels(np.around(y_compact[take_y], 3))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    if UseAtlasTag:
        pas.makeATLAStag(ax, fig, AtlasTag, SecondTag, xmin=0.05, ymax=0.1)

    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


def plot_prob(
    df_results: pd.DataFrame,
    plot_name: str,
    tagger_name: str,
    class_labels: list,
    flavour: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    nBins: int = 50,
    logy: bool = False,
    figsize: list = None,
    labelFontSize: int = 10,
    loc_legend: str = "best",
    ncol: int = 2,
    x_label: str = None,
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    dpi: int = 400,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
):
    """Plot the probability output for the given flavour for one model.

    Parameters
    ----------
    df_results : pd.DataFrame
        Pandas DataFrame with the probability values of the jets inside.
    plot_name : str
        Path, Name and format of the resulting plot file.
    tagger_name : str
        Name of the tagger/model that is to be plotted.
    class_labels : list
        A list of the class_labels which are used.
    flavour : str
        Probability of this flavour is plotted.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample"
    nBins : int, optional
        Number of bins to use, by default 50
    logy : bool, optional
        Plot a logarithmic y-axis, by default False
    figsize : list, optional
        Size of the resulting figure as a list with two elements. First
        is the width, second the height. By default None.
    labelFontSize : int, optional
        Fontsize of the axis labels, by default 10
    loc_legend : str, optional
        Position of the legend in the plot, by default "best"
    ncol : int, optional
        Number of columns in the legend, by default 2
    x_label : str, optional
        x-label, by default None
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag, by default 1.3
    yAxisAtlasTag : float, optional
        y position of the ATLAS tag, by default 0.9
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    ymin : float, optional
        Minimum value of the y-axis, by default None
    ymax : float, optional
        Maximum value of the y-axis, by default None
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {f"{label}": i for i, label in enumerate(class_labels)}

    # Calculate the binning for all flavours
    _, Binning = np.histogram(
        df_results.query(f"labels=={index_dict[flavour]}")[
            f'{tagger_name}_{flav_cat[flavour]["prob_var_name"]}'
        ],
        bins=nBins,
    )

    # Clear the figure and init a new one
    if figsize is None:
        plt.clf()
        plt.figure(figsize=(8, 6))

    else:
        plt.clf()
        plt.figure(figsize=(figsize[0], figsize[1]))

    for iter_flavour in class_labels:
        # Select correct jets
        flav_tracks = df_results.query(f"labels=={index_dict[iter_flavour]}")[
            f'{tagger_name}_{flav_cat[flavour]["prob_var_name"]}'
        ]

        bins, weights, unc, band = hist_w_unc(
            a=flav_tracks,
            bins=Binning,
        )

        plt.hist(
            x=bins[:-1],
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.0,
            color=flav_cat[iter_flavour]["colour"],
            stacked=False,
            fill=False,
            label=flav_cat[iter_flavour]["legend_label"],
        )

        if iter_flavour == class_labels[-1]:
            plt.hist(
                x=bins[:-1],
                bins=bins,
                bottom=band,
                weights=unc * 2,
                label="stat. unc.",
                **global_config.hist_err_style,
            )

        else:
            plt.hist(
                x=bins[:-1],
                bins=bins,
                bottom=band,
                weights=unc * 2,
                **global_config.hist_err_style,
            )

    # overwriting xlim and ylim if defined in config
    xmin = plt.xlim()[0] if xmin is None else xmin
    xmax = plt.xlim()[1] if xmax is None else xmax
    ymin = plt.ylim()[0] if ymin is None else ymin
    ymax = plt.ylim()[1] if ymax is None else ymax

    if logy is True:
        plt.yscale("log")
        ymin = max(ymin, 1e-08)

    # Increase ymax so atlas tag don't cut plot
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)
    plt.xlim(xmin=ymin, xmax=xmax)

    # Set legend
    plt.legend(loc=loc_legend, ncol=ncol)

    # Set label
    xlabel = (
        f'{flav_cat[flavour]["legend_label"]} Probability'
        if x_label is None
        else x_label
    )
    plt.xlabel(
        xlabel,
        horizontalalignment="right",
        fontsize=labelFontSize,
        x=1.0,
    )
    plt.ylabel(
        "Normalised Number of Jets",
        horizontalalignment="right",
        fontsize=labelFontSize,
        y=1.0,
    )

    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.tight_layout()
    plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


def plot_prob_comparison(
    df_list: list,
    model_labels: list,
    tagger_list: list,
    class_labels_list: list,
    flavour: str,
    plot_name: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    nBins: int = 50,
    figsize: list = None,
    labelpad: int = None,
    labelFontSize: int = 10,
    legFontSize: int = 10,
    loc_legend: str = "best",
    ncol: int = 2,
    Ratio_Cut=None,
    which_axis="left",
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    ycolor: str = "black",
    title: str = None,
    logy: bool = False,
    dpi: int = 400,
):
    """Plot the probability output for the given flavour for
    multiple models with a ratio plot in a subpanel.

    Parameters
    ----------
    df_list : list
        List of pandas DataFrames with the probability values inside
    model_labels : list
        List of labels for the given models
    tagger_list : list
        List of tagger names
    class_labels_list : list
        List with the class_labels used in the different taggers
    flavour : str
        Probability of this flavour is plotted.
    plot_name : str
        Path, Name and format of the resulting plot file.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample""
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag, by default 1.3
    yAxisAtlasTag : float, optional
        y position of the ATLAS tag, by default 0.9
    nBins : int, optional
        Number of bins to use, by default 50
    figsize : list, optional
        Size of the resulting figure as a list with two elements. First
        is the width, second the height. By default None.
    labelpad : int, optional
        Spacing in points from the axes bounding box including
        ticks and tick labels, by default None
    labelFontSize : int, optional
        Fontsize of the axis labels, by default 10
    legFontSize : int, optional
        Fontsize of the legend, by default 10
    loc_legend : str, optional
        Position of the legend in the plot, by default "best"
    ncol : int, optional
        Number of columns in the legend, by default 2
    Ratio_Cut : list, optional
        List of the lower and upper y-limit for the ratio plot,
        by default None
    which_axis : list, optional
        List which y-axis to use for the given models, by default "left"
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    ymin : float, optional
        Minimum value of the y-axis, by default None
    ymax : float, optional
        Maximum value of the y-axis, by default None
    ycolor : str, optional
        y-axis color, by default "black"
    title : str, optional
        Title over the plot, by default None
    logy : bool, optional
        y-axis in log format, by default True
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {f"{iter_flav}": i for i, iter_flav in enumerate(class_labels_list[0])}

    if not isinstance(which_axis, list):
        which_axis = [which_axis] * len(df_list)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    if figsize is None:
        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(8, 1, figure=fig)
    axis_dict["left"] = {}
    axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
    axis_dict["left"]["ratio"] = fig.add_subplot(
        gs[6:, 0], sharex=axis_dict["left"]["top"]
    )

    # Get binning for the plot
    _, Binning = np.histogram(
        df_list[0].query(f"labels=={index_dict[flavour]}")[
            f'{tagger_list[0]}_{flav_cat[flavour]["prob_var_name"]}'
        ],
        bins=nBins,
    )

    # Init bincout and unc dict for ratio calculation
    bincounts = {}
    bincounts_unc = {}

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    for i, (
        df_results,
        linestyle,
        which_a,
        model_label,
        tagger,
        class_labels,
    ) in enumerate(
        zip(
            df_list,
            linestyles,
            which_axis,
            model_labels,
            tagger_list,
            class_labels_list,
        )
    ):
        # Get index dict
        index_dict = {f"{iter_flav}": i for i, iter_flav in enumerate(class_labels)}

        for iter_flav in class_labels:
            # Select correct jets
            flav_tracks = df_results.query(f"labels=={index_dict[iter_flav]}")[
                f'{tagger}_{flav_cat[flavour]["prob_var_name"]}'
            ]

            # Calculate bins
            bins, weights, unc, band = hist_w_unc(
                a=flav_tracks,
                bins=Binning,
            )

            hist_counts, _, _ = axis_dict[which_a]["top"].hist(
                x=bins[:-1],
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2.0,
                linestyle=linestyle,
                color=flav_cat[iter_flav]["colour"],
                stacked=False,
                fill=False,
                label=flav_cat[iter_flav]["legend_label"] + f" {model_label}",
            )

            if (iter_flav == class_labels[-1]) and i == 0:
                axis_dict[which_a]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    label="stat. unc.",
                    **global_config.hist_err_style,
                )

            else:
                axis_dict[which_a]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    **global_config.hist_err_style,
                )

            bincounts.update({f"{iter_flav}{i}": hist_counts})
            bincounts_unc.update({f"{iter_flav}{i}": unc})

        # Start ratio plot
        if i != 0:
            for iter_flav in class_labels:

                # Calculate the step and step_unc for ratio
                step, step_unc = hist_ratio(
                    numerator=bincounts[f"{iter_flav}{i}"],
                    denominator=bincounts[f"{iter_flav}0"],
                    numerator_unc=bincounts_unc[f"{iter_flav}{i}"],
                    denominator_unc=bincounts_unc[f"{iter_flav}0"],
                )

                axis_dict["left"]["ratio"].step(
                    x=Binning,
                    y=step,
                    color=flav_cat[iter_flav]["colour"],
                    linestyle=linestyles[i],
                )

                axis_dict["left"]["ratio"].fill_between(
                    x=Binning,
                    y1=step - step_unc,
                    y2=step + step_unc,
                    step="pre",
                    facecolor="none",
                    edgecolor=global_config.hist_err_style["edgecolor"],
                    linewidth=global_config.hist_err_style["linewidth"],
                    hatch=global_config.hist_err_style["hatch"],
                )

    # Add axes, titels and the legend
    axis_dict["left"]["top"].set_ylabel(
        "Normalised Number of Jets",
        fontsize=labelFontSize,
        horizontalalignment="right",
        y=1.0,
        color=ycolor,
    )
    if title is not None:
        axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["ratio"].set_xlabel(
        f'{flav_cat[flavour]["legend_label"]} Probability',
        horizontalalignment="right",
        x=1.0,
    )

    axis_dict["left"]["ratio"].set_ylabel(
        "Ratio",
        labelpad=labelpad,
        fontsize=labelFontSize,
    )

    if Ratio_Cut is not None:
        axis_dict["left"]["ratio"].set_ylim(bottom=Ratio_Cut[0], top=Ratio_Cut[1])

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    if xmin is not None:
        axis_dict["left"]["top"].set_xlim(left=xmin)

    else:
        axis_dict["left"]["top"].set_xlim(left=Binning[0])

    if xmax is not None:
        axis_dict["left"]["top"].set_xlim(right=xmax)

    else:
        axis_dict["left"]["top"].set_xlim(right=Binning[-1])

    if ymin is not None:
        axis_dict["left"]["top"].set_ylim(bottom=ymin)

    if ymax is not None:
        axis_dict["left"]["top"].set_ylim(top=ymax)

    # Add black line at one
    axis_dict["left"]["ratio"].axhline(
        y=1,
        xmin=0,
        xmax=1,
        color="black",
        alpha=0.5,
    )

    if logy is True:
        axis_dict["left"]["top"].set_yscale("log")

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    if logy is False:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0], left_y_limits[1] * yAxisIncrease
        )

    elif logy is True:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0] * 0.5,
            left_y_limits[0] * ((left_y_limits[1] / left_y_limits[0]) ** yAxisIncrease),
        )

    axis_dict["left"]["top"].legend(
        loc=loc_legend,
        fontsize=legFontSize,
        ncol=ncol,
    )  # , title="DL1r")

    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True, dpi=dpi)
    plt.close()


def plot_confusion(
    df_results: dict,
    tagger_name: str,
    class_labels: list,
    plot_name: str,
    colorbar: bool = True,
    show_absolute: bool = False,
    show_normed: bool = True,
    transparent_bkg: bool = True,
    dpi: int = 400,
):
    """Plotting the confusion matrix for a given tagger.

    Parameters
    ----------
    df_results : dict
        Loaded pandas dataframe from the evaluation file
    tagger_name : str
        Name of the tagger in the evaluation file
    class_labels : list
        List of class labels used
    plot_name : str
        Full path + name of the plot with the extension
    colorbar : bool, optional
        Decide, if colourbar is shown or not, by default True
    show_absolute : bool, optional
        Show the absolute, by default False
    show_normed : bool, optional
        Show the output normed, by default True
    transparent_bkg : bool, optional
        Decide, if the background is transparent or not, by default True
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Get a list of the tagger prob variables
    prob_list = []
    for prob in class_labels:
        prob_list.append(
            f'{tagger_name}_{global_config.flavour_categories[prob]["prob_var_name"]}'
        )

    # Get the truth
    y_target = df_results["labels"]

    # Get the probabilities of the tagger and select the highest
    y_predicted = np.argmax(df_results[prob_list].values, axis=1)

    # Define the confusion matrix
    cm = confusion_matrix(y_target=y_target, y_predicted=y_predicted, binary=False)

    # Plot the colormap
    mlxtend_plot_cm(
        conf_mat=cm,
        colorbar=colorbar,
        show_absolute=show_absolute,
        show_normed=show_normed,
        class_names=class_labels,
    )

    # Set tight layout for the plot
    plt.tight_layout()

    # Save the plot to path
    plt.savefig(plot_name, transparent=transparent_bkg, dpi=dpi)
    plt.close()


def plotFractionContour(  # pylint: disable=W0102
    df_results_list: list,
    tagger_list: list,
    label_list: list,
    colour_list: list,
    linestyle_list: list,
    rejections_to_plot: list,
    plot_name: str,
    rejections_to_fix_list: list,
    marked_points_list: list,
    ApplyAtlasStyle: bool = True,
    transparent_bkg: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Simulation Internal",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow,\n$t\\bar{t}$ Test Sample, WP = 77%",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.3,
    figsize: list = [11.69 * 0.8, 8.27 * 0.8],
    legcols: int = 1,
    labelFontSize: int = 10,
    legFontSize: int = 10,
    loc_legend: str = "best",
    xlim: list = None,
    ylim: list = None,
    grid: bool = True,
    title: str = "",
    xcolour: str = "black",
    ycolour: str = "black",
    dpi: int = 400,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot contour plots for the given taggers for two rejections.
    The rejections are calulated with different fractions. If more
    than two rejections are available, the others need to be set to
    a fixed value.

    Parameters
    ----------
    df_results_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    label_list : list
        List with the labels for the given taggers.
    colour_list : list
        List with colours for the given taggers. If an empty list is given,
        the colours will be set automatically.
    linestyle_list : list
        List with linestyles for the given taggers. If an empty list is given,
        the linestyles will be set automatically.
    rejections_to_plot : list
        List with two elements. The elements are the names for the two
        rejections that are plotted against each other. For example,
        ["cjets", "ujets"].
    plot_name : str
        Path, Name and format of the resulting plot file.
    rejections_to_fix_list : list
        List of dicts with the extra rejections. If more than two rejections are
        available, you need to fix the other rejections to a specific
        value. The dict entry key is the name of the rejection, for
        example "bbjets", and the entry is the value that it is set to,
        for example 0.2.
    marked_points_list : list
        List with marker dicts for each model provided. The dict contains
        the information needed to plot the marker, like the fraction values,
        which colour is used etc.
    ApplyAtlasStyle : bool, optional
        Apply ATLAS style for matplotlib, by default True
    transparent_bkg : bool, optional
        if plot is saved with transparent background, by default True
    UseAtlasTag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    AtlasTag : str, optional
        First row of the ATLAS Tag, by default "Simulation Internal"
    SecondTag : str, optional
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically, by default,
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample, WP = 77"
    yAxisAtlasTag : float, optional
        y position where the ATLAS logo is placed in parts of the full y axis
        (0 is bottom, 1 is tom). By default 0.9
    yAxisIncrease : float, optional
        Increasing the y axis to fit the ATLAS Tag, by default 1.3
    figsize : list, optional
        List with the figure size, first entry is the width, second is the
        height. By default [11.69 * 0.8, 8.27 * 0.8]
    legcols : int, optional
        Number of legend columns, by default 1
    labelFontSize : int, optional
        Fontsize of the axis labels, by default 10
    legFontSize : int, optional
        Fontsize of the legend, by default 10
    loc_legend : str, optional
        Position of the legend in the plot, by default "best"
    xlim : list, optional
        List with two elements, lower and upper limit for the x-axis,
        by default None
    ylim : list, optional
        List with two elements, lower and upper limit for the y-axis,
        by default None
    grid : bool, optional
        Decide, if a grid is plotted or not, by default True
    title : str, optional
        Title of the plot, by default ""
    xcolour : str, optional
        Color of the x-axis, by default "black"
    ycolour : str, optional
        Color of the y-axis, by default "black"
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    **kwargs
        Arbitrary keyword arguments.

    Raises
    ------
    IndexError
        If the given number of tagger names, labels and data dicts are not
        the same.
    """

    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get a full colour list
    if len(colour_list) == 0:
        colour_list_tmp = []

        # Create new colour list
        for i in range(len(df_results_list)):
            colour_list_tmp.append(f"C{i}")

        # Set the tmp colour list as the real one
        colour_list = colour_list_tmp

    # Get a full colour list
    if len(linestyle_list) == 0:
        linestyle_list = ["--" for i in range(len(df_results_list))]

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Init new list for the fraction values
    fraction_list = []

    # Extract fraction steps from dict
    for _, dict_values in df_results_list[0].items():
        fraction_list.append(dict_values[f"{rejections_to_plot[0]}"])

    # Remove all doubled items
    fraction_list = list(dict.fromkeys(fraction_list))

    # Set figure size
    plt.figure(figsize=(figsize[0], figsize[1]))

    # Ensure that for each model, a tagger name and a label is provided and vice versa
    # TODO Change in Python 3.10 to strict=True in the zip() function which will ensure
    # same length
    if not all(
        len(lst) == len(df_results_list)
        for lst in [
            tagger_list,
            label_list,
            colour_list,
            linestyle_list,
            rejections_to_fix_list,
            marked_points_list,
        ]
    ):
        raise IndexError(
            "Not the same amount of Evaluation files, names and labels are given! "
            "Please check that!"
        )

    # Loop over the combinations of the models
    for (
        df_results,
        tagger,
        label,
        colour,
        linestyle,
        fixed_rejections,
        marked_point_dict,
    ) in zip(
        df_results_list,
        tagger_list,
        label_list,
        colour_list,
        linestyle_list,
        rejections_to_fix_list,
        marked_points_list,
    ):
        # Init a dict for the rejections with an empty list for each rejection
        df = {f"{rejection}": [] for rejection in rejections_to_plot}

        # Loop over the fraction values
        for frac in fraction_list:

            # Loop over the entries in the provided results
            for dict_key, dict_values in df_results.items():

                # Init a rej_to_fix bool
                rej_to_fix_bool = True

                # Check if all not-plotted rejections have a fixed value given
                if fixed_rejections:
                    for (
                        rej_to_fix_key,
                        rej_to_fix_key_value,
                    ) in fixed_rejections.items():
                        if (
                            not dict_values[f"{rej_to_fix_key}_rej"]
                            == rej_to_fix_key_value
                        ):
                            rej_to_fix_bool = False

                # Check that the correct combination of fraction value and
                # rejection is chosen
                if (
                    f"{tagger}" in dict_key
                    and dict_values[f"{rejections_to_plot[0]}"] == frac
                    and rej_to_fix_bool
                ):
                    for rejection in rejections_to_plot:
                        df[rejection].append(dict_values[f"{rejection}_rej"])

                    if (
                        marked_point_dict
                        and marked_point_dict[f"{rejections_to_plot[0]}"]
                        == dict_values[f"{rejections_to_plot[0]}"]
                    ):
                        plot_point_x = dict_values[f"{rejections_to_plot[0]}_rej"]
                        plot_point_y = dict_values[f"{rejections_to_plot[1]}_rej"]

        # Plot the contour
        plt.plot(
            df[rejections_to_plot[0]],
            df[rejections_to_plot[1]],
            label=label,
            color=colour,
            linestyle=linestyle,
        )

        if marked_point_dict:
            # Build the correct for the point
            frac_label_x = flav_cat[rejections_to_plot[0]]["prob_var_name"]
            frac_x_value = marked_point_dict[f"{rejections_to_plot[0]}"]
            frac_label_y = flav_cat[rejections_to_plot[1]]["prob_var_name"]
            frac_y_value = marked_point_dict[f"{rejections_to_plot[1]}"]

            point_label = (
                rf"{label} $f_{{{frac_label_x}}} = {frac_x_value}$,"
                rf" $f_{{{frac_label_y}}} = {frac_y_value}$"
            )

            # Plot the marker
            plt.plot(
                plot_point_x,
                plot_point_y,
                color=marked_point_dict["colour"]
                if "colour" in marked_point_dict
                and marked_point_dict["colour"] is not None
                else colour,
                marker=marked_point_dict["marker_style"]
                if "marker_style" in marked_point_dict
                and marked_point_dict["marker_style"] is not None
                else "x",
                label=marked_point_dict["marker_label"]
                if "marker_label" in marked_point_dict
                and marked_point_dict["marker_label"] is not None
                else point_label,
                markersize=marked_point_dict["markersize"]
                if "markersize" in marked_point_dict
                and marked_point_dict["markersize"] is not None
                else 15,
                markeredgewidth=marked_point_dict["markeredgewidth"]
                if "markeredgewidth" in marked_point_dict
                and marked_point_dict["markeredgewidth"] is not None
                else 2,
            )

    # Set x and y label
    plt.xlabel(
        flav_cat[rejections_to_plot[0]]["legend_label"] + " rejection",
        fontsize=labelFontSize,
        horizontalalignment="right",
        x=1.0,
        color=ycolour,
    )
    plt.ylabel(
        flav_cat[rejections_to_plot[1]]["legend_label"] + " rejection",
        fontsize=labelFontSize,
        horizontalalignment="right",
        y=1.0,
        color=xcolour,
    )

    # Set limits if defined
    if xlim:
        plt.xlim(bottom=xlim[0], top=xlim[1])

    if ylim:
        plt.xlim(bottom=ylim[0], top=ylim[1])

    # Increase y limit for ATLAS tag
    y_min, y_max = plt.ylim()
    plt.ylim(bottom=y_min, top=y_max * yAxisIncrease)

    # Check to use a meshgrid
    if grid is True:
        plt.grid()

    # Set title of the plot
    plt.title(title)

    # Define ATLAS tag
    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Set legend
    plt.legend(loc=loc_legend, prop={"size": legFontSize}, ncol=legcols)

    # Set tight layout
    plt.tight_layout()

    # Save plot
    plt.savefig(plot_name, transparent=transparent_bkg, dpi=dpi)
    plt.close()
    plt.clf()
