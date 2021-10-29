from umami.configuration import global_config, logger  # isort:skip

import matplotlib as mtp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import pchip

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.tools import applyATLASstyle


def eff_err(x, N):
    return np.sqrt(x * (1 - x) / N)


def rej_err(x, N):
    return np.sqrt((1 / x) * (1 - (1 / x)) / N)


def FlatEfficiencyPerBin(df, predictions, variable, var_bins, wp=0.7):
    """
    For each bin in var_bins of variable, cuts the score in
    predictions column to get the desired WP (working point)
    df must (at least) contain the following columns:
        - bscore
        - value of variable
        - labels (with the true labels)
    Creates a column 'btag' with the tagged (1/0) info in df.

    Note: labels indicate in fact the column in Y_true, so:
        - labels = 3 for taus,
        - labels = 2 for b,
        - labels = 1 for c,
        - labels = 0 for u.
    """
    df["btag"] = 0
    for i in range(len(var_bins) - 1):
        index_jets_in_bin = (var_bins[i] <= df[variable]) & (
            df[variable] < var_bins[i + 1]
        )
        df_in_bin = df[index_jets_in_bin]
        if len(df_in_bin) == 0:
            continue
        bscores_b_in_bin = df_in_bin[df_in_bin["labels"] == 2][predictions]
        if len(bscores_b_in_bin) == 0:
            continue
        cutvalue_in_bin = np.percentile(bscores_b_in_bin, 100.0 * (1.0 - wp))
        df.loc[index_jets_in_bin, ["btag"]] = (
            df_in_bin[predictions] > cutvalue_in_bin
        ) * 1

    return df["btag"]


def calc_bins(input_array, Binning):
    """
    This method calculates the bin content and uncertainty
    for a given input.
    Returns the bins, weights, uncertainites and lower band
    as numpy arrays.
    """
    # Calculate the number of jets per flavour
    arr_length = len(input_array)

    # Calculate the counts and the bin edges
    counts, bins = np.histogram(input_array, bins=Binning)

    unc = np.sqrt(counts) / arr_length
    band = counts / arr_length - unc
    weights = counts / arr_length

    return bins, weights, unc, band


def calc_ratio(counter, denominator, counter_unc, denominator_unc):
    """
    This method calculates the ratio of the given bincounts and
    returns the input for a step function that plots the ratio
    """
    step = np.divide(
        counter,
        denominator,
        out=np.ones(
            counter.shape,
            dtype=float,
        ),
        where=(denominator != 0),
    )

    # Add an extra bin in the beginning to have the same binning as the input
    # Otherwise, the ratio will not be exactly above each other (due to step)
    step = np.append(np.array([step[0]]), step)

    # Calculate rel uncertainties
    counter_rel_unc = np.divide(
        counter_unc,
        counter,
        out=np.zeros(
            counter.shape,
            dtype=float,
        ),
        where=(counter != 0),
    )

    denominator_rel_unc = np.divide(
        denominator_unc,
        denominator,
        out=np.zeros(
            denominator.shape,
            dtype=float,
        ),
        where=(denominator != 0),
    )

    # Calculate rel uncertainty
    step_rel_unc = np.sqrt(counter_rel_unc ** 2 + denominator_rel_unc ** 2)

    # Add the first value again (same reason as for the step calculation)
    step_rel_unc = np.append(np.array([step_rel_unc[0]]), step_rel_unc)

    # Calculate final uncertainty
    step_unc = step * step_rel_unc

    return step, step_unc


def plotEfficiencyVariable(
    plot_name,
    df,
    variable,
    var_bins,
    fc=0.018,
    ftau=None,
    efficiency=0.70,
    include_taus=False,
    centralise_bins=True,
    xticksval=None,
    xticks=None,
    minor_ticks_frequency=None,
    xlogscale=False,
    xlabel=None,
    UseAtlasTag=True,
    AtlasTag="Internal",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$",
    ThirdTag="DL1r",
    SaveData=False,
):
    """
    For a given variable (string) in the panda dataframe df, plots:
    - the b-eff of b, c, and light jets as a function of variable
                 (discretised in bins as indicated by var_bins input)
    - the variable distribution.

    The following options are needed:
    - plot_name: the full path + the start of the name of the plot.
    - df: panda dataframe with columns
        - The efficiency is computed from the btag column of df.
        - variable (next parameter)
        - labels (2 for b, 1 for c, 0 for u, and 3 for taus)
    - variable: String giving the variable to use in df
    - var_bins: numpy array listing the bins of variable
        Entry i and i+1 in var_bins define bin i of the variable.

    The following options are optional:
    - fc: the charm fraction used to compute the score (btag)
    - ftau: the tau fraction used to compute the score (btag)
    - efficiency: the working point (b tagging). NOT IN PERCENT
    - include_taus: boolean whether to plot taus or not
    - centralise_bins: boolean to centralise point in the bins
    - xticksval: list of ticks values (must agree with the one below)
    - xticks: list of ticks (must agree with the one above)
    - minor_ticks_frequency: int,
                    if given, sets the frequency of minor ticks
    - xlogscale: boolean, whether to set the x-axis in log-scale.
    - xlabel: String: label to use for the x-axis.
    - UseAtlasTag: boolean, whether to use the ATLAS tag or not.
    - AtlasTag: string: tag to attached to ATLAS
    - SecondTag: string: second line of the ATLAS tag.
    - ThirdTag: tag on the top left of the plot, indicate model
        and fractions used.

    Note: to get a flat efficiency plot, you need to produce a 'btag' column
    in df using FlatEfficiencyPerBin (defined above).
    """
    data = df.copy()
    total_var = []
    b_lst, b_err_list = [], []
    c_lst, c_err_list = [], []
    u_lst, u_err_list = [], []
    tau_lst, tau_err_list = [], []
    null_value = np.nan
    for i in range(len(var_bins) - 1):
        df_in_bin = data[
            (var_bins[i] <= data[variable])
            & (data[variable] < var_bins[i + 1])
        ]
        total_b_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 2])
        total_c_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 1])
        total_u_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 0])
        total_tau_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 3])
        df_in_bin = df_in_bin.query("btag == 1")
        total_in_bin = (
            total_b_tag_in_bin + total_c_tag_in_bin + total_u_tag_in_bin
        )
        if include_taus:
            total_in_bin += total_tau_tag_in_bin

        if total_in_bin == 0:
            total_var.append(null_value)
            u_lst.append(null_value)
            u_err_list.append(null_value)
            c_lst.append(null_value)
            c_err_list.append(null_value)
            b_lst.append(null_value)
            b_err_list.append(null_value)
            tau_lst.append(null_value)
            tau_err_list.append(null_value)
        else:
            total_var.append(total_in_bin)
            index, counts = np.unique(
                df_in_bin["labels"].values, return_counts=True
            )
            in_b, in_c, in_u, in_tau = False, False, False, False
            for item, count in zip(index, counts):
                if item == 0:
                    eff = count / total_u_tag_in_bin
                    u_lst.append(eff)
                    u_err_list.append(eff_err(eff, total_u_tag_in_bin))
                    in_u = True
                elif item == 1:
                    eff = count / total_c_tag_in_bin
                    c_lst.append(eff)
                    c_err_list.append(eff_err(eff, total_c_tag_in_bin))
                    in_c = True
                elif item == 2:
                    eff = count / total_b_tag_in_bin
                    b_lst.append(eff)
                    b_err_list.append(eff_err(eff, total_b_tag_in_bin))
                    in_b = True
                elif item == 3:
                    eff = count / total_tau_tag_in_bin
                    tau_lst.append(eff)
                    tau_err_list.append(eff_err(eff, total_tau_tag_in_bin))
                    in_tau = True
                else:
                    logger.info(f"Invaled value of index from labels: {item}")
            if not (in_u):
                u_lst.append(null_value)
                u_err_list.append(null_value)
            if not (in_c):
                c_lst.append(null_value)
                c_err_list.append(null_value)
            if not (in_b):
                b_lst.append(null_value)
                b_err_list.append(null_value)
            if not (in_tau):
                tau_lst.append(null_value)
                tau_err_list.append(null_value)
    if plot_name[-4:] == ".pdf":
        plot_name = plot_name[:-4]

    if xlabel is None:
        xlabel = variable
        if variable == "actualInteractionsPerCrossing":
            xlabel = "Actual interactions per bunch crossing"
        elif variable == "pt":
            xlabel = r"$p_T$ [GeV]"

    if ftau is None:
        ThirdTag = (
            ThirdTag
            + "\n"
            + "$\\epsilon_b$ = {}%, $f_c$ = {}".format(efficiency, fc)
        )
    else:
        ThirdTag = (
            ThirdTag
            + "\n"
            + "$\\epsilon_b$ = {}%, $f_c$ = {}, ".format(efficiency, fc)
            + "$f_{tau}$ = "
            + "{}".format(ftau)
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
            for i in range(len(x_label)):
                if i in selected_indices:
                    trimmed_label.append(
                        np.format_float_scientific(x_label[i], precision=3)
                    )
                    trimmed_label_val.append(x_label[i])
        else:
            trimmed_label = x_label
            trimmed_label_val = x_label

    if SaveData:
        x_arr = np.asarray(x_value)
        x_err_arr = np.asarray(x_err)
        b_arr = np.asarray(b_lst)
        b_err_arr = np.asarray(b_err_list)
        c_arr = np.asarray(c_lst)
        c_err_arr = np.asarray(c_err_list)
        u_arr = np.asarray(u_lst)
        u_err_arr = np.asarray(u_err_list)
        if include_taus:
            tau_arr = np.asarray(tau_lst)
            tau_err_arr = np.asarray(tau_err_list)
            np.savetxt(
                plot_name + "_effTable_" + "old" + ".txt",
                (
                    x_arr,
                    x_err_arr,
                    b_arr,
                    b_err_arr,
                    c_arr,
                    c_err_arr,
                    u_arr,
                    u_err_arr,
                    tau_arr,
                    tau_err_arr,
                ),
                delimiter=" ",
            )
        else:
            np.savetxt(
                plot_name + "_effTable_" + "old" + ".txt",
                (
                    x_arr,
                    x_err_arr,
                    b_arr,
                    b_err_arr,
                    c_arr,
                    c_err_arr,
                    u_arr,
                    u_err_arr,
                ),
                delimiter=" ",
            )

    # First plot: variable
    fig, ax1 = plt.subplots()
    ax1.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 3), useMathText=True
    )
    ax1.ticklabel_format(
        style="sci", axis="x", scilimits=(0, 3), useMathText=True
    )
    ax1.errorbar(x=x_value, y=total_var, xerr=x_err, fmt="o", markersize=2.0)
    ax1.set_ylabel(r"Count")
    if xlogscale:
        ax1.set_xscale("log")
        if xticks is not None:
            ax1.set_xticks(trimmed_label_val)
            ax1.set_xticklabels(trimmed_label)
    else:
        ax1.set_xticks(trimmed_label_val)
        ax1.set_xticklabels(trimmed_label)
        ax1.set_xlim(trimmed_label_val[0], trimmed_label_val[-1])
    ax1.set_xlabel(xlabel)
    if minor_ticks_frequency is not None:
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
    ax_r = ax1.secondary_yaxis("right")
    ax_t = ax1.secondary_xaxis("top")
    ax_r.set_yticklabels([])
    if xlogscale:
        ax_t.set_xscale("log")
        if xticks is not None:
            ax_t.set_xticks(trimmed_label_val)
    else:
        ax_t.set_xticks(trimmed_label_val)
    if minor_ticks_frequency is not None:
        ax_t.xaxis.set_minor_locator(
            plt.MultipleLocator(minor_ticks_frequency)
        )
    ax_t.set_xticklabels([])
    ax_r.tick_params(axis="y", direction="in", which="both")
    ax_t.tick_params(axis="x", direction="in", which="both")
    ax1.grid(color="grey", linestyle="--", linewidth=0.5)
    if UseAtlasTag:
        pas.makeATLAStag(ax1, fig, AtlasTag, SecondTag, xmin=0.7, ymax=0.88)
    ax1.text(
        0.05,
        0.815,
        ThirdTag,
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax1.transAxes,
        color="Black",
        fontsize=10,
        linespacing=1.7,
    )
    fig.tight_layout()
    plt.savefig(plot_name + "_distr.pdf", transparent=True, dpi=200)
    plt.close()

    # Second plot: efficiency
    fig, ax = plt.subplots()
    ax.ticklabel_format(
        style="sci", axis="x", scilimits=(0, 3), useMathText=True
    )
    ax.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 3), useMathText=True
    )
    ax.axhline(y=1, color="#696969", linestyle="-", zorder=1)
    ax.errorbar(
        x=x_value,
        y=b_lst,
        yerr=b_err_list,
        xerr=x_err,
        label=r"$b$-jets",
        fmt="o",
        markersize=2.0,
        markeredgecolor="#1f77b4",
        markerfacecolor="#1f77b4",
        c="#1f77b4",
        alpha=1,
        zorder=10,
    )
    ax.errorbar(
        x=x_value,
        y=c_lst,
        yerr=c_err_list,
        xerr=x_err,
        label=r"$c$-jets",
        fmt="s",
        markersize=2.0,
        markeredgecolor="#ff7f0e",
        markerfacecolor="#ff7f0e",
        c="#ff7f0e",
        alpha=1,
        zorder=9,
    )
    ax.errorbar(
        x=x_value,
        y=u_lst,
        yerr=u_err_list,
        xerr=x_err,
        label=r"$l$-jets",
        fmt="o",
        markersize=2.0,
        markeredgecolor="#2ca02c",
        markerfacecolor="#2ca02c",
        c="#2ca02c",
        alpha=0.7,
        zorder=8,
    )
    if include_taus:
        ax.errorbar(
            x=x_value,
            y=tau_lst,
            yerr=tau_err_list,
            xerr=x_err,
            label="tau-jets",
            fmt="s",
            markersize=2.0,
            markeredgecolor="#7c5295",
            markerfacecolor="#7c5295",
            c="#7c5295",
            alpha=0.7,
            zorder=7,
        )
    if xlogscale:
        ax.set_xscale("log")
        if xticks is not None:
            ax.set_xticks(trimmed_label_val)
            ax.set_xticklabels(trimmed_label)
    else:
        ax.set_xticks(trimmed_label_val)
        ax.set_xticklabels(trimmed_label)
        ax.set_xlim(trimmed_label_val[0], trimmed_label_val[-1])
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2e1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Efficiency")
    if minor_ticks_frequency is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_ticks_frequency))
    ax_r = ax.secondary_yaxis("right")
    ax_t = ax.secondary_xaxis("top")
    ax_r.set_yticklabels([])
    if xlogscale:
        ax_t.set_xscale("log")
        if xticks is not None:
            ax_t.set_xticks(trimmed_label_val)
    else:
        ax_t.set_xticks(trimmed_label_val)
    if minor_ticks_frequency is not None:
        ax_t.xaxis.set_minor_locator(
            plt.MultipleLocator(minor_ticks_frequency)
        )
    ax_t.set_xticklabels([])
    ax_r.set_yscale("log")
    ax_r.tick_params(axis="y", direction="in", which="both")
    ax_t.tick_params(axis="x", direction="in", which="both")
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right")
    if UseAtlasTag:
        pas.makeATLAStag(ax, fig, AtlasTag, SecondTag, xmin=0.55, ymax=0.88)
    ax.text(
        0.05,
        0.815,
        ThirdTag,
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="Black",
        fontsize=10,
        linespacing=1.7,
    )
    fig.tight_layout()
    plt.savefig(plot_name + "_efficiency.pdf", transparent=True, dpi=200)
    plt.close()


def plotPtDependence(
    df_list,
    tagger_list,
    model_labels,
    plot_name: str,
    class_labels: list,
    main_class: str,
    flavour: str,
    WP: float = 0.77,
    Disc_Cut_Value: float = None,
    SWP_Comparison: bool = False,
    SWP_label_list: list = [],
    Passed: bool = True,
    Fixed_WP_Bin: bool = False,
    bin_edges=[0, 20, 50, 90, 150, 300, 1000],
    WP_Line: bool = False,
    figsize: list = None,
    Grid: bool = False,
    binomialErrors: bool = True,
    xlabel: str = r"$p_T$ in GeV",
    Log: bool = False,
    colors: list = None,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample, fc=0.018",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.1,
    frameon: bool = False,
    ncol: int = 1,
    ymin: float = None,
    ymax: float = None,
    alpha: float = 0.8,
    trans: bool = True,
    linewidth: float = 1.6,
):
    """
    For a given list of models, plot the b-eff, l and c-rej as a function
    of jet pt for customizable WP and fc values.

    Following options are needed to be given:
    - df_list: List of the dataframes from the model. See plotting_umami.py
    - tagger_list: List of strings with the tagger names (MUST BE THE NAMES IN FILE)
    - model_labels: Labels for the legend of the plot
    - plot_name: Path, Name and format of the resulting plot file
    - class_labels: A list of the class_labels which are used
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - flavour: Flavour which is to be plotted.

    Following options are preset and can be changed:
    - WP: Which Working point is used
    - Disc_Cut_Value: Set a disc cut value for all models if SWP is off.
    - SWP_Comparison: Use the same cut value on the discriminant for all
                      models with the same SWP_label. Not works with Fixed_WP_Bin True.
    - SWP_label_list: List of labels for each model. Models with same SWP label get
                      the same Disc_cut_value
    - Passed: Select if the selected jets need to pass the discriminant WP cut
    - Fixed_WP_Bin: Calculate the WP cut on the discriminant per bin
    - bin_edges: As the name says, the edges of the bins used
    - WP_Line: Print a WP line in the upper plot
    - figsize: Size of the resulting figure
    - Grid: Use a grid in the plots
    - binomialErrors: Use binomial errors
    - xlabel: Label for x axis
    - Log: Set yscale to Log
    - colors: Custom color list for the different models
    - ApplyAtlasStyle: Apply ATLAS style for matplotlib
    - UseAtlasTag: Use the ATLAS Tag in the plots
    - AtlasTag: First row of the ATLAS Tag
    - SecondTag: Second Row of the ATLAS Tag. No need to add WP or fc.
                 It added automatically
    - yAxisAtlasTag: Relative y axis position of the ATLAS Tag in
    - yAxisIncrease: Increasing the y axis to fit the ATLAS Tag in
    - frameon: Set the frame around legend off/on
    - ncol: Number of columns in the legend
    - ymin: y axis minimum
    - ymax: y axis maximum
    - alpha: Value for visibility of the plot lines
    - trans: Sets the transparity of the background. If true, the background erased.
             If False, the background is white
    """

    # Apply ATLAS style
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get global config of the classes
    flav_cat = global_config.flavour_categories

    # Get the bins for the histogram
    pt_midpts = (np.asarray(bin_edges)[:-1] + np.asarray(bin_edges)[1:]) / 2.0
    bin_widths = (np.asarray(bin_edges)[1:] - np.asarray(bin_edges)[:-1]) / 2.0
    Npts = pt_midpts.size

    if SWP_Comparison is True:
        SWP_Cut_Dict = {}

    # Get the indicies of the flavours
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

    # Set color if not provided
    if colors is None:
        colors = ["C{}".format(i) for i in range(len(df_list))]

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    # Ratio save. Used for ratio calculation
    ratio_eff = {
        "pt_midpts": [],
        "effs": [],
        "rej": [],
        "bin_widths": [],
        "yerr": [],
    }

    # Init new figure
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

    for i, (df_results, model_label, tagger, color) in enumerate(
        zip(df_list, model_labels, tagger_list, colors)
    ):

        # Placeholder for the sig eff and bkg rejections
        effs = np.zeros(Npts)

        # Get truth labels
        truth_labels = df_results["labels"]

        if Fixed_WP_Bin is False:
            if SWP_Comparison is False:
                if Disc_Cut_Value is None:
                    # Calculate WP cutoff for b-disc
                    disc_cut = np.percentile(
                        df_results.query(f"labels=={index_dict[main_class]}")[
                            f"disc_{tagger}"
                        ],
                        (1 - WP) * 100,
                    )

                elif Disc_Cut_Value is not None:
                    disc_cut = Disc_Cut_Value

            elif (
                SWP_Comparison is True
                and SWP_label_list[i] not in SWP_Cut_Dict
            ):
                # Calc disc cut value for the SWP label model
                disc_cut = np.percentile(
                    df_results.query(f"labels=={index_dict[main_class]}")[
                        f"disc_{tagger}"
                    ],
                    (1 - WP) * 100,
                )

                # Set Value globally for the SWP label
                SWP_Cut_Dict.update({SWP_label_list[i]: disc_cut})

            elif SWP_Comparison is True and SWP_label_list[i] in SWP_Cut_Dict:
                # Set disc_cut for the model after its SWP label
                disc_cut = SWP_Cut_Dict[SWP_label_list[i]]

        # Get jet pts
        jetPts = df_results["pt"] / 1000

        # For calculating the binomial errors, let nTest be an array of
        # the same shape as the the # of pT bins that we have
        nTest = np.zeros(Npts)

        for j, pt_min, pt_max in zip(
            np.arange(Npts), bin_edges[:-1], bin_edges[1:]
        ):

            # Cut on selected flavour jets with the wanted pt range
            den_mask = (
                (jetPts > pt_min)
                & (jetPts < pt_max)
                & (truth_labels == index_dict[flavour])
            )

            if Fixed_WP_Bin is False:
                # Cut on jets which passed the WP
                if Passed is True:
                    num_mask = den_mask & (
                        df_results[f"disc_{tagger}"] > disc_cut
                    )

                else:
                    num_mask = den_mask & (
                        df_results[f"disc_{tagger}"] <= disc_cut
                    )

            else:
                # Setting pT mask for the selected bin to calculate
                # the disc cut value fot the particular bin
                pT_mask = (jetPts > pt_min) & (jetPts < pt_max)

                # If SWP is used, calculate the disc cut for the model if
                # its not added to the dict yet. If its already added,
                # the value is loaded. If SWP is false, the disc value
                # will be calculated for each of the models independently
                if SWP_Comparison is False:
                    disc_cut = np.percentile(
                        df_results.query(f"labels=={index_dict[main_class]}")[
                            f"disc_{tagger}"
                        ][pT_mask],
                        (1 - WP) * 100,
                    )

                elif (
                    SWP_Comparison is True
                    and SWP_label_list[i] not in SWP_Cut_Dict
                ):
                    # Calc disc cut value for the SWP label model
                    disc_cut = np.percentile(
                        df_results.query(f"labels=={index_dict[main_class]}")[
                            f"disc_{tagger}"
                        ][pT_mask],
                        (1 - WP) * 100,
                    )

                    # Set Value globally for the SWP label
                    SWP_Cut_Dict.update({SWP_label_list[i]: disc_cut})

                elif (
                    SWP_Comparison is True
                    and SWP_label_list[i] in SWP_Cut_Dict
                ):
                    # Set disc_cut for the model after its SWP label
                    disc_cut = SWP_Cut_Dict[SWP_label_list[i]]

                # Cut on jets which passed the WP
                if Passed is True:
                    num_mask = den_mask & (
                        df_results[f"disc_{tagger}"] > disc_cut
                    )

                else:
                    num_mask = den_mask & (
                        df_results[f"disc_{tagger}"] <= disc_cut
                    )

            # Sum masks for binominal error calculation
            nTest[j] = den_mask.sum()
            effs[j] = num_mask.sum() / nTest[j]

        # For b-jets, plot the eff: for l and c-jets, look at the rej
        if flavour == main_class:
            yerr = eff_err(effs, nTest) if binomialErrors else None

            axis_dict["left"]["top"].errorbar(
                pt_midpts,
                effs,
                xerr=bin_widths,
                yerr=yerr,
                color=color,
                fmt=".",
                label=model_label,
                alpha=alpha,
                linewidth=linewidth,
            )

            # Calculate Ratio
            # Check if its not the first model which is used as reference
            if i != 0:
                effs_ratio = effs / ratio_eff["effs"]
                yerr_ratio = (
                    (yerr / effs) + (ratio_eff["yerr"] / ratio_eff["effs"])
                ) * effs_ratio

                axis_dict["left"]["ratio"].errorbar(
                    pt_midpts,
                    effs_ratio,
                    xerr=bin_widths,
                    yerr=yerr_ratio,
                    color=color,
                    fmt=".",
                    label=model_label,
                    alpha=alpha,
                    linewidth=linewidth,
                )

            else:
                ratio_eff["pt_midpts"] = pt_midpts
                ratio_eff["effs"] = effs
                ratio_eff["bin_widths"] = bin_widths
                ratio_eff["yerr"] = yerr

        else:
            # Calculate rejection
            rej = 1 / effs
            yerr = (
                np.power(rej, 2) * eff_err(effs, nTest)
                if binomialErrors
                else None
            )

            # Plot the "hists"
            axis_dict["left"]["top"].errorbar(
                pt_midpts,
                rej,
                xerr=bin_widths,
                yerr=yerr,
                color=color,
                fmt=".",
                label=model_label,
                alpha=alpha,
                linewidth=linewidth,
            )

            # Calculate Ratio
            # Check if its not the first model which is used as reference
            if i != 0:
                rej_ratio = rej / ratio_eff["rej"]
                yerr_ratio = (
                    (yerr / rej) + (ratio_eff["yerr"] / ratio_eff["rej"])
                ) * rej_ratio

                axis_dict["left"]["ratio"].errorbar(
                    pt_midpts,
                    rej_ratio,
                    xerr=bin_widths,
                    yerr=yerr_ratio,
                    color=color,
                    fmt=".",
                    label=model_label,
                    alpha=alpha,
                    linewidth=linewidth,
                )

            else:
                ratio_eff["pt_midpts"] = pt_midpts
                ratio_eff["rej"] = rej
                ratio_eff["bin_widths"] = bin_widths
                ratio_eff["yerr"] = yerr

    # Set labels
    axis_dict["left"]["ratio"].set_xlabel(
        xlabel, horizontalalignment="right", x=1.0
    )

    # Set metric
    if flavour == main_class:
        metric = "efficiency"

    else:
        metric = "rejection"

    # Set addition to y label if fixed WP bin is True
    if Fixed_WP_Bin is False:
        Fixed_WP_Label = "Inclusive"

    else:
        Fixed_WP_Label = "Fixed"

    # Set y label
    axis_dict["left"]["top"].set_ylabel(
        f'{Fixed_WP_Label} {flav_cat[flavour]["legend_label"]} {metric}',
        horizontalalignment="right",
        y=1.0,
    )

    # Set ratio y label
    axis_dict["left"]["ratio"].set_ylabel("Ratio")

    # Check for Logscale
    if Log is True:
        axis_dict["left"]["top"].set_yscale("log")

    # Set limits
    axis_dict["left"]["top"].set_xlim(bin_edges[0], bin_edges[-1])

    # Increase ymax so atlas tag don't cut plot
    if (ymin is None) and (ymax is None):
        ymin, ymax = axis_dict["left"]["top"].get_ylim()

    elif ymin is None:
        ymin, _ = axis_dict["left"]["top"].get_ylim()

    elif ymax is None:
        _, ymax = axis_dict["left"]["top"].get_ylim()

    # Increase the yaxis limit upper part by given factor to fit ATLAS Tag in
    if Log is True:
        axis_dict["left"]["top"].set_ylim(
            ymin,
            ymin * ((ymax / ymin) ** yAxisIncrease),
        )

    else:
        axis_dict["left"]["top"].set_ylim(
            bottom=ymin, top=yAxisIncrease * ymax
        )

    # Get xlim for the horizontal and vertical lines
    xmin, xmax = axis_dict["left"]["top"].get_xlim()

    # Set WP Line
    if WP_Line is True:
        axis_dict["left"]["top"].hlines(
            y=WP,
            xmin=xmin,
            xmax=xmax,
            colors="black",
            linestyle="dotted",
            alpha=0.5,
        )

    # Ratio line
    axis_dict["left"]["ratio"].hlines(
        y=1,
        xmin=xmin,
        xmax=xmax,
        colors=colors[0],
        linestyle="dotted",
        alpha=0.5,
    )

    # Set grid
    if Grid is True:
        axis_dict["left"]["top"].grid()
        axis_dict["left"]["ratio"].grid()

    # Define legend
    axis_dict["left"]["top"].legend(
        loc="upper right", ncol=ncol, frameon=frameon
    )

    # Redefine Second Tag with inclusive or fixed tag
    if Fixed_WP_Bin is True:
        SecondTag = (
            SecondTag
            + "\nConstant "
            + r"$\epsilon_b$ = {}% per bin".format(int(WP * 100))
        )

    else:
        SecondTag = (
            SecondTag
            + "\nInclusive "
            + r"$\epsilon_b$ = {}%".format(int(WP * 100))
        )

    # Set the ATLAS Tag
    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Set tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(plot_name, transparent=trans)
    plt.close()


def plotROCRatio(
    df_results_list: list,
    tagger_list: list,
    rej_class_list: list,
    labels: list,
    plot_name: str,
    main_class: str,
    df_eff_key: str = "effs",
    title: str = "",
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample, fc=0.018",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.3,
    styles: list = None,
    colors: list = None,
    xmin: float = None,
    ymax: float = None,
    ymin: float = None,
    ymax_right: float = None,
    ymin_right: float = None,
    legFontSize: int = 10,
    loc_legend: str = "best",
    rrange: list = None,
    rlabel: str = "Ratio",
    binomialErrors: bool = True,
    nTest: list = 0,
    alabel: dict = None,
    figsize: list = None,
    legcols: int = 1,
    labelpad: int = None,
    which_axis: list = "left",
    WorkingPoints: list = None,
    same_height_WP: bool = True,
    ratio_id: int = 0,
    ycolor: str = "black",
    ycolor_right: str = "black",
    set_logy: bool = True,
):
    """
    Plot the ROC curves with binomial errors with the ratio plot in a subpanel
    underneath. This function all accepts the same inputs as plotROC, and the
    additional ones are listed below.

    Addtional Inputs:
    - rrange: The range on the y-axis for the ratio panel
    - rlabel: The label for the y-axis for the ratio panel
    - binomialErrors: whether to include binomial errors for the rejection
                      curves
    - nTest: A list of the same length as class_rejections, with the number of events used
             to calculate the background efficiencies.
             We need this To calculate the binomial errors on the background
             rejection,
             using the formula given by
             http://home.fnal.gov/~paterno/images/effic.pdf.
    """
    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    if binomialErrors is True:
        # Check if nTest is provided in all samples
        nTest_in_file = []
        for df_results in df_results_list:
            if "N_test" not in df_results:
                nTest_in_file.append(False)

            else:
                nTest_in_file.append(True)

        if nTest == 0 and not all(nTest_in_file):
            logger.error(
                "Requested binomialErrors, but not all models have nTest. Will NOT plot rej errors."
            )
            binomialErrors = False

    if styles is None:
        styles = ["-" for i in labels]
    if colors is None:
        colors = ["C{}".format(i) for i in range(len(labels))]
        colors_WP = "C{}".format(len(colors) + 1)

    else:
        colors_WP = "red"

    if type(nTest) != list:
        nTest = [nTest] * len(labels)

    if type(which_axis) != list:
        which_axis = [which_axis] * len(labels)

    if type(ratio_id) != list:
        ratio_id = [ratio_id] * len(labels)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    if figsize is None:
        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(8, 1, figure=fig)
    axis_dict["left"] = {}
    axis_dict["left"]["top"] = fig.add_subplot(gs[:5, 0])
    axis_dict["left"]["ratio"] = fig.add_subplot(
        gs[5:, 0], sharex=axis_dict["left"]["top"]
    )
    if "right" in which_axis:
        axis_dict["right"] = {}
        axis_dict["right"]["top"] = axis_dict["left"]["top"].twinx()

    if WorkingPoints is not None:
        for WP in WorkingPoints:

            # Set y-point of the WP lines/text
            ytext = 0.65 if same_height_WP else 1.25 - WP

            axis_dict["left"]["top"].axvline(
                x=WP,
                ymax=ytext,
                color=colors_WP,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Set the number above the line
            axis_dict["left"]["top"].text(
                x=WP - 0.005,
                y=ytext + 0.005,
                s=f"{int(WP * 100)}%",
                transform=axis_dict["left"]["top"].get_xaxis_text1_transform(
                    0
                )[0],
                fontsize=10,
            )

            axis_dict["left"]["ratio"].axvline(
                x=WP, color=colors_WP, linestyle="dashed", linewidth=1.0
            )

    # Create lines list and ratio dict for looping
    lines = []
    f0_ratio = {}

    # Loop over the models with the different settings for each model
    for i, (
        df_results,
        tagger,
        rej_class,
        label,
        style,
        color,
        nte,
        which_a,
        r_id,
    ) in enumerate(
        zip(
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
            styles,
            colors,
            nTest,
            which_axis,
            ratio_id,
        )
    ):

        # Get the main class efficency for x-axis
        main_class_effs = df_results[df_eff_key]

        # Get the rejections
        class_rejections = df_results[f"{tagger}_{rej_class}_rej"]

        # Check which rejection is on which axis and set label
        if which_a == "right":
            ylabel_right = f'{flav_cat[rej_class]["legend_label"]} Rejection'

        elif which_a == "left":
            ylabel = f'{flav_cat[rej_class]["legend_label"]} Rejection'

        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(main_class_effs)))

        # Also mask the rejections that are 0
        nonzero = (class_rejections != 0) & (dx > 0)
        if xmin:
            nonzero = nonzero & (main_class_effs > xmin)
        x = main_class_effs[nonzero]
        y = class_rejections[nonzero]

        # Plot the lines in the main plot and add them to lines list
        lines = lines + axis_dict[which_a]["top"].plot(
            x, y, linestyle=style, color=color, label=label, zorder=2
        )

        # Calculate and plot binominal errors for main plot
        if binomialErrors is True:
            yerr = np.power(y, 2) * rej_err(class_rejections[nonzero], nte)
            y1 = y - yerr
            y2 = y + yerr

            axis_dict[which_a]["top"].fill_between(
                x, y1, y2, color=color, alpha=0.3, zorder=2
            )

        # Interpolate the rejection function for nicer plotting
        f = pchip(x, y)

        # Check if the ratio_id divisor was already used or not
        # If not, calculate the divisor for ratio_id and add it to list
        if r_id not in f0_ratio:
            f0_ratio[r_id] = f
            axis_dict["left"]["ratio"].plot(
                x, np.ones(len(x)), linestyle=style, color=color, linewidth=1.6
            )
            if binomialErrors is True:
                axis_dict["left"]["ratio"].fill_between(
                    x,
                    1 - yerr / y,
                    1 + yerr / y,
                    color=color,
                    alpha=0.3,
                    zorder=1,
                )
            continue

        # If ratio_id divisor already calculated, plot calculate ratio and plot
        ratio_ix = f(x) / f0_ratio[r_id](x)
        axis_dict["left"]["ratio"].plot(
            x, ratio_ix, linestyle=style, color=color, linewidth=1.6
        )
        if binomialErrors is True:
            axis_dict["left"]["ratio"].fill_between(
                x,
                ratio_ix - yerr / f(x),
                ratio_ix + yerr / f(x),
                color=color,
                alpha=0.3,
                zorder=1,
            )

    # Add axes, titles and the legend
    axis_dict["left"]["top"].set_ylabel(
        ylabel, fontsize=12, horizontalalignment="right", y=1.0, color=ycolor
    )
    axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["top"].grid()
    if set_logy:
        axis_dict["left"]["top"].set_yscale("log")
    axis_dict["left"]["ratio"].set_xlabel(
        f'{flav_cat[main_class]["legend_label"]} Efficiency',
        fontsize=12,
        horizontalalignment="right",
        x=1.0,
    )
    axis_dict["left"]["ratio"].set_ylabel(
        rlabel, labelpad=labelpad, fontsize=12
    )
    axis_dict["left"]["ratio"].grid()

    if "right" in axis_dict:
        axis_dict["right"]["top"].set_ylabel(
            ylabel_right,
            fontsize=12,
            horizontalalignment="right",
            y=1.0,
            color=ycolor_right,
        )
        axis_dict["right"]["top"].tick_params(
            axis="y", labelcolor=ycolor_right
        )
        if set_logy:
            axis_dict["right"]["top"].set_yscale("log")

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    # Print label
    if alabel is not None:
        axis_dict["left"]["top"].text(
            **alabel, transform=axis_dict["left"]["top"].transAxes
        )

    # Auto set x-limit
    axis_dict["left"]["top"].set_xlim(
        df_results_list[0][df_eff_key].iloc[0],
        df_results_list[0][df_eff_key].iloc[-1],
    )

    # Manually set xmin
    if xmin:
        axis_dict["left"]["top"].set_xlim(xmin, 1)

    # Check for ymin/ymax and set y-axis
    if set_logy is True:
        left_y_limits = axis_dict["left"]["top"].get_ylim()
        yAxisIncrease = (
            left_y_limits[0]
            * ((left_y_limits[1] / left_y_limits[0]) ** yAxisIncrease)
            / left_y_limits[1]
        )

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    new_ymin_left = left_y_limits[0] if ymin is None else ymin
    new_ymax_left = left_y_limits[1] * yAxisIncrease if ymax is None else ymax
    axis_dict["left"]["top"].set_ylim(new_ymin_left, new_ymax_left)

    if "right" in axis_dict:
        right_y_limits = axis_dict["right"]["top"].get_ylim()
        new_ymin_right = (
            right_y_limits[0] if ymin_right is None else ymin_right
        )
        new_ymax_right = (
            right_y_limits[1] * yAxisIncrease
            if ymax_right is None
            else ymax_right
        )
        axis_dict["right"]["top"].set_ylim(new_ymin_right, new_ymax_right)

    # Increase the ratio y-axis if wanted
    if rrange is not None:
        axis_dict["left"]["ratio"].set_ylim(rrange)

    # Define the legend
    axis_dict["left"]["top"].legend(
        handles=lines,
        labels=[line.get_label() for line in lines],
        loc=loc_legend,
        fontsize=legFontSize,
        ncol=legcols,
    )

    # Define ATLAS tag
    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Set tight layout
    plt.tight_layout()

    # Save plot
    plt.savefig(plot_name, transparent=True)
    plt.close()
    plt.clf()


def plotROCRatioComparison(
    df_results_list: list,
    tagger_list: list,
    rej_class_list: list,
    labels: list,
    plot_name: str,
    df_eff_key: str = "effs",
    main_class: str = "bjets",
    title: str = "",
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.3,
    linestyles: list = [],
    colors: list = [],
    xmin: float = None,
    ymax: float = None,
    ymin: float = None,
    labelFontSize: int = 10,
    legFontSize: int = 10,
    loc_legend: str = "best",
    Ratio_Cut: list = None,
    binomialErrors: bool = True,
    nTest: list = 0,
    alabel: dict = None,
    figsize: list = None,
    legcols: int = 1,
    labelpad: int = None,
    WorkingPoints: list = None,
    same_height_WP: bool = True,
    ratio_id: list = 0,
    ycolor: str = "black",
    ycolor_right: str = "black",
    set_logy: bool = True,
):
    """
    Plot the ROC curves with binomial errors with the two ratio plot in a subpanel
    underneath. This function is steered by the plotting_umami.py. Documentation is
    provided in the docs folder.

    Addtional Inputs:
    - Ratio_Cut: The range on the y-axis for the ratio panel
    - rlabel: The label for the y-axis for the ratio panel
    - binomialErrors: whether to include binomial errors for the rejection
                      curves
    - nTest: A list of the same length as beffs, with the number of events used
            to calculate the background efficiencies.
            We need this To calculate the binomial errors on the background
            rejection,
            using the formula given by
            http://home.fnal.gov/~paterno/images/effic.pdf.
    """
    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Loop over the given rejection types and add them to a lists
    flav_list = list(dict.fromkeys(rej_class_list))

    # Append a linestyles for each model determined by the rejections
    if len(linestyles) == 0:
        for which_j in rej_class_list:
            for i, flav in enumerate(flav_list):
                if which_j == flav:
                    if i == 0:
                        # This is solids
                        linestyles.append("-")

                    elif i == 1:
                        # This is densly dashed dotted
                        linestyles.append((0, (3, 1, 1, 1)))

                    else:
                        raise ValueError("Can't plot more than 2 rejections!")

    # Create list for the models
    model_list = []
    for label in labels:
        if label not in model_list:
            model_list.append(label)

    # Fill in the colors for the models given
    if len(colors) == 0:
        for label in labels:
            for i, model in enumerate(model_list):
                if label == model:
                    colors.append(f"C{i}")

    # Set WP colors
    colors_WP = "red"

    if binomialErrors is True:
        # Check if nTest is provided in all samples
        nTest_in_file = []
        for df_results in df_results_list:
            if "N_test" not in df_results:
                nTest_in_file.append(False)

            else:
                nTest_in_file.append(True)

        if nTest == 0 and not all(nTest_in_file):
            logger.error(
                "Requested binomialErrors, but not all models have nTest. Will NOT plot rej errors."
            )
            binomialErrors = False

    if type(nTest) != list:
        nTest = [nTest] * len(labels)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    # Create figure with the given size, if provided.
    if figsize is None:
        fig = plt.figure(figsize=(8, 8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    # Define the grid of the subplots
    gs = gridspec.GridSpec(11, 1, figure=fig)
    axis_dict["left"] = {}

    # Define the top subplot for the curves
    axis_dict["left"]["top"] = fig.add_subplot(gs[:5, 0])

    # Define the ratio plots for the rejections
    axis_dict["left"][flav_list[0]] = fig.add_subplot(
        gs[5:8, 0], sharex=axis_dict["left"]["top"]
    )
    axis_dict["left"][flav_list[1]] = fig.add_subplot(
        gs[8:, 0], sharex=axis_dict["left"]["top"]
    )

    # Draw WP lines at the specifed WPs
    if WorkingPoints is not None:
        for WP in WorkingPoints:

            # Set y-point of the WP lines/text
            ytext = 0.65 if same_height_WP else 1.25 - WP

            # Plot the vertical WP lines for top plot
            axis_dict["left"]["top"].axvline(
                x=WP,
                ymax=ytext,
                color=colors_WP,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Set the number above the line
            axis_dict["left"]["top"].text(
                x=WP - 0.005,
                y=ytext + 0.005,
                s=f"{int(WP * 100)}%",
                transform=axis_dict["left"]["top"].get_xaxis_text1_transform(
                    0
                )[0],
                fontsize=10,
            )

            # Draw the WP lines in the ratio plots
            for flav in flav_list:
                axis_dict["left"][flav].axvline(
                    x=WP, color=colors_WP, linestyle="dashed", linewidth=1.0
                )

    # Create lines list and ratio dict for looping
    lines = []
    f0_ratio = {}

    assert (
        len(df_results_list)
        == len(tagger_list)
        == len(rej_class_list)
        == len(labels)
        == len(linestyles)
        == len(colors)
        == len(nTest)
        == len(ratio_id)
    ), "Input configs must be given for each model. Dimension Error!"

    # Loop over the models with the different settings for each model
    for i, (
        df_results,
        tagger,
        rej_class,
        label,
        style,
        color,
        nte,
        r_id,
    ) in enumerate(
        zip(
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
            linestyles,
            colors,
            nTest,
            ratio_id,
        )
    ):

        # Get the main class efficency for x-axis
        main_class_effs = df_results[df_eff_key]

        # Get the rejections
        class_rejections = df_results[f"{tagger}_{rej_class}_rej"]

        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(main_class_effs)))

        # Also mask the rejections that are 0
        nonzero = (class_rejections != 0) & (dx > 0)
        if xmin:
            nonzero = nonzero & (main_class_effs > xmin)
        x = main_class_effs[nonzero]
        y = class_rejections[nonzero]

        # Plot the lines in the main plot and add them to lines list
        lines = lines + axis_dict["left"]["top"].plot(
            x, y, linestyle=style, color=color, label=label, zorder=2
        )

        # Calculate and plot binominal errors for main plot
        if binomialErrors:
            yerr = np.power(y, 2) * rej_err(class_rejections[nonzero], nte)

            y1 = y - yerr
            y2 = y + yerr

            axis_dict["left"]["top"].fill_between(
                x, y1, y2, color=color, alpha=0.3, zorder=2
            )

        # Interpolate the rejection function for nicer plotting
        f = pchip(x, y)

        # Check if the ratio_id divisor was already used or not
        # If not, calculate the divisor for ratio_id and add it to list
        if r_id not in f0_ratio:
            f0_ratio[r_id] = f
            axis_dict["left"][rej_class].plot(
                x, np.ones(len(x)), linestyle=style, color=color, linewidth=1.6
            )
            if binomialErrors:
                axis_dict["left"][rej_class].fill_between(
                    x,
                    1 - yerr / y,
                    1 + yerr / y,
                    color=color,
                    alpha=0.3,
                    zorder=1,
                )
            continue

        # If ratio_id divisor already calculated, plot calculate ratio and plot
        ratio_ix = f(x) / f0_ratio[r_id](x)
        axis_dict["left"][rej_class].plot(
            x, ratio_ix, linestyle=style, color=color, linewidth=1.6
        )
        if binomialErrors:
            axis_dict["left"][rej_class].fill_between(
                x,
                ratio_ix - yerr / f(x),
                ratio_ix + yerr / f(x),
                color=color,
                alpha=0.3,
                zorder=1,
            )

    # Add axes, titles and the legend
    axis_dict["left"]["top"].set_ylabel(
        "Background rejection",
        fontsize=labelFontSize,
        horizontalalignment="right",
        y=1.0,
        color=ycolor,
    )
    axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["top"].grid()

    # Check for log scale
    if set_logy:
        axis_dict["left"]["top"].set_yscale("log")

    # Set grid for the ratio plots and set ylabel
    for flav in flav_list:
        axis_dict["left"][flav].grid()
        rlabel = f'{flav_cat[flav]["legend_label"]} ratio'

        axis_dict["left"][flav].set_ylabel(
            rlabel,
            labelpad=labelpad,
            fontsize=labelFontSize,
        )

    # Set xlabel for lowest ratio plot
    axis_dict["left"][flav_list[1]].set_xlabel(
        f'{flav_cat[main_class]["legend_label"]} efficiency',
        fontsize=labelFontSize,
        horizontalalignment="right",
        x=1.0,
    )

    # Hide the xlabels of the upper ratio and the main plot
    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)
    plt.setp(axis_dict["left"][flav_list[0]].get_xticklabels(), visible=False)

    # Print label om plot
    if alabel is not None:
        axis_dict["left"]["top"].text(
            **alabel, transform=axis_dict["left"]["top"].transAxes
        )

    # Auto set x-limit
    axis_dict["left"]["top"].set_xlim(
        df_results_list[0][df_eff_key].iloc[0],
        df_results_list[0][df_eff_key].iloc[-1],
    )

    # Manually set xmin
    if xmin:
        axis_dict["left"]["top"].set_xlim(xmin, 1)

    # Check for ymin/ymax and set y-axis
    if set_logy is True:
        left_y_limits = axis_dict["left"]["top"].get_ylim()
        yAxisIncrease = (
            left_y_limits[0]
            * ((left_y_limits[1] / left_y_limits[0]) ** yAxisIncrease)
            / left_y_limits[1]
        )

    # Check for ymin/ymax and set y-axis
    left_y_limits = axis_dict["left"]["top"].get_ylim()
    new_ymin_left = left_y_limits[0] if ymin is None else ymin
    new_ymax_left = left_y_limits[1] * yAxisIncrease if ymax is None else ymax
    axis_dict["left"]["top"].set_ylim(new_ymin_left, new_ymax_left)

    # Set ratio range
    if Ratio_Cut is not None:
        for flav in flav_list:
            axis_dict["left"][flav].set_ylim(Ratio_Cut[0], Ratio_Cut[1])

    # Create the two legends for rejection and model
    line_list_rej = []
    for i in range(2):
        label = f'{flav_cat[flav_list[i]]["legend_label"]} rejection'
        line = axis_dict["left"]["top"].plot(
            np.nan, np.nan, color="k", label=label, linestyle=["-", "--"][i]
        )
        line_list_rej += line

    legend1 = axis_dict["left"]["top"].legend(
        handles=line_list_rej,
        labels=[tmp.get_label() for tmp in line_list_rej],
        loc="upper center",
        fontsize=legFontSize,
        ncol=legcols,
    )

    # Add the second legend to plot
    axis_dict["left"]["top"].add_artist(legend1)

    # Get the labels for the legends
    labels_list = []
    lines_list = []

    for line in lines:
        for model in model_list:
            if (
                line.get_label() == model
                and line.get_label() not in labels_list
            ):
                labels_list.append(line.get_label())
                lines_list.append(line)

    # Define the legend
    axis_dict["left"]["top"].legend(
        handles=lines_list,
        labels=labels_list,
        loc=loc_legend,
        fontsize=legFontSize,
        ncol=legcols,
    )

    # Define ATLASTag
    if UseAtlasTag is True:
        pas.makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Set tight layout
    plt.tight_layout()

    # Set filename and save figure
    plt.savefig(plot_name, transparent=True)
    plt.close()
    plt.clf()


def plotSaliency(
    maps_dict,
    plot_name,
    title,
    target_beff=0.77,
    jet_flavour="bjets",
    PassBool=True,
    nFixedTrks=8,
    fontsize=14,
    xlabel="Tracks sorted by $s_{d0}$",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag=r"$\sqrt{s}$ = 13 TeV, $t\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    FlipAxis=False,
):
    # Transform to percent
    target_beff = 100 * target_beff

    # Little Workaround
    AtlasTag = " " + AtlasTag

    gradient_map = maps_dict[
        "{}_{}_{}".format(int(target_beff), jet_flavour, PassBool)
    ]

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

        plt.yticks(
            np.arange(nFeatures), yticklabels[:nFeatures], fontsize=fontsize
        )

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
    plt.savefig(plot_name, transparent=True, bbox_inches="tight")


def plot_score(
    df_results,
    plot_name: str,
    tagger_name: str,
    class_labels: list,
    main_class: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    WorkingPoints: list = None,
    nBins: int = 50,
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    xlabel: str = None,
    WorkingPoints_Legend: bool = False,
):
    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

    # Calculate the binning for all flavours
    _, Binning = np.histogram(
        df_results.query(f"labels=={index_dict[main_class]}")[
            f"disc_{tagger_name}"
        ],
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
        bins, weights, unc, band = calc_bins(
            input_array=flav_tracks,
            Binning=Binning,
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

    # Define handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

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
                colors="red",
                linestyles="dashed",
                linewidth=2.0,
            )

            # Set the number above the line
            ax.annotate(
                "{}%".format(int(WP * 100)),
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
    plt.savefig(plot_name, transparent=True)
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
    AtlasTag: str = "Internal Simulation",
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
    x_label=r"$D_b$",
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    ycolor: str = "black",
    ycolor_right: str = "black",
    title: str = None,
):
    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {
        f"{flavour}": i for i, flavour in enumerate(class_labels_list[0])
    }

    if type(which_axis) != list:
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
        df_list[0].query(f"labels=={index_dict[main_class]}")[
            f"disc_{tagger_list[0]}"
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
        index_dict = {
            f"{flavour}": i for i, flavour in enumerate(class_labels)
        }

        for flavour in class_labels:
            # Select correct jets
            flav_tracks = df_results.query(f"labels=={index_dict[flavour]}")[
                f"disc_{tagger}"
            ]

            bins, weights, unc, band = calc_bins(
                input_array=flav_tracks,
                Binning=Binning,
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
                step, step_unc = calc_ratio(
                    counter=bincounts["{}{}".format(flavour, i)],
                    denominator=bincounts["{}{}".format(flavour, 0)],
                    counter_unc=bincounts_unc["{}{}".format(flavour, i)],
                    denominator_unc=bincounts_unc["{}{}".format(flavour, 0)],
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
        axis_dict["left"]["ratio"].set_ylim(
            bottom=Ratio_Cut[0], top=Ratio_Cut[1]
        )

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
            tagger_name_WP = (
                tagger_list[0] if tagger_for_WP is None else tagger_for_WP
            )

            # Calculate x value of WP line
            x_value = np.percentile(
                df_results.query(f"labels=={index_dict[main_class]}")[
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
                "{}%".format(int(WP * 100)),
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
    plt.savefig(plot_name, transparent=True)
    plt.close()


def plotFractionScan(
    data,
    label,
    plot_name,
    x_val,
    y_val,
    UseAtlasTag=True,
    AtlasTag="Internal",
    SecondTag="$\\sqrt{s}=13$ TeV, PFlow Jets, $t\\bar{t}$",
):
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
        plt.savefig(plot_name, transparent=True)
    plt.close()


def plot_prob(
    df_results,
    plot_name: str,
    tagger_name: str,
    class_labels: list,
    flavour: str,
    ApplyAtlasStyle: bool = True,
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    nBins: int = 50,
    Log: bool = False,
    figsize: list = None,
    labelFontSize: int = 10,
    loc_legend: str = "best",
    ncol: int = 2,
    x_label: str = None,
    yAxisIncrease: float = 1.3,
    yAxisAtlasTag: float = 0.9,
):
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
            f'{tagger_name}_{flav_cat[iter_flavour]["prob_var_name"]}'
        ]

        bins, weights, unc, band = calc_bins(
            input_array=flav_tracks,
            Binning=Binning,
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

    if Log is True:
        plt.yscale("log")
        ymin, ymax = plt.ylim()

        if ymin <= 1e-8:
            ymin = 1e-8

        # Increase ymax so atlas tag don't cut plot
        plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)

    else:
        # Increase ymax so atlas tag don't cut plot
        ymin, ymax = plt.ylim()
        plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)

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
    plt.savefig(plot_name, transparent=True)
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
    AtlasTag: str = "Internal Simulation",
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
    x_label: str = None,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    ycolor: str = "black",
    ycolor_right: str = "black",
    title: str = None,
    set_logy: bool = False,
):
    # Apply the ATLAS Style with the bars on the axes
    if ApplyAtlasStyle is True:
        applyATLASstyle(mtp)

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Get index dict
    index_dict = {
        f"{iter_flav}": i for i, iter_flav in enumerate(class_labels_list[0])
    }

    if type(which_axis) != list:
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
        index_dict = {
            f"{iter_flav}": i for i, iter_flav in enumerate(class_labels)
        }

        for iter_flav in class_labels:
            # Select correct jets
            flav_tracks = df_results.query(f"labels=={index_dict[iter_flav]}")[
                f'{tagger}_{flav_cat[flavour]["prob_var_name"]}'
            ]

            # Calculate bins
            bins, weights, unc, band = calc_bins(
                input_array=flav_tracks,
                Binning=Binning,
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
                step, step_unc = calc_ratio(
                    counter=bincounts["{}{}".format(iter_flav, i)],
                    denominator=bincounts["{}{}".format(iter_flav, 0)],
                    counter_unc=bincounts_unc["{}{}".format(iter_flav, i)],
                    denominator_unc=bincounts_unc["{}{}".format(iter_flav, 0)],
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
        axis_dict["left"]["ratio"].set_ylim(
            bottom=Ratio_Cut[0], top=Ratio_Cut[1]
        )

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

    if set_logy is True:
        axis_dict["left"]["top"].set_yscale("log")

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    if set_logy is False:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0], left_y_limits[1] * yAxisIncrease
        )

    elif set_logy is True:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0] * 0.5,
            left_y_limits[0]
            * ((left_y_limits[1] / left_y_limits[0]) ** yAxisIncrease),
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
        plt.savefig(plot_name, transparent=True)
    plt.close()
    # plt.show()


def plot_confusion(
    df_results: dict,
    tagger_name: str,
    class_labels: list,
    plot_name: str,
    colorbar: bool = True,
    show_absolute: bool = False,
    show_normed: bool = True,
    transparent_bkg: bool = True,
):
    """
    Plotting the confusion matrix for a given tagger.

    Input:
    - df_results: Loaded pandas dataframe from the evaluation file.
    - tagger_name: Name of the tagger in the evaluation file.
    - class_labels: List of class labels used.
    - plot_name: Full path + name of the plot with the extension.
    - colorbar: Decide, if colourbar is shown or not.
    - show_absolute: Show the absolute.
    - show_normed: Show the output normed.
    - transparent_bkg: Decide, if the background is transparent or not.

    Output:
    - Confusion Matrix
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
    cm = confusion_matrix(
        y_target=y_target, y_predicted=y_predicted, binary=False
    )

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
    plt.savefig(plot_name, transparent=transparent_bkg)
    plt.close()
