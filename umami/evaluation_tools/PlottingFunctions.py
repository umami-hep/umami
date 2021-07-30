from umami.configuration import global_config, logger  # isort:skip
import pickle

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import pchip

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.tools import applyATLASstyle


def eff_err(x, N):
    return np.sqrt(x * (1 - x) / N)


def GetScore(pb, pc, pu, ptau=None, fc=0.018, ftau=None):
    pb = pb.astype("float64")
    pc = pc.astype("float64")
    pu = pu.astype("float64")
    add_small = 1e-10
    if ptau is not None:
        if ftau is None:
            flight = 1 - fc
            ftau = flight
        else:
            flight = 1 - fc - ftau
        ptau = ptau.astype("float64")
        return np.log(
            (pb + add_small)
            / (flight * pu + ftau * ptau + fc * pc + add_small)
        )
    return np.log((pb + add_small) / ((1.0 - fc) * pu + fc * pc + add_small))


def GetScoreC(pb, pc, pu, ptau=None, fb=0.2, ftau=None):
    pb = pb.astype("float64")
    pc = pc.astype("float64")
    pu = pu.astype("float64")
    add_small = 1e-10
    if ptau is not None:
        if ftau is None:
            flight = 1 - fb
            ftau = flight
        else:
            flight = 1 - fb - ftau
        ptau = ptau.astype("float64")
        return np.log(
            (pc + add_small)
            / (flight * pu + ftau * ptau + fb * pb + add_small)
        )
    return np.log((pc + add_small) / ((1.0 - fb) * pu + fb * pb + add_small))


def GetCutDiscriminant(pb, pc, pu, ptau=None, fc=0.018, ftau=None, wp=0.7):
    """
    Return the cut value on the b-discrimant to reach desired WP
    (working point).
    pb, pc, and pu (ptau) are the proba for the b-jets.
    """
    bscore = GetScore(pb, pc, pu, ptau=ptau, fc=fc, ftau=ftau)
    cutvalue = np.percentile(bscore, 100.0 * (1.0 - wp))
    return cutvalue


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


def discriminant_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    return (shape[0],)


def get_gradients(model, X, nJets):
    """
    Calculating the gradients with respect to the input variables.
    Note that only Keras backend functions can be used here because
    the gradients are tensorflow tensors and are not compatible with
    numpy.
    """
    gradients = K.gradients(model.output, model.inputs)

    input_tensors = model.inputs + [K.learning_phase()]
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # Pass in the cts and categorical inputs, as well as the learning phase
    # (0 for test mode)
    gradients = compute_gradients([X[:nJets], 0])

    return gradients[0]


def getDiscriminant(x, fc=0.018):
    """
    This method returns the score of the input (like GetScore)
    but calculated with the Keras Backend due to conflicts of
    numpy functions inside a layer in a keras model.
    Note: not yet compatible with taus
    """
    return K.log(x[:, 2] / (fc * x[:, 1] + (1 - fc) * x[:, 0]))


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
            total_var.append(0)
            u_lst.append(1e5)
            u_err_list.append(1)
            c_lst.append(1e5)
            c_err_list.append(1)
            b_lst.append(1e5)
            b_err_list.append(1)
            tau_lst.append(1e5)
            tau_err_list.append(1)
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
                u_lst.append(1e5)
                u_err_list.append(1)
            if not (in_c):
                c_lst.append(1e5)
                c_err_list.append(1)
            if not (in_b):
                b_lst.append(1e5)
                b_err_list.append(1)
            if not (in_tau):
                tau_lst.append(1e5)
                tau_err_list.append(1)
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
            + "$\\epsilon_b$ = {}%, $f_c$ = {}, $f_\tau$ = {}".format(
                efficiency, fc, ftau
            )
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
    prediction_labels,
    model_labels,
    plot_name,
    flavor=2,
    WP=0.77,
    Disc_Cut_Value=None,
    fc_list=[],
    fc=0.018,
    SWP_label_list=[],
    Passed=True,
    Fixed_WP_Bin=False,
    Same_WP_Cut_Comparison=False,
    Same_WP_Cut_Comparison_Model=0,
    bin_edges=[0, 20, 50, 90, 150, 300, 1000],
    WP_Line=False,
    figsize=None,
    Grid=False,
    binomialErrors=True,
    xlabel=r"$p_T$ in GeV",
    Log=None,
    colors=None,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample, fc=0.018",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.1,
    frameon=True,
    ncol=2,
    ymin=None,
    ymax=None,
    alpha=0.8,
    trans=True,
):
    """
    For a given list of models, plot the b-eff, l and c-rej as a function
    of jet pt for customizable WP and fc values.

    Following options are needed to be given:
    - df_list: List of the dataframes from the model. See plotting_umami.py
    - prediction_labels: Probability label names. I.e. [dips_pb, dips_pc, dips_pu]
                         The order is important!
    - model_labels: Labels for the legend of the plot
    - plot_name: Path, Name and format of the resulting plot file

    Following options are preset and can be changed:
    - flavor: Flavor ID of the jets type that should plotted.
              2: b jet (eff)
              1: c jet (rej)
              0: u jet (rej)
    - WP: Which Working point is used
    - fc_list: fc values for the different models. Give one for all
               or no model, otherwise you get an error
    - fc: If all models have the same fc value and none are given with
          the fc_list, this value is used for calculation
    - Passed: Select if the selected jets need to pass the discriminant WP cut
    - Fixed_WP_Bin: Calculate the WP cut on the discriminant per bin
    - Same_WP_Cut_Comparison: Use the same cut value on the b-tagging
                              discriminant for all models in the plot. Not works
                              with Fixed_WP_Bin True.
    - Same_WP_Cut_Comparison_Model: Model which cut is used in the Same WP Cut.
                                    0 is the first defined, 1 the second and so
                                    on.
    - bin_edges: As the name says, the edges of the bins used
    - WP_Line: Print a WP line in the upper plot
    - figsize: Size of the resulting figure
    - Grid: Use a grid in the plots
    - binomialErrors: Use binomial errors
    - xlabel: Label for x axis
    - Log: Set yscale to Log
    - colors: Custom color list for the different models
    - UseAtlasTag: Use the ATLAS Tag in the plots
    - AtlasTag: First row of the ATLAS Tag
    - SecondTag: Second Row of the ATLAS Tag. No need to add WP or fc.
                 It added automatically
    - yAxisAtlasTag: Relative y axis position of the ATLAS Tag in
    - yAxisIncrease: Increasing the y axis to fit the ATLAS Tag in
    - ymin: y axis minimum
    - ymax: y axis maximum
    - alpha: Value for visibility of the plot lines
    - trans: Sets the transparicy of the background. If true, the background erased.
             If False, the background is white
    """
    # Get the bins for the histogram
    pt_midpts = (np.asarray(bin_edges)[:-1] + np.asarray(bin_edges)[1:]) / 2.0
    bin_widths = (np.asarray(bin_edges)[1:] - np.asarray(bin_edges)[:-1]) / 2.0
    Npts = pt_midpts.size

    if Same_WP_Cut_Comparison is True:
        SWP_Cut_Dict = {}

    # Get flavor indices
    b_index, c_index, u_index = 2, 1, 0

    # Set fc default for proper labeling
    fc_default = False

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

    # Check if fc values or given
    if len(fc_list) == 0:
        fc_default = True

        for i in range(len(df_list)):
            fc_list.append(0.018)

    # Init new figure
    if figsize is None:
        fig = plt.figure(figsize=(8.27 * 0.8, 11.69 * 0.8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(8, 1, figure=fig)
    axis_dict["left"] = {}
    axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
    axis_dict["left"]["ratio"] = fig.add_subplot(
        gs[6:, 0], sharex=axis_dict["left"]["top"]
    )

    for i, (df_results, model_label, pred_labels, color, fcs) in enumerate(
        zip(df_list, model_labels, prediction_labels, colors, fc_list)
    ):
        # Adding fc value to model label if needed
        if fc_default is False:
            model_label += "\nfc={}".format(fcs)

        # Calculate b-discriminant
        df_results["discs"] = GetScore(
            *[df_results[pX] for pX in pred_labels], fc=fcs
        )

        # Placeholder for the sig eff and bkg rejections
        effs = np.zeros(Npts)

        # Get truth labels
        truth_labels = df_results["labels"]

        if Fixed_WP_Bin is False:
            if Same_WP_Cut_Comparison is False:
                if Disc_Cut_Value is None:
                    # Calculate WP cutoff for b-disc
                    disc_cut = np.percentile(
                        df_results.query(
                            f"labels=={global_config.flavour_labels['b']}"
                        )["discs"],
                        (1 - WP) * 100,
                    )

                elif Disc_Cut_Value is not None:
                    disc_cut = Disc_Cut_Value

            elif (
                Same_WP_Cut_Comparison is True
                and SWP_label_list[i] not in SWP_Cut_Dict
            ):
                # Calc disc cut value for the SWP label model
                disc_cut = np.percentile(
                    df_results.query(
                        f"labels=={global_config.flavour_labels['b']}"
                    )["discs"],
                    (1 - WP) * 100,
                )

                # Set Value globally for the SWP label
                SWP_Cut_Dict.update({SWP_label_list[i]: disc_cut})

            elif (
                Same_WP_Cut_Comparison is True
                and SWP_label_list[i] in SWP_Cut_Dict
            ):
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

            # Cut on selected flavor jets with the wanted pt range
            den_mask = (
                (jetPts > pt_min)
                & (jetPts < pt_max)
                & (truth_labels == flavor)
            )

            if Fixed_WP_Bin is False:
                # Cut on jets which passed the WP
                if Passed is True:
                    num_mask = den_mask & (df_results["discs"] > disc_cut)

                else:
                    num_mask = den_mask & (df_results["discs"] <= disc_cut)

            else:
                # Setting pT mask for the selected bin to calculate
                # the disc cut value fot the particular bin
                pT_mask = (jetPts > pt_min) & (jetPts < pt_max)

                # If SWP is used, calculate the disc cut for the model if
                # its not added to the dict yet. If its already added,
                # the value is loaded. If SWP is false, the disc value
                # will be calculated for each of the models independently
                if Same_WP_Cut_Comparison is False:
                    disc_cut = np.percentile(
                        df_results.query(
                            f"labels=={global_config.flavour_labels['b']}"
                        )["discs"][pT_mask],
                        (1 - WP) * 100,
                    )

                elif (
                    Same_WP_Cut_Comparison is True
                    and SWP_label_list[i] not in SWP_Cut_Dict
                ):
                    # Calc disc cut value for the SWP label model
                    disc_cut = np.percentile(
                        df_results.query(
                            f"labels=={global_config.flavour_labels['b']}"
                        )["discs"][pT_mask],
                        (1 - WP) * 100,
                    )

                    # Set Value globally for the SWP label
                    SWP_Cut_Dict.update({SWP_label_list[i]: disc_cut})

                elif (
                    Same_WP_Cut_Comparison is True
                    and SWP_label_list[i] in SWP_Cut_Dict
                ):
                    # Set disc_cut for the model after its SWP label
                    disc_cut = SWP_Cut_Dict[SWP_label_list[i]]

                # Cut on jets which passed the WP
                if Passed is True:
                    num_mask = den_mask & (df_results["discs"] > disc_cut)

                else:
                    num_mask = den_mask & (df_results["discs"] <= disc_cut)

            # Sum masks for binominal error calculation
            nTest[j] = den_mask.sum()
            effs[j] = num_mask.sum() / nTest[j]

        # For b-jets, plot the eff: for l and c-jets, look at the rej
        if flavor == b_index:
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
    if flavor == b_index:
        metric = "efficiency"

    else:
        metric = "rejection"

    # Set addition to y label if fixed WP bin is True
    if Fixed_WP_Bin is False:
        Fixed_WP_Label = "Inclusive"

    else:
        Fixed_WP_Label = ""

    # Set flavor label for the y axis
    if flavor == b_index:
        flav_label = r"$b$-jet"

    elif flavor == c_index:
        flav_label = r"$c$-jet"

    elif flavor == u_index:
        flav_label = "light-flavour jet"

    # Set y label
    axis_dict["left"]["top"].set_ylabel(
        f"{Fixed_WP_Label} {flav_label}-{metric}",
        horizontalalignment="right",
        y=1.0,
    )

    # Set ratio y label
    axis_dict["left"]["ratio"].set_ylabel("Ratio")

    # Check for Logscale
    if Log:
        axis_dict["left"]["top"].set_yscale("log")

    # Set limits
    axis_dict["left"]["top"].set_xlim(bin_edges[0], bin_edges[-1])

    if flavor == b_index:
        axis_dict["left"]["top"].set_ylim(ymin=ymin, ymax=ymax)

    # Increase ymax so atlas tag don't cut plot
    if (ymin is None) and (ymax is None):
        ymin, ymax = axis_dict["left"]["top"].get_ylim()

    elif ymin is None:
        ymin, _ = axis_dict["left"]["top"].get_ylim()

    elif ymax is None:
        _, ymax = axis_dict["left"]["top"].get_ylim()

    # Increase the yaxis limit upper part by given factor to fit ATLAS Tag in
    axis_dict["left"]["top"].set_ylim(bottom=ymin, top=yAxisIncrease * ymax)

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

    # Set the ATLAS Tag
    if fc_default is True:
        if UseAtlasTag is True:
            pas.makeATLAStag(
                ax=axis_dict["left"]["top"],
                fig=fig,
                first_tag=AtlasTag,
                second_tag=(
                    SecondTag
                    + ", fc={}".format(fc)
                    + "\nWP = {}%".format(int(WP * 100))
                ),
                ymax=yAxisAtlasTag,
            )

    elif fc_default is False:
        if UseAtlasTag is True:
            pas.makeATLAStag(
                ax=axis_dict["left"]["top"],
                fig=fig,
                first_tag=AtlasTag,
                second_tag=(SecondTag + "\nWP = {}%".format(int(WP * 100))),
                ymax=yAxisAtlasTag,
            )

    # Set tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(plot_name, transparent=trans)
    plt.close()


def plotROCRatio(
    teffs,
    beffs,
    labels,
    title="",
    ylabel="Background rejection",
    tag="",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample, fc=0.018",
    yAxisAtlasTag=0.9,
    figDir="../figures",
    subDir="mc16d",
    styles=None,
    colors=None,
    xmin=None,
    ymax=None,
    ymin=None,
    legFontSize=10,
    loc_legend="best",
    rrange=None,
    rlabel="Ratio",
    binomialErrors=False,
    nTest=0,
    plot_name=None,
    alabel=None,
    figsize=None,
    legcols=2,
    labelpad=None,
    which_axis="left",
    WorkingPoints=None,
    x_label="$b$-jet efficiency",
    ylabel_right=None,
    ratio_id=0,
    ycolor="black",
    ycolor_right="black",
    set_logy=True,
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
    - nTest: A list of the same length as beffs, with the number of events used
            to calculate the background efficiencies.
            We need this To calculate the binomial errors on the background
            rejection,
            using the formula given by
            http://home.fnal.gov/~paterno/images/effic.pdf.
    """
    # set ylabel
    if ylabel == "light":
        ylabel = r"Light-Flavour Jet Rejection ($1/\epsilon_{l}$)"
    elif ylabel == "c":
        ylabel = r"$c$-Jet Rejection ($1/\epsilon_{c}$)"
    elif ylabel == "t":
        ylabel = r"Tau-Jet Rejection ($1/\epsilon_{\tau}$)"
    elif ylabel == "b":
        ylabel = r"$b$-Jet Rejection ($1/\epsilon_{b}$)"
    if ylabel_right == "light":
        ylabel_right = r"Light-Flavour Jet Rejection ($1/\epsilon_{l}$)"
    elif ylabel_right == "c":
        ylabel_right = r"$c$-Jet Rejection ($1/\epsilon_{c}$)"
    elif ylabel_right == "t":
        ylabel_right = r"Tau-Jet Rejection ($1/\epsilon_{\tau}$)"
    elif ylabel_right == "b":
        ylabel_right = r"$b$-Jet Rejection ($1/\epsilon_{b}$)"

    if binomialErrors and nTest == 0:
        logger.error(
            "Error: Requested binomialErrors, but did not pass nTest. Will NOT plot rej errors."
        )
        binomialErrors = False

    if styles is None:
        styles = ["-" for i in teffs]
    if colors is None:
        colors = ["C{}".format(i) for i in range(len(teffs))]
        colors_WP = "C{}".format(len(colors) + 1)

    else:
        colors_WP = "red"

    if type(nTest) != list:
        nTest = [nTest] * len(teffs)

    if type(which_axis) != list:
        which_axis = [which_axis] * len(teffs)

    if type(ratio_id) != list:
        ratio_id = [ratio_id] * len(teffs)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    if figsize is None:
        fig = plt.figure(figsize=(8.27 * 0.8, 11.69 * 0.8))

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
            axis_dict["left"]["top"].axvline(
                x=WP,
                ymax=0.65,
                color=colors_WP,
                linestyle="dashed",
                linewidth=1.0,
            )

            axis_dict["left"]["ratio"].axvline(
                x=WP, color=colors_WP, linestyle="dashed", linewidth=1.0
            )

            # Set the number above the line
            axis_dict["left"]["top"].annotate(
                text="{}%".format(int(WP * 100)),
                xy=(WP, 0.79),
                xytext=(WP, 0.79),
                textcoords="offset points",
                xycoords=("data", "figure fraction"),
                ha="center",
                va="bottom",
                size=10,
            )

    lines = []
    f0_ratio = {}
    for i, (teff, beff, label, style, color, nte, which_a, r_id) in enumerate(
        zip(teffs, beffs, labels, styles, colors, nTest, which_axis, ratio_id)
    ):

        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(teff)))

        # Also mask the rejections that are 0
        nonzero = (beff != 0) & (dx > 0)
        if xmin:
            nonzero = nonzero & (teff > xmin)
        x = teff[nonzero]
        y = np.divide(1, beff[nonzero])

        lines = lines + axis_dict[which_a]["top"].plot(
            x, y, linestyle=style, color=color, label=label, zorder=2
        )
        if binomialErrors:
            yerr = np.power(y, 2) * eff_err(beff[nonzero], nte)

            y1 = y - yerr
            y2 = y + yerr

            axis_dict[which_a]["top"].fill_between(
                x, y1, y2, color=color, alpha=0.3, zorder=2
            )

        f = pchip(x, y)

        if r_id not in f0_ratio:
            f0_ratio[r_id] = f
            axis_dict["left"]["ratio"].plot(
                x, np.ones(len(x)), linestyle=style, color=color, linewidth=1.6
            )
            if binomialErrors:
                axis_dict["left"]["ratio"].fill_between(
                    x,
                    1 - yerr / y,
                    1 + yerr / y,
                    color=color,
                    alpha=0.3,
                    zorder=1,
                )
            continue
        ratio_ix = f(x) / f0_ratio[r_id](x)
        axis_dict["left"]["ratio"].plot(
            x, ratio_ix, linestyle=style, color=color, linewidth=1.6
        )
        if binomialErrors:
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
        x_label, fontsize=12, horizontalalignment="right", x=1.0
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

    axis_dict["left"]["top"].set_xlim(teffs[0].iloc[0], teffs[0].iloc[-1])
    if xmin:
        axis_dict["left"]["top"].set_xlim(xmin, 1)

    if ymax is not None:
        if ymin is not None:
            axis_dict["left"]["top"].set_ylim(ymin, ymax)
        else:
            axis_dict["left"]["top"].set_ylim(1, ymax)
    elif ymin is not None:
        _, top = axis_dict["left"]["top"].get_ylim()
        axis_dict["left"]["top"].set_ylim(ymin, top)

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    axis_dict["left"]["top"].set_ylim(left_y_limits[0], left_y_limits[1] * 1.2)
    if "right" in axis_dict:
        right_y_limits = axis_dict["right"]["top"].get_ylim()
        axis_dict["right"]["top"].set_ylim(
            right_y_limits[0], right_y_limits[1] * 1.2
        )

    if rrange is not None:
        axis_dict["left"]["ratio"].set_ylim(rrange)
    axis_dict["left"]["top"].legend(
        handles=lines,
        labels=[line.get_label() for line in lines],
        loc=loc_legend,
        fontsize=legFontSize,
        ncol=legcols,
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
    if len(tag) != 0:
        plt.savefig(
            "{}/{}/rocRatio_{}.pdf".format(figDir, subDir, tag),
            bbox_inches="tight",
            transparent=True,
        )
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True)
    plt.close()
    # plt.show()


def plotROCRatioComparison(
    teffs,
    beffs,
    labels,
    which_rej,
    title="",
    ylabel="Background Rejection",
    tag="",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample, fc=0.018",
    yAxisAtlasTag=0.9,
    figDir="../figures",
    subDir="mc16d",
    xmin=None,
    ymax=None,
    ymin=None,
    legFontSize=10,
    loc_legend="best",
    Ratio_Cut=None,
    binomialErrors=False,
    nTest=0,
    plot_name=None,
    alabel=None,
    figsize=None,
    legcols=2,
    labelpad=None,
    WorkingPoints=None,
    x_label="$b$-jet efficiency",
    ratio_id=0,
    ycolor="black",
    ycolor_right="black",
    set_logy=True,
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
    applyATLASstyle(mtp)

    # Define empty lists for the styles, colors and rejs
    styles = []
    colors = []
    flav_list = []

    # Loop over the given rejection types and add them to a lists
    for which_j in which_rej:
        if which_j not in flav_list:
            flav_list.append(which_j)

    # Append a styles for each model determined by the rejections
    if len(styles) == 0:
        for which_j in which_rej:
            for i, flav in enumerate(flav_list):
                if which_j == flav:
                    if i == 0:
                        # This is solids
                        styles.append("-")

                    elif i == 1:
                        # This is densly dashed dotted
                        styles.append((0, (3, 1, 1, 1)))

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

    if binomialErrors and nTest == 0:
        logger.error(
            "Error: Requested binomialErrors, but did not pass nTest. Will NOT plot rej errors."
        )
        binomialErrors = False

    if type(nTest) != list:
        nTest = [nTest] * len(teffs)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    # Create figure with the given size, if provided.
    if figsize is None:
        fig = plt.figure(figsize=(8.27 * 0.8, 11.69 * 0.8))

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
            axis_dict["left"]["top"].axvline(
                x=WP,
                ymax=0.65,
                color=colors_WP,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Draw the WP lines in the ratio plots
            for flav in flav_list:
                axis_dict["left"][flav].axvline(
                    x=WP, color=colors_WP, linestyle="dashed", linewidth=1.0
                )

            # Set the number above the line
            axis_dict["left"]["top"].annotate(
                text="{}%".format(int(WP * 100)),
                xy=(WP, 0.85),
                xytext=(WP, 0.85),
                textcoords="offset points",
                xycoords=("data", "figure fraction"),
                ha="center",
                va="bottom",
                size=10,
            )

    # Create lines list and ratio dict for looping
    lines = []
    f0_ratio = {}

    # Loop over the models with the different settings for each model
    for i, (teff, beff, label, style, color, nte, which_j, r_id) in enumerate(
        zip(teffs, beffs, labels, styles, colors, nTest, which_rej, ratio_id)
    ):

        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(teff)))

        # Also mask the rejections that are 0
        nonzero = (beff != 0) & (dx > 0)
        if xmin:
            nonzero = nonzero & (teff > xmin)
        x = teff[nonzero]

        # Calculate Rejection
        y = np.divide(1, beff[nonzero])

        # Plot the lines in the main plot and add them to lines list
        lines = lines + axis_dict["left"]["top"].plot(
            x, y, linestyle=style, color=color, label=label, zorder=2
        )

        # Calculate and plot binominal errors for main plot
        if binomialErrors:
            yerr = np.power(y, 2) * eff_err(beff[nonzero], nte)

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
            axis_dict["left"][which_j].plot(
                x, np.ones(len(x)), linestyle=style, color=color, linewidth=1.6
            )
            if binomialErrors:
                axis_dict["left"][which_j].fill_between(
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
        axis_dict["left"][which_j].plot(
            x, ratio_ix, linestyle=style, color=color, linewidth=1.6
        )
        if binomialErrors:
            axis_dict["left"][which_j].fill_between(
                x,
                ratio_ix - yerr / f(x),
                ratio_ix + yerr / f(x),
                color=color,
                alpha=0.3,
                zorder=1,
            )

    # Add axes, titles and the legend
    axis_dict["left"]["top"].set_ylabel(
        ylabel, fontsize=10, horizontalalignment="right", y=1.0, color=ycolor
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

        if flav != "c" or flav != "b":
            rlabel = f"{flav} Ratio"

        else:
            rlabel = r"${}$ Ratio"

        axis_dict["left"][flav].set_ylabel(
            rlabel,
            labelpad=labelpad,
            fontsize=10,
        )

    # Set xlabel for lowest ratio plot
    axis_dict["left"][flav_list[1]].set_xlabel(
        x_label, fontsize=10, horizontalalignment="right", x=1.0
    )

    # Hide the xlabels of the upper ratio and the main plot
    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)
    plt.setp(axis_dict["left"][flav_list[0]].get_xticklabels(), visible=False)

    # Print label om plot
    if alabel is not None:
        axis_dict["left"]["top"].text(
            **alabel, transform=axis_dict["left"]["top"].transAxes
        )

    # Set xlimit
    axis_dict["left"]["top"].set_xlim(teffs[0].iloc[0], teffs[0].iloc[-1])
    if xmin:
        axis_dict["left"]["top"].set_xlim(xmin, 1)

    # Check for ylimit and set according to options
    if ymax is not None:
        if ymin is not None:
            axis_dict["left"]["top"].set_ylim(ymin, ymax)
        else:
            axis_dict["left"]["top"].set_ylim(1, ymax)
    elif ymin is not None:
        _, top = axis_dict["left"]["top"].get_ylim()
        axis_dict["left"]["top"].set_ylim(ymin, top)

    # Increase y-axis for atlas label
    left_y_limits = axis_dict["left"]["top"].get_ylim()
    axis_dict["left"]["top"].set_ylim(left_y_limits[0], left_y_limits[1] * 1.2)

    # Set ratio range
    if Ratio_Cut is not None:
        for flav in flav_list:
            axis_dict["left"][flav].set_ylim(Ratio_Cut[0], Ratio_Cut[1])

    # Create the two legends for rejection and model
    line_list_rej = []
    for i in range(2):
        if flav_list[i] == "b" or flav_list[i] == "c":
            label = r"${}$ Rejection".format(flav_list[i])

        else:
            label = f"{flav_list[i]} Rejection"
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
    if len(tag) != 0:
        plt.savefig(
            "{}/{}/rocRatio_{}.pdf".format(figDir, subDir, tag),
            bbox_inches="tight",
            transparent=True,
        )
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True)
    plt.close()
    plt.clf()


def plotSaliency(
    plot_name,
    FileDir,
    epoch,
    data_set_name,
    title,
    target_beff=0.77,
    jet_flavour=2,
    PassBool=True,
    nFixedTrks=8,
    fs=14,
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

    with open(FileDir + f"/saliency_{epoch}_{data_set_name}.pkl", "rb") as f:
        maps_dict = pickle.load(f)

    gradient_map = maps_dict[
        "{}_{}_{}".format(int(target_beff), jet_flavour, PassBool)
    ]

    colorScale = np.max(np.abs(gradient_map))
    cmaps = ["RdBu", "PuOr", "PiYG"]

    nFeatures = gradient_map.shape[0]

    if FlipAxis is True:
        fig = plt.figure(figsize=(0.7 * nFeatures, 0.7 * nFixedTrks))
        gradient_map = np.swapaxes(gradient_map, 0, 1)

        plt.yticks(
            np.arange(nFixedTrks), np.arange(1, nFixedTrks + 1), fontsize=fs
        )

        plt.ylabel(xlabel, fontsize=fs)
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
            r"$\log \ p_T^{frac}$",
            r"$\log \ \Delta R$",
            "nPixHits",
            "nSCTHits",
            "$d_0$",
            r"$z_0 \sin \theta$",
        ]

        plt.xticks(np.arange(nFeatures), xticklabels[:nFeatures], rotation=45)

    else:
        fig = plt.figure(figsize=(0.7 * nFixedTrks, 0.7 * nFeatures))

        plt.xticks(
            np.arange(nFixedTrks), np.arange(1, nFixedTrks + 1), fontsize=fs
        )

        plt.xlabel(xlabel, fontsize=fs)
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
            r"$\log \ p_T^{frac}$",
            r"$\log \ \Delta R$",
            "nPixHits",
            "nSCTHits",
            "$d_0$",
            r"$z_0 \sin \theta$",
        ]

        plt.yticks(np.arange(nFeatures), yticklabels[:nFeatures])

    im = plt.imshow(
        gradient_map,
        cmap=cmaps[jet_flavour],
        origin="lower",
        vmin=-colorScale,
        vmax=colorScale,
    )

    plt.title(title, fontsize=fs)

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
        r"$\frac{\partial D_{b}}{\partial x_{ik}}$", size=1.5 * fs
    )

    # Save the figure
    plt.savefig(plot_name, transparent=True, bbox_inches="tight")


def plot_score(
    plot_name,
    plot_config,
    eval_params,
    eval_file_dir,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    WorkingPoints=None,
    nBins=50,
    yAxisIncrease=1.3,
    yAxisAtlasTag=0.9,
):
    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])
    if "bool_use_taus" in eval_params:
        bool_use_taus = eval_params["bool_use_taus"]

    else:
        bool_use_taus = False

    if "discriminant" in plot_config:
        discriminant = plot_config["discriminant"]

    else:
        discriminant = "b"

    # Read file, change to specific file if defined
    if ("evaluation_file" not in plot_config) or (
        plot_config["evaluation_file"] is None
    ):
        df_results = pd.read_hdf(
            eval_file_dir + f"/results-{eval_epoch}.h5",
            plot_config["data_set_name"],
        )

    else:
        df_results = pd.read_hdf(
            plot_config["evaluation_file"], plot_config["data_set_name"]
        )

    # Calculate the scores for the NN outputs
    if discriminant == "c":
        df_results["discs"] = GetScoreC(
            *[df_results[pX] for pX in plot_config["prediction_labels"]]
        )
    else:
        df_results["discs"] = GetScore(
            *[df_results[pX] for pX in plot_config["prediction_labels"]]
        )

    # Calculate the binning for all flavours
    _, Binning = np.histogram(
        df_results.query(f"labels=={global_config.flavour_labels['b']}")[
            "discs"
        ],
        bins=nBins,
    )

    # Define the flavours used
    if bool_use_taus is True:
        flav_list = ["b", "c", "u", "tau"]

    else:
        flav_list = ["b", "c", "u"]

    # Clear the figure and init a new one
    plt.clf()
    fig = plt.figure()
    ax = fig.gca()

    for flavour in flav_list:
        # Select correct jets
        flav_tracks = df_results.query(
            f"labels=={global_config.flavour_labels[flavour]}"
        )["discs"]

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
            color=global_config.flavour_colors[flavour],
            stacked=False,
            fill=False,
            label=global_config.flavour_legend_labels[flavour],
        )

        # Check for last item so the label for the legend is
        # only printed once
        if flavour == flav_list[-1]:
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

    # Set WP vertical lines if given in config
    if WorkingPoints is not None:

        # Iterate over WPs
        for WP in WorkingPoints:

            # Calculate x value of WP line
            if discriminant == "c":
                x_value = np.percentile(
                    df_results.query("labels==1")["discs"], (1 - WP) * 100
                )
                color = "#ff7f0e"
            else:
                x_value = np.percentile(
                    df_results.query(
                        f"labels=={global_config.flavour_labels['b']}"
                    )["discs"],
                    (1 - WP) * 100,
                )
                color = "red"

            # Draw WP line
            plt.vlines(
                x=x_value,
                ymin=ymin,
                ymax=WP * ymax,
                colors=color,
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

    plt.legend()
    if discriminant == "c":
        plt.xlabel("$D_{c}$")
    else:
        plt.xlabel("$D_{b}$")
    plt.ylabel("Normalised Number of Jets")

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
    df_list,
    prediction_labels_list,
    model_labels,
    plot_name,
    bool_use_taus=False,
    discriminant="b",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    tag="",
    WorkingPoints=None,
    nBins=50,
    figsize=None,
    labelpad=None,
    legFontSize=10,
    loc_legend="best",
    ncol=2,
    Ratio_Cut=None,
    which_axis="left",
    x_label=r"$D_b$",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    ycolor="black",
    ycolor_right="black",
    title=None,
    ylabel="Normalised Number of Jets",
    yAxisIncrease=1.3,
    yAxisAtlasTag=0.9,
):
    # Calculate the scores for the NN outputs
    for (df_results, prediction_labels) in zip(
        df_list, prediction_labels_list
    ):
        if discriminant == "b":
            df_results["discs"] = GetScore(
                *[df_results[pX] for pX in prediction_labels]
            )
        elif discriminant == "c":
            df_results["discs"] = GetScoreC(
                *[df_results[pX] for pX in prediction_labels]
            )
            if x_label == r"$D_b$":
                # Swap to c discriminant
                x_label = r"$D_c$"
        else:
            raise ValueError("Unknown discriminant {}!".format(discriminant))

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
        df_list[0].query(f"labels=={global_config.flavour_labels['b']}")[
            "discs"
        ],
        bins=nBins,
    )

    # Define the flavours used
    if bool_use_taus is True:
        flav_list = ["b", "c", "u", "tau"]

    else:
        flav_list = ["b", "c", "u"]

    # Init bincout and unc dict for ratio calculation
    bincounts = {}
    bincounts_unc = {}

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    for i, (df_results, linestyle, which_a, model_label) in enumerate(
        zip(df_list, linestyles, which_axis, model_labels)
    ):

        for flavour in flav_list:
            # Select correct jets
            flav_tracks = df_results.query(
                f"labels=={global_config.flavour_labels[flavour]}"
            )["discs"]

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
                color=global_config.flavour_colors[flavour],
                stacked=False,
                fill=False,
                label=global_config.flavour_legend_labels[flavour]
                + f" {model_label}",
            )

            if (flavour == flav_list[-1]) and i == 0:
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
            for flavour in flav_list:

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
                    color=global_config.flavour_colors[flavour],
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
        ylabel, fontsize=12, horizontalalignment="right", y=1.0, color=ycolor
    )
    if title is not None:
        axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["ratio"].set_xlabel(
        x_label, fontsize=12, horizontalalignment="right", x=1.0
    )

    axis_dict["left"]["ratio"].set_ylabel(
        "Ratio",
        labelpad=labelpad,
        fontsize=12,
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

            # Calculate x value of WP line
            if discriminant == "c":
                x_value = np.percentile(
                    df_list[0].query("labels==1")["discs"], (1 - WP) * 100
                )
                color = "#ff7f0e"
            else:
                x_value = np.percentile(
                    df_list[0].query(
                        f"labels=={global_config.flavour_labels['b']}"
                    )["discs"],
                    (1 - WP) * 100,
                )
                color = "#FF0000"

            # Draw WP line
            axis_dict["left"]["top"].axvline(
                x=x_value,
                ymin=0,
                ymax=0.75 * WP,
                color=color,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Draw WP line
            axis_dict["left"]["ratio"].axvline(
                x=x_value,
                ymin=0,
                ymax=1,
                color=color,
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
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True)
    plt.close()
    # plt.show()


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
    fig, ax = plt.subplots()
    plot = ax.imshow(
        z_data_table,
        cmap="RdBu",
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
    plot_name,
    plot_config,
    eval_params,
    eval_file_dir,
    bool_use_taus=False,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    nBins=50,
    Log=False,
    figsize=None,
    loc_legend="best",
    ncol=2,
    x_label="DNN Output",
    yAxisIncrease=1.3,
    yAxisAtlasTag=0.9,
):
    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])
    if "bool_use_taus" in eval_params:
        bool_use_taus = eval_params["bool_use_taus"]

    else:
        bool_use_taus = False

    # Read file, change to specific file if defined
    if ("evaluation_file" not in plot_config) or (
        plot_config["evaluation_file"] is None
    ):
        df_results = pd.read_hdf(
            eval_file_dir + f"/results-{eval_epoch}.h5",
            plot_config["data_set_name"],
        )

    else:
        df_results = pd.read_hdf(
            plot_config["evaluation_file"], plot_config["data_set_name"]
        )

    # Calculate the binning for all flavours
    _, Binning = np.histogram(
        df_results.query(f"labels=={global_config.flavour_labels['b']}")[
            plot_config["prediction_labels"]
        ],
        bins=nBins,
    )

    # Define the flavours used
    if bool_use_taus is True:
        flav_list = ["b", "c", "u", "tau"]

    else:
        flav_list = ["b", "c", "u"]

    # Clear the figure and init a new one
    if figsize is None:
        plt.clf()
        plt.figure(figsize=(8, 6))

    else:
        plt.clf()
        plt.figure(figsize=(figsize[0], figsize[1]))

    for flavour in flav_list:
        # Select correct jets
        flav_tracks = df_results.query(
            f"labels=={global_config.flavour_labels[flavour]}"
        )[plot_config["prediction_labels"]]

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
            color=global_config.flavour_colors[flavour],
            stacked=False,
            fill=False,
            label=global_config.flavour_legend_labels[flavour],
        )

        if flavour == flav_list[-1]:
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

    # Set x label
    if "x_label" in plot_config and plot_config["x_label"] is not None:
        plt.xlabel(plot_config["x_label"])

    else:
        plt.xlabel(plot_config["prediction_labels"])

    # Set y label
    plt.ylabel("Normalised Number of Jets")

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
    df_list,
    prediction_labels_list,
    model_labels,
    plot_name,
    bool_use_taus=False,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    tag="",
    nBins=50,
    figsize=None,
    labelpad=None,
    legFontSize=10,
    loc_legend="best",
    ncol=2,
    Log=False,
    Ratio_Cut=None,
    which_axis="left",
    x_label="DNN Output",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    ycolor="black",
    ycolor_right="black",
    title=None,
    ylabel="Normalised Number of Jets",
    yAxisIncrease=1.3,
    yAxisAtlasTag=0.9,
):
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
    if "right" in which_axis:
        axis_dict["right"] = {}
        axis_dict["right"]["top"] = axis_dict["left"]["top"].twinx()

    # Get binning for the plot
    _, Binning = np.histogram(
        df_list[0].query(f"labels=={global_config.flavour_labels['b']}")[
            prediction_labels_list[0]
        ],
        bins=nBins,
    )

    # Define the flavours used
    if bool_use_taus is True:
        flav_list = ["b", "c", "u", "tau"]

    else:
        flav_list = ["b", "c", "u"]

    # Init bincout and unc dict for ratio calculation
    bincounts = {}
    bincounts_unc = {}

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    for i, (
        df_results,
        prediction_label,
        linestyle,
        which_a,
        model_label,
    ) in enumerate(
        zip(
            df_list,
            prediction_labels_list,
            linestyles,
            which_axis,
            model_labels,
        )
    ):

        for flavour in flav_list:
            # Select correct jets
            flav_tracks = df_results.query(
                f"labels=={global_config.flavour_labels[flavour]}"
            )[prediction_label]

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
                color=global_config.flavour_colors[flavour],
                stacked=False,
                fill=False,
                label=global_config.flavour_legend_labels[flavour]
                + f" {model_label}",
            )

            if (flavour == flav_list[-1]) and i == 0:
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
            for flavour in flav_list:

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
                    color=global_config.flavour_colors[flavour],
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
        ylabel, fontsize=12, horizontalalignment="right", y=1.0, color=ycolor
    )
    if title is not None:
        axis_dict["left"]["top"].set_title(title)
    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)
    axis_dict["left"]["ratio"].set_xlabel(
        x_label, fontsize=12, horizontalalignment="right", x=1.0
    )

    axis_dict["left"]["ratio"].set_ylabel(
        "Ratio",
        labelpad=labelpad,
        fontsize=12,
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

    if Log is True:
        axis_dict["left"]["top"].set_yscale("log")

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    if Log is False:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0], left_y_limits[1] * yAxisIncrease
        )

    elif Log is True:
        axis_dict["left"]["top"].set_ylim(
            left_y_limits[0] * 0.5,
            left_y_limits[0]
            * (left_y_limits[1] / left_y_limits[0]) ** yAxisIncrease,
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
