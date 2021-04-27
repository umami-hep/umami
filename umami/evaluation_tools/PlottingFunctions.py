import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import pchip

import umami.tools.PyATLASstyle.PyATLASstyle as pas


def eff_err(x, N):
    return np.sqrt(x * (1 - x) / N)


def GetScore(pb, pc, pu, fc=0.018):
    pb = pb.astype("float64")
    pc = pc.astype("float64")
    pu = pu.astype("float64")
    add_small = 1e-10
    return np.log((pb + add_small) / ((1.0 - fc) * pu + fc * pc + add_small))


def GetCutDiscriminant(pb, pc, pu, ptau=None, fc=0.018, wp=0.7):
    """
    Return the cut value on the b-discrimant to reach desired WP
    (working point).
    pb, pc, and pu (ptau) are the proba for the b-jets.
    """
    bscore = GetScore(pb, pc, pu, ptau=ptau, fc=fc)
    cutvalue = np.percentile(bscore, 100.0 * (1.0 - wp))
    return cutvalue


def FlatEfficiencyPerBin(df, predictions, variable, var_bins, wp=0.7):
    """
    For each bin in var_bins of variable, cuts the score in
    bscore to get the desired WP (working point)
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
        bscores_b_in_bin = df_in_bin[df_in_bin["labels"] == 2]["bscore"]
        if len(bscores_b_in_bin) == 0:
            continue
        # print("There are {} b-jets in bin {}".format(len(bscores_b_in_bin), var_bins[i]))
        cutvalue_in_bin = np.percentile(bscores_b_in_bin, 100.0 * (1.0 - wp))
        df.loc[index_jets_in_bin, ["btag"]] = (
            df_in_bin["bscore"] > cutvalue_in_bin
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
    """
    return K.log(x[:, 2] / (fc * x[:, 1] + (1 - fc) * x[:, 0]))


def plotEfficiencyVariable(
    plot_name,
    df,
    variable,
    var_bins,
    fc=0.018,
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
    For a given variable (string) in the panda dataframe df, plot:
    - the b-eff in b, c, and light jets as a function of variable
                 (discretised in bins as indicated by var_bins input)
    - the variable distribution.

    The efficiency is computed from score. e.g., for the recommended DL1r
    at 70% WP:
        score = (data["disc_DL1r"]> 3.245)*1

    Entry i and i+1 in var_bins define a bin of the variable.

    df must at least contain columns with names:
    - variable
    - labels (2 for b, 1 for c, 0 for u, and optionally 3 for taus)
    - btag (with value 1 for b-tagged jets and 0 for others)

    Option centralise_bins can be used if the variable is made of
    integer so that datapoints are centralised to int value

    Note: to get a flat efficiency plot, you need to produce a 'btag' column
    in df using FlatEfficiencyPerBin (defined above).
    """
    data = df.copy()
    total_var = []
    b_lst, b_err_list = [], []
    c_lst, c_err_list = [], []
    u_lst, u_err_list = [], []
    t_lst, t_err_list = [], []

    for i in range(len(var_bins) - 1):
        df_in_bin = data[
            (var_bins[i] <= data[variable])
            & (data[variable] < var_bins[i + 1])
        ]
        total_b_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 2])
        total_c_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 1])
        total_u_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 0])
        total_t_tag_in_bin = len(df_in_bin[df_in_bin["labels"] == 3])
        df_in_bin = df_in_bin.query("btag == 1")
        total_in_bin = (
            total_b_tag_in_bin + total_c_tag_in_bin + total_u_tag_in_bin
        )
        if include_taus:
            total_in_bin += total_t_tag_in_bin

        if total_in_bin == 0:
            total_var.append(0)
            u_lst.append(1e5)
            u_err_list.append(1)
            c_lst.append(1e5)
            c_err_list.append(1)
            b_lst.append(1e5)
            b_err_list.append(1)
            t_lst.append(1e5)
            t_err_list.append(1)
        else:
            total_var.append(total_in_bin)
            index, counts = np.unique(
                df_in_bin["labels"].values, return_counts=True
            )
            in_b, in_c, in_u, in_t = False, False, False, False
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
                    eff = count / total_u_tag_in_bin
                    t_lst.append(eff)
                    t_err_list.append(eff_err(eff, total_t_tag_in_bin))
                    in_t = True
                else:
                    print("Invaled value of index from labels: ", item)
            if not (in_u):
                u_lst.append(1e5)
                u_err_list.append(1)
            if not (in_c):
                c_lst.append(1e5)
                c_err_list.append(1)
            if not (in_b):
                b_lst.append(1e5)
                b_err_list.append(1)
            if not (in_t):
                t_lst.append(1e5)
                t_err_list.append(1)
    if plot_name[-4:] == ".pdf":
        plot_name = plot_name[:-4]

    if xlabel is None:
        xlabel = variable
        if variable == "actualInteractionsPerCrossing":
            xlabel = "Actual interactions per bunch crossing"
        elif variable == "pt":
            xlabel = r"$p_T$ [GeV]"

    ThirdTag = (
        ThirdTag
        + "\n"
        + "$\\epsilon_b$ = {}%, $f_c$ = {}".format(efficiency, fc)
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
            y=t_lst,
            yerr=u_err_list,
            xerr=x_err,
            label=r"$\tau$-jets",
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
        pas.makeATLAStag(ax, fig, AtlasTag, SecondTag, xmin=0.57, ymax=0.88)
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
    fc_list=[],
    fc=0.018,
    Passed=True,
    Fixed_WP_Bin=False,
    bin_edges=[20, 50, 90, 150, 300, 1000],
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
    """
    # Get the bins for the histogram
    pt_midpts = (np.asarray(bin_edges)[:-1] + np.asarray(bin_edges)[1:]) / 2.0
    bin_widths = (np.asarray(bin_edges)[1:] - np.asarray(bin_edges)[:-1]) / 2.0
    Npts = pt_midpts.size

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
            # Calculate WP cutoff for b-disc
            disc_cut = np.percentile(
                df_results.query("labels==2")["discs"], (1 - WP) * 100
            )

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
                pT_mask = (jetPts > pt_min) & (jetPts < pt_max)

                disc_cut = np.percentile(
                    df_results.query("labels==2")["discs"][pT_mask],
                    (1 - WP) * 100,
                )

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

    if Fixed_WP_Bin is False:
        Fixed_WP_Label = "Inclusive"

    else:
        Fixed_WP_Label = ""

    if flavor == b_index:
        flav_label = r"$b$-jet"

    elif flavor == c_index:
        flav_label = r"$c$-jet"

    elif flavor == u_index:
        flav_label = "light-flavour jet"

    axis_dict["left"]["top"].set_ylabel(
        f"{Fixed_WP_Label} {flav_label}-{metric}",
        horizontalalignment="right",
        y=1.0,
    )

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

    axis_dict["left"]["top"].set_ylim(bottom=ymin, top=yAxisIncrease * ymax)

    # Set WP Line
    if WP_Line is True:
        xmin, xmax = axis_dict["left"]["top"].get_xlim()

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
    axis_dict["left"]["top"].legend(loc="upper right", ncol=2)

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
    legFontSize=10,
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
    if ylabel_right == "light":
        ylabel_right = r"Light-Flavour Jet Rejection ($1/\epsilon_{l}$)"
    elif ylabel_right == "c":
        ylabel_right = r"$c$-Jet Rejection ($1/\epsilon_{c}$)"

    if binomialErrors and nTest == 0:
        print(
            "Error: Requested binomialErrors, but did not pass nTest.",
            "Will NOT plot rej errors.",
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
            x, y, style, color=color, label=label, zorder=2
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
                x, np.ones(len(x)), style, color=color, linewidth=1.6
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
            x, ratio_ix, style, color=color, linewidth=1.6
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

    # Add axes, titels and the legend
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
    if ymax:
        axis_dict["left"]["top"].set_ylim(1, ymax)

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
        loc="best",
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
    df_results["discs"] = GetScore(
        *[df_results[pX] for pX in plot_config["prediction_labels"]]
    )

    # Clear the figure and init a new one
    plt.clf()
    fig = plt.figure()
    ax = fig.gca()

    # Define the length of b, c, and light
    len_b = len(df_results.query("labels==2"))
    len_c = len(df_results.query("labels==1"))
    len_u = len(df_results.query("labels==0"))

    # Calculate the hists and bin edges for errorbands
    counts_b, bins_b = np.histogram(
        df_results.query("labels==2")["discs"], bins=nBins
    )

    counts_c, bins_c = np.histogram(
        df_results.query("labels==1")["discs"],
        # Use the calculated binning to ensure its equal
        bins=bins_b,
    )

    counts_u, bins_u = np.histogram(
        df_results.query("labels==0")["discs"],
        # Use the calculated binning to ensure its equal
        bins=bins_b,
    )

    # Calculate the bin centers
    bincentres = [
        (bins_b[i] + bins_b[i + 1]) / 2.0 for i in range(len(bins_b) - 1)
    ]

    # Calculate poisson uncertainties and lower bands
    unc_b = np.sqrt(counts_b) / len_b
    band_lower_b = counts_b / len_b - unc_b

    unc_c = np.sqrt(counts_c) / len_c
    band_lower_c = counts_c / len_c - unc_c

    unc_u = np.sqrt(counts_u) / len_u
    band_lower_u = counts_u / len_u - unc_u

    # Hist the scores and their corresponding errors
    plt.hist(
        x=bins_b[:-1],
        bins=bins_b,
        weights=(counts_b / len_b),
        histtype="step",
        linewidth=2.0,
        color="C0",
        stacked=False,
        fill=False,
        label=r"$b$-jets",
    )

    plt.hist(
        x=bincentres,
        bins=bins_b,
        bottom=band_lower_b,
        weights=unc_b * 2,
        fill=False,
        hatch="/////",
        linewidth=0,
        edgecolor="#666666",
    )

    plt.hist(
        x=bins_c[:-1],
        bins=bins_c,
        weights=counts_c / len_c,
        histtype="step",
        linewidth=2.0,
        color="C1",
        stacked=False,
        fill=False,
        label=r"$c$-jets",
    )

    plt.hist(
        x=bincentres,
        bins=bins_c,
        bottom=band_lower_c,
        weights=unc_c * 2,
        fill=False,
        hatch="/////",
        linewidth=0,
        edgecolor="#666666",
    )

    plt.hist(
        x=bins_u[:-1],
        bins=bins_u,
        weights=counts_u / len_u,
        histtype="step",
        linewidth=2.0,
        color="C2",
        stacked=False,
        fill=False,
        label=r"light-flavour jets",
    )

    plt.hist(
        x=bincentres,
        bins=bins_u,
        bottom=band_lower_u,
        weights=unc_u * 2,
        fill=False,
        hatch="/////",
        linewidth=0,
        edgecolor="#666666",
        label="stat. unc.",
    )

    # Increase ymax so atlas tag don't cut plot
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)

    # Set WP vertical lines if given in config
    if WorkingPoints is not None:

        # Iterate over WPs
        for WP in WorkingPoints:

            # Calculate x value of WP line
            x_value = np.percentile(
                df_results.query("labels==2")["discs"], (1 - WP) * 100
            )

            # Draw WP line
            plt.vlines(
                x=x_value,
                ymin=ymin,
                ymax=WP * ymax,
                colors="C3",
                linestyles="dashed",
                linewidth=1.0,
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
    prediction_labels,
    model_labels,
    plot_name,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    tag="",
    WorkingPoints=None,
    nBins=50,
    figsize=None,
    labelpad=None,
    legFontSize=10,
    RatioType="Ratio",
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
    for df_results in df_list:
        df_results["discs"] = GetScore(
            *[df_results[pX] for pX in prediction_labels]
        )

    if type(which_axis) != list:
        which_axis = [which_axis] * len(df_list)

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

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
    if "right" in which_axis:
        axis_dict["right"] = {}
        axis_dict["right"]["top"] = axis_dict["left"]["top"].twinx()

    # Get binning for the plot
    _, Binning = np.histogram(
        df_list[0].query("labels==2")["discs"], bins=nBins
    )

    # Calculate the bin centers
    bincentres = [
        (Binning[i] + Binning[i + 1]) / 2.0 for i in range(len(Binning) - 1)
    ]

    # Init bincout dict for ratio calculation
    bincounts = {}

    linestyles = ["solid", "dashed"]
    for i, (df_results, linestyle, which_a) in enumerate(
        zip(df_list, linestyles, which_axis)
    ):
        # Define the length of b, c, and light
        len_b = len(df_results.query("labels==2"))
        len_c = len(df_results.query("labels==1"))
        len_u = len(df_results.query("labels==0"))

        # Calculate the hists and bin edges for errorbands
        counts_b, bins_b = np.histogram(
            df_results.query("labels==2")["discs"],
            # Use the calculated binning to ensure its equal
            bins=Binning,
        )

        counts_c, bins_c = np.histogram(
            df_results.query("labels==1")["discs"],
            # Use the calculated binning to ensure its equal
            bins=Binning,
        )

        counts_u, bins_u = np.histogram(
            df_results.query("labels==0")["discs"],
            # Use the calculated binning to ensure its equal
            bins=Binning,
        )

        # Calculate poisson uncertainties and lower bands
        unc_b = np.sqrt(counts_b) / len_b
        band_lower_b = counts_b / len_b - unc_b

        unc_c = np.sqrt(counts_c) / len_c
        band_lower_c = counts_c / len_c - unc_c

        unc_u = np.sqrt(counts_u) / len_u
        band_lower_u = counts_u / len_u - unc_u

        hist_counts_b, _, _ = axis_dict[which_a]["top"].hist(
            x=bins_b[:-1],
            bins=bins_b,
            weights=(counts_b / len_b),
            histtype="step",
            linewidth=2.0,
            linestyle=linestyle,
            color="C0",
            stacked=False,
            fill=False,
            label=r"$b$-jets {}".format(model_labels[i]),
        )

        axis_dict[which_a]["top"].hist(
            x=bincentres,
            bins=bins_b,
            bottom=band_lower_b,
            weights=unc_b * 2,
            fill=False,
            hatch="/////",
            linewidth=0,
            edgecolor="#666666",
        )

        hist_counts_c, _, _ = axis_dict[which_a]["top"].hist(
            x=bins_c[:-1],
            bins=bins_c,
            weights=counts_c / len_c,
            histtype="step",
            linewidth=2.0,
            linestyle=linestyle,
            color="C1",
            stacked=False,
            fill=False,
            label=r"$c$-jets {}".format(model_labels[i]),
        )

        axis_dict[which_a]["top"].hist(
            x=bincentres,
            bins=bins_c,
            bottom=band_lower_c,
            weights=unc_c * 2,
            fill=False,
            hatch="/////",
            linewidth=0,
            edgecolor="#666666",
        )

        hist_counts_u, _, _ = axis_dict[which_a]["top"].hist(
            x=bins_u[:-1],
            bins=bins_u,
            weights=counts_u / len_u,
            histtype="step",
            linewidth=2.0,
            linestyle=linestyle,
            color="C2",
            stacked=False,
            fill=False,
            label=r"light-flavour jets {}".format(model_labels[i]),
        )

        if i == 0:
            axis_dict[which_a]["top"].hist(
                x=bincentres,
                bins=bins_u,
                bottom=band_lower_u,
                weights=unc_u * 2,
                fill=False,
                hatch="/////",
                linewidth=0,
                edgecolor="#666666",
                label="stat. unc.",
            )

        else:
            axis_dict[which_a]["top"].hist(
                x=bincentres,
                bins=bins_u,
                bottom=band_lower_u,
                weights=unc_u * 2,
                fill=False,
                hatch="/////",
                linewidth=0,
                edgecolor="#666666",
            )

        bincounts.update({"b{}".format(i): hist_counts_b})
        bincounts.update({"c{}".format(i): hist_counts_c})
        bincounts.update({"u{}".format(i): hist_counts_u})

    # Start ratio plot
    for i, (flavor, color) in enumerate(zip(["b", "c", "u"], [0, 1, 2])):
        if RatioType == "Ratio":
            axis_dict["left"]["ratio"].step(
                x=Binning[:-1],
                y=np.divide(
                    bincounts["{}{}".format(flavor, 1)],
                    bincounts["{}{}".format(flavor, 0)],
                    out=np.ones(
                        bincounts["{}{}".format(flavor, 1)].shape, dtype=float
                    )
                    * bincounts["{}{}".format(flavor, 1)]
                    + 1,
                    where=(bincounts["{}{}".format(flavor, 0)] != 0),
                ),
                color="C{}".format(color),
            )

        elif RatioType == "Absolute":
            axis_dict["left"]["ratio"].step(
                x=Binning[:-1],
                y=bincounts["{}{}".format(flavor, 1)]
                - bincounts["{}{}".format(flavor, 0)],
                color="C{}".format(color),
            )

    # Add black line at one
    if RatioType == "Ratio":
        axis_dict["left"]["ratio"].axhline(
            y=1,
            xmin=axis_dict["left"]["ratio"].get_xlim()[0],
            xmax=axis_dict["left"]["ratio"].get_xlim()[1],
            color="black",
            alpha=0.5,
        )

    elif RatioType == "Absolute":
        axis_dict["left"]["ratio"].axhline(
            y=0,
            xmin=axis_dict["left"]["ratio"].get_xlim()[0],
            xmax=axis_dict["left"]["ratio"].get_xlim()[1],
            color="black",
            alpha=0.5,
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
    if RatioType == "Absolute":
        axis_dict["left"]["ratio"].set_ylabel(
            "{} - {}".format(model_labels[1], model_labels[0]),
            labelpad=labelpad,
            fontsize=12,
        )

    elif RatioType == "Ratio":
        axis_dict["left"]["ratio"].set_ylabel(
            "{} / {}".format(model_labels[1], model_labels[0]),
            labelpad=labelpad,
            fontsize=12,
        )

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    if xmin is not None:
        axis_dict["left"]["top"].set_xlim(left=xmin)
    if xmax is not None:
        axis_dict["left"]["top"].set_xlim(right=xmax)
    if ymin is not None:
        axis_dict["left"]["top"].set_ylim(left=ymin)
    if ymax is not None:
        axis_dict["left"]["top"].set_ylim(right=ymax)

    left_y_limits = axis_dict["left"]["top"].get_ylim()
    axis_dict["left"]["top"].set_ylim(
        left_y_limits[0], left_y_limits[1] * yAxisIncrease
    )

    axis_dict["left"]["top"].legend(
        loc="best",
        fontsize=legFontSize,
        ncol=2,
    )  # , title="DL1r")

    # Set WP vertical lines if given in config
    if WorkingPoints is not None:

        # Iterate over WPs
        for WP in WorkingPoints:

            # Calculate x value of WP line
            x_value = np.percentile(
                df_list[0].query("labels==2")["discs"], (1 - WP) * 100
            )

            # Draw WP line
            axis_dict["left"]["top"].axvline(
                x=x_value,
                ymin=0,
                ymax=0.75 * WP,
                color="C3",
                linestyle="dashed",
                linewidth=1.0,
            )

            # Draw WP line
            axis_dict["left"]["ratio"].axvline(
                x=x_value,
                ymin=0,
                ymax=1,
                color="C3",
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
