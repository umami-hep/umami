import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import pchip

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.tools.PyATLASstyle.PyATLASstyle import makeATLAStag


def eff_err(x, N):
    return np.sqrt(x * (1 - x) / N)


def GetScore(pb, pc, pu, fc=0.018):
    pb = pb.astype("float64")
    pc = pc.astype("float64")
    pu = pu.astype("float64")
    add_small = 1e-10
    return np.log((pb + add_small) / ((1.0 - fc) * pu + fc * pc + add_small))


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
        makeATLAStag(
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
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.tight_layout()
    plt.savefig(plot_name, transparent=True)
    plt.close()
