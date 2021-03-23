import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from umami.tools.PyATLASstyle.PyATLASstyle import makeATLAStag


def plot_dumper_evaluation(
    plot_name,
    plot_config,
    eval_params,
    eval_file_dir,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    x_label=r"$|p_{light}^{LWTNN} - p_{light}^{DIPS}|$",
    y_scale=True,
    nBins=50,
    yAxisIncrease=100,
    yAxisAtlasTag=0.9,
):
    # Read file, change to specific file if defined
    if ("evaluation_file" not in plot_config) or (
        plot_config["evaluation_file"] is None
    ):
        df_results = pd.read_csv(eval_file_dir + "/dumper_val.csv")

    else:
        df_results = pd.read_csv(plot_config["evaluation_file"])

    # Clear the figure and init a new one
    plt.clf()

    # Define the length of b, c, and light
    len_diff = len(df_results["diff"])

    # Calculate the hists and bin edges for errorbands
    counts_diff, bins_diff = np.histogram(df_results["diff"], bins=nBins)

    # Calculate the bin centers
    bincentres = [
        (bins_diff[i] + bins_diff[i + 1]) / 2.0
        for i in range(len(bins_diff) - 1)
    ]

    # Calculate poisson uncertainties and lower bands
    unc_diff = np.sqrt(counts_diff) / len_diff
    band_lower_diff = counts_diff / len_diff - unc_diff

    # Hist the scores and their corresponding errors
    plt.hist(
        x=bins_diff[:-1],
        bins=bins_diff,
        weights=(counts_diff / len_diff),
        histtype="step",
        linewidth=2.0,
        color="C0",
        stacked=False,
        fill=False,
        label=r"$b$-jets",
    )

    plt.hist(
        x=bincentres,
        bins=bins_diff,
        bottom=band_lower_diff,
        weights=unc_diff * 2,
        fill=False,
        hatch="/////",
        linewidth=0,
        edgecolor="#666666",
    )

    # Set yscale
    if y_scale is True:
        plt.yscale("log")

    # Increase ymax so atlas tag don't cut plot
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)

    # plt.legend()
    plt.xlabel(x_label)
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
