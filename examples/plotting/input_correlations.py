"""
    This plots a linear correlation matrix and scatterplots between
    all variables like (var_1, var_2) -> (x_xis, y_axis).

    It will execute on test data for illustration. Adjust filepath to the h5
    ntuple path and var_file to your yaml variable file.
"""

import os
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy import stats

import umami.data_tools as udt
from umami.configuration import logger
from umami.preprocessing_tools import get_variable_dict

# get test data
TESTDATA = (
    "https://umami-ci-provider.web.cern.ch/plot_input_vars/plot_input_vars_r22_check.h5"
)
TESTDIR = "/tmp/umami/plot_input_vars/"
TESTFILE = "plot_input_vars_r22_check.h5"
FILEEXISTS = os.path.isfile(TESTDIR + TESTFILE)
if not FILEEXISTS:
    os.makedirs(TESTDIR, exist_ok=True)
    run(["wget", TESTDATA, "--directory-prefix", TESTDIR], check=True)

TESTVAR = {
    "train_variables": {
        "some_vars": [
            "pt_btagJes",
            "JetFitter_mass",
            "JetFitter_energyFraction",
            "JetFitter_significance3d",
            "JetFitter_nVTX",
            "JetFitter_deltaR",
        ]
    }
}
with open(TESTDIR + "test_var.yaml", "w") as outfile:
    yaml.dump(TESTVAR, outfile, default_flow_style=False)

# adjust it to your h5 ntuple path and yaml variable file
FILEPATH = TESTDIR + TESTFILE
VARFILE = TESTDIR + "test_var.yaml"
NJETS = 1000

# load jets
class_labels = ["bjets", "cjets", "ujets"]
variable_config = get_variable_dict(VARFILE)
jetsVarsAll = [
    i
    for j in variable_config["train_variables"]
    for i in variable_config["train_variables"][j]
]

jetsAll, _ = udt.load_jets_from_file(
    filepath=FILEPATH,
    class_labels=class_labels,
    variables=jetsVarsAll,
    n_jets=NJETS,
    print_logger=False,
)


def correlation_matrix(jets, jet_vars, fig_size=(9, 11)) -> None:
    """
    Plots a Correlation Matrix

    Parameters
    ----------
    jets : pandas.DataFrame
        The jets as numpy ndarray
    jet_vars : list
        List of variables to plot
    fig_size : tuple(int, int)
        size of figure
    """

    logger.info("Plotting Correlation Matrix ...")

    jets = jets[jet_vars]
    corr = jets.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.subplots(figsize=fig_size)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=None,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")


def scatter_matrix(
    jets,
    jet_vars,
    std_outliers=5,
    show_contours=True,
    contour_level=4,
    del_upper_right_triangle=True,
) -> None:
    """
    Plots 2D scatter plots between all variables

    Parameters
    ----------
    jets : pandas.DataFrame
        The jets as numpy ndarray
    jet_vars : list
        List of variables to plot
    std_outliers : float
        outside of how many std's distance sort out outliers
    show_contours : bool
        Show contour lines on lower triangle (expensive)
    contour_level : int
        how many contour levels
    del_upper_right_triangle : bool
        if upper right triangle plots are plotted
    """

    # delete NaN
    jets = jets.dropna()

    # how many std's distance for sorting out outliers
    jets = jets[(np.abs(stats.zscore(jets)) < std_outliers).all(axis=1)]
    jets["Flavour"] = jets["Umami_labels"]

    # flavor strings
    jets.loc[jets["Flavour"] == 0, "Flavour"] = "b-jets"
    jets.loc[jets["Flavour"] == 1, "Flavour"] = "c-jets"
    jets.loc[jets["Flavour"] == 2, "Flavour"] = "u-jets"

    logger.info("Plotting Scatter Matrix ... ")
    logger.info("This can take a while depending on the amount of variables and jets.")

    jet_vars.append("Flavour")
    jets_for_plot = jets[jet_vars]

    # seaborn plot
    sns.set_theme(style="ticks")
    #   b: "#1f77b4"
    #   c: "#ff7f0e"
    #   u: "#2ca02c"
    graph = sns.pairplot(
        jets_for_plot,
        hue="Flavour",
        palette=[
            "#1f77b4",
            "#2ca02c",
            "#ff7f0e",
        ],
        corner=del_upper_right_triangle,
        height=3,
    )

    if show_contours is True:
        graph.map_lower(sns.kdeplot, levels=contour_level, fill=True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("scatterplot_matrix.png")


correlation_matrix(jetsAll, jetsVarsAll)
scatter_matrix(jetsAll, jetsVarsAll, std_outliers=5)
