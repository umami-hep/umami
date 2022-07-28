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
test_data = (
    "https://umami-ci-provider.web.cern.ch/plot_input_vars/plot_input_vars_r22_check.h5"
)
test_dir = "/tmp/umami/plot_input_vars/"
test_file = "plot_input_vars_r22_check.h5"
file_exists = os.path.isfile(test_dir + test_file)
if not file_exists:
    os.makedirs(test_dir, exist_ok=True)
    run(["wget", test_data, "--directory-prefix", test_dir], check=True)

test_var = {
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
with open(test_dir + "test_var.yaml", "w") as outfile:
    yaml.dump(test_var, outfile, default_flow_style=False)

# adjust it to your h5 ntuple path and yaml variable file
filepath = test_dir + test_file
var_file = test_dir + "test_var.yaml"
n_jets = 1000

# load jets
class_labels = ["bjets", "cjets", "ujets"]
variable_config = get_variable_dict(var_file)
jetsVarsAll = [
    i
    for j in variable_config["train_variables"]
    for i in variable_config["train_variables"][j]
]

jetsAll, _ = udt.LoadJetsFromFile(
    filepath=filepath,
    class_labels=class_labels,
    variables=jetsVarsAll,
    n_jets=n_jets,
    print_logger=False,
)


def CorrelationMatrix(jets, jetsVars, fig_size=(9, 11)) -> None:
    """
    Plots a Correlation Matrix

    Parameters
    ----------
    jets : pandas.DataFrame
        The jets as numpy ndarray
    jetsVars : list
        List of variables to plot
    fig_size : tuple(int, int)
        size of figure
    """

    logger.info("Plotting Correlation Matrix ...")

    jets = jets[jetsVars]
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


def ScatterMatrix(
    jets,
    jetsVars,
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
    jetsVars : list
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

    # drop string coloumn
    jets = jets.drop(["Umami_string_labels"], axis=1)

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

    jetsVars.append("Flavour")
    jets_for_plot = jets[jetsVars]

    # seaborn plot
    sns.set_theme(style="ticks")
    #   b: "#1f77b4"
    #   c: "#ff7f0e"
    #   u: "#2ca02c"
    g = sns.pairplot(
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
        g.map_lower(sns.kdeplot, levels=contour_level, fill=True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("scatterplot_matrix.png")


CorrelationMatrix(jetsAll, jetsVarsAll)
ScatterMatrix(jetsAll, jetsVarsAll, std_outliers=5)
