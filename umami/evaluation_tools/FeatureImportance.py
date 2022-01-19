"""Integrates shapeley package to rank feature importance in NN training."""
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib import colors


def ShapleyOneFlavor(
    model,
    test_data,
    model_output=2,
    feature_sets=200,
    plot_size=(11, 11),
    plot_path=None,
    plot_name="shapley_b-jets",
):
    """
    https://github.com/slundberg/shap

    Calculates shap values from shap package and plots results as beeswarm plot
    (Explainers are chosen automatically by shap depending on the feature size)

    model_output: is the output node of the model like:
    tau_index, b_index, c_index, u_index = 3, 2, 1, 0

    feature_sets: how many whole sets of features to be calculated over.
    Corresponds to the number of dots per feature in the beeswarm plot

    plot_size: (11,11) works well for DL1r

    """

    explainer = shap.Explainer(model.predict, masker=test_data.values)
    shap_values = explainer(test_data.values[:feature_sets, :])

    # From ReDevVerse comments https://github.com/slundberg/shap/issues/1460
    # model_output in np.take takes the according flavor
    # max_display defines how many features will be shown in the plot
    shap.summary_plot(
        shap_values=np.take(shap_values.values, model_output, axis=-1),
        features=test_data.values[:feature_sets, :],
        feature_names=list(test_data.keys()),
        plot_size=plot_size,
        max_display=100,
    )
    plt.tight_layout()
    if not os.path.exists(os.path.abspath(plot_path + "/plots")):
        os.makedirs(os.path.abspath(plot_path + "/plots"))
    plt.savefig(plot_path + "plots/" + plot_name + ".pdf")
    plt.close("all")


def ShapleyAllFlavors(
    model,
    test_data,
    feature_sets=200,
    averaged_sets=50,
    plot_size=(11, 11),
    plot_path=None,
    plot_name="shapley_all_flavors",
):

    """
    Makes a bar plot for the influence of features for all flavour outputs as
    categories in one plot.

    averaged_sets: let's you average over input features before
    they are handed to the shap framework to decrease runtime.

    """

    # it is just calculating mean values, not an actual kmeans algorithm
    averaged_data = shap.kmeans(test_data.values[:feature_sets, :], averaged_sets)
    explainer = shap.KernelExplainer(model.predict, data=averaged_data)
    shap_values = explainer.shap_values(test_data.values[:feature_sets, :])

    #   b: "#1f77b4"
    #   c: "#ff7f0e"
    #   u: "#2ca02c"
    # make colors for flavor outputs
    jet_cmap = colors.ListedColormap(["#2ca02c", "#ff7f0e", "#1f77b4"])
    # class_inds="original" gives you the right label order
    # max_display: defines how many features will be shown in the plot
    # class_names: plot labels

    shap.summary_plot(
        shap_values=shap_values,
        features=test_data,
        feature_names=list(test_data.keys()),
        class_names=["u-jets", "c-jets", "b-jets"],
        class_inds="original",
        plot_type="bar",
        color=jet_cmap,
        plot_size=plot_size,
        max_display=100,
    )

    plt.tight_layout()
    plt.savefig(plot_path + "/plots/" + plot_name + ".pdf")
    plt.close("all")
