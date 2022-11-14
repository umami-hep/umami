"""Integrates shapeley package to rank feature importance in NN training."""
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib import colors


def shapley_one_flavour(
    model: object,
    test_data: np.ndarray,
    model_output: int = 2,
    feature_sets: int = 200,
    plot_size: tuple = (11, 11),
    plot_path: str = None,
    plot_name: str = "shapley_b-jets",
) -> None:
    """
    https://github.com/slundberg/shap

    Calculates shap values from shap package and plots results as beeswarm plot
    (Explainers are chosen automatically by shap depending on the feature size)

    model_output: is the output node of the model like:
    tau_index, b_index, c_index, u_index = 3, 2, 1, 0

    Parameters
    ----------
    model : Keras Model
        Loaded model which is to be evaluated.
    test_data : np.ndarray
        Array with the test data
    model_output : int, optional
        How many outputs the model has, by default 2
    feature_sets : int, optional
        How many whole sets of features to be calculated over.
        Corresponds to the number of dots per feature in the
        beeswarm plot , by default 200
    plot_size : tuple, optional
        Tuple with the plot size, by default (11, 11)
    plot_path : str, optional
        Path where the plot is aved, by default None
    plot_name : str, optional
        Name of the output file, by default "shapley_b-jets"
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


def shapley_all_flavours(
    model: object,
    test_data: np.ndarray,
    feature_sets: int = 200,
    averaged_sets: int = 50,
    plot_size: tuple = (11, 11),
    plot_path: str = None,
    plot_name: str = "shapley_all_flavors",
) -> None:
    """
    Makes a bar plot for the influence of features for all flavour outputs as
    categories in one plot.

    averaged_sets: let's you average over input features before
    they are handed to the shap framework to decrease runtime.

    Parameters
    ----------
    model : Keras Model
        Loaded model which is to be evaluated.
    test_data : np.ndarray
        Array with the test data
    feature_sets : int, optional
        How many whole sets of features to be calculated over.
        Corresponds to the number of dots per feature in the
        beeswarm plot , by default 200
    averaged_sets : int, optional
        Average sets, by default 50
    plot_size : tuple, optional
        Tuple with the plot size, by default (11, 11)
    plot_path : str, optional
        Path where the plot is aved, by default None
    plot_name : str, optional
        Name of the output file, by default "shapley_b-jets"
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
