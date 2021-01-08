#!/usr/bin/env python

"""
This script allows to plot the ROC curves (and ratios to other models),
the confusion matrix and the output scores (pb, pc, pu).
A configuration file has to be provided.
See umami/examples/plotting_umami_config*.yaml for examples.
This script works on the output of the evaluate_model.py script
and has to be specified in the config file as 'evaluation_file'.
"""

import h5py
import argparse
import os
import numpy as np
import pandas as pd
import yaml
from yaml.loader import FullLoader
import umami.evaluation_tools as uet
import matplotlib.pyplot as plt
from umami.evaluate_model import GetScore
from umami.tools.PyATLASstyle.PyATLASstyle import makeATLAStag


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Plotting command line options."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Name of the plot config file",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="pdf",
        help="file extension for the plots",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="umami_plotting_plots",
        help="Name of the directory in which the plots will be saved.",
    )
    args = parser.parse_args()
    return args


def plot_ROC(plot_name, plot_config):
    teffs = []
    beffs = []
    labels = []
    if "nTest" not in plot_config["plot_settings"].keys():
        nTest_provided = False
        plot_config["plot_settings"]["nTest"] = []
    else:
        nTest_provided = True

    for model_name, model_config in plot_config["models_to_plot"].items():
        print("model", model_name)
        model_config["df_results_eff_rej"] = pd.read_hdf(
            model_config["evaluation_file"], model_config["data_set_name"]
        )
        model_config["rej_rates"] = (
            1.0 / model_config["df_results_eff_rej"][model_config["df_key"]]
        )

        teffs.append(model_config["df_results_eff_rej"]["beff"][:150])
        beffs.append(model_config["rej_rates"])
        labels.append(model_config["label"])

        # nTest is only needed to calculate binomial errors
        if not nTest_provided and (
            "binomialErrors" in plot_config["plot_settings"]
            and plot_config["plot_settings"]["binomialErrors"]
        ):
            h5_file = h5py.File(model_config["evaluation_file"], "r")
            plot_config["plot_settings"]["nTest"].append(
                h5_file.attrs["N_test"]
            )
            h5_file.close()

    uet.plotROCRatio(
        teffs=teffs,
        beffs=beffs,
        labels=labels,
        plot_name=plot_name,
        **plot_config["plot_settings"]
    )


def plot_confusion_matrix(plot_name, plot_config):
    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm

    df_results = pd.read_hdf(
        plot_config["evaluation_file"], plot_config["data_set_name"]
    )
    y_target = df_results["labels"]
    y_predicted = np.argmax(
        df_results[plot_config["prediction_labels"]].values, axis=1
    )
    cm = confusion_matrix(
        y_target=y_target, y_predicted=y_predicted, binary=False
    )
    class_names = ["light", "c", "b"]
    mlxtend_plot_cm(
        conf_mat=cm,
        colorbar=True,
        show_absolute=False,
        show_normed=True,
        class_names=class_names,
    )
    plt.tight_layout()
    plt.savefig(plot_name, transparent=True)


def plot_score(plot_name, plot_config):
    df_results = pd.read_hdf(
        plot_config["evaluation_file"], plot_config["data_set_name"]
    )
    df_results["discs"] = GetScore(
        *[df_results[pX] for pX in plot_config["prediction_labels"]]
    )
    plt.clf()
    plt.hist(
        [
            df_results.query("labels==2")["discs"],
            df_results.query("labels==1")["discs"],
            df_results.query("labels==0")["discs"],
        ],
        50,
        histtype="step",
        stacked=False,
        fill=False,
        density=1,
        label=["b-jets", "c-jets", "l-jets"],
    )
    plt.legend()
    plt.xlabel(u"$D_{b}$")
    text = ''
    if "text" in plot_config:
        text = plot_config["text"]
    makeATLAStag(plt.gca(), plt.gcf(), "Internal Simulation", text)
    plt.tight_layout()
    plt.savefig(plot_name, transparent=True)


def SetUpPlots(plotting_config, plot_directory, format):
    for plot_name, plot_config in plotting_config.items():
        print("Processing:", plot_name)
        save_plot_to = os.path.join(plot_directory, plot_name + "." + format)

        if plot_config["type"] == "ROC":
            plot_ROC(save_plot_to, plot_config)
        elif plot_config["type"] == "confusion_matrix":
            plot_confusion_matrix(save_plot_to, plot_config)
        elif plot_config["type"] == "scores":
            plot_score(save_plot_to, plot_config)

        print("saved plot as:", save_plot_to)


def main(args):
    with open(args.config_file) as yaml_config:
        plotting_config = yaml.load(yaml_config, Loader=FullLoader)
    plot_directory = os.path.join(os.getcwd(), args.output_directory)
    os.makedirs(plot_directory, exist_ok=True)
    SetUpPlots(plotting_config, plot_directory, args.format)


if __name__ == "__main__":
    args = GetParser()
    main(args)
