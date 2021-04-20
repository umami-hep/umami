#!/usr/bin/env python

"""
This script allows to plot the ROC curves (and ratios to other models),
the confusion matrix and the output scores (pb, pc, pu).
A configuration file has to be provided.
See umami/examples/plotting_umami_config*.yaml for examples.
This script works on the output of the evaluate_model.py script
and has to be specified in the config file as 'evaluation_file'.
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import FullLoader

import umami.evaluation_tools as uet
import umami.validation_dumper_tools as uvdt


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


def plot_ROC(plot_name, plot_config, eval_params, eval_file_dir):
    teffs = []
    beffs = []
    labels = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    if "nTest" not in plot_config["plot_settings"].keys():
        nTest_provided = False
        plot_config["plot_settings"]["nTest"] = []
    else:
        nTest_provided = True

    for model_name, model_config in plot_config["models_to_plot"].items():
        print("model", model_name)
        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            model_config["df_results_eff_rej"] = pd.read_hdf(
                eval_file_dir + f"/results-rej_per_eff-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            model_config["df_results_eff_rej"] = pd.read_hdf(
                model_config["evaluation_file"], model_config["data_set_name"]
            )

        model_config["rej_rates"] = (
            1.0 / model_config["df_results_eff_rej"][model_config["df_key"]]
        )

        x_values = "beff"
        if "x_values_key" in model_config:
            x_values = model_config["x_values_key"]
        teffs.append(model_config["df_results_eff_rej"][x_values])
        beffs.append(model_config["rej_rates"])
        labels.append(model_config["label"])

        # nTest is only needed to calculate binomial errors
        if not nTest_provided and (
            "binomialErrors" in plot_config["plot_settings"]
            and plot_config["plot_settings"]["binomialErrors"]
        ):
            h5_file = h5py.File(
                eval_file_dir + f"/results-rej_per_eff-{eval_epoch}.h5", "r"
            )
            plot_config["plot_settings"]["nTest"].append(
                h5_file.attrs["N_test"]
            )
            h5_file.close()
        else:
            plot_config["plot_settings"]["nTest"] = 0

    uet.plotROCRatio(
        teffs=teffs,
        beffs=beffs,
        labels=labels,
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_confusion_matrix(plot_name, plot_config, eval_params, eval_file_dir):
    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

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
    plt.close()


def score_comparison(plot_name, plot_config, eval_params, eval_file_dir):
    # Init dataframe list
    df_list = []
    model_labels = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        print("model", model_name)
        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            df_results = pd.read_hdf(
                eval_file_dir + f"/results-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            df_results = pd.read_hdf(
                plot_config["evaluation_file"], plot_config["data_set_name"]
            )

        df_list.append(df_results)
        model_labels.append(model_config["label"])

    if len(df_list) > 2:
        raise ValueError(
            "Too many models for comparison plot! Only 2 are allowed"
        )

    else:
        uet.plot_score_comparison(
            df_list=df_list,
            prediction_labels=plot_config["prediction_labels"],
            model_labels=model_labels,
            plot_name=plot_name,
            **plot_config["plot_settings"],
        )


def SetUpPlots(plotting_config, plot_directory, eval_file_dir, format):
    # Extract the eval parameters
    eval_params = plotting_config["Eval_parameters"]

    # Iterate over the different plots which are to be plotted
    for plot_name, plot_config in plotting_config.items():

        # Skip Eval parameters
        if plot_name == "Eval_parameters":
            continue

        # Define the path to the new plot
        print("Processing:", plot_name)
        save_plot_to = os.path.join(
            plot_directory,
            plot_name + "_{}".format(int(eval_params["epoch"])) + "." + format,
        )

        # Check for plot type and use the needed function
        if plot_config["type"] == "ROC":
            plot_ROC(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "confusion_matrix":
            plot_confusion_matrix(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "scores":
            uet.plot_score(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                **plot_config["plot_settings"],
            )

        elif plot_config["type"] == "saliency":
            uet.plotSaliency(
                plot_name=save_plot_to,
                FileDir=eval_file_dir,
                epoch=int(eval_params["epoch"]),
                **plot_config["plot_settings"],
            )

        elif plot_config["type"] == "dumper_validation":
            uvdt.plot_dumper_evaluation(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                **plot_config["plot_settings"],
            )

        elif plot_config["type"] == "scores_comparison":
            score_comparison(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )
        else:
            raise NameError(
                "Plot type {} is not supported".format(plot_config["type"])
            )

        print(
            "saved plot as:",
            save_plot_to.replace(eval_params["Path_to_models_dir"], ""),
            "\n",
        )


def main(args):
    # Open and load the config files used in the eval process
    with open(args.config_file) as yaml_config:
        plotting_config = yaml.load(yaml_config, Loader=FullLoader)

    # Define the output dir and make it
    plot_directory = os.path.join(
        plotting_config["Eval_parameters"]["Path_to_models_dir"],
        plotting_config["Eval_parameters"]["model_name"],
        args.output_directory,
    )

    os.makedirs(plot_directory, exist_ok=True)

    # Define the path to the results from the model defined in train_config
    eval_file_dir = os.path.join(
        plotting_config["Eval_parameters"]["Path_to_models_dir"],
        plotting_config["Eval_parameters"]["model_name"],
        "results",
    )

    # Start plotting
    SetUpPlots(plotting_config, plot_directory, eval_file_dir, args.format)


if __name__ == "__main__":
    args = GetParser()
    main(args)
