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
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from yaml.loader import FullLoader

import umami.evaluation_tools as uet
from umami.configuration import logger


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

    parser.add_argument(
        "-p",
        "--print_plotnames",
        action="store_true",
        help="Print the model names of the plots to the terminal.",
    )

    args = parser.parse_args()
    return args


def plot_probability_comparison(
    plot_name, plot_config, eval_params, eval_file_dir, print_model
):
    # Init dataframe list
    df_list = []
    model_labels = []
    tagger_list = []
    class_labels_list = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info(f"model: {model_name}")

        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            df_results = pd.read_hdf(
                eval_file_dir + f"/results-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            df_results = pd.read_hdf(
                model_config["evaluation_file"], model_config["data_set_name"]
            )

        df_list.append(df_results)
        tagger_list.append(model_config["tagger_name"])
        model_labels.append(model_config["label"])
        class_labels_list.append(model_config["class_labels"])

    uet.plot_prob_comparison(
        df_list=df_list,
        model_labels=model_labels,
        tagger_list=tagger_list,
        class_labels_list=class_labels_list,
        flavour=plot_config["prob_class"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_ROC(plot_name, plot_config, eval_params, eval_file_dir, print_model):
    df_results_list = []
    tagger_list = []
    rej_class_list = []
    labels = []
    linestyles = []
    colors = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    if "nTest" not in plot_config["plot_settings"].keys():
        nTest_provided = False
        plot_config["plot_settings"]["nTest"] = []
    else:
        nTest_provided = True

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info(f"model: {model_name}")

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

        df_results_list.append(model_config["df_results_eff_rej"])
        tagger_list.append(model_config["tagger_name"])
        rej_class_list.append(model_config["rejection_class"])
        labels.append(model_config["label"])

        if "linestyle" in model_config:
            linestyles.append(model_config["linestyle"])

        if "color" in model_config:
            colors.append(model_config["color"])

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

    if len(colors) == 0:
        colors = None

    if len(linestyles) == 0:
        linestyles = None

    uet.plotROCRatio(
        df_results_list=df_results_list,
        tagger_list=tagger_list,
        rej_class_list=rej_class_list,
        labels=labels,
        plot_name=plot_name,
        styles=linestyles,
        colors=colors,
        **plot_config["plot_settings"],
    )


def plot_ROC_Comparison(
    plot_name, plot_config, eval_params, eval_file_dir, print_model
):
    df_results_list = []
    tagger_list = []
    rej_class_list = []
    labels = []
    linestyles = []
    colors = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    if "nTest" not in plot_config["plot_settings"].keys():
        nTest_provided = False
        plot_config["plot_settings"]["nTest"] = []
    else:
        nTest_provided = True

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info(f"model: {model_name}")

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

        df_results_list.append(model_config["df_results_eff_rej"])
        tagger_list.append(model_config["tagger_name"])
        rej_class_list.append(model_config["rejection_class"])
        labels.append(model_config["label"])

        if "linestyle" in model_config:
            linestyles.append(model_config["linestyle"])

        if "color" in model_config:
            colors.append(model_config["color"])

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

    # Get the right ratio id for correct ratio calculation
    ratio_dict = {}
    ratio_id = []

    for i, which_a in enumerate(rej_class_list):
        if which_a not in ratio_dict:
            ratio_dict.update({which_a: i})
            ratio_id.append(i)

        else:
            ratio_id.append(ratio_dict[which_a])

    uet.plotROCRatioComparison(
        df_results_list=df_results_list,
        tagger_list=tagger_list,
        rej_class_list=rej_class_list,
        labels=labels,
        plot_name=plot_name,
        ratio_id=ratio_id,
        linestyles=linestyles,
        colors=colors,
        **plot_config["plot_settings"],
    )


def plot_ROCvsVar(plot_name, plot_config, eval_params, eval_file_dir):
    """
    "flat_eff": bool whether to plot a flat b-efficiency as a function of var
    "efficiency": the targeted efficiency
    "variable": which variable to plot the efficiency as a function of.
    "max_variable": maximum value of the range of variable.
    "min_variable": minimum value of the range of variable.
    "nbin": number of bin to use
    """
    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])
    bool_use_taus = eval_params["bool_use_taus"]
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
    # Whether to fix the b-efficiency in each bin of the variable analysed
    flat_eff = False
    if "flat_eff" in plot_config and plot_config["flat_eff"] is True:
        flat_eff = True

    if (
        "variable" not in plot_config
        or plot_config["variable"] not in df_results
    ):
        logger.warning(
            "Forgot to specify a variable that is contained in the dataframe"
        )
        logger.warning("Defaulting to pT")
        plot_config["variable"] = "pt"

    if "prediction_labels" not in plot_config:
        logger.warning("Forgot to specify the prediction labels")

    fc = 0.018
    if "fc" in plot_config and plot_config["fc"] is not None:
        fc = plot_config["fc"]

    if "ptau" in plot_config["prediction_labels"] and bool_use_taus:
        bool_use_taus = True
        ftau = 1 - fc
        if "ftau" in plot_config and plot_config["ftau"] is not None:
            ftau = plot_config["ftau"]
    else:
        bool_use_taus = False
        if "ptau" in plot_config["prediction_labels"]:
            plot_config["prediction_labels"].remove("ptau")
        ftau = None

    # Compute the score
    df_results["bscore"] = uet.GetScore(
        *[df_results[pX] for pX in plot_config["prediction_labels"]],
        fc=fc,
        ftau=ftau,
    )

    max_given, min_given, nbin_given = False, False, False
    if "max_variable" in plot_config:
        max_given = True
    if "min_variable" in plot_config:
        min_given = True
    if "nbin" in plot_config:
        nbin_given = True

    xticksval = None
    xticks = None

    if plot_config["variable"] == "pt":
        maxval = 6000000
        minval = 10000
        nbin = 100
        xticksval = [10, 1.5e3, 3e3, 4.5e3, 6e3]
        xticks = [
            r"$10$",
            r"$1.5 \times 10^3$",
            r"$3 \times  10^3$",
            r"$4.5 \times 10^3$",
            r"$6 \times 10^3$",
        ]

    elif plot_config["variable"] == "eta":
        maxval = 2.5
        minval = 0
        nbin = 20

    elif plot_config["variable"] == "actualInteractionsPerCrossing":
        maxval = 81
        minval = 0
        nbin = 82

    else:  # No special range
        maxval = df_results[plot_config["variable"]].max()
        minval = df_results[plot_config["variable"]].min()
        nbin = 100

    if max_given:
        maxval = plot_config["max_variable"]
        if plot_config["variable"] == "pt":
            maxval = maxval * 1000
    if min_given:
        minval = plot_config["min_variable"]
        if plot_config["variable"] == "pt":
            minval = minval * 1000
    if nbin_given:
        nbin = plot_config["nbin"]

    var_bins = np.linspace(minval, maxval, nbin)

    if "var_bins" in plot_config:
        var_bins = np.asarray(plot_config["var_bins"])
        if plot_config["variable"] == "pt":
            var_bins = var_bins * 1000

    if flat_eff:
        df_results["btag"] = uet.FlatEfficiencyPerBin(
            df_results,
            "bscore",
            plot_config["variable"],
            var_bins,
            wp=plot_config["efficiency"] / 100,
        )
    else:
        cutvalue = np.percentile(
            df_results[df_results["labels"] == 2]["bscore"],
            100.0 * (1.0 - plot_config["efficiency"] / 100.0),
        )
        df_results["btag"] = (df_results["bscore"] > cutvalue) * 1

    if "xticksval" in plot_config:
        xticksval = plot_config["xticksval"]
    if "xticks" in plot_config:
        xticks = plot_config["xticks"]

    uet.plotEfficiencyVariable(
        plot_name=plot_name,
        df=df_results,
        variable=plot_config["variable"],
        var_bins=var_bins,
        include_taus=bool_use_taus,
        fc=fc,
        ftau=ftau,
        efficiency=plot_config["efficiency"],
        xticksval=xticksval,
        xticks=xticks,
        **plot_config["plot_settings"],
    )


def plot_confusion_matrix(plot_name, plot_config, eval_params, eval_file_dir):
    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])
    bool_use_taus = eval_params["bool_use_taus"]

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

    # Need to flip the prediciton labels list here.
    # b's are in y_target a 2 while with the given order in prediction labels
    # pb is a 0. So just switch it
    y_predicted = np.argmax(
        df_results[plot_config["prediction_labels"][::-1]].values, axis=1
    )
    cm = confusion_matrix(
        y_target=y_target, y_predicted=y_predicted, binary=False
    )
    if bool_use_taus:
        class_names = ["b", "c", "light", r"$\tau$"]
    else:
        class_names = ["b", "c", "light"]

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


def score_comparison(
    plot_name, plot_config, eval_params, eval_file_dir, print_model
):
    # Init dataframe list
    df_list = []
    tagger_list = []
    class_labels_list = []
    model_labels = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info(f"model: {model_name}")

        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            df_results = pd.read_hdf(
                eval_file_dir + f"/results-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            df_results = pd.read_hdf(
                model_config["evaluation_file"], model_config["data_set_name"]
            )

        df_list.append(df_results)
        model_labels.append(model_config["label"])
        tagger_list.append(model_config["tagger_name"])
        class_labels_list.append(model_config["class_labels"])

    uet.plot_score_comparison(
        df_list=df_list,
        model_labels=model_labels,
        tagger_list=tagger_list,
        class_labels_list=class_labels_list,
        main_class=plot_config["main_class"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_pT_vs_eff(
    plot_name, plot_config, eval_params, eval_file_dir, print_model
):
    # Init label and dataframe list
    df_list = []
    model_labels = []
    tagger_list = []
    SWP_label_list = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if model_name == "evaluation_file":
            continue

        if print_model:
            logger.info(f"model: {model_name}")

        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            df_results = pd.read_hdf(
                eval_file_dir + f"/results-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            df_results = pd.read_hdf(
                model_config["evaluation_file"],
                model_config["data_set_name"],
            )

        if (
            "SWP_label" in model_config
            and model_config["SWP_label"] is not None
        ):
            SWP_label_list.append(model_config["SWP_label"])

        df_list.append(df_results)
        model_labels.append(model_config["label"])
        tagger_list.append(model_config["tagger"])

    uet.plotPtDependence(
        df_list=df_list,
        tagger_list=tagger_list,
        model_labels=model_labels,
        plot_name=plot_name,
        SWP_label_list=SWP_label_list,
        **plot_config["plot_settings"],
    )


def plot_fraction_scan(plot_name, plot_config, eval_params, eval_file_dir):
    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    if ("evaluation_file" not in plot_config) or (
        plot_config["evaluation_file"] is None
    ):
        df_results = pd.read_hdf(
            eval_file_dir + f"/results-rej_per_frac-{eval_epoch}.h5",
            plot_config["data_set_name"],
        )
    else:
        df_results = pd.read_hdf(
            plot_config["evaluation_file"], plot_config["data_set_name"]
        )

    # default values
    y_values = "fraction_c"
    x_values = "fraction_taus"
    if "xlabel" in plot_config:
        x_values = plot_config["xlabel"]
    if "ylabel" in plot_config:
        y_values = plot_config["ylabel"]
    # labels.append(model_config["label"])

    uet.plotFractionScan(
        data=df_results,
        label=plot_config["label"],
        plot_name=plot_name,
        x_val=x_values,
        y_val=y_values,
        **plot_config["plot_settings"],
    )


def plot_score(plot_name, plot_config, eval_params, eval_file_dir):
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

    uet.plot_score(
        df_results=df_results,
        tagger_name=plot_config["tagger_name"],
        class_labels=plot_config["class_labels"],
        main_class=plot_config["main_class"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_prob(plot_name, plot_config, eval_params, eval_file_dir):
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

    uet.plot_prob(
        df_results=df_results,
        plot_name=plot_name,
        tagger_name=plot_config["tagger_name"],
        class_labels=plot_config["class_labels"],
        flavour=plot_config["prob_class"],
        **plot_config["plot_settings"],
    )


def plot_saliency(plot_name, plot_config, eval_params, eval_file_dir):
    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    # Read file, change to specific file if defined
    if ("evaluation_file" not in plot_config) or (
        plot_config["evaluation_file"] is None
    ):
        with open(
            eval_file_dir
            + f'/saliency_{eval_epoch}_{plot_config["data_set_name"]}.pkl',
            "rb",
        ) as f:
            maps_dict = pickle.load(f)

    else:
        with open(plot_config["evaluation_file"], "rb") as f:
            maps_dict = pickle.load(f)

    uet.plotSaliency(
        maps_dict=maps_dict,
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def SetUpPlots(
    plotting_config, plot_directory, eval_file_dir, format, print_model
):
    # Extract the eval parameters
    eval_params = plotting_config["Eval_parameters"]

    # Extract the print epoch bool
    if (
        "epoch_to_name" not in plotting_config["Eval_parameters"]
        or plotting_config["Eval_parameters"]["epoch_to_name"] is None
    ):
        epoch_to_name = True

    else:
        epoch_to_name = plotting_config["Eval_parameters"]["epoch_to_name"]

    # Iterate over the different plots which are to be plotted
    for plot_name, plot_config in tqdm(plotting_config.items()):

        # Skip Eval parameters
        if plot_name == "Eval_parameters":
            continue

        # Define the path to the new plot
        if print_model:
            logger.info(f"Processing: {plot_name}")

        if epoch_to_name is True:
            save_plot_to = os.path.join(
                plot_directory,
                plot_name
                + "_{}".format(int(eval_params["epoch"]))
                + "."
                + format,
            )

        else:
            save_plot_to = os.path.join(
                plot_directory,
                plot_name + "." + format,
            )

        # Check for plot type and use the needed function
        if plot_config["type"] == "ROC":
            plot_ROC(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                print_model=print_model,
            )

        elif plot_config["type"] == "ROC_Comparison":
            plot_ROC_Comparison(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                print_model=print_model,
            )

        elif plot_config["type"] == "confusion_matrix":
            plot_confusion_matrix(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "scores":
            plot_score(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "probability":
            plot_prob(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "saliency":
            plot_saliency(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "probability_comparison":
            plot_probability_comparison(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                print_model=print_model,
            )

        elif plot_config["type"] == "scores_comparison":
            score_comparison(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                print_model=print_model,
            )

        elif plot_config["type"] == "pT_vs_eff":
            plot_pT_vs_eff(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
                print_model=print_model,
            )

        elif plot_config["type"] == "ROCvsVar":
            plot_ROCvsVar(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        elif plot_config["type"] == "FracScan":
            plot_fraction_scan(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_dir,
            )

        else:
            raise NameError(
                "Plot type {} is not supported".format(plot_config["type"])
            )

        if print_model:
            logger.info(
                f'saved plot as: {save_plot_to.replace(eval_params["Path_to_models_dir"], "")} \n'
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
    SetUpPlots(
        plotting_config,
        plot_directory,
        eval_file_dir,
        args.format,
        args.print_plotnames,
    )


if __name__ == "__main__":
    args = GetParser()
    main(args)
