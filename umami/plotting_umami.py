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
import pandas as pd
import yaml
from tqdm import tqdm
from yaml.loader import FullLoader

import umami.plotting_tools as uet
from umami.configuration import logger, set_log_level


def get_parser():
    """
    Argument parser for Preprocessing script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Plotting command line options.")

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

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
    )

    return parser.parse_args()


def plot_probability(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
    print_model: bool,
) -> None:
    """
    Plots probability comparison.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    print_model : bool
        Print the models which are plotted while plotting.
    """
    # Init dataframe list
    df_list = []
    model_labels = []
    tagger_list = []
    class_labels_list = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info("model: %s", model_name)

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
        tagger_list.append(model_config["tagger_name"].casefold())
        model_labels.append(model_config["label"])
        class_labels_list.append(model_config["class_labels"])

    uet.plot_prob(
        df_list=df_list,
        model_labels=model_labels,
        tagger_list=tagger_list,
        class_labels_list=class_labels_list,
        flavour=plot_config["prob_class"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_roc(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
    print_model: bool,
) -> None:
    """
    Plot ROCs.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    print_model : bool
        Print the models which are plotted while plotting.

    Raises
    ------
    AttributeError
        If the needed n_jets per class used to calculate the
        rejections is not in the rej_per_epoch results file.
    """
    df_results_list = []
    tagger_list = []
    rej_class_list = []
    labels = []
    linestyles = []
    colours = []
    n_jets_test = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info("model: %s", model_name)

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
        tagger_list.append(model_config["tagger_name"].casefold())
        rej_class_list.append(model_config["rejection_class"])
        labels.append(model_config["label"])

        if "linestyle" in model_config:
            linestyles.append(model_config["linestyle"])

        if "colour" in model_config:
            colours.append(model_config["colour"])

        # n_test is only needed to calculate binomial errors
        if (
            "draw_errors" in plot_config["plot_settings"]
            and plot_config["plot_settings"]["draw_errors"] is False
        ):
            n_jets_test.append(None)

        else:
            try:
                with h5py.File(
                    eval_file_dir + f"/results-rej_per_eff-{eval_epoch}.h5", "r"
                ) as h5_file:
                    n_jets_test.append(
                        h5_file.attrs[f"n_jets_{model_config['rejection_class']}"]
                    )

            except KeyError as error:
                raise AttributeError(
                    "You set draw_errors to True but the needed number of jets per "
                    "class used to calculate the rejections is not in the results "
                    "file! Please re-run the evaluation to add this values correctly!"
                ) from error

    # Get the right ratio id for correct ratio calculation
    ratio_dict = {}
    ratio_id = []

    if len(colours) == 0:
        colours = None

    for i, which_a in enumerate(rej_class_list):
        if which_a not in ratio_dict:
            ratio_dict.update({which_a: i})
            ratio_id.append(i)

        else:
            ratio_id.append(ratio_dict[which_a])

    uet.plot_roc(
        df_results_list=df_results_list,
        tagger_list=tagger_list,
        rej_class_list=rej_class_list,
        labels=labels,
        plot_name=plot_name,
        ratio_id=ratio_id,
        linestyles=linestyles,
        colours=colours,
        n_test=n_jets_test,
        **plot_config["plot_settings"],
    )


def plot_confusion_matrix(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    """
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

    uet.plot_confusion_matrix(
        df_results=df_results,
        tagger_name=plot_config["tagger_name"].casefold(),
        class_labels=plot_config["class_labels"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_score(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
    print_model: bool,
) -> None:
    """
    Plot score comparison.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    print_model : bool
        Print the models which are plotted while plotting.
    """
    # Init dataframe list
    df_list = []
    tagger_list = []
    class_labels_list = []
    model_labels = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info("model: %s", model_name)

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
        tagger_list.append(model_config["tagger_name"].casefold())
        class_labels_list.append(model_config["class_labels"])

    uet.plot_score(
        df_list=df_list,
        model_labels=model_labels,
        tagger_list=tagger_list,
        class_labels_list=class_labels_list,
        main_class=plot_config["main_class"],
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_pt_vs_eff(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
    print_model: bool,
) -> None:
    """
    Plot pT vs efficiency.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    print_model : bool
        Print the models which are plotted while plotting.
    """
    # Init label and dataframe list
    df_list = []
    model_labels = []
    tagger_list = []

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    for model_name, model_config in plot_config["models_to_plot"].items():
        if model_name == "evaluation_file":
            continue

        if print_model:
            logger.info("model: %s", model_name)

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

        df_list.append(df_results)
        model_labels.append(model_config["label"])
        tagger_list.append(model_config["tagger_name"].casefold())

    uet.plot_pt_dependence(
        df_list=df_list,
        tagger_list=tagger_list,
        model_labels=model_labels,
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def plot_saliency(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
) -> None:
    """
    Plot saliency maps.

    Parameters
    ----------
    plot_name : str
        Full path of the plot.
    plot_config : dict
        Dict with the plot configs.
    eval_params : dict
        Dict with the evaluation parameters.
    eval_file_dir : str
        Path to the results directory of the model.
    """

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
        ) as pkl_file:
            maps_dict = pickle.load(pkl_file)

    else:
        with open(plot_config["evaluation_file"], "rb") as pkl_file:
            maps_dict = pickle.load(pkl_file)

    uet.plot_saliency(
        maps_dict=maps_dict,
        plot_name=plot_name,
        target_eff=plot_config["target_eff"],
        jet_flavour=plot_config["jet_flavour"],
        PassBool=plot_config["PassBool"],
        nFixedTrks=plot_config["nFixedTrks"],
        **plot_config["plot_settings"],
    )


def plot_frac_contour(
    plot_name: str,
    plot_config: dict,
    eval_params: dict,
    eval_file_dir: str,
    print_model: bool,
) -> None:
    """Plot the fraction contour plot.

    Parameters
    ----------
    plot_name : str
        Full path + name of the plot
    plot_config : dict
        Loaded plotting config as dict.
    eval_params : dict
        Evaluation parameters from the plotting config.
    eval_file_dir : str
        File which is to use for plotting.
    print_model : bool
        Print the logger while plotting.
    """

    # Get the epoch which is to be evaluated
    eval_epoch = int(eval_params["epoch"])

    df_results_list = []
    tagger_list = []
    labels = []
    linestyles = []
    colours = []
    fixed_rejections = []
    marker_list = []

    for model_name, model_config in plot_config["models_to_plot"].items():
        if print_model:
            logger.info("model: %s", model_name)

        if ("evaluation_file" not in model_config) or (
            model_config["evaluation_file"] is None
        ):
            model_config["df_results_frac_rej"] = pd.read_hdf(
                eval_file_dir + f"/results-rej_per_fractions-{eval_epoch}.h5",
                model_config["data_set_name"],
            )

        else:
            model_config["df_results_frac_rej"] = pd.read_hdf(
                model_config["evaluation_file"], model_config["data_set_name"]
            )

        if "fixed_rejections" not in model_config:
            fixed_rejections.append(None)

        else:
            fixed_rejections.append(model_config["fixed_rejections"])

        if "marker" not in model_config:
            marker_list.append(None)

        else:
            marker_list.append(model_config["marker"])

        df_results_list.append(model_config["df_results_frac_rej"])
        tagger_list.append(model_config["tagger_name"].casefold())
        labels.append(model_config["label"])
        linestyles.append(model_config["linestyle"])
        colours.append(model_config["colour"])

    uet.plot_fraction_contour(
        df_results_list=df_results_list,
        tagger_list=tagger_list,
        label_list=labels,
        colour_list=colours,
        linestyle_list=linestyles,
        rejections_to_fix_list=fixed_rejections,
        rejections_to_plot=plot_config["rejections"],
        marked_points_list=marker_list,
        plot_name=plot_name,
        **plot_config["plot_settings"],
    )


def set_up_plots(
    plot_config_dict: dict,
    plot_dir: str,
    eval_file_path: str,
    file_format: str,
    print_model: bool,
) -> None:
    """Setting up plot settings.

    Parameters
    ----------
    plot_config_dict : dict
        Dict with the plot settings.
    plot_dir : str
        Path to the output directory of the plots.
    eval_file_path : str
        Path to the directory where the result files are saved.
    file_format : str
        String of the file format.
    print_model : bool
        Print the logger while plotting.

    Raises
    ------
    NameError
        If given plottype is not supported.
    """

    # Extract the eval parameters
    eval_params = plot_config_dict["Eval_parameters"]

    # Extract the print epoch bool
    if (
        "epoch_to_name" not in plot_config_dict["Eval_parameters"]
        or plot_config_dict["Eval_parameters"]["epoch_to_name"] is None
    ):
        epoch_to_name = True

    else:
        epoch_to_name = plot_config_dict["Eval_parameters"]["epoch_to_name"]

    # Iterate over the different plots which are to be plotted
    for plot_name, plot_config in tqdm(plot_config_dict.items()):

        # Skip general "Eval_parameters" and yaml anchors (have to start with a dot)
        if plot_name == "Eval_parameters" or plot_name.startswith("."):
            continue

        # Define the path to the new plot
        if print_model:
            logger.info("Processing: %s", plot_name)

        if epoch_to_name is True:
            save_plot_to = os.path.join(
                plot_dir,
                f"{plot_name}_{int(eval_params['epoch'])}.{file_format}",
            )

        else:
            save_plot_to = os.path.join(
                plot_dir,
                f"{plot_name}.{file_format}",
            )

        # Check for plot type and use the needed function
        if plot_config["type"] == "Frac_Contour":
            plot_frac_contour(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
                print_model=print_model,
            )

        elif plot_config["type"] == "ROC":
            plot_roc(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
                print_model=print_model,
            )

        elif plot_config["type"] == "confusion_matrix":
            plot_confusion_matrix(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
            )

        elif plot_config["type"] == "scores":
            plot_score(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
                print_model=print_model,
            )

        elif plot_config["type"] == "saliency":
            plot_saliency(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
            )

        elif plot_config["type"] == "probability":
            plot_probability(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
                print_model=print_model,
            )

        elif plot_config["type"] == "pT_vs_eff":
            plot_pt_vs_eff(
                plot_name=save_plot_to,
                plot_config=plot_config,
                eval_params=eval_params,
                eval_file_dir=eval_file_path,
                print_model=print_model,
            )

        else:
            raise NameError(f"Plot type {plot_config['type']} is not supported")

        if print_model:
            logger.info(
                "saved plot as: %s\n",
                save_plot_to.replace(eval_params["Path_to_models_dir"], ""),
            )


if __name__ == "__main__":
    args = get_parser()

    # Set logger level
    if args.verbose:
        set_log_level(logger, "DEBUG")

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
    eval_file_directory = os.path.join(
        plotting_config["Eval_parameters"]["Path_to_models_dir"],
        plotting_config["Eval_parameters"]["model_name"],
        "results",
    )

    # Start plotting
    set_up_plots(
        plot_config_dict=plotting_config,
        plot_dir=plot_directory,
        eval_file_path=eval_file_directory,
        file_format=args.format,
        print_model=args.print_plotnames,
    )
