#!/usr/bin/env python
"""
This script plots the given input variables of the given files and
also a comparison.
"""

import argparse

import yaml

import umami.input_vars_tools as uit
from umami.configuration import logger
from umami.plotting_tools.utils import translate_kwargs
from umami.tools import yaml_loader


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
        help="Name of the plotting config file",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        help="File type",
        default="pdf",
    )

    parser.add_argument(
        "--tracks",
        action="store_true",
        help="Plot track variables.",
    )

    parser.add_argument(
        "--jets",
        action="store_true",
        help="Plot jet variables.",
    )
    return parser.parse_args()


def plot_trks_variables(plot_config, plot_type):
    """Plot track variables.

    Parameters
    ----------
    plot_config : object
        plot configuration
    plot_type : str
        Plottype, like pdf or png
    """
    plot_config["Eval_parameters"] = translate_kwargs(plot_config["Eval_parameters"])
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        # Skip general "Eval_parameters" and yaml anchors (have to start with a dot)
        if plot_name == "Eval_parameters" or plot_name.startswith("."):
            continue

        if plotting_config["variables"] == "jets":
            continue

        logger.info(f"Start {plot_name}...\n")
        plotting_config["plot_settings"] = translate_kwargs(
            plotting_config["plot_settings"]
        )
        filepath_list = []
        labels_list = []
        tracks_list = []

        # Default to no selection based on track_origin
        trk_origins = ["All"]

        # Update list of track origins if specified
        if "track_origins" in plotting_config:
            trk_origins = plotting_config["track_origins"]

        for model_name, _ in plotting_config["Datasets_to_plot"].items():
            if (
                not plotting_config["Datasets_to_plot"][f"{model_name}"]["files"]
                is None
            ):
                filepath_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"]["files"]
                )
                labels_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"]["label"]
                )
                tracks_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"]["tracks_name"]
                )

        for trk_origin in trk_origins:
            if ("nTracks" in plotting_config) and (plotting_config["nTracks"] is True):
                uit.plot_n_tracks_per_jet(
                    datasets_filepaths=filepath_list,
                    datasets_labels=labels_list,
                    datasets_track_names=tracks_list,
                    class_labels=plotting_config["class_labels"],
                    n_jets=int(plot_config["Eval_parameters"]["n_jets"]),
                    output_directory=plotting_config["folder_to_save"]
                    if plotting_config["folder_to_save"]
                    else "input_vars_trks/",
                    plot_type=plot_type,
                    track_origin=trk_origin,
                    **plotting_config["plot_settings"],
                )

            else:
                uit.plot_input_vars_trks(
                    datasets_filepaths=filepath_list,
                    datasets_labels=labels_list,
                    datasets_track_names=tracks_list,
                    class_labels=plotting_config["class_labels"],
                    var_dict=plot_config["Eval_parameters"]["var_dict"],
                    n_jets=int(plot_config["Eval_parameters"]["n_jets"]),
                    binning=plotting_config["binning"],
                    output_directory=plotting_config["folder_to_save"]
                    if plotting_config["folder_to_save"]
                    else "input_vars_trks/",
                    plot_type=plot_type,
                    track_origin=trk_origin,
                    **plotting_config["plot_settings"],
                )


def plot_jets_variables(plot_config, plot_type):
    """Plot jet variables.

    Parameters
    ----------
    plot_config : object
        plot configuration
    plot_type : str
        Plottype, like pdf or png
    """
    plot_config["Eval_parameters"] = translate_kwargs(plot_config["Eval_parameters"])
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        # Skip general "Eval_parameters" and yaml anchors (have to start with a dot)
        if plot_name == "Eval_parameters" or plot_name.startswith("."):
            continue

        if plotting_config["variables"] == "tracks":
            continue

        logger.info(f"Start {plot_name}...\n")
        plotting_config["plot_settings"] = translate_kwargs(
            plotting_config["plot_settings"]
        )
        filepath_list = []
        labels_list = []

        for model_name, _ in plotting_config["Datasets_to_plot"].items():
            if (
                not plotting_config["Datasets_to_plot"][f"{model_name}"]["files"]
                is None
            ):
                filepath_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"]["files"]
                )
                labels_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"]["label"]
                )

        uit.plot_input_vars_jets(
            datasets_filepaths=filepath_list,
            datasets_labels=labels_list,
            class_labels=plotting_config["class_labels"],
            var_dict=plot_config["Eval_parameters"]["var_dict"],
            n_jets=int(plot_config["Eval_parameters"]["n_jets"]),
            binning=plotting_config["binning"],
            output_directory=plotting_config["folder_to_save"]
            if plotting_config["folder_to_save"]
            else "input_vars_jets/",
            plot_type=plot_type,
            special_param_jets=plotting_config["special_param_jets"],
            **plotting_config["plot_settings"],
        )


if __name__ == "__main__":
    args = get_parser()

    if not (args.jets or args.tracks):
        raise Exception(
            "Please provide '--tracks' or '--jets' to plot their input variables"
        )

    if args.tracks:
        # Open and load the config files used in the eval process
        with open(args.config_file) as yaml_config:
            plots_config = yaml.load(yaml_config, Loader=yaml_loader)

        plot_trks_variables(plots_config, args.format)

    if args.jets:
        # Open and load the config files used in the eval process
        with open(args.config_file) as yaml_config:
            plots_config = yaml.load(yaml_config, Loader=yaml_loader)

        plot_jets_variables(plots_config, args.format)
