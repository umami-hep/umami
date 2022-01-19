#!/usr/bin/env python

"""
This script plots the given input variables of the given files and
also a comparison.
"""

import argparse

import yaml

import umami.input_vars_tools as uit
from umami.configuration import logger
from umami.tools import yaml_loader


def GetParser():
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
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        if plot_name == "Eval_parameters":
            continue

        if plotting_config["variables"] == "jets":
            continue

        logger.info(f"Start {plot_name}...\n")
        filepath_list = []
        labels_list = []
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

        for trk_origin in trk_origins:
            if len(filepath_list) >= 2:
                if ("nTracks" in plotting_config) and (
                    plotting_config["nTracks"] is True
                ):
                    uit.plot_nTracks_per_Jet(
                        datasets_filepaths=filepath_list,
                        datasets_labels=labels_list,
                        class_labels=plotting_config["class_labels"],
                        nJets=int(plot_config["Eval_parameters"]["nJets"]),
                        output_directory="input_vars_trks/"
                        + plotting_config["folder_to_save"],
                        plot_type=plot_type,
                        **plotting_config["plot_settings"],
                        track_origin=trk_origin,
                    )

                else:
                    uit.plot_input_vars_trks_comparison(
                        datasets_filepaths=filepath_list,
                        datasets_labels=labels_list,
                        class_labels=plotting_config["class_labels"],
                        var_dict=plot_config["Eval_parameters"]["var_dict"],
                        nJets=int(plot_config["Eval_parameters"]["nJets"]),
                        binning=plotting_config["binning"],
                        output_directory="input_vars_trks/"
                        + plotting_config["folder_to_save"],
                        plot_type=plot_type,
                        **plotting_config["plot_settings"],
                        track_origin=trk_origin,
                    )

            else:
                if ("nTracks" in plotting_config) and (
                    plotting_config["nTracks"] is True
                ):
                    uit.plot_nTracks_per_Jet(
                        datasets_filepaths=filepath_list,
                        datasets_labels=labels_list,
                        class_labels=plotting_config["class_labels"],
                        nJets=int(plot_config["Eval_parameters"]["nJets"]),
                        output_directory="input_vars_trks/"
                        + plotting_config["folder_to_save"],
                        plot_type=plot_type,
                        **plotting_config["plot_settings"],
                        track_origin=trk_origin,
                    )

                else:
                    uit.plot_input_vars_trks(
                        datasets_filepaths=filepath_list,
                        datasets_labels=labels_list,
                        class_labels=plotting_config["class_labels"],
                        var_dict=plot_config["Eval_parameters"]["var_dict"],
                        nJets=int(plot_config["Eval_parameters"]["nJets"]),
                        binning=plotting_config["binning"],
                        output_directory="input_vars_trks/"
                        + plotting_config["folder_to_save"],
                        plot_type=plot_type,
                        **plotting_config["plot_settings"],
                        track_origin=trk_origin,
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
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        if plot_name == "Eval_parameters":
            continue

        if plotting_config["variables"] == "tracks":
            continue

        logger.info(f"Start {plot_name}...\n")
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

        if len(filepath_list) >= 2:
            uit.plot_input_vars_jets_comparison(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                class_labels=plotting_config["class_labels"],
                var_dict=plot_config["Eval_parameters"]["var_dict"],
                nJets=int(plot_config["Eval_parameters"]["nJets"]),
                binning=plotting_config["binning"],
                output_directory="input_vars_jets/" + plotting_config["folder_to_save"],
                plot_type=plot_type,
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

        else:
            uit.plot_input_vars_jets(
                datasets_filepaths=filepath_list,
                datasets_labels=labels_list,
                class_labels=plotting_config["class_labels"],
                var_dict=plot_config["Eval_parameters"]["var_dict"],
                nJets=int(plot_config["Eval_parameters"]["nJets"]),
                binning=plotting_config["binning"],
                output_directory="input_vars_jets/" + plotting_config["folder_to_save"],
                plot_type=plot_type,
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )


if __name__ == "__main__":
    args = GetParser()

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
