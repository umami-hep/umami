#!/usr/bin/env python

"""
This script plots the given input variables of the given files and
also a comparison.
"""

import argparse

import yaml

import umami.input_vars_tools as uit
from umami.tools import yaml_loader


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
    args = parser.parse_args()
    return args


def plot_trks_variables(plot_config, plot_type):
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        if plot_name == "Eval_parameters":
            continue

        if plotting_config["variables"] == "jets":
            continue

        print(f"Start {plot_name}...")
        print()
        filepath_list = []
        labels_list = []

        for model_name, model_config in plotting_config[
            "Datasets_to_plot"
        ].items():
            if (
                not plotting_config["Datasets_to_plot"][f"{model_name}"][
                    "files"
                ]
                is None
            ):
                filepath_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"][
                        "files"
                    ]
                )
                labels_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"][
                        "label"
                    ]
                )

        if len(filepath_list) >= 2:
            if ("nTracks" in plotting_config) and (
                plotting_config["nTracks"] is True
            ):
                uit.plot_nTracks_per_Jet(
                    filepaths=filepath_list,
                    labels=labels_list,
                    flavors=plotting_config["flavors"],
                    var_dict=plot_config["Eval_parameters"]["var_dict"],
                    nJets=int(plot_config["Eval_parameters"]["nJets"]),
                    output_directory="input_vars_trks/"
                    + plotting_config["folder_to_save"],
                    plot_type=plot_type,
                    **plotting_config["plot_settings"],
                )

            else:
                uit.plot_input_vars_trks_comparison(
                    filepaths=filepath_list,
                    labels=labels_list,
                    flavors=plotting_config["flavors"],
                    var_dict=plot_config["Eval_parameters"]["var_dict"],
                    nJets=int(plot_config["Eval_parameters"]["nJets"]),
                    binning=plotting_config["binning"],
                    output_directory="input_vars_trks/"
                    + plotting_config["folder_to_save"],
                    plot_type=plot_type,
                    **plotting_config["plot_settings"],
                )

        else:
            if ("nTracks" in plotting_config) and (
                plotting_config["nTracks"] is True
            ):
                uit.plot_nTracks_per_Jet(
                    filepaths=filepath_list,
                    labels=labels_list,
                    flavors=plotting_config["flavors"],
                    var_dict=plot_config["Eval_parameters"]["var_dict"],
                    nJets=int(plot_config["Eval_parameters"]["nJets"]),
                    output_directory="input_vars_trks/"
                    + plotting_config["folder_to_save"],
                    plot_type=plot_type,
                    **plotting_config["plot_settings"],
                )

            else:
                uit.plot_input_vars_trks(
                    filepaths=filepath_list,
                    labels=labels_list,
                    flavors=plotting_config["flavors"],
                    var_dict=plot_config["Eval_parameters"]["var_dict"],
                    nJets=int(plot_config["Eval_parameters"]["nJets"]),
                    binning=plotting_config["binning"],
                    output_directory="input_vars_trks/"
                    + plotting_config["folder_to_save"],
                    plot_type=plot_type,
                    **plotting_config["plot_settings"],
                )


def plot_jets_variables(plot_config, plot_type):
    # Iterate over the different plots which are to be plotted
    for plot_name, plotting_config in plot_config.items():
        if plot_name == "Eval_parameters":
            continue

        if plotting_config["variables"] == "tracks":
            continue

        print(f"Start {plot_name}...")
        print()
        filepath_list = []
        labels_list = []

        for model_name, model_config in plotting_config[
            "Datasets_to_plot"
        ].items():
            if (
                not plotting_config["Datasets_to_plot"][f"{model_name}"][
                    "files"
                ]
                is None
            ):
                filepath_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"][
                        "files"
                    ]
                )
                labels_list.append(
                    plotting_config["Datasets_to_plot"][f"{model_name}"][
                        "label"
                    ]
                )

        if len(filepath_list) >= 2:
            uit.plot_input_vars_jets_comparison(
                filepaths=filepath_list,
                labels=labels_list,
                flavors=plotting_config["flavors"],
                var_dict=plot_config["Eval_parameters"]["var_dict"],
                nJets=int(plot_config["Eval_parameters"]["nJets"]),
                binning=plotting_config["binning"],
                output_directory="input_vars_jets/"
                + plotting_config["folder_to_save"],
                plot_type=plot_type,
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )

        else:
            uit.plot_input_vars_jets(
                filepaths=filepath_list,
                labels=labels_list,
                flavors=plotting_config["flavors"],
                var_dict=plot_config["Eval_parameters"]["var_dict"],
                nJets=int(plot_config["Eval_parameters"]["nJets"]),
                binning=plotting_config["binning"],
                output_directory="input_vars_jets/"
                + plotting_config["folder_to_save"],
                plot_type=plot_type,
                special_param_jets=plotting_config["special_param_jets"],
                **plotting_config["plot_settings"],
            )


if __name__ == "__main__":
    args = GetParser()

    if args.tracks:
        # Open and load the config files used in the eval process
        with open(args.config_file) as yaml_config:
            plot_config = yaml.load(yaml_config, Loader=yaml_loader)

        plot_trks_variables(plot_config, args.format)

    if args.jets:
        # Open and load the config files used in the eval process
        with open(args.config_file) as yaml_config:
            plot_config = yaml.load(yaml_config, Loader=yaml_loader)

        plot_jets_variables(plot_config, args.format)
