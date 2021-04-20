#!/usr/bin/env python

"""
This script plots the given input variables of the given files and
also a comparison.
"""

import argparse

import umami.input_vars_tools as uit


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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = GetParser()
    plot_config = uit.Configuration(args.config_file)

    uit.plot_input_vars_trks(
        plot_config=plot_config,
        nJets=plot_config.nJets,
        binning=plot_config.binning,
        flavors=plot_config.flavors,
        **plot_config.plot_settings,
    )
