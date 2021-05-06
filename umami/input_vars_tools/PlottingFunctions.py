#!/usr/bin/env python

"""
This script plots the given input variables of the given files and
also a comparison.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.tools import yaml_loader


def plot_input_vars_trks(
    plot_config,
    nJets,
    binning,
    flavors,
    var_dict,
    bool_use_taus=False,
    sorting_variable="ptfrac",
    nLeading=None,
    plot_type="pdf",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag=r"$\sqrt{s}$ = 13 TeV, $t\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=10,
    output_directory="input_vars_trks",
    figsize=None,
):
    nBins_dict = {}

    for variable in binning:
        if type(binning[variable]) is list:
            nBins_dict.update({variable: np.asarray(binning[variable])})

        else:
            nBins_dict.update({variable: binning[variable]})

    # Init list for files
    file_list = []
    file_name_list = []

    # Check for given files
    if plot_config.test_file is not None:
        file_list.append(plot_config.test_file)
        file_name_list.append("Test")

    if plot_config.comparison_file is not None:
        file_list.append(plot_config.comparison_file)
        file_name_list.append("Comparison")

    # Load var dict
    with open(plot_config.plot_settings_tracks["var_dict"], "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Iterate over files
    for i, (file, release) in enumerate(
        zip(
            file_list,
            file_name_list,
        )
    ):
        print(f"File: {release}")

        # Loading the labels to remove jets that are not used
        labels = h5py.File(file, "r")["/jets"][:nJets][
            variable_config["label"]
        ]

        # Set up a bool list
        indices_toremove = np.where(labels > 5)[0]

        # Getting the flavor labels
        flavor_labels = np.delete(labels, indices_toremove, 0)

        # Load tracks
        trks = np.asarray(h5py.File(file, "r")["/tracks"][:nJets])

        # Delete all not b, c or light jets
        trks = np.delete(trks, indices_toremove, 0)

        # Loading track variables
        noNormVars = variable_config["track_train_variables"]["noNormVars"]
        logNormVars = variable_config["track_train_variables"]["logNormVars"]
        jointNormVars = variable_config["track_train_variables"][
            "jointNormVars"
        ]
        trksVars = noNormVars + logNormVars + jointNormVars

        # Sort after given variable
        sorting = np.argsort(-1 * trks[sorting_variable])

        # Check if path is existing, if not mkdir
        if nLeading is None:
            if not os.path.isdir(f"{output_directory}/{sorting_variable}/"):
                os.makedirs(f"{output_directory}/{sorting_variable}/")
            filedir = f"{output_directory}/{sorting_variable}"

        else:
            if not os.path.isdir(
                f"{output_directory}/{sorting_variable}/{nLeading}/"
            ):
                os.makedirs(
                    f"{output_directory}/{sorting_variable}/{nLeading}/"
                )
            filedir = f"{output_directory}/{sorting_variable}/{nLeading}"

        print(f"Sorting: {sorting_variable}")
        print(f"nLeading track: {nLeading}")
        print()

        # Loop over vars
        for var in trksVars:
            print(f"Plotting {var}...")

            # Sort the variables and tracks after given variable
            tmp = np.asarray(
                [
                    trks[var][i][sorting[i]]
                    for i in range(len(trks[sorting_variable]))
                ]
            )

            # Calculate unified Binning
            b = tmp[flavor_labels == 5]

            if nBins_dict[var] is None:
                _, Binning = np.histogram(
                    b[:, nLeading][~np.isnan(b[:, nLeading])]
                )

            else:
                _, Binning = np.histogram(
                    b[:, nLeading][~np.isnan(b[:, nLeading])],
                    bins=nBins_dict[var],
                )

            # Set up new figure
            if figsize is None:
                fig = plt.figure(figsize=(8.27 * 0.8, 11.69 * 0.8))

            else:
                fig = plt.figure(figsize=(figsize[0], figsize[1]))

            for i, flavor in enumerate(flavors):
                jets = tmp[flavor_labels == flavors[flavor]]

                # Get number of tracks
                nTracks = len(jets[:, nLeading][~np.isnan(jets[:, nLeading])])

                # Calculate Binning and counts for plotting
                counts, Bins = np.histogram(
                    np.clip(
                        jets[:, nLeading][~np.isnan(jets[:, nLeading])],
                        Binning[0],
                        Binning[-1],
                    ),
                    bins=Binning,
                )

                # Calculate the bin centers
                bincentres = [
                    (Binning[i] + Binning[i + 1]) / 2.0
                    for i in range(len(Binning) - 1)
                ]

                # Calculate poisson uncertainties and lower bands
                unc = np.sqrt(counts) / nTracks
                band_lower = counts / nTracks - unc

                plt.hist(
                    x=Bins[:-1],
                    bins=Bins,
                    weights=(counts / nTracks),
                    histtype="step",
                    linewidth=1.0,
                    color=f"C{i}",
                    stacked=False,
                    fill=False,
                    label=r"${}$-jets".format(flavor),
                )

                plt.hist(
                    x=bincentres,
                    bins=Bins,
                    bottom=band_lower,
                    weights=unc * 2,
                    fill=False,
                    hatch="/////",
                    linewidth=0,
                    edgecolor="#666666",
                )

            if nLeading is None:
                plt.xlabel(var)

            else:
                plt.xlabel(f"{nLeading+1} leading tracks {var}")
            plt.ylabel("Normalised Number of Tracks")
            plt.yscale("log")

            ymin, ymax = plt.ylim()
            plt.ylim(ymin=0.01 * ymin, ymax=yAxisIncrease * ymax)
            plt.legend(loc="best")
            plt.tight_layout()

            ax = plt.gca()
            if UseAtlasTag is True:
                pas.makeATLAStag(
                    ax,
                    fig,
                    first_tag=AtlasTag,
                    second_tag=SecondTag + " " + release + " File",
                    ymax=yAxisAtlasTag,
                )

            plt.savefig(f"{filedir}/{var}_{release}.{plot_type}")
            plt.close()
            plt.clf()
        print()


def plot_input_vars_jets(
    plot_config,
    nJets,
    binning,
    flavors,
    var_dict,
    bool_use_taus=False,
    special_param_jets=None,
    plot_type="pdf",
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag=r"$\sqrt{s}$ = 13 TeV,",
    save_dict=None,
    yAxisAtlasTag=0.925,
    yAxisIncrease=10,
    output_directory="input_vars_jets",
    figsize=None,
):
    nBins_dict = {}

    for variable in binning:
        if type(binning[variable]) is list:
            nBins_dict.update({variable: np.asarray(binning[variable])})

        else:
            nBins_dict.update({variable: binning[variable]})

    # Init list for files
    file_list = []
    file_name_list = []

    # Check for given files
    if plot_config.test_file is not None:
        file_list.append(plot_config.test_file)
        file_name_list.append("ttbar")

    if plot_config.test_file_Zext is not None:
        file_list.append(plot_config.test_file_Zext)
        file_name_list.append("zpext")

    if plot_config.comparison_file is not None:
        file_list.append(plot_config.comparison_file)
        file_name_list.append("Comparison")

    # Load var dict
    with open(plot_config.plot_settings_jets["var_dict"], "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Iterate over files
    for i, (file, release) in enumerate(
        zip(
            file_list,
            file_name_list,
        )
    ):
        print(f"File: {release}")

        # Loading the jets
        jets = pd.DataFrame(h5py.File(file, "r")["/jets"][:nJets][:])

        # Remove non-interesting flavours
        if bool_use_taus:
            jets.query(
                f"{variable_config['label']} in [0, 4, 5, 15]", inplace=True
            )
        else:
            jets.query(f"{variable_config['label']} <= 5", inplace=True)

        # Loading jet variables
        jetsVars = [
            i
            for j in variable_config["train_variables"]
            for i in variable_config["train_variables"][j]
        ]

        # Check if path is existing, if not mkdir
        if not os.path.isdir(f"{output_directory}/jets_variable/{release}/"):
            os.makedirs(f"{output_directory}/jets_variable/{release}/")
        filedir = f"{output_directory}/jets_variable/{release}/"

        colour_dict = {
            0: "#2ca02c",
            4: "#ff7f0e",
            5: "#1f77b4",
            15: "#7c5295",
        }
        # Loop over vars
        for var in jetsVars:
            print(f"Plotting {var}...")

            tmp = jets[[variable_config["label"], var]]
            tmp = tmp.dropna()
            # Calculate unified Binning
            b = tmp[tmp[variable_config["label"]] == 5]

            var_range = None
            if special_param_jets is not None and var in special_param_jets:
                if (
                    "lim_left" in special_param_jets[var]
                    and "lim_right" in special_param_jets[var]
                ):
                    lim_left = special_param_jets[var]["lim_left"]
                    lim_right = special_param_jets[var]["lim_right"]
                    var_range = (lim_left, lim_right)

            if nBins_dict[var] is None:
                _, Binning = np.histogram(b[var], range=var_range)
            else:
                _, Binning = np.histogram(
                    b[var],
                    bins=nBins_dict[var],
                    range=var_range,
                )

            # Set up new figure
            if figsize is None:
                fig = plt.figure(figsize=(8.27 * 0.8, 11.69 * 0.8))

            else:
                fig = plt.figure(figsize=(figsize[0], figsize[1]))

            for i, flavor in enumerate(flavors):
                jets_flavor = tmp[
                    tmp[variable_config["label"]] == flavors[flavor]
                ]

                # Get number of jets
                nJets_flavor = len(jets_flavor)

                # Calculate Binning and counts for plotting
                counts, Bins = np.histogram(
                    np.clip(
                        jets_flavor[var],
                        Binning[0],
                        Binning[-1],
                    ),
                    bins=Binning,
                )

                # Calculate the bin centers
                bincentres = [
                    (Binning[i] + Binning[i + 1]) / 2.0
                    for i in range(len(Binning) - 1)
                ]

                # Calculate poisson uncertainties and lower bands
                unc = np.sqrt(counts) / nJets_flavor
                band_lower = counts / nJets_flavor - unc

                plt.hist(
                    x=Bins[:-1],
                    bins=Bins,
                    weights=(counts / nJets_flavor),
                    histtype="step",
                    linewidth=1.0,
                    color=colour_dict[flavors[flavor]],
                    stacked=False,
                    fill=False,
                    label=r"${}$-jets".format(flavor),
                )

                plt.hist(
                    x=bincentres,
                    bins=Bins,
                    bottom=band_lower,
                    weights=unc * 2,
                    fill=False,
                    hatch="/////",
                    linewidth=0,
                    edgecolor="#666666",
                )

            plt.xlabel(var)
            plt.ylabel("Normalised Number of Jets")
            plt.yscale("log")

            ymin, ymax = plt.ylim()
            plt.ylim(ymin=0.01 * ymin, ymax=yAxisIncrease * ymax)
            plt.legend(loc="upper right")
            plt.tight_layout()

            ax = plt.gca()
            release_label = ""
            if release == "ttbar":
                release_label = r"$t\bar{t}$"
            elif release == "zpext":
                release_label = r"$Z'$"
            if UseAtlasTag is True:
                pas.makeATLAStag(
                    ax,
                    fig,
                    first_tag=AtlasTag,
                    second_tag=SecondTag + " " + release_label,
                    ymax=yAxisAtlasTag,
                )

            plt.savefig(f"{filedir}/{var}.{plot_type}")
            plt.close()
            plt.clf()
        print()
