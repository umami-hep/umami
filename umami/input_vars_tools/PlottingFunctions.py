#!/usr/bin/env python

"""
This script plots the given input variables of the given files and
also a comparison.
"""

import os
import re
from glob import glob

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import gridspec

import umami.evaluation_tools as uet
import umami.train_tools as utt
from umami.configuration import global_config, logger
from umami.tools import applyATLASstyle, makeATLAStag, yaml_loader


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def plot_nTracks_per_Jet(
    datasets_filepaths,
    datasets_labels,
    var_dict,
    nJets,
    class_labels,
    plot_type="pdf",
    UseAtlasTag=True,
    ApplyATLASStyle=False,
    AtlasTag="Internal Simulation",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=1,
    output_directory="input_vars_trks",
    figsize=None,
    Log=True,
    ylabel="Normalised Number of Tracks",
    ycolor="black",
    legFontSize=10,
    ncol=2,
    Ratio_Cut=None,
    Bin_Width_y_axis=True,
):
    """
    Plotting the number of tracks per jet as a histogram.

    Input:
    - datasets_filepaths: List of filepaths to the files.
    - datasets_labels: Label of the dataset for the legend.
    - var_dict: Variable dict where all variables of the files are saved.
    - nJets: Number of jets to use for plotting.
    - class_labels: List of class_labels which are to be plotted.
    - plot_type: Plottype, like pdf or png
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - output_directory: Name of the output directory. Only the dir name not path!
    - figsize: List of the figure size. i.e [5, 6]
    - Log: Set y-axis log True or False.
    - ylabel: Y-label.
    - ycolor: Y-axis-label colour.
    - legFontSize: Legend font size
    - ncol: Number of columns of the legend.
    - Ratio_Cut: List of y-axis cuts for the ratio block.
    - Bin_Width_y_axis: Show bin size on y-axis

    Output:
    - Number of tracks per jet plot
    """

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label) in zip(
        datasets_filepaths,
        datasets_labels,
    ):
        # Init jet counter
        nJets_counter = 0

        # Get the filepath of the dataset
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(
            sorted(filepath, key=natural_keys)
        ):
            if nJets_counter < nJets:
                tmp_trks, tmp_flavour_labels = utt.LoadTrksFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    print_logger=False,
                )

                if file_counter == 0:
                    # Append to array
                    trks = tmp_trks
                    flavour_labels = tmp_flavour_labels

                else:
                    # Append to array
                    trks = np.concatenate((trks, tmp_trks))
                    flavour_labels = np.concatenate(
                        (flavour_labels, tmp_flavour_labels)
                    )

                # Add number of jets to counter
                nJets_counter += len(tmp_trks)

            else:
                break

        if len(trks) < nJets:
            n_trks = len(trks)
            logger.warning(
                f"{nJets} were set to be used, but only {n_trks} are available for {label} files!"
            )

        # Append trks to dict
        trks_dict.update({label: trks[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    # Apply ATLAS style if true
    if ApplyATLASStyle:
        applyATLASstyle(mtp)

    # Set up new figure
    if not figsize:
        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(8, 1, figure=fig)
    axis_dict["left"] = {}
    axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
    axis_dict["left"]["ratio"] = fig.add_subplot(
        gs[6:, 0], sharex=axis_dict["left"]["top"]
    )

    # Init bincounts for ratio calculation
    bincounts = {}
    bincounts_unc = {}

    # Init lowest bincount
    Lowest_histcount = 1

    # Iterate over models
    for model_number, (label, linestyle) in enumerate(
        zip(datasets_labels, linestyles[: len(datasets_labels)])
    ):
        # Sort after given variable
        trks = np.asarray(trks_dict[label])
        nTracks = np.sum(~np.isnan(trks["ptfrac"]), axis=1)

        if model_number == 0:
            # Calculate unified Binning
            nTracks_first = nTracks[flavour_label_dict[label] == 0]

            _, Binning = np.histogram(
                nTracks_first,
                bins=np.arange(-0.5, 40.5, 1),
            )

        for flav_label, flavour in enumerate(class_labels):
            nTracks_flavour = nTracks[flavour_label_dict[label] == flav_label]

            # Calculate bins
            bins, weights, unc, band = uet.calc_bins(
                input_array=nTracks_flavour,
                Binning=Binning,
            )

            hist_counts, _, _ = axis_dict["left"]["top"].hist(
                x=bins[:-1],
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=1.0,
                linestyle=linestyle,
                color=global_config.flavour_categories[flavour]["colour"],
                stacked=False,
                fill=False,
                label=global_config.flavour_categories[flavour]["legend_label"]
                + f" {label}",
            )

            if flavour == class_labels[-1] and model_number == 0:
                axis_dict["left"]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    label="stat. unc.",
                    **global_config.hist_err_style,
                )

            else:
                axis_dict["left"]["top"].hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=band,
                    weights=unc * 2,
                    **global_config.hist_err_style,
                )

            bincounts.update({f"{flavour}{model_number}": hist_counts})
            bincounts_unc.update({f"{flavour}{model_number}": unc})

            for count in hist_counts:
                if count != 0 and count < Lowest_histcount:
                    Lowest_histcount = count

        # Start ratio plot
        if model_number != 0:
            for flavour in class_labels:
                step, step_unc = uet.calc_ratio(
                    counter=bincounts["{}{}".format(flavour, model_number)],
                    denominator=bincounts["{}{}".format(flavour, 0)],
                    counter_unc=bincounts_unc[
                        "{}{}".format(flavour, model_number)
                    ],
                    denominator_unc=bincounts_unc["{}{}".format(flavour, 0)],
                )

                axis_dict["left"]["ratio"].step(
                    x=Binning,
                    y=step,
                    color=global_config.flavour_categories[flavour]["colour"],
                    linestyle=linestyles[model_number],
                )

                axis_dict["left"]["ratio"].fill_between(
                    x=Binning,
                    y1=step - step_unc,
                    y2=step + step_unc,
                    step="pre",
                    facecolor="none",
                    edgecolor=global_config.hist_err_style["edgecolor"],
                    linewidth=global_config.hist_err_style["linewidth"],
                    hatch=global_config.hist_err_style["hatch"],
                )

        elif model_number == 0:
            # Add black line at one
            axis_dict["left"]["ratio"].axhline(
                y=1,
                xmin=axis_dict["left"]["ratio"].get_xlim()[0],
                xmax=axis_dict["left"]["ratio"].get_xlim()[1],
                color="black",
                alpha=0.5,
            )

    axis_dict["left"]["ratio"].set_xlim(
        left=Binning[0],
        right=Binning[-1],
    )

    # Add axes, titels and the legend
    if Bin_Width_y_axis is True:
        Bin_Width = abs(Binning[1] - Binning[0])
        axis_dict["left"]["top"].set_ylabel(
            ylabel + " / {:.2f}".format(Bin_Width),
            fontsize=12,
            horizontalalignment="right",
            y=1.0,
            color=ycolor,
        )

    else:
        axis_dict["left"]["top"].set_ylabel(
            ylabel,
            fontsize=12,
            horizontalalignment="right",
            y=1.0,
            color=ycolor,
        )

    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)

    axis_dict["left"]["ratio"].set_xlabel(
        "Number of tracks per Jet",
        fontsize=12,
        horizontalalignment="right",
        x=1.0,
    )

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    if Log is True:
        axis_dict["left"]["top"].set_yscale("log")

        if axis_dict["left"]["top"].get_ylim()[0] <= 0:
            # Set lower y limit
            left_y_limits = axis_dict["left"]["top"].get_ylim()
            axis_dict["left"]["top"].set_ylim(
                bottom=Lowest_histcount * 0.1,
                top=left_y_limits[1] * 10 ** (yAxisIncrease),
            )

        else:
            left_y_limits = axis_dict["left"]["top"].get_ylim()
            axis_dict["left"]["top"].set_ylim(
                bottom=left_y_limits[0] * 0.1,
                top=left_y_limits[1] * 10 ** (yAxisIncrease),
            )

    else:
        left_y_limits = axis_dict["left"]["top"].get_ylim()
        axis_dict["left"]["top"].set_ylim(
            bottom=left_y_limits[0],
            top=left_y_limits[1] * yAxisIncrease,
        )

    if Ratio_Cut is not None:
        axis_dict["left"]["ratio"].set_ylim(
            bottom=Ratio_Cut[0], top=Ratio_Cut[1]
        )

    # Set axis
    axis_dict["left"]["top"].legend(
        loc="upper right",
        fontsize=legFontSize,
        ncol=ncol,
    )

    # Set tight layout
    plt.tight_layout()

    # Set ATLAS Tag
    if UseAtlasTag is True:
        makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Save and close figure
    plt.savefig(f"{output_directory}/nTracks_per_Jet.{plot_type}")
    plt.close()
    plt.clf()


def plot_input_vars_trks_comparison(
    datasets_filepaths,
    datasets_labels,
    var_dict,
    nJets,
    binning,
    class_labels,
    sorting_variable="ptfrac",
    n_Leading=[None],
    plot_type="pdf",
    UseAtlasTag=True,
    ApplyATLASStyle=False,
    AtlasTag="Internal Simulation",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=1,
    output_directory="input_vars_trks",
    figsize=None,
    Log=True,
    ylabel="Normalised Number of Tracks",
    ycolor="black",
    legFontSize=10,
    ncol=2,
    Ratio_Cut=None,
    Bin_Width_y_axis=True,
):
    """
    Plotting the track variable in comparison to another model with ratio plot.

    Input:
    - datasets_filepaths: List of filepaths to the files.
    - datasets_labels: Label of the dataset for the legend.
    - var_dict: Variable dict where all variables of the files are saved.
    - nJets: Number of jets to use for plotting.
    - binning: Decide which binning is used.
    - class_labels: List of class_labels which are to be plotted.
    - sorting_variable: Variable which is used for sorting.
    - n_Leading: n-th leading jet which is plotted. For all, = None.
    - plot_type: Plottype, like pdf or png
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - output_directory: Name of the output directory. Only the dir name not path!
    - figsize: List of the figure size. i.e [5, 6]
    - Log: Set y-axis log True or False.
    - ylabel: Y-label.
    - ycolor: Y-axis-label colour.
    - legFontSize: Legend font size
    - ncol: Number of columns of the legend.
    - Ratio_Cut: List of y-axis cuts for the ratio block.
    - Bin_Width_y_axis: Show bin size on y-axis

    Output:
    - Track variable in comparison to another model with ratio plot.
    """

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if type(binning[variable]) is list:
            if variable.startswith("number"):
                nBins_dict.update(
                    {
                        variable: np.arange(
                            binning[variable][0] - 0.5,
                            binning[variable][1] - 0.5,
                            binning[variable][2],
                        )
                    }
                )

            else:
                nBins_dict.update(
                    {
                        variable: np.arange(
                            binning[variable][0],
                            binning[variable][1],
                            binning[variable][2],
                        )
                    }
                )

        elif type(binning[variable]) is None:
            nBins_dict.update({variable: int(100)})

        else:
            nBins_dict.update({variable: binning[variable]})

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for filepath, label in zip(
        datasets_filepaths,
        datasets_labels,
    ):
        # Init jet counter
        nJets_counter = 0

        # Get wildcard
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(
            sorted(filepath, key=natural_keys)
        ):
            if nJets_counter < nJets:
                tmp_trks, tmp_flavour_labels = utt.LoadTrksFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    print_logger=False,
                )

                if file_counter == 0:
                    # Append to array
                    trks = tmp_trks
                    flavour_labels = tmp_flavour_labels

                else:
                    # Append to array
                    trks = np.concatenate((trks, tmp_trks))
                    flavour_labels = np.concatenate(
                        (flavour_labels, tmp_flavour_labels)
                    )

                # Add number of jets to counter
                nJets_counter += len(tmp_trks)

            else:
                break

        if len(trks) < nJets:
            n_trks = len(trks)
            logger.warning(
                f"{nJets} were set to be used, but only {n_trks} are available for {label} files!"
            )

        # Append trks to dict
        trks_dict.update({label: trks[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    # Load var dict
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Loading track variables
    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trksVars = noNormVars + logNormVars + jointNormVars

    for nLeading in n_Leading:
        if nLeading == "None":
            nLeading = None

        # Check if path is existing, if not mkdir
        if nLeading is None:
            filedir = os.path.join(output_directory, sorting_variable, "All")

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        else:
            filedir = os.path.join(
                output_directory, sorting_variable, str(nLeading)
            )
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        logger.info(f"Path: {filedir}")
        logger.info(f"Sorting: {sorting_variable}")
        logger.info(f"nLeading track: {nLeading}\n")

        # Loop over vars
        for var in trksVars:
            if var in nBins_dict:
                logger.info(f"Plotting {var}...")

                # Define the figure with two subplots of unequal sizes
                axis_dict = {}

                # Apply ATLAS style if true
                if ApplyATLASStyle:
                    applyATLASstyle(mtp)

                # Set up new figure
                if figsize is None:
                    fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                else:
                    fig = plt.figure(figsize=(figsize[0], figsize[1]))

                gs = gridspec.GridSpec(8, 1, figure=fig)
                axis_dict["left"] = {}
                axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
                axis_dict["left"]["ratio"] = fig.add_subplot(
                    gs[6:, 0], sharex=axis_dict["left"]["top"]
                )

                # Init bincounts for ratio calculation
                bincounts = {}
                bincounts_unc = {}

                # Init lowest bincount
                Lowest_histcount = 1

                # Iterate over models
                for model_number, (label, linestyle) in enumerate(
                    zip(datasets_labels, linestyles[: len(datasets_labels)])
                ):
                    # Sort after given variable
                    sorting = np.argsort(
                        -1 * trks_dict[label][sorting_variable]
                    )

                    # Sort the variables and tracks after given variable
                    tmp = np.asarray(
                        [
                            trks_dict[label][var][k][sorting[k]]
                            for k in range(
                                len(trks_dict[label][sorting_variable])
                            )
                        ]
                    )

                    if model_number == 0:
                        # Calculate unified Binning
                        first_flav = tmp[flavour_label_dict[label] == 0]

                        if nBins_dict[var] is None:
                            _, Binning = np.histogram(
                                first_flav[:, nLeading][
                                    ~np.isnan(first_flav[:, nLeading])
                                ]
                            )

                        else:
                            _, Binning = np.histogram(
                                first_flav[:, nLeading][
                                    ~np.isnan(first_flav[:, nLeading])
                                ],
                                bins=nBins_dict[var],
                            )

                    for flav_label, flavour in enumerate(class_labels):
                        jets = tmp[flavour_label_dict[label] == flav_label]

                        # Get number of tracks
                        Tracks = jets[:, nLeading][
                            ~np.isnan(jets[:, nLeading])
                        ]

                        # Calculate bins
                        bins, weights, unc, band = uet.calc_bins(
                            input_array=Tracks,
                            Binning=Binning,
                        )

                        hist_counts, _, _ = axis_dict["left"]["top"].hist(
                            x=bins[:-1],
                            bins=bins,
                            weights=weights,
                            histtype="step",
                            linewidth=1.0,
                            linestyle=linestyle,
                            color=global_config.flavour_categories[flavour][
                                "colour"
                            ],
                            stacked=False,
                            fill=False,
                            label=global_config.flavour_categories[flavour][
                                "legend_label"
                            ]
                            + f" {label}",
                        )

                        if flavour == class_labels[-1] and model_number == 0:
                            axis_dict["left"]["top"].hist(
                                x=bins[:-1],
                                bins=bins,
                                bottom=band,
                                weights=unc * 2,
                                label="stat. unc.",
                                **global_config.hist_err_style,
                            )

                        else:
                            axis_dict["left"]["top"].hist(
                                x=bins[:-1],
                                bins=bins,
                                bottom=band,
                                weights=unc * 2,
                                **global_config.hist_err_style,
                            )

                        bincounts.update(
                            {f"{flavour}{model_number}": hist_counts}
                        )
                        bincounts_unc.update({f"{flavour}{model_number}": unc})

                        for count in hist_counts:
                            if count != 0 and count < Lowest_histcount:
                                Lowest_histcount = count

                    # Start ratio plot
                    if model_number != 0:
                        for flavour in class_labels:
                            step, step_unc = uet.calc_ratio(
                                counter=bincounts[
                                    "{}{}".format(flavour, model_number)
                                ],
                                denominator=bincounts[
                                    "{}{}".format(flavour, 0)
                                ],
                                counter_unc=bincounts_unc[
                                    "{}{}".format(flavour, model_number)
                                ],
                                denominator_unc=bincounts_unc[
                                    "{}{}".format(flavour, 0)
                                ],
                            )

                            axis_dict["left"]["ratio"].step(
                                x=Binning,
                                y=step,
                                color=global_config.flavour_categories[
                                    flavour
                                ]["colour"],
                                linestyle=linestyles[model_number],
                            )

                            axis_dict["left"]["ratio"].fill_between(
                                x=Binning,
                                y1=step - step_unc,
                                y2=step + step_unc,
                                step="pre",
                                facecolor="none",
                                edgecolor=global_config.hist_err_style[
                                    "edgecolor"
                                ],
                                linewidth=global_config.hist_err_style[
                                    "linewidth"
                                ],
                                hatch=global_config.hist_err_style["hatch"],
                            )

                    elif model_number == 0:
                        # Add black line at one
                        axis_dict["left"]["ratio"].axhline(
                            y=1,
                            xmin=axis_dict["left"]["ratio"].get_xlim()[0],
                            xmax=axis_dict["left"]["ratio"].get_xlim()[1],
                            color="black",
                            alpha=0.5,
                        )

                axis_dict["left"]["ratio"].set_xlim(
                    left=Binning[0],
                    right=Binning[-1],
                )

                # Add axes, titels and the legend
                if Bin_Width_y_axis is True:
                    Bin_Width = abs(Binning[1] - Binning[0])
                    axis_dict["left"]["top"].set_ylabel(
                        ylabel + " / {:.2f}".format(Bin_Width),
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=ycolor,
                    )

                else:
                    axis_dict["left"]["top"].set_ylabel(
                        ylabel,
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=ycolor,
                    )

                axis_dict["left"]["top"].tick_params(
                    axis="y", labelcolor=ycolor
                )

                if nLeading is None:
                    axis_dict["left"]["ratio"].set_xlabel(
                        var, fontsize=12, horizontalalignment="right", x=1.0
                    )

                else:
                    axis_dict["left"]["ratio"].set_xlabel(
                        f"{nLeading+1} leading tracks {var}",
                        fontsize=12,
                        horizontalalignment="right",
                        x=1.0,
                    )

                plt.setp(
                    axis_dict["left"]["top"].get_xticklabels(), visible=False
                )

                if Log is True:
                    axis_dict["left"]["top"].set_yscale("log")

                    if axis_dict["left"]["top"].get_ylim()[0] <= 0:
                        # Set lower y limit
                        left_y_limits = axis_dict["left"]["top"].get_ylim()
                        axis_dict["left"]["top"].set_ylim(
                            bottom=Lowest_histcount * 0.1,
                            top=left_y_limits[1] * 10 ** (yAxisIncrease),
                        )

                    else:
                        left_y_limits = axis_dict["left"]["top"].get_ylim()
                        axis_dict["left"]["top"].set_ylim(
                            bottom=left_y_limits[0] * 0.1,
                            top=left_y_limits[1] * 10 ** (yAxisIncrease),
                        )

                else:
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=left_y_limits[0],
                        top=left_y_limits[1] * yAxisIncrease,
                    )

                if Ratio_Cut is not None:
                    axis_dict["left"]["ratio"].set_ylim(
                        bottom=Ratio_Cut[0], top=Ratio_Cut[1]
                    )

                # Set axis
                axis_dict["left"]["top"].legend(
                    loc="upper right",
                    fontsize=legFontSize,
                    ncol=ncol,
                )

                # Set tight layout
                plt.tight_layout()

                # Set ATLAS Tag
                if UseAtlasTag is True:
                    makeATLAStag(
                        ax=axis_dict["left"]["top"],
                        fig=fig,
                        first_tag=AtlasTag,
                        second_tag=SecondTag,
                        ymax=yAxisAtlasTag,
                    )

                # Save and close figure
                plt.savefig(f"{filedir}/{var}_{nLeading}.{plot_type}")
                plt.close()
                plt.clf()
        logger.info(
            "\n-------------------------------------------------------------------------------"
        )


def plot_input_vars_trks(
    datasets_filepaths,
    datasets_labels,
    var_dict,
    nJets,
    binning,
    class_labels,
    sorting_variable="ptfrac",
    n_Leading=None,
    plot_type="pdf",
    UseAtlasTag=True,
    ApplyATLASStyle=False,
    AtlasTag="Internal Simulation",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=10,
    output_directory="input_vars_trks",
    figsize=None,
    Log=True,
    ylabel="Normalised number of tracks",
    ycolor="black",
    legFontSize=10,
    ncol=2,
    Bin_Width_y_axis=True,
):
    """
    Plotting the track variable.

    Input:
    - datasets_filepaths: List of filepaths to the files.
    - datasets_labels: Label of the dataset for the legend.
    - var_dict: Variable dict where all variables of the files are saved.
    - nJets: Number of jets to use for plotting.
    - binning: Decide which binning is used.
    - class_labels: List of class_labels which are to be plotted.
    - sorting_variable: Variable which is used for sorting.
    - n_Leading: n-th leading jet which is plotted. For all, = None.
    - plot_type: Plottype, like pdf or png
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - output_directory: Name of the output directory. Only the dir name not path!
    - figsize: List of the figure size. i.e [5, 6]
    - Log: Set y-axis log True or False.
    - ylabel: Y-label.
    - ycolor: Y-axis-label colour.
    - legFontSize: Legend font size
    - ncol: Number of columns of the legend.
    - Bin_Width_y_axis: Show bin size on y-axis

    Output:
    - Track variable for the n-th leading jets
    """

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if type(binning[variable]) is list:
            if variable.startswith("number"):
                nBins_dict.update(
                    {
                        variable: np.arange(
                            binning[variable][0] - 0.5,
                            binning[variable][1] - 0.5,
                            binning[variable][2],
                        )
                    }
                )

            else:
                nBins_dict.update(
                    {
                        variable: np.arange(
                            binning[variable][0],
                            binning[variable][1],
                            binning[variable][2],
                        )
                    }
                )

        elif type(binning[variable]) is None:
            nBins_dict.update({variable: int(100)})

        else:
            nBins_dict.update({variable: binning[variable]})

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label) in zip(
        datasets_filepaths,
        datasets_labels,
    ):
        # Init jet counter
        nJets_counter = 0

        # Get wildcard
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(
            sorted(filepath, key=natural_keys)
        ):
            if nJets_counter < nJets:
                tmp_trks, tmp_flavour_labels = utt.LoadTrksFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    print_logger=False,
                )

                if file_counter == 0:
                    # Append to array
                    trks = tmp_trks
                    flavour_labels = tmp_flavour_labels

                else:
                    # Append to array
                    trks = np.concatenate((trks, tmp_trks))
                    flavour_labels = np.concatenate(
                        (flavour_labels, tmp_flavour_labels)
                    )

                # Add number of jets to counter
                nJets_counter += len(tmp_trks)

            else:
                break

        if len(trks) < nJets:
            n_trks = len(trks)
            logger.warning(
                f"{nJets} were set to be used, but only {n_trks} are available for {label} files!"
            )

        # Append trks to dict
        trks_dict.update({label: trks[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    # Load var dict
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Loading track variables
    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trksVars = noNormVars + logNormVars + jointNormVars

    for nLeading in n_Leading:
        if nLeading == "None":
            nLeading = None

        # Check if path is existing, if not mkdir
        if nLeading is None:
            filedir = os.path.join(output_directory, sorting_variable, "All/")

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        else:
            filedir = os.path.join(
                output_directory, sorting_variable, str(nLeading)
            )
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        logger.info(f"Path: {filedir}")
        logger.info(f"Sorting: {sorting_variable}")
        logger.info(f"nLeading track: {nLeading}\n")

        for var in nBins_dict:
            if var not in trksVars:
                logger.info(
                    f"{var} in config, but not in Variables yaml! Skipping..."
                )

        # Loop over vars
        for var in trksVars:
            if var not in nBins_dict:
                logger.info(
                    f"{var} in Variables yaml but not in config! Skipping..."
                )

            else:
                logger.info(f"Plotting {var}...")

                # Iterate over models
                for (label, linestyle) in zip(
                    datasets_labels, linestyles[: len(datasets_labels)]
                ):
                    # Sort after given variable
                    sorting = np.argsort(
                        -1 * trks_dict[label][sorting_variable]
                    )

                    # Sort the variables and tracks after given variable
                    tmp = np.asarray(
                        [
                            trks_dict[label][var][k][sorting[k]]
                            for k in range(
                                len(trks_dict[label][sorting_variable])
                            )
                        ]
                    )

                    # Calculate unified Binning
                    first_flav = tmp[flavour_label_dict[label] == 0]

                    # Check if binning is already set
                    if nBins_dict[var] is None:

                        # Get Binning
                        _, Binning = np.histogram(
                            first_flav[:, nLeading][
                                ~np.isnan(first_flav[:, nLeading])
                            ]
                        )

                    else:
                        # Get Binning
                        _, Binning = np.histogram(
                            first_flav[:, nLeading][
                                ~np.isnan(first_flav[:, nLeading])
                            ],
                            bins=nBins_dict[var],
                        )

                    # Apply ATLAS style if true
                    if ApplyATLASStyle:
                        applyATLASstyle(mtp)

                    # Set up new figure
                    if figsize is None:
                        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                    else:
                        fig = plt.figure(figsize=(figsize[0], figsize[1]))

                    # Iterate over flavours
                    for flav_label, flavour in enumerate(class_labels):

                        # Get all jets with wanted flavour
                        jets = tmp[flavour_label_dict[label] == flav_label]

                        # Get number of tracks
                        Tracks = jets[:, nLeading][
                            ~np.isnan(jets[:, nLeading])
                        ]

                        # Calculate bins
                        bins, weights, unc, band = uet.calc_bins(
                            input_array=Tracks,
                            Binning=Binning,
                        )

                        plt.hist(
                            x=bins[:-1],
                            bins=bins,
                            weights=weights,
                            histtype="step",
                            linewidth=1.0,
                            linestyle=linestyle,
                            color=global_config.flavour_categories[flavour][
                                "colour"
                            ],
                            stacked=False,
                            fill=False,
                            label=global_config.flavour_categories[flavour][
                                "legend_label"
                            ]
                            + f" {label}",
                        )

                        if flavour == class_labels[-1]:
                            plt.hist(
                                x=bins[:-1],
                                bins=bins,
                                bottom=band,
                                weights=unc * 2,
                                label="stat. unc.",
                                **global_config.hist_err_style,
                            )

                        else:
                            plt.hist(
                                x=bins[:-1],
                                bins=bins,
                                bottom=band,
                                weights=unc * 2,
                                **global_config.hist_err_style,
                            )

                    if nLeading is None:
                        plt.xlabel(var)

                    else:
                        plt.xlabel(f"{nLeading+1} leading tracks {var}")

                    # Add axes, titels and the legend
                    if Bin_Width_y_axis is True:
                        Bin_Width = abs(Binning[1] - Binning[0])
                        plt.ylabel(
                            ylabel + " / {:.2f}".format(Bin_Width),
                            fontsize=12,
                            horizontalalignment="right",
                            y=1.0,
                            color=ycolor,
                        )

                    else:
                        plt.ylabel(
                            ylabel,
                            fontsize=12,
                            horizontalalignment="right",
                            y=1.0,
                            color=ycolor,
                        )

                    if Log is True:
                        plt.yscale("log")
                        ymin, ymax = plt.ylim()
                        plt.ylim(ymin=0.01 * ymin, ymax=yAxisIncrease * ymax)

                    else:
                        ymin, ymax = plt.ylim()
                        plt.ylim(ymin=0.8 * ymin, ymax=yAxisIncrease * ymax)

                    plt.legend(
                        loc="upper right", fontsize=legFontSize, ncol=ncol
                    )
                    plt.tight_layout()

                    ax = plt.gca()
                    if UseAtlasTag is True:
                        makeATLAStag(
                            ax=ax,
                            fig=fig,
                            first_tag=AtlasTag,
                            second_tag=SecondTag,
                            ymax=yAxisAtlasTag,
                        )

                    plt.savefig(f"{filedir}/{var}_{nLeading}.{plot_type}")
                    plt.close()
                    plt.clf()
                    logger.info(f"{filedir}/{var}.{plot_type}\n")


def plot_input_vars_jets(
    datasets_filepaths,
    datasets_labels,
    var_dict,
    nJets,
    binning,
    class_labels,
    special_param_jets=None,
    plot_type="pdf",
    UseAtlasTag=True,
    ApplyATLASStyle=False,
    AtlasTag="Internal Simulation",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=10,
    output_directory="input_vars_jets",
    figsize=None,
    Log=True,
    ylabel="Normalised number of tracks",
    ycolor="black",
    legFontSize=10,
    ncol=2,
    Bin_Width_y_axis=True,
):
    """
    Plotting the jet variable.

    Input:
    - datasets_filepaths: List of filepaths to the files.
    - datasets_labels: Label of the dataset for the legend.
    - var_dict: Variable dict where all variables of the files are saved.
    - nJets: Number of jets to use for plotting.
    - binning: Decide which binning is used.
    - class_labels: List of class_labels which are to be plotted.
    - special_param_jets: Give specific x-axis-limits for variable.
    - plot_type: Plottype, like pdf or png
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - output_directory: Name of the output directory. Only the dir name not path!
    - figsize: List of the figure size. i.e [5, 6]
    - Log: Set y-axis log True or False.
    - ylabel: Y-label.
    - ycolor: Y-axis-label colour.
    - legFontSize: Legend font size
    - ncol: Number of columns of the legend.
    - Bin_Width_y_axis: Show bin size on y-axis

    Output:
    - Jet variable plot
    """

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if type(binning[variable]) is list:
            nBins_dict.update(
                {
                    variable: np.arange(
                        binning[variable][0],
                        binning[variable][1],
                        binning[variable][2],
                    )
                }
            )

        elif type(binning[variable]) is None:
            nBins_dict.update({variable: int(100)})

        else:
            nBins_dict.update({variable: binning[variable]})

    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Init trks and flavour label dicts
    jets_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label) in zip(
        datasets_filepaths,
        datasets_labels,
    ):
        # Init jet counter
        nJets_counter = 0

        # Get wildcard
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(
            sorted(filepath, key=natural_keys)
        ):
            if nJets_counter < nJets:
                tmp_jets, tmp_flavour_labels = utt.LoadJetsFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    print_logger=False,
                )

                if file_counter == 0:
                    # Append to array
                    jets = tmp_jets
                    flavour_labels = tmp_flavour_labels

                else:
                    # Append to array
                    jets = jets.append(tmp_jets, ignore_index=True)
                    flavour_labels = np.append(
                        flavour_labels,
                        tmp_flavour_labels,
                    )

                # Add number of loaded jets to counter
                nJets_counter += len(tmp_jets)

            else:
                break

        if len(jets) < nJets:
            n_jets = len(jets)
            logger.warning(
                f"{nJets} were set to be used, but only {n_jets} are available for {label} files!"
            )

        # Append jets to dict
        jets_dict.update({label: jets[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    # Loading jet variables
    jetsVars = [
        i
        for j in variable_config["train_variables"]
        for i in variable_config["train_variables"][j]
    ]

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")
    filedir = f"{output_directory}/"

    # Loop over vars
    for var in jetsVars:
        if var in nBins_dict:
            logger.info(f"Plotting {var}...")

            for label in datasets_labels:
                # Get variable and the labels of the jets
                jets_var = jets_dict[label][var]
                flavour_labels_var = flavour_label_dict[label]

                # Clean both from nans
                jets_var_clean = jets_var[~np.isnan(jets_var)]
                flavour_label_clean = flavour_labels_var[~np.isnan(jets_var)]

                # Calculate unified Binning
                first_flav = jets_var_clean[flavour_label_clean == 0]

                var_range = None
                if (
                    special_param_jets is not None
                    and var in special_param_jets
                ):
                    if (
                        "lim_left" in special_param_jets[var]
                        and "lim_right" in special_param_jets[var]
                    ):
                        lim_left = special_param_jets[var]["lim_left"]
                        lim_right = special_param_jets[var]["lim_right"]
                        var_range = (lim_left, lim_right)

                if nBins_dict[var] is None:
                    _, Binning = np.histogram(first_flav, range=var_range)

                else:
                    _, Binning = np.histogram(
                        first_flav,
                        bins=nBins_dict[var],
                        range=var_range,
                    )

                # Apply ATLAS style if true
                if ApplyATLASStyle:
                    applyATLASstyle(mtp)

                # Set up new figure
                if figsize is None:
                    fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                else:
                    fig = plt.figure(figsize=(figsize[0], figsize[1]))

                for flav_label, flavour in enumerate(class_labels):
                    jets_flavour = jets_var_clean[
                        flavour_label_clean == flav_label
                    ]

                    # Calculate bins
                    bins, weights, unc, band = uet.calc_bins(
                        input_array=jets_flavour,
                        Binning=Binning,
                    )

                    plt.hist(
                        x=bins[:-1],
                        bins=bins,
                        weights=weights,
                        histtype="step",
                        linewidth=1.0,
                        color=global_config.flavour_categories[flavour][
                            "colour"
                        ],
                        stacked=False,
                        fill=False,
                        label=global_config.flavour_categories[flavour][
                            "legend_label"
                        ],
                    )

                    if flavour == class_labels[-1]:
                        plt.hist(
                            x=bins[:-1],
                            bins=bins,
                            bottom=band,
                            weights=unc * 2,
                            label="stat. unc.",
                            **global_config.hist_err_style,
                        )

                    else:
                        plt.hist(
                            x=bins[:-1],
                            bins=bins,
                            bottom=band,
                            weights=unc * 2,
                            **global_config.hist_err_style,
                        )

                plt.xlabel(var)

                # Add axes, titels and the legend
                if Bin_Width_y_axis is True:
                    Bin_Width = abs(Binning[1] - Binning[0])
                    plt.ylabel(
                        ylabel + " / {:.2f}".format(Bin_Width),
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=ycolor,
                    )

                else:
                    plt.ylabel(
                        ylabel,
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=ycolor,
                    )

                if Log is True:
                    plt.yscale("log")
                    ymin, ymax = plt.ylim()
                    plt.ylim(ymin=0.01 * ymin, ymax=yAxisIncrease * ymax)

                else:
                    ymin, ymax = plt.ylim()
                    plt.ylim(ymin=0.8 * ymin, ymax=yAxisIncrease * ymax)

                plt.legend(loc="upper right", fontsize=legFontSize, ncol=ncol)
                plt.tight_layout()

                ax = plt.gca()
                if UseAtlasTag is True:
                    makeATLAStag(
                        ax=ax,
                        fig=fig,
                        first_tag=AtlasTag,
                        second_tag=SecondTag,
                        ymax=yAxisAtlasTag,
                    )

                plt.savefig(f"{filedir}/{var}.{plot_type}")
                plt.close()
                plt.clf()


def plot_input_vars_jets_comparison(
    datasets_filepaths,
    datasets_labels,
    var_dict,
    nJets,
    binning,
    class_labels,
    special_param_jets=None,
    plot_type="pdf",
    UseAtlasTag=True,
    ApplyATLASStyle=False,
    AtlasTag="Internal Simulation",
    SecondTag="$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
    yAxisAtlasTag=0.925,
    yAxisIncrease=10,
    output_directory="input_vars_jets",
    figsize=None,
    Log=True,
    ylabel="Normalised Number of Tracks",
    ycolor="black",
    legFontSize=10,
    ncol=2,
    Ratio_Cut=None,
    Bin_Width_y_axis=True,
):
    """
    Plotting the jet variable comparison for the given datasets.

    Input:
    - datasets_filepaths: List of filepaths to the files.
    - datasets_labels: Label of the dataset for the legend.
    - var_dict: Variable dict where all variables of the files are saved.
    - nJets: Number of jets to use for plotting.
    - binning: Decide which binning is used.
    - class_labels: List of class_labels which are to be plotted.
    - special_param_jets: Give specific x-axis-limits for variable.
    - plot_type: Plottype, like pdf or png
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - output_directory: Name of the output directory. Only the dir name not path!
    - figsize: List of the figure size. i.e [5, 6]
    - Log: Set y-axis log True or False.
    - ylabel: Y-label.
    - ycolor: Y-axis-label colour.
    - legFontSize: Legend font size
    - ncol: Number of columns of the legend.
    - Ratio_Cut: List of y-axis cuts for the ratio block.
    - Bin_Width_y_axis: Show bin size on y-axis

    Output:
    - Jet variable comparison plot for the given datasets
    """

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if type(binning[variable]) is list:
            nBins_dict.update(
                {
                    variable: np.arange(
                        binning[variable][0],
                        binning[variable][1],
                        binning[variable][2],
                    )
                }
            )

        elif type(binning[variable]) is None:
            nBins_dict.update({variable: int(100)})

        else:
            nBins_dict.update({variable: binning[variable]})

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    jets_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label) in zip(
        datasets_filepaths,
        datasets_labels,
    ):
        # Init jet counter
        nJets_counter = 0

        # Get wildcard
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(
            sorted(filepath, key=natural_keys)
        ):
            if nJets_counter < nJets:
                tmp_jets, tmp_flavour_labels = utt.LoadJetsFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    print_logger=False,
                )

                if file_counter == 0:
                    # Append to array
                    jets = tmp_jets
                    flavour_labels = tmp_flavour_labels

                else:
                    # Append to array
                    jets = jets.append(tmp_jets, ignore_index=True)
                    flavour_labels = np.append(
                        flavour_labels,
                        tmp_flavour_labels,
                    )

                # Add number of jets to counter
                nJets_counter += len(tmp_jets)

            else:
                break

        if len(jets) < nJets:
            n_jets = len(jets)
            logger.warning(
                f"{nJets} were set to be used, but only {n_jets} are available for {label} files!"
            )

        # Append jets to dict
        jets_dict.update({label: jets[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Loading jet variables
    jetsVars = [
        i
        for j in variable_config["train_variables"]
        for i in variable_config["train_variables"][j]
    ]

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")
    filedir = f"{output_directory}/"

    # Loop over vars
    for var in jetsVars:
        if var in nBins_dict:
            logger.info(f"Plotting {var}...")

            # Define the figure with two subplots of unequal sizes
            axis_dict = {}

            # Apply ATLAS style if true
            if ApplyATLASStyle:
                applyATLASstyle(mtp)

            # Set up new figure
            if figsize is None:
                fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

            else:
                fig = plt.figure(figsize=(figsize[0], figsize[1]))

            gs = gridspec.GridSpec(8, 1, figure=fig)
            axis_dict["left"] = {}
            axis_dict["left"]["top"] = fig.add_subplot(gs[:6, 0])
            axis_dict["left"]["ratio"] = fig.add_subplot(
                gs[6:, 0], sharex=axis_dict["left"]["top"]
            )

            # Init bincounts for ratio calculation
            bincounts = {}
            bincounts_unc = {}

            # Init lowest bincount
            Lowest_histcount = 1

            # Iterate over models
            for model_number, (label, linestyle) in enumerate(
                zip(datasets_labels, linestyles[: len(datasets_labels)])
            ):
                # Get variable and the labels of the jets
                jets_var = jets_dict[label][var]
                flavour_labels_var = flavour_label_dict[label]

                # Clean both from nans
                jets_var_clean = jets_var[~np.isnan(jets_var)]
                flavour_label_clean = flavour_labels_var[~np.isnan(jets_var)]

                if model_number == 0:
                    # Calculate unified Binning
                    first_flav = jets_var_clean[flavour_label_clean == 0]

                    var_range = None
                    if (
                        special_param_jets is not None
                        and var in special_param_jets
                    ):
                        if (
                            "lim_left" in special_param_jets[var]
                            and "lim_right" in special_param_jets[var]
                        ):
                            lim_left = special_param_jets[var]["lim_left"]
                            lim_right = special_param_jets[var]["lim_right"]
                            var_range = (lim_left, lim_right)

                    if nBins_dict[var] is None:
                        _, Binning = np.histogram(first_flav, range=var_range)

                    else:
                        _, Binning = np.histogram(
                            first_flav,
                            bins=nBins_dict[var],
                            range=var_range,
                        )

                for flav_label, flavour in enumerate(class_labels):
                    jets_flavour = jets_var_clean[
                        flavour_label_clean == flav_label
                    ]

                    # Calculate bins
                    bins, weights, unc, band = uet.calc_bins(
                        input_array=jets_flavour,
                        Binning=Binning,
                    )

                    hist_counts, _, _ = axis_dict["left"]["top"].hist(
                        x=bins[:-1],
                        bins=bins,
                        weights=weights,
                        histtype="step",
                        linewidth=1.0,
                        linestyle=linestyle,
                        color=global_config.flavour_categories[flavour][
                            "colour"
                        ],
                        stacked=False,
                        fill=False,
                        label=global_config.flavour_categories[flavour][
                            "legend_label"
                        ]
                        + f" {label}",
                    )

                    if flavour == class_labels[-1] and model_number == 0:
                        axis_dict["left"]["top"].hist(
                            x=bins[:-1],
                            bins=bins,
                            bottom=band,
                            weights=unc * 2,
                            label="stat. unc.",
                            **global_config.hist_err_style,
                        )

                    else:
                        axis_dict["left"]["top"].hist(
                            x=bins[:-1],
                            bins=bins,
                            bottom=band,
                            weights=unc * 2,
                            **global_config.hist_err_style,
                        )

                    bincounts.update({f"{flavour}{model_number}": hist_counts})
                    bincounts_unc.update({f"{flavour}{model_number}": unc})

                    for count in hist_counts:
                        if count != 0 and count < Lowest_histcount:
                            Lowest_histcount = count

                # Start ratio plot
                if model_number != 0:
                    for flavour in class_labels:
                        step, step_unc = uet.calc_ratio(
                            counter=bincounts[
                                "{}{}".format(flavour, model_number)
                            ],
                            denominator=bincounts["{}{}".format(flavour, 0)],
                            counter_unc=bincounts_unc[
                                "{}{}".format(flavour, model_number)
                            ],
                            denominator_unc=bincounts_unc[
                                "{}{}".format(flavour, 0)
                            ],
                        )

                        axis_dict["left"]["ratio"].step(
                            x=Binning,
                            y=step,
                            color=global_config.flavour_categories[flavour][
                                "colour"
                            ],
                            linestyle=linestyles[model_number],
                        )

                        axis_dict["left"]["ratio"].fill_between(
                            x=Binning,
                            y1=step - step_unc,
                            y2=step + step_unc,
                            step="pre",
                            facecolor="none",
                            edgecolor=global_config.hist_err_style[
                                "edgecolor"
                            ],
                            linewidth=global_config.hist_err_style[
                                "linewidth"
                            ],
                            hatch=global_config.hist_err_style["hatch"],
                        )

                elif model_number == 0:
                    # Add black line at one
                    axis_dict["left"]["ratio"].axhline(
                        y=1,
                        xmin=axis_dict["left"]["ratio"].get_xlim()[0],
                        xmax=axis_dict["left"]["ratio"].get_xlim()[1],
                        color="black",
                        alpha=0.5,
                    )

            axis_dict["left"]["ratio"].set_xlim(
                left=Binning[0],
                right=Binning[-1],
            )

            # Add axes, titels and the legend
            if Bin_Width_y_axis is True:
                Bin_Width = abs(Binning[1] - Binning[0])
                axis_dict["left"]["top"].set_ylabel(
                    ylabel + " / {:.2f}".format(Bin_Width),
                    fontsize=12,
                    horizontalalignment="right",
                    y=1.0,
                    color=ycolor,
                )

            else:
                axis_dict["left"]["top"].set_ylabel(
                    ylabel,
                    fontsize=12,
                    horizontalalignment="right",
                    y=1.0,
                    color=ycolor,
                )

            axis_dict["left"]["top"].tick_params(axis="y", labelcolor=ycolor)

            axis_dict["left"]["ratio"].set_xlabel(
                var, fontsize=12, horizontalalignment="right", x=1.0
            )

            plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

            if Log is True:
                axis_dict["left"]["top"].set_yscale("log")

                if axis_dict["left"]["top"].get_ylim()[0] <= 0:
                    # Set lower y limit
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=Lowest_histcount * 0.1,
                        top=left_y_limits[1] * 10 ** (yAxisIncrease),
                    )

                else:
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=left_y_limits[0] * 0.1,
                        top=left_y_limits[1] * 10 ** (yAxisIncrease),
                    )

            else:
                left_y_limits = axis_dict["left"]["top"].get_ylim()
                axis_dict["left"]["top"].set_ylim(
                    bottom=left_y_limits[0],
                    top=left_y_limits[1] * yAxisIncrease,
                )

            if Ratio_Cut is not None:
                axis_dict["left"]["ratio"].set_ylim(
                    bottom=Ratio_Cut[0], top=Ratio_Cut[1]
                )

            # Set axis
            axis_dict["left"]["top"].legend(
                loc="upper right",
                fontsize=legFontSize,
                ncol=ncol,
            )

            # Set tight layout
            plt.tight_layout()

            # Set ATLAS Tag
            if UseAtlasTag is True:
                makeATLAStag(
                    ax=axis_dict["left"]["top"],
                    fig=fig,
                    first_tag=AtlasTag,
                    second_tag=SecondTag,
                    ymax=yAxisAtlasTag,
                )

            # Save and close figure
            plt.savefig(f"{filedir}/{var}.{plot_type}")
            plt.close()
            plt.clf()
    logger.info(
        "\n-------------------------------------------------------------------------------"
    )
