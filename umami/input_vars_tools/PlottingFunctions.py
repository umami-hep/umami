#!/usr/bin/env python

"""Plots the given input variables of the given files and also a comparison."""

import os
from glob import glob

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import umami.data_tools as udt
from umami.configuration import global_config, logger
from umami.helper_tools import hist_ratio, hist_w_unc
from umami.preprocessing_tools import GetVariableDict
from umami.tools import applyATLASstyle, makeATLAStag, natural_keys


def check_kwargs_var_plots(kwargs: dict, **custom_default):
    """
    Validate the kwargs for plotting functions from **kwargs in function definition.

    Parameters
    ----------
    kwargs: dict
        kwargs dictionary passed to the plotting functions
        - plot_type : str
            Plottype, like pdf or png
        - UseAtlasTag : bool
            Define if ATLAS Tag is used or not.
        - ApplyATLASStyle : bool
            Apply ATLAS Style of the plot (for approval etc.).
        - AtlasTag : str
            Main tag. Mainly "Internal Simulation".
        - SecondTag : str
            Lower tag in the ATLAS label with infos.
        - yAxisAtlasTag : float
            Y axis position of the ATLAS label.
        - yAxisIncrease : float
            Y axis increase factor to fit the ATLAS label.
        - figsize : list
            List of the figure size. i.e [5, 6]
        - Log : bool
            Set y-axis log True or False.
        - ylabel : str
            Y-label.
        - ycolor : str
            Y-axis-label colour.
        - legFontSize : int
            Legend font size.
        - ncol : int
            Number of columns of the legend.
    **custom_default: dict
        overwrites the default values defined in this function

    Returns
    -------
    kwargs: dict
        kwargs, replaced with custom default values if needed
    """
    # the following kwargs all plotting functions have in common
    default_kwargs = {
        "UseAtlasTag": True,
        "ApplyATLASStyle": False,
        "AtlasTag": "Internal Simulation",
        "SecondTag": "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
        "yAxisAtlasTag": 0.925,
        "yAxisIncrease": 1,
        "figsize": None,
        "Log": True,
        "ylabel": "Number of Tracks",
        "ycolor": "black",
        "legFontSize": 10,
        "ncol": 2,
        "Bin_Width_y_axis": True,
        "plot_type": "pdf",
    }

    updated_kwargs = {}
    for key, value in default_kwargs.items():
        if key in kwargs.keys():
            updated_kwargs[key] = kwargs[key]
        elif key in custom_default.keys():  # pylint: disable=C0201
            updated_kwargs[key] = custom_default[key]
        else:
            updated_kwargs[key] = value

    # Check for y axis scale
    if updated_kwargs["Log"] is True:
        updated_kwargs["ylabel"] = "Normalised " + updated_kwargs["ylabel"]

    return updated_kwargs


def plot_nTracks_per_Jet(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    nJets: int,
    class_labels: list,
    output_directory: str = "input_vars_trks",
    Ratio_Cut: list = None,
    track_origin: str = "All",
    **kwargs,
):
    """
    Plot the number of tracks per jet as a histogram.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    datasets_track_names : list
        List with the track names of the files.
    nJets : int
        Number of jets to use.
    class_labels : list
        List of classes that are to be plotted.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    Ratio_Cut : list
        List of y-axis cuts for the ratio block.
    track_origin : str
        Track set that is to be used for plotting.
    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`
    """
    # check the kwargs
    kwargs = check_kwargs_var_plots(kwargs)
    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label, tracks_name) in zip(
        datasets_filepaths, datasets_labels, datasets_track_names
    ):
        # Init jet counter
        nJets_counter = 0

        # Get the filepath of the dataset
        filepath = glob(filepath)

        # Loop over files and get the amount of jets needed.
        for file_counter, file in enumerate(sorted(filepath, key=natural_keys)):
            if nJets_counter < nJets:
                tmp_trks, tmp_flavour_labels = udt.LoadTrksFromFile(
                    filepath=file,
                    class_labels=class_labels,
                    nJets=nJets,
                    tracks_name=tracks_name,
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
                f"{nJets} were set to be used, but only {n_trks} are available"
                f" for {label} files!"
            )

        # Append trks to dict
        trks_dict.update({label: trks[:nJets]})
        flavour_label_dict.update({label: flavour_labels[:nJets]})

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")

    logger.info(f"Path: {output_directory}")
    logger.info(f"Track origin: {track_origin}\n")

    # Define the figure with two subplots of unequal sizes
    axis_dict = {}

    # Apply ATLAS style if true
    if kwargs["ApplyATLASStyle"]:
        applyATLASstyle(mtp)

    # Set up new figure
    if not kwargs["figsize"]:
        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

    else:
        fig = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))

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
        if track_origin == "All":
            nTracks = np.sum(~np.isnan(trks["ptfrac"]), axis=1)
        else:
            nTracks = np.sum(
                np.logical_and(
                    ~np.isnan(trks["ptfrac"]),
                    trks["truthOriginLabel"] == global_config.OriginType[track_origin],
                ),
                axis=1,
            )

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
            bins, weights, unc, band = hist_w_unc(
                a=nTracks_flavour,
                bins=Binning,
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
                step, step_unc = hist_ratio(
                    numerator=bincounts[f"{flavour}{model_number}"],
                    denominator=bincounts[f"{flavour}0"],
                    numerator_unc=bincounts_unc[f"{flavour}{model_number}"],
                    denominator_unc=bincounts_unc[f"{flavour}0"],
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
    if kwargs["Bin_Width_y_axis"] is True:
        Bin_Width = abs(Binning[1] - Binning[0])
        axis_dict["left"]["top"].set_ylabel(
            f"{kwargs['ylabel']} / {Bin_Width:.2f}",
            fontsize=12,
            horizontalalignment="right",
            y=1.0,
            color=kwargs["ycolor"],
        )

    else:
        axis_dict["left"]["top"].set_ylabel(
            kwargs["ylabel"],
            fontsize=12,
            horizontalalignment="right",
            y=1.0,
            color=kwargs["ycolor"],
        )

    axis_dict["left"]["top"].tick_params(axis="y", labelcolor=kwargs["ycolor"])

    axis_dict["left"]["ratio"].set_xlabel(
        "Number of tracks per Jet"
        if track_origin == "All"
        else f"Number of tracks per Jet ({track_origin})",
        fontsize=12,
        horizontalalignment="right",
        x=1.0,
    )

    plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

    if kwargs["Log"] is True:
        axis_dict["left"]["top"].set_yscale("log")

        if axis_dict["left"]["top"].get_ylim()[0] <= 0:
            # Set lower y limit
            left_y_limits = axis_dict["left"]["top"].get_ylim()
            axis_dict["left"]["top"].set_ylim(
                bottom=Lowest_histcount * 0.1,
                top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
            )

        else:
            left_y_limits = axis_dict["left"]["top"].get_ylim()
            axis_dict["left"]["top"].set_ylim(
                bottom=left_y_limits[0] * 0.1,
                top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
            )

    else:
        left_y_limits = axis_dict["left"]["top"].get_ylim()
        axis_dict["left"]["top"].set_ylim(
            bottom=left_y_limits[0],
            top=left_y_limits[1] * kwargs["yAxisIncrease"],
        )

    if Ratio_Cut is not None:
        axis_dict["left"]["ratio"].set_ylim(bottom=Ratio_Cut[0], top=Ratio_Cut[1])

    # Set axis
    axis_dict["left"]["top"].legend(
        loc="upper right",
        fontsize=kwargs["legFontSize"],
        ncol=kwargs["ncol"],
    )

    # Set tight layout
    plt.tight_layout()

    # Set ATLAS Tag
    if kwargs["UseAtlasTag"] is True:
        makeATLAStag(
            ax=axis_dict["left"]["top"],
            fig=fig,
            first_tag=kwargs["AtlasTag"],
            second_tag=kwargs["SecondTag"],
            ymax=kwargs["yAxisAtlasTag"],
        )

    # Save and close figure
    plt.savefig(
        f"{output_directory}/nTracks_per_Jet_{track_origin}.{kwargs['plot_type']}"
    )
    plt.close()
    plt.clf()


def plot_input_vars_trks_comparison(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    var_dict: dict,
    nJets: int,
    binning: dict,
    class_labels: list,
    sorting_variable: str = "ptfrac",
    n_Leading: list = None,
    output_directory: str = "input_vars_trks",
    Ratio_Cut: list = None,
    track_origin: str = "All",
    **kwargs,
):
    """
    Plot the track variable in comparison to another model with ratio plot.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    datasets_track_names : list
        List with the track names of the files.
    var_dict : dict
        Variable dict where all variables of the files are saved.
    nJets : int
        Number of jets to use for plotting.
    binning : dict
        Decide which binning is used.
    class_labels : list
        List of class_labels which are to be plotted.
    sorting_variable : str
        Variable which is used for sorting.
    n_Leading : list
        n-th leading jet which is plotted. For all, = None.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    Ratio_Cut : list
        List of y-axis cuts for the ratio block.
    track_origin: str
        Track set that is to be used for plotting.
    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`

    Raises
    ------
    ValueError
        If the type of the given binning is not supported.
    """
    # check the kwargs
    kwargs = check_kwargs_var_plots(kwargs)

    # check to avoid dangerous default value (list)
    if n_Leading is None:
        n_Leading = [None]
    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
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

        # If int, set to the given numbers
        elif isinstance(binning[variable], int):
            nBins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            nBins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for filepath, label, tracks_name in zip(
        datasets_filepaths,
        datasets_labels,
        datasets_track_names,
    ):

        # Get the tracks and the labels from the file/files
        trks, flavour_labels = udt.LoadTrksFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=nJets,
            tracks_name=tracks_name,
            print_logger=False,
        )

        # Append trks to dict
        trks_dict.update({label: trks})
        flavour_label_dict.update({label: flavour_labels})

    # Load var dict
    variable_config = GetVariableDict(var_dict)

    # Loading track variables
    try:
        trksVars = variable_config["tracks"]

    except KeyError:
        noNormVars = variable_config["track_train_variables"][datasets_track_names[0]][
            "noNormVars"
        ]
        logNormVars = variable_config["track_train_variables"][datasets_track_names[0]][
            "logNormVars"
        ]
        jointNormVars = variable_config["track_train_variables"][
            datasets_track_names[0]
        ]["jointNormVars"]
        trksVars = noNormVars + logNormVars + jointNormVars

        # Check for variables in the other
        for counter, track_names in enumerate(datasets_track_names):
            if counter != 0:
                noNormVars_tmp = variable_config["track_train_variables"][track_names][
                    "noNormVars"
                ]
                logNormVars_tmp = variable_config["track_train_variables"][track_names][
                    "logNormVars"
                ]
                jointNormVars_tmp = variable_config["track_train_variables"][
                    track_names
                ]["jointNormVars"]
                trksVars_tmp = noNormVars_tmp + logNormVars_tmp + jointNormVars_tmp

                for iter_var in trksVars_tmp:
                    if iter_var not in trksVars:
                        logger.warning(
                            f"Variable {iter_var} of {datasets_labels[counter]} "
                            f"not in {datasets_labels[0]} track collection. "
                            "Skipping..."
                        )

    for nLeading in n_Leading:
        if nLeading == "None":
            nLeading = None

        # Check if path is existing, if not mkdir
        if nLeading is None:
            filedir = os.path.join(output_directory, sorting_variable, "All")

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        else:
            filedir = os.path.join(output_directory, sorting_variable, str(nLeading))
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        logger.info(f"Path: {filedir}")
        logger.info(f"Sorting: {sorting_variable}")
        logger.info(f"nLeading track: {nLeading}")
        logger.info(f"Track origin: {track_origin}\n")

        # Loop over vars
        for var in trksVars:
            if var in nBins_dict:
                logger.info(f"Plotting {var}...")

                # Define the figure with two subplots of unequal sizes
                axis_dict = {}

                # Apply ATLAS style if true
                if kwargs["ApplyATLASStyle"]:
                    applyATLASstyle(mtp)

                # Set up new figure
                if kwargs["figsize"] is None:
                    fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                else:
                    fig = plt.figure(
                        figsize=(kwargs["figsize"][0], kwargs["figsize"][1])
                    )

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
                    sorting = np.argsort(-1 * trks_dict[label][sorting_variable])

                    # Sort the variables and tracks after given variable
                    if track_origin == "All":
                        tmp = np.asarray(
                            [
                                trks_dict[label][var][k][sorting[k]]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )
                    else:
                        # Select tracks of a given origin, so keep truthOriginLabel
                        tmp = np.asarray(
                            [
                                trks_dict[label][[var, "truthOriginLabel"]][k][
                                    sorting[k]
                                ]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )

                    if model_number == 0:
                        # Calculate unified Binning
                        if track_origin == "All":
                            first_flav = tmp[flavour_label_dict[label] == 0]
                        else:
                            first_flav = tmp[var][flavour_label_dict[label] == 0]

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
                        if track_origin == "All":
                            Tracks = jets[:, nLeading][~np.isnan(jets[:, nLeading])]
                        else:
                            mask_nan = ~np.isnan(jets[var][:, nLeading])
                            mask_origin = np.asarray(
                                jets[:, nLeading]["truthOriginLabel"]
                                == global_config.OriginType[track_origin]
                            )
                            Tracks = jets[:, nLeading][
                                np.logical_and(mask_nan, mask_origin)
                            ][var]

                        # Calculate bins
                        bins, weights, unc, band = hist_w_unc(
                            a=Tracks,
                            bins=Binning,
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
                            step, step_unc = hist_ratio(
                                numerator=bincounts[f"{flavour}{model_number}"],
                                denominator=bincounts[f"{flavour}0"],
                                numerator_unc=bincounts_unc[f"{flavour}{model_number}"],
                                denominator_unc=bincounts_unc[f"{flavour}0"],
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
                if kwargs["Bin_Width_y_axis"] is True:
                    Bin_Width = abs(Binning[1] - Binning[0])
                    axis_dict["left"]["top"].set_ylabel(
                        f"{kwargs['ylabel']} / {Bin_Width:.2f}",
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=kwargs["ycolor"],
                    )

                else:
                    axis_dict["left"]["top"].set_ylabel(
                        kwargs["ylabel"],
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=kwargs["ycolor"],
                    )

                axis_dict["left"]["top"].tick_params(
                    axis="y", labelcolor=kwargs["ycolor"]
                )

                if nLeading is None:
                    axis_dict["left"]["ratio"].set_xlabel(
                        f"{var}"
                        if track_origin == "All"
                        else f"{var} ({track_origin})",
                        fontsize=12,
                        horizontalalignment="right",
                        x=1.0,
                    )

                else:
                    axis_dict["left"]["ratio"].set_xlabel(
                        f"{nLeading+1} leading tracks {var}"
                        if track_origin == "All"
                        else (f"{nLeading+1} leading tracks {var} ({track_origin})"),
                        fontsize=12,
                        horizontalalignment="right",
                        x=1.0,
                    )

                plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

                if kwargs["Log"] is True:
                    axis_dict["left"]["top"].set_yscale("log")

                    if axis_dict["left"]["top"].get_ylim()[0] <= 0:
                        # Set lower y limit
                        left_y_limits = axis_dict["left"]["top"].get_ylim()
                        axis_dict["left"]["top"].set_ylim(
                            bottom=Lowest_histcount * 0.1,
                            top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
                        )

                    else:
                        left_y_limits = axis_dict["left"]["top"].get_ylim()
                        axis_dict["left"]["top"].set_ylim(
                            bottom=left_y_limits[0] * 0.1,
                            top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
                        )

                else:
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=left_y_limits[0],
                        top=left_y_limits[1] * kwargs["yAxisIncrease"],
                    )

                if Ratio_Cut is not None:
                    axis_dict["left"]["ratio"].set_ylim(
                        bottom=Ratio_Cut[0], top=Ratio_Cut[1]
                    )

                # Set axis
                axis_dict["left"]["top"].legend(
                    loc="upper right",
                    fontsize=kwargs["legFontSize"],
                    ncol=kwargs["ncol"],
                )

                # Set tight layout
                plt.tight_layout()

                # Set ATLAS Tag
                if kwargs["UseAtlasTag"] is True:
                    makeATLAStag(
                        ax=axis_dict["left"]["top"],
                        fig=fig,
                        first_tag=kwargs["AtlasTag"],
                        second_tag=kwargs["SecondTag"],
                        ymax=kwargs["yAxisAtlasTag"],
                    )

                # Save and close figure
                plt.savefig(
                    f"{filedir}/{var}_{nLeading}_{track_origin}.{kwargs['plot_type']}"
                )
                plt.close()
                plt.clf()
        logger.info(
            "\n-----------------------------------------------------------------------"
            "--------"
        )


def plot_input_vars_trks(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    var_dict: dict,
    nJets: int,
    binning: dict,
    class_labels: list,
    sorting_variable: str = "ptfrac",
    n_Leading: list = None,
    output_directory: str = "input_vars_trks",
    track_origin: str = "All",
    **kwargs,
):
    """
    Plot the track variable.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    datasets_track_names : list
        List with the track names of the files.
    var_dict : dict
        Variable dict where all variables of the files are saved.
    nJets : int
        Number of jets to use for plotting.
    binning : dict
        Decide which binning is used.
    class_labels : list
        List of class_labels which are to be plotted.
    sorting_variable : str
        Variable which is used for sorting.
    n_Leading : list
        n-th leading jet which is plotted. For all, = None.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    track_origin: str
        Track set that is to be used for plotting.
    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`

    Raises
    ------
    ValueError
        If the type of the given binning is not supported.
    """
    # check the kwargs
    kwargs = check_kwargs_var_plots(kwargs, yAxisIncrease=10)

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
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

        # If int, set to the given numbers
        elif isinstance(binning[variable], int):
            nBins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            nBins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label, tracks_name) in zip(
        datasets_filepaths,
        datasets_labels,
        datasets_track_names,
    ):

        # Get the tracks and the labels from the file/files
        trks, flavour_labels = udt.LoadTrksFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=nJets,
            tracks_name=tracks_name,
            print_logger=False,
        )

        # Append trks to dict
        trks_dict.update({label: trks})
        flavour_label_dict.update({label: flavour_labels})

    # Load var dict
    variable_config = GetVariableDict(var_dict)

    # Loading track variables
    noNormVars = variable_config["track_train_variables"][datasets_track_names[0]][
        "noNormVars"
    ]
    logNormVars = variable_config["track_train_variables"][datasets_track_names[0]][
        "logNormVars"
    ]
    jointNormVars = variable_config["track_train_variables"][datasets_track_names[0]][
        "jointNormVars"
    ]
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
            filedir = os.path.join(output_directory, sorting_variable, str(nLeading))
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        logger.info(f"Path: {filedir}")
        logger.info(f"Sorting: {sorting_variable}")
        logger.info(f"nLeading track: {nLeading}")
        logger.info(f"Track origin: {track_origin}\n")

        for var in nBins_dict:
            if var not in trksVars:
                logger.info(f"{var} in config, but not in Variables yaml! Skipping...")

        # Loop over vars
        for var in trksVars:
            if var not in nBins_dict:
                logger.info(f"{var} in Variables yaml but not in config! Skipping...")

            else:
                logger.info(f"Plotting {var}...")

                # Iterate over models
                for (label, linestyle) in zip(
                    datasets_labels, linestyles[: len(datasets_labels)]
                ):
                    # Sort after given variable
                    sorting = np.argsort(-1 * trks_dict[label][sorting_variable])

                    # Sort the variables and tracks after given variable
                    if track_origin == "All":
                        tmp = np.asarray(
                            [
                                trks_dict[label][var][k][sorting[k]]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )
                    else:
                        # Select tracks of a given origin, so keep truthOriginLabel
                        tmp = np.asarray(
                            [
                                trks_dict[label][[var, "truthOriginLabel"]][k][
                                    sorting[k]
                                ]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )

                    # Calculate unified Binning
                    if track_origin == "All":
                        first_flav = tmp[flavour_label_dict[label] == 0]
                    else:
                        first_flav = tmp[var][flavour_label_dict[label] == 0]

                    # Check if binning is already set
                    if nBins_dict[var] is None:

                        # Get Binning
                        _, Binning = np.histogram(
                            first_flav[:, nLeading][~np.isnan(first_flav[:, nLeading])]
                        )

                    else:
                        # Get Binning
                        _, Binning = np.histogram(
                            first_flav[:, nLeading][~np.isnan(first_flav[:, nLeading])],
                            bins=nBins_dict[var],
                        )

                    # Apply ATLAS style if true
                    if kwargs["ApplyATLASStyle"]:
                        applyATLASstyle(mtp)

                    # Set up new figure
                    if kwargs["figsize"] is None:
                        fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                    else:
                        fig = plt.figure(
                            figsize=(kwargs["figsize"][0], kwargs["figsize"][1])
                        )

                    # Iterate over flavours
                    for flav_label, flavour in enumerate(class_labels):

                        # Get all jets with wanted flavour
                        jets = tmp[flavour_label_dict[label] == flav_label]

                        # Get number of tracks
                        if track_origin == "All":
                            Tracks = jets[:, nLeading][~np.isnan(jets[:, nLeading])]
                        else:
                            mask_nan = ~np.isnan(jets[var][:, nLeading])
                            mask_origin = np.asarray(
                                jets[:, nLeading]["truthOriginLabel"]
                                == global_config.OriginType[track_origin]
                            )
                            Tracks = jets[:, nLeading][
                                np.logical_and(mask_nan, mask_origin)
                            ][var]

                        # Calculate bins
                        bins, weights, unc, band = hist_w_unc(
                            a=Tracks,
                            bins=Binning,
                        )

                        plt.hist(
                            x=bins[:-1],
                            bins=bins,
                            weights=weights,
                            histtype="step",
                            linewidth=1.0,
                            linestyle=linestyle,
                            color=global_config.flavour_categories[flavour]["colour"],
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
                        plt.xlabel(
                            f"{var}"
                            if track_origin == "All"
                            else f"{var} ({track_origin})"
                        )

                    else:
                        plt.xlabel(
                            f"{nLeading+1} leading tracks {var}"
                            if track_origin == "All"
                            else (f"{nLeading+1} leading tracks {var} ({track_origin})")
                        )

                    # Add axes, titels and the legend
                    if kwargs["Bin_Width_y_axis"] is True:
                        Bin_Width = abs(Binning[1] - Binning[0])
                        plt.ylabel(
                            f"{kwargs['ylabel']} / {Bin_Width:.2f}",
                            fontsize=12,
                            horizontalalignment="right",
                            y=1.0,
                            color=kwargs["ycolor"],
                        )

                    else:
                        plt.ylabel(
                            kwargs["ylabel"],
                            fontsize=12,
                            horizontalalignment="right",
                            y=1.0,
                            color=kwargs["ycolor"],
                        )

                    if kwargs["Log"] is True:
                        plt.yscale("log")
                        ymin, ymax = plt.ylim()
                        plt.ylim(ymin=0.01 * ymin, ymax=kwargs["yAxisIncrease"] * ymax)

                    else:
                        ymin, ymax = plt.ylim()
                        plt.ylim(ymin=0.8 * ymin, ymax=kwargs["yAxisIncrease"] * ymax)

                    plt.legend(
                        loc="upper right",
                        fontsize=kwargs["legFontSize"],
                        ncol=kwargs["ncol"],
                    )
                    plt.tight_layout()

                    ax = plt.gca()
                    if kwargs["UseAtlasTag"] is True:
                        makeATLAStag(
                            ax=ax,
                            fig=fig,
                            first_tag=kwargs["AtlasTag"],
                            second_tag=kwargs["SecondTag"],
                            ymax=kwargs["yAxisAtlasTag"],
                        )

                    plt.savefig(
                        f"{filedir}/{var}_{nLeading}_{track_origin}."
                        f"{kwargs['plot_type']}"
                    )
                    plt.close()
                    plt.clf()
                    logger.info(f"{filedir}/{var}.{kwargs['plot_type']}\n")


def plot_input_vars_jets(
    datasets_filepaths: list,
    datasets_labels: list,
    var_dict: dict,
    nJets: int,
    binning: dict,
    class_labels: list,
    special_param_jets: dict = None,
    output_directory: str = "input_vars_jets",
    normalise: bool = True,
    **kwargs,
):
    """
    Plot the jet variable.

    Parameters
    ----------
    datasets_filepaths: list
        List of filepaths to the files.
    datasets_labels: list
        Label of the dataset for the legend.
    var_dict: dict
        Variable dict where all variables of the files are saved.
    nJets: int
        Number of jets to use for plotting.
    binning: dict
        Decide which binning is used.
    class_labels: list
        List of class_labels which are to be plotted.
    special_param_jets: dict
        Give specific x-axis-limits for variable.
    output_directory: str
        Name of the output directory. Only the dir name not path!
    normalise: bool, optional
        Bool to specify if distributions are normalised, by default True.
    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`

    Raises
    ------
    ValueError
        If the type of the given binning is not supported.
    """

    # Set the ylabel to jets
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Number of Jets"

    # check the kwargs
    kwargs = check_kwargs_var_plots(kwargs, yAxisIncrease=10)

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
            nBins_dict.update(
                {
                    variable: np.arange(
                        binning[variable][0],
                        binning[variable][1],
                        binning[variable][2],
                    )
                }
            )

        # If int, set to the given numbers
        elif isinstance(binning[variable], int):
            nBins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            nBins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

    variable_config = GetVariableDict(var_dict)

    # Init trks and flavour label dicts
    jets_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels
    # defined in the config
    for (filepath, label) in zip(
        datasets_filepaths,
        datasets_labels,
    ):

        # Get the tracks and the labels from the file/files
        jets, flavour_labels = udt.LoadJetsFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=nJets,
            print_logger=False,
        )

        # Append jets to dict
        jets_dict.update({label: jets})
        flavour_label_dict.update({label: flavour_labels})

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
                if special_param_jets is not None and var in special_param_jets:
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
                if kwargs["ApplyATLASStyle"]:
                    applyATLASstyle(mtp)

                # Set up new figure
                if kwargs["figsize"] is None:
                    fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

                else:
                    fig = plt.figure(
                        figsize=(kwargs["figsize"][0], kwargs["figsize"][1])
                    )

                for flav_label, flavour in enumerate(class_labels):
                    jets_flavour = jets_var_clean[flavour_label_clean == flav_label]

                    # Calculate bins
                    bins, weights, unc, band = hist_w_unc(
                        a=jets_flavour, bins=Binning, normed=normalise
                    )

                    plt.hist(
                        x=bins[:-1],
                        bins=bins,
                        weights=weights,
                        histtype="step",
                        linewidth=1.0,
                        color=global_config.flavour_categories[flavour]["colour"],
                        stacked=False,
                        fill=False,
                        label=global_config.flavour_categories[flavour]["legend_label"],
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
                if kwargs["Bin_Width_y_axis"] is True:
                    Bin_Width = abs(Binning[1] - Binning[0])
                    plt.ylabel(
                        f"{kwargs['ylabel']} / {Bin_Width:.2f}",
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=kwargs["ycolor"],
                    )

                else:
                    plt.ylabel(
                        kwargs["ylabel"],
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=kwargs["ycolor"],
                    )

                if normalise is False:
                    plt.ylabel(
                        "Number of jets",
                        fontsize=12,
                        horizontalalignment="right",
                        y=1.0,
                        color=kwargs["ycolor"],
                    )

                if kwargs["Log"] is True:
                    plt.yscale("log")
                    ymin, ymax = plt.ylim()
                    plt.ylim(ymin=0.01 * ymin, ymax=kwargs["yAxisIncrease"] * ymax)

                else:
                    ymin, ymax = plt.ylim()
                    plt.ylim(ymin=0.8 * ymin, ymax=kwargs["yAxisIncrease"] * ymax)

                plt.legend(
                    loc="upper right",
                    fontsize=kwargs["legFontSize"],
                    ncol=kwargs["ncol"],
                )
                plt.tight_layout()

                ax = plt.gca()
                if kwargs["UseAtlasTag"] is True:
                    makeATLAStag(
                        ax=ax,
                        fig=fig,
                        first_tag=kwargs["AtlasTag"],
                        second_tag=kwargs["SecondTag"],
                        ymax=kwargs["yAxisAtlasTag"],
                    )

                plt.savefig(f"{filedir}/{var}.{kwargs['plot_type']}")
                plt.close()
                plt.clf()


def plot_input_vars_jets_comparison(
    datasets_filepaths: list,
    datasets_labels: list,
    var_dict: dict,
    nJets: int,
    binning: dict,
    class_labels: list,
    special_param_jets: dict = None,
    output_directory: str = "input_vars_jets",
    Ratio_Cut: list = None,
    normalise: bool = True,
    **kwargs,
):
    """
    Plot the jet variable comparison for the given datasets.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    var_dict : dict
        Variable dict where all variables of the files are saved.
    nJets : int
        Number of jets to use for plotting.
    binning : dict
        Decide which binning is used.
    class_labels : list
        List of class_labels which are to be plotted.
    special_param_jets : dict
        Dict with special x-axis cuts for the given variable.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    Ratio_Cut : list
        List of y-axis cuts for the ratio block.
    normalise: bool, optional
        Bool to specify if distributions are normalised, by default True.

    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`

    Raises
    ------
    ValueError
        If the type of the given binning is not supported.
    """
    # Set the ylabel to jets
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Number of Jets"

    # check the kwargs
    kwargs = check_kwargs_var_plots(kwargs, yAxisIncrease=10)

    nBins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
            nBins_dict.update(
                {
                    variable: np.arange(
                        binning[variable][0],
                        binning[variable][1],
                        binning[variable][2],
                    )
                }
            )

        # If int, set to the given numbers
        elif isinstance(binning[variable], int):
            nBins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            nBins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

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

        # Get the tracks and the labels from the file/files
        jets, flavour_labels = udt.LoadJetsFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=nJets,
            print_logger=False,
        )

        # Append jets to dict
        jets_dict.update({label: jets})
        flavour_label_dict.update({label: flavour_labels})

    variable_config = GetVariableDict(var_dict)

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
            if kwargs["ApplyATLASStyle"]:
                applyATLASstyle(mtp)

            # Set up new figure
            if kwargs["figsize"] is None:
                fig = plt.figure(figsize=(11.69 * 0.8, 8.27 * 0.8))

            else:
                fig = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))

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
                    if special_param_jets is not None and var in special_param_jets:
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
                    jets_flavour = jets_var_clean[flavour_label_clean == flav_label]

                    # Calculate bins
                    bins, weights, unc, band = hist_w_unc(
                        a=jets_flavour, bins=Binning, normed=normalise
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
                        step, step_unc = hist_ratio(
                            numerator=bincounts[f"{flavour}{model_number}"],
                            denominator=bincounts[f"{flavour}0"],
                            numerator_unc=bincounts_unc[f"{flavour}{model_number}"],
                            denominator_unc=bincounts_unc[f"{flavour}0"],
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

            axis_dict["left"]["ratio"].set_ylabel("Ratio", fontsize=12)

            # Add axes, titels and the legend
            if kwargs["Bin_Width_y_axis"] is True:
                Bin_Width = abs(Binning[1] - Binning[0])
                axis_dict["left"]["top"].set_ylabel(
                    f"{kwargs['ylabel']} / {Bin_Width:.2f}",
                    fontsize=12,
                    horizontalalignment="right",
                    y=1.0,
                    color=kwargs["ycolor"],
                )

            else:
                axis_dict["left"]["top"].set_ylabel(
                    kwargs["ylabel"],
                    fontsize=12,
                    horizontalalignment="right",
                    y=1.0,
                    color=kwargs["ycolor"],
                )

            if normalise is False:
                axis_dict["left"]["top"].set_ylabel(
                    "Number of jets",
                    fontsize=12,
                    horizontalalignment="right",
                    y=1.0,
                    color=kwargs["ycolor"],
                )

            axis_dict["left"]["top"].tick_params(axis="y", labelcolor=kwargs["ycolor"])

            axis_dict["left"]["ratio"].set_xlabel(
                var, fontsize=12, horizontalalignment="right", x=1.0
            )

            plt.setp(axis_dict["left"]["top"].get_xticklabels(), visible=False)

            if kwargs["Log"] is True:
                axis_dict["left"]["top"].set_yscale("log")

                if axis_dict["left"]["top"].get_ylim()[0] <= 0:
                    # Set lower y limit
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=Lowest_histcount * 0.1,
                        top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
                    )

                else:
                    left_y_limits = axis_dict["left"]["top"].get_ylim()
                    axis_dict["left"]["top"].set_ylim(
                        bottom=left_y_limits[0] * 0.1,
                        top=left_y_limits[1] * 10 ** (kwargs["yAxisIncrease"]),
                    )

            else:
                left_y_limits = axis_dict["left"]["top"].get_ylim()
                axis_dict["left"]["top"].set_ylim(
                    bottom=left_y_limits[0],
                    top=left_y_limits[1] * kwargs["yAxisIncrease"],
                )

            if Ratio_Cut is not None:
                axis_dict["left"]["ratio"].set_ylim(
                    bottom=Ratio_Cut[0], top=Ratio_Cut[1]
                )

            # Set axis
            axis_dict["left"]["top"].legend(
                loc="upper right",
                fontsize=kwargs["legFontSize"],
                ncol=kwargs["ncol"],
            )

            # Set tight layout
            plt.tight_layout()

            # Set ATLAS Tag
            if kwargs["UseAtlasTag"] is True:
                makeATLAStag(
                    ax=axis_dict["left"]["top"],
                    fig=fig,
                    first_tag=kwargs["AtlasTag"],
                    second_tag=kwargs["SecondTag"],
                    ymax=kwargs["yAxisAtlasTag"],
                )

            # Save and close figure
            plt.savefig(f"{filedir}/{var}.{kwargs['plot_type']}")
            plt.close()
            plt.clf()
    logger.info(
        "\n------------------------------------------------------------------"
        "-------------"
    )
