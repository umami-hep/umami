#!/usr/bin/env python

"""Plots the given input variables of the given files and also a comparison."""

import os

import numpy as np

import umami.data_tools as udt
from umami.configuration import global_config, logger
from umami.plotting import histogram, histogram_plot
from umami.plotting.utils import translate_kwargs
from umami.preprocessing_tools import GetVariableDict


def check_kwargs_var_plots(kwargs: dict, **custom_default):
    """
    Validate the kwargs for plotting functions from **kwargs in function definition.

    Parameters
    ----------
    kwargs: dict
        kwargs dictionary passed to the plotting functions
        - use_atlas_tag : bool
            Define if ATLAS Tag is used or not.
        - apply_atlas_style : bool
            Apply ATLAS Style of the plot (for approval etc.).
        - atlas_first_tag : str
            Main tag. Mainly "Internal Simulation".
        - atlas_second_tag : str
            Lower tag in the ATLAS label with infos.
        - y_scale : float
            Y axis increase factor to fit the ATLAS label.
        - figsize : list
            List of the figure size. i.e [5, 6]
        - logy : bool
            Set y-axis log True or False.
        - ylabel : str
            Y-label.
        - leg_fontsize : int
            Legend font size.
        - leg_ncol : int
            Number of columns of the legend.
        - leg_loc : str
            Location of the legend (matplotlib conventions)
        - bin_width_in_ylabel : bool
            Option to specify the bin width in the ylabel, by default True
        - plot_type: str
            File format for the output, by default "pdf"
        - transparent: bool
            Option to make the background of the plot transparent, by default True
        - norm: bool
            Normalise histograms, by default True
    **custom_default: dict
        overwrites the default values defined in this function

    Returns
    -------
    kwargs: dict
        kwargs, replaced with custom default values if needed
    """
    # the following kwargs all plotting functions have in common
    default_kwargs = {
        "use_atlas_tag": True,
        "apply_atlas_style": True,
        "atlas_first_tag": "Internal Simulation",
        "atlas_second_tag": "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets",
        "y_scale": 1,
        "figsize": (6.8, 5),
        "logy": True,
        "ylabel": "Number of Tracks",
        "leg_fontsize": 10,
        "leg_ncol": 1,
        "leg_loc": "upper right",
        "bin_width_in_ylabel": True,
        "plot_type": "pdf",
        "transparent": True,
        "norm": True,
    }

    updated_kwargs = {}
    for key, value in default_kwargs.items():
        if key in kwargs.keys():
            updated_kwargs[key] = kwargs[key]
        elif key in custom_default.keys():  # pylint: disable=C0201
            updated_kwargs[key] = custom_default[key]
        else:
            updated_kwargs[key] = value

    # Add "Normalised" to ylabel in case it was forgotten
    if (
        updated_kwargs["norm"] is True
        and "norm" not in updated_kwargs["ylabel"].lower()
    ):
        updated_kwargs["ylabel"] = "Normalised " + updated_kwargs["ylabel"]

    return updated_kwargs


def plot_n_tracks_per_jet(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    n_jets: int,
    class_labels: list,
    output_directory: str = "input_vars_trks",
    ratio_cut: list = None,
    track_origin: str = "All",
    **kwargs,
):
    """
    Plot the number of tracks per jet as a histogram. If multiple datasets are
    provided, a ratio plot comparing each flavour individually will be added below
    the main plot.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    datasets_track_names : list
        List with the track names of the files.
    n_jets : int
        Number of jets to use.
    class_labels : list
        List of classes that are to be plotted.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    ratio_cut : list
        List of y-axis cuts for the ratio block.
    track_origin : str
        Track set that is to be used for plotting.
    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`
    """
    # check the kwargs
    kwargs = translate_kwargs(kwargs)
    kwargs = check_kwargs_var_plots(kwargs)
    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels defined in the config
    for (filepath, label, tracks_name) in zip(
        datasets_filepaths, datasets_labels, datasets_track_names
    ):
        loaded_trks, loaded_flavour_labels = udt.LoadTrksFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=n_jets,
            tracks_name=tracks_name,
            print_logger=True,
        )

        # Append trks to dict
        trks_dict.update({label: loaded_trks})
        flavour_label_dict.update({label: loaded_flavour_labels})

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")

    logger.info(f"Path: {output_directory}")
    logger.info(f"Track origin: {track_origin}\n")

    # Initialise plot
    n_tracks_plot = histogram_plot(
        bins=np.arange(-0.5, 40.5, 1),
        ymin_ratio_1=ratio_cut[0] if ratio_cut is not None else None,
        ymax_ratio_1=ratio_cut[1] if ratio_cut is not None else None,
        n_ratio_panels=1 if len(datasets_filepaths) > 1 else 0,
        norm=kwargs["norm"],
        y_scale=kwargs["y_scale"],
        ylabel=kwargs["ylabel"],
        logy=kwargs["logy"],
        figsize=kwargs["figsize"],
        leg_ncol=kwargs["leg_ncol"],
        leg_loc=kwargs["leg_loc"],
        apply_atlas_style=kwargs["apply_atlas_style"],
        atlas_second_tag=kwargs["atlas_second_tag"],
        bin_width_in_ylabel=kwargs["bin_width_in_ylabel"],
    )
    # Set xlabel
    n_tracks_plot.xlabel = (
        "Number of tracks per jet"
        if track_origin == "All"
        else f"Number of tracks per jet ({track_origin})"
    )

    # Iterate over datasets
    for dataset_number, (label, linestyle) in enumerate(
        zip(datasets_labels, linestyles[: len(datasets_labels)])
    ):
        # Sort after given variable
        trks = np.asarray(trks_dict[label])
        if track_origin == "All":
            n_tracks = np.sum(~np.isnan(trks["ptfrac"]), axis=1)
        else:
            n_tracks = np.sum(
                np.logical_and(
                    ~np.isnan(trks["ptfrac"]),
                    trks["truthOriginLabel"] == global_config.OriginType[track_origin],
                ),
                axis=1,
            )

        for flav_label, flavour in enumerate(class_labels):

            n_tracks_flavour = n_tracks[flavour_label_dict[label] == flav_label]

            n_tracks_plot.add(
                histogram(
                    values=n_tracks_flavour,
                    flavour=flavour,
                    label=label,
                    linestyle=linestyle,
                ),
                reference=not bool(dataset_number),
            )

    n_tracks_plot.draw()
    n_tracks_plot.savefig(
        f"{output_directory}/nTracks_per_Jet_{track_origin}.{kwargs['plot_type']}"
    )


def plot_input_vars_trks(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    var_dict: dict,
    n_jets: int,
    binning: dict,
    class_labels: list,
    sorting_variable: str = "ptfrac",
    n_leading: list = None,
    output_directory: str = "input_vars_trks",
    ratio_cut: list = None,
    track_origin: str = "All",
    **kwargs,
):
    """
    Plot the track variable in comparison to another model with ratio plot. If multiple
    datasets are provided, a ratio plot comparing each flavour individually will be
    added below the main plot.

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
    n_jets : int
        Number of jets to use for plotting.
    binning : dict
        Decide which binning is used.
    class_labels : list
        List of class_labels which are to be plotted.
    sorting_variable : str
        Variable which is used for sorting.
    n_leading : list
        n-th leading jet which is plotted. For all, = None.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    ratio_cut : list
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
    # TODO: move remove ratio_cut and replace with "ratio_ymin" and "ratio_ymax" in
    # kwargs which are then handed to the histogram plot
    # check the kwargs
    kwargs = translate_kwargs(kwargs)
    kwargs = check_kwargs_var_plots(kwargs)

    # check to avoid dangerous default value (list)
    if n_leading is None:
        n_leading = [None]
    bins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
            if variable.startswith("number"):
                # TODO: change to |= in python 3.9
                bins_dict.update(
                    {
                        variable: np.arange(
                            binning[variable][0] - 0.5,
                            binning[variable][1] - 0.5,
                            binning[variable][2],
                        )
                    }
                )

            else:
                # TODO: change to |= in python 3.9
                bins_dict.update(
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
            # TODO: change to |= in python 3.9
            bins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            # TODO: change to |= in python 3.9
            bins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels defined in the config
    for filepath, label, tracks_name in zip(
        datasets_filepaths,
        datasets_labels,
        datasets_track_names,
    ):

        # Get the tracks and the labels from the file/files
        trks, flavour_labels = udt.LoadTrksFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=n_jets,
            tracks_name=tracks_name,
            print_logger=False,
        )

        # Append trks to dict
        # TODO: change to |= in python 3.9
        trks_dict.update({label: trks})
        # TODO: change to |= in python 3.9
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

    for n_lead in n_leading:
        if n_lead == "None":
            n_lead = None

        # Check if path is existing, if not mkdir
        if n_lead is None:
            filedir = os.path.join(output_directory, sorting_variable, "All")

            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        else:
            filedir = os.path.join(output_directory, sorting_variable, str(n_lead))
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

        logger.info(f"Path: {filedir}")
        logger.info(f"Sorting: {sorting_variable}")
        logger.info(f"nLeading track: {n_lead}")
        logger.info(f"Track origin: {track_origin}\n")

        # Loop over variables
        for var in trksVars:
            if var in bins_dict:
                logger.info(f"Plotting {var}...")

                # Initialise plot for this variable
                var_plot = histogram_plot(
                    bins=bins_dict[var],
                    ymin_ratio_1=ratio_cut[0] if ratio_cut is not None else None,
                    ymax_ratio_1=ratio_cut[1] if ratio_cut is not None else None,
                    n_ratio_panels=1 if len(datasets_filepaths) > 1 else 0,
                    norm=kwargs["norm"],
                    y_scale=kwargs["y_scale"],
                    ylabel=kwargs["ylabel"],
                    logy=kwargs["logy"],
                    figsize=kwargs["figsize"],
                    leg_ncol=kwargs["leg_ncol"],
                    apply_atlas_style=kwargs["apply_atlas_style"],
                    atlas_second_tag=kwargs["atlas_second_tag"],
                    bin_width_in_ylabel=kwargs["bin_width_in_ylabel"],
                )

                if n_lead is None:
                    var_plot.xlabel = (
                        f"{var}" if track_origin == "All" else f"{var} ({track_origin})"
                    )

                else:
                    var_plot.xlabel = (
                        f"{n_lead+1} leading tracks {var}"
                        if track_origin == "All"
                        else (f"{n_lead+1} leading tracks {var} ({track_origin})")
                    )

                # Iterate over datasets
                for dataset_number, (label, linestyle) in enumerate(
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

                    for flav_label, flavour in enumerate(class_labels):
                        jets = tmp[flavour_label_dict[label] == flav_label]

                        # Get number of tracks
                        if track_origin == "All":
                            track_values = jets[:, n_lead][~np.isnan(jets[:, n_lead])]
                        else:
                            mask_nan = ~np.isnan(jets[var][:, n_lead])
                            mask_origin = np.asarray(
                                jets[:, n_lead]["truthOriginLabel"]
                                == global_config.OriginType[track_origin]
                            )
                            track_values = jets[:, n_lead][
                                np.logical_and(mask_nan, mask_origin)
                            ][var]

                        # Add histogram to plot
                        var_plot.add(
                            histogram(
                                values=track_values,
                                flavour=flavour,
                                label=label,
                                linestyle=linestyle,
                            ),
                            reference=not bool(dataset_number),
                        )

                var_plot.draw()
                var_plot.savefig(
                    f"{filedir}/{var}_{n_lead}_{track_origin}.{kwargs['plot_type']}",
                    transparent=kwargs["transparent"],
                )
        logger.info(f"\n{80 * '-'}")


def plot_input_vars_jets(
    datasets_filepaths: list,
    datasets_labels: list,
    var_dict: dict,
    n_jets: int,
    binning: dict,
    class_labels: list,
    special_param_jets: dict = None,
    output_directory: str = "input_vars_jets",
    ratio_cut: list = None,
    **kwargs,
):
    """
    Plot the jet variable comparison for the given datasets. If multiple datasets are
    provided, a ratio plot comparing each flavour individually will be added below
    the main plot.

    Parameters
    ----------
    datasets_filepaths : list
        List of filepaths to the files.
    datasets_labels : list
        Label of the dataset for the legend.
    var_dict : dict
        Variable dict where all variables of the files are saved.
    n_jets : int
        Number of jets to use for plotting.
    binning : dict
        Decide which binning is used.
    class_labels : list
        List of class_labels which are to be plotted.
    special_param_jets : dict
        Dict with special x-axis cuts for the given variable.
    output_directory : str
        Name of the output directory. Only the dir name not path!
    ratio_cut : list
        List of y-axis cuts for the ratio block.

    **kwargs: dict
        additional arguments passed to `check_kwargs_var_plots`

    Raises
    ------
    ValueError
        If the type of the given binning is not supported.
    """

    # check the kwargs
    kwargs = translate_kwargs(kwargs)
    kwargs = check_kwargs_var_plots(kwargs, yAxisIncrease=10)

    # Set the ylabel to jets
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Number of Jets"

    bins_dict = {}

    # Load the given binning or set it
    for variable in binning:
        if isinstance(binning[variable], list):
            bins_dict.update(
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
            bins_dict.update({variable: binning[variable]})

        # If None, give default value
        elif binning[variable] is None:
            bins_dict.update({variable: int(100)})

        else:
            raise ValueError(f"Type {type(binning[variable])} is not supported!")

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    jets_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels defined in the config
    for (filepath, label) in zip(datasets_filepaths, datasets_labels):
        # Get the tracks and the labels from the file/files
        jets, flavour_labels = udt.LoadJetsFromFile(
            filepath=filepath,
            class_labels=class_labels,
            nJets=n_jets,
            print_logger=False,
        )

        # Append jets to dict
        jets_dict.update({label: jets})
        flavour_label_dict.update({label: flavour_labels})

    variable_config = GetVariableDict(var_dict)

    # Loading jet variables
    jet_variables = [
        i
        for j in variable_config["train_variables"]
        for i in variable_config["train_variables"][j]
    ]

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")
    filedir = f"{output_directory}/"

    # Loop over vars
    for var in jet_variables:
        if var in bins_dict:

            # Initialise plot for this variable
            var_plot = histogram_plot(
                bins=bins_dict[var],
                xlabel=var,
                ymin_ratio_1=ratio_cut[0] if ratio_cut is not None else None,
                ymax_ratio_1=ratio_cut[1] if ratio_cut is not None else None,
                n_ratio_panels=1 if len(datasets_filepaths) > 1 else 0,
                norm=kwargs["norm"],
                y_scale=kwargs["y_scale"],
                ylabel=kwargs["ylabel"],
                logy=kwargs["logy"],
                figsize=kwargs["figsize"],
                leg_ncol=kwargs["leg_ncol"],
                leg_loc=kwargs["leg_loc"],
                apply_atlas_style=kwargs["apply_atlas_style"],
                atlas_second_tag=kwargs["atlas_second_tag"],
                bin_width_in_ylabel=kwargs["bin_width_in_ylabel"],
            )
            # setting range based on value from config file
            if special_param_jets is not None and var in special_param_jets:
                if (
                    "lim_left" in special_param_jets[var]
                    and "lim_right" in special_param_jets[var]
                ):
                    lim_left = special_param_jets[var]["lim_left"]
                    lim_right = special_param_jets[var]["lim_right"]
                    var_plot.bins_range = (lim_left, lim_right)

            logger.info(f"Plotting {var} ...")

            # Iterate over datasets
            for dataset_number, (label, linestyle) in enumerate(
                zip(datasets_labels, linestyles[: len(datasets_labels)])
            ):
                # Get variable and the labels of the jets
                jets_var = jets_dict[label][var]
                flavour_labels_var = flavour_label_dict[label]

                # Clean both from nans
                jets_var_clean = jets_var[~np.isnan(jets_var)]
                flavour_label_clean = flavour_labels_var[~np.isnan(jets_var)]

                for flav_label, flavour in enumerate(class_labels):
                    jets_flavour = jets_var_clean[flavour_label_clean == flav_label]

                    # Add histogram to plot
                    var_plot.add(
                        histogram(
                            values=jets_flavour,
                            flavour=flavour,
                            label=label,
                            linestyle=linestyle,
                        ),
                        reference=not bool(dataset_number),
                    )

            # Draw and save the plot
            var_plot.draw()
            var_plot.savefig(
                f"{filedir}/{var}.{kwargs['plot_type']}",
                transparent=kwargs["transparent"],
            )

    logger.info(f"\n{80 * '-'}")
