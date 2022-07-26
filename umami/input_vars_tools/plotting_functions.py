"""Plots the given input variables of the given files and also a comparison."""
import os

import numpy as np
from pandas import DataFrame
from puma import Histogram, HistogramPlot

import umami.data_tools as udt
from umami.configuration import global_config, logger
from umami.plotting_tools.utils import translate_binning
from umami.preprocessing_tools import GetVariableDict


def check_kwargs_for_ylabel_and_n_ratio_panel(
    kwargs: dict,
    fallback_ylabel: str,
    n_datasets: int,
) -> dict:
    """Helper function to check the following keyword arguments + using fallback
    values if they are not set
    - ylabel
    - n_ratio_panels
    - norm (set to "True" if not provided)

    Parameters
    ----------
    kwargs : dict
        Keyword arguments handed to the plotting function
    fallback_ylabel : str
        Fallback value for the ylabel
    n_datasets : int
        Number of datasets that are plotted

    Returns
    -------
    kwargs
        Updated keyword arguments
    """

    # check the kwargs
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = fallback_ylabel
    if "norm" not in kwargs:
        kwargs["norm"] = True
    # Add "Normalised" to ylabel in case it was forgotten
    if kwargs["norm"] and "norm" not in kwargs["ylabel"].lower():
        kwargs["ylabel"] = f"Normalised {kwargs['ylabel']}"
    # Set number of ratio panels if not specified
    if "n_ratio_panels" not in kwargs:
        kwargs["n_ratio_panels"] = 1 if n_datasets > 1 else 0
    return kwargs


def plot_n_tracks_per_jet(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_track_names: list,
    n_jets: int,
    class_labels: list,
    output_directory: str = "input_vars_trks",
    plot_type: str = "pdf",
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
    plot_type: str, optional
        File format for the output, by default "pdf"
    track_origin : str, optional
        Track set that is to be used for plotting, by default "All"
    **kwargs: dict
        Keyword arguments passed to the plot. You can use all arguments that are
        supported by the `HistogramPlot` class in the plotting API.
    """

    kwargs = check_kwargs_for_ylabel_and_n_ratio_panel(
        kwargs,
        fallback_ylabel="Number of Jets",
        n_datasets=len(datasets_filepaths),
    )

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
            n_jets=n_jets,
            tracks_name=tracks_name,
            print_logger=True,
        )

        # Append trks to dict
        trks_dict.update({label: loaded_trks})
        flavour_label_dict.update({label: loaded_flavour_labels})

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")

    logger.info("Path: %s", output_directory)
    logger.info("Track origin: %s\n", track_origin)

    # Initialise plot
    n_tracks_plot = HistogramPlot(**kwargs)
    # Set xlabel
    n_tracks_plot.xlabel = (
        "Number of tracks per jet"
        if track_origin == "All"
        else f"Number of tracks per jet ({track_origin})"
    )

    # Store the means of the n_tracks distributions to print them at the end
    n_tracks_means = {label: {} for label in datasets_labels}
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
            n_tracks_means[label].update({flavour: n_tracks_flavour.mean()})

            n_tracks_plot.add(
                Histogram(
                    values=n_tracks_flavour,
                    flavour=flavour,
                    ratio_group=flavour,
                    label=label,
                    linestyle=linestyle,
                ),
                reference=not bool(dataset_number),
            )

    n_tracks_plot.draw()
    n_tracks_plot.savefig(
        f"{output_directory}/nTracks_per_Jet_{track_origin}.{plot_type}"
    )
    logger.info("Average number of tracks:\n%s", DataFrame.from_dict(n_tracks_means))


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
    plot_type: str = "pdf",
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
    plot_type: str, optional
        File format for the output, by default "pdf"
    track_origin : str, optional
        Track set that is to be used for plotting, by default "All"
    **kwargs: dict
        Keyword arguments passed to the plot. You can use all arguments that are
        supported by the `HistogramPlot` class in the plotting API.

    """

    kwargs = check_kwargs_for_ylabel_and_n_ratio_panel(
        kwargs,
        fallback_ylabel="Number of Tracks",
        n_datasets=len(datasets_filepaths),
    )

    # check to avoid dangerous default value (list)
    if n_leading is None:
        n_leading = [None]

    # Create dict that stores the binning for all the variables
    bins_dict = {}
    for variable in binning:
        bins_dict.update({variable: translate_binning(binning[variable], variable)})

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
            n_jets=n_jets,
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
        trks_vars = variable_config["tracks"]

    except KeyError:
        no_norm_vars = variable_config["track_train_variables"][
            datasets_track_names[0]
        ]["noNormVars"]
        log_norm_vars = variable_config["track_train_variables"][
            datasets_track_names[0]
        ]["logNormVars"]
        joint_norm_vars = variable_config["track_train_variables"][
            datasets_track_names[0]
        ]["jointNormVars"]
        trks_vars = no_norm_vars + log_norm_vars + joint_norm_vars

        # Check for variables in the other
        for counter, track_names in enumerate(datasets_track_names):
            if counter != 0:
                no_norm_vars_tmp = variable_config["track_train_variables"][
                    track_names
                ]["noNormVars"]
                log_norm_vars_tmp = variable_config["track_train_variables"][
                    track_names
                ]["logNormVars"]
                joint_norm_vars_tmp = variable_config["track_train_variables"][
                    track_names
                ]["jointNormVars"]
                trks_vars_tmp = (
                    no_norm_vars_tmp + log_norm_vars_tmp + joint_norm_vars_tmp
                )

                for iter_var in trks_vars_tmp:
                    if iter_var not in trks_vars:
                        logger.warning(
                            "Variable %s of %s not in %s track collection. Skipping...",
                            iter_var,
                            datasets_labels[counter],
                            datasets_labels[0],
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

        logger.info("Path: %s", filedir)
        logger.info("Sorting: %s", sorting_variable)
        logger.info("nLeading track: %s", n_lead)
        logger.info("Track origin: %s\n", track_origin)

        # Loop over variables
        for var in trks_vars:
            if var in bins_dict:
                logger.info("Plotting %s...", var)

                # Initialise plot for this variable
                var_plot = HistogramPlot(bins=bins_dict[var], **kwargs)

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
                        trks_array = np.asarray(
                            [
                                trks_dict[label][var][k][sorting[k]]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )

                    else:
                        # Select tracks of a given origin, so keep truthOriginLabel
                        trks_array = np.asarray(
                            [
                                trks_dict[label][[var, "truthOriginLabel"]][k][
                                    sorting[k]
                                ]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )

                    # Retrieve the track mask to figure out which track is a placeholder
                    try:
                        trk_mask = np.asarray(
                            [
                                trks_dict[label]["valid"][k][sorting[k]]
                                for k in range(len(trks_dict[label][sorting_variable]))
                            ]
                        )

                    except ValueError:
                        trk_mask = ~np.isnan(
                            np.asarray(
                                [
                                    trks_dict[label]["ptfrac"][k][sorting[k]]
                                    for k in range(
                                        len(trks_dict[label][sorting_variable])
                                    )
                                ]
                            )
                        )

                    for flav_label, flavour in enumerate(class_labels):
                        # Get the mask for the flavour which is to be plotted
                        tracks_flav_mask = flavour_label_dict[label] == flav_label

                        # Get the tracks and masks for one specific flavour
                        tracks = trks_array[tracks_flav_mask][:, n_lead]
                        tracks_mask = trk_mask[tracks_flav_mask][:, n_lead]

                        # Get number of tracks
                        if track_origin == "All":
                            track_values = tracks[tracks_mask]

                        else:
                            mask_origin = np.asarray(
                                tracks["truthOriginLabel"]
                                == global_config.OriginType[track_origin]
                            )
                            track_values = tracks[
                                np.logical_and(tracks_mask, mask_origin)
                            ][var]

                        # Add histogram to plot
                        var_plot.add(
                            Histogram(
                                values=track_values,
                                flavour=flavour,
                                ratio_group=flavour,
                                label=label,
                                linestyle=linestyle,
                            ),
                            reference=not bool(dataset_number),
                        )

                var_plot.draw()
                var_plot.savefig(f"{filedir}/{var}_{n_lead}_{track_origin}.{plot_type}")

            else:
                logger.debug("Variable %s not in the binning dict. Skipping ...", var)
        logger.info("\n%s", 80 * "-")


def plot_input_vars_jets(
    datasets_filepaths: list,
    datasets_labels: list,
    var_dict: dict,
    n_jets: int,
    binning: dict,
    class_labels: list,
    special_param_jets: dict = None,
    output_directory: str = "input_vars_jets",
    plot_type: str = "pdf",
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
    plot_type: str, optional
        File format for the output, by default "pdf"
    **kwargs: dict
        Keyword arguments passed to the plot. You can use all arguments that are
        supported by the `HistogramPlot` class in the plotting API.
    """

    kwargs = check_kwargs_for_ylabel_and_n_ratio_panel(
        kwargs,
        fallback_ylabel="Number of Jets",
        n_datasets=len(datasets_filepaths),
    )

    # Create dict that stores the binning for all the variables
    bins_dict = {}
    for variable in binning:
        bins_dict.update({variable: translate_binning(binning[variable], variable)})

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
            n_jets=n_jets,
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
            var_plot = HistogramPlot(bins=bins_dict[var], xlabel=var, **kwargs)
            # setting range based on value from config file
            if special_param_jets is not None and var in special_param_jets:
                if (
                    "lim_left" in special_param_jets[var]
                    and "lim_right" in special_param_jets[var]
                ):
                    lim_left = special_param_jets[var]["lim_left"]
                    lim_right = special_param_jets[var]["lim_right"]
                    var_plot.bins_range = (lim_left, lim_right)

            logger.info("Plotting %s ...", var)

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
                        Histogram(
                            values=jets_flavour,
                            flavour=flavour,
                            ratio_group=flavour,
                            label=label,
                            linestyle=linestyle,
                        ),
                        reference=not bool(dataset_number),
                    )

            # Draw and save the plot
            var_plot.draw()
            var_plot.savefig(f"{filedir}/{var}.{plot_type}")

    logger.info("\n%s", 80 * "-")
