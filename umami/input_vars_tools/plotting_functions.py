"""Plots the given input variables of the given files and also a comparison."""
import operator
import os

import numpy as np
from pandas import DataFrame
from puma import Histogram, HistogramPlot

import umami.data_tools as udt
from umami.configuration import global_config, logger
from umami.plotting_tools.utils import translate_binning


def get_datasets_configuration(plotting_config: dict, tracks: bool = False):
    """Helper function to transform dict that stores the configuration of the different
    datasets into lists of certain parameters.

    Parameters
    ----------
    plotting_config : dict
        Plotting configuration
    tracks : bool, optional
        Bool if the function should look for the `tracks_name` variable in the dataset
        configurations.

    Returns
    -------
    filepath_list : list
        List with the filepaths of all the datasets.
    labels_list : list
        List with the 'dataset label' of each dataset.
    class_labels_list : list
        List with the class labels for each dataset. If no dataset-specific class labels
        are provided, the globally defined class labels are used.
    tracks_name_list : list
        List with the track names of the datasets. Only returned if `tracks` is True.
    """

    filepath_list = []
    labels_list = []
    class_labels_list = []
    tracks_name_list = []

    datasets_config = plotting_config["Datasets_to_plot"]

    for dataset_name in datasets_config:
        if not datasets_config[dataset_name]["files"] is None:
            filepath_list.append(datasets_config[dataset_name]["files"])
            labels_list.append(datasets_config[dataset_name]["label"])
            # check if this dataset has a specific list of class labels
            class_labels_list.append(
                datasets_config[dataset_name]["class_labels"]
                if "class_labels" in datasets_config[dataset_name]
                else plotting_config["class_labels"]
            )
            if tracks:
                tracks_name_list.append(datasets_config[dataset_name]["tracks_name"])

    if tracks:
        return filepath_list, labels_list, class_labels_list, tracks_name_list
    return filepath_list, labels_list, class_labels_list


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
    datasets_class_labels: list,
    datasets_track_names: list,
    n_jets: int,
    cut_vars_dict: dict,
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
    datasets_class_labels : list
        List with dataset-specific class labels, e.g. [["ujets", "cjets"], ["cjets"]]
        to plot light-jets and c-jets for the first but only c-jets for the second
        dataset
    datasets_track_names : list
        List with the track names of the files.
    n_jets : int
        Number of jets to use.
    cut_vars_dict : dict
        Dict with cuts on variables applied to the input datasets.
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
    for filepath, label, tracks_name, class_labels in zip(
        datasets_filepaths, datasets_labels, datasets_track_names, datasets_class_labels
    ):
        loaded_trks, loaded_flavour_labels = udt.load_trks_from_file(
            filepath=filepath,
            class_labels=class_labels,
            n_jets=n_jets,
            tracks_name=tracks_name,
            cut_vars_dict=cut_vars_dict,
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
    for dataset_number, (label, linestyle, class_labels) in enumerate(
        zip(datasets_labels, linestyles[: len(datasets_labels)], datasets_class_labels)
    ):
        # Sort after given variable
        trks = np.asarray(trks_dict[label])
        if track_origin == "All":
            n_tracks = np.sum(~np.isnan(trks["ptfrac"]), axis=1)
        else:
            n_tracks = np.sum(
                np.logical_and(
                    ~np.isnan(trks["ptfrac"]),
                    trks["ftagTruthOriginLabel"]
                    == global_config.OriginType[track_origin],
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
    datasets_class_labels: list,
    datasets_track_names: list,
    n_jets: int,
    var_dict: dict,
    cut_vars_dict: dict,
    sorting_variable: str = "ptfrac",
    xlabels_dict: dict = None,
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
    datasets_class_labels : list
        List with dataset-specific class labels, e.g. [["ujets", "cjets"], ["cjets"]]
        to plot light-jets and c-jets for the first but only c-jets for the second
        dataset
    datasets_track_names : list
        List with the track names of the files.
    n_jets : int
        Number of jets to use for plotting.
    var_dict : dict
        Dict with all the variables you want to plot inside with their binning.
    cut_vars_dict : dict
        Dict with cuts on variables applied to the input datasets.
    sorting_variable : str, optional
        Variable which is used for sorting, by default "ptfrac"
    xlabels_dict : dict, optional
        Dict that stores the xlabels of the variables that are plotted. I.e. to
        specify a label for "pt_btagJes", use {"pt_btagJes": "$p_T$ [MeV]"}.
        Variables that do not appear in the dict will have the variable name as xlabel.
        By default None
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

    # Define operator dict
    operator_dict = {
        "+": operator.iadd,
        "-": operator.isub,
        "*": operator.imul,
        "/": operator.itruediv,
        "log": "log",
    }

    kwargs = check_kwargs_for_ylabel_and_n_ratio_panel(
        kwargs,
        fallback_ylabel="Number of Tracks",
        n_datasets=len(datasets_filepaths),
    )

    # check to avoid dangerous default value (list)
    if n_leading is None:
        n_leading = [None]
    if xlabels_dict is None:
        xlabels_dict = {}

    # Create dict that stores the binning for all the variables
    bins_dict = {}

    for variable, entry in var_dict.items():
        if isinstance(entry, dict):
            bins_dict.update({variable: translate_binning(entry["binning"], variable)})

        else:
            bins_dict.update({variable: translate_binning(entry, variable)})

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    trks_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels defined in the config
    for filepath, label, class_labels, tracks_name in zip(
        datasets_filepaths,
        datasets_labels,
        datasets_class_labels,
        datasets_track_names,
    ):
        # Get the tracks and the labels from the file/files
        trks, flavour_labels = udt.load_trks_from_file(
            filepath=filepath,
            class_labels=class_labels,
            cut_vars_dict=cut_vars_dict,
            n_jets=n_jets,
            tracks_name=tracks_name,
            print_logger=False,
        )

        # Append trks to dict
        # TODO: change to |= in python 3.9
        trks_dict.update({label: trks})

        # TODO: change to |= in python 3.9
        flavour_label_dict.update({label: flavour_labels})

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
        for var in bins_dict:
            logger.info("Plotting %s...", var)

            # Initialise plot for this variable
            var_plot = HistogramPlot(bins=bins_dict.get(var, 100), **kwargs)

            # Retrieve x label
            xlabel = xlabels_dict.get(var, var)

            if n_lead is None:
                var_plot.xlabel = (
                    xlabel if track_origin == "All" else f"{xlabel} ({track_origin})"
                )

            else:
                var_plot.xlabel = (
                    f"{n_lead+1} leading tracks {xlabel}"
                    if track_origin == "All"
                    else (f"{n_lead+1} leading tracks {xlabel} ({track_origin})")
                )

            # Iterate over datasets
            for dataset_number, (label, linestyle, class_labels) in enumerate(
                zip(
                    datasets_labels,
                    linestyles[: len(datasets_labels)],
                    datasets_class_labels,
                )
            ):
                # Sort after given variable
                sorting = np.argsort(-1 * trks_dict[label][sorting_variable])

                try:
                    if isinstance(var_dict[var], dict):
                        # Create the list to loop over
                        var_loop_list = var_dict[var]["variables"]

                        # Get an array to to append to
                        trks_array = np.zeros_like(trks_dict[label][var_loop_list[0]])

                        # Ensure that if log is used, only one variable is given
                        if (
                            var_dict[var]["operator"] == "log"
                            and len(var_loop_list) != 1
                        ):
                            raise ValueError(
                                f"You defined log for {var} which uses multiple"
                                " variables. For log, only one variable is supported!"
                            )

                        if (
                            var_dict[var]["operator"] == "log"
                            and len(var_loop_list) == 1
                        ):
                            # Get the correct operator to merge the variables
                            var_operator = "+"
                            use_log = True

                        else:
                            # Get the correct operator to merge the variables
                            var_operator = var_dict[var]["operator"]
                            use_log = False

                    else:
                        # Create the list to loop over
                        var_loop_list = [var]

                        # Get an array to to append to
                        trks_array = np.zeros_like(trks_dict[label][var])

                        # Get the correct operator to merge the variables
                        var_operator = "+"

                        # Set use_log to False
                        use_log = False

                except ValueError:
                    logger.error(
                        "Variable %s not available in %s. Skipping it for %s...",
                        var,
                        label,
                        label,
                    )
                    continue

                for iter_var in var_loop_list:
                    # Sort the variables and tracks after given variable
                    if track_origin == "All":
                        trks_array = operator_dict[var_operator](
                            trks_array,
                            np.asarray(
                                [
                                    trks_dict[label][iter_var][k][sorting[k]]
                                    for k in range(
                                        len(trks_dict[label][sorting_variable])
                                    )
                                ]
                            ),
                        )

                    else:
                        # Select tracks of a given origin, so keep ftagTruthOriginLabel
                        trks_array = operator_dict[var_operator](
                            trks_array,
                            np.asarray(
                                [
                                    trks_dict[label][
                                        [iter_var, "ftagTruthOriginLabel"]
                                    ][k][sorting[k]]
                                    for k in range(
                                        len(trks_dict[label][sorting_variable])
                                    )
                                ]
                            ),
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
                                for k in range(len(trks_dict[label][sorting_variable]))
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
                            tracks["ftagTruthOriginLabel"]
                            == global_config.OriginType[track_origin]
                        )
                        track_values = tracks[np.logical_and(tracks_mask, mask_origin)][
                            var
                        ]

                    # Apply log if chosen
                    if use_log:
                        track_values = np.log(track_values)

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

        logger.info("\n%s", 80 * "-")


def plot_input_vars_jets(
    datasets_filepaths: list,
    datasets_labels: list,
    datasets_class_labels: list,
    var_dict: dict,
    cut_vars_dict: dict,
    n_jets: int,
    xlabels_dict: dict = None,
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
    datasets_class_labels : list
        List with dataset-specific class labels, e.g. [["ujets", "cjets"], ["cjets"]]
        to plot light-jets and c-jets for the first but only c-jets for the second
        dataset
    var_dict : dict
        Dict with all the variables you want to plot inside with their binning.
    cut_vars_dict : dict
        Dict with cuts on variables applied to the input datasets.
    n_jets : int
        Number of jets to use for plotting.
    xlabels_dict : dict, optional
        Dict that stores the xlabels of the variables that are plotted. I.e. to
        specify a label for "pt_btagJes", use {"pt_btagJes": "$p_T$ [MeV]"}.
        Variables that do not appear in the dict will have the variable name as xlabel.
        By default None
    special_param_jets : dict, optional
        Dict with special x-axis cuts for the given variable.
    output_directory : str, optional
        Name of the output directory. Only the dir name not path!
        By default "input_vars_jets"
    plot_type: str, optional
        File format for the output, by default "pdf"
    **kwargs: dict
        Keyword arguments passed to the plot. You can use all arguments that are
        supported by the `HistogramPlot` class in the plotting API.

    Raises
    ------
    ValueError
        When operator log is chosen but more than one variable is given
    """

    # Define operator dict
    operator_dict = {
        "+": operator.iadd,
        "-": operator.isub,
        "*": operator.imul,
        "/": operator.itruediv,
    }

    if xlabels_dict is None:
        xlabels_dict = {}

    kwargs = check_kwargs_for_ylabel_and_n_ratio_panel(
        kwargs,
        fallback_ylabel="Number of Jets",
        n_datasets=len(datasets_filepaths),
    )

    # Create dict that stores the binning for all the variables
    bins_dict = {}

    for variable, entry in var_dict.items():
        if isinstance(entry, dict):
            bins_dict.update({variable: translate_binning(entry["binning"], variable)})

        else:
            bins_dict.update({variable: translate_binning(entry, variable)})

    # Init Linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Init trks and flavour label dicts
    jets_dict = {}
    flavour_label_dict = {}

    # Iterate over the different dataset filepaths and labels defined in the config
    for filepath, label, class_labels in zip(
        datasets_filepaths, datasets_labels, datasets_class_labels
    ):
        # Get the tracks and the labels from the file/files
        jets, flavour_labels = udt.load_jets_from_file(
            filepath=filepath,
            class_labels=class_labels,
            n_jets=n_jets,
            cut_vars_dict=cut_vars_dict,
            print_logger=False,
        )

        # Append jets to dict
        jets_dict.update({label: jets})
        flavour_label_dict.update({label: flavour_labels})

    # Check if path is existing, if not mkdir
    if not os.path.isdir(f"{output_directory}/"):
        os.makedirs(f"{output_directory}/")
    filedir = f"{output_directory}/"

    # Loop over vars
    for var, entry in bins_dict.items():
        # Initialise plot for this variable
        var_plot = HistogramPlot(
            bins=entry,
            xlabel=xlabels_dict.get(var, var),
            **kwargs,
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

        logger.info("Plotting %s ...", var)

        # Iterate over datasets
        for dataset_number, (label, linestyle, class_labels) in enumerate(
            zip(
                datasets_labels,
                linestyles[: len(datasets_labels)],
                datasets_class_labels,
            )
        ):
            try:
                if isinstance(var_dict[var], dict):
                    # Create the list to loop over
                    var_loop_list = var_dict[var]["variables"]

                    # Get an array to append to
                    jets_array = np.zeros_like(jets_dict[label][var_loop_list[0]])
                    jets_bool = np.ones_like(jets_dict[label][var_loop_list[0]])

                    # Ensure that if log is used, only one variable is given
                    if var_dict[var]["operator"] == "log" and len(var_loop_list) != 1:
                        raise ValueError(
                            f"You defined log for {var} which uses multiple variables. "
                            "For log, only one variable is supported!"
                        )

                    if var_dict[var]["operator"] == "log" and len(var_loop_list) == 1:
                        # Get the correct operator to merge the variables
                        var_operator = "+"
                        use_log = True

                    else:
                        # Get the correct operator to merge the variables
                        var_operator = var_dict[var]["operator"]
                        use_log = False

                else:
                    # Create the list to loop over
                    var_loop_list = [var]

                    # Get an array to to append to
                    jets_array = np.zeros_like(jets_dict[label][var])
                    jets_bool = np.ones_like(jets_dict[label][var])

                    # Get the correct operator to merge the variables
                    var_operator = "+"

                    # Set use_log to False
                    use_log = False

            except KeyError:
                logger.error(
                    "Variable %s not available in %s. Skipping it for %s...",
                    var,
                    label,
                    label,
                )
                continue

            for iter_var in var_loop_list:
                # Get the variable from all jets and check for NaNs
                iter_jets_var = jets_dict[label][iter_var]
                iter_jets_bool = np.isnan(iter_jets_var)

                # Convert all NaNs to 0
                iter_jets_var_clean = np.nan_to_num(iter_jets_var)

                # Add them to the prepared zero array
                jets_array = operator_dict[var_operator](
                    jets_array,
                    iter_jets_var_clean,
                )

                # Combine the info if all given values were NaNs
                jets_bool = jets_bool & iter_jets_bool

            # Get the labels of the jets
            flavour_labels_var = flavour_label_dict[label]
            flavour_label_clean = flavour_labels_var[~jets_bool]

            # Remove all jets where the only NaNs are
            jets_array = jets_array[~jets_bool]

            for flav_label, flavour in enumerate(class_labels):
                jets_flavour = jets_array[flavour_label_clean == flav_label]

                # Apply log if chosen
                if use_log:
                    jets_flavour = np.log(jets_flavour)

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
