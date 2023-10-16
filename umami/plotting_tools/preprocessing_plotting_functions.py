"""Plotting functions for the preprocessing."""

import os

import h5py
import numpy as np
import pandas as pd
from puma import Histogram, HistogramPlot

from umami.configuration import global_config, logger
from umami.data_tools import load_jets_from_file
from umami.plotting_tools.utils import translate_kwargs


def plot_unique_jet_appearence(
    sample: str,
    class_labels: list,
    output_dir: str,
    n_jets: int,
    fileformat: str = "pdf",
    **kwargs,
) -> None:
    """
    Plot a histogram which shows how often the jets are duplicated.
    This is done per class.

    Parameters
    ----------
    sample : str
        Path to the resampled file (after resampling before writing)
    class_labels : list
        List with the class labels used (flavours)
    output_dir : str
        Path where the plot wil be stored. The filename is always
        Duplicated_jet_multiplicity + the format of the file.
    n_jets : int
        Number of jets used for the histogram
    fileformat : str, optional
        Fileformat used for the plot, by default "pdf"
    **kwargs : kwargs
        kwargs from `plot_object`
    """

    jets, _ = load_jets_from_file(
        filepath=sample,
        class_labels=class_labels,
        n_jets=n_jets,
        variables=["eventNumber", "jetPtRank"],
        print_logger=False,
    )

    # Init the histogram plot object
    histo_plot = HistogramPlot(
        bins=30,
        bins_range=[0, 30],
        **kwargs,
    )

    # Init a total unique to sum to
    total_unique = 0

    # Loop over the different flavours
    for counter, class_label in enumerate(class_labels):
        # Select only the wanted category
        tmp_jets = jets[jets["Umami_labels"] == counter]

        # Check that the needed variables are loaded and given
        try:
            unique, count = np.unique(
                tmp_jets["eventNumber"] + tmp_jets["jetPtRank"] / 10,
                return_counts=True,
            )

            # Add the number of unique jets for this flavour to the total counter
            total_unique += len(unique)

        # If not, print error and break the loop
        except KeyError:
            logger.error(
                """
                Duplicate plot can't be made without the eventNumber variable
                and the jetPtRank variable! Please check that these are given!
                Plot will be empty!
                """
            )
            break

        # Add the flavour to the histogram
        histo_plot.add(
            Histogram(
                values=count,
                flavour=class_label,
                label=r"($N_{Unique} = $" + f"{len(unique)})",
            ),
            reference=False,
        )

    # Set the x-label
    histo_plot.xlabel = "Number of duplicates"

    if histo_plot.atlas_second_tag == "" or histo_plot.atlas_second_tag is None:
        histo_plot.atlas_second_tag = (
            f"Total number of unique jets: {total_unique}"
            + f",\nTotal number of jets: {len(jets)}"
        )

    else:
        histo_plot.atlas_second_tag += (
            f",\nTotal number of unique jets: {total_unique}"
            + f",\nTotal number of jets: {len(jets)}"
        )

    # Draw and save the plot
    histo_plot.draw()
    histo_plot.savefig(
        plot_name=os.path.join(
            output_dir,
            f"Duplicated_jet_multiplicity.{fileformat}",
        ),
        **kwargs,
    )


def plot_variable(
    df_in,
    labels: np.ndarray,
    variable: str,
    variable_index: int,
    var_type: str,
    class_labels: list,
    output_dir: str,
    fileformat: str = "pdf",
    **kwargs,
) -> None:
    """
    Plot a given variable.

    Parameters
    ----------
    df_in : pd.DataFrame or np.ndarray
        DataFrame (for jets) or ndarray (for tracks) with
        the jets/tracks inside.
    labels : np.ndarray
        One hot encoded array with the truth values.
    variable : str
        Name of the variable which is to be plotted.
    variable_index : int
        Index of the variable in the final training set. This
        is used to identify the variables in the final training
        set.
    var_type : str
        Type of the variable that is used. Either `jets` or
        `tracks`.
    class_labels : list
        List with the flavours used (ORDER IMPORTANT).
    output_dir : str
        Directory where the plot is saved.
    fileformat : str, optional
        Fileformat of the plots, by default "pdf"
    **kwargs : kwargs
        kwargs from `plot_object`

    Raises
    ------
    TypeError
        If the given variable type is not supported.
    """

    # Translate the kwargs
    kwargs = translate_kwargs(kwargs)

    # Give a debug logger
    logger.debug("Plotting variable %s...", variable)

    # Init the histogram plot object
    histo_plot = HistogramPlot(**kwargs)

    # Set the x-label
    if histo_plot.xlabel is None:
        histo_plot.xlabel = variable

    # Loop over the flavours
    for flav_counter, flavour in enumerate(class_labels):
        # This is the case if a pandas Dataframe is given
        try:
            flavour_jets = df_in[variable][labels == flav_counter].values.flatten()

        # This is the case when a numpy ndarray is given
        except AttributeError:
            flavour_jets = df_in[variable][labels == flav_counter].flatten()

        # This is the case if the training set is already converted to X_train etc.
        except IndexError as error:
            if var_type.casefold() == "jets":
                flavour_jets = df_in[:, variable_index][
                    labels == flav_counter
                ].flatten()

            elif var_type.casefold() == "tracks":
                flavour_jets = df_in[:, :, variable_index][
                    labels == flav_counter
                ].flatten()

            else:
                raise TypeError(
                    f"Variable type {var_type} not supported! Only jets and tracks!"
                ) from error

        # Add the flavour to the histogram
        histo_plot.add(
            Histogram(
                values=np.nan_to_num(flavour_jets),
                flavour=flavour,
            ),
            reference=False,
        )

    # Draw and save the plot
    histo_plot.draw()
    histo_plot.savefig(
        plot_name=os.path.join(
            output_dir,
            f"{variable}.{fileformat}",
        ),
        **kwargs,
    )


def plot_resampling_variables(
    concat_samples: dict,
    var_positions: list,
    variable_names: list,
    sample_categories: list,
    output_dir: str,
    bins_dict: dict,
    sample_id_position: int = 3,
    fileformat: str = "pdf",
    **kwargs,
) -> None:
    """
    Plot the variables which are used for resampling before the resampling
    starts.

    Parameters
    ----------
    concat_samples : dict
        Dict with the format given in the Undersampling class by the class object
        `concat_samples`.
    var_positions : list
        The position where the variables are stored in the sub-dict `jets`.
    variable_names : list
        The name of the 2 variables which will be plotted.
    sample_categories : list
        List with the names of the sample categories (e.g. ["ttbar", "zprime"]).
    output_dir : str
        Name of the output directory where the plots will be saved.
    bins_dict : dict
        Dict with the binning for the resampling variables. First key must be the
        variable name with a tuple of 3 int which gives the lower limit, upper limit
        and the number of bins to use.
    sample_id_position : int, optional
        Position in the numpy.ndarray of the concat_samples where the sample
        id is stored. By default 3
    fileformat : str, optional
        Format of the plot file, by default "pdf".
    **kwargs : kwargs
        kwargs from `plot_object`

    Raises
    ------
    ValueError
        If unsupported binning is provided.
    """

    # Check if output directory exists
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Defining linestyles for the resampling variables
    linestyles = ["-", "--", "-."]

    # Translate the kwargs to new naming scheme
    kwargs = translate_kwargs(kwargs)

    # Loop over the variables which are used for resampling
    for var, varpos in zip(variable_names, var_positions):
        if isinstance(bins_dict[var], int):
            bins = bins_dict[var]
            bins_range = None

        elif isinstance(bins_dict[var], (list, tuple)) and len(bins_dict[var]) == 3:
            bins = bins_dict[var][2]
            bins_range = (
                bins_dict[var]["bins_range"][0],
                bins_dict[var]["bins_range"][1],
            )

        else:
            raise ValueError(
                "Provided binning for plot_resampling_variables is "
                "neither a list with three entries nor an int!"
            )

        # Init a new histogram
        histo_plot = HistogramPlot(
            bins=bins,
            bins_range=bins_range,
            **kwargs,
        )

        # Set the x-label
        if histo_plot.xlabel is None:
            histo_plot.xlabel = f"{var}"

        # Check if the variable is pT (which is in the files in MeV)
        # and set the scale value to make it GeV in the plots
        if var in ["pT", "pt_btagJes"] or var == global_config.pTvariable:
            scale_val = 1e3
            histo_plot.xlabel += " [GeV]"

        else:
            scale_val = 1

        # Loop over the different flavours
        for flavour in concat_samples:
            # Loop over sample ids (ttbar and zprime for example)
            for sample_id in np.unique(
                concat_samples[flavour]["jets"][:, sample_id_position]
            ).astype("int"):
                # Add the histogram for the flavour
                histo_plot.add(
                    Histogram(
                        values=concat_samples[flavour]["jets"][:, varpos] / scale_val,
                        flavour=flavour,
                        label=sample_categories[sample_id]
                        if sample_categories
                        else None,
                        linestyle=linestyles[sample_id],
                    ),
                    reference=False,
                )

        # Draw and save the plot
        histo_plot.draw()
        histo_plot.savefig(
            plot_name=os.path.join(
                output_dir,
                f"{var}_before_resampling.{fileformat}",
            )
        )


def preprocessing_plots(
    sample: str,
    var_dict: dict,
    class_labels: list,
    plots_dir: str,
    use_random_jets: bool = False,
    jet_collection: str = "jets",
    track_collection_list: list = None,
    n_jets: int = 3e4,
    seed: int = 42,
    **kwargs,
):
    """
    Plotting the different track and jet variables after
    the preprocessing steps.

    Parameters
    ----------
    sample : str
        Path to output file of the preprocessing step.
    var_dict : dict
        Loaded variable dict.
    class_labels : list
        List with the flavours used (ORDER IMPORTANT).
    plots_dir : str
        Path to folder where the plots are saved.
    use_random_jets : bool, optional
        Decide if random jets are drawn from the sample to
        ensure correct mixing. Otherwise the first n_jets are
        used for plotting, by default False
    jet_collection : str, optional
        Name of the jet collection, by default "jets"
    track_collection_list : list, optional
        List of str of the track collections which are to be
        plotted, by default None
    n_jets : int, optional
        Number of jets to plot, by default int(3e4)
    seed : int, optional
        Random seed for the selection of the jets, by default 42
    **kwargs : kwargs
        kwargs from `plot_object`

    Raises
    ------
    TypeError
        If the provided track collection list is neither a string or
        a list.
    """
    logger.info("Plots will be saved in the directory %s", plots_dir)
    # Get max number of available jets
    with h5py.File(sample, "r") as f_h5:
        try:
            n_jets_infile = len(f_h5["/jets/inputs"])

        except KeyError:
            n_jets_infile = len(f_h5["jets"])

    # Check if random values are used or not
    if use_random_jets is True:
        # Get a random generator with specified seed
        rng = np.random.default_rng(seed=seed)

        # Mix the chunks
        selected_indicies = sorted(
            rng.choice(
                np.arange(n_jets_infile, dtype=int),
                int(n_jets),
                replace=False,
            )
        )

    else:
        # if number of requested jets is larger that what is available,
        # plot all available jets.
        if n_jets > n_jets_infile:
            logger.warning(
                "You requested %i jets,but there are only %i jets in the input!",
                n_jets,
                n_jets_infile,
            )
        selected_indicies = np.arange(min(n_jets, n_jets_infile), dtype=int)

    # Check if track collection list is valid
    if isinstance(track_collection_list, str):
        track_collection_list = [track_collection_list]

    elif track_collection_list is None:
        track_collection_list = []

    elif not isinstance(track_collection_list, list):
        raise TypeError(
            "Track Collection list for variable plotting must be a list or a string!"
        )

    # Open the file which is to be plotted
    with h5py.File(sample, "r") as infile:
        # Get the labels of the jets to plot
        try:
            labels = infile["jets/labels"][selected_indicies]
        except KeyError:
            labels = infile["/labels"][selected_indicies]

        # Check if jet collection is given
        if jet_collection:
            # Check if output directory exists
            os.makedirs(
                plots_dir,
                exist_ok=True,
            )

            # Extract the correct variables
            variables_header = var_dict["train_variables"]
            jet_var_list = [i for j in variables_header for i in variables_header[j]]

            # Get the jets from file
            try:
                jets = np.asarray(infile["jets/inputs"][selected_indicies])

            except KeyError:
                jets = pd.DataFrame(
                    infile["/jets"].fields(jet_var_list)[selected_indicies]
                )

            # Loop over variables
            for jet_var_counter, jet_var in enumerate(jet_var_list):
                # Plotting
                plot_variable(
                    df_in=jets,
                    labels=labels,
                    variable=jet_var,
                    variable_index=jet_var_counter,
                    var_type="jets",
                    class_labels=class_labels,
                    output_dir=plots_dir,
                    **kwargs,
                )

        # Loop over the track selections
        for track_collection in track_collection_list:
            # Check if output directory exists
            os.makedirs(
                os.path.join(
                    plots_dir,
                    track_collection,
                ),
                exist_ok=True,
            )

            # Loading track variables for given collection
            no_norm_vars = var_dict["track_train_variables"][track_collection][
                "noNormVars"
            ]
            log_norm_vars = var_dict["track_train_variables"][track_collection][
                "logNormVars"
            ]
            joint_norm_vars = var_dict["track_train_variables"][track_collection][
                "jointNormVars"
            ]
            trks_vars = no_norm_vars + log_norm_vars + joint_norm_vars

            # Get the tracks from file
            try:
                tracks = np.asarray(
                    infile[f"{track_collection}/inputs"][selected_indicies]
                )

            except KeyError:
                tracks = np.asarray(infile[f"/{track_collection}"][selected_indicies])

            # Loop over track variables
            for trk_var_counter, trk_var in enumerate(trks_vars):
                # Plotting
                plot_variable(
                    df_in=tracks,
                    labels=labels,
                    variable=trk_var,
                    variable_index=trk_var_counter,
                    var_type="tracks",
                    class_labels=class_labels,
                    output_dir=os.path.join(plots_dir, track_collection),
                    **kwargs,
                )
