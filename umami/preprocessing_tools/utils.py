"""Collection of utility functions for preprocessing tools."""
import os

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import global_config, logger
from umami.plotting import histogram, histogram_plot
from umami.plotting.utils import translate_kwargs
from umami.tools import yaml_loader


def GetVariableDict(yaml_file: str) -> dict:
    """
    Reads yaml_file containig the variables and exports
    them to a dict.

    Parameters
    ----------
    yaml_file : str
        Input yaml file containing trainig variables

    Returns
    -------
    out_dict : dict
        Dictionary containing training variables
    """
    with open(yaml_file, "r") as conf:
        in_dict = yaml.load(conf, Loader=yaml_loader)
        out_dict = in_dict.copy()

    if "track_train_variables" in out_dict.keys():
        if (
            "noNormVars" in out_dict["track_train_variables"]
            or "logNormVars" in out_dict["track_train_variables"]
            or "jointNormVas" in out_dict["track_train_variables"]
        ):
            del out_dict["track_train_variables"]
            out_dict["track_train_variables"] = {}
            out_dict["track_train_variables"]["tracks"] = in_dict[
                "track_train_variables"
            ]
            logger.warning(
                "'track_train_varibles' should be a nested dictionary. Default tracks"
                "name 'tracks' being used"
            )

    return out_dict


def GetBinaryLabels(
    df: pd.DataFrame,
    column: str = "label",
) -> np.ndarray:
    """
    Transforms labels to binary labels

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the labels inside.
    column : str, optional
        Label name to be used to binarise, by default "label"

    Returns
    -------
    np.ndarray
        containing binary label with shape (len(df), n_classes)
    """

    lb = LabelBinarizer()
    if isinstance(df, np.ndarray):
        return lb.fit_transform(df)

    labels = np.array(df[column].values)
    return lb.fit_transform(labels)


def plot_variable(
    df,
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
    df : pd.DataFrame or np.ndarray
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

    # Remove all ratio panels
    kwargs["n_ratio_panels"] = 0

    # Translate the kwargs
    kwargs = translate_kwargs(kwargs)

    # Give a debug logger
    logger.debug(f"Plotting variable {variable}...")

    # Init the histogram plot object
    histo_plot = histogram_plot(**kwargs)

    # Set the x-label
    if histo_plot.xlabel is None:
        histo_plot.xlabel = variable

    # Loop over the flavours
    for flav_counter, flavour in enumerate(class_labels):

        # This is the case if a pandas Dataframe is given
        try:
            flavour_jets = df[variable][labels[:, flav_counter] == 1].values.flatten()

        # This is the case when a numpy ndarray is given
        except AttributeError:
            flavour_jets = df[variable][labels[:, flav_counter] == 1].flatten()

        # This is the case if the training set is already converted to X_train etc.
        except IndexError as error:
            if var_type.casefold() == "jets":
                flavour_jets = df[:, variable_index][
                    labels[:, flav_counter] == 1
                ].flatten()

            elif var_type.casefold() == "tracks":
                flavour_jets = df[:, :, variable_index][
                    labels[:, flav_counter] == 1
                ].flatten()

            else:
                raise TypeError(
                    f"Variable type {var_type} not supported! Only jets and tracks!"
                ) from error

        # Add the flavour to the histogram
        histo_plot.add(
            histogram(
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

    # Defining two linestyles for the resampling variables
    linestyles = ["-", "--"]

    # Translate the kwargs to new naming scheme
    kwargs = translate_kwargs(kwargs)

    # Deactivate the ratio panel
    kwargs["n_ratio_panels"] = 0

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
        histo_plot = histogram_plot(
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
                    histogram(
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
            ),
            **kwargs,
        )


def preprocessing_plots(
    sample: str,
    var_dict: dict,
    class_labels: list,
    plots_dir: str,
    use_random_jets: bool = False,
    jet_collection: str = "jets",
    track_collection_list: list = None,
    nJets: int = 3e4,
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
        ensure correct mixing. Otherwise the first nJets are
        used for plotting, by default False
    jet_collection : str, optional
        Name of the jet collection, by default "jets"
    track_collection_list : list, optional
        List of str of the track collections which are to be
        plotted, by default None
    nJets : int, optional
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

    # Get max number of available jets
    with h5py.File(sample, "r") as f:
        try:
            nJets_infile = len(f["/jets"])

        except KeyError:
            nJets_infile = len(f["/X_train"])

    # Check if random values are used or not
    if use_random_jets is True:

        # Get a random generator with specified seed
        rng = np.random.default_rng(seed=seed)

        # Mix the chunks
        selected_indicies = sorted(
            rng.choice(
                np.arange(nJets_infile, dtype=int),
                int(nJets),
                replace=False,
            )
        )

    else:

        # if number of requested jets is larger that what is available,
        # plot all available jets.
        if nJets > nJets_infile:
            logger.warning(
                f"You requested {nJets} jets,"
                f"but there are only {nJets_infile} jets in the input!"
            )
        selected_indicies = np.arange(min(nJets, nJets_infile), dtype=int)

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
            labels = infile["/labels"][selected_indicies]

        except KeyError:
            labels = infile["Y_train"][selected_indicies]

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
                jets = pd.DataFrame(
                    infile["/jets"].fields(jet_var_list)[selected_indicies]
                )

            except KeyError:
                jets = np.asarray(infile["X_train"][selected_indicies])

            # Loop over variables
            for jet_var_counter, jet_var in enumerate(jet_var_list):

                # Plotting
                plot_variable(
                    df=jets,
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
            noNormVars = var_dict["track_train_variables"][track_collection][
                "noNormVars"
            ]
            logNormVars = var_dict["track_train_variables"][track_collection][
                "logNormVars"
            ]
            jointNormVars = var_dict["track_train_variables"][track_collection][
                "jointNormVars"
            ]
            trksVars = noNormVars + logNormVars + jointNormVars

            # Get the tracks from file
            try:
                tracks = np.asarray(infile[f"/{track_collection}"][selected_indicies])

            except KeyError:
                tracks = np.asarray(
                    infile[f"X_{track_collection}_train"][selected_indicies]
                )

            # Loop over track variables
            for trk_var_counter, trk_var in enumerate(trksVars):

                # Plotting
                plot_variable(
                    df=tracks,
                    labels=labels,
                    variable=trk_var,
                    variable_index=trk_var_counter,
                    var_type="tracks",
                    class_labels=class_labels,
                    output_dir=os.path.join(plots_dir, track_collection),
                    **kwargs,
                )
