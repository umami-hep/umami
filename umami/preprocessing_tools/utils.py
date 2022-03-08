"""Collection of utility functions for preprocessing tools."""
import os

import h5py
import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import global_config, logger
from umami.helper_tools import hist_w_unc
from umami.tools import applyATLASstyle, makeATLAStag, yaml_loader


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
    binning: dict = None,
    figsize: list = None,
    normed: bool = True,
    fileformat: str = "pdf",
    UseAtlasTag: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "$\\sqrt{s}=13$ TeV, PFlow Jets",
    y_scale: float = 1.3,
    yAxisAtlasTag: float = 0.9,
    leg_loc: str = "upper right",
    label_fontsize: int = 12,
    leg_fontsize: int = 10,
    leg_ncol: int = 1,
    logy: bool = True,
    **kwargs,  # pylint: disable=unused-argument
):
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
    binning : dict, optional
        Dict with the variables as keys and binning as item,
        by default None
    figsize : list, optional
        List with the size of the figure, by default None
    normed : bool, optional
        Normalise the flavours, by default True
    fileformat : str, optional
        Fileformat of the plots, by default "pdf"
    UseAtlasTag : bool, optional
        Use a ATLAS tag, by default True
    AtlasTag : str, optional
        First line of ATLAS tag, by default "Internal Simulation"
    SecondTag : str, optional
        Second line of ATLAS tag, by default "$sqrt{s}=13$ TeV, PFlow Jets"
    y_scale : float, optional
        Increase the y-axis to fit the ATALS tag in, by default 1.3
    yAxisAtlasTag : float, optional
        Relative y axis position of the ATLAS Tag, by default 0.9
    leg_loc : str, optional
        Position of the legend in the plot, by default "upper right"
    label_fontsize : int, optional
        Fontsize of the axis labels, by default 12
    leg_fontsize : int, optional
        Fontsize of the legend, by default 10
    leg_ncol : int, optional
        Number of columns in the legend, by default 1
    logy : bool, optional
        Plot a logarithmic y-axis, by default True
    **kwargs : kwargs
        kwargs from `plot_object`

    Raises
    ------
    TypeError
        If the given variable type is not supported.
    """

    # Check if binning is given. If not, init an empty dict
    if not binning:
        binning = {}

    # Check if figsize is given. If not, init default size
    if not figsize:
        figsize = [11.69 * 0.8, 8.27 * 0.8]

    # Set ATLAS plot style
    applyATLASstyle(mtp)

    # Give a debug logger
    logger.debug(f"Plotting variable {variable}...")

    # Get the binning
    try:
        _, bins = np.histogram(
            a=np.nan_to_num(df[variable]),
            bins=binning[variable]
            if variable in binning and binning is not None
            else 50,
        )

    except IndexError as Error:
        if var_type.casefold() == "jets":
            array = np.nan_to_num(df[:, variable_index])

        elif var_type.casefold() == "tracks":
            array = np.nan_to_num(df[:, :, variable_index])

        else:
            raise TypeError(
                f"Variable type {var_type} not supported! Only jets and tracks!"
            ) from Error

        _, bins = np.histogram(
            a=array,
            bins=binning[variable]
            if variable in binning and binning is not None
            else 50,
        )

    # Init a new figure
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    ax = fig.subplots()

    # Loop over the flavours
    for flav_counter, flavour in enumerate(class_labels):

        # Get all jets with the correct flavour
        try:
            flavour_jets = df[variable][labels[:, flav_counter] == 1].values

        except AttributeError:
            flavour_jets = df[variable][labels[:, flav_counter] == 1]

        except IndexError as Error:
            if var_type.casefold() == "jets":
                flavour_jets = df[:, variable_index][
                    labels[:, flav_counter] == 1
                ].flatten()

            elif var_type.casefold() == "tracks":
                flavour_jets = df[:, :, variable_index][labels[:, flav_counter] == 1]

            else:
                raise TypeError(
                    f"Variable type {var_type} not supported! Only jets and tracks!"
                ) from Error

        # Calculate bins
        hist_bins, weights, unc, band = hist_w_unc(
            a=flavour_jets,
            bins=bins,
            normed=normed,
        )

        # Plot the bins
        ax.hist(
            x=hist_bins[:-1],
            bins=hist_bins,
            weights=weights,
            histtype="step",
            linewidth=1.0,
            color=global_config.flavour_categories[flavour]["colour"],
            stacked=False,
            fill=False,
            label=global_config.flavour_categories[flavour]["legend_label"],
        )

        # Plot uncertainty
        ax.hist(
            x=hist_bins[:-1],
            bins=hist_bins,
            bottom=band,
            weights=unc * 2,
            label="stat. unc." if flavour == class_labels[-1] else None,
            **global_config.hist_err_style,
        )

    # Set xlabel
    ax.set_xlabel(
        variable,
        fontsize=label_fontsize,
        horizontalalignment="right",
        x=1.0,
    )

    if normed:
        ax.set_ylabel(
            "Normalised Number of Jets",
            fontsize=label_fontsize,
            horizontalalignment="right",
            y=1.0,
        )

    else:
        ax.set_ylabel(
            "Number of Jets",
            fontsize=label_fontsize,
            horizontalalignment="right",
            y=1.0,
        )

    # Set logscale for y axis
    if logy is True:
        ax.set_yscale("log")

        # Increase ymax so atlas tag don't cut plot
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(
            ymin,
            ymax * np.log(ymax / ymin) * 10 * y_scale,
        )

    else:

        # Increase ymax so atlas tag don't cut plot
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=y_scale * ymax)

    # ATLAS tag
    if UseAtlasTag is True:
        makeATLAStag(
            ax=ax,
            fig=fig,
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    # Set legend
    ax.legend(
        loc=leg_loc,
        ncol=leg_ncol,
        fontsize=leg_fontsize,
    )

    # Set the tight layout
    plt.tight_layout()

    # Save figure and clean it.
    plt.savefig(
        os.path.join(
            output_dir,
            f"{variable}.{fileformat}",
        )
    )
    plt.close()
    plt.clf()


def preprocessing_plots(
    sample: str,
    var_dict: dict,
    class_labels: list,
    plots_dir: str,
    use_random_jets: bool = False,
    jet_collection: str = "jets",
    track_collection_list: list = None,
    nJets: int = 3e4,
    figsize: list = None,
    seed: int = 42,
    **kwargs,  # pylint: disable=unused-argument
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
    figsize : list, optional
        List with the size of the figure, by default None
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


def ResamplingPlots(
    concat_samples: dict,
    positions_x_y: list = None,
    variable_names: list = None,
    plot_base_name: str = "plots/resampling-plot",
    binning: dict = None,
    Log: bool = True,
    after_sampling: bool = False,
    normalised: bool = False,
    use_weights: bool = False,
    hist_input: bool = False,
    second_tag: str = "",
    fileformat: str = "pdf",
):
    """
    Plots pt and eta distribution as nice plots for presentation.

    Parameters
    ----------
    concat_samples : dict
        dict with the format given in the Undersampling class by the class object
        `concat_samples` or the `x_y_after_sampling` depending on the `after_sampling`
        option
    positions_x_y :  list
        The position where the variables are stored the sub-dict `jets`
    variable_names : list
        The name of the 2 variables which will be plotted
    plot_base_name : str
        Folder and name of the plot w/o extension, this will be appened as well as
        the variable name
    binning : dict
        dict of the bin_edges used for plotting
    Log : bool
        boolean indicating if plot is in log scale or not (default True)
    after_sampling: bool
        If False (default) using the syntax of `concat_samples`
    normalised: bool
        Normalises the integral of the histogram to 1 (default False)
    use_weights: bool
        If True, the weights are used for the histogram (default False)
    hist_input: bool
        If True the concat_samples is a dictionary of histograms and binning is
        already the full axes (default False)
    second_tag: str
        Second tag which is inserted below the ATLAS tag (using the makeATLAStag
        function)
    fileformat : str
        Fileending of the plot. Default is "pdf"
    """
    if positions_x_y is None:
        positions_x_y = [0, 1]
    if variable_names is None:
        variable_names = ["pT", "abseta"]
    if binning is None:
        binning = {
            "pT": np.linspace(10000, 2000000, 200),
            "abseta": np.linspace(0, 2.5, 26),
        }

    applyATLASstyle(mtp)

    for varname, varpos in zip(variable_names, positions_x_y):
        # Loop over flavours
        plt.figure()
        for flav in concat_samples:
            if normalised:
                norm_factor = (
                    len(concat_samples[flav])
                    if after_sampling
                    else len(concat_samples[flav]["jets"])
                )
            else:
                norm_factor = 1.0

            scale_val = 1

            if varname in ["pT", "pt_btagJes"] or varname == global_config.pTvariable:
                scale_val = 1e3

            if hist_input:
                # Working directly on the x-D array
                direction_sum = tuple(  # pylint: disable=R1728
                    [i for i in positions_x_y if i != varpos]
                )
                counts = np.sum(concat_samples[flav], axis=direction_sum)
                Bins = binning[varname] / scale_val

            else:
                # Calculate Binning and counts for plotting
                counts, Bins = np.histogram(
                    concat_samples[flav][:, varpos] / scale_val
                    if after_sampling
                    else concat_samples[flav]["jets"][:, varpos] / scale_val,
                    bins=binning[varname],
                    weights=concat_samples[flav]["weight"] if use_weights else None,
                )

            # Calculate the bin centers
            bincentres = [(Bins[i] + Bins[i + 1]) / 2.0 for i in range(len(Bins) - 1)]
            # Calculate poisson uncertainties and lower bands
            unc = np.sqrt(counts) / norm_factor
            band_lower = counts / norm_factor - unc

            plt.hist(
                x=Bins[:-1],
                bins=Bins,
                weights=(counts / norm_factor),
                histtype="step",
                linewidth=1.0,
                color=global_config.flavour_categories[flav]["colour"],
                stacked=False,
                fill=False,
                label=global_config.flavour_categories[flav]["legend_label"],
            )

            plt.hist(
                x=bincentres,
                bins=Bins,
                bottom=band_lower,
                weights=unc * 2,
                **global_config.hist_err_style,
            )

        if Log is True:
            plt.yscale("log")
            ymin, ymax = plt.ylim()

            if varname == "pT":
                plt.ylim(ymin=ymin, ymax=100 * ymax)

            else:
                plt.ylim(ymin=ymin, ymax=10 * ymax)

        elif Log is False:
            ymin, ymax = plt.ylim()
            plt.ylim(ymin=ymin, ymax=1.2 * ymax)

        if varname == global_config.pTvariable:
            plt.xlabel(r"$p_T$ in GeV")

        elif varname == global_config.etavariable:
            plt.xlabel(r"$\eta$")
        else:
            plt.xlabel(varname)

        plt.ylabel(r"Number of Jets")
        plt.legend(loc="upper right")

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag="Internal Simulation",
            second_tag=second_tag,
            ymax=0.9,
        )

        plt.tight_layout()
        if not os.path.exists(os.path.abspath("./plots")):
            os.makedirs(os.path.abspath("./plots"))

        plt.savefig(f"{plot_base_name}{varname}.{fileformat}")
        plt.close()
        plt.clf()


def generate_process_tag(
    preparation_ntuples_keys: list,
) -> str:
    """
    Builds a tag that contains the used processes (e.g. Z' and ttbar)
    which can then be used for plots.

    Parameters
    ----------
    preparation_ntuples_keys : dict_keys
        Dict keys from the preparation.ntuples section of the preprocessing
        config

    Returns
    -------
    second_tag_for_plot : str
        String which is used as 'second_tag' parameter  by the makeATLAStag
        function.

    Raises
    ------
    KeyError
        If the plot label for the process was not found.
    """

    # Loop over the keys in the "preparation.ntuples" section of the
    # preprocessing config. For each process, try to extract the
    # corresponding plot label from the global config and add it to
    # the string that is inserted as "second_tag" in the plot
    processes = ""
    combined_sample = False
    logger.info("Looking for different processes in preprocessing config.")
    for process in preparation_ntuples_keys:
        try:
            label = global_config.process_labels[process]["label"]
            if processes == "":
                processes += f"{label}"
            else:
                combined_sample = True
                processes += f" + {label}"
            logger.info(f"Found the process '{process}' with the label '{label}'")
        except KeyError as Error:
            raise KeyError(
                f"Plot label for the process {process} was not"
                "found. Make sure your entries in the 'ntuples'"
                "section are valid entries that have a matching entry"
                "in the global config."
            ) from Error
    # Combine the string that contains the latex code for the processes
    # and the "sqrt(s)..." and "PFlow Jets" part
    if combined_sample is True:
        second_tag_for_plot = r"$\sqrt{s}$ = 13 TeV, Combined " + processes
    else:
        second_tag_for_plot = r"$\sqrt{s}$ = 13 TeV, " + processes

    second_tag_for_plot += " PFlow Jets"

    return second_tag_for_plot
