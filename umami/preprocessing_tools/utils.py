"""Collection of utility functions for preprocessing tools."""
import os

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import global_config, logger
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
