"""Plotting functions for NN training."""
import copy
import os

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from umami.configuration import global_config, logger
from umami.data_tools import LoadJetsFromFile
from umami.metrics import GetRejection
from umami.preprocessing_tools import GetBinaryLabels
from umami.tools import applyATLASstyle, check_main_class_input, makeATLAStag


def plot_validation_files(
    metric_identifier: str,
    df_results: dict,
    label_prefix: str = "",
    label_suffix: str = "",
    val_files: dict = None,
):
    """Helper function which loops over the validation files and plots the chosen
    metric for each epoch. Meant to be called in other plotting functions.

    The fuction loops over the validation files and plots a line with the label
    f"{label_prefix}{validation_file_label}{label_suffix}" for each file.

    Parameters
    ----------
    metric_identifier : str
        Identifier for the metric you want to plot, e.g. "val_loss" or "disc_cut".
    df_results : dict
        Dict which contains the results of the training.
    label_prefix : str, optional
        This string is put at the beginning of the plot label for each line.
        For accuracy for example choose "validation accuracy - ", by default ""
    label_suffix : str, optional
        This string is put at the end of the plot label for each line, by default ""
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    """

    if val_files is None:
        logger.warning(
            f"No validation files provided --> not plotting {metric_identifier}."
        )
    else:
        for val_file_identifier, val_file_config in val_files.items():
            plt.plot(
                df_results["epoch"],
                df_results[f"{metric_identifier}_{val_file_identifier}"],
                label=f"{label_prefix}{val_file_config['label']}{label_suffix}",
            )


def CompTaggerRejectionDict(
    file: str,
    unique_identifier: str,
    tagger_comp_var: list,
    recommended_frac_dict: dict,
    nJets: int,
    WP: float,
    class_labels: list,
    main_class: str,
    cut_vars_dict: dict = None,
):
    """Load the comparison tagger probability variables from the validation
    file and calculate the rejections.

    Parameters
    ----------
    file : str
        Filename of the validation file.
    unique_identifier: str
        Unique identifier of the used dataset (e.g. ttbar_r21)
    tagger_comp_var : list
        List of the comparison tagger probability variable names.
    recommended_frac_dict : dict
        Dict with the fractions.
    nJets : int
        Number of jets to use for calculation of the comparison tagger
        rejections
    WP : float
        Working point at which the rejections should be evaluated.
    class_labels : list
        List with the used class_labels.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    cut_vars_dict : dict
        Dict with the cut variables and the values for selecting jets.

    Returns
    -------
    dict
        Dict with the rejections for against the main class for all given flavours.
    """

    df, labels = LoadJetsFromFile(
        filepath=file,
        class_labels=class_labels,
        nJets=nJets,
        variables=tagger_comp_var,
        cut_vars_dict=cut_vars_dict,
        print_logger=False,
        chunk_size=1e6,
    )

    # Binarize the labels
    y_true = GetBinaryLabels(labels)

    # Calculate rejections
    recomm_rej_dict, _ = GetRejection(
        y_pred=df[tagger_comp_var].values,
        y_true=y_true,
        unique_identifier=unique_identifier,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=recommended_frac_dict,
        target_eff=WP,
    )

    return recomm_rej_dict


def PlotDiscCutPerEpoch(
    df_results: dict,
    plot_name: str,
    frac_class: str,
    val_files: dict = None,
    trained_taggers: list = None,
    target_beff: float = 0.77,
    frac: float = 0.018,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.3,
    ncol: int = 1,
    plot_datatype: str = "pdf",
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the discriminant cut value for a specific working point
    over all epochs.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and disc cuts.
    plot_name : str
        Path where the plots is saved + plot name.
    frac_class : str
        Define which fraction is shown in ATLAS Tag.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    trained_taggers : list, optional
        List of trained taggers, by default None
    target_beff : float, optional
        Working Point to use, by default 0.77
    frac : float, optional
        Fraction value for ATLAS Tag, by default 0.018
    UseAtlasTag : bool, optional
        Define if ATLAS Tag is used or not, by default True
    ApplyATLASStyle : bool, optional
        Apply ATLAS Style of the plot (for approval etc.), by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.3
    ncol : int, optional
        Number of columns in the legend, by default 1
    plot_datatype : str, optional
        Datatype of the plot, by default "pdf"
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut",
        df_results=df_results,
        label_prefix="",
        label_suffix=" validation sample",
        val_files=val_files,
    )

    if UseAtlasTag is True:
        SecondTag = (
            SecondTag
            + f"\n{frac_class} fraction = {frac}"
            + f"\nWP={int(target_beff * 100):02d}%"
        )

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.legend(loc="upper right")
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)
    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel(
        r"$b$-Tagging discriminant Cut Value",
        fontsize=12,
        horizontalalignment="right",
        y=1.0,
    )
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotDiscCutPerEpochUmami(
    df_results: dict,
    plot_name: str,
    frac_class: str,
    val_files: dict = None,
    target_beff: float = 0.77,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag: float = 0.9,
    yAxisIncrease: float = 1.35,
    ncol: int = 1,
    plot_datatype: str = "pdf",
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the discriminant cut value for a specific working point over all epochs.
    DIPS and Umami are both shown.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and disc cuts.
    plot_name : str
        Path where the plots is saved + plot name.
    frac_class : str
        Define which fraction is shown in ATLAS Tag
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    target_beff : float, optional
        Working Point to use, by default 0.77
    UseAtlasTag : bool, optional
        Define if ATLAS Tag is used or not, by default True
    ApplyATLASStyle : bool, optional
        Apply ATLAS Style of the plot (for approval etc.), by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.35
    ncol : int, optional
        Number of columns in the legend, by default 1
    plot_datatype : str, optional
        Datatype of the plot, by default "pdf"
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut_umami",
        df_results=df_results,
        label_prefix="Umami - ",
        label_suffix=" validation sample",
        val_files=val_files,
    )
    plot_validation_files(
        metric_identifier="disc_cut_dips",
        df_results=df_results,
        label_prefix="DIPS - ",
        label_suffix=" validation sample",
        val_files=val_files,
    )

    if UseAtlasTag is True:
        SecondTag = SecondTag + f"\nWP={int(target_beff * 100):02d}%"

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.legend(loc="upper right")
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=yAxisIncrease * ymax)
    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel(
        r"$b$-Tagging discriminant Cut Value",
        fontsize=12,
        horizontalalignment="right",
        y=1.0,
    )
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotRejPerEpochComparison(
    df_results: dict,
    tagger_label: str,
    comp_tagger_rej_dict: dict,
    unique_identifier: str,
    plot_name: str,
    class_labels: list,
    main_class: str,
    label_extension: str,
    rej_string: str,
    taggers_from_file: dict = None,
    trained_taggers: list = None,
    target_beff: float = 0.77,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag: float = 0.95,
    yAxisIncrease: float = 1.1,
    ncol: int = 1,
    figsize: list = None,
    legend_loc: str = "upper right",
    plot_datatype: str = "pdf",
    legFontSize: int = 10,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plotting the Rejections per Epoch for the trained tagger and the provided
    comparison taggers.

    This is only available for two rejections at once due to limiting of axes.

    Parameters
    ----------
    df_results : dict
        Dict with the rejections of the trained tagger.
    tagger_label : str
        Name of trained tagger.
    comp_tagger_rej_dict : dict
        Dict with the rejections of the comp taggers.
    unique_identifier : str
        Unique identifier of the used dataset (e.g. ttbar_r21).
    plot_name : str
        Path where the plots is saved.
    class_labels : list
        A list of the class_labels which are used
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    label_extension : str
        Extension of the legend label giving the process type.
    rej_string : str
        String that is added after the class for the key.
    taggers_from_file : dict
        Dict with the comparison taggers as keys and their labels for
        the plots as values, by default None
    trained_taggers : list, optional
        List of dicts with needed info about local available taggers, by default None
    target_beff : float, optional
        Target Working point., by default 0.77
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.95
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.1
    ncol : int, optional
        Number of columns in the legend., by default 1
    figsize : list, optional
        Size of the figure., by default None
    legend_loc : str, optional
        Position of the legend., by default "upper right"
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    legFontSize : int, optional
        Fontsize of the legend., by default 10
    **kwargs
        Arbitrary keyword arguments.
    """
    if figsize is None:
        figsize = [8, 6]

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a deep copy of the class labels as set
    class_labels_wo_main = copy.deepcopy(set(class_labels))

    # Remove the main classes from the copy
    class_labels_wo_main.difference_update(main_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories
    linestyle_list = [
        "-",
        (0, (5, 1)),
        "--",
        (0, (1, 1)),
        (0, (5, 10)),
    ]

    # Set global plot configs
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot()
    ax1.set_xlabel("Epoch")
    ax2 = ax1.twinx()
    axes = [ax1, ax2]

    # Define a list for the lines which are plotted
    lines = []

    for counter, iter_class in enumerate(class_labels_wo_main):
        # Init a linestyle counter
        counter_models = 0

        # Plot rejection
        lines = lines + axes[counter].plot(
            df_results["epoch"],
            df_results[f"{iter_class}_{rej_string}"],
            linestyle=linestyle_list[counter],
            color=f"C{counter_models}",
            label=tagger_label,
        )

        # Set up the counter
        counter_models += 1

        # Set y label
        axes[counter].set_ylabel(
            f'{flav_cat[iter_class]["legend_label"]} Rejection',
        )

        if comp_tagger_rej_dict is None:
            logger.info("No comparison tagger defined. Not plotting those!")

        else:
            for _, comp_tagger in enumerate(comp_tagger_rej_dict):
                try:
                    tmp_line = axes[counter].axhline(
                        comp_tagger_rej_dict[comp_tagger][
                            f"{iter_class}_rej_{unique_identifier}"
                        ],
                        0,
                        df_results["epoch"].max(),
                        color=f"C{counter_models}",
                        linestyle=linestyle_list[counter],
                        label=taggers_from_file[comp_tagger]
                        if taggers_from_file
                        else comp_tagger,
                    )

                    # Set up the counter
                    counter_models += 1

                    # Add the horizontal line to the lines list
                    lines += [tmp_line]

                except KeyError:
                    logger.info(
                        f"{iter_class} rejection for {comp_tagger} and file "
                        f"{unique_identifier} not in dict! Skipping ..."
                    )

        if trained_taggers is None:
            if counter == 0:
                logger.debug("No local taggers defined. Not plotting those!")

        else:
            for _, tt in enumerate(trained_taggers):
                try:
                    # Get the needed rejection info from json
                    tt_rej_dict = pd.read_json(trained_taggers[tt]["path"])

                except FileNotFoundError(
                    f'No .json file found at {trained_taggers[tt]["path"]}.'
                    f' Skipping {trained_taggers[tt]["label"]}'
                ):
                    continue

                lines = lines + axes[counter].plot(
                    tt_rej_dict["epoch"],
                    tt_rej_dict[f"{iter_class}_{rej_string}"],
                    linestyle=linestyle_list[counter],
                    color=f"C{counter_models}",
                    label=trained_taggers[tt]["label"],
                )

                # Set up the counter
                counter_models += 1

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Increase y limit for ATLAS Logo
    ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)
    ax2.set_ylim(top=ax2.get_ylim()[1] * yAxisIncrease)

    # Create the two legends for rejection and model
    line_list_rej = []

    # Get lines for each rejection
    for counter, iter_class in enumerate(class_labels_wo_main):
        line = ax1.plot(
            np.nan,
            np.nan,
            color="k",
            label=f'{flav_cat[iter_class]["legend_label"]} rejection',
            linestyle=linestyle_list[counter],
        )
        line_list_rej += line

    # Get the middle legend
    legend1 = ax1.legend(
        handles=line_list_rej,
        labels=[tmp.get_label() for tmp in line_list_rej],
        loc="upper center",
        fontsize=legFontSize,
        ncol=ncol,
    )

    # Add the second legend to plot
    ax1.add_artist(legend1)

    # Get the labels for the legends
    labels_list = []
    lines_list = []

    for line in lines:
        if line.get_label() not in labels_list:
            labels_list.append(line.get_label())
            lines_list.append(line)

    if UseAtlasTag is True:
        SecondTag = SecondTag + f"\nWP={int(target_beff * 100):02d}% "

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag + label_extension + " sample",
            ymax=yAxisAtlasTag,
        )

    # Define the legend
    ax1.legend(
        handles=lines_list,
        labels=labels_list,
        loc=legend_loc,
        fontsize=legFontSize,
        ncol=ncol,
    )
    plt.tight_layout()
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotRejPerEpoch(
    df_results: dict,
    tagger_label: str,
    comp_tagger_rej_dict: dict,
    unique_identifier: str,
    plot_name: str,
    class_labels: list,
    main_class: str,
    label_extension: str,
    rej_string: str,
    taggers_from_file: dict = None,
    trained_taggers: list = None,
    target_beff: float = 0.77,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag: float = 0.95,
    yAxisIncrease: float = 1.1,
    ncol: int = 1,
    figsize: list = None,
    legend_loc: str = "upper right",
    plot_datatype: str = "pdf",
    legFontSize: int = 10,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plotting the Rejections per Epoch for the trained tagger and the provided
    comparison taggers in separate plots. One per rejection.

    Parameters
    ----------
    df_results : dict
        Dict with the rejections of the trained tagger.
    tagger_label : str
        Name of trained tagger.
    comp_tagger_rej_dict : dict
        Dict with the rejections of the comp taggers.
    unique_identifier: str
        Unique identifier of the used dataset (e.g. ttbar_r21).
    plot_name : str
        Path where the plots is saved.
    class_labels : list
        A list of the class_labels which are used
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    label_extension : str
        Extension of the legend label giving the process type.
    rej_string : str
        String that is added after the class for the key.
    taggers_from_file : dict
        Dict with the comparison taggers as keys and their labels for
        the plots as values, by default None
    trained_taggers : list, optional
        List of dicts with needed info about local available taggers, by default None
    target_beff : float, optional
        Target Working point., by default 0.77
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.95
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.1
    ncol : int, optional
        Number of columns in the legend., by default 1
    figsize : list, optional
        Size of the figure., by default None
    legend_loc : str, optional
        Position of the legend., by default "upper right"
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    legFontSize : int, optional
        Fontsize of the legend., by default 10
    **kwargs
        Arbitrary keyword arguments.
    """
    if figsize is None:
        figsize = [8, 6]
    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a deep copy of the class labels as set
    class_labels_wo_main = copy.deepcopy(set(class_labels))

    # Remove the main classes from the copy
    class_labels_wo_main.difference_update(main_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories

    for _, iter_class in enumerate(class_labels_wo_main):
        # Set global plot configs
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot()
        axes.set_xlabel("Epoch")
        axes.set_ylabel(f'{flav_cat[iter_class]["legend_label"]} Rejection')

        # Init a linestyle counter
        counter_models = 0

        # Plot rejection
        axes.plot(
            df_results["epoch"],
            df_results[f"{iter_class}_{rej_string}"],
            linestyle="-",
            color=f"C{counter_models}",
            label=tagger_label,
        )

        # Set up the counter
        counter_models += 1

        if comp_tagger_rej_dict is None:
            logger.info("No comparison tagger defined. Not plotting those!")

        else:
            for _, comp_tagger in enumerate(comp_tagger_rej_dict):
                try:
                    axes.axhline(
                        comp_tagger_rej_dict[comp_tagger][
                            f"{iter_class}_rej_{unique_identifier}"
                        ],
                        0,
                        df_results["epoch"].max(),
                        color=f"C{counter_models}",
                        linestyle="-",
                        label=taggers_from_file[comp_tagger]
                        if taggers_from_file
                        else comp_tagger,
                    )

                    # Set up the counter
                    counter_models += 1

                except KeyError:
                    logger.info(
                        f"{iter_class} rejection for {comp_tagger} not in"
                        " dict! Skipping ..."
                    )

        if trained_taggers is None:
            logger.debug("No local taggers defined. Not plotting those!")

        else:
            for _, tt in enumerate(trained_taggers):
                try:
                    # Get the needed rejection info from json
                    tt_rej_dict = pd.read_json(trained_taggers[tt]["path"])

                except FileNotFoundError(
                    f'No .json file found at {trained_taggers[tt]["path"]}.'
                    f' Skipping {trained_taggers[tt]["label"]}'
                ):
                    continue

                axes.plot(
                    tt_rej_dict["epoch"],
                    tt_rej_dict[f"{iter_class}_{rej_string}"],
                    linestyle="-",
                    color=f"C{counter_models}",
                    label=trained_taggers[tt]["label"],
                )

                # Set up the counter
                counter_models += 1

        # Increase y limit for ATLAS Logo
        axes.set_ylim(top=axes.get_ylim()[1] * yAxisIncrease)

        if UseAtlasTag is True:
            makeATLAStag(
                ax=plt.gca(),
                fig=plt.gcf(),
                first_tag=AtlasTag,
                second_tag=(
                    SecondTag
                    + f"\nWP={int(target_beff * 100):02d}% "
                    + label_extension
                    + " sample"
                ),
                ymax=yAxisAtlasTag,
            )

        # Define the legend
        axes.legend(
            loc=legend_loc,
            fontsize=legFontSize,
            ncol=ncol,
        )
        plt.tight_layout()
        plt.savefig(
            plot_name + f"_{iter_class}_rejection.{plot_datatype}",
            transparent=True,
        )
        plt.cla()
        plt.clf()


def PlotLosses(
    df_results: dict,
    plot_name: str,
    train_history_dict: dict,
    val_files: dict = None,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease: float = 1.2,
    yAxisAtlasTag: float = 0.9,
    plot_datatype: str = "pdf",
    ymin: float = None,
    ymax: float = None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training loss and the validation losses per epoch.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and losses.
    plot_name : str
        Path where the plots is saved.
    train_history_dict : dict
        Dict that stores the results of the training.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.2
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    ymin : float, optional
        Manually set ymin. Overwrites yAxisIncrease, by default None
    ymax : float, optional
        Manually set ymax. Overwrites yAxisIncrease, by default None
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # plots training loss
    if "loss" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["loss"],
            label="training loss - hybrid sample",
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_loss",
        df_results=df_results,
        label_prefix="validation loss - ",
        label_suffix=" sample",
        val_files=val_files,
    )

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.legend(loc="upper right")
    old_ymin, old_ymax = plt.ylim()
    if ymin is not None:
        plt.ylim(ymin=ymin, ymax=old_ymax)

    if ymax is not None:
        plt.ylim(ymin=old_ymin, ymax=ymax)

    if ymin is None and ymax is None:
        plt.ylim(ymin=old_ymin, ymax=yAxisIncrease * old_ymax)

    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss", fontsize=12, horizontalalignment="right", y=1.0)
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuracies(
    df_results: dict,
    plot_name: str,
    train_history_dict: dict,
    val_files: dict = None,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease: float = 1.2,
    yAxisAtlasTag: float = 0.9,
    plot_datatype: str = "pdf",
    ymin: float = None,
    ymax: float = None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training and validation accuracies per epoch.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and accuracies.
    plot_name : str
        Path where the plots is saved.
    train_history_dict : dict
        Dict that stores the results of the training.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.2
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    ymin : float, optional
        Manually set ymin. Overwrites yAxisIncrease, by default None
    ymax : float, optional
        Manually set ymax. Overwrites yAxisIncrease, by default None
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    if "accuracy" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["accuracy"],
            label="training accuracy - hybrid sample",
        )

    # loop over the validation files using the unique identifiers and plot the accuracy
    # for each file
    plot_validation_files(
        metric_identifier="val_acc",
        df_results=df_results,
        label_prefix="validation accuracy - ",
        label_suffix=" sample",
        val_files=val_files,
    )

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.legend(loc="upper right")
    old_ymin, old_ymax = plt.ylim()
    if ymin is not None:
        plt.ylim(ymin=ymin, ymax=old_ymax)

    if ymax is not None:
        plt.ylim(ymin=old_ymin, ymax=ymax)

    if ymin is None and ymax is None:
        plt.ylim(ymin=old_ymin, ymax=yAxisIncrease * old_ymax)

    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy", fontsize=12, horizontalalignment="right", y=1.0)
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotLossesUmami(
    df_results: dict,
    plot_name: str,
    train_history_dict: dict,
    val_files: dict = None,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease: float = 1.4,
    yAxisAtlasTag: float = 0.9,
    plot_datatype: str = "pdf",
    ymin: float = None,
    ymax: float = None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training loss and the validation losses per epoch for Umami model
    (with DIPS and Umami losses).

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and losses.
    plot_name : str
        Path where the plots is saved.
    train_history_dict : dict
        Dict that stores the results of the training.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.4
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    ymin : float, optional
        Manually set ymin. Overwrites yAxisIncrease, by default None
    ymax : float, optional
        Manually set ymax. Overwrites yAxisIncrease, by default None
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # Plot umami and dips training loss
    if "loss_umami" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["loss_umami"],
            label="training loss Umami - hybrid sample",
        )

    if "loss_dips" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["loss_dips"],
            label="training loss DIPS - hybrid sample",
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_loss_umami",
        df_results=df_results,
        label_prefix="validation loss Umami - ",
        label_suffix=" sample",
        val_files=val_files,
    )
    plot_validation_files(
        metric_identifier="val_loss_dips",
        df_results=df_results,
        label_prefix="validation loss DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
    )

    plt.legend(loc="upper right")
    old_ymin, old_ymax = plt.ylim()
    if ymin is not None:
        plt.ylim(ymin=ymin, ymax=old_ymax)

    if ymax is not None:
        plt.ylim(ymin=old_ymin, ymax=ymax)

    if ymin is None and ymax is None:
        plt.ylim(ymin=old_ymin, ymax=yAxisIncrease * old_ymax)

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss", fontsize=12, horizontalalignment="right", y=1.0)
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuraciesUmami(
    df_results: dict,
    plot_name: str,
    train_history_dict: dict,
    val_files: dict = None,
    UseAtlasTag: bool = True,
    ApplyATLASStyle: bool = True,
    AtlasTag: str = "Internal Simulation",
    SecondTag: str = "\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease: float = 1.4,
    yAxisAtlasTag: float = 0.9,
    plot_datatype: str = "pdf",
    ymin: float = None,
    ymax: float = None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training and validation accuracies per epoch for Umami model
    (with DIPS and Umami accuracies).

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and accuracies.
    plot_name : str
        Path where the plots is saved.
    train_history_dict : dict
        Dict that stores the results of the training.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    UseAtlasTag : bool, optional
        Bool to decide if you want the ATLAS tag, by default True
    ApplyATLASStyle : bool, optional
        Decide, if the ATLAS Plotting style is used, by default True
    AtlasTag : str, optional
        Main tag. Mainly "Internal Simulation", by default "Internal Simulation"
    SecondTag : str, optional
        Lower tag in the ATLAS label with infos,
        by default "$sqrt{s}=13$ TeV, PFlow jets"
    yAxisIncrease : float, optional
        Y axis increase factor to fit the ATLAS label, by default 1.4
    yAxisAtlasTag : float, optional
        Y axis position of the ATLAS label, by default 0.9
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    ymin : float, optional
        Manually set ymin. Overwrites yAxisIncrease, by default None
    ymax : float, optional
        Manually set ymax. Overwrites yAxisIncrease, by default None
    **kwargs
        Arbitrary keyword arguments.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # Plot umami and dips training loss
    if "accuracy_umami" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["accuracy_umami"],
            label="training accuracy Umami - hybrid sample",
        )

    if "accuracy_dips" in train_history_dict:
        plt.plot(
            df_results["epoch"],
            train_history_dict["accuracy_dips"],
            label="training accuracy DIPS - hybrid sample",
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_acc_umami",
        df_results=df_results,
        label_prefix="validation accuracy Umami - ",
        label_suffix=" sample",
        val_files=val_files,
    )
    plot_validation_files(
        metric_identifier="val_acc_dips",
        df_results=df_results,
        label_prefix="validation accuracy DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
    )

    plt.legend(loc="upper right")
    old_ymin, old_ymax = plt.ylim()
    if ymin is not None:
        plt.ylim(ymin=ymin, ymax=old_ymax)

    if ymax is not None:
        plt.ylim(ymin=old_ymin, ymax=ymax)

    if ymin is None and ymax is None:
        plt.ylim(ymin=old_ymin, ymax=yAxisIncrease * old_ymax)

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    plt.xlabel("Epoch", fontsize=12, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy", fontsize=12, horizontalalignment="right", y=1.0)
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def RunPerformanceCheck(
    train_config: object,
    tagger: str,
    tagger_comp_vars: dict = None,
    dict_file_name: str = None,
    train_history_dict_file_name: str = None,
    WP: float = None,
):
    """Loading the validation metrics from the trained model and calculate
    the metrics for the comparison taggers and plot them.

    Parameters
    ----------
    train_config : object
        Loaded train_config object.
    tagger : str
        String name of the tagger used.
    tagger_comp_vars : dict, optional
        Dict of the tagger probability variables as lists, by default None
    dict_file_name : str, optional
        Path to the json file with the per epoch metrics of the trained tagger,
        by default None
    train_history_dict_file_name : str, optional
        Path to the history file from the training, by default None
    WP : float, optional
        Working point to evaluate, by default None
    """

    logger.info(f"Running performance check for {tagger}.")

    # Load parameters from train config
    Eval_parameters = train_config.Eval_parameters_validation
    Val_settings = train_config.Validation_metrics_settings
    frac_dict = Eval_parameters["frac_values"]
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    recommended_frac_dict = Eval_parameters["frac_values_comp"]
    # Load the unique identifiers of the validation files and the corresponding plot
    # labels. These are used several times in this function
    val_files = train_config.validation_files

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a WP
    if WP is None:
        WP = Val_settings["WP"]

    # Get dict with training results from json
    tagger_rej_dict = pd.read_json(dict_file_name)

    # Check if history file exists
    if os.path.isfile(train_history_dict_file_name):
        train_history_dict = pd.read_json(train_history_dict_file_name)

    elif "accuracy" in tagger_rej_dict:
        logger.warning(
            "Metrics history file not found! Extract metrics from validation file."
        )

        train_history_dict = tagger_rej_dict

    else:
        logger.warning("Not training metrics found! Not plotting acc/loss!")
        train_history_dict = None

    if tagger_comp_vars is not None:
        # Dict
        comp_tagger_rej_dict = {}

        # Loop over taggers that are used for comparsion
        # After the double loop, the resulting dict looks something e.g. for the comp
        # taggers rnnip and dl1r like this:
        # {"rnnip": <dict_with_rej_values_rnnip>, "dl1r": <dict_with_rej_values_dl1r>}
        for comp_tagger in tagger_comp_vars:
            # loop over the different kinds of validation files specified in the
            # training config. Calculate the rejection values for each of the validation
            # files and add the values to the rejection dict
            comp_tagger_rej_dict[f"{comp_tagger}"] = {}
            for val_file_identifier, val_file_config in val_files.items():
                comp_tagger_rej_dict[f"{comp_tagger}"].update(
                    CompTaggerRejectionDict(
                        file=val_file_config["path"],
                        unique_identifier=val_file_identifier,
                        tagger_comp_var=tagger_comp_vars[comp_tagger],
                        recommended_frac_dict=recommended_frac_dict[comp_tagger],
                        nJets=Val_settings["n_jets"],
                        cut_vars_dict=val_file_config["variable_cuts"]
                        if "variable_cuts" in val_file_config
                        else None,
                        WP=WP,
                        class_labels=class_labels,
                        main_class=main_class,
                    )
                )

    else:
        # Define the dicts as None if compare tagger is False
        comp_tagger_rej_dict = None

    # Define dir where the plots are saved
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)

    # Get a deep copy of the class labels as set
    class_labels_wo_main = copy.deepcopy(set(class_labels))

    # Remove the main classes from the copy
    class_labels_wo_main.difference_update(main_class)

    n_rej = len(class_labels_wo_main)

    if tagger.casefold() == "umami" or tagger.casefold() == "umami_cond_att":
        for subtagger in ["umami", "dips"]:
            if n_rej == 2:
                for val_file_identifier, val_file_config in val_files.items():
                    # Plot comparsion for the comparison taggers
                    PlotRejPerEpochComparison(
                        df_results=tagger_rej_dict,
                        tagger_label=subtagger,
                        frac_dict=frac_dict,
                        comp_tagger_rej_dict=comp_tagger_rej_dict,
                        unique_identifier=val_file_identifier,
                        comp_tagger_frac_dict=recommended_frac_dict,
                        plot_name=(
                            f"{plot_dir}/rej-plot_val_{subtagger}_{val_file_identifier}"
                        ),
                        class_labels=class_labels,
                        main_class=main_class,
                        label_extension=val_file_config["label"],
                        rej_string=f"rej_{subtagger}_{val_file_identifier}",
                        target_beff=WP,
                        **Val_settings,
                    )

                    PlotRejPerEpoch(
                        df_results=tagger_rej_dict,
                        tagger_label=subtagger,
                        frac_dict=frac_dict,
                        unique_identifier=val_file_identifier,
                        comp_tagger_rej_dict=comp_tagger_rej_dict,
                        comp_tagger_frac_dict=recommended_frac_dict,
                        plot_name=(
                            f"{plot_dir}/rej-plot_val_{subtagger}_{val_file_identifier}"
                        ),
                        class_labels=class_labels,
                        main_class=main_class,
                        label_extension=val_file_config["label"],
                        rej_string=f"rej_{subtagger}_{val_file_identifier}",
                        target_beff=WP,
                        **Val_settings,
                    )

        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpochUmami(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            frac_class="cjets",
            target_beff=WP,
            **Val_settings,
        )

        # Check if metrics are present
        if train_history_dict is not None:
            plot_name = f"{plot_dir}/loss-plot"
            PlotLossesUmami(
                tagger_rej_dict,
                plot_name,
                train_history_dict=train_history_dict,
                val_files=val_files,
                **Val_settings,
            )
            plot_name = f"{plot_dir}/accuracy-plot"
            PlotAccuraciesUmami(
                tagger_rej_dict,
                plot_name,
                train_history_dict=train_history_dict,
                val_files=val_files,
                **Val_settings,
            )

    else:
        # If no freshly trained tagger label is given, give tagger
        if not (
            "tagger_label" in Val_settings and Val_settings["tagger_label"] is not None
        ):
            Val_settings["tagger_label"] = tagger

        if n_rej == 2:
            # Plot comparsion for the comparison taggers
            # Loop over validation files
            for val_file_identifier, val_file_config in val_files.items():
                PlotRejPerEpochComparison(
                    df_results=tagger_rej_dict,
                    frac_dict=frac_dict,
                    comp_tagger_rej_dict=comp_tagger_rej_dict,
                    comp_tagger_frac_dict=recommended_frac_dict,
                    unique_identifier=val_file_identifier,
                    plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                    class_labels=class_labels,
                    main_class=main_class,
                    label_extension=val_file_config["label"],
                    rej_string=f"rej_{val_file_identifier}",
                    target_beff=WP,
                    **Val_settings,
                )
        for val_file_identifier, val_file_config in val_files.items():
            # Plot rejections in one plot per rejection
            PlotRejPerEpoch(
                df_results=tagger_rej_dict,
                frac_dict=frac_dict,
                comp_tagger_rej_dict=comp_tagger_rej_dict,
                comp_tagger_frac_dict=recommended_frac_dict,
                unique_identifier=val_file_identifier,
                plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                class_labels=class_labels,
                main_class=main_class,
                label_extension=val_file_config["label"],
                rej_string=f"rej_{val_file_identifier}",
                target_beff=WP,
                **Val_settings,
            )

        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpoch(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            val_files=val_files,
            target_beff=WP,
            frac_class="cjets",
            frac=frac_dict["cjets"],
            **Val_settings,
        )

        # Check if metrics are present
        if train_history_dict is not None:
            plot_name = f"{plot_dir}/loss-plot"
            PlotLosses(
                tagger_rej_dict,
                plot_name,
                train_history_dict=train_history_dict,
                val_files=val_files,
                **Val_settings,
            )
            plot_name = f"{plot_dir}/accuracy-plot"
            PlotAccuracies(
                tagger_rej_dict,
                plot_name,
                train_history_dict=train_history_dict,
                val_files=val_files,
                **Val_settings,
            )
