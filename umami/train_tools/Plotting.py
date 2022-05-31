"""Plotting functions for NN training."""
# pylint: disable=invalid-name
# TODO: switch to new plotting API with pep8 conform naming
import copy
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from puma import PlotBase

from umami.configuration import global_config, logger
from umami.data_tools import LoadJetsFromFile
from umami.metrics import get_rejection
from umami.preprocessing_tools import GetBinaryLabels
from umami.tools import check_main_class_input


def plot_validation_files(
    metric_identifier: str,
    df_results: dict,
    ax,
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
    ax : matplotlib.axes.Axes
        Axis you want to plot the curves on. If None, the currently active axis
        is used. By default None
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
            ax.plot(
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
    working_point: float,
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
    working_point : float
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

    # Get a list with all per-jets variables loaded
    avai_variables = list(df.keys())

    # Init a bool if the rejection should be skipped
    Skip_rej_calc = False

    # Check if the tagger variables are available
    for tagger_var in tagger_comp_var:
        if tagger_var not in avai_variables:
            logger.warning(
                f"Tagger probability {tagger_var} not in validation file"
                f" {os.path.basename(file)}. Skipping ..."
            )
            Skip_rej_calc = True

    # Init an empty dict so the loop while plotting will not break
    if Skip_rej_calc:
        recomm_rej_dict = {}

    else:
        # Calculate rejections
        recomm_rej_dict, _ = get_rejection(
            y_pred=df[tagger_comp_var].values,
            y_true=y_true,
            unique_identifier=unique_identifier,
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=recommended_frac_dict,
            target_eff=working_point,
        )

    return recomm_rej_dict


def PlotDiscCutPerEpoch(
    df_results: dict,
    plot_name: str,
    frac_class: str,
    val_files: dict = None,
    target_beff: float = 0.77,
    frac: float = 0.018,
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
    target_beff : float, optional
        Working Point to use, by default 0.77
    frac : float, optional
        Fraction value for ATLAS Tag, by default 0.018
    plot_datatype : str, optional
        Datatype of the plot, by default "pdf"
    **kwargs
        Keyword arguments handed to the plotting API
    """
    disc_cut_plot = PlotBase(
        xlabel="Epoch",
        ylabel="$b$-tagging discriminant cut value",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    disc_cut_plot.initialise_figure()

    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut",
        df_results=df_results,
        label_prefix="",
        label_suffix=" validation sample",
        val_files=val_files,
        ax=disc_cut_plot.axis_top,
    )

    disc_cut_plot.atlas_second_tag += (
        f"\n{frac_class} fraction = {frac}\nWP={int(target_beff * 100):02d}%"
    )
    disc_cut_plot.initialise_plot()
    disc_cut_plot.savefig(f"{plot_name}.{plot_datatype}", transparent=True)


def PlotDiscCutPerEpochUmami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    target_beff: float = 0.77,
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
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    target_beff : float, optional
        Working Point to use, by default 0.77
    plot_datatype : str, optional
        Datatype of the plot, by default "pdf"
    **kwargs
        Keyword arguments handed to the plotting API
    """
    disc_cut_plot = PlotBase(
        xlabel="Epoch",
        ylabel="$b$-tagging discriminant cut value",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    disc_cut_plot.initialise_figure()
    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut_umami",
        df_results=df_results,
        label_prefix="Umami - ",
        label_suffix=" validation sample",
        val_files=val_files,
        ax=disc_cut_plot.axis_top,
    )
    plot_validation_files(
        metric_identifier="disc_cut_dips",
        df_results=df_results,
        label_prefix="DIPS - ",
        label_suffix=" validation sample",
        val_files=val_files,
        ax=disc_cut_plot.axis_top,
    )
    disc_cut_plot.atlas_second_tag += f"\nWP={int(target_beff * 100):02d}%"
    disc_cut_plot.initialise_plot()
    disc_cut_plot.savefig(f"{plot_name}.{plot_datatype}", transparent=True)


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
    plot_datatype: str = "pdf",
    leg_fontsize: int = 10,
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
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    leg_fontsize : int, optional
        Fontsize of the legend., by default 10
    **kwargs
        Keyword arguments handed to the plotting API
    """
    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a deep copy of the class labels
    class_labels_wo_main = copy.deepcopy(list(dict.fromkeys(class_labels)))

    # Remove the main classes from the copy
    for m_class in main_class:
        class_labels_wo_main.remove(m_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories
    linestyle_list = [
        "-",
        (0, (5, 1)),
        "--",
        (0, (1, 1)),
        (0, (5, 10)),
    ]

    rej_plot = PlotBase(
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    rej_plot.initialise_figure()

    ax_left = rej_plot.axis_top
    ax_right = ax_left.twinx()
    axes = [ax_left, ax_right]

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
        rej_plot.set_ylabel(
            ax_mpl=axes[counter],
            label=f'{flav_cat[iter_class]["legend_label"]} rejection',
            align_right=False,
        )

        if comp_tagger_rej_dict is None:
            logger.info("No comparison tagger defined. Not plotting those!")

        else:
            for _, comp_tagger in enumerate(comp_tagger_rej_dict):
                try:
                    tmp_line = axes[counter].axhline(
                        y=comp_tagger_rej_dict[comp_tagger][
                            f"{iter_class}_rej_{unique_identifier}"
                        ],
                        xmin=0,
                        xmax=df_results["epoch"].max(),
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

    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create the two legends for rejection and model
    line_list_rej = []

    # Get lines for each rejection
    for counter, iter_class in enumerate(class_labels_wo_main):
        line = ax_left.plot(
            np.nan,
            np.nan,
            color="k",
            label=f'{flav_cat[iter_class]["legend_label"]} rejection',
            linestyle=linestyle_list[counter],
        )
        line_list_rej += line

    # Get the middle legend
    legend1 = ax_left.legend(
        handles=line_list_rej,
        labels=[tmp.get_label() for tmp in line_list_rej],
        loc="upper center",
        fontsize=leg_fontsize,
        frameon=False,
    )

    # Add the second legend to plot
    ax_left.add_artist(legend1)

    # Get the labels for the legends
    labels_list = []
    lines_list = []

    for line in lines:
        if line.get_label() not in labels_list:
            labels_list.append(line.get_label())
            lines_list.append(line)

    rej_plot.atlas_second_tag += (
        f"\nWP={int(target_beff * 100):02d}% {label_extension} sample"
    )

    # Define the legend
    ax_left.legend(
        handles=lines_list,
        labels=labels_list,
        loc="upper right",
        fontsize=leg_fontsize,
        frameon=False,
    )
    if "y_scale" not in kwargs:
        kwargs["y_scale"] = 1.3
    ax_right.set_ylim(top=ax_right.get_ylim()[1] * kwargs["y_scale"])

    rej_plot.set_log()
    rej_plot.set_y_lim()
    rej_plot.axis_top.set_xlabel("Epoch")
    rej_plot.set_tick_params()
    rej_plot.fig.tight_layout()
    rej_plot.plotting_done = True
    rej_plot.atlasify()
    rej_plot.savefig(f"{plot_name}.{plot_datatype}", transparent=True)


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
    plot_datatype: str = "pdf",
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
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    **kwargs
         Keyword arguments handed to the plotting API
    """
    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a deep copy of the class labels
    class_labels_wo_main = copy.deepcopy(list(dict.fromkeys(class_labels)))

    # Remove the main classes from the copy
    for m_class in main_class:
        class_labels_wo_main.remove(m_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories

    for _, iter_class in enumerate(class_labels_wo_main):
        rej_plot = PlotBase(
            xlabel="Epoch",
            ylabel=f'{flav_cat[iter_class]["legend_label"]} rejection',
            n_ratio_panels=0,
            logy=False,
            **kwargs,
        )
        rej_plot.initialise_figure()
        # Init a linestyle counter
        counter_models = 0

        # Plot rejection
        rej_plot.axis_top.plot(
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
                    rej_plot.axis_top.axhline(
                        y=comp_tagger_rej_dict[comp_tagger][
                            f"{iter_class}_rej_{unique_identifier}"
                        ],
                        xmin=0,
                        xmax=df_results["epoch"].max(),
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

                rej_plot.axis_top.plot(
                    tt_rej_dict["epoch"],
                    tt_rej_dict[f"{iter_class}_{rej_string}"],
                    linestyle="-",
                    color=f"C{counter_models}",
                    label=trained_taggers[tt]["label"],
                )

                # Set up the counter
                counter_models += 1

        rej_plot.atlas_second_tag += (
            f"\nWP={int(target_beff * 100):02d}% {label_extension} sample"
        )

        rej_plot.initialise_plot()
        rej_plot.savefig(
            f"{plot_name}_{iter_class}_rejection.{plot_datatype}",
            transparent=True,
        )


def PlotLosses(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training loss and the validation losses per epoch.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and losses.
    plot_name : str
        Path where the plots is saved.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    **kwargs
        Keyword arguments handed to the plotting API
    """
    loss_plot = PlotBase(
        xlabel="Epoch",
        ylabel="Loss",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    loss_plot.initialise_figure()

    # plots training loss
    if "loss" in df_results:
        loss_plot.axis_top.plot(
            df_results["epoch"],
            df_results["loss"],
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
        ax=loss_plot.axis_top,
    )

    loss_plot.initialise_plot()
    loss_plot.savefig(plot_name + f".{plot_datatype}", transparent=True)


def PlotAccuracies(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,  # pylint: disable=unused-argument
):
    """Plot the training and validation accuracies per epoch.

    Parameters
    ----------
    df_results : dict
        Dict with the epochs and accuracies.
    plot_name : str
        Path where the plots is saved.
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    **kwargs

    """
    acc_plot = PlotBase(
        xlabel="Epoch",
        ylabel="Accuracy",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    acc_plot.initialise_figure()

    if "accuracy" in df_results:
        acc_plot.axis_top.plot(
            df_results["epoch"],
            df_results["accuracy"],
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
        ax=acc_plot.axis_top,
    )
    acc_plot.initialise_plot()
    acc_plot.savefig(plot_name + f".{plot_datatype}", transparent=True)


def PlotLossesUmami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
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
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    **kwargs
        Keyword arguments handed to the plotting API
    """
    loss_plot = PlotBase(
        xlabel="Epoch",
        ylabel="Loss",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    loss_plot.initialise_figure()

    # Plot umami and dips training loss
    if "loss_umami" in df_results:
        loss_plot.axis_top.plot(
            df_results["epoch"],
            df_results["loss_umami"],
            label="training loss Umami - hybrid sample",
        )

    if "loss_dips" in df_results:
        loss_plot.axis_top.plot(
            df_results["epoch"],
            df_results["loss_dips"],
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
        ax=loss_plot.axis_top,
    )
    plot_validation_files(
        metric_identifier="val_loss_dips",
        df_results=df_results,
        label_prefix="validation loss DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
        ax=loss_plot.axis_top,
    )
    loss_plot.initialise_plot()
    loss_plot.savefig(plot_name + f".{plot_datatype}", transparent=True)


def PlotAccuraciesUmami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
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
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    **kwargs
        Keyword arguments handed to the plotting API
    """
    acc_plot = PlotBase(
        xlabel="Epoch",
        ylabel="Accuracy",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    acc_plot.initialise_figure()

    # Plot umami and dips training loss
    if "accuracy_umami" in df_results:
        acc_plot.axis_top.plot(
            df_results["epoch"],
            df_results["accuracy_umami"],
            label="training accuracy Umami - hybrid sample",
        )

    if "accuracy_dips" in df_results:
        acc_plot.axis_top.plot(
            df_results["epoch"],
            df_results["accuracy_dips"],
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
        ax=acc_plot.axis_top,
    )
    plot_validation_files(
        metric_identifier="val_acc_dips",
        df_results=df_results,
        label_prefix="validation accuracy DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
        ax=acc_plot.axis_top,
    )

    acc_plot.initialise_plot()
    acc_plot.savefig(plot_name + f".{plot_datatype}", transparent=True)


def RunPerformanceCheck(
    train_config: object,
    tagger: str,
    tagger_comp_vars: dict = None,
    train_metrics_file_name: str = None,
    val_metrics_file_name: str = None,
    working_point: float = None,
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
    train_metrics_file_name : str, optional
        Path to the json file with the per epoch train metrics, by default None
    val_metrics_file_name : str, optional
        Path to the json file with the per epoch validation metrics, by default None
    working_point : float, optional
        Working point to evaluate, by default None
    """

    logger.info(f"Running performance check for {tagger}.")

    # Load parameters from train config
    Eval_parameters = train_config.Eval_parameters_validation
    Val_settings = train_config.Validation_metrics_settings
    plot_args = train_config.plot_args
    frac_dict = Eval_parameters["frac_values"]
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    recommended_frac_dict = Eval_parameters["frac_values_comp"]
    # Load the unique identifiers of the validation files and the corresponding plot
    # labels. These are used several times in this function
    val_files = train_config.validation_files

    # Printing the given plot args for debugging
    logger.debug(f"plot_args = {plot_args}")

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a working point
    if working_point is None:
        working_point = Val_settings["WP"]

    # Get dict with training results from json
    try:
        train_metrics_dict = pd.read_json(train_metrics_file_name)

    except ValueError:
        logger.warning(f"Train metrics json {train_metrics_file_name} not found!")
        train_metrics_dict = None

    # Get dict with validation results from json
    try:
        val_metrics_dict = pd.read_json(val_metrics_file_name)

    except ValueError:
        logger.warning(
            f"Validation results json {val_metrics_file_name} could not be found! "
            "Check your train config values (the name of the file loaded depends "
            "on them). If you want to use a specific json file, use the -d option "
            "of the plotting_epoch_performance script!"
        )
        val_metrics_dict = None

    # Merge the train and val metrics dicts
    if train_metrics_dict is not None and val_metrics_dict is not None:

        # Join the two dataframes in one
        tagger_rej_dict = train_metrics_dict.join(
            val_metrics_dict.set_index("epoch"), on="epoch"
        )

    elif train_metrics_dict is not None:
        tagger_rej_dict = train_metrics_dict

    elif val_metrics_dict is not None:
        tagger_rej_dict = val_metrics_dict

    else:
        raise ValueError(
            "No train or validation metrics file could be loaded! Check files"
        )

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
                        working_point=working_point,
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

    # Get a deep copy of the class labels
    class_labels_wo_main = copy.deepcopy(list(dict.fromkeys(class_labels)))

    # Remove the main classes from the copy
    for m_class in main_class:
        class_labels_wo_main.remove(m_class)

    n_rej = len(class_labels_wo_main)

    if tagger.casefold() == "umami" or tagger.casefold() == "umami_cond_att":
        for subtagger in ["umami", "dips"]:
            if n_rej == 2:
                for val_file_identifier, val_file_config in val_files.items():
                    # Plot comparsion for the comparison taggers
                    PlotRejPerEpochComparison(
                        df_results=tagger_rej_dict,
                        tagger_label=subtagger,
                        comp_tagger_rej_dict=comp_tagger_rej_dict,
                        unique_identifier=val_file_identifier,
                        plot_name=(
                            f"{plot_dir}/rej-plot_val_{subtagger}_{val_file_identifier}"
                        ),
                        class_labels=class_labels,
                        main_class=main_class,
                        label_extension=val_file_config["label"],
                        rej_string=f"rej_{subtagger}_{val_file_identifier}",
                        target_beff=working_point,
                        **plot_args,
                    )

                    PlotRejPerEpoch(
                        df_results=tagger_rej_dict,
                        tagger_label=subtagger,
                        comp_tagger_rej_dict=comp_tagger_rej_dict,
                        unique_identifier=val_file_identifier,
                        plot_name=(
                            f"{plot_dir}/rej-plot_val_{subtagger}_{val_file_identifier}"
                        ),
                        class_labels=class_labels,
                        main_class=main_class,
                        label_extension=val_file_config["label"],
                        rej_string=f"rej_{subtagger}_{val_file_identifier}",
                        target_beff=working_point,
                        **plot_args,
                    )

        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpochUmami(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            val_files=val_files,
            target_beff=working_point,
            **plot_args,
        )

        # Check if metrics are present
        plot_name = f"{plot_dir}/loss-plot"
        PlotLossesUmami(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        PlotAccuraciesUmami(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
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
                    tagger_label=Val_settings["tagger_label"],
                    comp_tagger_rej_dict=comp_tagger_rej_dict,
                    unique_identifier=val_file_identifier,
                    plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                    class_labels=class_labels,
                    main_class=main_class,
                    label_extension=val_file_config["label"],
                    rej_string=f"rej_{val_file_identifier}",
                    target_beff=working_point,
                    **plot_args,
                )
        for val_file_identifier, val_file_config in val_files.items():
            # Plot rejections in one plot per rejection
            PlotRejPerEpoch(
                df_results=tagger_rej_dict,
                comp_tagger_rej_dict=comp_tagger_rej_dict,
                unique_identifier=val_file_identifier,
                plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                class_labels=class_labels,
                main_class=main_class,
                label_extension=val_file_config["label"],
                rej_string=f"rej_{val_file_identifier}",
                target_beff=working_point,
                tagger_label=Val_settings["tagger_label"],
                **plot_args,
            )

        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpoch(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            val_files=val_files,
            target_beff=working_point,
            frac_class=list(frac_dict.keys())[0],
            frac=frac_dict[list(frac_dict.keys())[0]],
            **plot_args,
        )

        # Check if metrics are present
        plot_name = f"{plot_dir}/loss-plot"
        PlotLosses(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        PlotAccuracies(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
