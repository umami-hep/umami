"""Plotting functions for NN training."""
# pylint: disable=invalid-name
# TODO: switch to new plotting API with pep8 conform naming
import copy
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from puma import Line2D, Line2DPlot

from umami.configuration import global_config, logger
from umami.data_tools import LoadJetsFromFile
from umami.metrics import get_rejection
from umami.preprocessing_tools.utils import binarise_jet_labels
from umami.tools import check_main_class_input


def plot_validation_files(
    metric_identifier: str,
    df_results: dict,
    plot_object: Line2DPlot,
    label_prefix: str = "",
    label_suffix: str = "",
    val_files: dict = None,
    **kwargs,
):
    """Helper function which loops over the validation files and adds the line object
    for the chosen metric for each epoch. Meant to be called in other plotting
    functions.

    The fuction loops over the validation files and plots a line with the label
    f"{label_prefix}{validation_file_label}{label_suffix}" for each file.

    Parameters
    ----------
    metric_identifier : str
        Identifier for the metric you want to plot, e.g. "val_loss" or "disc_cut".
    df_results : dict
        Dict which contains the results of the training.
    plot_object : puma.Line2DPlot
        Plot object you want to plot the curves on.
    label_prefix : str, optional
        This string is put at the beginning of the plot label for each line.
        For accuracy for example choose "validation accuracy - ", by default ""
    label_suffix : str, optional
        This string is put at the end of the plot label for each line, by default ""
    val_files: dict, optional
        Dict that contains the configuration of all the validation files listed in the
        train config. If None, nothing happens and a warning is printed to the logs,
        by default None
    **kwargs
        Keyword arguments handed to the plotting API
    """
    # Check for xmin and xmax
    if "xmin" not in kwargs:
        kwargs["xmin"] = 0

    if "xmax" not in kwargs:
        kwargs["xmax"] = df_results["epoch"].max()

    if val_files is None:
        logger.warning(
            "No validation files provided --> not plotting %s.", metric_identifier
        )

    else:
        for val_file_identifier, val_file_config in val_files.items():
            plot_object.add(
                Line2D(
                    df_results["epoch"],
                    df_results[f"{metric_identifier}_{val_file_identifier}"],
                    label=f"{label_prefix}{val_file_config['label']}{label_suffix}",
                )
            )


def get_comp_tagger_rej_dict(
    file: str,
    unique_identifier: str,
    tagger_comp_var: list,
    recommended_frac_dict: dict,
    n_jets: int,
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
    n_jets : int
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
        n_jets=n_jets,
        variables=tagger_comp_var,
        cut_vars_dict=cut_vars_dict,
        print_logger=False,
        chunk_size=1e6,
    )

    # Binarize the labels
    y_true = binarise_jet_labels(
        labels=labels,
        internal_labels=list(range(len(class_labels))),
    )

    # Get a list with all per-jets variables loaded
    avai_variables = list(df.keys())

    # Init a bool if the rejection should be skipped
    Skip_rej_calc = False

    # Check if the tagger variables are available
    for tagger_var in tagger_comp_var:
        if tagger_var not in avai_variables:
            logger.warning(
                "Tagger probability %s not in validation file %s. Skipping ...",
                tagger_var,
                os.path.basename(file),
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


def plot_disc_cut_per_epoch(
    df_results: dict,
    plot_name: str,
    frac_class: str,
    val_files: dict = None,
    target_beff: float = 0.77,
    frac: float = 0.018,
    plot_datatype: str = "pdf",
    **kwargs,
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
    disc_cut_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="$b$-tagging discriminant cut value",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut",
        df_results=df_results,
        label_prefix="",
        label_suffix=" validation sample",
        val_files=val_files,
        plot_object=disc_cut_plot,
        **kwargs,
    )

    disc_cut_plot.atlas_second_tag += (
        f"\n{frac_class} fraction = {frac}\nWP={int(target_beff * 100):02d}%"
    )
    disc_cut_plot.draw()
    disc_cut_plot.savefig(f"{plot_name}.{plot_datatype}")


def plot_disc_cut_per_epoch_umami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    target_beff: float = 0.77,
    plot_datatype: str = "pdf",
    **kwargs,
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
    disc_cut_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="$b$-tagging discriminant cut value",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )
    # loop over the validation files using the unique identifiers and plot the disc cut
    # value for each file
    plot_validation_files(
        metric_identifier="disc_cut_umami",
        df_results=df_results,
        label_prefix="Umami - ",
        label_suffix=" validation sample",
        val_files=val_files,
        plot_object=disc_cut_plot,
        **kwargs,
    )
    plot_validation_files(
        metric_identifier="disc_cut_dips",
        df_results=df_results,
        label_prefix="DIPS - ",
        label_suffix=" validation sample",
        val_files=val_files,
        plot_object=disc_cut_plot,
    )
    disc_cut_plot.atlas_second_tag += f"\nWP={int(target_beff * 100):02d}%"
    disc_cut_plot.draw()
    disc_cut_plot.savefig(f"{plot_name}.{plot_datatype}")


def plot_rej_per_epoch_comp(
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
    trained_taggers: dict = None,
    target_beff: float = 0.77,
    plot_datatype: str = "pdf",
    leg_fontsize: int = 10,
    **kwargs,
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
    # Check for xmin and xmax
    if "xmin" not in kwargs:
        kwargs["xmin"] = 0

    if "xmax" not in kwargs:
        kwargs["xmax"] = df_results["epoch"].max()

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

    rej_plot = Line2DPlot(
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    ax_left = rej_plot.axis_top
    ax_left.set_xlim(left=kwargs["xmin"], right=kwargs["xmax"])
    ax_right = ax_left.twinx()
    axes = [ax_left, ax_right]

    # Define a list for the lines which are plotted
    lines = []

    for counter, iter_class in enumerate(class_labels_wo_main):
        # Init a linestyle counter
        counter_models = 0

        if comp_tagger_rej_dict is None:
            logger.info("No comparison tagger defined. Not plotting those!")

        else:
            for comp_tagger in comp_tagger_rej_dict:
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
                        "%s rejection for %s and file %s not in dict! Skipping ...",
                        iter_class,
                        comp_tagger,
                        unique_identifier,
                    )

        if trained_taggers is None:
            if counter == 0:
                logger.debug("No local taggers defined. Not plotting those!")

        else:
            for tt in trained_taggers:
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
    rej_plot.savefig(f"{plot_name}.{plot_datatype}")


def plot_rej_per_epoch(
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
    trained_taggers: dict = None,
    target_beff: float = 0.77,
    plot_datatype: str = "pdf",
    **kwargs,
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
    # Check for xmin and xmax
    if "xmin" not in kwargs:
        kwargs["xmin"] = 0

    if "xmax" not in kwargs:
        kwargs["xmax"] = df_results["epoch"].max()

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
        rej_plot = Line2DPlot(
            xlabel="Epoch",
            ylabel=f'{flav_cat[iter_class]["legend_label"]} rejection',
            n_ratio_panels=0,
            logy=False,
            **kwargs,
        )
        rej_plot.axis_top.set_xlim(left=kwargs["xmin"], right=kwargs["xmax"])

        # Init a linestyle counter
        counter_models = 0

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
                        "%s rejection for %s not in dict! Skipping ...",
                        iter_class,
                        comp_tagger,
                    )

        if trained_taggers is None:
            logger.debug("No local taggers defined. Not plotting those!")

        else:
            for tt in trained_taggers:
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

        rej_plot.atlas_second_tag += (
            f"\nWP={int(target_beff * 100):02d}% {label_extension} sample"
        )

        rej_plot.initialise_plot()
        rej_plot.savefig(f"{plot_name}_{iter_class}_rejection.{plot_datatype}")


def plot_losses(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,
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
    loss_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="Loss",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    # plots training loss
    if "loss" in df_results:
        loss_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["loss"],
                label="training loss - hybrid sample",
            )
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_loss",
        df_results=df_results,
        label_prefix="validation loss - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=loss_plot,
        **kwargs,
    )

    loss_plot.draw()
    loss_plot.savefig(plot_name + f".{plot_datatype}")


def plot_accuracies(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,
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
    **kwargs : kwargs
        kwargs for `PlotBase` function

    """
    acc_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="Accuracy",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    if "accuracy" in df_results:
        acc_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["accuracy"],
                label="training accuracy - hybrid sample",
            )
        )

    # loop over the validation files using the unique identifiers and plot the accuracy
    # for each file
    plot_validation_files(
        metric_identifier="val_acc",
        df_results=df_results,
        label_prefix="validation accuracy - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=acc_plot,
        **kwargs,
    )
    acc_plot.draw()
    acc_plot.savefig(plot_name + f".{plot_datatype}")


def plot_losses_umami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,
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
    loss_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="Loss",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    # Plot umami and dips training loss
    if "loss_umami" in df_results:
        loss_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["loss_umami"],
                label="training loss Umami - hybrid sample",
            )
        )

    if "loss_dips" in df_results:
        loss_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["loss_dips"],
                label="training loss DIPS - hybrid sample",
            )
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_loss_umami",
        df_results=df_results,
        label_prefix="validation loss Umami - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=loss_plot,
        **kwargs,
    )
    plot_validation_files(
        metric_identifier="val_loss_dips",
        df_results=df_results,
        label_prefix="validation loss DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=loss_plot,
        **kwargs,
    )
    loss_plot.draw()
    loss_plot.savefig(plot_name + f".{plot_datatype}")


def plot_accuracies_umami(
    df_results: dict,
    plot_name: str,
    val_files: dict = None,
    plot_datatype: str = "pdf",
    **kwargs,
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
    plot_datatype : str, optional
        Datatype of the plot., by default "pdf"
    **kwargs : kwargs
        kwargs for `PlotBase` function
    """
    acc_plot = Line2DPlot(
        xlabel="Epoch",
        ylabel="Accuracy",
        n_ratio_panels=0,
        logy=False,
        **kwargs,
    )

    # Plot umami and dips training loss
    if "accuracy_umami" in df_results:
        acc_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["accuracy_umami"],
                label="training accuracy Umami - hybrid sample",
            )
        )

    if "accuracy_dips" in df_results:
        acc_plot.add(
            Line2D(
                df_results["epoch"],
                df_results["accuracy_dips"],
                label="training accuracy DIPS - hybrid sample",
            )
        )

    # loop over the validation files using the unique identifiers and plot the loss
    # for each file
    plot_validation_files(
        metric_identifier="val_acc_umami",
        df_results=df_results,
        label_prefix="validation accuracy Umami - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=acc_plot,
        **kwargs,
    )
    plot_validation_files(
        metric_identifier="val_acc_dips",
        df_results=df_results,
        label_prefix="validation accuracy DIPS - ",
        label_suffix=" sample",
        val_files=val_files,
        plot_object=acc_plot,
        **kwargs,
    )

    acc_plot.draw()
    acc_plot.savefig(plot_name + f".{plot_datatype}")


def run_validation_check(
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

    Raises
    ------
    ValueError
        When no training metrics json could be found.
    """

    logger.info("Running performance check for %s.", tagger)

    # Load parameters from train config
    val_settings = train_config.validation_settings
    plot_args = train_config.plot_args
    frac_dict = train_config.evaluation_settings["frac_values"]
    class_labels = train_config.nn_structure["class_labels"]
    main_class = train_config.nn_structure["main_class"]
    recommended_frac_dict = train_config.evaluation_settings.get("frac_values_comp")
    # Load the unique identifiers of the validation files and the corresponding plot
    # labels. These are used several times in this function
    val_files = train_config.validation_files

    # Printing the given plot args for debugging
    logger.debug("plot_args = %s", plot_args)

    # Check the main class input and transform it into a set
    main_class = check_main_class_input(main_class)

    # Get a working point
    if working_point is None:
        working_point = val_settings["working_point"]

    # Get dict with training results from json
    try:
        train_metrics_dict = pd.read_json(train_metrics_file_name)

    except ValueError:
        logger.warning("Train metrics json %s not found!", train_metrics_file_name)
        train_metrics_dict = None

    # Get dict with validation results from json
    try:
        val_metrics_dict = pd.read_json(val_metrics_file_name)

    except ValueError:
        logger.warning(
            "Validation results json %s could not be found! "
            "Check your train config values (the name of the file loaded depends "
            "on them). If you want to use a specific json file, use the -d option "
            "of the plotting_epoch_performance script!",
            val_metrics_file_name,
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
                    get_comp_tagger_rej_dict(
                        file=val_file_config["path"],
                        unique_identifier=val_file_identifier,
                        tagger_comp_var=tagger_comp_vars[comp_tagger],
                        recommended_frac_dict=recommended_frac_dict[comp_tagger]
                        if recommended_frac_dict
                        else None,
                        n_jets=val_settings["n_jets"],
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
    logger.info("saving plots to %s", plot_dir)
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
                    plot_rej_per_epoch_comp(
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
                        taggers_from_file=val_settings["taggers_from_file"],
                        trained_taggers=val_settings["trained_taggers"],
                        **plot_args,
                    )

                    plot_rej_per_epoch(
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
                        taggers_from_file=val_settings["taggers_from_file"],
                        trained_taggers=val_settings["trained_taggers"],
                        **plot_args,
                    )

        plot_name = f"{plot_dir}/disc-cut-plot"
        plot_disc_cut_per_epoch_umami(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            val_files=val_files,
            target_beff=working_point,
            **plot_args,
        )

        # Check if metrics are present
        plot_name = f"{plot_dir}/loss-plot"
        plot_losses_umami(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        plot_accuracies_umami(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )

    else:
        # If no freshly trained tagger label is given, give tagger
        if not (
            "tagger_label" in val_settings and val_settings["tagger_label"] is not None
        ):
            val_settings["tagger_label"] = tagger

        if n_rej == 2:
            # Plot comparsion for the comparison taggers
            # Loop over validation files
            for val_file_identifier, val_file_config in val_files.items():
                plot_rej_per_epoch_comp(
                    df_results=tagger_rej_dict,
                    tagger_label=val_settings["tagger_label"],
                    comp_tagger_rej_dict=comp_tagger_rej_dict,
                    unique_identifier=val_file_identifier,
                    plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                    class_labels=class_labels,
                    main_class=main_class,
                    label_extension=val_file_config["label"],
                    rej_string=f"rej_{val_file_identifier}",
                    target_beff=working_point,
                    taggers_from_file=val_settings["taggers_from_file"],
                    trained_taggers=val_settings["trained_taggers"],
                    **plot_args,
                )

        for val_file_identifier, val_file_config in val_files.items():
            # Plot rejections in one plot per rejection
            plot_rej_per_epoch(
                df_results=tagger_rej_dict,
                comp_tagger_rej_dict=comp_tagger_rej_dict,
                unique_identifier=val_file_identifier,
                plot_name=f"{plot_dir}/rej-plot_val_{val_file_identifier}",
                class_labels=class_labels,
                main_class=main_class,
                label_extension=val_file_config["label"],
                rej_string=f"rej_{val_file_identifier}",
                target_beff=working_point,
                tagger_label=val_settings["tagger_label"],
                taggers_from_file=val_settings["taggers_from_file"],
                trained_taggers=val_settings["trained_taggers"],
                **plot_args,
            )

        plot_name = f"{plot_dir}/disc-cut-plot"
        plot_disc_cut_per_epoch(
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
        plot_losses(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        plot_accuracies(
            tagger_rej_dict,
            plot_name,
            val_files=val_files,
            **plot_args,
        )
