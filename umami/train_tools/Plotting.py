import copy
import os

import h5py
import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from umami.configuration import global_config, logger
from umami.preprocessing_tools import GetBinaryLabels
from umami.tools import applyATLASstyle, makeATLAStag
from umami.train_tools import (
    GetRejection,
    get_class_label_ids,
    get_class_label_variables,
)


def CompTaggerRejectionDict(
    file,
    tagger_comp_name: str,
    tagger_comp_var: list,
    recommended_frac_dict: dict,
    WP: float,
    class_labels: list,
    main_class: str,
):
    """
    Load the comparison tagger probability variables from the validation
    file and calculate the rejections.

    Input:
    - file: Filepath to validation file.
    - tagger_comp_name: Name of the comparison tagger.
    - tagger_comp_var: List of the comparison tagger probability variable names.
    - recommended_frac_dict: Dict with the fractions.
    - class_labels: List with the used class_labels.
    - main_class: The main discriminant class. For b-tagging obviously "bjets"

    Output:
    - Dict with the rejections for against the main class for all given flavours.
    """

    # Get class_labels variables
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(
        class_labels
    )

    # Get the tagger variables and the class label variables
    variables = list(dict.fromkeys(class_label_vars)) + tagger_comp_var

    # Load the Jets
    df = pd.DataFrame(h5py.File(file, "r")["/jets"][:][variables])

    # Init new column for string labels
    df["Umami_string_labels"] = np.zeros_like(df[class_label_vars[0]])
    df["Umami_labels"] = np.zeros_like(df[class_label_vars[0]])

    # Change type of column to string
    df = df.astype({"Umami_string_labels": "str"})

    # Iterate over the classes and add the correct labels to Umami columns
    for class_id, class_label_var, class_label in zip(
        class_ids, class_label_vars, flatten_class_labels
    ):
        indices_tochange = np.where(df[class_label_var].values == class_id)

        # Add a string description which this class is
        df["Umami_string_labels"].values[indices_tochange] = class_label

        # Add the right column label to class
        df["Umami_labels"].values[indices_tochange] = class_labels.index(
            class_label
        )

    # Get the indices of the jets that are not used
    indices_toremove = np.where(df["Umami_string_labels"] == "0")[0]

    # Remove all unused jets
    df = df.drop(indices_toremove)

    # Binarize the labels
    y_true = GetBinaryLabels(df["Umami_labels"].values)

    # Calculate rejections
    recomm_rej_dict, _ = GetRejection(
        y_pred=df[tagger_comp_var].values,
        y_true=y_true,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=recommended_frac_dict,
        target_eff=WP,
    )

    return recomm_rej_dict


def PlotDiscCutPerEpoch(
    df_results,
    plot_name,
    frac_class: str,
    target_beff=0.77,
    frac: float = 0.018,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
    plot_datatype="pdf",
):
    """
    Plot the discriminant cut value for a specific working point
    over all epochs.

    Input:
    - df_results: Dict with the epochs and disc cuts.
    - plot_name: Path where the plots is saved + plot name.
    - frac_class: Define which fraction is shown in ATLAS Tag.
    - target_beff: Working Point to use.
    - frac: Fraction value for ATLAS Tag.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - ncol: Number of columns in the legend.
    - plot_datatype: Datatype of the plot.

    Output:
    - Discriminant Cut per epoch plotted.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        df_results["disc_cut"],
        label=r"$t\bar{t}$ validation sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["disc_cut_add"],
        label=r"$Z'$ validation sample",
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
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel(r"$b$-Tagging discriminant Cut Value")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotDiscCutPerEpochUmami(
    df_results,
    plot_name,
    frac_class: str,
    target_beff=0.77,
    frac: float = 0.018,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
    plot_datatype="pdf",
):
    """
    Plot the discriminant cut value for a specific working point
    over all epochs.

    Input:
    - df_results: Dict with the epochs and disc cuts.
    - plot_name: Path where the plots is saved + plot name.
    - frac_class: Define which fraction is shown in ATLAS Tag.
    - target_beff: Working Point to use.
    - frac: Fraction value for ATLAS Tag.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - ncol: Number of columns in the legend.
    - plot_datatype: Datatype of the plot.

    Output:
    - Discriminant Cut per epoch plotted. DIPS and UMAMI both shown.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        df_results["disc_cut_dips"],
        label=r"$DIPS - t\bar{t}$ validation sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["disc_cut_dips_add"],
        label=r"DIPS - $Z'$ validation sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["disc_cut_umami"],
        label=r"$Umami - t\bar{t}$ validation sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["disc_cut_umami_add"],
        label=r"Umami - $Z'$ validation sample",
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
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel(r"$b$-Tagging discriminant Cut Value")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotRejPerEpoch(
    df_results: dict,
    frac_dict: dict,
    comp_tagger_rej_dict: dict,
    comp_tagger_frac_dict: dict,
    plot_name: str,
    class_labels: list,
    main_class: str,
    label_extension: str,
    rej_string: str,
    target_beff=0.77,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
    legend_loc=(0.6, 0.65),
    plot_datatype="pdf",
):
    """
    Plotting the Rejections per Epoch for the trained tagger and
    the provided comparison taggers.

    Input:
    - df_results: Dict with the rejections of the trained tagger.
    - frac_dict: Dict with the fractions of the trained tagger.
    - comp_tagger_rej_dict: Dict with the rejections of the comp taggers.
    - comp_tagger_frac_dict: Dict with the fractions of the comp taggers.
    - plot_name: Path where the plots is saved.
    - class_labels: A list of the class_labels which are used
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - label_extension: Extension of the legend label giving the process type.
    - rej_string: String that is added after the class for the key.
    - target_beff: Target Working point.
    - UseAtlasTag: Bool to decide if you want the ATLAS tag.
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - ncol: Number of columns in the legend.
    - plot_datatype: Datatype of the plot.

    Output:
    - Plot of the rejections of the taggers per epoch.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    # Get a list of the background classes
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories
    linestyle_list = ["-", (0, (5, 10)), (0, (5, 5)), (0, (5, 1))]

    if len(class_labels_wo_main) == 2:
        # Set global plot configs
        fig, ax1 = plt.subplots(constrained_layout=True)
        ax1.set_xlabel("Epoch")
        ax2 = ax1.twinx()
        axes = [ax1, ax2]

        for counter, iter_class in enumerate(class_labels_wo_main):
            # Plot rejection
            axes[counter].plot(
                df_results["epoch"],
                df_results[f"{iter_class}_{rej_string}"],
                ":",
                color=flav_cat[iter_class]["colour"],
                label=f"{flav_cat[iter_class]['legend_label']} - {label_extension}",
            )
            axes[counter].set_ylabel(
                f'{flav_cat[iter_class]["legend_label"]} Rejection',
                color=flav_cat[iter_class]["colour"],
            )

            if comp_tagger_rej_dict is None:
                logger.info(
                    "No comparison tagger defined. Plotting only trained tagger rejections!"
                )

            else:
                for comp_counter, comp_tagger in enumerate(
                    comp_tagger_rej_dict
                ):
                    try:
                        axes[counter].axhline(
                            comp_tagger_rej_dict[comp_tagger][
                                f"{iter_class}_rej"
                            ],
                            0,
                            df_results["epoch"].max(),
                            color=flav_cat[iter_class]["colour"],
                            lw=1.0,
                            alpha=1,
                            linestyle=linestyle_list[comp_counter]
                            if comp_counter < len(linestyle_list)
                            else "-",
                            label=f"Recomm. {comp_tagger}",
                        )

                    except KeyError:
                        logger.info(
                            f"{iter_class} rejection for {comp_tagger} not in dict! Skipping ..."
                        )

            # Set color of axis
            axes[counter].tick_params(
                axis="y", labelcolor=flav_cat[iter_class]["colour"]
            )

        # Set the x axis locator
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Increase y limit for ATLAS Logo
        ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)
        ax2.set_ylim(top=ax2.get_ylim()[1] * yAxisIncrease)

    else:
        fig, ax1 = plt.subplots(constrained_layout=True)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Rejection")

        for counter, iter_class in enumerate(class_labels_wo_main):
            # Plot rejection
            ax1.plot(
                df_results["epoch"],
                df_results[f"{iter_class}_{rej_string}"],
                fmt=":",
                color=flav_cat[iter_class]["colour"],
                label=flav_cat[iter_class]["legend_label"],
            )

            if comp_tagger_rej_dict is None:
                logger.info(
                    "No comparison tagger defined. Plotting only trained tagger rejections!"
                )

            else:
                for comp_tagger in comp_tagger_rej_dict:
                    try:
                        ax1.axhline(
                            comp_tagger_rej_dict[comp_tagger][
                                f"{iter_class}_rej"
                            ],
                            0,
                            df_results["epoch"].max(),
                            color=flav_cat[iter_class]["colour"],
                            lw=1.0,
                            alpha=1,
                            linestyle=(0, (5, 10)),
                            label=f"Recomm. {comp_tagger} - {flav_cat[iter_class]['legend_label']}",
                        )

                    except KeyError:
                        logger.info(
                            f"{iter_class} rejection for {comp_tagger} not in dict! Skipping ..."
                        )

        # Increase y limit for ATLAS Logo
        ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)

    if UseAtlasTag is True:
        SecondTag = SecondTag + f"\nWP={int(target_beff * 100):02d}%"

        # Increase y limit for ATLAS Logo
        ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    fig.legend(ncol=ncol, loc=legend_loc)
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotLosses(
    df_results,
    plot_name,
    train_history_dict,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease=1.2,
    yAxisAtlasTag=0.9,
    plot_datatype="pdf",
    ymin=None,
    ymax=None,
):
    """
    Plot the training loss and the validation losses per epoch.

    Input:
    - df_results: Dict with the epochs and losses.
    - plot_name: Path where the plots is saved + plot name.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - plot_datatype: Datatype of the plot.
    - ymin: Manually set ymin. Overwrites yAxisIncrease.
    - ymax: Manually set ymax. Overwrites yAxisIncrease.

    Output:
    - Plot with the training and validation losses per epoch.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        train_history_dict["loss"],
        label="training loss - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_loss"],
        label=r"validation loss - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_loss_add"],
        label=r"validation loss - ext. $Z'$ sample",
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

    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuracies(
    df_results,
    plot_name,
    train_history_dict,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease=1.2,
    yAxisAtlasTag=0.9,
    plot_datatype="pdf",
    ymin=None,
    ymax=None,
):
    """
    Plot the training and validation accuracies per epoch.

    Input:
    - df_results: Dict with the epochs and accuracies.
    - plot_name: Path where the plots is saved + plot name.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - plot_datatype: Datatype of the plot.
    - ymin: Manually set ymin. Overwrites yAxisIncrease.
    - ymax: Manually set ymax. Overwrites yAxisIncrease.

    Output:
    - Plot with the training and validation accuracies per epoch.
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        train_history_dict["accuracy"],
        label="training accuracy - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_acc"],
        label=r"validation accuracy - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_acc_add"],
        label=r"validation accuracy - ext. $Z'$ sample",
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

    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotLossesUmami(
    df_results,
    plot_name,
    train_history_dict,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease=1.4,
    yAxisAtlasTag=0.9,
    plot_datatype="pdf",
    ymin=None,
    ymax=None,
):
    """
    Plot the training loss and the validation losses per epoch for
    UMAMI model (with DIPS and UMAMI losses).

    Input:
    - df_results: Dict with the epochs and losses.
    - plot_name: Path where the plots is saved + plot name.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - plot_datatype: Datatype of the plot.
    - ymin: Manually set ymin. Overwrites yAxisIncrease.
    - ymax: Manually set ymax. Overwrites yAxisIncrease.

    Output:
    - Plot with the training and validation losses per epoch for
      UMAMI model (with DIPS and UMAMI losses).
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        train_history_dict["umami_loss"],
        label="training loss UMAMI - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_loss"],
        label=r"val loss UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        train_history_dict["dips_loss"],
        label="training loss DIPS - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_val_loss"],
        label=r"val loss DIPS - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_loss_add"],
        label=r"val loss UMAMI - ext. $Z'$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_val_loss_add"],
        label=r"val loss DIPS - ext. $Z'$ sample",
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

    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuraciesUmami(
    df_results,
    plot_name,
    train_history_dict,
    UseAtlasTag=True,
    ApplyATLASStyle=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisIncrease=1.4,
    yAxisAtlasTag=0.9,
    plot_datatype="pdf",
    ymin=None,
    ymax=None,
):
    """
    Plot the training and validation accuracies per epoch for
    UMAMI model (with DIPS and UMAMI accuracies).

    Input:
    - df_results: Dict with the epochs and accuracies.
    - plot_name: Path where the plots is saved + plot name.
    - UseAtlasTag: Define if ATLAS Tag is used or not.
    - ApplyATLASStyle: Apply ATLAS Style of the plot (for approval etc.).
    - AtlasTag: Main tag. Mainly "Internal Simulation".
    - SecondTag: Lower tag in the ATLAS label with infos.
    - yAxisAtlasTag: Y axis position of the ATLAS label.
    - yAxisIncrease: Y axis increase factor to fit the ATLAS label.
    - plot_datatype: Datatype of the plot.
    - ymin: Manually set ymin. Overwrites yAxisIncrease.
    - ymax: Manually set ymax. Overwrites yAxisIncrease.

    Output:
    - Plot with the training and validation accuracies per epoch for
      UMAMI model (with DIPS and UMAMI accuracies).
    """

    # Apply ATLAS style
    if ApplyATLASStyle is True:
        applyATLASstyle(mtp)

    plt.plot(
        df_results["epoch"],
        train_history_dict["umami_accuracy"],
        label="training acc UMAMI - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_acc"],
        label=r"val acc UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        train_history_dict["dips_accuracy"],
        label="training acc DIPS - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_val_acc"],
        label=r"val acc DIPS - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_acc_add"],
        label=r"val acc UMAMI - ext. $Z'$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_val_acc_add"],
        label=r"val acc DIPS - ext. $Z'$ sample",
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

    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plt.savefig(plot_name + f".{plot_datatype}", transparent=True)
    plt.cla()
    plt.clf()


def RunPerformanceCheck(
    train_config,
    tagger: str,
    tagger_comp_vars: dict = None,
    dict_file_name: str = None,
    train_history_dict: str = None,
    WP: float = None,
):
    """
    Loading the validation metrics from the trained model and calculate
    the metrics for the comparison taggers and plot them.

    Input:
    - train_config: Loaded train_config object.
    - tagger: String name of the tagger used.
    - tagger_comp_vars: Dict of the tagger probability variables as lists.
    - dict_file_name: Path to the json file with the per epoch metrics of the
                      trained tagger.

    Output:
    - Validation metrics plots
    """

    logger.info("Running performance check for DIPS.")

    # Load parameters from train config
    Eval_parameters = train_config.Eval_parameters_validation
    Plotting_settings = train_config.Plotting_settings
    frac_dict = Eval_parameters["frac_values"]
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    recommended_frac_dict = Eval_parameters["frac_values_comp"]

    if WP is None:
        WP = Eval_parameters["WP"]

    # Get dict with training results from json
    tagger_rej_dict = pd.read_json(dict_file_name)
    train_history_dict = pd.read_json(train_history_dict)

    if tagger_comp_vars is not None:
        # Dict
        comp_tagger_rej_dict = {}
        comp_tagger_rej_dict_add = {}

        # Loop over taggers that are used for comparsion
        for tagger in tagger_comp_vars:
            comp_tagger_rej_dict[tagger] = CompTaggerRejectionDict(
                file=train_config.validation_file,
                tagger_comp_name=tagger,
                tagger_comp_var=tagger_comp_vars[tagger],
                recommended_frac_dict=recommended_frac_dict[tagger],
                WP=WP,
                class_labels=class_labels,
                main_class=main_class,
            )

            if train_config.add_validation_file is not None:
                comp_tagger_rej_dict_add[tagger] = CompTaggerRejectionDict(
                    file=train_config.add_validation_file,
                    tagger_comp_name=tagger,
                    tagger_comp_var=tagger_comp_vars[tagger],
                    recommended_frac_dict=recommended_frac_dict[tagger],
                    WP=WP,
                    class_labels=class_labels,
                    main_class=main_class,
                )

            else:
                comp_tagger_rej_dict_add[tagger] = None

    else:
        # Define the dicts as None if compare tagger is False
        comp_tagger_rej_dict = None
        comp_tagger_rej_dict_add = None

    # Define dir where the plots are saved
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)

    if tagger == "umami":
        for subtagger in ["umami", "dips"]:
            PlotRejPerEpoch(
                df_results=tagger_rej_dict,
                frac_dict=frac_dict,
                comp_tagger_rej_dict=comp_tagger_rej_dict,
                comp_tagger_frac_dict=recommended_frac_dict,
                plot_name=f"{plot_dir}/rej-plot_val_{subtagger}",
                class_labels=class_labels,
                main_class=main_class,
                label_extension=r"$t\bar{t}$",
                rej_string=f"rej_{subtagger}",
                target_beff=WP,
                **Plotting_settings,
            )

            if tagger_comp_vars is not None:
                PlotRejPerEpoch(
                    df_results=tagger_rej_dict,
                    frac_dict=frac_dict,
                    comp_tagger_rej_dict=comp_tagger_rej_dict_add,
                    comp_tagger_frac_dict=recommended_frac_dict,
                    plot_name=f"{plot_dir}/rej-plot_val_{subtagger}_add",
                    class_labels=class_labels,
                    main_class=main_class,
                    label_extension=r"ext. $Z'$",
                    rej_string="rej_{subtagger}_add",
                    target_beff=WP,
                    **Plotting_settings,
                )

        plot_name = f"{plot_dir}/loss-plot"
        PlotLossesUmami(
            tagger_rej_dict,
            plot_name,
            train_history_dict=train_history_dict,
            **Plotting_settings,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        PlotAccuraciesUmami(
            tagger_rej_dict,
            plot_name,
            train_history_dict=train_history_dict,
            **Plotting_settings,
        )
        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpochUmami(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            frac_class="cjets",
            target_beff=WP,
            frac=frac_dict["cjets"],
            **Plotting_settings,
        )

    else:
        # Plot comparsion for the comparison taggers
        PlotRejPerEpoch(
            df_results=tagger_rej_dict,
            frac_dict=frac_dict,
            comp_tagger_rej_dict=comp_tagger_rej_dict,
            comp_tagger_frac_dict=recommended_frac_dict,
            plot_name=f"{plot_dir}/rej-plot_val",
            class_labels=class_labels,
            main_class=main_class,
            label_extension=r"$t\bar{t}$",
            rej_string="rej",
            target_beff=WP,
            **Plotting_settings,
        )

        if tagger_comp_vars is not None:
            PlotRejPerEpoch(
                df_results=tagger_rej_dict,
                frac_dict=frac_dict,
                comp_tagger_rej_dict=comp_tagger_rej_dict_add,
                comp_tagger_frac_dict=recommended_frac_dict,
                plot_name=f"{plot_dir}/rej-plot_val_add",
                class_labels=class_labels,
                main_class=main_class,
                label_extension=r"ext. $Z'$",
                rej_string="rej_add",
                target_beff=WP,
                **Plotting_settings,
            )

        plot_name = f"{plot_dir}/loss-plot"
        PlotLosses(
            tagger_rej_dict,
            plot_name,
            train_history_dict=train_history_dict,
            **Plotting_settings,
        )
        plot_name = f"{plot_dir}/accuracy-plot"
        PlotAccuracies(
            tagger_rej_dict,
            plot_name,
            train_history_dict=train_history_dict,
            **Plotting_settings,
        )
        plot_name = f"{plot_dir}/disc-cut-plot"
        PlotDiscCutPerEpoch(
            df_results=tagger_rej_dict,
            plot_name=plot_name,
            target_beff=WP,
            frac_class="cjets",
            frac=frac_dict["cjets"],
            **Plotting_settings,
        )
