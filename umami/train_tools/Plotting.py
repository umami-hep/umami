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


def PlotDiscCutPerEpoch(
    df_results,
    plot_name,
    frac_class: str,
    target_beff=0.77,
    frac: float = 0.018,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
    plot_datatype="",
):
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
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotDiscCutPerEpochUmami(
    df_results,
    plot_name,
    target_beff=0.77,
    fc_value=0.018,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
    plot_datatype="",
):
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
            + "\nfc={}".format(fc_value)
            + ", WP={:02d}%".format(int(target_beff * 100))
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
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotRejPerEpoch(
    df_results,
    plot_name: str,
    frac_dict: dict,
    class_labels: list,
    main_class: str,
    recomm_rej_dict: dict,
    label_extension: str,
    comp_tagger_name="DL1r",
    target_beff=0.77,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    yAxisIncrease=1.3,
    ncol=1,
):
    applyATLASstyle(mtp)

    # Get a list of the background classes
    class_labels_wo_main = class_labels
    class_labels_wo_main.remove(main_class)

    # Get flavour categories from global config
    flav_cat = global_config.flavour_categories

    if len(class_labels_wo_main) == 2:
        # Set global plot configs
        fig, ax1 = plt.subplots(constrained_layout=True)
        legend_loc = (0.6, 0.75)
        ax1.set_xlabel("Epoch")
        ax2 = ax1.twinx()
        axes = [ax1, ax2]

        for counter, iter_class in enumerate(class_labels_wo_main):
            # Plot rejection
            axes[counter].plot(
                df_results["epoch"],
                df_results[f"{iter_class}_rej"],
                ":",
                color=flav_cat[iter_class]["colour"],
                label=f"{flav_cat[iter_class]['legend_label']} - {label_extension}",
            )
            axes[counter].set_ylabel(
                f'{flav_cat[iter_class]["legend_label"]} Rejection',
                color=flav_cat[iter_class]["colour"],
            )

            if (
                recomm_rej_dict
                and recomm_rej_dict[f"{iter_class}_rej"] is not None
                and comp_tagger_name is not None
            ):
                axes[counter].axhline(
                    recomm_rej_dict[f"{iter_class}_rej"],
                    0,
                    df_results["epoch"].max(),
                    color=flav_cat[iter_class]["colour"],
                    lw=1.0,
                    alpha=1,
                    linestyle=(0, (5, 10)),
                    label=f"Recomm. {comp_tagger_name}",
                )

            axes[counter].tick_params(
                axis="y", labelcolor=flav_cat[iter_class]["colour"]
            )

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Increase y limit for ATLAS Logo
        ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)
        ax2.set_ylim(top=ax2.get_ylim()[1] * yAxisIncrease)

    else:
        fig, ax1 = plt.subplots(constrained_layout=True)
        legend_loc = (0.6, 0.75)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Rejection")

        for counter, iter_class in enumerate(class_labels_wo_main):
            # Plot rejection
            ax1.plot(
                df_results["epoch"],
                df_results[f"{iter_class}_rej"],
                fmt=":",
                color=flav_cat[iter_class]["colour"],
                label=flav_cat[iter_class]["legend_label"],
            )

            if (
                recomm_rej_dict is not None
                and f"{iter_class}_rej" in recomm_rej_dict
                and recomm_rej_dict[f"{iter_class}_rej"] is not None
            ):
                ax1.axhline(
                    recomm_rej_dict[f"{iter_class}_rej"],
                    0,
                    df_results["epoch"].max(),
                    color=flav_cat[iter_class]["colour"],
                    lw=1.0,
                    alpha=1,
                    linestyle=(0, (5, 10)),
                    label=f"Recomm. {comp_tagger_name} - {flav_cat[iter_class]['legend_label']}",
                )

        # Increase y limit for ATLAS Logo
        ax1.set_ylim(top=ax1.get_ylim()[1] * yAxisIncrease)

    if UseAtlasTag is True:
        SecondTag = (
            SecondTag
            + "\n{} fraction = {}".format(
                class_labels_wo_main[0], frac_dict[class_labels_wo_main[0]]
            )
            + "\nWP={:02d}%".format(int(target_beff * 100))
        )

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=yAxisAtlasTag,
        )

    fig.legend(ncol=ncol, loc=legend_loc)
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotLosses(
    df_results,
    plot_name,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    plot_datatype="",
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["loss"],
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
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=1.2 * ymax)
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuracies(
    df_results,
    plot_name,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    yAxisAtlasTag=0.9,
    plot_datatype="",
    ymin=None,
    ymax=None,
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["acc"],
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
    plot_ymin, plot_ymax = plt.ylim()
    if ymin is not None:
        plot_ymin = ymin
    if ymax is not None:
        plot_ymax = ymax
    else:
        plot_ymax = 1.2 * plot_ymax
    plt.ylim(ymin=plot_ymin, ymax=plot_ymax)
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotLossesUmami(
    df_results,
    plot_name,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    plot_datatype="",
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["umami_loss"],
        label="training loss UMAMI - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_loss"],
        label=r"val loss UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_loss"],
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

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
        )

    plt.legend()
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuraciesUmami(
    df_results,
    plot_name,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
    plot_datatype="",
    ymin=None,
    ymax=None,
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["umami_acc"],
        label="training acc UMAMI - hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_acc"],
        label=r"val acc UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_acc"],
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

    if UseAtlasTag is True:
        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
        )

    plt.legend()
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plot_ymin, plot_ymax = plt.ylim()
    if ymin is not None:
        plot_ymin = ymin
    if ymax is not None:
        plot_ymax = ymax
    else:
        plot_ymax = 1.2 * plot_ymax
    plt.ylim(ymin=plot_ymin, ymax=plot_ymax)
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def RunPerformanceCheck(
    train_config,
    compare_tagger=True,
    tagger_comp_var=["DL1r_pu", "DL1r_pc", "DL1r_pb"],
    comp_tagger_name="DL1r",
    WP_b=0.77,
    fc=0.018,
    fb=0.2,
    dict_file_name=None,
):
    logger.info("Running performance check.")
    Eval_parameters = train_config.Eval_parameters_validation
    plot_datatype = train_config.Eval_parameters_validation["plot_datatype"]
    bool_use_taus = train_config.bool_use_taus
    recommended_fc_values = {"DL1r": 0.018, "RNNIP": 0.08}
    recommended_fb_values = {"DL1r": 0.2, "RNNIP": 0.08}

    if (
        "fc_value" in Eval_parameters
        and Eval_parameters["fc_value"] is not None
    ):
        fc = Eval_parameters["fc_value"]
    if (
        "fb_value" in Eval_parameters
        and Eval_parameters["fb_value"] is not None
    ):
        fb = Eval_parameters["fb_value"]
    if bool_use_taus:
        if "ftauforb_value" in Eval_parameters:
            ftauforb = Eval_parameters["ftauforb_value"]
        else:
            ftauforb = None
        if "ftauforc_value" in Eval_parameters:
            ftauforc = Eval_parameters["ftauforc_value"]
        else:
            ftauforc = None
    else:
        ftauforc = None
        ftauforb = None

    c_rej, u_rej = None, None
    if bool_use_taus:
        tau_rej = None

    if compare_tagger:
        variables = ["HadronConeExclTruthLabelID"]
        variables += tagger_comp_var[:]
        df = pd.DataFrame(
            h5py.File(train_config.validation_file, "r")["/jets"][:][variables]
        )
        if bool_use_taus:
            df.query(
                "HadronConeExclTruthLabelID in [0, 4, 5, 15]", inplace=True
            )
            df.replace(
                {"HadronConeExclTruthLabelID": {4: 1, 5: 2, 15: 3}},
                inplace=True,
            )

            tagger_comp_var.append("DL1r_ptau")
            df["DL1r_ptau"] = 0
        else:
            df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
            df.replace(
                {"HadronConeExclTruthLabelID": {4: 1, 5: 2}}, inplace=True
            )

        y_true = GetBinaryLabels(df["HadronConeExclTruthLabelID"].values)
        if bool_use_taus:
            c_rej, u_rej, tau_rej = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                frac=recommended_fc_values[comp_tagger_name],
                use_taus=True,
            )
            b_rejC, u_rejC, tau_rejC = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                d_type="c",
                frac=recommended_fb_values[comp_tagger_name],
                use_taus=True,
            )
        else:
            c_rej, u_rej, _ = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                frac=recommended_fc_values[comp_tagger_name],
            )
            b_rejC, u_rejC, _ = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                d_type="c",
                frac=recommended_fc_values[comp_tagger_name],
            )
            tau_rej = None
            tau_rejC = None

    df_results = pd.read_json(dict_file_name)
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"Saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    if comp_tagger_name == "RNNIP" or comp_tagger_name == "DL1r":
        plot_name = f"{plot_dir}/rej-plot_val.{plot_datatype}"
        PlotRejPerEpoch(
            df_results=df_results,
            plot_name=plot_name,
            c_rej=c_rej,
            u_rej=u_rej,
            tau_rej=tau_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
                "tau_rej": r"tau-rej. - $t\bar{t}$",
            },
            rej_keys={
                "c_rej": "c_rej",
                "u_rej": "u_rej",
                "tau_rej": "tau_rej",
            },
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            ftau_value=ftauforb,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )
        plot_name = f"{plot_dir}/rej-plot_valC.pdf"
        PlotRejPerEpoch(
            df_results,
            plot_name,
            b_rej=b_rejC,
            u_rej=u_rejC,
            tau_rej=tau_rejC,
            rej_keys={
                "b_rej": "b_rejC",
                "u_rej": "u_rejC",
                "tau_rej": "tau_rejC",
            },
            labels={
                "b_rej": r"$b$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
                "tau_rej": r"tau-rej. - $t\bar{t}$",
            },
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fb_value=fb,
            ftau_value=ftauforc,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )

    if train_config.add_validation_file is not None:
        c_rej, u_rej = None, None
        if compare_tagger:
            variables = ["HadronConeExclTruthLabelID"]
            variables += tagger_comp_var[:]
            if bool_use_taus:
                variables.remove("DL1r_ptau")
            df = pd.DataFrame(
                h5py.File(train_config.add_validation_file, "r")["/jets"][:][
                    variables
                ]
            )
            if bool_use_taus:
                df.query(
                    "HadronConeExclTruthLabelID in [0, 4, 5, 15]",
                    inplace=True,
                )
                df.replace(
                    {"HadronConeExclTruthLabelID": {4: 1, 5: 2, 15: 3}},
                    inplace=True,
                )
                df["DL1r_ptau"] = 0
            else:
                df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
                df.replace(
                    {"HadronConeExclTruthLabelID": {4: 1, 5: 2}}, inplace=True
                )
            y_true = GetBinaryLabels(df["HadronConeExclTruthLabelID"].values)
            if bool_use_taus:
                c_rej, u_rej, tau_rej, _ = GetRejection(
                    df[tagger_comp_var[:]].values,
                    y_true,
                    WP_b,
                    frac=recommended_fc_values[comp_tagger_name],
                    use_taus=True,
                )
                b_rejC, u_rejC, tau_rejC, _ = GetRejection(
                    df[tagger_comp_var[:]].values,
                    y_true,
                    WP_b,
                    d_type="c",
                    frac=recommended_fb_values[comp_tagger_name],
                    use_taus=True,
                )
            else:
                c_rej, u_rej, _ = GetRejection(
                    df[tagger_comp_var[:]].values,
                    y_true,
                    WP_b,
                    frac=recommended_fc_values[comp_tagger_name],
                )
                b_rejC, u_rejC, _ = GetRejection(
                    df[tagger_comp_var[:]].values,
                    y_true,
                    WP_b,
                    d_type="c",
                    frac=recommended_fb_values[comp_tagger_name],
                )
                tau_rej = None
                tau_rejC = None

        if comp_tagger_name == "RNNIP" or comp_tagger_name == "DL1r":
            plot_name = f"{plot_dir}/rej-plot_val_add.{plot_datatype}"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                c_rej=c_rej,
                u_rej=u_rej,
                tau_rej=tau_rej,
                labels={
                    "c_rej": r"$c$-rej. - ext. $Z'$",
                    "u_rej": r"light-rej. - ext. $Z'$",
                    "tau_rej": r"tau-rej. - $t\bar{t}$",
                },
                rej_keys={
                    "c_rej": "c_rej_add",
                    "u_rej": "u_rej_add",
                    "tau_rej": "tau_rej_add",
                },
                comp_tagger_name=comp_tagger_name,
                target_beff=WP_b,
                fc_value=fc,
                ftau_value=ftauforb,
                UseAtlasTag=Eval_parameters["UseAtlasTag"],
                AtlasTag=Eval_parameters["AtlasTag"],
                SecondTag=Eval_parameters["SecondTag"],
            )
            plot_name = f"{plot_dir}/rej-plot_val_addC.pdf"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                b_rej=b_rejC,
                u_rej=u_rejC,
                tau_rej=tau_rejC,
                rej_keys={
                    "b_rej": "b_rejC_add",
                    "u_rej": "u_rejC_add",
                    "tau_rej": "tau_rejC_add",
                },
                labels={
                    "b_rej": r"$b$-rej. - $t\bar{t}$",
                    "u_rej": r"light-rej. - $t\bar{t}$",
                    "tau_rej": r"tau-rej. - $t\bar{t}$",
                },
                comp_tagger_name=comp_tagger_name,
                target_beff=WP_b,
                fb_value=fb,
                ftau_value=ftauforc,
                UseAtlasTag=Eval_parameters["UseAtlasTag"],
                AtlasTag=Eval_parameters["AtlasTag"],
                SecondTag=Eval_parameters["SecondTag"],
            )

    plot_name = f"{plot_dir}/loss-plot.{plot_datatype}"
    PlotLosses(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )
    acc_ymin, acc_ymax = None, None
    if "acc_ymin" in Eval_parameters:
        acc_ymin = Eval_parameters["acc_ymin"]
    if "acc_ymax" in Eval_parameters:
        acc_ymax = Eval_parameters["acc_ymax"]
    plot_name = f"{plot_dir}/accuracy-plot.{plot_datatype}"
    PlotAccuracies(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
        ymin=acc_ymin,
        ymax=acc_ymax,
    )


def RunPerformanceCheckDips(
    train_config,
    compare_tagger=True,
    tagger_comp_var=["rnnip_pu", "rnnip_pc", "rnnip_pb"],
    comp_tagger_name="rnnip",
    WP: float = 0.77,
    dict_file_name=None,
):
    logger.info("Running performance check.")

    # Load parameters from train config
    Eval_parameters = train_config.Eval_parameters_validation
    plot_datatype = Eval_parameters["plot_datatype"]
    frac_dict = Eval_parameters["frac_dict"]
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    recommended_frac_dict = Eval_parameters["frac_values_comp"]

    # Get class_labels variables etc. from global config
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(
        class_labels
    )

    if compare_tagger:

        # Get the tagger variables and the class label variables
        variables = class_label_vars + tagger_comp_var[:]

        # Load the Jets
        df = pd.DataFrame(
            h5py.File(train_config.validation_file, "r")["/jets"][:][variables]
        )

        # Iterate over the classes and remove all not used jets
        for class_id, class_label_var in zip(class_ids, class_label_vars):
            df.query(f"{class_label_var} in {class_id}", inplace=True)

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

        # Binarize the labels
        y_true = GetBinaryLabels(df["Umami_labels"].values)

        # Calculate rejections
        recomm_rej_dict, _ = GetRejection(
            y_pred=df[tagger_comp_var[:]].values,
            y_true=y_true,
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=recommended_frac_dict[comp_tagger_name],
            target_eff=WP,
        )

    else:
        recomm_rej_dict = None

    # Get dict from json
    df_results = pd.read_json(dict_file_name)

    # Define dir where the plots are saved
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot comparsion for the comparison taggers
    plot_name = f"{plot_dir}/rej-plot_val.{plot_datatype}"
    PlotRejPerEpoch(
        df_results=df_results,
        plot_name=plot_name,
        frac_dict=frac_dict,
        class_labels=class_labels,
        main_class=main_class,
        recomm_rej_dict=recomm_rej_dict,
        label_extension="$t\bar{t}$",
        comp_tagger_name=comp_tagger_name,
        target_beff=WP,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )

    if train_config.add_validation_file is not None:
        if compare_tagger:
            # Get the tagger variables and the class label variables
            variables = class_label_vars + tagger_comp_var[:]

            # Load the Jets
            df = pd.DataFrame(
                h5py.File(train_config.add_validation_file, "r")["/jets"][:][
                    variables
                ]
            )

            # Iterate over the classes and remove all not used jets
            for class_id, class_label_var in zip(class_ids, class_label_vars):
                df.query(f"{class_label_var} in {class_id}", inplace=True)

            # Init new column for string labels
            df["Umami_string_labels"] = np.zeros_like(df[class_label_vars[0]])
            df["Umami_labels"] = np.zeros_like(df[class_label_vars[0]])

            # Change type of column to string
            df = df.astype({"Umami_string_labels": "str"})

            # Iterate over the classes and add the correct labels to Umami columns
            for class_id, class_label_var, class_label in zip(
                class_ids, class_label_vars, flatten_class_labels
            ):
                indices_tochange = np.where(
                    df[class_label_var].values == class_id
                )

                # Add a string description which this class is
                df["Umami_string_labels"].values[
                    indices_tochange
                ] = class_label

                # Add the right column label to class
                df["Umami_labels"].values[
                    indices_tochange
                ] = class_labels.index(class_label)

            # Binarize the labels
            y_true = GetBinaryLabels(df["Umami_labels"].values)

            # Calculate rejections
            recomm_rej_dict, _ = GetRejection(
                y_pred=df[tagger_comp_var[:]].values,
                y_true=y_true,
                class_labels=class_labels,
                main_class=main_class,
                frac_dict=recommended_frac_dict[comp_tagger_name],
                target_eff=WP,
            )

        else:
            recomm_rej_dict = None

        plot_name = f"{plot_dir}/rej-plot_val_add.{plot_datatype}"
        PlotRejPerEpoch(
            df_results=df_results,
            plot_name=plot_name,
            frac_dict=frac_dict,
            class_labels=class_labels,
            main_class=main_class,
            recomm_rej_dict=recomm_rej_dict,
            label_extension="$ext. $Z'$",
            comp_tagger_name=comp_tagger_name,
            target_beff=WP,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )

    plot_name = f"{plot_dir}/loss-plot.{plot_datatype}"
    PlotLosses(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )
    acc_ymin, acc_ymax = None, None
    if "acc_ymin" in Eval_parameters:
        acc_ymin = Eval_parameters["acc_ymin"]
    if "acc_ymax" in Eval_parameters:
        acc_ymax = Eval_parameters["acc_ymax"]
    plot_name = f"{plot_dir}/accuracy-plot.{plot_datatype}"
    PlotAccuracies(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
        ymin=acc_ymin,
        ymax=acc_ymax,
    )
    plot_name = f"{plot_dir}/disc-cut-plot.{plot_datatype}"
    PlotDiscCutPerEpoch(
        df_results=df_results,
        plot_name=plot_name,
        target_beff=WP,
        frac_class="cjets",
        frac=frac_dict["cjets"],
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )


def RunPerformanceCheckUmami(
    train_config,
    compare_tagger=True,
    tagger_comp_var=["DL1r_pu", "DL1r_pc", "DL1r_pb"],
    comp_tagger_name="DL1r",
    WP_b=0.77,
    fc=0.018,
    dict_file_name=None,
):
    logger.info("Running performance check.")
    Eval_parameters = train_config.Eval_parameters_validation
    plot_datatype = train_config.Eval_parameters_validation["plot_datatype"]
    recommended_fc_values = {"DL1r": 0.018, "RNNIP": 0.08}
    c_rej, u_rej = None, None
    if compare_tagger:
        variables = ["HadronConeExclTruthLabelID"]
        variables += tagger_comp_var[:]
        df = pd.DataFrame(
            h5py.File(train_config.validation_file, "r")["/jets"][:][variables]
        )
        df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
        df.replace({"HadronConeExclTruthLabelID": {4: 1, 5: 2}}, inplace=True)
        y_true = GetBinaryLabels(df["HadronConeExclTruthLabelID"].values)
        c_rej, u_rej, _ = GetRejection(
            df[tagger_comp_var[:]].values,
            y_true,
            WP_b,
            frac=recommended_fc_values[comp_tagger_name],
        )

    df_results = pd.read_json(dict_file_name)
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    if comp_tagger_name == "RNNIP":
        plot_name = f"{plot_dir}/rej-plot_val_dips.{plot_datatype}"
        PlotRejPerEpoch(
            df_results,
            plot_name,
            c_rej=c_rej,
            u_rej=u_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
            },
            rej_keys={"c_rej": "c_rej_dips", "u_rej": "u_rej_dips"},
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )

    else:
        plot_name = f"{plot_dir}/rej-plot_val_umami.{plot_datatype}"
        PlotRejPerEpoch(
            df_results,
            plot_name,
            c_rej=c_rej,
            u_rej=u_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
            },
            rej_keys={"c_rej": "c_rej_umami", "u_rej": "u_rej_umami"},
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )

    if train_config.add_validation_file is not None:
        c_rej, u_rej = None, None
        if compare_tagger:
            variables = ["HadronConeExclTruthLabelID"]
            variables += tagger_comp_var[:]
            df = pd.DataFrame(
                h5py.File(train_config.add_validation_file, "r")["/jets"][:][
                    variables
                ]
            )
            df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
            df.replace(
                {"HadronConeExclTruthLabelID": {4: 1, 5: 2}}, inplace=True
            )
            y_true = GetBinaryLabels(df["HadronConeExclTruthLabelID"].values)
            c_rej, u_rej, _ = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                frac=recommended_fc_values[comp_tagger_name],
            )

        if comp_tagger_name == "RNNIP":
            plot_name = f"{plot_dir}/rej-plot_val_add_dips.{plot_datatype}"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                c_rej=c_rej,
                u_rej=u_rej,
                labels={
                    "c_rej": r"$c$-rej. - ext. $Z'$",
                    "u_rej": r"light-rej. - ext. $Z'$",
                },
                rej_keys={
                    "c_rej": "c_rej_dips_add",
                    "u_rej": "u_rej_dips_add",
                },
                comp_tagger_name=comp_tagger_name,
                target_beff=WP_b,
                fc_value=fc,
                UseAtlasTag=Eval_parameters["UseAtlasTag"],
                AtlasTag=Eval_parameters["AtlasTag"],
                SecondTag=Eval_parameters["SecondTag"],
            )

        else:
            plot_name = f"{plot_dir}/rej-plot_val_add_umami.{plot_datatype}"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                c_rej=c_rej,
                u_rej=u_rej,
                labels={
                    "c_rej": r"$c$-rej. - ext. $Z'$",
                    "u_rej": r"light-rej. - ext. $Z'$",
                },
                rej_keys={
                    "c_rej": "c_rej_umami_add",
                    "u_rej": "u_rej_umami_add",
                },
                comp_tagger_name=comp_tagger_name,
                target_beff=WP_b,
                fc_value=fc,
                UseAtlasTag=Eval_parameters["UseAtlasTag"],
                AtlasTag=Eval_parameters["AtlasTag"],
                SecondTag=Eval_parameters["SecondTag"],
            )

    plot_name = f"{plot_dir}/loss-plot.{plot_datatype}"
    PlotLossesUmami(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )
    acc_ymin, acc_ymax = None, None
    if "acc_ymin" in Eval_parameters:
        acc_ymin = Eval_parameters["acc_ymin"]
    if "acc_ymax" in Eval_parameters:
        acc_ymax = Eval_parameters["acc_ymax"]
    plot_name = f"{plot_dir}/accuracy-plot.{plot_datatype}"
    PlotAccuraciesUmami(
        df_results,
        plot_name,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
        ymin=acc_ymin,
        ymax=acc_ymax,
    )
    plot_name = f"{plot_dir}/disc-cut-plot.{plot_datatype}"
    PlotDiscCutPerEpochUmami(
        df_results=df_results,
        plot_name=plot_name,
        target_beff=WP_b,
        fc_value=fc,
        UseAtlasTag=Eval_parameters["UseAtlasTag"],
        AtlasTag=Eval_parameters["AtlasTag"],
        SecondTag=Eval_parameters["SecondTag"],
    )


def plot_validation(train_config, beff, cfrac, dict_file_name):
    RunPerformanceCheckUmami(
        train_config,
        compare_tagger=True,
        tagger_comp_var=["rnnip_pu", "rnnip_pc", "rnnip_pb"],
        comp_tagger_name="RNNIP",
        WP_b=beff,
        fc=cfrac,
        dict_file_name=dict_file_name,
    )
    RunPerformanceCheckUmami(
        train_config,
        compare_tagger=True,
        WP_b=beff,
        fc=cfrac,
        dict_file_name=dict_file_name,
    )
