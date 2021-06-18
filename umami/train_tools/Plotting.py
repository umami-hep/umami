import os

import h5py
import matplotlib as mtp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from umami.configuration import logger
from umami.preprocessing_tools import GetBinaryLabels
from umami.tools import applyATLASstyle, makeATLAStag
from umami.train_tools import GetRejection


def PlotDiscCutPerEpoch(
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
    plot_name,
    b_rej=None,
    c_rej=None,
    u_rej=None,
    tau_rej=None,
    rej_keys={
        "c_rej": "c_rej",
        "u_rej": "u_rej",
        "tau_rej": "tau_rej",
        "b_rej": "b_rej",
    },
    labels={
        "b_rej": r"$b$-rej. - val. sample",
        "c_rej": r"$c$-rej. - val. sample",
        "u_rej": "light-rej. - val. sample",
        "tau_rej": "tau-rej. - val. sample",
    },
    comp_tagger_name="DL1r",
    target_beff=0.77,
    fc_value=0.018,
    fb_value=0.2,
    ftau_value=None,
    UseAtlasTag=True,
    AtlasTag="Internal Simulation",
    SecondTag="\n$\\sqrt{s}=13$ TeV, PFlow jets",
):
    applyATLASstyle(mtp)
    fig, ax1 = plt.subplots(constrained_layout=True)
    legend_loc = (0.6, 0.75)
    color = "#2ca02c"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("light flavour jet rejection", color=color)
    ax1.plot(
        df_results["epoch"],
        df_results[rej_keys["u_rej"]],
        ":",
        color=color,
        label=labels["u_rej"],
    )
    if u_rej is not None:
        ax1.axhline(
            u_rej,
            0,
            df_results["epoch"].max(),
            color=color,
            lw=1.0,
            alpha=1,
            linestyle=(0, (5, 10)),
            label=f"recomm. {comp_tagger_name}",
        )
    ax1.tick_params(axis="y", labelcolor=color)

    if tau_rej is not None:
        color = "#7c5295"
        ax1.set_ylabel("light and tau flavour jet rejection", color="k")
        ax1.tick_params(axis="y", labelcolor="k")
        ax1.plot(
            df_results["epoch"],
            df_results[rej_keys["tau_rej"]],
            ":",
            color=color,
            label=labels["tau_rej"],
        )
        ax1.axhline(
            tau_rej,
            0,
            df_results["epoch"].max(),
            color=color,
            lw=1.0,
            alpha=1,
            linestyle=(0, (5, 10)),
            label=f"recomm. {comp_tagger_name}",
        )
        # Also change the location of the legend
        legend_loc = (0.6, 0.68)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    f_name = "fc"
    # we already handled the x-label with ax1
    if b_rej is not None:
        color = "#1f77b4"
        ax2.set_ylabel(r"$b$-jet rejection", color=color)
        ax2.plot(
            df_results["epoch"],
            df_results[rej_keys["b_rej"]],
            ":",
            color=color,
            label=labels["b_rej"],
        )

        ax2.axhline(
            b_rej,
            0,
            df_results["epoch"].max(),
            color=color,
            lw=1.0,
            alpha=1,
            linestyle=(7, (5, 10)),
            label=f"recomm. {comp_tagger_name}",
        )
        f_name = "fb"
        f_value = fb_value
    if c_rej is not None:
        color = "#ff7f0e"
        ax2.set_ylabel(r"$c$-jet rejection", color=color)
        ax2.plot(
            df_results["epoch"],
            df_results[rej_keys["c_rej"]],
            ":",
            color=color,
            label=labels["c_rej"],
        )

        ax2.axhline(
            c_rej,
            0,
            df_results["epoch"].max(),
            color=color,
            lw=1.0,
            alpha=1,
            linestyle=(7, (5, 10)),
            label=f"recomm. {comp_tagger_name}",
        )
        f_name = "fc"
        f_value = fc_value
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.2)
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.2)

    if UseAtlasTag is True:
        if tau_rej is None:
            SecondTag = (
                SecondTag
                + "\n{}={}".format(f_name, f_value)
                + ", WP={:02d}%".format(int(target_beff * 100))
            )
        else:
            if ftau_value is None:
                SecondTag = (
                    SecondTag
                    + "\n{}={}".format(f_name, f_value)
                    + "\n{}={}".format(r"$f_{tau}$", 1 - f_value)
                    + ", WP={:02d}%".format(int(target_beff * 100))
                )
            else:
                SecondTag = (
                    SecondTag
                    + "\n{}={}".format(f_name, f_value)
                    + "\n{}={}".format(r"$f_{tau}$", ftau_value)
                    + ", WP={:02d}%".format(int(target_beff * 100))
                )

        makeATLAStag(
            ax=plt.gca(),
            fig=plt.gcf(),
            first_tag=AtlasTag,
            second_tag=SecondTag,
            ymax=0.9,
        )

    fig.legend(ncol=1, loc=legend_loc)
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
    comp_tagger_name="RNNIP",
    WP_b=0.77,
    fc=0.018,
    fb=0.2,
    dict_file_name=None,
):
    logger.info("Running performance check.")
    Eval_parameters = train_config.Eval_parameters_validation
    plot_datatype = train_config.Eval_parameters_validation["plot_datatype"]
    recommended_fc_values = {"DL1r": 0.018, "RNNIP": 0.08}

    if (
        "fc_value" in Eval_parameters
        and Eval_parameters["fc_value"] is not None
    ):
        fc = Eval_parameters["fc_value"]

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

    else:
        c_rej, u_rej = None, None

    df_results = pd.read_json(dict_file_name)
    plot_dir = f"{train_config.model_name}/plots"
    logger.info(f"saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    if comp_tagger_name == "RNNIP" or comp_tagger_name == "DL1r":
        plot_name = f"{plot_dir}/rej-plot_val.{plot_datatype}"
        PlotRejPerEpoch(
            df_results=df_results,
            plot_name=plot_name,
            c_rej=c_rej,
            u_rej=u_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
            },
            rej_keys={
                "c_rej": "c_rej",
                "u_rej": "u_rej",
            },
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            UseAtlasTag=Eval_parameters["UseAtlasTag"],
            AtlasTag=Eval_parameters["AtlasTag"],
            SecondTag=Eval_parameters["SecondTag"],
        )

    if train_config.add_validation_file is not None:
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
        else:
            c_rej, u_rej = None, None

        if comp_tagger_name == "RNNIP" or comp_tagger_name == "DL1r":
            plot_name = f"{plot_dir}/rej-plot_val_add.{plot_datatype}"
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
                    "c_rej": "c_rej_add",
                    "u_rej": "u_rej_add",
                },
                comp_tagger_name=comp_tagger_name,
                target_beff=WP_b,
                fc_value=fc,
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
        target_beff=WP_b,
        fc_value=fc,
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


def plot_validation_dips(train_config, beff, cfrac, dict_file_name):
    RunPerformanceCheckDips(
        train_config,
        compare_tagger=True,
        tagger_comp_var=["rnnip_pu", "rnnip_pc", "rnnip_pb"],
        comp_tagger_name="RNNIP",
        WP_b=beff,
        fc=cfrac,
        dict_file_name=dict_file_name,
    )
