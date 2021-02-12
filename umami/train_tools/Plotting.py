import os

import h5py
import matplotlib as mtp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from umami.preprocessing_tools import GetBinaryLabels
from umami.tools import applyATLASstyle, makeATLAStag
from umami.train_tools import GetRejection


def PlotRejPerEpoch(
    df_results,
    plot_name,
    c_rej=None,
    u_rej=None,
    rej_keys={"c_rej": "c_rej", "u_rej": "u_rej"},
    labels={
        "c_rej": r"$c$-rej. - val. sample",
        "u_rej": "light-rej. - val. sample",
    },
    comp_tagger_name="DL1r",
    target_beff=0.77,
    fc_value=0.018,
    plot_label="\n$\\sqrt{s}=13$ TeV, PFlow jets",
):
    applyATLASstyle(mtp)
    fig, ax1 = plt.subplots(constrained_layout=True)

    color = "tab:red"
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

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    # we already handled the x-label with ax1
    ax2.set_ylabel(r"$c$-jet rejection", color=color)
    ax2.plot(
        df_results["epoch"],
        df_results[rej_keys["c_rej"]],
        ":",
        color=color,
        label=labels["c_rej"],
    )
    if c_rej is not None:
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

    ax2.tick_params(axis="y", labelcolor=color)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.2)
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.2)

    makeATLAStag(
        plt.gca(),
        plt.gcf(),
        "Internal Simulation",
        (
            plot_label
            + "\nfc={}".format(fc_value)
            + ", WP={:02d}%".format(int(target_beff * 100))
        ),
    )
    fig.legend(ncol=1, loc=(0.6, 0.1))
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotLosses(
    df_results, plot_name, plot_label="\n$\\sqrt{s}=13$ TeV, PFlow jets"
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["loss"],
        label="training loss - downsampled hybrid sample",
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
    makeATLAStag(plt.gca(), plt.gcf(), "Internal Simulation", plot_label)
    plt.legend()
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=1.1 * ymax)
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuracies(
    df_results, plot_name, plot_label="\n$\\sqrt{s}=13$ TeV, PFlow jets"
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["acc"],
        label="training accuracy - downsampled hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_acc"],
        label=r"validation accuracy - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["val_accuracy_add"],
        label=r"validation accuracy - ext. $Z'$ sample",
    )
    makeATLAStag(plt.gca(), plt.gcf(), "Internal Simulation", plot_label)
    plt.legend()
    ymin, ymax = plt.ylim()
    plt.ylim(ymin=ymin, ymax=1.1 * ymax)
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotLossesUmami(
    df_results, plot_name, plot_label="\n$\\sqrt{s}=13$ TeV, PFlow jets"
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["umami_loss"],
        label="training loss UMAMI - downsampled hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_loss"],
        label=r"val loss UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_loss"],
        label="training loss DIPS - downsampled hybrid sample",
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
    makeATLAStag(plt.gca(), plt.gcf(), "Internal Simulation", plot_label)
    plt.legend()
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Loss")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuraciesUmami(
    df_results, plot_name, plot_label="\n$\\sqrt{s}=13$ TeV, PFlow jets"
):
    applyATLASstyle(mtp)
    plt.plot(
        df_results["epoch"],
        df_results["umami_acc"],
        label="training acc UMAMI - downsampled hybrid sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["umami_val_acc"],
        label=r"val acc UMAMI - $t\bar{t}$ sample",
    )
    plt.plot(
        df_results["epoch"],
        df_results["dips_acc"],
        label="training acc DIPS - downsampled hybrid sample",
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
    makeATLAStag(plt.gca(), plt.gcf(), "Internal Simulation", plot_label)
    plt.legend()
    plt.xlabel("Epoch", fontsize=14, horizontalalignment="right", x=1.0)
    plt.ylabel("Accuracy")
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def RunPerformanceCheck(
    train_config,
    compare_tagger=True,
    tagger_comp_var=["DL1r_pu", "DL1r_pc", "DL1r_pb"],
    comp_tagger_name="DL1r",
):
    print("Running performance check.")
    WP_b = train_config.Eval_parameters_validation["WP_b"]
    fc_value = train_config.Eval_parameters_validation["fc_value"]
    plot_label = train_config.Eval_parameters_validation["plot_label"]

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
        c_rej, u_rej = GetRejection(
            df[tagger_comp_var[:]].values, y_true, WP_b, fc_value
        )

    dictfile = f"{train_config.model_name}/DictFile.json"
    df_results = pd.read_json(dictfile)
    plot_dir = f"{train_config.model_name}/plots"
    print("saving plots to", plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = f"{plot_dir}/rej-plot_val.pdf"
    PlotRejPerEpoch(
        df_results,
        plot_name,
        c_rej,
        u_rej,
        labels={
            "c_rej": r"$c$-rej. - $t\bar{t}$",
            "u_rej": r"light-rej. - $t\bar{t}$",
        },
        comp_tagger_name=comp_tagger_name,
        target_beff=WP_b,
        fc_value=fc_value,
        plot_label=plot_label,
    )

    if train_config.add_validation_file is not None:
        plot_name = f"{plot_dir}/rej-plot_val_add.pdf"
        PlotRejPerEpoch(
            df_results,
            plot_name,  # c_rej, u_rej,
            rej_keys={"c_rej": "c_rej_add", "u_rej": "u_rej_add"},
            labels={
                "c_rej": r"$c$-rej. - ext. $Z'$",
                "u_rej": r"light-rej. - ext. $Z'$",
            },
            target_beff=WP_b,
            fc_value=fc_value,
            plot_label=plot_label,
        )

    plot_name = f"{plot_dir}/loss-plot.pdf"
    PlotLosses(df_results, plot_name, plot_label=plot_label)

    plot_name = f"{plot_dir}/accuracy-plot.pdf"
    PlotAccuracies(df_results, plot_name, plot_label=plot_label)


def RunPerformanceCheckUmami(
    train_config,
    compare_tagger=True,
    tagger_comp_var=["DL1r_pu", "DL1r_pc", "DL1r_pb"],
    comp_tagger_name="DL1r",
    WP_b=0.77,
    fc=0.018,
    dict_file_name=None,
):
    print("Running performance check.")
    plot_label = train_config.Eval_parameters_validation["plot_label"]
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
        c_rej, u_rej = GetRejection(
            df[tagger_comp_var[:]].values,
            y_true,
            WP_b,
            recommended_fc_values[comp_tagger_name],
        )

    df_results = pd.read_json(dict_file_name)
    plot_dir = f"{train_config.model_name}/plots"
    print("saving plots to", plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    if comp_tagger_name == "RNNIP":
        plot_name = f"{plot_dir}/rej-plot_val_dips.pdf"
        PlotRejPerEpoch(
            df_results,
            plot_name,
            c_rej,
            u_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
            },
            rej_keys={"c_rej": "c_rej_dips", "u_rej": "u_rej_dips"},
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            plot_label=plot_label,
        )

    else:
        plot_name = f"{plot_dir}/rej-plot_val_umami.pdf"
        PlotRejPerEpoch(
            df_results,
            plot_name,
            c_rej,
            u_rej,
            labels={
                "c_rej": r"$c$-rej. - $t\bar{t}$",
                "u_rej": r"light-rej. - $t\bar{t}$",
            },
            rej_keys={"c_rej": "c_rej_umami", "u_rej": "u_rej_umami"},
            comp_tagger_name=comp_tagger_name,
            target_beff=WP_b,
            fc_value=fc,
            plot_label=plot_label,
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
            c_rej, u_rej = GetRejection(
                df[tagger_comp_var[:]].values,
                y_true,
                WP_b,
                recommended_fc_values[comp_tagger_name],
            )

        if comp_tagger_name == "RNNIP":
            plot_name = f"{plot_dir}/rej-plot_val_add_dips.pdf"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                c_rej,
                u_rej,
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
                plot_label=plot_label,
            )

        else:
            plot_name = f"{plot_dir}/rej-plot_val_add_umami.pdf"
            PlotRejPerEpoch(
                df_results,
                plot_name,
                c_rej,
                u_rej,
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
                plot_label=plot_label,
            )

    plot_name = f"{plot_dir}/loss-plot.pdf"
    PlotLossesUmami(df_results, plot_name, plot_label=plot_label)

    plot_name = f"{plot_dir}/accuracy-plot.pdf"
    PlotAccuraciesUmami(df_results, plot_name, plot_label=plot_label)


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
