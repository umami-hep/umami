import pandas as pd
import os
import h5py
import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from umami.tools import applyATLASstyle
from umami.preprocessing_tools import GetBinaryLabels
from umami.train_tools import GetRejection


def PlotRejPerEpoch(df_results, plot_name, c_rej=None, u_rej=None,
                    rej_keys={"c_rej": "c_rej", "u_rej": "u_rej"},
                    labels={"c_rej": "c-rej. - val. sample",
                            "u_rej": "l-rej. - val. sample"},
                    comp_tagger_name='DL1r'):
    applyATLASstyle(mtp)
    fig, ax1 = plt.subplots(constrained_layout=True)

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('light flavour jet rejection', color=color)
    ax1.plot(df_results["epoch"], df_results[rej_keys['u_rej']], ':',
             color=color, label=labels['u_rej'])
    if u_rej is not None:
        plt.axhline(u_rej, 0, df_results["epoch"].max(), color=color, lw=1.,
                    alpha=0.3, linestyle='--',
                    label=f'recomm. {comp_tagger_name}')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel(r'$c$-jet rejection', color=color)
    ax2.plot(df_results["epoch"], df_results[rej_keys['c_rej']], ':',
             color=color, label=labels['c_rej'])
    if c_rej is not None:
        plt.axhline(c_rej, 0, df_results["epoch"].max(), color=color, lw=1.,
                    alpha=0.3, linestyle='--',
                    label=f'recomm. {comp_tagger_name}')

    ax2.tick_params(axis='y', labelcolor=color)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.legend(ncol=1, loc=(0.6, 0.1))
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotLosses(df_results, plot_name,):
    applyATLASstyle(mtp)
    plt.plot(df_results['epoch'], df_results['loss'],
             label='training loss - downsampled hybrid sample')
    plt.plot(df_results['epoch'], df_results['val_loss'],
             label='validation loss - ttbar sample')
    plt.plot(df_results['epoch'], df_results['val_loss_add'],
             label="validation loss - Z' sample")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def PlotAccuracies(df_results, plot_name,):
    applyATLASstyle(mtp)
    plt.plot(df_results['epoch'], df_results['acc'],
             label='training accuracy - downsampled hybrid sample')
    plt.plot(df_results['epoch'], df_results['val_acc'],
             label='validation accuracy - ttbar sample')
    plt.plot(df_results['epoch'], df_results['val_accuracy_add'],
             label="validation accuracy - Z' sample")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(plot_name, transparent=True)
    plt.cla()
    plt.clf()


def RunPerformanceCheck(train_config, compare_tagger=True,
                        tagger_comp_var=["DL1r_pu", "DL1r_pc", "DL1r_pb"],
                        comp_tagger_name='DL1r'):
    print("Running performance check.")
    c_rej, u_rej = None, None
    if compare_tagger:
        variables = ["HadronConeExclTruthLabelID"]
        variables += tagger_comp_var[:]
        df = pd.DataFrame(
            h5py.File(train_config.validation_file, 'r')['/jets'][:][
                variables])
        df.query('HadronConeExclTruthLabelID <= 5', inplace=True)
        df.replace({'HadronConeExclTruthLabelID': {4: 1, 5: 2}}, inplace=True)
        y_true = GetBinaryLabels(df['HadronConeExclTruthLabelID'].values)
        c_rej, u_rej = GetRejection(
            df[tagger_comp_var[:]].values,
            y_true)

    dictfile = f"{train_config.model_name}/DictFile.json"
    df_results = pd.read_json(dictfile)
    plot_dir = f"{train_config.model_name}/plots"
    print("saving plots to", plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = f"{plot_dir}/rej-plot_val.pdf"
    PlotRejPerEpoch(df_results, plot_name, c_rej, u_rej,
                    labels={"c_rej": r"c-rej. - $t\bar{t}$",
                            "u_rej": r"l-rej. - $t\bar{t}$"},
                    comp_tagger_name=comp_tagger_name)

    if train_config.add_validation_file is not None:
        plot_name = f"{plot_dir}/rej-plot_val_add.pdf"
        PlotRejPerEpoch(df_results, plot_name,  # c_rej, u_rej,
                        rej_keys={"c_rej": "c_rej_add",
                                  "u_rej": "u_rej_add"},
                        labels={"c_rej": r"c-rej. - ext. $Z'$",
                                "u_rej": r"l-rej. - ext. $Z'$"})

    plot_name = f"{plot_dir}/loss-plot.pdf"
    PlotLosses(df_results, plot_name)

    plot_name = f"{plot_dir}/accuracy-plot.pdf"
    PlotAccuracies(df_results, plot_name)
