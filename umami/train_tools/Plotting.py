import pandas as pd
import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from umami.tools import applyATLASstyle


def PlotRejPerEpoch(df_results, plot_name, c_rej=None, u_rej=None,
                    rej_keys={"c_rej": "c_rej", "u_rej": "u_rej"},
                    labels={"c_rej": "c-rej. - val. sample",
                            "u_rej": "l-rej. - val. sample"}):
    applyATLASstyle(mtp)
    fig, ax1 = plt.subplots(constrained_layout=True)

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('light flavour jet rejection', color=color)
    ax1.plot(df_results["epoch"], df_results[rej_keys['u_rej']], ':',
             color=color, label=labels['u_rej'])
    if u_rej is not None:
        plt.axhline(u_rej, 0, df_results["epoch"].max(), color=color, lw=1.,
                    alpha=0.3, linestyle='--', label='recomm. DL1r')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel(r'$c$-jet rejection', color=color)
    ax2.plot(df_results["epoch"], df_results[rej_keys['c_rej']], ':',
             color=color, label=labels['c_rej'])
    if c_rej is not None:
        plt.axhline(c_rej, 0, df_results["epoch"].max(), color=color, lw=1.,
                    alpha=0.3, linestyle='--', label='recomm. DL1r')

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
