import argparse
import os
import numpy as np
import pandas as pd
import umami.evaluation_tools as uet


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Name of the plot config file")
    args = parser.parse_args()
    return args


def PlotROCRatio(plot_name, df_eff_rej):
    uet.plotROCRatio(teffs=[df_eff_rej["beff"] for x in range(25)],
                     beffs=[
                         1. / df_eff_rej['dl1r_urej'],
                         1. / df_eff_rej['umami_urej']
                            ],
                     labels=["DL1r", "DL1r retrained"],
                     ymax=5e3, ylabel='light flavour jet rejection',
                     binomialErrors=True,
                     nTest=1960184,
                     # styles=['--', "--", "--", "--"],
                     styles=['-', "-", "-", "-"],
                     # styles=[':', "-.", "--", "-."],
                     plot_name=plot_name + "/PFlow_u-rej_beff-dl1r.pdf",
                     alabel={"x": 0.63, "y": 0.7,
                             "s": r"$\sqrt{s}=13$ TeV, PFlow jets, " \
                             r"$t\bar{t}$ Sim." + "\next. hybrid training, " \
                             "fc=0.018",
                             "fontsize": 10},
                     # alabel={"x": 0.85, "y": 3e2, "s":r"$\sqrt{s}=13$ TeV,
                     # PFlow jets, $t\bar{t}$ Sim.", "fontsize":10},
                     # text='DL1r with different SMT variables - ext. hybrid
                     # training'
                     )

    uet.plotROCRatio(teffs=[df_eff_rej["beff"] for x in range(25)],
                     beffs=[
                         1. / df_eff_rej['dl1r_crej'],
                         1. / df_eff_rej['umami_crej']
                            ],
                     labels=["DL1r", "DL1r retrained"],
                     ymax=1e2, ylabel='c-jet rejection',
                     binomialErrors=True,
                     nTest=1960184,
                     # styles=['--', "--", "--", "--"],
                     styles=['-', "-", "-", "-"],
                     # styles=[':', "-.", "--", "-."],
                     plot_name=plot_name + "/PFlow_c-rej_beff-dl1r.pdf",
                     alabel={"x": 0.63, "y": 0.7,
                             "s": r"$\sqrt{s}=13$ TeV, PFlow jets, " \
                                  r"$t\bar{t}$ Sim." \
                                  "\next. hybrid training, fc=0.018",
                                  "fontsize": 10},
                     # alabel={"x": 0.85, "y": 3e2, "s":r"$\sqrt{s}=13$ TeV,
                     # PFlow jets, $t\bar{t}$ Sim.", "fontsize":10},
                     # text='DL1r with different SMT variables - ext. hybrid
                     # training'
                     )


def GetScore(pb, pc, pu, fc=0.018):
    pb = pb.astype('float64')
    pc = pc.astype('float64')
    pu = pu.astype('float64')
    add_small = 1e-10
    return np.log((pb + add_small) / ((1. - fc) * pu + fc * pc + add_small))


def SetUpPlots(plot_config):
    os.system(f"mkdir -p plots/{plot_config.plot_name}")
    # f_results = f"{plot_config.models}/results/results-{args.epoch}.h5"
    model_info = plot_config.models[0]
    # f_results = f"{model_info['name']}/results/"\
    #     f"results-{model_info['epoch']}.h5"
    # df_discs = pd.read_hdf(f_results, 'ttbar')
    f_results_eff_rej = f"{model_info['name']}/results/"\
        f"results-rej_per_eff-{model_info['epoch']}.h5"
    df_eff_rej = pd.read_hdf(f_results_eff_rej, 'ttbar')
    plot_name = f"plots/{plot_config.plot_name}"
    PlotROCRatio(plot_name, df_eff_rej)
    print("saved plot as", plot_name)

    # from mlxtend.evaluate import confusion_matrix

    # y_target = df_discs['labels']
    # y_predicted = np.argmax(df_discs[['umami_pu', 'umami_pc',
    # 'umami_pb']].values,
    #                         axis=1)
    # cm = confusion_matrix(y_target=y_target,
    #                     y_predicted=y_predicted,
    #                     binary=False)
    # import matplotlib.pyplot as plt
    # from mlxtend.plotting import plot_confusion_matrix
    # class_names = ['light', 'c', 'b']
    # fig, ax = plot_confusion_matrix(conf_mat=cm,
    #                                 colorbar=True,
    #                                 show_absolute=False,
    #                                 show_normed=True,
    #                                 class_names=class_names)
    # plt.tight_layout()
    # plt.savefig(f'{plot_name}/confusion_matrix_umami.pdf', transparent=True)

    # y_predicted = np.argmax(df_discs[['dips_pu', 'dips_pc',
    # 'dips_pb']].values,
    #                         axis=1)
    # cm = confusion_matrix(y_target=y_target,
    #                     y_predicted=y_predicted,
    #                     binary=False)
    # fig, ax = plot_confusion_matrix(conf_mat=cm,
    #                                 colorbar=True,
    #                                 show_absolute=False,
    #                                 show_normed=True,
    #                                 class_names=class_names)
    # plt.tight_layout()
    # plt.savefig(f'{plot_name}/confusion_matrix_dips.pdf', transparent=True)
    # df_discs["discs_umami"] = GetScore(df_discs["umami_pb"],
    #                                    df_discs["umami_pc"],
    #                                    df_discs["umami_pu"])
    # plt.clf()
    # plt.hist([df_discs.query("labels==2")["discs_umami"],
    #           df_discs.query("labels==1")["discs_umami"],
    #           df_discs.query("labels==0")["discs_umami"]],
    #          50, histtype='step',
    #          stacked=False, fill=False, density=1, label=["b-jets", "c-jets",
    #                                                       "l-jets"])
    # plt.legend()
    # plt.xlabel(u"$D_{b}$(umami)")
    # plt.tight_layout()
    # plt.savefig(f'{plot_name}/discriminant-umami.pdf', transparent=True)

    # df_discs["discs_dips"] = GetScore(df_discs["dips_pb"],
    #                                    df_discs["dips_pc"],
    #                                    df_discs["dips_pu"], 0.08)
    # plt.clf()
    # plt.hist([df_discs.query("labels==2")["discs_dips"],
    #           df_discs.query("labels==1")["discs_dips"],
    #           df_discs.query("labels==0")["discs_dips"]],
    #          50, histtype='step',
    #          stacked=False, fill=False, density=1, label=["b-jets", "c-jets",
    #                                                       "l-jets"])
    # plt.legend()
    # plt.xlabel(u"$D_{b}$(dips)")
    # plt.tight_layout()
    # plt.savefig(f'{plot_name}/discriminant-dips.pdf', transparent=True)

    # df_discs.to_hdf(
    #     f"{train_config.model_name}/results/results-{args.epoch}.h5",
    # "ttbar")
    # df_eff_rej.to_hdf(f"{train_config.model_name}/results/results-rej_per_eff"
    #                   f"-{args.epoch}.h5", "ttbar")


if __name__ == '__main__':
    args = GetParser()
    plot_config = uet.Configuration(args.config_file)
    SetUpPlots(plot_config)
    # train_config = utt.Configuration(args.config_file)
    # preprocess_config = Configuration(train_config.preprocess_config)
    # EvaluateModel(args, train_config, preprocess_config)
