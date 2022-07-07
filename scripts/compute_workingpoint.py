"""Script to determine efficiency WP cut values from tagger scores in input samples."""
from umami.configuration import logger, global_config  # isort:skip
from argparse import ArgumentParser

import numpy as np

import umami.train_tools as utt


def getArgumentParser():
    """Argparse option for compute_workingpoint script.

    Returns
    -------
    args: parse_args
    """
    parser = ArgumentParser()
    parser.add_argument("input_file_path")
    parser.add_argument("-t", "--tagger", default="DL1dLoose20210824r22")
    parser.add_argument("-c", "--class_labels", default=["bjets", "cjets", "ujets"])
    parser.add_argument("-p", "--prob_var_names", default=["pb", "pc", "pu"])
    parser.add_argument("-f", "--fractions", default=[0.0, 0.018, 0.982])
    parser.add_argument("-m", "--main_class", default="bjets")
    parser.add_argument("-w", "--working_points", default=[0.60, 0.70, 0.77, 0.85])
    parser.add_argument("-n", "--n_jets", default=22_000_000)
    return parser


def main():
    """Main function called when executing script."""
    args = getArgumentParser().parse_args()
    tagger = args.tagger
    main_class = args.main_class
    class_labels = args.class_labels
    prob_var_names = args.prob_var_names

    # fractions of jet flavours used in training
    frac_dict = {}
    for jet_class, fraction in zip(class_labels, args.fractions):
        if abs(fraction) > 0.0:
            frac_dict[jet_class] = fraction

    # list of variables for tagger
    variables = [tagger + "_" + scorename for scorename in ["pb", "pc", "pu"]]
    variables += [
        "HadronConeExclTruthLabelID",
        "HadronConeExclExtendedTruthLabelID",
        global_config.pTvariable,
        global_config.etavariable,
    ]

    # load jets from file
    logger.info("Loading jets from file(s) %s...", args.input_file_path)
    jets, _ = utt.LoadJetsFromFile(
        args.input_file_path,
        class_labels=class_labels,
        n_jets=int(args.n_jets),
        variables=variables,
    )

    # filter jets for efficiency determination (typically b-jets)
    selected_jets = jets[jets["Umami_string_labels"] == main_class]

    # compute b-tagging score using function in umami training tools
    for flav_index, _ in enumerate(class_labels):
        if flav_index == 0:
            tmp = selected_jets[f"{tagger}_{prob_var_names[flav_index]}"].values
        else:
            tmp = np.append(
                tmp,
                selected_jets[f"{tagger}_{prob_var_names[flav_index]}"].values,
            )
    # reshape to wrong sorted (transpose change it to correct shape)
    y_pred = tmp.reshape((len(class_labels), -1))
    y_pred = np.transpose(y_pred)
    d_b = utt.GetScore(
        y_pred=y_pred,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
    )

    # obtain cut values for efficiency working points
    cut_values = {}
    for eff_wp in args.working_points:
        d_b_cutvalue = np.quantile(d_b, 1.0 - float(eff_wp))
        cut_values[f"{eff_wp}"] = d_b_cutvalue

    logger.info("Determined efficiency working point cut values:")
    for eff, cut in cut_values.items():
        logger.info("Efficiency: %.3f | d_b cut value: %.3f", eff, cut)


if __name__ == "__main__":
    main()
