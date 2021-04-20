#!/usr/bin/env python
import argparse
import os

import tensorflow as tf

import umami.train_tools as utt
from umami.preprocessing_tools import Configuration

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing command line" "options."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Name of the training config file",
    )

    parser.add_argument(
        "--nJets",
        type=int,
        default=300000,
        help="Number of validation jets used",
    )

    parser.add_argument(
        "--beff",
        type=float,
        default=0.77,
        help="b-eff working point",
    )

    parser.add_argument(
        "--cfrac",
        type=float,
        default=0.018,
        help="charm fraction in likelihood",
    )

    parser.add_argument(
        "-d",
        "--dict",
        type=str,
        default=None,
        help="""Name of the json file which should be plotted. With
        this option the validation metrics are NOT calculated! Only
        the available values are plotted. This also means that
        --cfrac, --beff and --nJets have no impact on anything!""",
    )

    parser.add_argument(
        "--dips",
        action="store_true",
        help="Setting the model epoch performance to dips.",
    )

    parser.add_argument(
        "--dl1",
        action="store_true",
        help="Setting the model epoch performance to DL1.",
    )

    args = parser.parse_args()
    return args


def main(args, train_config, preprocess_config):
    if args.dl1:
        dictfile = f"{train_config.model_name}/DictFile.json"
        utt.RunPerformanceCheck(
            train_config,
            dict_file_name=dictfile,
        )
    else:
        if args.dict:
            output_file_name = args.dict
            parameters = utt.get_parameters_from_validation_dict_name(
                output_file_name
            )
            beff = parameters["WP_b"]
            cfrac = parameters["fc_value"]

        else:
            if args.dips:
                output_file_name = utt.calc_validation_metrics_dips(
                    train_config,
                    preprocess_config,
                    args.beff,
                    args.cfrac,
                    args.nJets,
                )
                beff = args.beff
                cfrac = args.cfrac

            else:
                output_file_name = utt.calc_validation_metrics(
                    train_config,
                    preprocess_config,
                    args.beff,
                    args.cfrac,
                    args.nJets,
                )
                beff = args.beff
                cfrac = args.cfrac

        if args.dips:
            utt.plot_validation_dips(
                train_config, beff, cfrac, dict_file_name=output_file_name
            )

        else:
            utt.plot_validation(
                train_config, beff, cfrac, dict_file_name=output_file_name
            )


if __name__ == "__main__":
    args = GetParser()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    main(args, train_config, preprocess_config)
