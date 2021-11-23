#!/usr/bin/env python
from umami.configuration import global_config  # isort:skip  # noqa
import argparse

import tensorflow as tf

import umami.train_tools as utt
from umami.preprocessing_tools import Configuration
from umami.train_tools import RunPerformanceCheck, get_class_prob_var_names


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
        help="Number of validation jets used",
    )

    parser.add_argument(
        "--beff",
        type=float,
        default=0.77,
        help="b-eff working point",
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
        "-t",
        "--tagger",
        type=str,
        required=True,
        help="""Model type which is used.
        You can either use 'dips', 'dips_cond_att', 'dl1' or 'umami'.""",
    )

    args = parser.parse_args()
    return args


def main(args, train_config, preprocess_config):
    # Check for nJets args
    if args.nJets is None:
        nJets = int(train_config.Eval_parameters_validation["n_jets"])

    else:
        nJets = args.nJets

    # Check if the tagger given is supported
    if args.tagger in ["umami", "dl1", "dips", "dips_cond_att"]:

        # If dict is given, the re-calculation is skipped
        if args.dict:
            output_file_name = args.dict
            parameters = utt.get_parameters_from_validation_dict_name(
                output_file_name
            )
            beff = parameters["WP"]

        else:
            # Calculate the validation metrics and save them
            output_file_name = utt.calc_validation_metrics(
                train_config=train_config,
                preprocess_config=preprocess_config,
                target_beff=args.beff
                if args.beff
                else train_config.Eval_parameters_validation["WP"],
                nJets=nJets,
                tagger=args.tagger,
            )
            beff = train_config.Eval_parameters_validation["WP"]

        # Run the Performance check with the values from the dict and plot them
        RunPerformanceCheck(
            train_config=train_config,
            tagger=args.tagger,
            tagger_comp_vars={
                f"{comp_tagger}": get_class_prob_var_names(
                    tagger_name=f"{comp_tagger}",
                    class_labels=train_config.NN_structure["class_labels"],
                )
                for comp_tagger in train_config.Eval_parameters_validation[
                    "tagger"
                ]
            },
            dict_file_name=output_file_name,
            train_history_dict=f"{train_config.model_name}/history.json",
            WP=beff,
        )

    else:
        raise ValueError(
            """
            You need to define a model type!\n
            You can either use 'dips', 'dips_cond_att', 'dl1' or 'umami'.
            """
        )


if __name__ == "__main__":
    args = GetParser()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    main(args, train_config, preprocess_config)
