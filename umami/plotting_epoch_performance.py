#!/usr/bin/env python
"""Execution script for epoch performance plotting."""
from umami.configuration import (  # isort:skip # noqa # pylint: disable=unused-import
    global_config,
)
import argparse

import tensorflow as tf

import umami.train_tools as utt
from umami.classification_tools import get_class_prob_var_names
from umami.preprocessing_tools import Configuration
from umami.train_tools import RunPerformanceCheck


def GetParser():
    """
    Argument parser for Preprocessing script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Preprocessing command lineoptions.")

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
        the available values are plotted.""",
    )

    parser.add_argument(
        "-r",
        "--recalculate",
        action="store_true",
        help="""The validation json (with the values per epoch inside)
        will be recalculated using the values from the train config.""",
    )

    parser.add_argument(
        "-t",
        "--tagger",
        type=str,
        help="""Model type which is used.
        You can either use 'dips', 'cads', 'dl1' or 'umami'.""",
    )

    return parser.parse_args()


def main(args, train_config, preprocess_config):
    """Executes plotting of epoch performance plots

    Parameters
    ----------
    args : parser.parse_args
        command line argument parser options
    train_config : object
        configuration file used for training
    preprocess_config : object
        configuration file used for preprocessing

    Raises
    ------
    ValueError
        If the given tagger is not supported.
    """ """"""
    # Get the eval and val params from the train config
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation

    # Check for nJets args
    if args.nJets is None:
        nJets = (
            int(val_params["n_jets"])
            if "n_jets" in val_params
            else int(eval_params["n_jets"])
        )

    else:
        nJets = args.nJets

    # Get the b eff
    if args.beff:
        WP = args.beff

    else:
        WP = float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])

    # Get the tagger from args. If not given, use the one from train config
    if args.tagger:
        tagger = args.tagger

    else:
        tagger = train_config.NN_structure["tagger"]

    # Check if the tagger given is supported
    if tagger.casefold() in ["umami", "dl1", "dips", "cads"]:

        # If dict is given, the re-calculation is skipped
        if args.dict:
            output_file_name = args.dict
            parameters = utt.get_parameters_from_validation_dict_name(output_file_name)
            beff = parameters["WP"]

        elif args.recalculate:
            # Calculate the validation metrics and save them
            output_file_name = utt.calc_validation_metrics(
                train_config=train_config,
                preprocess_config=preprocess_config,
                target_beff=WP,
                nJets=nJets,
                tagger=tagger,
            )
            beff = WP

        else:
            output_file_name = utt.get_validation_dict_name(
                WP=WP,
                n_jets=nJets,
                dir_name=train_config.model_name,
            )
            beff = WP

        # Run the Performance check with the values from the dict and plot them
        RunPerformanceCheck(
            train_config=train_config,
            tagger=tagger,
            tagger_comp_vars={
                f"{comp_tagger}": get_class_prob_var_names(
                    tagger_name=f"{comp_tagger}",
                    class_labels=train_config.NN_structure["class_labels"],
                )
                for comp_tagger in train_config.Validation_metrics_settings[
                    "taggers_from_file"
                ]
            }
            if "taggers_from_file" in train_config.Validation_metrics_settings
            else None,
            dict_file_name=output_file_name,
            train_history_dict_file_name=f"{train_config.model_name}/history.json",
            WP=beff,
        )

    else:
        raise ValueError(
            """
            You need to define a model type!\n
            You can either use 'dips', 'cads', 'dl1' or 'umami'.
            """
        )


if __name__ == "__main__":
    parser_args = GetParser()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    training_config = utt.Configuration(parser_args.config_file)
    preprocessing_config = Configuration(training_config.preprocess_config)
    main(parser_args, training_config, preprocessing_config)
