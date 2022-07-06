#!/usr/bin/env python
"""Execution script for epoch performance plotting."""
from umami.configuration import logger, set_log_level  # isort:skip

import argparse

import tensorflow as tf

import umami.train_tools as utt
from umami.helper_tools import get_class_prob_var_names
from umami.plotting_tools import run_validation_check
from umami.preprocessing_tools import Configuration


def get_parser():
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
        "--n_jets",
        type=int,
        help="Number of validation jets used",
    )

    parser.add_argument(
        "--beff",
        type=float,
        default=None,
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
        "-f",
        "--format",
        type=str,
        default=None,
        help="""Overwrite the file format for the plots given in the train
        config file""",
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
        You can either use 'dips', 'dips_attention', 'cads', 'dl1',
        'umami' or 'umami_cond_att'.
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
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

    # Check for format option
    if args.format:
        train_config.Validation_metrics_settings["plot_datatype"] = args.format

    # Check for n_jets args
    if args.n_jets is None:
        n_jets = (
            int(val_params["n_jets"])
            if "n_jets" in val_params
            else int(eval_params["n_jets"])
        )

    else:
        n_jets = args.n_jets

    # Get the b eff
    if args.beff is None:
        working_point = (
            float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
        )
    else:
        working_point = args.beff

    # Get the tagger from args. If not given, use the one from train config
    if args.tagger:
        tagger = args.tagger

    else:
        tagger = train_config.NN_structure["tagger"]

    # Check if the tagger given is supported
    if tagger.casefold() in [
        "umami",
        "umami_cond_att",
        "dl1",
        "dips",
        "dips_attention",
        "cads",
    ]:

        # If dict is given, the re-calculation is skipped
        if args.dict:
            output_file_name = args.dict
            parameters = utt.get_parameters_from_validation_dict_name(output_file_name)
            beff = parameters["WP"]

        elif args.recalculate:
            # Get the filename of the train metrics file
            train_metrics_file_name, _ = utt.get_metrics_file_name(
                working_point=working_point,
                n_jets=n_jets,
                dir_name=train_config.model_name,
            )

            # Calculate the validation metrics and save them
            val_metrics_file_name = utt.calc_validation_metrics(
                train_config=train_config,
                preprocess_config=preprocess_config,
                target_beff=working_point,
                n_jets=n_jets,
                tagger=tagger,
            )
            beff = working_point

        else:
            (
                train_metrics_file_name,
                val_metrics_file_name,
            ) = utt.get_metrics_file_name(
                working_point=working_point,
                n_jets=n_jets,
                dir_name=train_config.model_name,
            )
            beff = working_point

        # Get the comparison tagger variables
        if val_params["taggers_from_file"]:
            comp_tagger_list = (
                val_params["taggers_from_file"].keys()
                if isinstance(val_params["taggers_from_file"], dict)
                else val_params["taggers_from_file"]
            )

        # Run the Performance check with the values from the dict and plot them
        run_validation_check(
            train_config=train_config,
            tagger=tagger,
            tagger_comp_vars={
                f"{comp_tagger}": get_class_prob_var_names(
                    tagger_name=f"{comp_tagger}",
                    class_labels=train_config.NN_structure["class_labels"],
                )
                for comp_tagger in comp_tagger_list
            }
            if "taggers_from_file" in val_params
            and val_params["taggers_from_file"] is not None
            else None,
            train_metrics_file_name=train_metrics_file_name,
            val_metrics_file_name=val_metrics_file_name,
            working_point=beff,
        )

    else:
        raise ValueError(
            f"""
            Model type {tagger} is not supported.
            You can either use 'dips', 'dips_attention', 'cads', 'dl1',
            'umami' or 'umami_cond_att'.
            """
        )


if __name__ == "__main__":
    arg_parser = get_parser()

    # Set logger level
    if arg_parser.verbose:
        set_log_level(logger, "DEBUG")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    training_config = utt.Configuration(arg_parser.config_file)
    preprocessing_config = Configuration(training_config.preprocess_config)
    main(arg_parser, training_config, preprocessing_config)
