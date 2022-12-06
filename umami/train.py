#!/usr/bin/env python
"""Training script to perform various tagger trainings."""
import argparse

import h5py
import tensorflow as tf

import umami.models as utm
import umami.train_tools as utt
from umami.configuration import logger, set_log_level


def get_parser():
    """
    Argument parser for the train executable.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Train command line options.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to the training config file",
    )

    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs.")

    parser.add_argument(
        "-o",
        "--overwrite_config",
        action="store_true",
        help="Overwrite the configs files saved in metadata folder",
    )

    parser.add_argument(
        "-p",
        "--prepare",
        action="store_true",
        help="Only prepare the metadata folder and the model directory.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
    )

    parse_args = parser.parse_args()
    return parse_args


def check_train_file_format(input_file: str):
    """_summary_

    Parameters
    ----------
    input_file : str
        Path to input h5 file to check

    Raises
    ------
    KeyError
        If the specified key is not present in the input file
    """

    # If not using h5, don't check
    if not input_file.endswith(".h5"):
        return

    # Open the file and check for jets
    with h5py.File(input_file, "r") as f_h5:
        if "jets" not in f_h5.keys():
            raise KeyError(
                f"The input h5 file {input_file} does not contain a 'jets' "
                "group, suggesting this file has been produced with a version"
                "of umami <=0.13. You can fix this problem by checking out "
                "version 0.13 or older and rerunning this script, or re-make "
                "the training file in the newer format by re-running the --write "
                "preprocessing stage with umami >0.13."
            )


if __name__ == "__main__":
    # Get the args from parser
    args = get_parser()

    # Set logger level
    if args.verbose:
        set_log_level(logger, "DEBUG")

    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get the train and preprocess config
    train_config = utt.TrainConfiguration(args.config_file)

    # Get the tagger which is to be trained from the train config
    tagger_name = train_config.nn_structure.tagger

    # Check file format
    check_train_file_format(train_config.general.train_file)

    # Create the metadatafolder
    utt.create_metadata_folder(
        train_config_path=args.config_file,
        var_dict_path=train_config.general.var_dict,
        model_name=train_config.general.model_name,
        preprocess_config_path=train_config.general.preprocess_config.yaml_config,
        overwrite_config=bool(args.overwrite_config),
        model_file_path=train_config.general.model_file,
    )

    if not args.prepare:
        # Check for DIPS
        # TODO: Switch to case syntax with python 3.10
        if tagger_name.casefold() == "dips":
            utm.train_dips(
                args=args,
                train_config=train_config,
            )

        elif tagger_name.casefold() == "dl1":
            utm.train_dl1(
                args=args,
                train_config=train_config,
            )

        elif tagger_name.casefold() == "umami":
            utm.train_umami(
                args=args,
                train_config=train_config,
            )

        elif tagger_name.casefold() in ("cads", "dips_attention"):
            utm.train_cads(
                args=args,
                train_config=train_config,
            )

        elif tagger_name == "umami_cond_att":
            utm.train_umami_cond_att(
                args=args,
                train_config=train_config,
            )

        else:
            raise ValueError(
                f"""
                Tagger {tagger_name} is not supported! Possible taggers are
                'dips', 'dips_attention', 'cads', 'dl1',
                'umami' or 'umami_cond_att'.
                """
            )
