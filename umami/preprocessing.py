#!/usr/bin/env python
"""Execution script to run preprocessing steps."""
import argparse
import os
from pathlib import Path

from upp.classes.preprocessing_config import PreprocessingConfig

import umami.preprocessing_tools as upt
from umami.configuration import logger, set_log_level
from umami.preprocessing_tools.configuration import GeneralSettings


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
        help="Enter the name of the config file to create the hybrid sample.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Choose sample type for hybrid sample preparation and merging.",
    )
    parser.add_argument(
        "--shuffle_array",
        action="store_true",
        help="Shuffle output arrays in hybrid sample preparation.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(1e6),
        help="Set the chunk size of the generators.",
    )
    parser.add_argument(
        "--upp",
        default=None,
        help=(
            "Specify if UPP should be used for this"
            "preprocessing config. 1 for True, 0 for False."
        ),
    )

    # possible job options for the different preprocessing steps
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "-p",
        "--prepare",
        action="store_true",
        help="Prepares hybrid sample (choose sample type).",
    )

    action.add_argument(
        "-u",
        "--resampling",
        action="store_true",
        help="Runs resampling.",
    )

    action.add_argument(
        "--weighting",
        action="store_true",
        help="Runs weighting.",
    )

    action.add_argument(
        "-s",
        "--scaling",
        action="store_true",
        help="Retrieves scaling and shifting factors.",
    )

    action.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="""Shuffles sample, applies scaling and writes
        training sample and training labels to disk""",
    )

    action.add_argument(
        "-r",
        "--to_records",
        action="store_true",
        help="convert h5 file into tf records",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
    )

    parser.add_argument(
        "--hybrid_validation",
        action="store_true",
        help="""When running the resampling, by giving this option, the hybrid
        validation sample is resampled and not the training sample.""",
    )

    parser.add_argument(
        "--flavour",
        nargs="+",
        default=None,
        help=(
            "Sets the flavour to process for PDF sampling. List with target,"
            " combining, plotting or index of flavour."
        ),
    )
    # To pass a list, let a space between the list entries:
    # e.g., --flavour target 0 1 2 plotting combining
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # Set logger level
    if args.verbose:
        set_log_level(logger, "DEBUG")

    # Load preprocess config file
    if args.upp is not None:
        if args.upp == "0":
            USE_UPP = False
            config = upt.PreprocessConfiguration(yaml_config=args.config_file)
        elif args.upp == "1":
            USE_UPP = True
            config = PreprocessingConfig.from_file(
                Path(args.config_file), split="train"
            )
            genral_dict = config.get_umami_general()
            general = GeneralSettings(**genral_dict)
            config.mimic_umami_config(general)
        else:
            raise ValueError(
                "You need to define if UPP should be used for this"
                "preprocessing config. 1 for True, 0 for False."
            )
    else:
        try:
            config = upt.PreprocessConfiguration(args.config_file)
            USE_UPP = False
        except TypeError:
            config = PreprocessingConfig.from_file(
                Path(args.config_file), split="train"
            )
            genral_dict = config.get_umami_general()
            general = GeneralSettings(**genral_dict)
            config.mimic_umami_config(general)
            USE_UPP = True

    # Check for preparation
    if args.prepare:
        if USE_UPP:
            logger.error("UPP does not support preparation!")
        else:
            # Copy config to output directory
            config.copy_to_out_dir("preparation")

            # Check if one specific sample is given
            if args.sample:
                samples = [args.sample]
            # If no specific sample is given iterate
            # over the samples defined in the config
            else:
                samples = config.preparation.samples.keys()
                logger.warning(
                    "No --sample was selected, using all in config file! This can"
                    " take a lot of time!"
                )
            for iter_sample in samples:
                # Set the argument in args to this sample and run prepare
                args.sample = iter_sample
                preparation_tool = upt.PrepareSamples(args, config)
                preparation_tool.run()

    # Check for resampling
    elif args.resampling:
        if USE_UPP:
            logger.info("Using UPP for resampling")
            os.system(f"preprocess --config {args.config_file} --no-norm --split all")
        else:
            # ensure output dir exists
            out_dir = Path(config.config["parameters"]["file_path"])
            out_dir.mkdir(parents=True, exist_ok=True)

            # Check if hybrid validation sample should be produced and set the option in
            # the config
            config.sampling.use_validation_samples = args.hybrid_validation

            # Copy config to output directory
            config.copy_to_out_dir("resampling")

            # Check the method which should be used for resampling
            if config.sampling.method == "count":
                sampler = upt.UnderSampling(config=config)

            elif config.sampling.method == "pdf":
                sampler = upt.PDFSampling(config=config, flavour=args.flavour)

            elif config.sampling.method == "importance_no_replace":
                sampler = upt.UnderSamplingNoReplace(config=config)

            elif config.sampling.method == "weighting":
                sampler = upt.Weighting(config=config)

            else:
                raise ValueError(
                    f"{config.sampling.method} as sampling method is not supported!"
                )

            # Run the sampling with the selected method
            sampler.Run()

            # Set the option back to False to ensure correct naming
            config.sampling.use_validation_samples = False

    # Calculate the scale dicts of the previous resampled files
    elif args.scaling:
        Scaling = upt.CalculateScaling(config)
        Scaling.get_scale_dict(chunk_size=args.chunk_size)

    # Check for final writing to disk in train format
    elif args.write:
        # Check if we wish to write the validation files instead
        config.sampling.use_validation_samples = args.hybrid_validation
        # Copy config to output directory
        config.copy_to_out_dir("write")

        Writer = upt.TrainSampleWriter(config, compression=config.general.compression)
        Writer.write_train_sample(chunk_size=args.chunk_size)

    elif args.to_records:
        import umami.tf_tools as utft

        Converter = utft.H5ToTFRecords(config)
        Converter.write_tfrecord()

    # Give error when nothing is used
    else:
        raise ValueError(
            "You need to define which part of the preprocessing you want to run!"
        )
