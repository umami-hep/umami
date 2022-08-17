"""Execution script to run preprocessing steps."""
import argparse

import umami.preprocessing_tools as upt
from umami.configuration import logger, set_log_level


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
    config = upt.PreprocessConfiguration(args.config_file)

    # Check for preparation
    if args.prepare:

        # Copy config to output directory
        config.copy_to_out_dir("preparation")

        # Check if one specific sample is given
        if args.sample:
            samples = [args.sample]
        # If no specific sample is given iterate over the samples defined in the config
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

        # Copy config to output directory
        config.copy_to_out_dir("resampling")

        # Check the method which should be used for resampling
        if config.sampling["method"] == "count":
            sampler = upt.UnderSampling(config)

        elif config.sampling["method"] == "pdf":
            sampler = upt.PDFSampling(config, flavour=args.flavour)

        elif config.sampling["method"] == "importance_no_replace":
            sampler = upt.UnderSamplingNoReplace(config)

        elif config.sampling["method"] == "weighting":
            sampler = upt.Weighting(config)
        else:
            raise ValueError(
                f'{config.sampling["method"]} as sampling method is not supported!'
            )

        # Run the sampling with the selected method
        sampler.Run()

    # Calculate the scale dicts of the previous resampled files
    elif args.scaling:
        Scaling = upt.CalculateScaling(config)
        Scaling.get_scale_dict(chunk_size=args.chunk_size)

    # Check for final writing to disk in train format
    elif args.write:
        # Copy config to output directory
        config.copy_to_out_dir("write")

        Writer = upt.TrainSampleWriter(config, compression=config.config["compression"])
        Writer.write_train_sample()

    elif args.to_records:
        import umami.tf_tools as utft

        Converter = utft.h5_to_tf_record_converter(config)
        Converter.write_tfrecord()

    # Give error when nothing is used
    else:
        raise ValueError(
            "You need to define which part of the preprocessing you want to run!"
        )
