"""Execution script to run preprocessing steps."""
import argparse

import umami.preprocessing_tools as upt
from umami.configuration import logger


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
        "-a",
        "--apply_scales",
        action="store_true",
        help="Apllies scaling and shifting factors.",
    )

    action.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="""Shuffles sample and writes training sample and
        training labels to disk""",
    )

    action.add_argument(
        "-r",
        "--to_records",
        action="store_true",
        help="convert h5 file into tf records",
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
    args = GetParser()
    config = upt.Configuration(args.config_file)

    # Check for preparation
    if args.prepare:
        # Check if one specific sample is given
        if args.sample:
            preparation_tool = upt.PrepareSamples(args, config)
            preparation_tool.Run()

        # If no specific sample is given
        else:
            logger.warning(
                "No --sample was selected, using all in config file! This can"
                " take a lot of time!"
            )

            # Iterate over the samples defined in the config
            for iter_sample in config.preparation["samples"].keys():

                # Set the argument in args to this sample and run prepare
                args.sample = iter_sample
                preparation_tool = upt.PrepareSamples(args, config)
                preparation_tool.Run()

    # Check for resampling
    elif args.resampling:

        # Check the method which should be used for resampling
        if config.sampling["method"] == "count":
            sampler = upt.UnderSampling(config)

        elif config.sampling["method"] == "pdf":
            sampler = upt.PDFSampling(config, flavour=args.flavour)

        elif config.sampling["method"] == "probability_ratio":
            sampler = upt.ProbabilityRatioUnderSampling(config)

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
        Scaling = upt.Scaling(config)
        Scaling.GetScaleDict(chunkSize=args.chunk_size)

    # Apply scaling of the previous calculated scale dicts
    elif args.apply_scales:
        Scaling = upt.Scaling(config)
        Scaling.ApplyScales()

    # Check for final writing to disk in train format
    elif args.write:
        Writer = upt.TrainSampleWriter(config, compression=config.config["compression"])
        Writer.WriteTrainSample()

    elif args.to_records:
        Converter = upt.h5toTFRecordConverter(config)
        Converter.write_tfrecord()

    # Give error when nothing is used
    else:
        raise ValueError(
            "You need to define which part of the preprocessing you want to run!"
        )
