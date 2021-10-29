import argparse

import umami.preprocessing_tools as upt


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
        help="Enter the name of the config file to create the"
        " hybrid sample.",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        action="store_true",
        help="Stores also track information.",
    )
    parser.add_argument(
        "--tracks_name",
        default="tracks",
        help="Enter the name of the tracks dataset.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Choose sample type for hybrid sample preparation"
        " and merging.",
    )
    parser.add_argument(
        "--shuffle_array",
        action="store_true",
        help="Shuffle output arrays in hybrid sample" " preparation.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(1e5),
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
    # action.add_argument('--weighting', action='store_true',
    #                     help="Runs weighting.")

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = GetParser()
    config = upt.Configuration(args.config_file)

    if args.prepare:
        preparation_tool = upt.PrepareSamples(args, config)
        preparation_tool.Run()
    if args.resampling:
        if config.sampling["method"] == "count":
            us = upt.UnderSampling(config)
            us.Run()
        # here the other options such as PDFSampling etc. would be called
        if config.sampling["method"] == "probability_ratio":
            ust = upt.ProbabilityRatioUnderSampling(config)
            ust.Run()
    if args.scaling:
        Scaling = upt.Scaling(config)
        Scaling.GetScaleDict(chunkSize=args.chunk_size)
    if args.apply_scales:
        Scaling = upt.Scaling(config)
        Scaling.ApplyScales()
    if args.write:
        Writer = upt.TrainSampleWriter(config)
        Writer.WriteTrainSample()
