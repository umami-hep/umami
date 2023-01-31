#!/usr/bin/env python
"""Execution script to run merging of ttbar samples."""
import argparse

import umami.preprocessing_tools as upt


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
        "-i",
        "--index_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Enter the directory containing the index files to pass "
            "to the merging script. (Optional)"
        ),
    )
    parser.add_argument(
        "-p",
        "--prepare",
        action="store_true",
        help="Get list of indices for merging.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge samples.",
    )
    parser.add_argument(
        "--file_range",
        nargs=2,
        help=(
            "The start and end output file number that this script "
            "should merge. e.g. if you want to merge files 0-10, then"
            "you would pass 0 11 (end is exclusive)."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    if not args.prepare and not args.merge:
        raise ValueError("Must set either -p or -m")

    if args.file_range and not args.merge:
        raise ValueError("file_range can only be passed with merge.")

    # Load preprocess config file
    # config = upt.PreprocessConfiguration(args.config_file)

    # load yaml config file
    config = upt.MergeConfig(args.config_file)

    # Setup merging tool
    merge_tool = upt.TTbarMerge(config)

    if args.prepare:
        merge_tool.get_indices()

    elif args.merge:
        merge_tool.merge(args.file_range, args.index_dir)

    # Give error when nothing is used
    else:
        raise ValueError(
            "You need to define which part of the preprocessing you want to run!"
        )
