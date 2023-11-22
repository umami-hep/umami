"""
This script automatically creates a yaml file with the different
samples needed for the prepare step based on the given
parameters.
"""
import os
from argparse import ArgumentParser
from pathlib import Path

from umami.preprocessing_tools import PreprocessConfiguration
from umami.tools.yaml_tools import YAML


def get_parser():
    """
    Argument parser for the train executable.

    Returns
    -------
    args: parse_args
    """
    parser = ArgumentParser(description="Create Samples command line options.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help=(
            "Path to the preprocessing config file for which the samples"
            " will be created"
        ),
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path where the samples dict yaml file will be saved.",
    )

    return parser.parse_args()


def build_preparation_sample_dict(
    preprocess_config: str,
    output_path: str,
    categories: list = None,
    sample_types: list = None,
    n_jets: dict = None,
) -> None:
    """Build the samples dict from given arguments and dump it into yaml

    Parameters
    ----------
    preprocess_config : str
        Path to your preprocess config file you want to use
    output_path : str
        Output path for the yaml file where the samples are
        dumped in.
    categories : list, optional
        Categories that are to be used, by default None
    sample_types : list, optional
        Which samples type are to be used, by default None
    n_jets : dict, optional
        Dict with the number of jets to use for the different
        steps (training/validation/testing), by default None
    """

    # Set default values
    if categories is None:
        categories = ["ujets", "cjets", "bjets"]

    if sample_types is None:
        sample_types = ["ttbar", "zprime"]

    if n_jets is None:
        n_jets = {
            "training": int(10e6),
            "validation": int(4e6),
            "testing": int(4e6),
        }

    # Get everything needed for the yaml file
    yaml = YAML(typ="safe", pure=True)
    samples_file = Path(output_path)

    # Load the preprocessing config file
    config = PreprocessConfiguration(preprocess_config)

    # Define and start building the samples dict
    samples = {}

    # Iterate over the steps training, validation, testing
    for step, _ in n_jets.items():
        # Create the samples which are used for the resampling
        # Note: Validation is also used to create the samples
        # needed to create the hybrid validation file
        if step in ("training", "validation"):
            # Iterate over the ttbar, zprime
            for sample_type in sample_types:
                # Iterate over the flavours
                for category in categories:
                    samples[f"{step}_{sample_type}_{category}"] = {
                        "type": sample_type,
                        "category": category,
                        "n_jets": n_jets[step],
                        "cuts": config.config["cut_parameters"][
                            f".cuts_template_{step}_{sample_type}"
                        ]["cuts"]
                        if step == "training"
                        else config.config["cut_parameters"][f".cuts_template_{step}"][
                            "cuts"
                        ],
                        "output_name": os.path.join(
                            config.config["parameters"]["sample_path"],
                            f"{category}_{step}_{sample_type}_PFlow.h5",
                        ),
                    }

        # Create the (not resampled) validation/testing samples
        if step in ("validation", "testing"):
            for sample_type in sample_types:
                samples[f"{step}_{sample_type}"] = {
                    "type": sample_type,
                    "category": "inclusive",
                    "n_jets": n_jets[step],
                    "cuts": config.config["cut_parameters"][f".cuts_template_{step}"][
                        "cuts"
                    ],
                    "output_name": os.path.join(
                        config.config["parameters"]["sample_path"],
                        f"inclusive_{step}_{sample_type}_PFlow.h5",
                    ),
                }

    # Dump samples dicts
    yaml.dump(samples, samples_file)


if __name__ == "__main__":
    args = get_parser()

    categories_to_extract = ["ujets", "cjets", "bjets"]
    sample_types_to_extract = ["ttbar", "zprime"]
    n_jets_categories = {
        "training": int(10e6),
        "validation": int(4e6),
        "testing": int(4e6),
    }

    build_preparation_sample_dict(
        preprocess_config=args.config_file,
        output_path=args.output_path,
        categories=categories_to_extract,
        sample_types=sample_types_to_extract,
        n_jets=n_jets_categories,
    )
