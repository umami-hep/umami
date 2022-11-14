"""Python script to create the variable json for lwtnn input for DL1."""

import argparse
import json

from umami.configuration import logger
from umami.preprocessing_tools import get_variable_dict


def get_parser():
    """
    Argparse option for create_vardict script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="""Options for DL1 create_vardict""",
    )

    parser.add_argument(
        "-s",
        "--scale_dict",
        required=True,
        type=str,
        help="""scale_dict file containing scaling and shifting values.""",
    )
    parser.add_argument(
        "-v",
        "--var_dict",
        required=True,
        type=str,
        help="""Dictionary (json) with training variables.""",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="lwtnn_vars.json",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--tagger",
        type=str,
        required=True,
        help="tagger shortcut, will be variable in ntuples",
    )
    parser.add_argument(
        "-n",
        "--sequence_name",
        type=str,
        help="Track selection name.",
    )
    parser.add_argument(
        "--tracks_name",
        type=str,
        help="Tracks dataset name in .h5 training/testing files.",
    )

    return parser.parse_args()


def get_trk_variables(
    scale_dict: dict,
    variable_config: dict,
    tracks_name: str,
) -> list:
    """Retrieve track variable scaling factors.

    Parameters
    ----------
    scale_dict : dict
        scale configuration
    variable_config : dict
        variable config
    tracks_name : str
        name of the track collection

    Returns
    -------
    list
        list with track variables and scaling factors

    Raises
    ------
    ValueError
        if variable associated to logNormVars but not defined for it.
    """

    # Get track variables from variable config
    no_norm_vars = variable_config["track_train_variables"][tracks_name]["noNormVars"]
    log_norm_vars = variable_config["track_train_variables"][tracks_name]["logNormVars"]
    joint_norm_vars = variable_config["track_train_variables"][tracks_name][
        "jointNormVars"
    ]

    # Select correct track collection from scale dict
    track_dict = scale_dict[tracks_name]
    track_variables = []

    # Iterate over the not-normalised variables
    for elem in no_norm_vars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = 0.0
        v_dict["scale"] = 1.0
        track_variables.append(v_dict)

    # Iterate over the log-normalised variables
    for elem in log_norm_vars:
        v_dict = {}
        if elem == "ptfrac":
            v_dict["name"] = "log_ptfrac"
        elif elem == "dr":
            v_dict["name"] = "log_dr_nansafe"
        elif elem == "z0RelativeToBeamspotUncertainty":
            v_dict["name"] = "log_z0RelativeToBeamspotUncertainty"
        else:
            raise ValueError(f"{elem} not known in logNormVars. Please check.")
        v_dict["offset"] = -1.0 * track_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / track_dict[elem]["scale"]
        track_variables.append(v_dict)

    # Iterate over the joint-normalised variables
    for elem in joint_norm_vars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = -1.0 * track_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / track_dict[elem]["scale"]
        track_variables.append(v_dict)

    # Return the track variables ready for the json.
    return track_variables


def get_jet_variables(
    scale_dict: dict,
    variable_config: dict,
) -> list:
    """Retrieve jet variable scaling factors.

    Parameters
    ----------
    scale_dict : dict
        scale configuration
    variable_config : dict
        variable config

    Returns
    -------
    list
        list with jet variables and scaling factors
    """

    # Get the training jet variables
    jet_vars = [
        item
        for sublist in variable_config["train_variables"].values()
        for item in sublist
    ]
    jet_variables = []
    jet_dict = scale_dict["jets"]

    # Process the jet variables and add them to the list
    for elem in jet_vars:
        v_dict = {}
        if jet_dict[elem]["default"] is not None:
            v_dict["default"] = jet_dict[elem]["default"]
        if jet_dict[elem]["name"] == "absEta_btagJes":
            jet_dict[elem]["name"] = "abs_eta"
        if jet_dict[elem]["name"] == "pt_btagJes":
            jet_dict[elem]["name"] = "pt"
        v_dict["name"] = jet_dict[elem]["name"]
        v_dict["offset"] = -1.0 * jet_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / jet_dict[elem]["scale"]
        jet_variables.append(v_dict)

    # Return the jet variables ready for the json.
    return jet_variables


def __run():
    """main part of script generating json file"""
    args = get_parser()
    variable_config = get_variable_dict(args.var_dict)

    if "dips" in args.tagger.lower():
        logger.info("Starting processing DIPS variables.")

        # Load the given scale dict
        with open(args.scale_dict, "r") as f_scale:
            scale_dict = json.load(f_scale)

        # Get the track variables with scales ready for json
        track_variables = get_trk_variables(
            scale_dict,
            variable_config,
            args.tracks_name,
        )

        logger.info("Found %i variables", len(track_variables))
        inputs = {}

        # Set the name of the track collection (for athena)
        inputs["name"] = args.sequence_name

        # Get the track variables
        inputs["variables"] = track_variables

        # Correctly shape the track variables
        inputs = [inputs]

        # Create the lwtnn variable dict
        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = inputs
        lwtnn_var_dict["inputs"] = []

        # Set the output
        lwtnn_var_dict["outputs"] = [
            {
                "labels": ["pu", "pc", "pb"],
                "name": args.tagger,
            }
        ]

        logger.info("Saving %s.", args.output)

        # Save the lwtnn variable dict in json in correct format
        with open(args.output, "w") as dips_vars:
            json.dump(lwtnn_var_dict, dips_vars, indent=4)

    elif "dl1" in args.tagger.lower():
        logger.info("Starting processing DL1* variables.")

        # Load the given scale dict
        with open(args.scale_dict, "r") as f_scale:
            scale_dict = json.load(f_scale)

        # Get the jet variables with scales ready for json
        jet_variables = get_jet_variables(
            scale_dict,
            variable_config,
        )

        logger.info("Found %i jet variables", len(jet_variables))

        # Define a dict for the jet inputs
        jet_inputs = {}

        # Name the jet inputs correctly
        jet_inputs["name"] = "b-tagging"

        # Save the variables in the dict
        jet_inputs["variables"] = jet_variables
        jet_inputs = [jet_inputs]

        # Create the lwtnn variable dict
        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = []
        lwtnn_var_dict["inputs"] = jet_inputs

        # Set the output
        if "tau" in args.tagger:
            logger.info("Detected tau output in tagger.")
            labels_tau = ["pu", "pc", "pb", "ptau"]
            logger.info("Using labels %s", labels_tau)
            lwtnn_var_dict["outputs"] = [
                {
                    "labels": labels_tau,
                    "name": args.tagger,
                }
            ]

        else:
            lwtnn_var_dict["outputs"] = [
                {
                    "labels": ["pu", "pc", "pb"],
                    "name": args.tagger,
                }
            ]

        logger.info("Saving %s.", args.output)

        # Save the lwtnn variable dict in json in correct format
        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)

    elif "umami" in args.tagger.lower():
        logger.info("Starting processing UMAMI variables.")

        # Load the given scale dict
        with open(args.scale_dict, "r") as f_scale:
            scale_dict = json.load(f_scale)

        # Get the jet variables with scales ready for json
        jet_variables = get_jet_variables(
            scale_dict,
            variable_config,
        )

        # Get the track variables with scales ready for json
        track_variables = get_trk_variables(
            scale_dict,
            variable_config,
            args.tracks_name,
        )

        logger.info("Found %i track variables", len(track_variables))
        logger.info("Found %i jet variables", len(jet_variables))

        # Init track input dict
        track_inputs = {}

        # Set the name of the track collection (for athena)
        track_inputs["name"] = args.sequence_name

        # Get the track variables
        track_inputs["variables"] = track_variables

        # Correctly shape the track variables
        track_inputs = [track_inputs]

        # Init jet inputs dict with correct naming
        jet_inputs = {}
        jet_inputs["name"] = "b-tagging"

        # Get the jet variables
        jet_inputs["variables"] = jet_variables

        # Correctly shape the jet variables
        jet_inputs = [jet_inputs]

        # Create the lwtnn variable dict
        lwtnn_var_dict = {}

        # Set the track and jet inputs
        lwtnn_var_dict["input_sequences"] = track_inputs
        lwtnn_var_dict["inputs"] = jet_inputs

        # Set the output
        lwtnn_var_dict["outputs"] = [
            {
                "labels": ["pu", "pc", "pb"],
                "name": f"dips_{args.tagger}",
            },
            {
                "labels": ["pu", "pc", "pb"],
                "name": args.tagger,
            },
        ]

        logger.info("Saving %s.", args.output)

        # Save the lwtnn variable dict in json in correct format
        with open(args.output, "w") as umami_vars:
            json.dump(lwtnn_var_dict, umami_vars, indent=4)


if __name__ == "__main__":
    __run()
