"""Python script to create the variable json for lwtnn input for DL1."""

import argparse
import json

from umami.configuration import logger
from umami.preprocessing_tools import GetVariableDict


def GetParser():
    """
    Argparse option for create_vardict script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="""Options for DL1
                                     create_vardict"""
    )

    parser.add_argument(
        "-s",
        "--scale_dict",
        required=True,
        type=str,
        help="""scale_dict file containing scaling and shifting
                        values.""",
    )
    parser.add_argument(
        "-v",
        "--var_dict",
        required=True,
        type=str,
        help="""Dictionary (json) with training variables.""",
    )
    parser.add_argument("-o", "--output", type=str, required=True)
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
        default="tracks_ip3d_sd0sort",
        help="Track selection name.",
    )
    parser.add_argument(
        "--tracks_name",
        type=str,
        default="tracks",
        help="Tracks dataset name in .h5 training/testing files.",
    )

    return parser.parse_args()


def GetTrackVariables(scale_dict: dict, variable_config: dict, tracks_name: str):
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
    noNormVars = variable_config["track_train_variables"][tracks_name]["noNormVars"]
    logNormVars = variable_config["track_train_variables"][tracks_name]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"][tracks_name][
        "jointNormVars"
    ]

    track_dict = scale_dict[tracks_name]
    track_variables = []
    for elem in noNormVars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = 0.0
        v_dict["scale"] = 1.0
        track_variables.append(v_dict)

    for elem in logNormVars:
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

    for elem in jointNormVars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = -1.0 * track_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / track_dict[elem]["scale"]
        track_variables.append(v_dict)
    return track_variables


def GetJetVariables(scale_dict: dict, variable_config: dict):
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
    jetVars = [
        item
        for sublist in variable_config["train_variables"].values()
        for item in sublist
    ]
    jet_variables = []
    jet_dict = {}

    for elem in scale_dict["jets"]:
        jet_dict[elem["name"]] = elem

    for elem in jetVars:
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

    return jet_variables


def __run():
    """main part of script generating json file"""
    args = GetParser()
    variable_config = GetVariableDict(args.var_dict)

    if "dips" in args.tagger.lower():
        logger.info("Starting processing DIPS variables.")
        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)

        track_variables = GetTrackVariables(
            scale_dict, variable_config, args.tracks_name
        )

        logger.info(f"Found {len(track_variables)} variables")
        inputs = {}
        # inputs["name"] = "b-tagging" # only for DL1
        inputs["name"] = args.sequence_name
        inputs["variables"] = track_variables
        inputs = [inputs]

        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = inputs
        lwtnn_var_dict["inputs"] = []
        # lwtnn_var_dict["input_sequences"] = []
        # lwtnn_var_dict["inputs"] = inputs
        lwtnn_var_dict["outputs"] = [
            {"labels": ["pu", "pc", "pb"], "name": args.tagger}
        ]

        logger.info(f"Saving {args.output}.")
        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)

    elif "dl1" in args.tagger.lower():
        logger.info("Starting processing DL1* variables.")

        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)

        jet_variables = GetJetVariables(scale_dict, variable_config)

        logger.info(f"Found {len(jet_variables)} jet variables")
        jet_inputs = {}
        jet_inputs["name"] = "b-tagging"
        jet_inputs["variables"] = jet_variables
        jet_inputs = [jet_inputs]

        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = []
        lwtnn_var_dict["inputs"] = jet_inputs

        if "tau" in args.tagger:
            logger.info("Detected tau output in tagger.")
            labels_tau = ["pu", "pc", "pb", "ptau"]
            logger.info(f"Using labels {labels_tau}")
            lwtnn_var_dict["outputs"] = [{"labels": labels_tau, "name": args.tagger}]
        else:
            lwtnn_var_dict["outputs"] = [
                {"labels": ["pu", "pc", "pb"], "name": args.tagger}
            ]

        logger.info(f"Saving {args.output}.")
        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)

    elif "umami" in args.tagger.lower():
        logger.info("Starting processing UMAMI variables.")

        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)

        jet_variables = GetJetVariables(scale_dict, variable_config)
        track_variables = GetTrackVariables(
            scale_dict, variable_config, args.tracks_name
        )

        logger.info(f"Found {len(track_variables)} track variables")
        logger.info(f"Found {len(jet_variables)} jet variables")

        track_inputs = {}
        track_inputs["name"] = args.sequence_name
        track_inputs["variables"] = track_variables
        track_inputs = [track_inputs]

        jet_inputs = {}
        jet_inputs["name"] = "b-tagging"
        jet_inputs["variables"] = jet_variables
        jet_inputs = [jet_inputs]

        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = track_inputs
        lwtnn_var_dict["inputs"] = jet_inputs
        lwtnn_var_dict["outputs"] = [
            {"labels": ["pu", "pc", "pb"], "name": f"dips_{args.tagger}"},
            {"labels": ["pu", "pc", "pb"], "name": args.tagger},
        ]

        logger.info(f"Saving {args.output}.")
        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)


if __name__ == "__main__":
    __run()
