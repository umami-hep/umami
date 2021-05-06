"""Python script to create the variable json for lwtnn input for DL1."""

import argparse
import json

import yaml

from umami.tools import yaml_loader


def GetParser():
    """Argparse option for create_vardict script."""
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

    return parser.parse_args()


def GetTrackVariables(scale_dict, variable_config):
    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]

    track_dict = scale_dict["tracks"]
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
        else:
            print("SOMETHING IS WRONG")
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


def GetJetVariables(scale_dict, variable_config):
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
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    if "dips" in args.tagger.lower():

        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)["tracks"]

        track_variables = GetJetVariables(scale_dict, variable_config)

        print("Found %i variables" % len(track_variables))
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

        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)

    elif "dl1" in args.tagger.lower():

        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)

        jet_variables = GetJetVariables(scale_dict, variable_config)

        print(f"Found {len(jet_variables)} jet variables")
        jet_inputs = {}
        jet_inputs["name"] = "b-tagging"
        jet_inputs["variables"] = jet_variables
        jet_inputs = [jet_inputs]

        lwtnn_var_dict = {}
        lwtnn_var_dict["input_sequences"] = []
        lwtnn_var_dict["inputs"] = jet_inputs

        lwtnn_var_dict["outputs"] = [
            {"labels": ["pu", "pc", "pb"], "name": args.tagger}
        ]

        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)

    elif "umami" in args.tagger.lower():

        with open(args.scale_dict, "r") as f:
            scale_dict = json.load(f)

        jet_variables = GetJetVariables(scale_dict, variable_config)
        track_variables = GetTrackVariables(scale_dict, variable_config)

        print(f"Found {len(track_variables)} track variables")
        print(f"Found {len(jet_variables)} jet variables")

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

        with open(args.output, "w") as dl1_vars:
            json.dump(lwtnn_var_dict, dl1_vars, indent=4)


if __name__ == "__main__":
    __run()
