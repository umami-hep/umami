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


def __run():
    """main part of script generating json file"""
    args = GetParser()
    with open(args.scale_dict, "r") as f:
        scale_dict = json.load(f)["tracks"]
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]

    variables = []
    # for elem in scale_dict:
    #     if elem["name"] not in var_dict:
    #         continue
    #     v_dict = {}
    #     if elem['default'] is not None:
    #         v_dict['default'] = elem['default']
    #     if elem["name"] == "abs_eta_uncalib":
    #         elem["name"] = "abs_eta"
    #     if elem["name"] == "pt_uncalib":
    #         elem["name"] = "pt"
    #     v_dict["name"] = elem["name"]
    #     v_dict["offset"] = -1. * elem["shift"]
    #     v_dict["scale"] = 1. / elem["scale"]
    #     variables.append(v_dict)
    for elem in noNormVars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = 0.0
        v_dict["scale"] = 1.0
        variables.append(v_dict)

    for elem in logNormVars:
        v_dict = {}
        if elem == "ptfrac":
            v_dict["name"] = "log_ptfrac"
        elif elem == "dr":
            v_dict["name"] = "log_dr_nansafe"
        else:
            print("SOMETHING IS WRONG")
        v_dict["offset"] = -1.0 * scale_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / scale_dict[elem]["scale"]
        variables.append(v_dict)

    for elem in jointNormVars:
        v_dict = {}
        v_dict["name"] = elem
        v_dict["offset"] = -1.0 * scale_dict[elem]["shift"]
        v_dict["scale"] = 1.0 / scale_dict[elem]["scale"]
        variables.append(v_dict)

    print("Found %i variables" % len(variables))
    inputs = {}
    # inputs["name"] = "b-tagging" # only for DL1
    inputs["name"] = args.sequence_name
    inputs["variables"] = variables
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


if __name__ == "__main__":
    __run()
