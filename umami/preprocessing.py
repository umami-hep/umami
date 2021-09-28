import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import yaml
from numpy.lib.recfunctions import repack_fields, structured_to_unstructured

import umami.preprocessing_tools as upt
from umami.configuration import logger
from umami.preprocessing_tools import (
    ProbabilityRatioUnderSampling,
    UnderSampling,
)
from umami.tools import yaml_loader


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
        "-v",
        "--var_dict",
        default=None,
        help="Dictionary with input variables of tagger.",
        type=str,
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


def GetScaleDict(args, config):
    """
    Calculates the scaling, shifting and default values and saves them to json.
    The calculation is done only on the first iteration.
    """
    # TODO: find good way to get file names, breaks if no iterations

    # check if var_dict is provided, otherwise exit
    if not args.var_dict:
        logger.error(
            "Provide --var_dict to retrieve scaling and shifting factors"
        )
        sys.exit(1)

    input_file = config.GetFileName(iteration=1, option="downsampled")
    logger.info(input_file)
    infile_all = h5py.File(input_file, "r")
    take_taus = config.bool_process_taus

    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    variables_header = variable_config["train_variables"]
    var_list = [i for j in variables_header for i in variables_header[j]]

    bjets = pd.DataFrame(infile_all["bjets"][:][var_list])
    cjets = pd.DataFrame(infile_all["cjets"][:][var_list])
    ujets = pd.DataFrame(infile_all["ujets"][:][var_list])
    if take_taus:
        taujets = pd.DataFrame(infile_all["taujets"][:][var_list])
        X = pd.concat([bjets, cjets, ujets, taujets])
        del taujets
    else:
        X = pd.concat([bjets, cjets, ujets])
    del bjets, cjets, ujets

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("Retrieving scaling and shifting values for the jet variables")

    scale_dict = []
    for var in X.columns.values:
        if var in [variable_config["label"], "weight", "category"]:
            continue
        elif "isDefaults" in var:
            # no scaling and shifting is applied to the check variables
            scale_dict.append(upt.dict_in(var, 0.0, 1.0, None))
        else:
            dict_entry = upt.GetScales(
                vec=X[var].values,
                # TODO: implement weights
                w=np.ones(len(X)),
                varname=var,
                custom_defaults_vars=variable_config["custom_defaults_vars"],
            )
            scale_dict.append(upt.dict_in(*dict_entry))

    scale_dict_trk = {}
    if args.tracks:
        logger.info(
            "Retrieving scaling and shifting values for the track variables"
        )
        logNormVars = variable_config["track_train_variables"]["logNormVars"]
        jointNormVars = variable_config["track_train_variables"][
            "jointNormVars"
        ]
        trkVars = logNormVars + jointNormVars

        btrks = np.asarray(infile_all["btrk"][:])
        ctrks = np.asarray(infile_all["ctrk"][:])
        utrks = np.asarray(infile_all["utrk"][:])
        if take_taus:
            tautrks = np.asarray(infile_all["tautrk"][:])
            trks = np.concatenate((tautrks, utrks, ctrks, btrks))
        else:
            trks = np.concatenate((utrks, ctrks, btrks))

        X_trk_train = np.stack(
            [np.nan_to_num(trks[v]) for v in trkVars], axis=-1
        )

        mask = ~np.all(X_trk_train == 0, axis=-1)

        eps = 1e-8

        # Take the log of the desired variables
        for i, v in enumerate(logNormVars):
            X_trk_train[:, :, i][mask] = np.log(
                X_trk_train[:, :, i][mask] + eps
            )

        scale_dict_trk = upt.ScaleTracks(
            X_trk_train[:, :, :], logNormVars + jointNormVars
        )

    # save scale/shift dictionary to json file
    scale_dict = {"jets": scale_dict, "tracks": scale_dict_trk}
    os.makedirs(os.path.dirname(config.dict_file), exist_ok=True)
    with open(config.dict_file, "w") as outfile:
        json.dump(scale_dict, outfile, indent=4)
    logger.info(f"saved scale dictionary as {config.dict_file}")


def ApplyScalesTrksNumpy(args, config, iteration=1):
    if not args.var_dict:
        logger.error(
            "Provide --var_dict to apply scaling and shifting factors"
        )
        sys.exit(1)
    logger.info("Track scaling")
    input_file = config.GetFileName(iteration=iteration, option="downsampled")
    take_taus = config.bool_process_taus
    logger.info(input_file)
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trkVars = noNormVars + logNormVars + jointNormVars

    dsets = [h5py.File(input_file, "r")["/btrk"][:]]
    dsets.append(h5py.File(input_file, "r")["/ctrk"][:])
    dsets.append(h5py.File(input_file, "r")["/utrk"][:])
    if take_taus:
        dsets.append(h5py.File(input_file, "r")["/tautrk"][:])
    arrays = [np.asarray(dset) for dset in dsets]
    logger.info("concatenate all datasets")
    trks = np.concatenate(arrays, axis=0)
    logger.info("concatenated")

    with open(config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["tracks"]

    var_arr_list = []
    trk_mask = ~np.isnan(trks["ptfrac"])
    for var in trkVars:
        if var in logNormVars:
            x = np.log(trks[var])
        else:
            x = trks[var]
        if var in logNormVars:
            x -= scale_dict[var]["shift"]
            x /= scale_dict[var]["scale"]
        elif var in jointNormVars:
            x = np.where(trk_mask, x - scale_dict[var]["shift"], x)
            x = np.where(trk_mask, x / scale_dict[var]["scale"], x)
        var_arr_list.append(np.nan_to_num(x))

    d_arr = np.stack(var_arr_list, axis=-1)
    out_file = config.GetFileName(option="preprocessed", iteration=iteration)
    logger.info(f"saving file: {out_file}")
    with h5py.File(out_file, "a") as h5file:
        h5file.create_dataset("trks", data=d_arr, compression="gzip")
        # TODO: Add plotting


def ApplyScalesNumpy(args, config, iteration=1):
    """
    Apply the scaling and shifting to dataset using numpy
    """
    if not args.var_dict:
        logger.error(
            "Provide --var_dict to apply scaling and shifting factors"
        )
        sys.exit(1)

    input_file = config.GetFileName(iteration=iteration, option="downsampled")
    take_taus = config.bool_process_taus

    if take_taus:
        jets = pd.DataFrame(
            np.concatenate(
                [
                    h5py.File(input_file, "r")["/bjets"][:],
                    h5py.File(input_file, "r")["/cjets"][:],
                    h5py.File(input_file, "r")["/ujets"][:],
                    h5py.File(input_file, "r")["/taujets"][:],
                ]
            )
        )
    else:
        jets = pd.DataFrame(
            np.concatenate(
                [
                    h5py.File(input_file, "r")["/bjets"][:],
                    h5py.File(input_file, "r")["/cjets"][:],
                    h5py.File(input_file, "r")["/ujets"][:],
                ]
            )
        )
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    variables_header = variable_config["train_variables"]
    variables = [i for j in variables_header for i in variables_header[j]]
    variables += variable_config["spectator_variables"][:]
    variables += [variable_config["label"], "weight", "category"]
    if "weight" not in jets.columns.values:
        jets["weight"] = np.ones(len(jets))
    jets = jets[variables]
    jets = jets.replace([np.inf, -np.inf], np.nan)
    with open(config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["jets"]
    logger.info("Replacing default values.")
    default_dict = upt.Gen_default_dict(scale_dict)
    jets = jets.fillna(default_dict)
    # var_list = variable_config["train_variables"]
    logger.info("Applying scaling and shifting.")
    for elem in scale_dict:
        if "isDefaults" in elem["name"]:
            continue
        else:
            jets[elem["name"]] -= elem["shift"]
            jets[elem["name"]] /= elem["scale"]

    out_file = config.GetFileName(option="preprocessed", iteration=iteration)
    logger.info(f"Saving file: {out_file}")
    with h5py.File(out_file, "w") as h5file:
        h5file.create_dataset(
            "jets", data=jets.to_records(index=False), compression="gzip"
        )
    # TODO: Add plotting


def ApplyScales(args, config):
    for iteration in range(1, config.iterations + 1):
        ApplyScalesNumpy(args, config, iteration)
        if args.tracks:
            ApplyScalesTrksNumpy(args, config, iteration)


def WriteTrainSample(args, config):
    if not args.var_dict:
        logger.error("Please provide --var_dict to write training samples")
        sys.exit(1)
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    input_files = [
        config.GetFileName(option="preprocessed", iteration=it)
        for it in range(1, config.iterations + 1)
    ]

    size, ranges = upt.get_size(input_files)
    out_file = config.GetFileName(option="preprocessed_shuffled")

    variables_header = variable_config["train_variables"]
    variables = [i for j in variables_header for i in variables_header[j]]

    logger.info(f"Saving sample to {out_file}")
    with h5py.File(out_file, "w") as output:
        for i, file in enumerate(input_files):
            logger.info(f"Start processing file {i + 1} of {len(input_files)}")
            with h5py.File(file, "r") as in_file:
                jets = in_file["/jets"][:]
                labels = upt.GetBinaryLabels(jets[variable_config["label"]])
                np.random.seed(42)
                np.random.shuffle(labels)

                weights = jets["weight"]
                np.random.seed(42)
                np.random.shuffle(weights)

                jets = repack_fields(jets[variables])
                jets = structured_to_unstructured(jets)
                np.random.seed(42)
                np.random.shuffle(jets)

                if i == 0:
                    source = {
                        "X_train": jets[:1],
                        "Y_train": labels[:1],
                        "weight": weights[:1],
                    }
                    if args.tracks:
                        source["X_trk_train"] = in_file["/trks"][:1]
                    upt.create_datasets(output, source, size)

                output["X_train"][ranges[file][0] : ranges[file][1]] = jets[:]
                output["Y_train"][ranges[file][0] : ranges[file][1]] = labels[
                    :
                ]
                output["weight"][ranges[file][0] : ranges[file][1]] = weights[
                    :
                ]

                if args.tracks:
                    logger.info("adding tracks")
                    trks = in_file["/trks"][:]
                    np.random.seed(42)
                    np.random.shuffle(trks)
                    output["X_trk_train"][
                        ranges[file][0] : ranges[file][1]
                    ] = trks


if __name__ == "__main__":
    args = GetParser()
    config = upt.Configuration(args.config_file)

    if args.prepare:
        preparation_tool = upt.PrepareSamples(args, config)
        preparation_tool.Run()
    if args.resampling:
        if config.sampling["method"] == "count":
            us = UnderSampling(config)
            us.Run()
        # here the other options such as PDFSampling etc. would be called
        if config.sampling["method"] == "probability_ratio":
            ust = ProbabilityRatioUnderSampling(config)
            ust.Run()
    if args.scaling:
        GetScaleDict(args, config)
    if args.apply_scales:
        ApplyScales(args, config)
    if args.write:
        WriteTrainSample(args, config)
