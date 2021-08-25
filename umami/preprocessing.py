import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
import yaml
from numpy.lib.recfunctions import (
    append_fields,
    repack_fields,
    structured_to_unstructured,
)

import umami.preprocessing_tools as upt
from umami.configuration import global_config, logger
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
        "--undersampling",
        action="store_true",
        help="Runs undersampling.",
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


def RunUndersampling(args, config):
    """
    Applies required cuts to the samples and applies the downsampling.
    The downsampling in this case takes as many jets of each flavour
    per pT and eta bin.

    Can optionally run on taus (taujets, PID = 15) if the configuration,
    contains:
        bool_process_taus: True
    Undersampling method is based on sampling_method value in config:
    - 'sampling_method = count'  same number of flavour per bin
    - 'sampling_method = weight' same distribution of total per bin
                                 (e.g., 2% of b, 2% of l, ... in bin x)
    - 'sampling_method = count_tau_weight' for count of flavour of b, l,
                                 and c but weight for taus. All fixed to
                                 same density distribution per bin.
    - 'sampling_method = template_b' uses the b distribution as the template
                                 to make c, l and tau distributions b-shaped.
    WARNING: count sampling with upt.UnderSampling is not advised for tau,
    given their small fraction in the data.
    """

    N_list = upt.GetNJetsPerIteration(config)
    # TODO: switch to dask

    take_taus = config.bool_process_taus
    extended_labelling = config.bool_extended_labelling
    if config.sampling_method == "count":
        sampling_method = "count"
    elif config.sampling_method == "weight":
        sampling_method = "weight"
    elif config.sampling_method == "count_tau_weight" and take_taus:
        # if no taus, is equivalent to using count
        logger.info(
            "Undersampling based on weights, but then equalise counts of b, c, and l"
        )
        sampling_method = "count_bcl_weight_tau"
    elif config.sampling_method == "template_b":
        sampling_method = "template_b"
    elif config.sampling_method == "template_b_count":
        sampling_method = "template_b_count"
    else:
        logger.info("Unspecified sampling method, default is count")
        sampling_method = "count"

    # initialise input files (they are not yet loaded to memory)
    f_Z = h5py.File(config.f_z, "r")
    f_tt_bjets = h5py.File(config.f_tt_bjets, "r")
    f_tt_cjets = h5py.File(config.f_tt_cjets, "r")
    f_tt_ujets = h5py.File(config.f_tt_ujets, "r")

    if take_taus:
        f_tt_taujets = h5py.File(config.f_tt_taujets, "r")
        N_list = upt.GetNJetsPerIteration(
            config, f_tt_taujets["jets"].shape[0]
        )

    for x in range(config.iterations):
        logger.info(f"Iteration {x + 1} of {config.iterations}")
        vec_Z = f_Z["jets"][N_list[x]["nZ"] : N_list[x + 1]["nZ"]]
        vec_Z = append_fields(
            vec_Z,
            "category",
            np.zeros(len(vec_Z)),
            dtypes="<f4",
            asrecarray=True,
        )
        vec_tt_bjets = f_tt_bjets["jets"][
            N_list[x]["nbjets"] : N_list[x + 1]["nbjets"]
        ]
        vec_tt_bjets = append_fields(
            vec_tt_bjets,
            "category",
            np.ones(len(vec_tt_bjets)),
            dtypes="<f4",
            asrecarray=True,
        )
        vec_tt_cjets = f_tt_cjets["jets"][
            N_list[x]["ncjets"] : N_list[x + 1]["ncjets"]
        ]
        vec_tt_cjets = append_fields(
            vec_tt_cjets,
            "category",
            np.ones(len(vec_tt_cjets)),
            dtypes="<f4",
            asrecarray=True,
        )
        vec_tt_ujets = f_tt_ujets["jets"][
            N_list[x]["nujets"] : N_list[x + 1]["nujets"]
        ]
        vec_tt_ujets = append_fields(
            vec_tt_ujets,
            "category",
            np.ones(len(vec_tt_ujets)),
            dtypes="<f4",
            asrecarray=True,
        )
        if take_taus:
            vec_tt_taujets = f_tt_taujets["jets"][
                N_list[x]["ntaujets"] : N_list[x + 1]["ntaujets"]
            ]
            vec_tt_taujets = append_fields(
                vec_tt_taujets,
                "category",
                np.ones(len(vec_tt_taujets)),
                dtypes="<f4",
                asrecarray=True,
            )
        else:
            vec_tt_taujets = None

        if args.tracks:
            tnp_Zprime = np.asarray(
                f_Z["tracks"][N_list[x]["nZ"] : N_list[x + 1]["nZ"]]
            )
            tnp_tt_b = np.asarray(
                f_tt_bjets["tracks"][
                    N_list[x]["nbjets"] : N_list[x + 1]["nbjets"]
                ]
            )
            tnp_tt_c = np.asarray(
                f_tt_cjets["tracks"][
                    N_list[x]["ncjets"] : N_list[x + 1]["ncjets"]
                ]
            )
            tnp_tt_u = np.asarray(
                f_tt_ujets["tracks"][
                    N_list[x]["nujets"] : N_list[x + 1]["nujets"]
                ]
            )
            if take_taus:
                tnp_tt_tau = np.asarray(
                    f_tt_taujets["tracks"][
                        N_list[x]["ntaujets"] : N_list[x + 1]["ntaujets"]
                    ]
                )

        logger.info("starting pruning")
        # Print some statistics on the sample formed
        statistics_dict = upt.RunStatSamples(
            vec_tt_bjets, vec_tt_cjets, vec_tt_ujets, vec_tt_taujets
        )
        indices_toremove_Zprime = upt.GetCuts(
            vec_Z, config, "Zprime", extended_labelling
        )
        indices_toremove_bjets = upt.GetCuts(
            vec_tt_bjets, config, "ttbar", extended_labelling
        )
        indices_toremove_cjets = upt.GetCuts(
            vec_tt_cjets, config, "ttbar", extended_labelling
        )
        indices_toremove_ujets = upt.GetCuts(
            vec_tt_ujets, config, "ttbar", extended_labelling
        )
        if take_taus:
            indices_toremove_taujets = upt.GetCuts(vec_tt_taujets, config)

        vec_Z = np.delete(vec_Z, indices_toremove_Zprime, 0)
        vec_tt_bjets = np.delete(vec_tt_bjets, indices_toremove_bjets, 0)
        vec_tt_cjets = np.delete(vec_tt_cjets, indices_toremove_cjets, 0)
        vec_tt_ujets = np.delete(vec_tt_ujets, indices_toremove_ujets, 0)
        if take_taus:
            vec_tt_taujets = np.delete(
                vec_tt_taujets, indices_toremove_taujets, 0
            )

        if args.tracks:
            tnp_Zprime = np.delete(tnp_Zprime, indices_toremove_Zprime, 0)
            tnp_tt_b = np.delete(tnp_tt_b, indices_toremove_bjets, 0)
            tnp_tt_c = np.delete(tnp_tt_c, indices_toremove_cjets, 0)
            tnp_tt_u = np.delete(tnp_tt_u, indices_toremove_ujets, 0)
            if take_taus:
                tnp_tt_tau = np.delete(tnp_tt_tau, indices_toremove_taujets, 0)

        if extended_labelling:
            bjets = np.concatenate(
                [
                    vec_Z[
                        vec_Z["HadronConeExclExtendedTruthLabelID"]
                        == 5 | vec_Z["HadronConeExclExtendedTruthLabelID"]
                        == 54
                    ],
                    vec_tt_bjets,
                ]
            )
            cjets = np.concatenate(
                [
                    vec_Z[
                        vec_Z["HadronConeExclExtendedTruthLabelID"]
                        == 4 | vec_Z["HadronConeExclExtendedTruthLabelID"]
                        == 44
                    ],
                    vec_tt_cjets,
                ]
            )
            ujets = np.concatenate(
                [
                    vec_Z[vec_Z["HadronConeExclExtendedTruthLabelID"] == 0],
                    vec_tt_ujets,
                ]
            )
        else:
            bjets = np.concatenate(
                [vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 5], vec_tt_bjets]
            )
            cjets = np.concatenate(
                [vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 4], vec_tt_cjets]
            )
            ujets = np.concatenate(
                [vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 0], vec_tt_ujets]
            )
        if take_taus:
            if extended_labelling:
                taujets = np.concatenate(
                    [
                        vec_Z[
                            vec_Z["HadronConeExclExtendedTruthLabelID"] == 15
                        ],
                        vec_tt_taujets,
                    ]
                )
            else:
                taujets = np.concatenate(
                    [
                        vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 15],
                        vec_tt_taujets,
                    ]
                )
        else:
            taujets = None

        # New
        if args.tracks:
            if extended_labelling:
                btrk = np.concatenate(
                    [
                        tnp_Zprime[
                            vec_Z["HadronConeExclExtendedTruthLabelID"]
                            == 5 | vec_Z["HadronConeExclExtendedTruthLabelID"]
                            == 54
                        ],
                        tnp_tt_b,
                    ]
                )
                ctrk = np.concatenate(
                    [
                        tnp_Zprime[
                            vec_Z["HadronConeExclExtendedTruthLabelID"]
                            == 4 | vec_Z["HadronConeExclExtendedTruthLabelID"]
                            == 44
                        ],
                        tnp_tt_c,
                    ]
                )
                utrk = np.concatenate(
                    [
                        tnp_Zprime[
                            vec_Z["HadronConeExclExtendedTruthLabelID"] == 0
                        ],
                        tnp_tt_u,
                    ]
                )
            else:
                btrk = np.concatenate(
                    [
                        tnp_Zprime[vec_Z["HadronConeExclTruthLabelID"] == 5],
                        tnp_tt_b,
                    ]
                )
                ctrk = np.concatenate(
                    [
                        tnp_Zprime[vec_Z["HadronConeExclTruthLabelID"] == 4],
                        tnp_tt_c,
                    ]
                )
                utrk = np.concatenate(
                    [
                        tnp_Zprime[vec_Z["HadronConeExclTruthLabelID"] == 0],
                        tnp_tt_u,
                    ]
                )
            if take_taus:
                if extended_labelling:
                    tautrk = np.concatenate(
                        [
                            tnp_Zprime[
                                vec_Z["HadronConeExclExtendedTruthLabelID"]
                                == 15
                            ],
                            tnp_tt_tau,
                        ]
                    )
                else:
                    tautrk = np.concatenate(
                        [
                            tnp_Zprime[
                                vec_Z["HadronConeExclTruthLabelID"] == 15
                            ],
                            tnp_tt_tau,
                        ]
                    )
            else:
                tautrk = None
        else:
            btrk, ctrk, utrk, tautrk = None, None, None, None

        # Plots pt and eta before downsampling
        plot_name_clean = config.GetFileName(
            x + 1,
            extension="",
            option="pt_eta-wider_bins",
            custom_path="plots/",
        )
        upt.MakePresentationPlots(
            bjets=bjets,
            ujets=ujets,
            cjets=cjets,
            taujets=taujets,
            plots_path=plot_name_clean,
            binning={
                global_config.pTvariable: 200,
                global_config.etavariable: 20,
            },
            Log=True,
        )

        logger.info("starting undersampling")
        # Print some statistics on the sample formed
        statistics_dict = upt.RunStatSamples(bjets, cjets, ujets, taujets)

        # Do the sampling:
        (
            bjets,
            cjets,
            ujets,
            taujets,
            btrk,
            ctrk,
            utrk,
            tautrk,
            downs,
        ) = upt.RunSampling(
            bjets,
            cjets,
            ujets,
            taujets,
            btrk,
            ctrk,
            utrk,
            tautrk,
            sampling_method,
            take_taus,
            args.tracks,
            pT_max=config.pT_max,
        )
        # Print some statistics on the sample formed
        logger.info("finished undersampling")
        statistics_dict = upt.RunStatSamples(bjets, cjets, ujets, taujets)

        if config.enforce_ttbar_frac:
            # If one wants to enforce the ttbar fraction demanded
            # Normally not required, except if target number of jets is above total available
            bjets, bindices = upt.EnforceFraction(
                bjets, config.ttbar_frac, statistics_dict, label="b"
            )
            cjets, cindices = upt.EnforceFraction(
                cjets, config.ttbar_frac, statistics_dict, label="c"
            )
            ujets, uindices = upt.EnforceFraction(
                ujets, config.ttbar_frac, statistics_dict, label="u"
            )
            if take_taus:
                taujets, tauindices = upt.EnforceFraction(
                    taujets, config.ttbar_frac, statistics_dict, label="tau"
                )
            if args.tracks:
                if bindices is not None:
                    btrk = btrk[bindices]
                if cindices is not None:
                    ctrk = ctrk[cindices]
                if uindices is not None:
                    utrk = utrk[uindices]
                if take_taus and tauindices is not None:
                    tautrk = tautrk[tauindices]
            # Need to re-sample to make sure flavours are still distributed as demanded.
            # Do it separately for ttbar and Z', then concatenate.
            (
                bjets,
                cjets,
                ujets,
                taujets,
                btrk,
                ctrk,
                utrk,
                tautrk,
                _,
            ) = upt.RunSampling(
                bjets,
                cjets,
                ujets,
                taujets,
                btrk,
                ctrk,
                utrk,
                tautrk,
                sampling_method,
                take_taus,
                args.tracks,
            )
            statistics_dict = upt.RunStatSamples(bjets, cjets, ujets, taujets)

        out_file = config.GetFileName(x + 1, option="downsampled")
        # ensure output path exists
        os.makedirs(pathlib.Path(out_file).parent.absolute(), exist_ok=True)
        logger.info(f"saving file: {out_file}")

        h5f = h5py.File(out_file, "w")
        h5f.create_dataset("bjets", data=bjets, compression="gzip")
        h5f.create_dataset("cjets", data=cjets, compression="gzip")
        h5f.create_dataset("ujets", data=ujets, compression="gzip")
        if take_taus:
            h5f.create_dataset("taujets", data=taujets, compression="gzip")
        if args.tracks:
            h5f.create_dataset("btrk", data=btrk, compression="gzip")
            h5f.create_dataset("ctrk", data=ctrk, compression="gzip")
            h5f.create_dataset("utrk", data=utrk, compression="gzip")
            if take_taus:
                h5f.create_dataset("tautrk", data=tautrk, compression="gzip")

        h5f.close()
        # TODO: verify track handling
        logger.info("Plotting ...")
        plot_name = config.GetFileName(
            x + 1,
            option="downsampled-pt_eta",
            extension=".pdf",
            custom_path="plots/",
        )
        upt.MakePlots(
            bjets,
            cjets,
            ujets,
            taujets,
            plot_name=plot_name,
            binning={
                global_config.pTvariable: downs.pt_bins,
                global_config.etavariable: downs.eta_bins,
            },
        )
        plot_name = config.GetFileName(
            x + 1,
            extension=".pdf",
            option="downsampled-pt_eta-wider_bins",
            custom_path="plots/",
        )
        upt.MakePlots(
            bjets,
            cjets,
            ujets,
            taujets,
            plot_name=plot_name,
            binning={
                global_config.pTvariable: 200,
                global_config.etavariable: 20,
            },
        )
        plot_name_clean = config.GetFileName(
            x + 1,
            extension="",
            option="downsampled-pt_eta-wider_bins",
            custom_path="plots/",
        )
        upt.MakePresentationPlots(
            bjets,
            ujets,
            cjets,
            taujets,
            plots_path=plot_name_clean,
            binning={
                global_config.pTvariable: 200,
                global_config.etavariable: 20,
            },
        )


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
    if args.undersampling:
        RunUndersampling(args, config)
    if args.scaling:
        GetScaleDict(args, config)
    if args.apply_scales:
        ApplyScales(args, config)
    if args.write:
        WriteTrainSample(args, config)
