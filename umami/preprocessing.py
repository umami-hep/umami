import umami.preprocessing_tools as upt
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields, structured_to_unstructured
from numpy.lib.recfunctions import repack_fields
import pandas as pd
import argparse
import yaml
from umami.tools import yaml_loader
import json


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Enter the name of the config file to create the"
                        "hybrid sample.")
    parser.add_argument('-t', '--tracks', action='store_true',
                        help="Stores also track information.")
    parser.add_argument('-v', '--var_dict', required=True, default=None,
                        help="Dictionary with input variables of tagger.",
                        type=str)
    # possible job options for the different preprocessing steps
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-u', '--undersampling', action='store_true',
                        help="Runs undersampling.")
    # action.add_argument('--weighting', action='store_true',
    #                     help="Runs weighting.")
    action.add_argument('-s', '--scaling', action='store_true',
                        help="Retrieves scaling and shifting factors.")
    action.add_argument('-a', '--apply_scales', action='store_true',
                        help="Apllies scaling and shifting factors.")
    action.add_argument('-w', '--write', action='store_true',
                        help="Shuffles sample and writes training sample and"
                             "training labels to disk")
    args = parser.parse_args()
    return args


def RunUndersampling(args, config):
    """Applies required cuts to the samples and applies the downsampling."""
    N_list = upt.GetNJetsPerIteration(config)
    # TODO: switch to dask

    # initialise input files (they are not yet loaded to memory)
    f_Z = h5py.File(config.f_z, 'r')
    f_tt_bjets = h5py.File(config.f_tt_bjets, 'r')
    f_tt_cjets = h5py.File(config.f_tt_cjets, 'r')
    f_tt_ujets = h5py.File(config.f_tt_ujets, 'r')

    for x in range(config.iterations):
        print("Iteration", x + 1, "of", config.iterations)
        vec_Z = f_Z['jets'][N_list[x]["nZ"]:N_list[x + 1]["nZ"]]
        vec_Z = append_fields(vec_Z, "category",
                              np.zeros(len(vec_Z)),
                              dtypes='<f4', asrecarray=True)
        vec_tt_bjets = f_tt_bjets['jets'][N_list[x]["nbjets"]:N_list[x + 1]
                                          ["nbjets"]]
        vec_tt_bjets = append_fields(vec_tt_bjets, "category",
                                     np.ones(len(vec_tt_bjets)),
                                     dtypes='<f4', asrecarray=True)
        vec_tt_cjets = f_tt_cjets['jets'][N_list[x]["ncjets"]:N_list[x + 1]
                                          ["ncjets"]]
        vec_tt_cjets = append_fields(vec_tt_cjets, "category",
                                     np.ones(len(vec_tt_cjets)),
                                     dtypes='<f4', asrecarray=True)
        vec_tt_ujets = f_tt_ujets['jets'][N_list[x]["nujets"]:N_list[x + 1]
                                          ["nujets"]]
        vec_tt_ujets = append_fields(vec_tt_ujets, "category",
                                     np.ones(len(vec_tt_ujets)),
                                     dtypes='<f4', asrecarray=True)
        if args.tracks:
            tnp_Zprime = np.asarray(f_Z['tracks'][N_list[x]["nZ"]:N_list[x + 1]
                                                  ["nZ"]])
            tnp_tt_b = np.asarray(f_tt_bjets['tracks'][N_list[x]["nbjets"]:
                                  N_list[x + 1]["nbjets"]])
            tnp_tt_c = np.asarray(f_tt_cjets['tracks'][N_list[x]["ncjets"]:
                                  N_list[x + 1]["ncjets"]])
            tnp_tt_u = np.asarray(f_tt_ujets['tracks'][N_list[x]["nujets"]:
                                  N_list[x + 1]["nujets"]])

        indices_toremove_Zprime = upt.GetCuts(vec_Z, config, 'Zprime')
        indices_toremove_bjets = upt.GetCuts(vec_tt_bjets, config)
        indices_toremove_cjets = upt.GetCuts(vec_tt_cjets, config)
        indices_toremove_ujets = upt.GetCuts(vec_tt_ujets, config)

        vec_Z = np.delete(vec_Z, indices_toremove_Zprime, 0)
        vec_tt_bjets = np.delete(vec_tt_bjets, indices_toremove_bjets, 0)
        vec_tt_cjets = np.delete(vec_tt_cjets, indices_toremove_cjets, 0)
        vec_tt_ujets = np.delete(vec_tt_ujets, indices_toremove_ujets, 0)

        if args.tracks:
            tnp_Zprime = np.delete(tnp_Zprime, indices_toremove_Zprime, 0)
            tnp_tt_b = np.delete(tnp_tt_b, indices_toremove_bjets, 0)
            tnp_tt_c = np.delete(tnp_tt_c, indices_toremove_cjets, 0)
            tnp_tt_u = np.delete(tnp_tt_u, indices_toremove_ujets, 0)

        print("starting undersampling")
        bjets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 5
                                      ], vec_tt_bjets])
        cjets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 4
                                      ], vec_tt_cjets])
        ujets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 0
                                      ], vec_tt_ujets])
        downs = upt.UnderSampling(bjets, cjets, ujets)
        b_indices, c_indices, u_indices = downs.GetIndices()

        bjets = bjets[b_indices]
        cjets = cjets[c_indices]
        ujets = ujets[u_indices]

        if args.tracks:
            btrk = np.concatenate([tnp_Zprime[
                vec_Z["HadronConeExclTruthLabelID"] == 5], tnp_tt_b])[
                    b_indices]
            ctrk = np.concatenate([tnp_Zprime[
                vec_Z["HadronConeExclTruthLabelID"] == 4], tnp_tt_c])[
                    c_indices]
            utrk = np.concatenate([tnp_Zprime[
                vec_Z["HadronConeExclTruthLabelID"] == 0], tnp_tt_u])[
                    u_indices]

        ttfrac = float(len(bjets[bjets["category"] == 1]) + len(
            cjets[cjets["category"] == 1]) + len(
                ujets[ujets["category"] == 1])) / float(len(bjets) + len(
                    cjets) + len(ujets))
        print("ttbar fraction:", round(ttfrac, 2))

        out_file = config.GetFileName(x + 1, option="downsampled")
        print("saving file:", out_file)
        h5f = h5py.File(out_file, 'w')
        h5f.create_dataset('bjets', data=bjets)
        h5f.create_dataset('cjets', data=cjets)
        h5f.create_dataset('ujets', data=ujets)
        if args.tracks:
            h5f.create_dataset('btrk', data=btrk)
            h5f.create_dataset('ctrk', data=ctrk)
            h5f.create_dataset('utrk', data=utrk)

        h5f.close()
        # TODO: verify track handling
        print("Plotting ...")
        plot_name = config.GetFileName(x + 1, option="downsampled-pt_eta",
                                       extension=".pdf", custom_path="plots/")
        upt.MakePlots(bjets, cjets, ujets, plot_name=plot_name,
                      binning={"pt_btagJes":  downs.pt_bins,
                               "absEta_btagJes": downs.eta_bins})
        plot_name = config.GetFileName(x + 1, extension=".pdf",
                                       option="downsampled-pt_eta-wider_bins",
                                       custom_path="plots/")
        upt.MakePlots(bjets, cjets, ujets, plot_name=plot_name,
                      binning={"pt_btagJes":  200,
                               "absEta_btagJes": 20})


def GetScaleDict(args, config):
    """
    Calculates the scaling, shifting and default values and saves them to json.
    The calculation is done only on the first iteration.
    """
    # TODO: find good way to get file names, breaks if no iterations
    input_file = config.GetFileName(iteration=1, option='downsampled')
    print(input_file)
    infile_all = h5py.File(input_file, 'r')

    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    var_list = variable_config["train_variables"][:]

    bjets = pd.DataFrame(infile_all['bjets'][:][var_list])
    cjets = pd.DataFrame(infile_all['cjets'][:][var_list])
    ujets = pd.DataFrame(infile_all['ujets'][:][var_list])
    X = pd.concat([bjets, cjets, ujets])
    del bjets, cjets, ujets

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Retrieving scaling and shifting values for the jet variables")

    scale_dict = []
    for var in X.columns.values:
        if var in [variable_config["label"], 'weight', 'category']:
            continue
        elif 'isDefaults' in var:
            # no scaling and shifting is applied to the check variables
            scale_dict.append(upt.dict_in(var, 0., 1., None))
        else:
            dict_entry = upt.GetScales(
                vec=X[var].values,
                # TODO: implement weights
                w=np.ones(len(X)), varname=var,
                custom_defaults_vars=variable_config["custom_defaults_vars"])
            scale_dict.append(upt.dict_in(*dict_entry))

    scale_dict_trk = {}
    if args.tracks:
        print("Retrieving scaling and shifting values for the track variables")
        logNormVars = variable_config["track_train_variables"]["logNormVars"]
        jointNormVars = variable_config["track_train_variables"][
            "jointNormVars"]
        trkVars = logNormVars + jointNormVars

        btrks = np.asarray(infile_all['btrk'][:])
        ctrks = np.asarray(infile_all['ctrk'][:])
        utrks = np.asarray(infile_all['utrk'][:])

        trks = np.concatenate((utrks, ctrks, btrks))

        X_trk_train = np.stack([np.nan_to_num(trks[v])for v in trkVars],
                               axis=-1)

        mask = ~ np.all(X_trk_train == 0, axis=-1)

        eps = 1e-8

        # Take the log of the desired variables
        for i, v in enumerate(logNormVars):
            X_trk_train[:, :, i][mask] = np.log(X_trk_train[:, :, i][mask] +
                                                eps)

        scale_dict_trk = upt.ScaleTracks(X_trk_train[:, :, :],
                                         logNormVars+jointNormVars)

    # save scale/shift dictionary to json file
    scale_dict = {"jets": scale_dict, "tracks": scale_dict_trk}
    with open(config.dict_file, 'w') as outfile:
        json.dump(scale_dict, outfile, indent=4)
    print("saved scale dictionary as", config.dict_file)


def ApplyScalesTrksNumpy(args, config, iteration=1):
    print("Track scaling")
    input_file = config.GetFileName(iteration=iteration, option='downsampled')
    print(input_file)
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trkVars = noNormVars + logNormVars + jointNormVars

    dsets = [h5py.File(input_file, 'r')['/btrk'][:]]
    dsets.append(h5py.File(input_file, 'r')['/ctrk'][:])
    dsets.append(h5py.File(input_file, 'r')['/utrk'][:])
    arrays = [np.asarray(dset) for dset in dsets]
    print("concatenate all datasets")
    trks = np.concatenate(arrays, axis=0)
    print("concatenated")

    with open(config.dict_file, 'r') as infile:
        scale_dict = json.load(infile)["tracks"]

    var_arr_list = []
    for var in trkVars:
        if var in logNormVars:
            x = np.log(trks[var])
        else:
            x = trks[var]
        if var in logNormVars:
            x -= scale_dict[var]["shift"]
            x /= scale_dict[var]["scale"]
        elif var in jointNormVars:
            x = np.where(x == 0, x, x - scale_dict[var]["shift"])
            x = np.where(x == 0, x, x / scale_dict[var]["scale"])
        var_arr_list.append(np.nan_to_num(x))

    d_arr = np.stack(var_arr_list, axis=-1)
    out_file = config.GetFileName(option='preprocessed', iteration=iteration)
    print("saving file:", out_file)
    with h5py.File(out_file, 'a') as h5file:
        h5file.create_dataset('trks', data=d_arr)


def ApplyScalesNumpy(args, config, iteration=1):
    """
        Apply the scaling and shifting to dataset using numpy
    """
    input_file = config.GetFileName(iteration=iteration, option='downsampled')

    jets = h5py.File(input_file, 'r')['/bjets'][:]
    jets = pd.DataFrame(
        np.concatenate([h5py.File(input_file, 'r')['/bjets'][:],
                        h5py.File(input_file, 'r')['/cjets'][:],
                        h5py.File(input_file, 'r')['/ujets'][:]]))
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    variables = variable_config["train_variables"][:]
    variables += variable_config["spectator_variables"][:]
    variables += [variable_config["label"], 'weight', 'category']
    if 'weight' not in jets.columns.values:
        jets['weight'] = np.ones(len(jets))
    jets = jets[variables]
    jets = jets.replace([np.inf, -np.inf], np.nan)
    with open(config.dict_file, 'r') as infile:
        scale_dict = json.load(infile)['jets']
    print("Replacing default values.")
    default_dict = upt.Gen_default_dict(scale_dict)
    jets = jets.fillna(default_dict)
    # var_list = variable_config["train_variables"]
    print("Applying scaling and shifting.")
    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        else:
            jets[elem['name']] -= elem['shift']
            jets[elem['name']] /= elem['scale']

    out_file = config.GetFileName(option='preprocessed', iteration=iteration)
    print("Saving file:", out_file)
    with h5py.File(out_file, 'w') as h5file:
        h5file.create_dataset('jets', data=jets.to_records(index=False))


def ApplyScales(args, config):
    for iteration in range(1, config.iterations + 1):
        ApplyScalesNumpy(args, config, iteration)
        if args.tracks:
            ApplyScalesTrksNumpy(args, config, iteration)


def WriteTrainSample(args, config):
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    input_files = [config.GetFileName(
        option='preprocessed', iteration=it) for it in range(
            1, config.iterations + 1)]

    size, ranges = upt.get_size(input_files)
    out_file = config.GetFileName(option='preprocessed_shuffled')
    print("Saving sample to", out_file)
    with h5py.File(out_file, 'w') as output:
        for i, file in enumerate(input_files):
            print("Start processing file", i+1, "of", len(input_files))
            with h5py.File(file, 'r') as in_file:
                jets = in_file['/jets'][:]
                labels = upt.GetBinaryLabels(jets[variable_config['label']])
                np.random.seed(42)
                np.random.shuffle(labels)

                weights = jets['weight']
                np.random.seed(42)
                np.random.shuffle(weights)

                jets = repack_fields(jets[variable_config[
                    'train_variables'][:]])
                jets = structured_to_unstructured(jets)
                np.random.seed(42)
                np.random.shuffle(jets)

                if i == 0:
                    source = {"X_train": jets[:1], "Y_train": labels[:1],
                              "weight": weights[:1]}
                    if args.tracks:
                        source["X_trk_train"] = in_file['/trks'][:1]
                    upt.create_datasets(output, source, size)

                output["X_train"][ranges[file][0]:ranges[file][1]] = jets[:]
                output["Y_train"][ranges[file][0]:ranges[file][1]] = labels[:]
                output["weight"][ranges[file][0]:ranges[file][1]] = weights[:]

                if args.tracks:
                    print("adding tracks")
                    trks = in_file['/trks'][:]
                    np.random.seed(42)
                    np.random.shuffle(trks)
                    output["X_trk_train"][ranges[
                        file][0]:ranges[file][1]] = trks


if __name__ == '__main__':
    args = GetParser()
    config = upt.Configuration(args.config_file)

    if args.undersampling:
        RunUndersampling(args, config)
    if args.scaling:
        GetScaleDict(args, config)
    if args.apply_scales:
        ApplyScales(args, config)
    if args.write:
        WriteTrainSample(args, config)
