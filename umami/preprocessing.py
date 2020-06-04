import umami.preprocessing_tools as upt
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import argparse
import yaml
from umami.tools import yaml_loader
import json
import dask.array as da
import dask.dataframe as dd
import dask

dask.config.set({'temporary_directory': '/tmp'})


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
    # possible job options for the different preprocessing steps
    parser.add_argument('-u', '--undersampling', action='store_true',
                        help="Runs undersampling.")
    parser.add_argument('-s', '--scaling', action='store_true',
                        help="Retrieves scaling and shifting factors.")
    parser.add_argument('-a', '--apply_scales', action='store_true',
                        help="Apllies scaling and shifting factors.")
    parser.add_argument('-w', '--write', action='store_true',
                        help="Shuffles sample and writes training sample and"
                             "training labels to disk")
    # the variable dictionary is always needed except for downsampling
    parser.add_argument('-v', '--var_dict', required=False, default=None,
                        help="Dictionary with input variables of tagger.",
                        type=str)
    args = parser.parse_args()
    need_var_dict = (args.scaling or args.apply_scales or args.write or not (
                     args.scaling or args.write or args.apply_scales or
                     args.undersampling))

    if need_var_dict and args.var_dict is None:
        parser.error('It is required to specify --var_dict [-v]')

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
    # TODO: switch to dask
    config = upt.Configuration(args.config_file)
    # TODO: find good way to get file names, breaks if no iterations
    input_file = config.GetFileName(iteration=1, option='downsampled')
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

    print("Retrieving scaling and shifting values")

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

    # save scale/shift dictionary to json file
    with open(config.dict_file, 'w') as outfile:
        json.dump(scale_dict, outfile, indent=4)
    print("saved scale dictionary as", config.dict_file)


def ApplyScales(args, config):
    """
        Apply the scaling and shifting to dataset
    """
    input_files = [config.GetFileName(iteration=x+1, option='downsampled') for
                   x in range(config.iterations)]

    dsets = [h5py.File(fn, 'r')['/bjets'][:] for fn in input_files]
    dsets += [h5py.File(fn, 'r')['/cjets'][:] for fn in input_files]
    dsets += [h5py.File(fn, 'r')['/ujets'][:] for fn in input_files]
    arrays = [da.from_array(dset) for dset in dsets]
    x = da.concatenate(arrays, axis=0)  # Concatenate arrays along first axis
    df = x.to_dask_dataframe()
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    variables = variable_config["train_variables"][:]
    variables += variable_config["spectator_variables"][:]
    variables += [variable_config["label"], 'weight', 'category']
    # df_len = len(df)
    if 'weight' not in df.columns.values:
        # df['weight'] = dd.from_array(np.ones(df_len))
        # TODO: ugly workaround, but if not used a lot of NaNs are appended
        # instad of ones (see also https://stackoverflow.com/questions/46923274/appending-new-column-to-dask-dataframe?rq=1) # noqa
        df['weight'] = df["category"] * 0 + 1
    df = df[variables]
    df = df.replace([np.inf, -np.inf], np.nan)
    with open(config.dict_file, 'r') as infile:
        scale_dict = json.load(infile)
    print("Replacing default values.")
    default_dict = upt.Gen_default_dict(scale_dict)
    df = df.fillna(default_dict)
    # var_list = variable_config["train_variables"]
    print("Applying scaling and shifting.")
    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        else:
            df[elem['name']] -= elem['shift']
            df[elem['name']] /= elem['scale']

    out_file = config.GetFileName(option='preprocessed')
    print("Saving file:", out_file)
    df.to_hdf(out_file, '/jets')


def WriteTrainSample(args, config):
    with open(args.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    in_file = config.GetFileName(option='preprocessed')
    df = dd.read_hdf(in_file, '/jets')
    # df = dd.from_pandas(df.head(1000), npartitions=3)
    df_len = len(df)

    print("Calculating binary labels.")
    labels = upt.GetBinaryLabels(df, variable_config['label'])
    np.random.seed(42)
    np.random.shuffle(labels)

    out_file = config.GetFileName(option='preprocessed_shuffled')
    print("Saving labels to", out_file)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('/Y_train', data=labels, compression="gzip")
    del labels

    print("Shuffling sample")
    d_arr = upt.ShuffleDataFrame(df[variable_config['train_variables'][:]],
                                 df_len=df_len)
    print("Saving sample to", out_file)
    with h5py.File(out_file, 'a') as f:
        d = f.require_dataset('/X_train', shape=d_arr.shape,
                              dtype=d_arr.dtype)
        da.store(d_arr, d, compression='lzf')


if __name__ == '__main__':
    args = GetParser()
    config = upt.Configuration(args.config_file)
    if not (args.undersampling or args.scaling or args.apply_scales or
            args.write):
        args.undersampling = True
        args.scaling = True
        args.apply_scales = True
        args.write = True
    if args.undersampling:
        RunUndersampling(args, config)
    if args.scaling:
        GetScaleDict(args, config)
    if args.apply_scales:
        ApplyScales(args, config)
    if args.write:
        WriteTrainSample(args, config)
