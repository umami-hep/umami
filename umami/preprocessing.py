import umami.preprocessing_tools as upt
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import argparse
import os
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
    return parser.parse_args()


def RunDownsampling():
    """Applies required cuts to the samples and applies the downsampling."""
    args = GetParser()
    config = upt.Configuration(args.config_file)
    N_list = upt.GetNJetsPerIteration(config)

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

        print("starting downsampling")
        bjets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 5
                                      ], vec_tt_bjets])
        cjets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 4
                                      ], vec_tt_cjets])
        ujets = np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 0
                                      ], vec_tt_ujets])
        downs = upt.DownSampling(bjets, cjets, ujets)
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
        # TODO: Implement plotting
        # TODO: verify track handling
        # print("Plotting ...")
        # tp.MakePlots(b, u, c, plot_name=config.plot_name, option=str(x),
        #              binning={"pt_uncalib": 200, "abs_eta_uncalib": 200})


# python Preprocessing.py --no_writing --downsampled --only_scale --dummy_weights --input_file ${INPUTFILE} -f params_MC16D-2019-VRjets -o ""

def GetScaleDict():
    args = GetParser()
    config = upt.Configuration(args.config_file)
    # TODO: find good way to get file names
    input_file = config.GetFileName(iteration=1, option='downsampled')
    infile_all = h5py.File(input_file, 'r')

    # TODO: add properly Variable config
    config.variable_config = "/home/fr/fr_fr/fr_mg1150/workspace/btagging/umami/umami/configs/DL1r_Variables.yaml"
    # TODO: check if dictfile already exists in proper way
    dict_dir = "./"
    dict_file = "test.json"
    # if not (os.path.isfile("%s/%s.json" % (dict_dir, dict_file))):
    #     print("Scaler file ", "%s/%s.json" % (dict_dir, dict_file),
    #           "does not exist.",
    #           "Run first in training mode --add_dl1 0 or specify dict",
    #           "file via --dict_file.")
    #     exit()
    input_file = "test.h5"
    print('Preprocessing', input_file)

    with open(config.variable_config, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    var_list = [variable_config["label"], "category"]
    var_list += variable_config["train_variables"]
    var_list += variable_config["spectator_variables"]

    bjets = pd.DataFrame(infile_all['bjets'][:][var_list])
    cjets = pd.DataFrame(infile_all['cjets'][:][var_list])
    ujets = pd.DataFrame(infile_all['ujets'][:][var_list])
    X = pd.concat([bjets, cjets, ujets])
    del bjets, cjets, ujets
    X['weight'] = np.ones(len(X))

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # print("Apply scaling and shifting")
    # scale_dict = []
    # for var in X.columns.values:
    #     if var in [variable_config["label"], 'weight', 'category']:
    #         continue
    #     elif 'isDefaults' in var:
    #         # no scaling and shifting is applied to the check variables
    #         scale_dict.append(tp.dict_in(var, 0., 1., None))
    #     else:
    #         dict_entry = tp.Get_Shift_Scale(vec=X[var].values,
    #                                         w=X['weight'].values, varname=var)
    #         scale_dict.append(tp.dict_in(*dict_entry))

    # # save scale/shift dictionary to json file
    # scale_name = '%s/%s.json' % (args.dict_dir, args.dict_file)
    # with open(scale_name, 'w') as outfile:
    #     json.dump(scale_dict, outfile, indent=4)
    # print("saved scale dictionary as", scale_name)




if __name__ == '__main__':
    # RunDownsampling()
    GetScaleDict()
