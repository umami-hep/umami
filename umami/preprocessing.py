import umami.preprocessing_tools as upt
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields
import argparse


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Enter the name of the config file to create the"
                        "hybrid sample.")
    parser.add_argument('--cut_config_file', type=str, default=None,
                        help="Enter the name of the cut config file.")
    parser.add_argument('-t', '--tracks', action='store_true',
                        help="Stores also track information.")
    return parser.parse_args()


def main():
    args = GetParser()
    config = upt.Configuration(args.config)
    N_list = upt.GetNJetsPerIteration(config)

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

        indices_toremove_Zprime = GetPtCuts(vec_Z, config, 'Zprime')
        indices_toremove_bjets = GetPtCuts(vec_tt_bjets, config)
        indices_toremove_cjets = GetPtCuts(vec_tt_cjets, config)
        indices_toremove_ujets = GetPtCuts(vec_tt_ujets, config)

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
        b_indices, c_indices, u_indices = upt.DownSampling(
            np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 5],
                            vec_tt_bjets]),
            np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 4],
                            vec_tt_cjets]),
            np.concatenate([vec_Z[vec_Z["HadronConeExclTruthLabelID"] == 0],
                            vec_tt_ujets])
        )
        ttfrac = float(len(b[b["category"] == 1]) + len(c[c["category"] == 1])
                       + len(u[u["category"] == 1])) / float(len(b) + len(c) +
                                                             len(u))
        print("ttbar fraction:", ttfrac)
        out_file = config.outfile_name
        if config.iterations > 1:
            idx = out_file.index(".h5")
            inserttxt = "-file%i_%i" % (x + 1, config.iterations)
            out_file = out_file[:idx] + inserttxt + out_file[idx:]
        print("saving file:", out_file)
        h5f = h5py.File(out_file, 'w')
        h5f.create_dataset('bjets', data=b)
        h5f.create_dataset('cjets', data=c)
        h5f.create_dataset('ujets', data=u)
        h5f.close()
        print("Plotting ...")
        tp.MakePlots(b, u, c, plot_name=config.plot_name, option=str(x),
                     binning={"pt_uncalib": 200, "abs_eta_uncalib": 200})




if __name__ == '__main__':
    main()
