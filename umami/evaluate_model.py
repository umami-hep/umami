import h5py
import argparse
import os

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import CustomObjectScope

import umami.train_tools as utt
from umami.evaluation_tools.PlottingFunctions import GetScore
from umami.preprocessing_tools import Configuration
# from plottingFunctions import sigBkgEff


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing command line options."
    )

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        required=True,
        help="Name of the training config file"
    )

    parser.add_argument(
        '-e',
        '--epoch',
        type=int,
        required=True,
        default=None,
        help="Epoch which should be evaluated."
    )

    # TODO: implement vr_overlap
    # parser.add_argument('--vr_overlap', action='store_true', help='''Option
    #                     toenable vr overlap removall for validation sets.''')

    parser.add_argument(
        '--dl1',
        action='store_true',
        help="Evaluating DL1 like tagger with one loss."
    )

    parser.add_argument(
        '--dips',
        action='store_true',
        help="Evaluating Dips tagger with one loss."
    )

    parser.add_argument(
        '--nJets',
        type=int,
        default=None,
        help='''Number of jets used for the testing. By default it will
        use all available jets in the test files.'''
    )
    parser.add_argument(
        '--beff',
        type=float,
        default=0.77,
        help='b efficiency used for the charm fraction scan'
    )
    parser.add_argument(
        '--cfrac',
        type=float,
        default=0.018,
        help='charm fraction used for the b efficiency scan. The charm fraction for the recommended taggers are not affected by this! They are 0.018 / 0.08 for DL1r / RNNIP.'
    )

    args = parser.parse_args()
    return args



def EvaluateModel(
    args, train_config, preprocess_config, test_file, data_set_name
):
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    print("Evaluating", model_file)
    exclude=[]
    if "exclude" in train_config.config:
        exclude=train_config.config["exclude"]
    X_test, X_test_trk, Y_test = utt.GetTestFile(
        test_file, train_config.var_dict, preprocess_config, nJets=args.nJets, exclude=exclude
    )
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    pred_dips, pred_umami = model.predict(
        [X_test_trk, X_test], batch_size=5000, verbose=0
    )
    y_true = np.argmax(Y_test, axis=1)
    b_index, c_index, u_index = 2, 1, 0
    variables = ['absEta_btagJes', 'pt_btagJes', 'DL1r_pb', 'DL1r_pc',
                 'DL1r_pu', 'rnnip_pb', 'rnnip_pc', 'rnnip_pu',
                 'HadronConeExclTruthLabelID']
    df = pd.DataFrame(
        h5py.File(test_file, 'r')['/jets'][:args.nJets][
            variables])
    print("Jets used for testing:", len(df))
    df.query('HadronConeExclTruthLabelID <= 5', inplace=True)
    df_discs = pd.DataFrame({
        "umami_pb": pred_umami[:, b_index],
        "umami_pc": pred_umami[:, c_index],
        "umami_pu": pred_umami[:, u_index],
        "dips_pb": pred_dips[:, b_index],
        "dips_pc": pred_dips[:, c_index],
        "dips_pu": pred_dips[:, u_index],
        "pt": df["pt_btagJes"],
        "eta": df['absEta_btagJes'],
        "labels": y_true,
        "disc_DL1r": GetScore(df["DL1r_pb"], df["DL1r_pc"], df["DL1r_pu"]),
        "disc_rnnip": GetScore(df["rnnip_pb"], df["rnnip_pc"], df["rnnip_pu"],
                               fc=0.08)
        })

    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name)

    x_axis_granularity = 100

    print("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, x_axis_granularity)
    crej_arr_umami = []
    urej_arr_umami = []
    crej_arr_dips = []
    urej_arr_dips = []
    crej_arr_dl1r = []
    urej_arr_dl1r = []
    crej_arr_rnnip = []
    urej_arr_rnnip = []

    for eff in b_effs:
        crej_i, urej_i = utt.GetRejection(pred_umami, Y_test, target_beff=eff,
                                          cfrac=args.cfrac)
        crej_arr_umami.append(crej_i)
        urej_arr_umami.append(urej_i)
        crej_i, urej_i = utt.GetRejection(pred_dips, Y_test, target_beff=eff,
                                          cfrac=args.cfrac)
        crej_arr_dips.append(crej_i)
        urej_arr_dips.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["DL1r_pu", "DL1r_pc",
                                              "DL1r_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.018)
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["rnnip_pu", "rnnip_pc",
                                              "rnnip_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.08)
        crej_arr_rnnip.append(crej_i)
        urej_arr_rnnip.append(urej_i)

    print("calculating rejections per fc")
    fc_values = np.linspace(0.001, 0.1, x_axis_granularity)


    crej_arr_umami_cfrac = []
    urej_arr_umami_cfrac = []
    crej_arr_dips_cfrac = []
    urej_arr_dips_cfrac = []
    crej_arr_dl1r_cfrac = []
    urej_arr_dl1r_cfrac = []
    crej_arr_rnnip_cfrac = []
    urej_arr_rnnip_cfrac = []

    for fc in fc_values:
        crej_i, urej_i = utt.GetRejection(pred_umami, Y_test, target_beff=args.beff,
                                          cfrac=fc)
        crej_arr_umami_cfrac.append(crej_i)
        urej_arr_umami_cfrac.append(urej_i)
        crej_i, urej_i = utt.GetRejection(pred_dips, Y_test, target_beff=args.beff,
                                          cfrac=fc)
        crej_arr_dips_cfrac.append(crej_i)
        urej_arr_dips_cfrac.append(urej_i)

        crej_dl1r, urej_dl1r = utt.GetRejection(df[["DL1r_pu", "DL1r_pc",
                                                "DL1r_pb"]].values,
                                            Y_test, target_beff=args.beff,
                                            cfrac=fc)
        crej_arr_dl1r_cfrac.append(crej_dl1r)
        urej_arr_dl1r_cfrac.append(urej_dl1r)

        crej_rnnip, urej_rnnip = utt.GetRejection(df[["rnnip_pu", "rnnip_pc",
                                                "rnnip_pb"]].values,
                                            Y_test, target_beff=args.beff,
                                            cfrac=fc)
        crej_arr_rnnip_cfrac.append(crej_rnnip)
        urej_arr_rnnip_cfrac.append(urej_rnnip)


    df_eff_rej = pd.DataFrame({
        "beff": b_effs,
        "umami_crej": crej_arr_umami,
        "umami_urej": urej_arr_umami,
        "dips_crej": crej_arr_dips,
        "dips_urej": urej_arr_dips,
        "dl1r_crej": crej_arr_dl1r,
        "dl1r_urej": urej_arr_dl1r,
        "rnnip_crej": crej_arr_rnnip,
        "rnnip_urej": urej_arr_rnnip,
        "fc_values": fc_values,
        "dl1r_cfrac_crej" : crej_arr_dl1r_cfrac,
        "dl1r_cfrac_urej" : urej_arr_dl1r_cfrac,
        "rnnip_cfrac_crej" : crej_arr_rnnip_cfrac,
        "rnnip_cfrac_urej" : urej_arr_rnnip_cfrac,
        "umami_cfrac_crej" : crej_arr_umami_cfrac,
        "umami_cfrac_urej" : urej_arr_umami_cfrac,
        "dips_cfrac_crej" : crej_arr_dips_cfrac,
        "dips_cfrac_urej" : urej_arr_dips_cfrac,
    })
    df_eff_rej.to_hdf(f"{train_config.model_name}/results/results-rej_per_eff"
                      f"-{args.epoch}.h5", data_set_name)

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    f = h5py.File(
        f"{train_config.model_name}/results/" +
        f"results-rej_per_eff-{args.epoch}.h5",
        "a")
    f.attrs["N_test"] = len(df)
    f.close()


def EvaluateModelDips(
    args, train_config, preprocess_config, test_file, data_set_name
):

    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    print("Evaluating", model_file)
    _, X_test_trk, Y_test = utt.GetTestFile(
        test_file, train_config.var_dict, preprocess_config, nJets=args.nJets
    )
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    pred_dips = model.predict(
        X_test_trk, batch_size=5000, verbose=0
    )
    y_true = np.argmax(Y_test, axis=1)
    b_index, c_index, u_index = 2, 1, 0
    variables = ['absEta_btagJes', 'pt_btagJes', 'DL1r_pb', 'DL1r_pc',
                 'DL1r_pu', 'rnnip_pb', 'rnnip_pc', 'rnnip_pu',
                 'HadronConeExclTruthLabelID']
    df = pd.DataFrame(
        h5py.File(test_file, 'r')['/jets'][:args.nJets][
            variables])
    print("Jets used for testing:", len(df))
    df.query('HadronConeExclTruthLabelID <= 5', inplace=True)
    df_discs = pd.DataFrame({
        "dips_pb": pred_dips[:, b_index],
        "dips_pc": pred_dips[:, c_index],
        "dips_pu": pred_dips[:, u_index],
        "pt": df["pt_btagJes"],
        "eta": df['absEta_btagJes'],
        "labels": y_true,
        "disc_DL1r": GetScore(df["DL1r_pb"], df["DL1r_pc"], df["DL1r_pu"]),
        "disc_rnnip": GetScore(df["rnnip_pb"], df["rnnip_pc"], df["rnnip_pu"],
                               fc=0.08)
        })

    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name)

    print("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, 150)
    crej_arr_dips = []
    urej_arr_dips = []
    crej_arr_dl1r = []
    urej_arr_dl1r = []
    crej_arr_rnnip = []
    urej_arr_rnnip = []

    for eff in b_effs:
        crej_i, urej_i = utt.GetRejection(pred_dips, Y_test, target_beff=eff,
                                          cfrac=0.018)
        crej_arr_dips.append(crej_i)
        urej_arr_dips.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["DL1r_pu", "DL1r_pc",
                                              "DL1r_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.018)
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["rnnip_pu", "rnnip_pc",
                                              "rnnip_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.08)
        crej_arr_rnnip.append(crej_i)
        urej_arr_rnnip.append(urej_i)

    df_eff_rej = pd.DataFrame({
        "beff": b_effs,
        "dips_crej": crej_arr_dips,
        "dips_urej": urej_arr_dips,
        "dl1r_crej": crej_arr_dl1r,
        "dl1r_urej": urej_arr_dl1r,
        "rnnip_crej": crej_arr_rnnip,
        "rnnip_urej": urej_arr_rnnip
    })
    df_eff_rej.to_hdf(f"{train_config.model_name}/results/results-rej_per_eff"
                      f"-{args.epoch}.h5", data_set_name)

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    f = h5py.File(
        f"{train_config.model_name}/results/" +
        f"results-rej_per_eff-{args.epoch}.h5",
        "a")
    f.attrs["N_test"] = len(df)
    f.close()


def EvaluateModelDL1(
    args, train_config, preprocess_config, test_file, data_set_name
):
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    print("Evaluating", model_file)
    X_test, Y_test = utt.GetTestSample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config, nJets=args.nJets)
    # with CustomObjectScope({'Sum': Sum}):
    model = load_model(model_file)

    pred = model.predict(X_test, batch_size=5000, verbose=0)
    y_true = np.argmax(Y_test, axis=1)
    b_index, c_index, u_index = 2, 1, 0
    variables = ['absEta_btagJes', 'pt_btagJes', 'DL1r_pb', 'DL1r_pc',
                 'DL1r_pu', 'rnnip_pb', 'rnnip_pc', 'rnnip_pu',
                 'HadronConeExclTruthLabelID']
    df = pd.DataFrame(
        h5py.File(test_file, 'r')['/jets'][:args.nJets][
            variables])
    print("Jets used for testing:", len(df))
    df.query('HadronConeExclTruthLabelID <= 5', inplace=True)
    df_discs = pd.DataFrame({
        "pb": pred[:, b_index],
        "pc": pred[:, c_index],
        "pu": pred[:, u_index],
        "pt": df["pt_btagJes"],
        "eta": df['absEta_btagJes'],
        "labels": y_true,
        "disc_DL1r": GetScore(df["DL1r_pb"], df["DL1r_pc"], df["DL1r_pu"]),
        "disc_rnnip": GetScore(df["rnnip_pb"], df["rnnip_pc"], df["rnnip_pu"],
                               fc=0.08)
        })

    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name)

    print("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, 150)
    crej_arr = []
    urej_arr = []
    crej_arr_dl1r = []
    urej_arr_dl1r = []
    crej_arr_rnnip = []
    urej_arr_rnnip = []

    for eff in b_effs:
        crej_i, urej_i = utt.GetRejection(pred, Y_test, target_beff=eff,
                                          cfrac=0.018)
        crej_arr.append(crej_i)
        urej_arr.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["DL1r_pu", "DL1r_pc",
                                              "DL1r_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.018)
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        crej_i, urej_i = utt.GetRejection(df[["rnnip_pu", "rnnip_pc",
                                              "rnnip_pb"]].values,
                                          Y_test, target_beff=eff,
                                          cfrac=0.08)
        crej_arr_rnnip.append(crej_i)
        urej_arr_rnnip.append(urej_i)

    df_eff_rej = pd.DataFrame({
        "beff": b_effs,
        "umami_crej": crej_arr,
        "umami_urej": urej_arr,
        "dl1r_crej": crej_arr_dl1r,
        "dl1r_urej": urej_arr_dl1r,
        "rnnip_crej": crej_arr_rnnip,
        "rnnip_urej": urej_arr_rnnip
    })
    df_eff_rej.to_hdf(f"{train_config.model_name}/results/results-rej_per_eff"
                      f"-{args.epoch}.h5", data_set_name)

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binominal errors
    f = h5py.File(
        f"{train_config.model_name}/results/" +
        f"results-rej_per_eff-{args.epoch}.h5",
        "a")
    f.attrs["N_test"] = len(df)
    f.close()


if __name__ == "__main__":
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    if args.dl1:
        EvaluateModelDL1(
            args,
            train_config,
            preprocess_config,
            train_config.test_file,
            "ttbar",
        )

        if train_config.add_test_file is not None:
            EvaluateModelDL1(
                args,
                train_config,
                preprocess_config,
                train_config.add_test_file,
                "zpext",
            )

    elif args.dips:
        print('Start evaluating DIPS with ttbar...')
        EvaluateModelDips(
            args,
            train_config,
            preprocess_config,
            train_config.test_file,
            "ttbar",
        )

        if train_config.add_test_file is not None:
            print('Start evaluating DIPS with Zprime...')
            EvaluateModelDips(
                args,
                train_config,
                preprocess_config,
                train_config.add_test_file,
                "zpext",
            )

    else:
        EvaluateModel(
            args,
            train_config,
            preprocess_config,
            train_config.test_file,
            "ttbar",
        )
        if train_config.add_test_file is not None:
            EvaluateModel(
                args,
                train_config,
                preprocess_config,
                train_config.add_test_file,
                "zpext",
            )
