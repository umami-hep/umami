import argparse
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.train_tools as utt
from umami.configuration import global_config
from umami.evaluation_tools.PlottingFunctions import (
    GetScore,
    GetScoreC,
    discriminant_output_shape,
    get_gradients,
    getDiscriminant,
)
from umami.preprocessing_tools import Configuration

# from plottingFunctions import sigBkgEff
tf.compat.v1.disable_eager_execution()


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing command line options."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Name of the training config file",
    )

    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        required=True,
        help="Epoch which should be evaluated.",
    )

    # TODO: implement vr_overlap
    # parser.add_argument('--vr_overlap', action='store_true', help='''Option
    #                     toenable vr overlap removall for validation sets.''')

    parser.add_argument(
        "--dl1",
        action="store_true",
        help="Evaluating DL1 like tagger with one loss.",
    )

    parser.add_argument(
        "--dips",
        action="store_true",
        help="Evaluating Dips tagger with one loss.",
    )

    parser.add_argument(
        "--nJets",
        type=int,
        default=None,
        help="""Number of jets used for the testing. By default it will
        use all available jets in the test files.""",
    )

    parser.add_argument(
        "--beff",
        type=float,
        default=0.77,
        help="b efficiency used for the charm fraction scan",
    )

    parser.add_argument(
        "--cfrac",
        type=float,
        default=0.018,
        help="""charm fraction used for the b efficiency scan.
        The charm fraction for the recommended taggers are not
        affected by this! They are 0.018 / 0.08 for DL1r / RNNIP.""",
    )

    args = parser.parse_args()
    return args


def EvaluateModel(
    args, train_config, preprocess_config, test_file, data_set_name
):
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    print("Evaluating", model_file)
    exclude = []
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]
    X_test, X_test_trk, Y_test = utt.GetTestFile(
        test_file,
        train_config.var_dict,
        preprocess_config,
        nJets=args.nJets,
        exclude=exclude,
    )
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    pred_dips, pred_umami = model.predict(
        [X_test_trk, X_test], batch_size=5000, verbose=0
    )
    y_true = np.argmax(Y_test, axis=1)
    b_index, c_index, u_index = 2, 1, 0
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "DL1r_pb",
        "DL1r_pc",
        "DL1r_pu",
        "rnnip_pb",
        "rnnip_pc",
        "rnnip_pu",
        "HadronConeExclTruthLabelID",
    ]
    df = pd.DataFrame(
        h5py.File(test_file, "r")["/jets"][: args.nJets][variables]
    )
    print("Jets used for testing:", len(df))
    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
    df_discs = pd.DataFrame(
        {
            "umami_pb": pred_umami[:, b_index],
            "umami_pc": pred_umami[:, c_index],
            "umami_pu": pred_umami[:, u_index],
            "dips_pb": pred_dips[:, b_index],
            "dips_pc": pred_dips[:, c_index],
            "dips_pu": pred_dips[:, u_index],
            "pt": df[global_config.pTvariable],
            "eta": df[global_config.etavariable],
            "labels": y_true,
            "disc_DL1r": GetScore(df["DL1r_pb"], df["DL1r_pc"], df["DL1r_pu"]),
            "disc_rnnip": GetScore(
                df["rnnip_pb"], df["rnnip_pc"], df["rnnip_pu"], fc=0.08
            ),
        }
    )

    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

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
        crej_i, urej_i = utt.GetRejection(
            pred_umami, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_arr_umami.append(crej_i)
        urej_arr_umami.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_arr_dips.append(crej_i)
        urej_arr_dips.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
            Y_test,
            target_eff=eff,
            frac=0.018,
        )
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
            Y_test,
            target_eff=eff,
            frac=0.08,
        )
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
        crej_i, urej_i = utt.GetRejection(
            pred_umami, Y_test, target_eff=args.beff, frac=fc
        )
        crej_arr_umami_cfrac.append(crej_i)
        urej_arr_umami_cfrac.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=args.beff, frac=fc
        )
        crej_arr_dips_cfrac.append(crej_i)
        urej_arr_dips_cfrac.append(urej_i)

        crej_dl1r, urej_dl1r = utt.GetRejection(
            df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
            Y_test,
            target_eff=args.beff,
            frac=fc,
        )
        crej_arr_dl1r_cfrac.append(crej_dl1r)
        urej_arr_dl1r_cfrac.append(urej_dl1r)

        crej_rnnip, urej_rnnip = utt.GetRejection(
            df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
            Y_test,
            target_eff=args.beff,
            frac=fc,
        )
        crej_arr_rnnip_cfrac.append(crej_rnnip)
        urej_arr_rnnip_cfrac.append(urej_rnnip)

    df_eff_rej = pd.DataFrame(
        {
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
            "dl1r_cfrac_crej": crej_arr_dl1r_cfrac,
            "dl1r_cfrac_urej": urej_arr_dl1r_cfrac,
            "rnnip_cfrac_crej": crej_arr_rnnip_cfrac,
            "rnnip_cfrac_urej": urej_arr_rnnip_cfrac,
            "umami_cfrac_crej": crej_arr_umami_cfrac,
            "umami_cfrac_urej": urej_arr_umami_cfrac,
            "dips_cfrac_crej": crej_arr_dips_cfrac,
            "dips_cfrac_urej": urej_arr_dips_cfrac,
        }
    )

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff"
        f"-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    f = h5py.File(
        f"{train_config.model_name}/results/"
        + f"results-rej_per_eff-{args.epoch}.h5",
        "a",
    )
    f.attrs["N_test"] = len(df)
    f.close()


def EvaluateModelDips(
    args, train_config, preprocess_config, test_file, data_set_name
):
    # Set number of nJets for testing
    if args.nJets is None:
        nJets_test = int(train_config.Eval_parameters_validation["n_jets"])

    else:
        nJets_test = args.nJets

    # Define model path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    print("Evaluating", model_file)

    # Get the testfile with the needed configs
    X_test_trk, Y_test = utt.GetTestSampleTrks(
        test_file,
        train_config.var_dict,
        preprocess_config,
        nJets=nJets_test,
    )

    # Load pretrained model
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    # Get predictions from trained model
    pred_dips = model.predict(
        X_test_trk,
        batch_size=train_config.NN_structure["batch_size"],
        verbose=0,
    )

    # Setting y_true
    y_true = np.argmax(Y_test, axis=1)

    # Define the index values for the different flavours
    b_index, c_index, u_index = 2, 1, 0

    # Define the needed extra variables
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "DL1r_pb",
        "DL1r_pc",
        "DL1r_pu",
        "rnnip_pb",
        "rnnip_pc",
        "rnnip_pu",
        "HadronConeExclTruthLabelID",
    ]

    # Load the test data
    df = pd.DataFrame(
        h5py.File(test_file, "r")["/jets"][:nJets_test][variables]
    )
    print("Jets used for testing:", len(df))

    # Define the jets used
    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

    # Fill the new dataframe with the evaluated parameters
    df_discs = pd.DataFrame(
        {
            "dips_pb": pred_dips[:, b_index],
            "dips_pc": pred_dips[:, c_index],
            "dips_pu": pred_dips[:, u_index],
            "rnnip_pb": df["rnnip_pb"],
            "rnnip_pc": df["rnnip_pc"],
            "rnnip_pu": df["rnnip_pu"],
            "DL1r_pb": df["DL1r_pb"],
            "DL1r_pc": df["DL1r_pc"],
            "DL1r_pu": df["DL1r_pu"],
            "pt": df[global_config.pTvariable],
            "eta": df[global_config.etavariable],
            "labels": y_true,
            "disc_DL1r": GetScore(df["DL1r_pb"], df["DL1r_pc"], df["DL1r_pu"]),
            "disc_rnnip": GetScore(
                df["rnnip_pb"], df["rnnip_pc"], df["rnnip_pu"], fc=0.08
            ),
        }
    )

    # Create results dir and .h5 file
    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    x_axis_granularity = 100

    print("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, x_axis_granularity)
    crej_arr_dips = []
    urej_arr_dips = []
    crej_arr_dl1r = []
    urej_arr_dl1r = []
    crej_arr_rnnip = []
    urej_arr_rnnip = []

    for eff in b_effs:
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_arr_dips.append(crej_i)
        urej_arr_dips.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
            Y_test,
            target_eff=eff,
            frac=0.018,
        )
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        crej_i, urej_i = utt.GetRejection(
            df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
            Y_test,
            target_eff=eff,
            frac=0.08,
        )
        crej_arr_rnnip.append(crej_i)
        urej_arr_rnnip.append(urej_i)

    print("calculating rejections per fc")
    fc_values = np.linspace(0.001, 0.1, x_axis_granularity)

    crej_arr_dips_cfrac = []
    urej_arr_dips_cfrac = []
    crej_arr_dl1r_cfrac = []
    urej_arr_dl1r_cfrac = []
    crej_arr_rnnip_cfrac = []
    urej_arr_rnnip_cfrac = []

    for fc in fc_values:
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=args.beff, frac=fc
        )
        crej_arr_dips_cfrac.append(crej_i)
        urej_arr_dips_cfrac.append(urej_i)

        crej_dl1r, urej_dl1r = utt.GetRejection(
            df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
            Y_test,
            target_eff=args.beff,
            frac=fc,
        )
        crej_arr_dl1r_cfrac.append(crej_dl1r)
        urej_arr_dl1r_cfrac.append(urej_dl1r)

        crej_rnnip, urej_rnnip = utt.GetRejection(
            df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
            Y_test,
            target_eff=args.beff,
            frac=fc,
        )
        crej_arr_rnnip_cfrac.append(crej_rnnip)
        urej_arr_rnnip_cfrac.append(urej_rnnip)

    df_eff_rej = pd.DataFrame(
        {
            "beff": b_effs,
            "dips_crej": crej_arr_dips,
            "dips_urej": urej_arr_dips,
            "dl1r_crej": crej_arr_dl1r,
            "dl1r_urej": urej_arr_dl1r,
            "rnnip_crej": crej_arr_rnnip,
            "rnnip_urej": urej_arr_rnnip,
            "fc_values": fc_values,
            "dl1r_cfrac_crej": crej_arr_dl1r_cfrac,
            "dl1r_cfrac_urej": urej_arr_dl1r_cfrac,
            "rnnip_cfrac_crej": crej_arr_rnnip_cfrac,
            "rnnip_cfrac_urej": urej_arr_rnnip_cfrac,
            "dips_cfrac_crej": crej_arr_dips_cfrac,
            "dips_cfrac_urej": urej_arr_dips_cfrac,
        }
    )

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff"
        f"-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    f = h5py.File(
        f"{train_config.model_name}/results/"
        + f"results-rej_per_eff-{args.epoch}.h5",
        "a",
    )
    f.attrs["N_test"] = len(df)
    f.close()

    print("Calculate gradients for inputs")
    # Cut off last layer of the model for saliency maps
    cutted_model = model.layers[-1].output

    # Define the last node for the discriminant output
    disc = Lambda(getDiscriminant, output_shape=discriminant_output_shape)(
        cutted_model
    )

    # Define the computation graph for the model
    model = Model(model.inputs, disc)

    # Define boolean mask to filter placeholder tracks
    boolMask = (np.sum(X_test_trk, axis=-1) != 0).astype(bool)

    # Define the number of true tracks per jet as a mask
    nTrks = np.sum(boolMask, axis=-1)

    # Get score for the dips prediction
    Db = GetScore(
        pb=pred_dips[:, b_index],
        pc=pred_dips[:, c_index],
        pu=pred_dips[:, u_index],
    )

    # Init small dict
    map_dict = {}

    # Iterate over different beff, jet flavours and passed options
    for target_beff in [60, 70, 77, 85]:
        for jet_flavour in [0, 1, 2]:
            for PassBool in [True, False]:

                # Get the Db value for a specific flavour
                Db_flavour = Db[Y_test[:, jet_flavour].astype(bool)]

                # Get the cutvalue for the specific WP
                cutvalue = np.percentile(Db_flavour, (100 - target_beff))

                # Set PassBool masking
                if PassBool is True:
                    mask = Y_test[:, jet_flavour].astype(bool)
                    mask = mask & (nTrks == 8)
                    mask = mask & (Db > cutvalue)

                elif PassBool is False:
                    mask = Y_test[:, jet_flavour].astype(bool)
                    mask = mask & (nTrks == 8)
                    mask = mask & (Db < cutvalue)

                # Get gradient map
                gradient_map = get_gradients(model, X_test_trk[mask], 15000)

                # Turn gradient map for plotting
                gradient_map = np.swapaxes(gradient_map, 1, 2)

                # Mean over the jets
                gradient_map = np.mean(gradient_map, axis=0)

                map_dict.update(
                    {f"{target_beff}_{jet_flavour}_{PassBool}": gradient_map}
                )

    # Create results dir and .h5 ile
    os.system(f"mkdir -p {train_config.model_name}/results")
    with open(
        f"{train_config.model_name}/results/saliency_{args.epoch}"
        + f"_{data_set_name}.pkl",
        "wb",
    ) as f:
        pickle.dump(map_dict, f)


def EvaluateModelDL1(
    args, train_config, preprocess_config, test_file, data_set_name
):
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"

    # Set fractions
    fc_value = train_config.Eval_parameters_validation["fc_value"]
    fb_value = train_config.Eval_parameters_validation["fb_value"]
    if "ftauforb_value" in train_config.Eval_parameters_validation:
        ftauforb_value = train_config.Eval_parameters_validation[
            "ftauforb_value"
        ]
    else:
        ftauforb_value = None
    if "ftauforc_value" in train_config.Eval_parameters_validation:
        ftauforc_value = train_config.Eval_parameters_validation[
            "ftauforc_value"
        ]
    else:
        ftauforc_value = None

    # Manage taus
    bool_use_taus = (
        train_config.bool_use_taus and preprocess_config.bool_process_taus
    )
    if bool_use_taus:
        print("Evaluating {} with taus".format(model_file))
    else:
        print("Evaluating", model_file)
        ftauforc_value = None
        ftauforb_value = None

    X_test, Y_test = utt.GetTestSample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        nJets=args.nJets,
        use_taus=bool_use_taus,
    )
    # with CustomObjectScope({'Sum': Sum}):
    model = load_model(model_file)

    pred = model.predict(X_test, batch_size=5000, verbose=0)
    y_true = np.argmax(Y_test, axis=1)
    tau_index, b_index, c_index, u_index = 3, 2, 1, 0

    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "DL1r_pb",
        "DL1r_pc",
        "DL1r_pu",
        "rnnip_pb",
        "rnnip_pc",
        "rnnip_pu",
        "HadronConeExclTruthLabelID",
    ]
    add_variables = train_config.Eval_parameters_validation[
        "add_variables_eval"
    ]
    add_variables_available = None
    if add_variables is not None:
        available_variables = list(
            pd.DataFrame(h5py.File(test_file, "r")["/jets"][:1][:]).columns
        )
        add_variables_available = []
        for item in add_variables:
            if item in available_variables:
                add_variables_available.append(item)
            else:
                print("Variable '{}' not available".format(item))
        variables.extend(add_variables_available)

    df = pd.DataFrame(
        h5py.File(test_file, "r")["/jets"][: args.nJets][variables]
    )
    print("Jets used for testing:", len(df))
    if bool_use_taus:
        df.query("HadronConeExclTruthLabelID in [0, 4, 5, 15]", inplace=True)
        if "DL1r_ptau" not in df:
            df["DL1r_ptau"] = 0
        if "rnnip_ptau" not in df:
            df["rnnip_ptau"] = 0
        df_discs = pd.DataFrame(
            {
                "ptau": pred[:, tau_index],
                "pb": pred[:, b_index],
                "pc": pred[:, c_index],
                "pu": pred[:, u_index],
                "pt": df[global_config.pTvariable],
                "eta": df[global_config.etavariable],
                "labels": y_true,
                "disc_DL1r": GetScore(
                    df["DL1r_pb"],
                    df["DL1r_pc"],
                    df["DL1r_pu"],
                    df["DL1r_ptau"],
                    fc=fc_value,
                    ftau=ftauforb_value,
                ),
                "disc_DL1rC": GetScoreC(
                    df["DL1r_pb"],
                    df["DL1r_pc"],
                    df["DL1r_pu"],
                    df["DL1r_ptau"],
                    fb=fb_value,
                    ftau=ftauforc_value,
                ),
                "disc_rnnip": GetScore(
                    df["rnnip_pb"],
                    df["rnnip_pc"],
                    df["rnnip_pu"],
                    df["rnnip_ptau"],
                    fc=fc_value,
                    ftau=ftauforb_value,
                ),
                "disc_rnnipC": GetScoreC(
                    df["rnnip_pb"],
                    df["rnnip_pc"],
                    df["rnnip_pu"],
                    df["rnnip_ptau"],
                    fb=fb_value,
                    ftau=ftauforc_value,
                ),
            }
        )
    else:
        df.query("HadronConeExclTruthLabelID <= 5", inplace=True)
        df_discs = pd.DataFrame(
            {
                "pb": pred[:, b_index],
                "pc": pred[:, c_index],
                "pu": pred[:, u_index],
                "pt": df[global_config.pTvariable],
                "eta": df[global_config.etavariable],
                "labels": y_true,
                "disc_DL1r": GetScore(
                    df["DL1r_pb"],
                    df["DL1r_pc"],
                    df["DL1r_pu"],
                    fc=fc_value,
                    ftau=ftauforb_value,
                ),
                "disc_DL1rC": GetScoreC(
                    df["DL1r_pb"],
                    df["DL1r_pc"],
                    df["DL1r_pu"],
                    fb=fb_value,
                    ftau=ftauforc_value,
                ),
                "disc_rnnip": GetScore(
                    df["rnnip_pb"],
                    df["rnnip_pc"],
                    df["rnnip_pu"],
                    fc=fc_value,
                    ftau=ftauforb_value,
                ),
                "disc_rnnipC": GetScoreC(
                    df["rnnip_pb"],
                    df["rnnip_pc"],
                    df["rnnip_pu"],
                    fb=fb_value,
                    ftau=ftauforc_value,
                ),
            }
        )
    if add_variables_available is not None:
        for item in add_variables_available:
            print("Adding ", item)
            df_discs[item] = df[item]

    os.system(f"mkdir -p {train_config.model_name}/results")
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    print("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, 150)
    c_effs = np.linspace(0.09, 1, 150)
    crej_arr = []
    urej_arr = []
    crej_arr_dl1r = []
    urej_arr_dl1r = []
    crej_arr_rnnip = []
    urej_arr_rnnip = []
    brej_arrC = []
    urej_arrC = []
    brej_arr_dl1rC = []
    urej_arr_dl1rC = []
    brej_arr_rnnipC = []
    urej_arr_rnnipC = []
    if bool_use_taus:
        taurej_arr = []
        taurej_arr_dl1r = []
        taurej_arr_rnnip = []
        taurej_arrC = []
        taurej_arr_dl1rC = []
        taurej_arr_rnnipC = []

    for ind_eff, b_eff in enumerate(b_effs):
        c_eff = c_effs[ind_eff]
        if bool_use_taus:
            crej_i, urej_i, taurej_i = utt.GetRejection(
                pred,
                Y_test,
                target_eff=b_eff,
                frac=fc_value,
                taufrac=ftauforb_value,
                use_taus=bool_use_taus,
            )
            brej_iC, urej_iC, taurej_iC = utt.GetRejection(
                pred,
                Y_test,
                d_type="c",
                target_eff=c_eff,
                frac=fb_value,
                taufrac=ftauforc_value,
                use_taus=bool_use_taus,
            )
            taurej_arr.append(taurej_i)
            taurej_arrC.append(taurej_iC)
        else:
            crej_i, urej_i = utt.GetRejection(
                pred, Y_test, target_eff=b_eff, frac=fc_value
            )
            brej_iC, urej_iC = utt.GetRejection(
                pred, Y_test, d_type="c", target_eff=c_eff, frac=fb_value
            )
        crej_arr.append(crej_i)
        urej_arr.append(urej_i)
        brej_arrC.append(brej_iC)
        urej_arrC.append(urej_iC)

        if bool_use_taus:
            crej_i, urej_i, taurej_i = utt.GetRejection(
                df[["DL1r_pu", "DL1r_pc", "DL1r_pb", "DL1r_ptau"]].values,
                Y_test,
                target_eff=b_eff,
                frac=fc_value,
                taufrac=ftauforb_value,
                use_taus=bool_use_taus,
            )
            brej_iC, urej_iC, taurej_iC = utt.GetRejection(
                df[["DL1r_pu", "DL1r_pc", "DL1r_pb", "DL1r_ptau"]].values,
                Y_test,
                d_type="c",
                target_eff=c_eff,
                frac=fb_value,
                taufrac=ftauforc_value,
                use_taus=bool_use_taus,
            )
            taurej_arr_dl1r.append(taurej_i)
            taurej_arr_dl1rC.append(taurej_iC)
        else:
            crej_i, urej_i = utt.GetRejection(
                df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
                Y_test,
                target_eff=b_eff,
                frac=fc_value,
            )
            brej_iC, urej_iC = utt.GetRejection(
                df[["DL1r_pu", "DL1r_pc", "DL1r_pb"]].values,
                Y_test,
                d_type="c",
                target_eff=c_eff,
                frac=fb_value,
            )
        crej_arr_dl1r.append(crej_i)
        urej_arr_dl1r.append(urej_i)
        brej_arr_dl1rC.append(brej_iC)
        urej_arr_dl1rC.append(urej_iC)

        if bool_use_taus:
            crej_i, urej_i, taurej_i = utt.GetRejection(
                df[["rnnip_pu", "rnnip_pc", "rnnip_pb", "rnnip_ptau"]].values,
                Y_test,
                target_eff=b_eff,
                frac=fc_value,
                taufrac=ftauforb_value,
                use_taus=bool_use_taus,
            )
            brej_iC, urej_iC, taurej_iC = utt.GetRejection(
                df[["rnnip_pu", "rnnip_pc", "rnnip_pb", "rnnip_ptau"]].values,
                Y_test,
                d_type="c",
                target_eff=c_eff,
                frac=fb_value,
                taufrac=ftauforc_value,
                use_taus=bool_use_taus,
            )
            taurej_arr_rnnip.append(taurej_i)
            taurej_arr_rnnipC.append(taurej_iC)
        else:
            crej_i, urej_i = utt.GetRejection(
                df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
                Y_test,
                target_eff=b_eff,
                frac=fc_value,
            )
            brej_iC, urej_iC = utt.GetRejection(
                df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
                Y_test,
                d_type="c",
                target_eff=c_eff,
                frac=fb_value,
            )
        crej_arr_rnnip.append(crej_i)
        urej_arr_rnnip.append(urej_i)
        brej_arr_rnnipC.append(brej_iC)
        urej_arr_rnnipC.append(urej_iC)

    if bool_use_taus:
        df_eff_rej = pd.DataFrame(
            {
                "beff": b_effs,
                "ceff": c_effs,
                "umami_crej": crej_arr,
                "umami_urej": urej_arr,
                "umami_taurej": taurej_arr,
                "umami_brejC": brej_arrC,
                "umami_urejC": urej_arrC,
                "umami_taurejC": taurej_arrC,
                "dl1r_crej": crej_arr_dl1r,
                "dl1r_urej": urej_arr_dl1r,
                "dl1r_taurej": taurej_arr_dl1r,
                "dl1r_brejC": brej_arr_dl1rC,
                "dl1r_urejC": urej_arr_dl1rC,
                "dl1r_taurejC": taurej_arr_dl1rC,
                "rnnip_crej": crej_arr_rnnip,
                "rnnip_urej": urej_arr_rnnip,
                "rnnip_taurej": taurej_arr_rnnip,
                "rnnip_brejC": brej_arr_rnnipC,
                "rnnip_urejC": urej_arr_rnnipC,
                "rnnip_taurejC": taurej_arr_rnnipC,
            }
        )
    else:
        df_eff_rej = pd.DataFrame(
            {
                "beff": b_effs,
                "ceff": c_effs,
                "umami_crej": crej_arr,
                "umami_urej": urej_arr,
                "umami_brejC": brej_arrC,
                "umami_urejC": urej_arrC,
                "dl1r_crej": crej_arr_dl1r,
                "dl1r_urej": urej_arr_dl1r,
                "dl1r_brejC": brej_arr_dl1rC,
                "dl1r_urejC": urej_arr_dl1rC,
                "rnnip_crej": crej_arr_rnnip,
                "rnnip_urej": urej_arr_rnnip,
                "rnnip_brejC": brej_arr_rnnipC,
                "rnnip_urejC": urej_arr_rnnipC,
            }
        )
    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff"
        f"-{args.epoch}.h5",
        data_set_name,
    )
    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binominal errors
    f = h5py.File(
        f"{train_config.model_name}/results/"
        + f"results-rej_per_eff-{args.epoch}.h5",
        "a",
    )
    f.attrs["N_test"] = len(df)
    f.close()

    if not bool_use_taus:
        return

    print("calculating rejections per frac for beff of 70% and ceff of 40%")
    target_beff = 0.7
    target_ceff = 0.4
    # The first two must have same number of element
    c_fracs = np.linspace(0.005, 0.1, 21)
    b_fracs = np.linspace(0.1, 0.3, 21)
    tau_fracs = np.linspace(0.005, 0.9, 41)
    crej_arr = []
    urej_arr = []
    taurej_arr = []
    brejC_arr = []
    urejC_arr = []
    taurejC_arr = []
    fraction_taus = []
    fraction_c = []
    fraction_b = []
    for ind, c_frac in enumerate(c_fracs):
        b_frac = b_fracs[ind]
        if ind % (len(c_fracs) // 5) == 0:
            print("{} % done".format(ind // (len(c_fracs) // 5) * 20))
        for tau_frac in tau_fracs:
            crej_i, urej_i, taurej_i = utt.GetRejection(
                pred,
                Y_test,
                target_eff=target_beff,
                frac=c_frac,
                taufrac=tau_frac,
                use_taus=bool_use_taus,
            )
            brejC_i, urejC_i, taurejC_i = utt.GetRejection(
                pred,
                Y_test,
                d_type="c",
                target_eff=target_ceff,
                frac=b_frac,
                taufrac=tau_frac,
                use_taus=bool_use_taus,
            )
            taurej_arr.append(taurej_i)
            crej_arr.append(crej_i)
            urej_arr.append(urej_i)
            taurejC_arr.append(taurejC_i)
            brejC_arr.append(brejC_i)
            urejC_arr.append(urejC_i)
            fraction_taus.append(tau_frac)
            fraction_c.append(c_frac)
            fraction_b.append(b_frac)

    df_frac_rej = pd.DataFrame(
        {
            "fraction_c": fraction_c,
            "fraction_b": fraction_b,
            "fraction_taus": fraction_taus,
            "umami_crej": crej_arr,
            "umami_urej": urej_arr,
            "umami_taurej": taurej_arr,
            "umami_brejC": brejC_arr,
            "umami_urejC": urejC_arr,
            "umami_taurejC": taurejC_arr,
        }
    )
    df_frac_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_frac"
        f"-{args.epoch}.h5",
        data_set_name,
    )


if __name__ == "__main__":
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    if args.dl1:
        print("Start evaluating DL1 with ttbar test files...")
        for ttbar_models in train_config.ttbar_test_files:
            EvaluateModelDL1(
                args,
                train_config,
                preprocess_config,
                train_config.ttbar_test_files[ttbar_models]["Path"],
                train_config.ttbar_test_files[ttbar_models]["data_set_name"],
            )

        print("Start evaluating DL1 with Z' test files...")
        for zpext_models in train_config.zpext_test_files:
            EvaluateModelDL1(
                args,
                train_config,
                preprocess_config,
                train_config.zpext_test_files[zpext_models]["Path"],
                train_config.zpext_test_files[zpext_models]["data_set_name"],
            )

    elif args.dips:
        print("Start evaluating DIPS with ttbar test files...")
        for ttbar_models in train_config.ttbar_test_files:
            EvaluateModelDips(
                args,
                train_config,
                preprocess_config,
                train_config.ttbar_test_files[ttbar_models]["Path"],
                train_config.ttbar_test_files[ttbar_models]["data_set_name"],
            )

        print("Start evaluating DIPS with Z' test files...")
        for zpext_models in train_config.zpext_test_files:
            EvaluateModelDips(
                args,
                train_config,
                preprocess_config,
                train_config.zpext_test_files[zpext_models]["Path"],
                train_config.zpext_test_files[zpext_models]["data_set_name"],
            )

    else:
        print("Start evaluating UMAMI with ttbar test files...")
        for ttbar_models in train_config.ttbar_test_files:
            EvaluateModel(
                args,
                train_config,
                preprocess_config,
                train_config.ttbar_test_files[ttbar_models]["Path"],
                train_config.ttbar_test_files[ttbar_models]["data_set_name"],
            )

        print("Start evaluating UMAMI with Z' test files...")
        for zpext_models in train_config.zpext_test_files:
            EvaluateModel(
                args,
                train_config,
                preprocess_config,
                train_config.zpext_test_files[zpext_models]["Path"],
                train_config.zpext_test_files[zpext_models]["data_set_name"],
            )
