from umami.configuration import global_config, logger  # isort:skip
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
    Eval_params = train_config.Eval_parameters_validation

    # Set number of nJets for testing
    if args.nJets is None:
        nJets = int(Eval_params["n_jets"])

    else:
        nJets = args.nJets

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]
        fc_list = Eval_params["fc_values_comp"]

    else:
        tagger_list = Eval_params["tagger"]
        fc_list = Eval_params["fc_values_comp"]

    # Get model file path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    logger.info(f"Evaluating {model_file}")

    # Define excluded variables and laod them
    exclude = []
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Load the test jets
    X_test, X_test_trk, Y_test = utt.GetTestFile(
        test_file,
        train_config.var_dict,
        preprocess_config,
        nJets=nJets,
        exclude=exclude,
    )

    # Load the model for evaluation. Note: The Sum is needed here!
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    # Predict the output of the model on the test jets
    pred_dips, pred_umami = model.predict(
        [X_test_trk, X_test], batch_size=5000, verbose=0
    )

    # Get truth labelling
    y_true = np.argmax(Y_test, axis=1)

    # Set flavour indicies
    b_index, c_index, u_index = 2, 1, 0

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "HadronConeExclTruthLabelID",
    ]

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += [f"{tagger}_pb", f"{tagger}_pc", f"{tagger}_pu"]

    # Load the jets with selected variables
    df = pd.DataFrame(h5py.File(test_file, "r")["/jets"][:nJets][variables])
    logger.info(f"Jets used for testing: {len(df)}")

    # Exclude all flavors that are not light, c or b
    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

    # Define new dict with the evaluation results
    df_discs_dict = {
        "umami_pb": pred_umami[:, b_index],
        "umami_pc": pred_umami[:, c_index],
        "umami_pu": pred_umami[:, u_index],
        "dips_pb": pred_dips[:, b_index],
        "dips_pc": pred_dips[:, c_index],
        "dips_pu": pred_dips[:, u_index],
        "pt": df[global_config.pTvariable],
        "eta": df[global_config.etavariable],
        "labels": y_true,
    }

    # Calculate dics values and add them to the dict
    for tagger in tagger_list:
        df_discs_dict.update(
            {
                f"{tagger}_pb": df[f"{tagger}_pb"],
                f"{tagger}_pc": df[f"{tagger}_pc"],
                f"{tagger}_pu": df[f"{tagger}_pu"],
                f"disc_{tagger}": GetScore(
                    df[f"{tagger}_pb"],
                    df[f"{tagger}_pc"],
                    df[f"{tagger}_pu"],
                ),
            }
        )

    # Add dict to Dataframe and delete dict
    df_discs = pd.DataFrame(df_discs_dict)
    del df_discs_dict

    # Create results dir
    os.makedirs(f"{train_config.model_name}/results", exist_ok=True)

    # Save dataframe to h5
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    # Define granularity of x
    x_axis_granularity = 100

    logger.info("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, x_axis_granularity)

    crej_dict = {
        "umami": [],
        "dips": [],
    }

    urej_dict = {
        "umami": [],
        "dips": [],
    }

    # Add taggers to dicts
    for tagger in tagger_list:
        crej_dict.update({tagger: []})
        urej_dict.update({tagger: []})

    # Loop over effs for ROC plots
    for eff in b_effs:
        # Add the rejections from the trained models
        # Umami
        crej_i, urej_i = utt.GetRejection(
            pred_umami, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_dict["umami"].append(crej_i)
        urej_dict["umami"].append(urej_i)

        # Dips part
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_dict["dips"].append(crej_i)
        urej_dict["dips"].append(urej_i)

        for tagger in tagger_list:
            crej_tmp, urej_tmp = utt.GetRejection(
                df[[f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]].values,
                Y_test,
                target_eff=eff,
                frac=fc_list[tagger],
            )

            crej_dict[tagger].append(crej_tmp)
            urej_dict[tagger].append(urej_tmp)

    logger.info("calculating rejections per fc")
    fc_values = np.linspace(0.001, 0.1, x_axis_granularity)

    crej_cfrac_dict = {
        "umami": [],
        "dips": [],
    }

    urej_cfrac_dict = {
        "umami": [],
        "dips": [],
    }

    for tagger in tagger_list:
        crej_cfrac_dict.update({tagger: []})
        urej_cfrac_dict.update({tagger: []})

    for fc in fc_values:
        # Add the rejections from the trained models
        # Umami
        crej_i, urej_i = utt.GetRejection(
            pred_umami, Y_test, target_eff=args.beff, frac=fc
        )
        crej_cfrac_dict["umami"].append(crej_i)
        urej_cfrac_dict["umami"].append(urej_i)

        # Dips part
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=args.beff, frac=fc
        )
        crej_cfrac_dict["dips"].append(crej_i)
        urej_cfrac_dict["dips"].append(urej_i)

        for tagger in tagger_list:
            crej_tmp, urej_tmp = utt.GetRejection(
                df[[f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]].values,
                Y_test,
                target_eff=args.beff,
                frac=fc,
            )

            crej_cfrac_dict[tagger].append(crej_tmp)
            urej_cfrac_dict[tagger].append(urej_tmp)

    df_eff_rej_dict = {
        "beff": b_effs,
        "umami_crej": crej_dict["umami"],
        "umami_urej": urej_dict["umami"],
        "dips_crej": crej_dict["dips"],
        "dips_urej": urej_dict["dips"],
        "fc_values": fc_values,
        "umami_cfrac_crej": crej_cfrac_dict["umami"],
        "umami_cfrac_urej": urej_cfrac_dict["umami"],
        "dips_cfrac_crej": crej_cfrac_dict["dips"],
        "dips_cfrac_urej": urej_cfrac_dict["dips"],
    }

    for tagger in tagger_list:
        df_eff_rej_dict.update(
            {
                f"{tagger}_crej": crej_dict[tagger],
                f"{tagger}_urej": urej_dict[tagger],
                f"{tagger}_cfrac_crej": crej_cfrac_dict[tagger],
                f"{tagger}_cfrac_urej": urej_cfrac_dict[tagger],
            }
        )

    df_eff_rej = pd.DataFrame(df_eff_rej_dict)
    del df_eff_rej_dict

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
    Eval_params = train_config.Eval_parameters_validation

    # Set number of nJets for testing
    if args.nJets is None:
        nJets = int(Eval_params["n_jets"])

    else:
        nJets = args.nJets

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]
        fc_list = Eval_params["fc_values_comp"]

    else:
        tagger_list = Eval_params["tagger"]
        fc_list = Eval_params["fc_values_comp"]

    # Get model file path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    logger.info(f"Evaluating {model_file}")

    # Get the testfile with the needed configs
    X_test_trk, Y_test = utt.GetTestSampleTrks(
        test_file,
        train_config.var_dict,
        preprocess_config,
        nJets=nJets,
    )

    # Load the model for evaluation. Note: The Sum is needed here!
    with CustomObjectScope({"Sum": utt.Sum}):
        model = load_model(model_file)

    # Get predictions from trained model
    pred_dips = model.predict(
        X_test_trk,
        batch_size=train_config.NN_structure["batch_size"],
        verbose=0,
    )

    # Get truth labelling
    y_true = np.argmax(Y_test, axis=1)

    # Set flavour indicies
    b_index, c_index, u_index = 2, 1, 0

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "HadronConeExclTruthLabelID",
    ]

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += [f"{tagger}_pb", f"{tagger}_pc", f"{tagger}_pu"]

    # Load the jets with selected variables
    df = pd.DataFrame(h5py.File(test_file, "r")["/jets"][:nJets][variables])
    logger.info(f"Jets used for testing: {len(df)}")

    # Exclude all flavours that are not light, c or b
    df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

    # Define new dict with the evaluation results
    df_discs_dict = {
        "dips_pb": pred_dips[:, b_index],
        "dips_pc": pred_dips[:, c_index],
        "dips_pu": pred_dips[:, u_index],
        "pt": df[global_config.pTvariable],
        "eta": df[global_config.etavariable],
        "labels": y_true,
    }

    # Calculate dics values and add them to the dict
    for tagger in tagger_list:
        df_discs_dict.update(
            {
                f"{tagger}_pb": df[f"{tagger}_pb"],
                f"{tagger}_pc": df[f"{tagger}_pc"],
                f"{tagger}_pu": df[f"{tagger}_pu"],
                f"disc_{tagger}": GetScore(
                    df[f"{tagger}_pb"],
                    df[f"{tagger}_pc"],
                    df[f"{tagger}_pu"],
                ),
            }
        )

    # Add dict to Dataframe and delete dict
    df_discs = pd.DataFrame(df_discs_dict)
    del df_discs_dict

    # Create results dir
    os.system(f"mkdir -p {train_config.model_name}/results")

    # Save dataframe to h5
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    # Define granularity of x
    x_axis_granularity = 100

    logger.info("calculating rejections per efficiency")
    b_effs = np.linspace(0.39, 1, x_axis_granularity)

    crej_dict = {
        "dips": [],
    }

    urej_dict = {
        "dips": [],
    }

    # Add taggers to dicts
    for tagger in tagger_list:
        crej_dict.update({tagger: []})
        urej_dict.update({tagger: []})

    # Loop over effs for ROC plots
    for eff in b_effs:
        # Add the rejections from the trained models
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=eff, frac=args.cfrac
        )
        crej_dict["dips"].append(crej_i)
        urej_dict["dips"].append(urej_i)

        for tagger in tagger_list:
            crej_tmp, urej_tmp = utt.GetRejection(
                df[[f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]].values,
                Y_test,
                target_eff=eff,
                frac=fc_list[tagger],
            )

            crej_dict[tagger].append(crej_tmp)
            urej_dict[tagger].append(urej_tmp)

    logger.info("calculating rejections per fc")
    fc_values = np.linspace(0.001, 0.1, x_axis_granularity)

    crej_cfrac_dict = {
        "dips": [],
    }

    urej_cfrac_dict = {
        "dips": [],
    }

    for tagger in tagger_list:
        crej_cfrac_dict.update({tagger: []})
        urej_cfrac_dict.update({tagger: []})

    for fc in fc_values:
        # Add the rejections from the trained models
        crej_i, urej_i = utt.GetRejection(
            pred_dips, Y_test, target_eff=args.beff, frac=fc
        )
        crej_cfrac_dict["dips"].append(crej_i)
        urej_cfrac_dict["dips"].append(urej_i)

        for tagger in tagger_list:
            crej_tmp, urej_tmp = utt.GetRejection(
                df[[f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]].values,
                Y_test,
                target_eff=args.beff,
                frac=fc,
            )

            crej_cfrac_dict[tagger].append(crej_tmp)
            urej_cfrac_dict[tagger].append(urej_tmp)

    df_eff_rej_dict = {
        "beff": b_effs,
        "dips_crej": crej_dict["dips"],
        "dips_urej": urej_dict["dips"],
        "fc_values": fc_values,
        "dips_cfrac_crej": crej_cfrac_dict["dips"],
        "dips_cfrac_urej": urej_cfrac_dict["dips"],
    }

    for tagger in tagger_list:
        df_eff_rej_dict.update(
            {
                f"{tagger}_crej": crej_dict[tagger],
                f"{tagger}_urej": urej_dict[tagger],
                f"{tagger}_cfrac_crej": crej_cfrac_dict[tagger],
                f"{tagger}_cfrac_urej": urej_cfrac_dict[tagger],
            }
        )

    df_eff_rej = pd.DataFrame(df_eff_rej_dict)
    del df_eff_rej_dict

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

    logger.info("Calculate gradients for inputs")
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
    Eval_params = train_config.Eval_parameters_validation

    # Get model file path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    logger.info(f"Evaluating {model_file}")

    # Set fractions
    fc_value = Eval_params["fc_value"]
    fb_value = Eval_params["fb_value"]
    if "ftauforb_value" in Eval_params:
        ftauforb_value = Eval_params["ftauforb_value"]
    else:
        ftauforb_value = None
    if "ftauforc_value" in Eval_params:
        ftauforc_value = Eval_params["ftauforc_value"]
    else:
        ftauforc_value = None

    # Manage taus
    bool_use_taus = (
        train_config.bool_use_taus and preprocess_config.bool_process_taus
    )
    if bool_use_taus:
        logger.info(f"Evaluating {model_file} with taus")
    else:
        logger.info(f"Evaluating {model_file}")
        ftauforc_value = None
        ftauforb_value = None

    # Set number of nJets for testing
    if args.nJets is None:
        nJets = int(Eval_params["n_jets"])

    else:
        nJets = args.nJets

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]
        fc_list = Eval_params["fc_values_comp"]
        fb_list = Eval_params["fb_values_comp"]

    else:
        tagger_list = Eval_params["tagger"]
        fc_list = Eval_params["fc_values_comp"]
        fb_list = Eval_params["fb_values_comp"]

    # Define excluded variables and laod them
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Load the test jets
    X_test, Y_test = utt.GetTestSample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        nJets=nJets,
        use_taus=bool_use_taus,
        exclude=exclude,
    )

    # Load the model for evaluation.
    model = load_model(model_file)

    # Predict the output of the model on the test jets
    pred = model.predict(X_test, batch_size=5000, verbose=0)

    # Get truth labelling
    y_true = np.argmax(Y_test, axis=1)

    # Set flavour indicies
    tau_index, b_index, c_index, u_index = 3, 2, 1, 0

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
        "HadronConeExclTruthLabelID",
    ]

    # Load extra variables
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
                logger.info(f"Variable '{item}' not available")
        variables.extend(add_variables_available)

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += [f"{tagger}_pb", f"{tagger}_pc", f"{tagger}_pu"]

    # Load the jets with selected variables
    df = pd.DataFrame(h5py.File(test_file, "r")["/jets"][:nJets][variables])
    logger.info(f"Jets used for testing: {len(df)}")

    if bool_use_taus:
        # Exclude all flavors that are not light, c, b or tau
        df.query("HadronConeExclTruthLabelID in [0, 4, 5, 15]", inplace=True)

        # Define new dict with the evaluation results
        df_discs_dict = {
            "umami_ptau": pred[:, tau_index],
            "umami_pb": pred[:, b_index],
            "umami_pc": pred[:, c_index],
            "umami_pu": pred[:, u_index],
            "pt": df[global_config.pTvariable],
            "eta": df[global_config.etavariable],
            "labels": y_true,
        }

        # Calculate dics values and add them to the dict
        for tagger in tagger_list:
            if f"{tagger}_ptau" not in df:
                df[f"{tagger}_ptau"] = 0

            df_discs_dict.update(
                {
                    f"{tagger}_pb": df[f"{tagger}_pb"],
                    f"{tagger}_pc": df[f"{tagger}_pc"],
                    f"{tagger}_pu": df[f"{tagger}_pu"],
                    f"{tagger}_ptau": df[f"{tagger}_ptau"],
                    f"disc_{tagger}": GetScore(
                        df[f"{tagger}_pb"],
                        df[f"{tagger}_pc"],
                        df[f"{tagger}_pu"],
                        df[f"{tagger}_ptau"],
                        fc=fc_list[tagger],
                        ftau=ftauforb_value,
                    ),
                    f"disc_{tagger}C": GetScoreC(
                        df[f"{tagger}_pb"],
                        df[f"{tagger}_pc"],
                        df[f"{tagger}_pu"],
                        df[f"{tagger}_ptau"],
                        fb=fb_list[tagger],
                        ftau=ftauforc_value,
                    ),
                }
            )

    else:
        # Exclude all flavors that are not light, c, b or tau
        df.query("HadronConeExclTruthLabelID <= 5", inplace=True)

        # Define new dict with the evaluation results
        df_discs_dict = {
            "umami_pb": pred[:, b_index],
            "umami_pc": pred[:, c_index],
            "umami_pu": pred[:, u_index],
            "pt": df[global_config.pTvariable],
            "eta": df[global_config.etavariable],
            "labels": y_true,
        }

        # Calculate dics values and add them to the dict
        for tagger in tagger_list:
            df_discs_dict.update(
                {
                    f"{tagger}_pb": df[f"{tagger}_pb"],
                    f"{tagger}_pc": df[f"{tagger}_pc"],
                    f"{tagger}_pu": df[f"{tagger}_pu"],
                    f"disc_{tagger}": GetScore(
                        df[f"{tagger}_pb"],
                        df[f"{tagger}_pc"],
                        df[f"{tagger}_pu"],
                        fc=fc_list[tagger],
                        ftau=ftauforb_value,
                    ),
                    f"disc_{tagger}C": GetScoreC(
                        df[f"{tagger}_pb"],
                        df[f"{tagger}_pc"],
                        df[f"{tagger}_pu"],
                        fb=fb_list[tagger],
                        ftau=ftauforc_value,
                    ),
                }
            )

    # Add dict to Dataframe and delete dict
    df_discs = pd.DataFrame(df_discs_dict)
    del df_discs_dict

    # Adding extra variables if available
    if add_variables_available is not None:
        for item in add_variables_available:
            logger.info(f"Adding {item}")
            df_discs[item] = df[item]

    # Create results dir
    os.system(f"mkdir -p {train_config.model_name}/results")

    # Save dataframe to h5
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    logger.info("calculating rejections per efficiency")

    b_effs = np.linspace(0.39, 1, 150)
    c_effs = np.linspace(0.09, 1, 150)

    crej_dict = {
        "umami": [],
    }

    brej_dict = {
        "umamiC": [],
    }

    urej_dict = {
        "umami": [],
        "umamiC": [],
    }

    if bool_use_taus:
        taurej_dict = {
            "umami": [],
            "umamiC": [],
        }

    # Add taggers to dicts
    for tagger in tagger_list:
        crej_dict.update({f"{tagger}": []})
        urej_dict.update({f"{tagger}": []})
        brej_dict.update({f"{tagger}C": []})
        urej_dict.update({f"{tagger}C": []})

        if bool_use_taus:
            taurej_dict.update({f"{tagger}": []})
            taurej_dict.update({f"{tagger}C": []})

    # Loop over effs for ROC plots
    for ind_eff, b_eff in enumerate(b_effs):
        c_eff = c_effs[ind_eff]

        # Add the rejections from the trained models
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
            taurej_dict["umami"].append(taurej_i)
            taurej_dict["umamiC"].append(taurej_iC)

        else:
            crej_i, urej_i = utt.GetRejection(
                pred, Y_test, target_eff=b_eff, frac=fc_value
            )
            brej_iC, urej_iC = utt.GetRejection(
                pred, Y_test, d_type="c", target_eff=c_eff, frac=fb_value
            )

        crej_dict["umami"].append(crej_i)
        urej_dict["umami"].append(urej_i)
        brej_dict["umamiC"].append(brej_iC)
        urej_dict["umamiC"].append(urej_iC)

        for tagger in tagger_list:
            # Add the comparison taggers
            if bool_use_taus:
                crej_i, urej_i, taurej_i = utt.GetRejection(
                    df[
                        [
                            f"{tagger}_pu",
                            f"{tagger}_pc",
                            f"{tagger}_pb",
                            f"{tagger}_ptau",
                        ]
                    ].values,
                    Y_test,
                    target_eff=b_eff,
                    frac=fc_list[tagger],
                    taufrac=ftauforb_value,
                    use_taus=bool_use_taus,
                )
                brej_iC, urej_iC, taurej_iC = utt.GetRejection(
                    df[
                        [
                            f"{tagger}_pu",
                            f"{tagger}_pc",
                            f"{tagger}_pb",
                            f"{tagger}_ptau",
                        ]
                    ].values,
                    Y_test,
                    d_type="c",
                    target_eff=c_eff,
                    frac=fb_list[tagger],
                    taufrac=ftauforc_value,
                    use_taus=bool_use_taus,
                )
                taurej_dict[f"{tagger}"].append(taurej_i)
                taurej_dict[f"{tagger}C"].append(taurej_iC)
            else:
                crej_i, urej_i = utt.GetRejection(
                    df[
                        [f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]
                    ].values,
                    Y_test,
                    target_eff=b_eff,
                    frac=fc_list[tagger],
                )
                brej_iC, urej_iC = utt.GetRejection(
                    df[
                        [f"{tagger}_pu", f"{tagger}_pc", f"{tagger}_pb"]
                    ].values,
                    Y_test,
                    d_type="c",
                    target_eff=c_eff,
                    frac=fb_list[tagger],
                )
            crej_dict[f"{tagger}"].append(crej_i)
            urej_dict[f"{tagger}"].append(urej_i)
            brej_dict[f"{tagger}C"].append(brej_iC)
            urej_dict[f"{tagger}C"].append(urej_iC)

    if bool_use_taus:
        df_eff_rej_dict = {
            "beff": b_effs,
            "ceff": c_effs,
            "umami_crej": crej_dict["umami"],
            "umami_urej": urej_dict["umami"],
            "umami_taurej": taurej_dict["umami"],
            "umami_brejC": brej_dict["umamiC"],
            "umami_urejC": urej_dict["umamiC"],
            "umami_taurejC": taurej_dict["umamiC"],
        }

        for tagger in tagger_list:
            df_eff_rej_dict.update(
                {
                    f"{tagger}_crej": crej_dict[f"{tagger}"],
                    f"{tagger}_urej": urej_dict[f"{tagger}"],
                    f"{tagger}_taurej": taurej_dict[f"{tagger}"],
                    f"{tagger}_brejC": brej_dict[f"{tagger}C"],
                    f"{tagger}_urejC": urej_dict[f"{tagger}C"],
                    f"{tagger}_taurejC": taurej_dict[f"{tagger}C"],
                }
            )

    else:
        df_eff_rej_dict = {
            "beff": b_effs,
            "ceff": c_effs,
            "umami_crej": crej_dict["umami"],
            "umami_urej": urej_dict["umami"],
            "umami_brejC": brej_dict["umamiC"],
            "umami_urejC": urej_dict["umamiC"],
        }

        for tagger in tagger_list:
            df_eff_rej_dict.update(
                {
                    f"{tagger}_crej": crej_dict[f"{tagger}"],
                    f"{tagger}_urej": urej_dict[f"{tagger}"],
                    f"{tagger}_brejC": brej_dict[f"{tagger}C"],
                    f"{tagger}_urejC": urej_dict[f"{tagger}C"],
                }
            )

    # Add dict to Dataframe and delete dict
    df_eff_rej = pd.DataFrame(df_eff_rej_dict)
    del df_eff_rej_dict

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

    if not bool_use_taus:
        return

    logger.info(
        "calculating rejections per frac for beff of 70% and ceff of 40%"
    )
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
            logger.info(f"{ind // (len(c_fracs) // 5) * 20} % done")
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
        if train_config.ttbar_test_files is not None:
            logger.info("Start evaluating DL1 with ttbar test files...")
            for ttbar_models in train_config.ttbar_test_files:
                EvaluateModelDL1(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.ttbar_test_files[ttbar_models]["Path"],
                    train_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                )

        if train_config.zpext_test_files is not None:
            logger.info("Start evaluating DL1 with Z' test files...")
            for zpext_models in train_config.zpext_test_files:
                EvaluateModelDL1(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.zpext_test_files[zpext_models]["Path"],
                    train_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                )

    elif args.dips:
        if train_config.ttbar_test_files is not None:
            logger.info("Start evaluating DIPS with ttbar test files...")
            for ttbar_models in train_config.ttbar_test_files:
                EvaluateModelDips(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.ttbar_test_files[ttbar_models]["Path"],
                    train_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                )

        if train_config.zpext_test_files is not None:
            logger.info("Start evaluating DIPS with Z' test files...")
            for zpext_models in train_config.zpext_test_files:
                EvaluateModelDips(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.zpext_test_files[zpext_models]["Path"],
                    train_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                )

    else:
        if train_config.zpext_test_files is not None:
            logger.info("Start evaluating UMAMI with ttbar test files...")
            for ttbar_models in train_config.ttbar_test_files:
                EvaluateModel(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.ttbar_test_files[ttbar_models]["Path"],
                    train_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                )

        if train_config.zpext_test_files is not None:
            logger.info("Start evaluating UMAMI with Z' test files...")
            for zpext_models in train_config.zpext_test_files:
                EvaluateModel(
                    args,
                    train_config,
                    preprocess_config,
                    train_config.zpext_test_files[zpext_models]["Path"],
                    train_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                )
