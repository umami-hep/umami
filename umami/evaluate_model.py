from umami.configuration import global_config, logger  # isort:skip
import argparse
import os
import pickle

import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.evaluation_tools as uet
import umami.tf_tools as utf
import umami.train_tools as utt
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
        help="""Number of jets used for the testing. By default it will
        use all available jets in the test files.""",
    )

    parser.add_argument(
        "--beff",
        type=float,
        help="b efficiency used for the charm fraction scan",
    )

    args = parser.parse_args()
    return args


def EvaluateModel(
    args, train_config, preprocess_config, test_file, data_set_name
):
    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]

    # Init the placeholder lists for tagger_names
    tagger_names = []
    tagger_preds = []

    # Set number of nJets for testing
    nJets = int(Eval_params["n_jets"]) if not args.nJets else args.nJets

    # Check the config if the trained model is also to be evaluated
    try:
        Eval_model_bool = train_config.evaluate_trained_model

    except AttributeError:
        Eval_model_bool = True

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]

    else:
        tagger_list = Eval_params["tagger"]

    if Eval_model_bool is True:
        if args.epoch is None:
            raise ValueError(
                "You need to give an epoch which is to be evaluated!"
            )

        # Get model file path
        model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
        logger.info(f"Evaluating {model_file}")

        # Define excluded variables and laod them
        exclude = None
        if "exclude" in train_config.config:
            exclude = train_config.config["exclude"]

        # Load the test jets
        X_test, X_test_trk, Y_test = utt.GetTestFile(
            file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            nJets=nJets,
            exclude=exclude,
        )

        # Load the model for evaluation. Note: The Sum is needed here!
        with CustomObjectScope({"Sum": utf.Sum}):
            model = load_model(model_file)

        # Predict the output of the model on the test jets
        pred_dips, pred_umami = model.predict(
            [X_test_trk, X_test], batch_size=5000, verbose=0
        )

        # Fill the tagger_names and tagger_preds
        tagger_names = ["dips", "umami"]
        tagger_preds = [pred_dips, pred_umami]

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = utt.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += utt.get_class_prob_var_names(
            tagger_name=f"{tagger}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = utt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
    )

    # Get the discriminant values and probabilities of each tagger
    # for each jet
    df_discs_dict = uet.GetScoresProbsDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=tagger_preds,
        tagger_names=tagger_names,
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={
            "dips": Eval_params["frac_values"]["dips"],
            "umami": Eval_params["frac_values"]["umami"],
        }
        if tagger_preds != []
        else None,
        frac_values_comp=frac_values_comp,
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

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.GetRejectionPerEfficiencyDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=tagger_preds,
        tagger_names=tagger_names,
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={
            "dips": Eval_params["frac_values"]["dips"],
            "umami": Eval_params["frac_values"]["umami"],
        }
        if tagger_preds != []
        else None,
        frac_values_comp=frac_values_comp,
        eff_min=0.49
        if "eff_min" not in Eval_params
        else Eval_params["eff_min"],
        eff_max=1.0
        if "eff_max" not in Eval_params
        else Eval_params["eff_max"],
    )

    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

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
    f.attrs["N_test"] = len(jets)
    f.close()


def EvaluateModelDips(
    args, train_config, preprocess_config, test_file, data_set_name
):
    # Check if epochs are set
    if args.epoch is None:
        raise ValueError("You need to give an epoch which is to be evaluated!")

    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]

    # Set number of nJets for testing
    nJets = int(Eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]

    else:
        tagger_list = Eval_params["tagger"]

    # Get model file path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    logger.info(f"Evaluating {model_file}")

    # Get the testfile with the needed configs
    X_test_trk, Y_test = utt.GetTestSampleTrks(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=nJets,
    )

    # Load the model for evaluation. Note: The Sum is needed here!
    with CustomObjectScope({"Sum": utf.Sum}):
        model = load_model(model_file)

    # Get predictions from trained model
    pred_dips = model.predict(
        X_test_trk,
        batch_size=train_config.NN_structure["batch_size"],
        verbose=0,
    )

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = utt.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += utt.get_class_prob_var_names(
            tagger_name=f"{tagger}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = utt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
    )

    # Get the discriminant values and probabilities of each tagger
    # for each jet
    df_discs_dict = uet.GetScoresProbsDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dips],
        tagger_names=["dips"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"dips": Eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
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

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.GetRejectionPerEfficiencyDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dips],
        tagger_names=["dips"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"dips": Eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
        eff_min=0.49
        if "eff_min" not in Eval_params
        else Eval_params["eff_min"],
        eff_max=1.0
        if "eff_max" not in Eval_params
        else Eval_params["eff_max"],
    )

    # Form the dict to a Dataframe and save it
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

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
    f.attrs["N_test"] = len(jets)
    f.close()

    # Get the saliency map dict
    saliency_map_dict = uet.GetSaliencyMapDict(
        model=model,
        model_pred=pred_dips,
        X_test=X_test_trk,
        Y_test=Y_test,
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=Eval_params["frac_values"],
    )

    # Create results dir and pickle file
    os.system(f"mkdir -p {train_config.model_name}/results")
    with open(
        f"{train_config.model_name}/results/saliency_{args.epoch}"
        + f"_{data_set_name}.pkl",
        "wb",
    ) as f:
        pickle.dump(saliency_map_dict, f)


def EvaluateModelDL1(
    args, train_config, preprocess_config, test_file, data_set_name
):
    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]

    # Check if epochs are set or not
    if args.epoch is None:
        raise ValueError("You need to give an epoch which is to be evaluated!")

    # Set number of nJets for testing
    nJets = int(Eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if type(Eval_params["tagger"]) is not list:
        tagger_list = [Eval_params["tagger"]]

    else:
        tagger_list = Eval_params["tagger"]

    # Get model file path
    model_file = f"{train_config.model_name}/model_epoch{args.epoch}.h5"
    logger.info(f"Evaluating {model_file}")

    # Define excluded variables and laod them
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Get the testfile with the needed configs
    X_test, Y_test = utt.GetTestSample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=nJets,
        exclude=exclude,
    )

    # Load the model for evaluation.
    model = load_model(model_file)

    # Predict the output of the model on the test jets
    pred_DL1 = model.predict(
        X_test,
        batch_size=train_config.NN_structure["batch_size"],
        verbose=0,
    )

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = utt.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += utt.get_class_prob_var_names(
            tagger_name=f"{tagger}", class_labels=class_labels
        )

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

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = utt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
    )

    # Get the discriminant values and probabilities of each tagger
    # for each jet
    df_discs_dict = uet.GetScoresProbsDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_DL1],
        tagger_names=["DL1"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"DL1": Eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
    )

    # Add dict to Dataframe and delete dict
    df_discs = pd.DataFrame(df_discs_dict)
    del df_discs_dict

    # Adding extra variables if available
    if add_variables_available is not None:
        for item in add_variables_available:
            logger.info(f"Adding {item}")
            df_discs[item] = jets[item]

    # Create results dir
    os.system(f"mkdir -p {train_config.model_name}/results")

    # Save dataframe to h5
    df_discs.to_hdf(
        f"{train_config.model_name}/results/results-{args.epoch}.h5",
        data_set_name,
    )

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.GetRejectionPerEfficiencyDict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_DL1],
        tagger_names=["DL1"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"DL1": Eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
        eff_min=0.49
        if "eff_min" not in Eval_params
        else Eval_params["eff_min"],
        eff_max=1.0
        if "eff_max" not in Eval_params
        else Eval_params["eff_max"],
    )

    # Add dict to Dataframe and delete dict
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

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
    f.attrs["N_test"] = len(jets)
    f.close()

    # TODO Rewrite
    # if not bool_use_taus:
    #     return

    # logger.info(
    #     "calculating rejections per frac for beff of 70% and ceff of 40%"
    # )
    # target_beff = 0.7
    # target_ceff = 0.4
    # # The first two must have same number of element
    # c_fracs = np.linspace(0.005, 0.1, 21)
    # b_fracs = np.linspace(0.1, 0.3, 21)
    # tau_fracs = np.linspace(0.005, 0.9, 41)
    # crej_arr = []
    # urej_arr = []
    # taurej_arr = []
    # brejC_arr = []
    # urejC_arr = []
    # taurejC_arr = []
    # fraction_taus = []
    # fraction_c = []
    # fraction_b = []
    # for ind, c_frac in enumerate(c_fracs):
    #     b_frac = b_fracs[ind]
    #     if ind % (len(c_fracs) // 5) == 0:
    #         logger.info(f"{ind // (len(c_fracs) // 5) * 20} % done")
    #     for tau_frac in tau_fracs:
    #         crej_i, urej_i, taurej_i, _ = utt.GetRejection(
    #             pred,
    #             Y_test,
    #             target_eff=target_beff,
    #             frac=c_frac,
    #             taufrac=tau_frac,
    #             use_taus=bool_use_taus,
    #         )
    #         brejC_i, urejC_i, taurejC_i, _ = utt.GetRejection(
    #             pred,
    #             Y_test,
    #             d_type="c",
    #             target_eff=target_ceff,
    #             frac=b_frac,
    #             taufrac=tau_frac,
    #             use_taus=bool_use_taus,
    #         )
    #         taurej_arr.append(taurej_i)
    #         crej_arr.append(crej_i)
    #         urej_arr.append(urej_i)
    #         taurejC_arr.append(taurejC_i)
    #         brejC_arr.append(brejC_i)
    #         urejC_arr.append(urejC_i)
    #         fraction_taus.append(tau_frac)
    #         fraction_c.append(c_frac)
    #         fraction_b.append(b_frac)

    # df_frac_rej = pd.DataFrame(
    #     {
    #         "fraction_c": fraction_c,
    #         "fraction_b": fraction_b,
    #         "fraction_taus": fraction_taus,
    #         "DL1_crej": crej_arr,
    #         "DL1_urej": urej_arr,
    #         "DL1_taurej": taurej_arr,
    #         "DL1_brejC": brejC_arr,
    #         "DL1_urejC": urejC_arr,
    #         "DL1_taurejC": taurejC_arr,
    #     }
    # )
    # df_frac_rej.to_hdf(
    #     f"{train_config.model_name}/results/results-rej_per_frac"
    #     f"-{args.epoch}.h5",
    #     data_set_name,
    # )


if __name__ == "__main__":
    args = GetParser()
    train_config = utt.Configuration(args.config_file)

    # Check for evaluation only is used
    try:
        train_config.evaluate_trained_model
        preprocess_config = train_config

    except AttributeError:
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

        # if train_config.zpext_test_files is not None:
        #     logger.info("Start evaluating DIPS with Z' test files...")
        #     for zpext_models in train_config.zpext_test_files:
        #         EvaluateModelDips(
        #             args,
        #             train_config,
        #             preprocess_config,
        #             train_config.zpext_test_files[zpext_models]["Path"],
        #             train_config.zpext_test_files[zpext_models][
        #                 "data_set_name"
        #             ],
        #         )

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
