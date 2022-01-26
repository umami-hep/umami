"""Execution script for training model evaluations."""

from umami.configuration import global_config, logger  # isort:skip
import argparse
import os
import pickle

import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model  # pylint: disable=no-name-in-module
from tensorflow.keras.utils import (
    CustomObjectScope,  # pylint: disable=no-name-in-module
)

import umami.classification_tools as uct
import umami.data_tools as udt
import umami.evaluation_tools as uet
import umami.tf_tools as utf
import umami.train_tools as utt
from umami.evaluation_tools import FeatureImportance
from umami.preprocessing_tools import Configuration

# from plottingFunctions import sigBkgEff
tf.compat.v1.disable_eager_execution()


def GetParser():
    """
    Argument parser for Preprocessing script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Preprocessing command line options.")

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
        "-t",
        "--tagger",
        action="store_true",
        help="Decide, which tagger was used and is to be evaluated.",
    )

    parser.add_argument(
        "--nJets",
        type=int,
        help="""Number of jets used for the testing. By default it will
        use all available jets in the test files.""",
    )

    parser.add_argument(
        "--shapley",
        action="store_true",
        help="Calculates feature importance for DL1",
    )

    return parser.parse_args()


def EvaluateModel(
    args: object,
    train_config: object,
    preprocess_config: object,
    test_file: str,
    data_set_name: str,
):
    """
    Evaluate only the taggers in the files or also the UMAMI tagger.

    Parameters
    ----------
    args : object
        Loaded argparser.
    train_config : object
        Loaded train config.
    preprocess_config : object
        Loaded preprocessing config.
    test_file : str
        Path to the files which are to be tested. Wildcards are supported.
    data_set_name : str
        Dataset name for the results files. The results will be saved in
        dicts. The key will be this dataset name.

    Raises
    ------
    ValueError
        If no epoch is given when evaluating UMAMI.
    ValueError
        If the given tagger argument in train config is not a list.
    """

    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]
    tracks_name = train_config.tracks_name
    var_cuts = (
        Eval_params["variable_cuts"][f"{data_set_name}"]
        if "variable_cuts" in Eval_params and Eval_params["variable_cuts"] is not None
        else None
    )

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

    # Set epoch to use for evaluation of trained model or dummy value if
    # tagger scores from derivations should be used
    epoch = args.epoch if Eval_model_bool else 0

    # Test if multiple taggers are given or not
    tagger_list = (
        [Eval_params["tagger"]]
        if isinstance(Eval_params["tagger"], str)
        else Eval_params["tagger"]
    )
    try:
        assert isinstance(tagger_list, list)
    except AssertionError as Error:
        raise ValueError(
            """
            Tagger given in Eval_parameters_validation
            is not a string or a list!
            """
        ) from Error

    # evaluate trained model file (for evaluate_trained_model: True in config)
    if Eval_model_bool:
        if epoch is None:
            raise ValueError("You need to give an epoch which is to be evaluated!")

        # Get model file path
        model_file = utt.GetModelPath(
            model_name=train_config.model_name, epoch=args.epoch
        )
        logger.info(f"Evaluating {model_file}")

        # Define excluded variables and laod them
        exclude = None
        if "exclude" in train_config.config:
            exclude = train_config.config["exclude"]

        # Load the test jets
        X_test, X_test_trk, _ = utt.GetTestFile(
            input_file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            tracks_name=tracks_name,
            nJets=nJets,
            exclude=exclude,
            cut_vars_dict=var_cuts,
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
    label_var_list, _ = uct.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger in tagger_list:
        variables += uct.get_class_prob_var_names(
            tagger_name=f"{tagger}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
        cut_vars_dict=var_cuts,
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
        if tagger_preds != []  # pylint: disable=use-implicit-booleaness-not-comparison
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
        f"{train_config.model_name}/results/results-{epoch}.h5",
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
        if tagger_preds != []  # pylint: disable=use-implicit-booleaness-not-comparison
        else None,
        frac_values_comp=frac_values_comp,
        eff_min=0.49 if "eff_min" not in Eval_params else Eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in Eval_params else Eval_params["eff_max"],
    )

    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff-{epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/" + f"results-rej_per_eff-{epoch}.h5",
        "a",
    ) as f:
        f.attrs["N_test"] = len(jets)


def EvaluateModelDips(
    args: object,
    train_config: object,
    preprocess_config: object,
    test_file: str,
    data_set_name: str,
    tagger: str,
):
    """
    Evaluate the DIPS models.

    Parameters
    ----------
    args : object
        Loaded argparser.
    train_config : object
        Loaded train config.
    preprocess_config : object
        Loaded preprocessing config.
    test_file : str
        Path to the files which are to be tested. Wildcards are supported.
    data_set_name : str
        Dataset name for the results files. The results will be saved in
        dicts. The key will be this dataset name.
    tagger : str
        Name of the tagger that is to be evaluated. Can either be dips or
        dips_cond_att depending which architecture is used.

    Raises
    ------
    ValueError
        If no epoch is given when evaluating.
    ValueError
        If the given tagger argument in train config is neither a
        list nor a string.
    """

    # Check if epochs are set
    if args.epoch is None:
        raise ValueError("You need to give an epoch which is to be evaluated!")

    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]
    tracks_name = train_config.tracks_name
    var_cuts = (
        Eval_params["variable_cuts"][f"{data_set_name}"]
        if "variable_cuts" in Eval_params and Eval_params["variable_cuts"] is not None
        else None
    )

    # Set number of nJets for testing
    nJets = int(Eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if isinstance(Eval_params["tagger"], str):
        tagger_list = [Eval_params["tagger"]]

    elif isinstance(Eval_params["tagger"], list):
        tagger_list = Eval_params["tagger"]

    else:
        raise ValueError(
            """
            Tagger given in Eval_parameters_validation
            is not a string or a list!
            """
        )

    # Get model file path
    model_file = utt.GetModelPath(model_name=train_config.model_name, epoch=args.epoch)
    logger.info(f"Evaluating {model_file}")

    # Check which test files need to be loaded depending on the DIPS version
    if tagger == "dips_cond_att":
        # Load the test jets
        X_test, X_test_trk, Y_test = utt.GetTestFile(
            input_file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            tracks_name=tracks_name,
            nJets=nJets,
            cut_vars_dict=var_cuts,
            jet_variables=[
                global_config.etavariable,
                global_config.pTvariable,
            ],
            print_logger=False,
        )

        # Form the inputs for the network
        X = [X_test_trk, X_test]

    else:
        # Get the testfile with the needed configs
        X, Y_test = utt.GetTestSampleTrks(
            input_file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            tracks_name=tracks_name,
            nJets=nJets,
            cut_vars_dict=var_cuts,
        )

    # Load the model for evaluation with the Custom layers
    # needed for DIPS and DIPS Conditional Attention
    with CustomObjectScope(
        {
            "Sum": utf.Sum,
            "Attention": utf.Attention,
            "DeepSet": utf.DeepSet,
            "AttentionPooling": utf.AttentionPooling,
            "DenseNet": utf.DenseNet,
            "ConditionalAttention": utf.ConditionalAttention,
            "ConditionalDeepSet": utf.ConditionalDeepSet,
        }
    ):
        model = load_model(model_file)

    # Get predictions from trained model
    pred_dips = model.predict(
        X,
        batch_size=train_config.NN_structure["batch_size"],
        verbose=0,
    )

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = uct.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger_i in tagger_list:
        variables += uct.get_class_prob_var_names(
            tagger_name=f"{tagger_i}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
        cut_vars_dict=var_cuts,
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
        eff_min=0.49 if "eff_min" not in Eval_params else Eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in Eval_params else Eval_params["eff_max"],
    )

    # Form the dict to a Dataframe and save it
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/" + f"results-rej_per_eff-{args.epoch}.h5",
        "a",
    ) as f:
        f.attrs["N_test"] = len(jets)

    if (
        "Calculate_Saliency" in Eval_params
        and Eval_params["Calculate_Saliency"] is True
    ):
        # Get the saliency map dict
        saliency_map_dict = uet.GetSaliencyMapDict(
            model=model,
            model_pred=pred_dips,
            X_test=X,
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
    args: object,
    train_config: object,
    preprocess_config: object,
    test_file: str,
    data_set_name: str,
    test_file_entry: str,
):
    """
    Evaluate the various DL1* models.

    Parameters
    ----------
    args : object
        Loaded argparser.
    train_config : object
        Loaded train config.
    preprocess_config : object
        Loaded preprocessing config.
    test_file : str
        Path to the files which are to be tested. Wildcards are supported.
    data_set_name : str
        Dataset name for the results files. The results will be saved in
        dicts. The key will be this dataset name.
    test_file_entry : str
        Name of the test files that are currently used for the SHAPley
        files.

    Raises
    ------
    ValueError
        If no epoch is given when evaluating.
    ValueError
        If the given tagger argument in train config is neither a
        list nor a string.
    """

    # Get train parameters
    Eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = Eval_params["frac_values_comp"]
    var_cuts = (
        Eval_params["variable_cuts"][f"{data_set_name}"]
        if "variable_cuts" in Eval_params and Eval_params["variable_cuts"] is not None
        else None
    )

    # Check if epochs are set or not
    if args.epoch is None:
        raise ValueError("You need to give an epoch which is to be evaluated!")

    # Set number of nJets for testing
    nJets = int(Eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if isinstance(Eval_params["tagger"], str):
        tagger_list = [Eval_params["tagger"]]

    elif isinstance(Eval_params["tagger"], list):
        tagger_list = Eval_params["tagger"]

    else:
        raise ValueError(
            """
            Tagger given in Eval_parameters_validation
            is not a string or a list!
            """
        )

    # Get model file path
    model_file = utt.GetModelPath(model_name=train_config.model_name, epoch=args.epoch)
    logger.info(f"Evaluating {model_file}")

    # Define excluded variables and laod them
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Get the testfile with the needed configs
    X_test, _ = utt.GetTestSample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=nJets,
        exclude=exclude,
        cut_vars_dict=var_cuts,
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
    label_var_list, _ = uct.get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger_i in tagger_list:
        variables += uct.get_class_prob_var_names(
            tagger_name=f"{tagger_i}", class_labels=class_labels
        )

    # Load extra variables
    add_variables = train_config.Eval_parameters_validation["add_variables_eval"]
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
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=nJets,
        variables=variables,
        cut_vars_dict=var_cuts,
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
        eff_min=0.49 if "eff_min" not in Eval_params else Eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in Eval_params else Eval_params["eff_max"],
    )

    # Add dict to Dataframe and delete dict
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/results-rej_per_eff-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/" + f"results-rej_per_eff-{args.epoch}.h5",
        "a",
    ) as f:
        f.attrs["N_test"] = len(jets)

    if args.shapley:
        logger.info("Explaining feature importance with SHAPley")
        FeatureImportance.ShapleyOneFlavor(
            model=model,
            test_data=X_test,
            model_output=Eval_params["shapley"]["model_output"],
            feature_sets=Eval_params["shapley"]["feature_sets"],
            plot_size=Eval_params["shapley"]["plot_size"],
            plot_path=f"{train_config.model_name}/",
            plot_name=test_file_entry + "_shapley_b-jets",
        )

        if Eval_params["shapley"]["bool_all_flavor_plot"]:
            FeatureImportance.ShapleyAllFlavors(
                model=model,
                test_data=X_test,
                feature_sets=Eval_params["shapley"]["feature_sets"],
                averaged_sets=Eval_params["shapley"]["averaged_sets"],
                plot_size=Eval_params["shapley"]["plot_size"],
                plot_path=f"{train_config.model_name}/",
                plot_name=test_file_entry + "_shapley_all_flavors",
            )


if __name__ == "__main__":
    parser_args = GetParser()
    training_config = utt.Configuration(parser_args.config_file)

    # Check for evaluation only is used
    try:
        training_config.evaluate_trained_model  # pylint: disable=pointless-statement
        preprocessing_config = training_config
    except AttributeError:
        preprocessing_config = Configuration(training_config.preprocess_config)

    # Get the tagger from args. If not given, use the one from train config
    if parser_args.tagger:
        tagger_name = parser_args.tagger

    else:
        try:
            tagger_name = training_config.NN_structure["tagger"]

        except KeyError:
            logger.info(
                "No tagger defined. Running evaluation without a freshly trained model!"
            )
            tagger_name = None

    if tagger_name == "dl1":
        if training_config.ttbar_test_files is not None:
            logger.info("Start evaluating DL1 with ttbar test files...")
            for ttbar_models in training_config.ttbar_test_files:
                EvaluateModelDL1(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.ttbar_test_files[ttbar_models]["Path"],
                    data_set_name=training_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                    test_file_entry=ttbar_models,
                )

        if training_config.zpext_test_files is not None:
            logger.info("Start evaluating DL1 with Z' test files...")
            for zpext_models in training_config.zpext_test_files:
                EvaluateModelDL1(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.zpext_test_files[zpext_models]["Path"],
                    data_set_name=training_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                    test_file_entry=zpext_models,
                )

    elif tagger_name in ("dips", "dips_cond_att"):
        if training_config.ttbar_test_files is not None:
            logger.info("Start evaluating DIPS with ttbar test files...")
            for ttbar_models in training_config.ttbar_test_files:
                EvaluateModelDips(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.ttbar_test_files[ttbar_models]["Path"],
                    data_set_name=training_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                    tagger=tagger_name,
                )

        if training_config.zpext_test_files is not None:
            logger.info("Start evaluating DIPS with Z' test files...")
            for zpext_models in training_config.zpext_test_files:
                EvaluateModelDips(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.zpext_test_files[zpext_models]["Path"],
                    data_set_name=training_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                    tagger=tagger_name,
                )

    elif tagger_name == "umami" or tagger_name is None:
        if training_config.ttbar_test_files is not None:
            if tagger_name is None:
                logger.info("Start evaluating taggers in ttbar test files...")

            else:
                logger.info("Start evaluating UMAMI with ttbar test files...")

            for ttbar_models in training_config.ttbar_test_files:
                EvaluateModel(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.ttbar_test_files[ttbar_models]["Path"],
                    data_set_name=training_config.ttbar_test_files[ttbar_models][
                        "data_set_name"
                    ],
                )

        if training_config.zpext_test_files is not None:
            if tagger_name is None:
                logger.info("Start evaluating taggers in Z' test files...")

            else:
                logger.info("Start evaluating UMAMI with Z' test files...")

            for zpext_models in training_config.zpext_test_files:
                EvaluateModel(
                    args=parser_args,
                    train_config=training_config,
                    preprocess_config=preprocessing_config,
                    test_file=training_config.zpext_test_files[zpext_models]["Path"],
                    data_set_name=training_config.zpext_test_files[zpext_models][
                        "data_set_name"
                    ],
                )

    else:
        raise ValueError(
            f"""
            Tagger {tagger_name} is not supported!.
            """
        )
