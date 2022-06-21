#!/usr/bin/env python
"""Execution script for training model evaluations."""

from umami.configuration import global_config, logger, set_log_level  # isort:skip
import argparse
import os
import pickle

import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model  # pylint: disable=import-error
from tensorflow.keras.utils import CustomObjectScope  # pylint: disable=import-error

import umami.data_tools as udt
import umami.evaluation_tools as uet
import umami.tf_tools as utf
import umami.train_tools as utt
from umami.evaluation_tools import FeatureImportance
from umami.helper_tools import get_class_label_variables, get_class_prob_var_names
from umami.preprocessing_tools import Configuration

# from plottingFunctions import sigBkgEff
tf.compat.v1.disable_eager_execution()


def get_parser():
    """
    Argument parser for the evaluation script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="Evaluation script command line options."
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

    parser.add_argument(
        "-t",
        "--tagger",
        action="store_true",
        help="Decide, which tagger was used and is to be evaluated.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
    )

    parser.add_argument(
        "--nJets",
        type=int,
        help="""Number of jets used for the testing. This will overwrite the value which
        is provided in the training config file. If no value is provided either in the
        command line or in the training config file it will use all available jets in
        the test files.""",
    )

    parser.add_argument(
        "--shapley",
        action="store_true",
        help="Calculates feature importance for DL1",
    )

    return parser.parse_args()


def evaluate_model(
    args: object,
    train_config: object,
    preprocess_config: object,
    test_file: str,
    data_set_name: str,
    tagger: str,
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
    tagger : str
        Name of the tagger that is to be evaluated. Can either be umami or
        umami_cond_att depending which architecture is used.

    Raises
    ------
    ValueError
        If no epoch is given when evaluating UMAMI.
    ValueError
        If the given tagger argument in train config is not a list.
    """

    # Get train parameters
    test_set_config = train_config.test_files[data_set_name]
    eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = eval_params["frac_values_comp"]
    working_point = eval_params["WP"]
    tracks_name = train_config.tracks_name
    var_cuts = (
        test_set_config["variable_cuts"] if "variable_cuts" in test_set_config else None
    )

    # Print a warning that no variable cuts are used for the file
    if var_cuts is None:
        logger.warning(
            f"No variable cuts are given for {data_set_name}."
            " Please check if you defined them!"
        )

    if (
        "results_filename_extension" in eval_params
        and eval_params["results_filename_extension"] is not None
    ):
        results_filename_extension = eval_params["results_filename_extension"]
        logger.warning(
            f"Results filename extension is set to {results_filename_extension}. "
            "This means you have to specify the 'evaluation_file' when plotting your "
            "results."
        )
    else:
        results_filename_extension = ""

    # Init the placeholder lists for tagger_names
    tagger_names = []
    tagger_preds = []

    # Set number of n_jets for testing
    n_jets = int(eval_params["n_jets"]) if not args.nJets else args.nJets

    # Check the config if the trained model is also to be evaluated
    try:
        eval_model_bool = train_config.evaluate_trained_model
    except AttributeError:
        eval_model_bool = True

    # Set epoch to use for evaluation of trained model or dummy value if
    # tagger scores from derivations should be used
    epoch = args.epoch if eval_model_bool else 0

    # Test if multiple taggers are given or not
    tagger_list = (
        [eval_params["tagger"]]
        if isinstance(eval_params["tagger"], str)
        else eval_params["tagger"]
    )
    try:
        assert isinstance(tagger_list, list)
    except AssertionError as error:
        raise ValueError(
            """
            Tagger given in Eval_parameters_validation
            is not a string or a list!
            """
        ) from error

    # evaluate trained model file (for evaluate_trained_model: True in config)
    if eval_model_bool:
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

        # Check which test files need to be loaded depending on the umami version
        logger.info("Start loading %s test file", data_set_name)
        if tagger.casefold() == "umami_cond_att".casefold():
            # Load the test jets
            x_test, x_test_trk, _ = utt.GetTestFile(
                input_file=test_file,
                var_dict=train_config.var_dict,
                preprocess_config=preprocess_config,
                class_labels=class_labels,
                tracks_name=tracks_name,
                nJets=n_jets,
                exclude=exclude,
                cut_vars_dict=var_cuts,
                print_logger=False,
            )

            # Form the inputs for the network
            x_comb = [
                x_test_trk,
                x_test[
                    [
                        global_config.etavariable,
                        global_config.pTvariable,
                    ]
                ],
                x_test,
            ]

        else:
            # Get the testfile with the needed configs
            x_test, x_test_trk, _ = utt.GetTestFile(
                input_file=test_file,
                var_dict=train_config.var_dict,
                preprocess_config=preprocess_config,
                class_labels=class_labels,
                tracks_name=tracks_name,
                nJets=n_jets,
                exclude=exclude,
                cut_vars_dict=var_cuts,
            )

            # Form the inputs for the network
            x_comb = [x_test_trk, x_test]

        # Load the model for evaluation. Note: The Sum is needed here!
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

        # Predict the output of the model on the test jets
        pred_dips, pred_umami = model.predict(
            x_comb,
            batch_size=eval_params["eval_batch_size"],
            verbose=0,
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
    label_var_list, _ = get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to variables list
    for tagger_iter in tagger_list:
        variables += get_class_prob_var_names(
            tagger_name=f"{tagger_iter}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=n_jets,
        variables=variables,
        cut_vars_dict=var_cuts,
    )

    # Get the discriminant values and probabilities of each tagger for each jet
    df_discs_dict = uet.get_scores_probs_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=tagger_preds,
        tagger_names=tagger_names,
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={
            "dips": eval_params["frac_values"]["dips"],
            "umami": eval_params["frac_values"]["umami"],
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
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-{epoch}.h5",
        data_set_name,
    )

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.get_rej_per_eff_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=tagger_preds,
        tagger_names=tagger_names,
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={
            "dips": eval_params["frac_values"]["dips"],
            "umami": eval_params["frac_values"]["umami"],
        }
        if tagger_preds != []  # pylint: disable=use-implicit-booleaness-not-comparison
        else None,
        frac_values_comp=frac_values_comp,
        eff_min=0.49 if "eff_min" not in eval_params else eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in eval_params else eval_params["eff_max"],
    )

    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )

    # Get the rejections, discs and f_* values for the taggers
    tagger_fraction_rej_dict = uet.get_rej_per_frac_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=tagger_preds,
        tagger_names=tagger_names,
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        target_eff=working_point,
        step=0.01 if "frac_step" not in eval_params else eval_params["frac_step"],
        frac_min=0.01 if "frac_min" not in eval_params else eval_params["frac_min"],
        frac_max=1.0 if "frac_max" not in eval_params else eval_params["frac_max"],
    )

    # Form the dict to a Dataframe and save it
    df_frac_rej = pd.DataFrame(tagger_fraction_rej_dict)
    del tagger_fraction_rej_dict

    df_frac_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )


def evaluate_model_dips(
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
        CADS depending which architecture is used.

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
    test_set_config = train_config.test_files[data_set_name]
    eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = eval_params["frac_values_comp"]
    working_point = eval_params["WP"]
    tracks_name = train_config.tracks_name
    var_cuts = (
        test_set_config["variable_cuts"] if "variable_cuts" in test_set_config else None
    )

    if (
        "results_filename_extension" in eval_params
        and eval_params["results_filename_extension"] is not None
    ):
        results_filename_extension = eval_params["results_filename_extension"]
        logger.warning(
            f"Results filename extension is set to {results_filename_extension}. "
            "This means you have to specify the 'evaluation_file' when plotting your "
            "results."
        )
    else:
        results_filename_extension = ""

    # Print a warning that no variable cuts are used for the file
    if var_cuts is None:
        logger.warning(
            f"No variable cuts are given for {data_set_name}."
            " Please check if you defined them!"
        )

    # Set number of n_jets for testing
    n_jets = int(eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if isinstance(eval_params["tagger"], str):
        tagger_list = [eval_params["tagger"]]

    elif isinstance(eval_params["tagger"], list):
        tagger_list = eval_params["tagger"]

    elif eval_params["tagger"] is None:
        tagger_list = []

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

    # Check which test files need to be loaded depending on the CADS version
    logger.info("Start loading %s test file", data_set_name)
    if tagger.casefold() == "CADS".casefold():
        # Load the test jets
        x_test, x_test_trk, y_test = utt.GetTestFile(
            input_file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            tracks_name=tracks_name,
            nJets=n_jets,
            cut_vars_dict=var_cuts,
            jet_variables=[
                global_config.etavariable,
                global_config.pTvariable,
            ],
            print_logger=False,
        )

        # Form the inputs for the network
        x_comb = [x_test_trk, x_test]

    else:
        # Get the testfile with the needed configs
        x_comb, y_test = utt.get_test_sample_trks(
            input_file=test_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=class_labels,
            tracks_name=tracks_name,
            nJets=n_jets,
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
        x_comb,
        batch_size=eval_params["eval_batch_size"],
        verbose=0,
    )

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to variables list
    for tagger_i in tagger_list:
        variables += get_class_prob_var_names(
            tagger_name=f"{tagger_i}", class_labels=class_labels
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=class_labels,
        nJets=n_jets,
        variables=variables,
        cut_vars_dict=var_cuts,
    )

    # Get the discriminant values and probabilities of each tagger for each jet
    df_discs_dict = uet.get_scores_probs_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dips],
        tagger_names=[f"{tagger.casefold()}"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={f"{tagger.casefold()}": eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
    )

    # Add dict to Dataframe and delete dict
    df_discs = pd.DataFrame(df_discs_dict)
    del df_discs_dict

    # Create results dir
    os.makedirs(f"{train_config.model_name}/results", exist_ok=True)

    # Save dataframe to h5
    df_discs.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-{args.epoch}.h5",
        data_set_name,
    )

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.get_rej_per_eff_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dips],
        tagger_names=["dips"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"dips": eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
        eff_min=0.49 if "eff_min" not in eval_params else eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in eval_params else eval_params["eff_max"],
    )

    # Form the dict to a Dataframe and save it
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{args.epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )

    # Get the rejections, discs and f_* values for the taggers
    tagger_fraction_rej_dict = uet.get_rej_per_frac_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dips],
        tagger_names=["dips"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        target_eff=working_point,
        step=0.01 if "frac_step" not in eval_params else eval_params["frac_step"],
        frac_min=0.01 if "frac_min" not in eval_params else eval_params["frac_min"],
        frac_max=1.0 if "frac_max" not in eval_params else eval_params["frac_max"],
    )

    df_frac_rej = pd.DataFrame(tagger_fraction_rej_dict)
    del tagger_fraction_rej_dict

    df_frac_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )

    if (
        "calculate_saliency" in eval_params
        and eval_params["calculate_saliency"] is True
    ):
        # Get the saliency map dict
        saliency_map_dict = uet.get_saliency_map_dict(
            model=model,
            model_pred=pred_dips,
            X_test=x_comb,
            Y_test=y_test,
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=eval_params["frac_values"],
            var_dict_path=train_config.var_dict,
            tracks_name=tracks_name,
            nTracks=eval_params["saliency_ntrks"]
            if "saliency_ntrks" in eval_params
            else None,
            effs=eval_params["saliency_effs"]
            if "saliency_effs" in eval_params
            else None,
        )

        # Create results dir and pickle file
        os.system(f"mkdir -p {train_config.model_name}/results")
        with open(
            f"{train_config.model_name}/results/saliency{results_filename_extension}"
            f"_{args.epoch}_{data_set_name}.pkl",
            "wb",
        ) as pkl_file:
            pickle.dump(saliency_map_dict, pkl_file)


def evaluate_model_dl1(
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
    test_set_config = train_config.test_files[data_set_name]
    eval_params = train_config.Eval_parameters_validation
    class_labels = train_config.NN_structure["class_labels"]
    main_class = train_config.NN_structure["main_class"]
    frac_values_comp = eval_params["frac_values_comp"]
    working_point = eval_params["WP"]
    var_cuts = (
        test_set_config["variable_cuts"] if "variable_cuts" in test_set_config else None
    )

    # Print a warning that no variable cuts are used for the file
    if var_cuts is None:
        logger.warning(
            f"No variable cuts are given for {data_set_name}."
            " Please check if you defined them!"
        )

    if (
        "results_filename_extension" in eval_params
        and eval_params["results_filename_extension"] is not None
    ):
        results_filename_extension = eval_params["results_filename_extension"]
        logger.warning(
            f"Results filename extension is set to {results_filename_extension}. "
            "This means you have to specify the 'evaluation_file' when plotting your "
            "results."
        )
    else:
        results_filename_extension = ""

    # Check if epochs are set or not
    if args.epoch is None:
        raise ValueError("You need to give an epoch which is to be evaluated!")

    # Set number of nJets for testing
    n_jets = int(eval_params["n_jets"]) if not args.nJets else args.nJets

    # Test if multiple taggers are given or not
    if isinstance(eval_params["tagger"], str):
        tagger_list = [eval_params["tagger"]]

    elif isinstance(eval_params["tagger"], list):
        tagger_list = eval_params["tagger"]

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
    logger.info("Start loading %s test file", data_set_name)
    x_test, _ = utt.get_test_sample(
        input_file=test_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=n_jets,
        exclude=exclude,
        cut_vars_dict=var_cuts,
    )

    # Load the model for evaluation.
    model = load_model(model_file)

    # Predict the output of the model on the test jets
    pred_dl1 = model.predict(
        x_test,
        batch_size=eval_params["eval_batch_size"],
        verbose=0,
    )

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Adding all needed truth info variables
    label_var_list, _ = get_class_label_variables(
        class_labels=class_labels,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to
    # variables list
    for tagger_i in tagger_list:
        variables += get_class_prob_var_names(
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
        nJets=n_jets,
        variables=variables,
        cut_vars_dict=var_cuts,
    )

    # Get the discriminant values and probabilities of each tagger
    # for each jet
    df_discs_dict = uet.get_scores_probs_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dl1],
        tagger_names=["DL1"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"DL1": eval_params["frac_values"]},
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
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-{args.epoch}.h5",
        data_set_name,
    )

    # Get the rejections, discs and effs of the taggers
    tagger_rej_dicts = uet.get_rej_per_eff_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dl1],
        tagger_names=["DL1"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        frac_values={"DL1": eval_params["frac_values"]},
        frac_values_comp=frac_values_comp,
        eff_min=0.49 if "eff_min" not in eval_params else eval_params["eff_min"],
        eff_max=1.0 if "eff_max" not in eval_params else eval_params["eff_max"],
    )

    # Add dict to Dataframe and delete dict
    df_eff_rej = pd.DataFrame(tagger_rej_dicts)
    del tagger_rej_dicts

    df_eff_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_eff-{args.epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )

    # Get the rejections, discs and f_* values for the taggers
    tagger_fraction_rej_dict = uet.get_rej_per_frac_dict(
        jets=jets,
        y_true=truth_internal_labels,
        tagger_preds=[pred_dl1],
        tagger_names=["DL1"],
        tagger_list=tagger_list,
        class_labels=class_labels,
        main_class=main_class,
        target_eff=working_point,
        step=0.01 if "frac_step" not in eval_params else eval_params["frac_step"],
        frac_min=0.01 if "frac_min" not in eval_params else eval_params["frac_min"],
        frac_max=1.0 if "frac_max" not in eval_params else eval_params["frac_max"],
    )

    # Form the dict to a Dataframe and save it
    df_frac_rej = pd.DataFrame(tagger_fraction_rej_dict)
    del tagger_fraction_rej_dict

    df_frac_rej.to_hdf(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        data_set_name,
    )

    # Save the number of jets in the test file to the h5 file.
    # This is needed to calculate the binomial errors
    with h5py.File(
        f"{train_config.model_name}/results/"
        f"results{results_filename_extension}-rej_per_fractions-{args.epoch}.h5",
        "a",
    ) as h5_file:
        # Put the number of jets per class in the dict for unc calculation
        for flav_counter, flavour in enumerate(class_labels):
            h5_file.attrs[f"njets_{flavour}"] = len(
                truth_internal_labels[truth_internal_labels == flav_counter]
            )

    if args.shapley:
        logger.info("Explaining feature importance with SHAPley")
        FeatureImportance.ShapleyOneFlavor(
            model=model,
            test_data=x_test,
            model_output=eval_params["shapley"]["model_output"],
            feature_sets=eval_params["shapley"]["feature_sets"],
            plot_size=eval_params["shapley"]["plot_size"],
            plot_path=f"{train_config.model_name}/",
            plot_name=test_file_entry + "_shapley_b-jets",
        )

        if eval_params["shapley"]["bool_all_flavor_plot"]:
            FeatureImportance.ShapleyAllFlavors(
                model=model,
                test_data=x_test,
                feature_sets=eval_params["shapley"]["feature_sets"],
                averaged_sets=eval_params["shapley"]["averaged_sets"],
                plot_size=eval_params["shapley"]["plot_size"],
                plot_path=f"{train_config.model_name}/",
                plot_name=test_file_entry + "_shapley_all_flavors",
            )


if __name__ == "__main__":
    arg_parser = get_parser()

    # Set logger level
    if arg_parser.verbose:
        set_log_level(logger, "DEBUG")

    training_config = utt.Configuration(arg_parser.config_file)

    # Check for evaluation only (= evaluation of tagger scores in files) is used:
    # if nothing is specified, assume that freshly trained tagger is evaluated
    try:
        evaluate_trained_model = training_config.evaluate_trained_model
    except AttributeError:
        evaluate_trained_model = True  # pylint: disable=invalid-name

    preprocessing_config = (
        Configuration(training_config.preprocess_config)
        if evaluate_trained_model
        else training_config
    )

    # Get the tagger from arg_parser. If not given, use the one from train config
    if arg_parser.tagger:
        tagger_name = arg_parser.tagger
    else:
        try:
            tagger_name = training_config.NN_structure["tagger"]

        except KeyError:
            logger.info(
                "No tagger defined. Running evaluation without a freshly trained model!"
            )
            tagger_name = None  # pylint: disable=invalid-name

    # TODO Change this in python 3.10
    if tagger_name == "dl1":
        logger.info("Start evaluating DL1 with test files...")
        for (
            test_file_identifier,
            test_file_config,
        ) in training_config.test_files.items():
            evaluate_model_dl1(
                args=arg_parser,
                train_config=training_config,
                preprocess_config=preprocessing_config,
                test_file=test_file_config["path"],
                data_set_name=test_file_identifier,
                test_file_entry=test_file_identifier,
            )

    elif tagger_name.casefold() in ("dips", "cads"):
        logger.info("Start evaluating DIPS with test files...")
        for (
            test_file_identifier,
            test_file_config,
        ) in training_config.test_files.items():
            evaluate_model_dips(
                args=arg_parser,
                train_config=training_config,
                preprocess_config=preprocessing_config,
                test_file=test_file_config["path"],
                data_set_name=test_file_identifier,
                tagger=tagger_name,
            )

    elif (
        tagger_name.casefold() in ("umami", "umami_cond_att")
        or not evaluate_trained_model
    ):
        if tagger_name.casefold() in ("umami", "umami_cond_att"):
            logger.info(f"Start evaluating {tagger_name} with test files...")
        else:
            logger.info("Start evaluating in-file taggers with test files...")

        for (
            test_file_identifier,
            test_file_config,
        ) in training_config.test_files.items():
            evaluate_model(
                args=arg_parser,
                train_config=training_config,
                preprocess_config=preprocessing_config,
                test_file=test_file_config["path"],
                data_set_name=test_file_identifier,
                tagger=tagger_name,
            )

    else:
        raise ValueError(
            f"""
            Tagger {tagger_name} is not supported!.
            """
        )
