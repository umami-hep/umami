"""Execution script for training model evaluations."""
from umami.configuration import global_config, logger, set_log_level  # isort:skip
import argparse
import pickle
from pathlib import Path

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
        "-s",
        "--step",
        type=str,
        default=None,
        help="""Decide which step of the evaluation to run. If this parameter is not
        given, all steps are run in order. The possible options for this are results,
        rej_per_eff, rej_per_frac and saliency.""",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set verbose level to debug for the logger.",
    )

    parser.add_argument(
        "--n_jets",
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
    eval_params = train_config.evaluation_settings
    class_labels = train_config.nn_structure["class_labels"]
    main_class = train_config.nn_structure["main_class"]
    frac_values_comp = eval_params.get("frac_values_comp")
    working_point = eval_params["working_point"]
    add_variables = eval_params.get("add_eval_variables")
    classes_to_evaluate = class_labels + eval_params.get(
        "extra_classes_to_evaluate", []
    )
    tracks_name = (
        train_config.tracks_name if hasattr(train_config, "tracks_name") else None
    )
    var_cuts = test_set_config.get("variable_cuts")
    tagger_from_file = eval_params.get("tagger")

    # Test if multiple taggers are given or not
    if isinstance(tagger_from_file, str):
        tagger_list = [tagger_from_file]

    elif isinstance(tagger_from_file, list):
        tagger_list = tagger_from_file

    elif tagger_from_file is None:
        tagger_list = []

    else:
        raise ValueError(
            """
            Tagger given in evaluation_settings
            is not a string or a list!
            """
        )

    if (
        "results_filename_extension" in eval_params
        and eval_params["results_filename_extension"] is not None
    ):
        results_filename_extension = eval_params["results_filename_extension"]
        logger.warning(
            "Results filename extension is set to %s. "
            "This means you have to specify the 'evaluation_file' when plotting your "
            "results.",
            results_filename_extension,
        )
    else:
        results_filename_extension = ""

    # Print a warning that no variable cuts are used for the file
    if var_cuts is None:
        logger.warning(
            "No variable cuts are given for %s. Please check if you defined them!",
            data_set_name,
        )

    # Init the placeholder lists for tagger_names
    tagger_names = None
    tagger_preds = None

    # Set number of n_jets for testing
    n_jets = int(eval_params["n_jets"]) if not args.n_jets else args.n_jets

    # Check the config if the trained model is also to be evaluated
    try:
        eval_model_bool = train_config.evaluate_trained_model

    except AttributeError:
        eval_model_bool = True

    # Set epoch to use for evaluation of trained model or dummy value if
    # tagger scores from derivations should be used
    epoch = args.epoch if eval_model_bool else 0

    # evaluate trained model file (for evaluate_trained_model: True in config)
    if eval_model_bool:
        if epoch is None:
            raise ValueError("You need to give an epoch which is to be evaluated!")

        # Get model file path
        model_file = utt.get_model_path(
            model_name=train_config.model_name,
            epoch=args.epoch,
        )
        logger.info("Evaluating %s", model_file)

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

        # Define excluded variables and laod them
        exclude = None
        if "exclude" in train_config.config:
            exclude = train_config.config["exclude"]

        # Check which test files need to be loaded depending on the umami version
        logger.info("Start loading %s test file", data_set_name)
        if tagger.casefold() == "dl1":
            x_comb, _ = utt.get_test_sample(
                input_file=test_file,
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=classes_to_evaluate,
                n_jets=n_jets,
                exclude=exclude,
                cut_vars_dict=var_cuts,
            )

            # Predict the output of the model on the test jets
            pred_dl1 = model.predict(
                x_comb,
                batch_size=eval_params["eval_batch_size"],
                verbose=0,
            )

            # Fill the tagger_names and tagger_preds
            tagger_names = [tagger]
            tagger_preds = [pred_dl1]

            # Define fraction value dict
            frac_values = {tagger.casefold(): eval_params["frac_values"]}

        elif tagger.casefold() == "cads":
            # Load the test jets
            x_test, x_test_trk, y_test = utt.get_test_file(
                input_file=test_file,
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=classes_to_evaluate,
                tracks_name=tracks_name,
                n_jets=n_jets,
                cut_vars_dict=var_cuts,
                jet_variables=[
                    global_config.etavariable,
                    global_config.pTvariable,
                ],
                print_logger=False,
            )

            # Form the inputs for the network
            x_comb = [x_test_trk, x_test]

            # Get predictions from trained model
            pred_cads = model.predict(
                x_comb,
                batch_size=eval_params["eval_batch_size"],
                verbose=0,
            )

            # Fill the tagger_names and tagger_preds
            tagger_names = [tagger]
            tagger_preds = [pred_cads]

            # Define fraction value dict
            frac_values = {tagger.casefold(): eval_params["frac_values"]}

        elif tagger.casefold() in ("dips", "dips_attention"):
            # Load the test jets
            x_comb, y_test = utt.get_test_sample_trks(
                input_file=test_file,
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=classes_to_evaluate,
                tracks_name=tracks_name,
                n_jets=n_jets,
                cut_vars_dict=var_cuts,
            )

            # Get predictions from trained model
            pred_dips = model.predict(
                x_comb,
                batch_size=eval_params["eval_batch_size"],
                verbose=0,
            )

            # Fill the tagger_names and tagger_preds
            tagger_names = [tagger]
            tagger_preds = [pred_dips]

            # Define fraction value dict
            frac_values = {tagger.casefold(): eval_params["frac_values"]}

        elif tagger.casefold() in ("umami", "umami_cond_att"):
            # Get the testfile with the needed configs
            x_test, x_test_trk, _ = utt.get_test_file(
                input_file=test_file,
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=classes_to_evaluate,
                tracks_name=tracks_name,
                n_jets=n_jets,
                exclude=exclude,
                cut_vars_dict=var_cuts,
            )

            # Form the inputs for the network
            if tagger.casefold() == "umami_cond_att":
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
                x_comb = [x_test_trk, x_test]

            # Predict the output of the model on the test jets
            pred_dips, pred_umami = model.predict(
                x_comb,
                batch_size=eval_params["eval_batch_size"],
                verbose=0,
            )

            # Fill the tagger_names and tagger_preds
            tagger_names = ["dips", "umami"]
            tagger_preds = [pred_dips, pred_umami]

            # Define fraction value dict
            frac_values = {
                "dips": eval_params["frac_values"]["dips"],
                "umami": eval_params["frac_values"]["umami"],
            }

        else:
            raise ValueError(f"Given tagger {tagger} is not supported!")

    # Define variables that need to be loaded
    variables = [
        global_config.etavariable,
        global_config.pTvariable,
    ]

    # Load extra variables
    add_variables_available = None
    if add_variables is not None:
        # Get list with all available variables
        available_variables = list(
            h5py.File(test_file, "r")["/jets"].dtype.fields.keys()
        )
        add_variables_available = []
        for item in add_variables:
            if item in available_variables:
                add_variables_available.append(item)
            else:
                logger.info("Variable '%s' not available", item)
        variables.extend(add_variables_available)

    # Adding all needed truth info variables
    label_var_list, _ = get_class_label_variables(
        class_labels=classes_to_evaluate,
    )
    variables += list(set(label_var_list))

    # Add the predictions labels for the defined taggers to variables list
    for tagger_iter in tagger_list:
        variables += get_class_prob_var_names(
            tagger_name=f"{tagger_iter}", class_labels=classes_to_evaluate
        )

    # Load the jets and truth labels (internal) with selected variables
    jets, truth_internal_labels = udt.LoadJetsFromFile(
        filepath=test_file,
        class_labels=classes_to_evaluate,
        n_jets=n_jets,
        variables=variables,
        cut_vars_dict=var_cuts,
    )

    # Create results dir
    Path(f"{train_config.model_name}/results").mkdir(parents=True, exist_ok=True)

    if args.step in (None, "results"):
        # Get the discriminant values and probabilities of each tagger for each jet
        df_discs_dict = uet.get_scores_probs_dict(
            jets=jets,
            y_true=truth_internal_labels,
            tagger_classes=class_labels,
            tagger_preds=tagger_preds,
            tagger_names=tagger_names,
            tagger_list=tagger_list,
            class_labels=classes_to_evaluate,
            main_class=main_class,
            frac_values=frac_values if tagger_preds else None,
            frac_values_comp=frac_values_comp,
        )

        # Adding truth label values to the dict
        for truth_variable in list(set(label_var_list)):
            df_discs_dict[truth_variable] = jets[truth_variable]

        # Add dict to Dataframe and delete dict
        df_discs = pd.DataFrame(df_discs_dict)
        del df_discs_dict

        # Adding extra variables if available
        if add_variables_available is not None:
            for item in add_variables_available:
                logger.info("Adding %s", item)
                df_discs[item] = jets[item]

        # Save dataframe to h5
        df_discs.to_hdf(
            f"{train_config.model_name}/results/"
            f"results{results_filename_extension}-{epoch}.h5",
            data_set_name,
        )

    if args.step in (None, "rej_per_eff"):
        # Get the rejections, discs and effs of the taggers
        tagger_rej_dicts = uet.get_rej_per_eff_dict(
            jets=jets,
            y_true=truth_internal_labels,
            tagger_classes=class_labels,
            tagger_preds=tagger_preds,
            tagger_names=tagger_names,
            tagger_list=tagger_list,
            class_labels=classes_to_evaluate,
            main_class=main_class,
            frac_values=frac_values if tagger_preds else None,
            frac_values_comp=frac_values_comp,
            eff_min=eval_params.get("eff_min", 0.49),
            eff_max=eval_params.get("eff_max", 1.0),
            x_axis_granularity=eval_params.get("x_axis_granularity", 100),
            progress_bar=bool(args.verbose),
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
            for flav_counter, flavour in enumerate(classes_to_evaluate):
                h5_file.attrs[f"n_jets_{flavour}"] = len(
                    truth_internal_labels[truth_internal_labels == flav_counter]
                )

    if args.step in (None, "rej_per_frac"):
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
            step=eval_params.get("frac_step", 0.01),
            frac_min=eval_params.get("frac_min", 0.01),
            frac_max=eval_params.get("frac_max", 1.0),
            progress_bar=bool(args.verbose),
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
                h5_file.attrs[f"n_jets_{flavour}"] = len(
                    truth_internal_labels[truth_internal_labels == flav_counter]
                )

    if args.step in (None, "saliency"):
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
                nTracks=eval_params.get("saliency_ntrks"),
                effs=eval_params.get("saliency_effs"),
            )

            # Pickle file
            with open(
                f"{train_config.model_name}"
                f"/results/saliency{results_filename_extension}"
                f"_{args.epoch}_{data_set_name}.pkl",
                "wb",
            ) as pkl_file:
                pickle.dump(saliency_map_dict, pkl_file)

    if args.shapley:
        logger.info("Explaining feature importance with SHAPley")
        FeatureImportance.ShapleyOneFlavor(
            model=model,
            test_data=x_test,
            model_output=eval_params["shapley"]["model_output"],
            feature_sets=eval_params["shapley"]["feature_sets"],
            plot_size=eval_params["shapley"]["plot_size"],
            plot_path=f"{train_config.model_name}/",
            plot_name=data_set_name + "_shapley_b-jets",
        )

        if eval_params["shapley"]["bool_all_flavor_plot"]:
            FeatureImportance.ShapleyAllFlavors(
                model=model,
                test_data=x_test,
                feature_sets=eval_params["shapley"]["feature_sets"],
                averaged_sets=eval_params["shapley"]["averaged_sets"],
                plot_size=eval_params["shapley"]["plot_size"],
                plot_path=f"{train_config.model_name}/",
                plot_name=data_set_name + "_shapley_all_flavors",
            )


if __name__ == "__main__":
    arg_parser = get_parser()

    # Set logger level
    if arg_parser.verbose:
        set_log_level(logger, "DEBUG")

    training_config = utt.Configuration(arg_parser.config_file)

    # Retrieve tagger name from train config
    tagger_name = training_config.nn_structure.get("tagger")
    if tagger_name is None:
        logger.info(
            "No tagger defined. Running evaluation without a freshly trained model!"
        )

    # Check for evaluation only (= evaluation of tagger scores in files) is used:
    # if nothing is specified, assume that freshly trained tagger is evaluated
    try:
        evaluate_trained_model = training_config.evaluate_trained_model

    except AttributeError:
        evaluate_trained_model = True  # pylint: disable=invalid-name

    # TODO Change this in python 3.10
    if evaluate_trained_model:
        logger.info("Start evaluating %s with test files...", tagger_name)

    else:
        logger.info("Start evaluating in-file taggers with test files...")

    for (
        test_file_identifier,
        test_file_config,
    ) in training_config.test_files.items():
        evaluate_model(
            args=arg_parser,
            train_config=training_config,
            test_file=test_file_config["path"],
            data_set_name=test_file_identifier,
            tagger=tagger_name,
        )
