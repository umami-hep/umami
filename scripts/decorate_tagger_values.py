"""Module handling training file writing to disk."""

import h5py
from tensorflow.keras.models import load_model  # pylint: disable=import-error
from tensorflow.keras.utils import CustomObjectScope  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.configuration import global_config, logger
from umami.data_tools import LoadJetsFromFile


def decorate_tagger_probabilites(
    input_file: str,
    output_file: str,
    model_type: str,
    model_h5_name: str,
    model_file: str,
    model_var_dict: str,
    model_scale_dict: str,
    class_labels: list,
    chunk_size: int = int(1e6),
    predict_batch_size: int = int(3e4),
    tracks_name: str = None,
    exclude: list = None,
    compression: str = None,
    n_jets: int = None,
):
    """
    Load the given input file and model file and
    calculate the output for the model for the given
    jets in the input file. The output values of the
    tagger are then added to the input_file and the full
    jets + predictions are saved in the output file.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    output_file : str
        Path to the output file.
    model_type : str
        Tagger type that is used, supported are 'dips',
        'dips_attention', 'cads', 'dl1', 'umami' and
        'umami_cond_att'.
    model_h5_name : str
        Name of the model in the h5 output file. E.g. if model_h5_name="dipsTEST", the
        predictions will be stored as "dipsTEST_pu", "dipsTEST_pc", etc.
    model_file : str
        Model file in which the tagger is stored.
    model_var_dict : str
        Path to the variable dict which was used to train
        the tagger which is to be used.
    model_scale_dict : str
        Scale dict with which the tagger was trained.
    class_labels : list
        Class labels with which the tagger was trained.
    chunk_size : int, optional
        Number of jets which are loaded per step to reduce memory
        usage, by default int(1e6)
    predict_batch_size : int, optional
        Batch size for the model.predict command, by default int(3e4)
    tracks_name : str, optional
        Name of the track dataset to use, by default None
    exclude : list, optional
        List of variables which are to be excluded, by default None
    compression : str, optional
        Compression of the output file, by default None
    n_jets : int, optional
        Number of jets to be evaluated from the input file, by default None, which will
        evaluate the whole input file

    Raises
    ------
    ValueError
        If the given model type is not supported.
    """
    # Get model file path
    logger.info("Using model saved in %s", model_file)
    logger.info("Will calculate the predictions for the jets in file %s", input_file)
    logger.info("File with predictions will be written to %s", output_file)

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

    # Retrieve the number of jets available
    n_jets_available = len(h5py.File(input_file, "r")["/jets"])
    if n_jets is None:
        logger.info(
            "No number of jets specified. Will evaluate the output for the whole file, "
            "which has %i jets",
            n_jets_available,
        )
        n_jets = n_jets_available
    else:
        if n_jets > n_jets_available:
            logger.warning(
                "Requested to evaluate %i jets, but only %i jets are available in the "
                "file. Will evaluate the whole file.",
                n_jets,
                n_jets_available,
            )
            n_jets = n_jets_available
        else:
            logger.info("Evaluating the first %i jets of the input file.", n_jets)

    # Get the indices
    start_ind = 0

    tupled_indices = []
    while start_ind < n_jets:
        end_ind = int(start_ind + chunk_size)
        end_ind = min(end_ind, n_jets)

        tupled_indices.append((start_ind, end_ind))
        start_ind = end_ind
        end_ind = int(start_ind + chunk_size)

    for i, indices_to_load in enumerate(tupled_indices):
        logger.info("Processing chunk %i/%i", i + 1, len(tupled_indices))
        if model_type.casefold() == "dl1":
            x_comb, labels = utt.get_test_sample(
                input_file=input_file,
                var_dict=model_var_dict,
                scale_dict=model_scale_dict,
                class_labels=class_labels,
                indices_to_load=indices_to_load,
                exclude=exclude,
                cut_vars_dict=None,
            )

        elif model_type.casefold() == "cads":
            # Load the test jets
            x_test, x_test_trk, labels = utt.get_test_file(
                input_file=input_file,
                var_dict=model_var_dict,
                scale_dict=model_scale_dict,
                class_labels=class_labels,
                tracks_name=tracks_name,
                indices_to_load=indices_to_load,
                cut_vars_dict=None,
                jet_variables=[
                    global_config.etavariable,
                    global_config.pTvariable,
                ],
                print_logger=False,
            )

            # Form the inputs for the network
            x_comb = [x_test_trk, x_test]

        elif model_type.casefold() in ("dips", "dips_attention"):
            # Load the test jets
            x_comb, labels = utt.get_test_sample_trks(
                input_file=input_file,
                var_dict=model_var_dict,
                scale_dict=model_scale_dict,
                class_labels=class_labels,
                tracks_name=tracks_name,
                indices_to_load=indices_to_load,
                cut_vars_dict=None,
            )

        elif model_type.casefold() in ("umami", "umami_cond_att"):
            # Get the testfile with the needed configs
            x_test, x_test_trk, labels = utt.get_test_file(
                input_file=input_file,
                var_dict=model_var_dict,
                scale_dict=model_scale_dict,
                class_labels=class_labels,
                tracks_name=tracks_name,
                indices_to_load=indices_to_load,
                exclude=exclude,
                cut_vars_dict=None,
            )

            # Form the inputs for the network
            if model_type.casefold() == "umami_cond_att":
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

        else:
            raise ValueError(f"Given model_type {model_type} is not supported!")

        # Get predictions from trained model
        prediction = model.predict(
            x_comb,
            batch_size=predict_batch_size,
            verbose=0,
        )

        # Load the jets which are to be decorated
        jets, _ = LoadJetsFromFile(
            filepath=input_file,
            class_labels=class_labels,
            indices_to_load=indices_to_load,
        )

        # Get flavour categories
        flavour_categories = global_config.flavour_categories

        # Append the tagger output probabilites to the jets dataset
        for flav_counter, flavour in enumerate(class_labels):
            jets[
                f'{model_h5_name}_{flavour_categories[flavour]["prob_var_name"]}'
            ] = prediction[:, flav_counter]

        # Write the jets and labels to file
        if indices_to_load[0] == 0:
            with h5py.File(output_file, "w") as outfile:
                logger.info("Creating output file %s", output_file)
                outfile.create_dataset(
                    "jets",
                    data=jets.to_records(index=False),
                    compression=compression,
                    maxshape=(None,),
                )
                outfile.create_dataset(
                    "labels",
                    data=labels,
                    compression=compression,
                    maxshape=(None, labels.shape[1]),
                )

        else:
            # appending to existing dataset
            with h5py.File(output_file, "a") as outfile:
                outfile["jets"].resize(
                    (outfile["jets"].shape[0] + jets.shape[0]),
                    axis=0,
                )
                outfile["jets"][-jets.shape[0] :] = jets.to_records(index=False)  # jets

                outfile["labels"].resize(
                    (outfile["labels"].shape[0] + labels.shape[0]),
                    axis=0,
                )
                outfile["labels"][-labels.shape[0] :] = labels
