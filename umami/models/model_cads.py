"""Keras model of the CADS tagger."""
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt


def cads_model(
    train_config: object,
    input_shape: tuple,
    continue_training: bool = False,
):
    """Keras model definition of CADS.

    Parameters
    ----------
    train_config : object
        training config
    input_shape : tuple
        dataset input shape
    continue_training : bool, optional
        Decide, if the training is continued using the latest
        model file, by default False

    Returns
    -------
    keras model
        CADS keras model
    int
        Number of epochs
    int
        Starting epoch number
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.NN_structure

    # Check if a prepared model is used or not
    cads, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if cads is None:
        # Init a new cads/dips attention model
        cads = utf.Deepsets_model(
            repeat_input_shape=input_shape,
            num_conditions=nn_structure["N_Conditions"],
            num_set_features=nn_structure["ppm_sizes"][-1],
            sets_nodes=nn_structure["ppm_sizes"][:-1],
            classif_nodes=nn_structure["dense_sizes"],
            classif_output=len(nn_structure["class_labels"]),
            pooling="attention",
            attention_nodes=nn_structure["attention_sizes"],
            condition_sets=nn_structure["ppm_condition"],
            condition_attention=nn_structure["attention_condition"],
            condition_classifier=nn_structure["dense_condition"],
            shortcut_inputs=False,
            sets_batch_norm=nn_structure["Batch_Normalisation"],
            classif_batch_norm=nn_structure["Batch_Normalisation"],
            activation="relu",
            attention_softmax=False,
        )

    if load_optimiser is False:
        # Set optimiser and loss
        model_optimiser = Adam(learning_rate=nn_structure["lr"])
        cads.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print CADS model summary when log level lower or equal INFO level
    if logger.level <= 20:
        cads.summary()

    return cads, nn_structure["epochs"], init_epoch


def cads_tagger(args, train_config, preprocess_config):
    """
    Training handling of CADS.

    Parameters
    ----------
    args : parser args
        Arguments from command line parser
    train_config : object
        training configuration
    preprocess_config : object
        preprocessing configuration

    Raises
    ------
    ValueError
        If input is neither a h5 nor a directory.
    """

    # Load NN Structure and training parameter from file
    nn_structure = train_config.NN_structure
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation
    tracks_name = train_config.tracks_name

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    working_point = (
        float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
    )
    n_jets_val = (
        int(val_params["n_jets"])
        if "n_jets" in val_params
        else int(eval_params["n_jets"])
    )

    if ".h5" in train_config.train_file:
        # Init a metadata dict
        metadata = {}

        # Get the shapes for training
        with h5py.File(train_config.train_file, "r") as f:
            metadata["n_jets"], metadata["n_trks"], metadata["n_trk_features"] = f[
                f"X_{tracks_name}_train"
            ].shape
            _, metadata["n_dim"] = f["Y_train"].shape

        if nn_structure["use_sample_weights"]:
            tensor_types = (
                {"input_1": tf.float32, "input_2": tf.float32},
                tf.float32,
                tf.float32,
            )
            tensor_shapes = (
                {
                    "input_1": tf.TensorShape(
                        [None, metadata["n_trks"], metadata["n_trk_features"]]
                    ),
                    "input_2": tf.TensorShape([None, nn_structure["N_Conditions"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
                tf.TensorShape([None]),
            )
        else:
            tensor_types = (
                {"input_1": tf.float32, "input_2": tf.float32},
                tf.float32,
            )
            tensor_shapes = (
                {
                    "input_1": tf.TensorShape(
                        [None, metadata["n_trks"], metadata["n_trk_features"]]
                    ),
                    "input_2": tf.TensorShape([None, nn_structure["N_Conditions"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        # Get training set from generator
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.cads_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    X_trk_Name=f"X_{tracks_name}_train",
                    Y_Name="Y_train",
                    n_jets=int(nn_structure["nJets_train"])
                    if "nJets_train" in nn_structure
                    and nn_structure["nJets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
                    nConds=nn_structure["N_Conditions"],
                    chunk_size=int(1e6),
                    sample_weights=nn_structure["use_sample_weights"],
                ),
                output_types=tensor_types,
                output_shapes=tensor_shapes,
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    elif os.path.isdir(train_config.train_file):
        train_dataset, metadata = utf.load_tfrecords_train_dataset(
            train_config=train_config
        )

    else:
        raise ValueError(
            f"input file {train_config.train_file} is neither a .h5 file nor "
            "a directory with TF Record Files. You should check this."
        )

    # Init CADS model
    cads, epochs, init_epoch = cads_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        continue_training=train_config.config["continue_training"]
        if "continue_training" in train_config.config
        and train_config.config["continue_training"] is not None
        else False,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = epochs

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    # Set ModelCheckpoint as callback
    cads_model_checkpoint = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(cads_model_checkpoint)

    if "LRR" in nn_structure and nn_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Load validation data for callback
    val_data_dict = None
    if n_jets_val > 0:
        if nn_structure["N_Conditions"] is None:
            val_data_dict = utt.load_validation_data_dips(
                train_config=train_config,
                preprocess_config=preprocess_config,
                nJets=n_jets_val,
                convert_to_tensor=True,
            )

            # TODO: Add a representative validation dataset for training (shown in
            # stdout)
            # Create the validation data tuple for the fit function
            # validation_data = (
            #     val_data_dict["X_valid"],
            #     val_data_dict["Y_valid"],
            # )

        else:
            val_data_dict = utt.load_validation_data_umami(
                train_config=train_config,
                preprocess_config=preprocess_config,
                nJets=n_jets_val,
                convert_to_tensor=True,
                jets_var_list=["absEta_btagJes", "pt_btagJes"],
            )

            # TODO: Add a representative validation dataset for training (shown in
            # stdout)
            # Create the validation data tuple for the fit function
            # validation_data = (
            #     [
            #         val_data_dict["X_valid_trk"],
            #         val_data_dict["X_valid"],
            #     ],
            #     val_data_dict["Y_valid"],
            # )

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallback(
        model_name=train_config.model_name,
        class_labels=nn_structure["class_labels"],
        main_class=nn_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=working_point,
        frac_dict=eval_params["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP=working_point,
            n_jets=n_jets_val,
            dir_name=train_config.model_name,
        ),
    )

    # Append the callback
    callbacks.append(my_callback)

    logger.info("Start training")
    history = cads.fit(
        train_dataset,
        epochs=n_epochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=validation_data,
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure["nJets_train"]) / nn_structure["batch_size"]
        if "nJets_train" in nn_structure and nn_structure["nJets_train"] is not None
        else metadata["n_jets"] / nn_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )

    # Dump dict into json
    logger.info(f"Dumping history file to {train_config.model_name}/history.json")

    # Make the history dict the same shape as the dict from the callbacks
    hist_dict = utt.prepare_history_dict(history.history)

    # Dump history file to json
    with open(f"{train_config.model_name}/history.json", "w") as outfile:
        json.dump(hist_dict, outfile, indent=4)
