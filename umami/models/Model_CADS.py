"""Keras model of the CADS tagger."""
from umami.configuration import logger  # isort:skip
import json

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.models import load_model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt


def Cads_model(train_config, input_shape):
    """Keras model definition of CADS.

    Parameters
    ----------
    train_config : object
        training config
    input_shape : tuple
        dataset input shape

    Returns
    -------
    keras model
        CADS keras model
    int
        Number of epochs
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    load_optimiser = (
        NN_structure["load_optimiser"] if "load_optimiser" in NN_structure else True
    )

    if train_config.model_file is not None:
        # Load CADS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        cads = load_model(
            train_config.model_file,
            {
                "Sum": utf.Sum,
                "Attention": utf.Attention,
                "DeepSet": utf.DeepSet,
                "AttentionPooling": utf.AttentionPooling,
                "DenseNet": utf.DenseNet,
                "ConditionalAttention": utf.ConditionalAttention,
                "ConditionalDeepSet": utf.ConditionalDeepSet,
            },
            compile=load_optimiser,
        )

    else:
        # Init a new cads/dips attention model
        cads = utf.Deepsets_model(
            repeat_input_shape=input_shape,
            num_conditions=NN_structure["N_Conditions"],
            num_set_features=NN_structure["ppm_sizes"][-1],
            sets_nodes=NN_structure["ppm_sizes"][:-1],
            classif_nodes=NN_structure["dense_sizes"],
            classif_output=len(NN_structure["class_labels"]),
            pooling="attention",
            attention_nodes=NN_structure["attention_sizes"],
            condition_sets=NN_structure["ppm_condition"],
            condition_attention=NN_structure["attention_condition"],
            condition_classifier=NN_structure["dense_condition"],
            shortcut_inputs=False,
            sets_batch_norm=NN_structure["Batch_Normalisation"],
            classif_batch_norm=NN_structure["Batch_Normalisation"],
            activation="relu",
            attention_softmax=False,
        )

    if not load_optimiser or train_config.model_file is None:
        # Set optimiser and loss
        model_optimiser = Adam(learning_rate=NN_structure["lr"])
        cads.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print CADS model summary when log level lower or equal INFO level
    if logger.level <= 20:
        cads.summary()

    return cads, NN_structure["epochs"]


def Cads(args, train_config, preprocess_config):
    """Training handling of CADS.

    Parameters
    ----------
    args : parser args
        Arguments from command line parser
    train_config : object
        training configuration
    preprocess_config : object
        preprocessing configuration

    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation
    tracks_key = train_config.tracks_key

    # Get the shapes for training
    with h5py.File(train_config.train_file, "r") as f:
        nJets, nTrks, nFeatures = f[tracks_key].shape
        nJets, nDim = f["Y_train"].shape

        if NN_structure["nJets_train"] is not None:
            nJets = int(NN_structure["nJets_train"])

    # Print how much jets are used
    logger.info(f"Number of Jets used for training: {nJets}")

    # Init CADS model
    cads, epochs = Cads_model(train_config=train_config, input_shape=(nTrks, nFeatures))

    if NN_structure["use_sample_weights"]:
        tensor_types = (
            {"input_1": tf.float32, "input_2": tf.float32},
            tf.float32,
            tf.float32,
        )
        tensor_shapes = (
            {
                "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                "input_2": tf.TensorShape([None, NN_structure["N_Conditions"]]),
            },
            tf.TensorShape([None, nDim]),
            tf.TensorShape([None]),
        )
    else:
        tensor_types = (
            {"input_1": tf.float32, "input_2": tf.float32},
            tf.float32,
        )
        tensor_shapes = (
            {
                "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                "input_2": tf.TensorShape([None, NN_structure["N_Conditions"]]),
            },
            tf.TensorShape([None, nDim]),
        )

    # Get training set from generator
    train_dataset = (
        tf.data.Dataset.from_generator(
            utf.cads_generator(
                train_file_path=train_config.train_file,
                X_Name="X_train",
                X_trk_Name=tracks_key,
                Y_Name="Y_train",
                n_jets=nJets,
                batch_size=NN_structure["batch_size"],
                nConds=NN_structure["N_Conditions"],
                chunk_size=int(1e6),
                sample_weights=NN_structure["use_sample_weights"],
            ),
            output_types=tensor_types,
            output_shapes=tensor_shapes,
        )
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    cads_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    if "LRR" in NN_structure and NN_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**NN_structure)

    # Load validation data for callback
    val_data_dict = None
    if Val_params["n_jets"] > 0:
        if NN_structure["N_Conditions"] is None:
            val_data_dict = utt.load_validation_data_dips(
                train_config=train_config,
                preprocess_config=preprocess_config,
                nJets=int(Val_params["n_jets"]),
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
                nJets=int(Val_params["n_jets"]),
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
        class_labels=NN_structure["class_labels"],
        main_class=NN_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=Val_params["WP"],
        frac_dict=Val_params["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP=Val_params["WP"],
            n_jets=int(Val_params["n_jets"]),
            dir_name=train_config.model_name,
        ),
    )

    logger.info("Start training")
    history = cads.fit(
        train_dataset,
        epochs=nEpochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=validation_data,
        callbacks=[cads_mChkPt, reduce_lr, my_callback],
        steps_per_epoch=nJets / NN_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
    )

    # Dump dict into json
    logger.info(f"Dumping history file to {train_config.model_name}/history.json")

    # Make the history dict the same shape as the dict from the callbacks
    hist_dict = utt.prepare_history_dict(history.history)

    # Dump history file to json
    with open(f"{train_config.model_name}/history.json", "w") as outfile:
        json.dump(hist_dict, outfile, indent=4)
