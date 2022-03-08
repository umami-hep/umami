#!/usr/bin/env python
"""Keras model of the UMAMI tagger."""
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
from tensorflow.keras import activations  # pylint: disable=import-error
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Masking,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.preprocessing_tools import GetVariableDict


def Umami_model(train_config=None, input_shape=None, njet_features=None):
    """Keras model definition of UMAMI tagger.

    Parameters
    ----------
    train_config : object, optional
        training config, by default None
    input_shape : tuple, optional
        dataset input shape, by default None
    njet_features: int, optional
        number of jet features, by default None

    Returns
    -------
    keras model
        UMAMI keras model
    int
        number of epochs
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]
    load_optimiser = (
        NN_structure["load_optimiser"] if "load_optimiser" in NN_structure else True
    )

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        umami = load_model(
            train_config.model_file, {"Sum": utf.Sum}, compile=load_optimiser
        )

    else:
        logger.info("No modelfile provided! Initialise a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(NN_structure["DIPS_ppm_units"]):

            tdd = TimeDistributed(
                Dense(phi_nodes, activation="linear"), name=f"Phi{i}_Dense"
            )(tdd)

            if batch_norm:
                tdd = TimeDistributed(
                    BatchNormalization(), name=f"Phi{i}_BatchNormalization"
                )(tdd)

            if dropout != 0:
                tdd = TimeDistributed(Dropout(rate=dropout), name=f"Phi{i}_Dropout")(
                    tdd
                )

            tdd = TimeDistributed(Activation(activations.relu), name=f"Phi{i}_ReLU")(
                tdd
            )

        # This is where the magic happens... sum up the track features!
        F = utf.Sum(name="Sum")(tdd)

        for j, (F_nodes, p) in enumerate(
            zip(
                NN_structure["DIPS_dense_units"],
                [dropout] * len(NN_structure["DIPS_dense_units"][:-1]) + [0],
            )
        ):

            F = Dense(F_nodes, activation="linear", name=f"F{j}_Dense")(F)
            if batch_norm:
                F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
            if dropout != 0:
                F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
            F = Activation(activations.relu, name=f"F{j}_ReLU")(F)

        dips_output = Dense(len(class_labels), activation="softmax", name="dips")(F)

        # Input layer
        jet_inputs = Input(shape=(njet_features,))

        # Adding the intermediate dense layers for DL1
        x = jet_inputs
        for unit in NN_structure["intermediate_units"]:
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        # Concatenate the inputs
        x = Concatenate()([F, x])

        # Loop to initialise the hidden layers
        for unit in NN_structure["DL1_units"]:
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        jet_output = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="umami",
        )(x)

        umami = Model(
            inputs=[trk_inputs, jet_inputs], outputs=[dips_output, jet_output]
        )

        # Set optimier and loss
        model_optimiser = Adam(learning_rate=NN_structure["lr"])
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    return umami, NN_structure["epochs"]


def Umami(args, train_config, preprocess_config):
    """Training handling of UMAMI tagger.

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
    NN_structure = train_config.NN_structure
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation

    # Init a list for the callbacks
    callbacks = []

    # Set the tracks collection name
    tracks_name = train_config.tracks_name

    # Get needed variable from the train config
    WP = float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
    n_jets_val = (
        int(val_params["n_jets"])
        if "n_jets" in val_params
        else int(eval_params["n_jets"])
    )

    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=n_jets_val,
            convert_to_tensor=True,
        )

    # Load the excluded variables from train_config
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    else:
        exclude = None

    # Load variable config
    variable_config = GetVariableDict(train_config.var_dict)

    # Get excluded variables
    _, _, excluded_var = utt.get_jet_feature_indices(
        variable_config["train_variables"], exclude
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
            _, metadata["n_jet_features"] = f["X_train"].shape
            if exclude is not None:
                metadata["n_jet_features"] -= len(exclude)
            logger.debug(
                f"Input shape of jet training set: {metadata['n_jet_features']}"
            )

        if NN_structure["use_sample_weights"]:
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
                    "input_2": tf.TensorShape([None, metadata["n_jet_features"]]),
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
                    "input_2": tf.TensorShape([None, metadata["n_jet_features"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        # Get training set from generator
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.umami_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    X_trk_Name=f"X_{tracks_name}_train",
                    Y_Name="Y_train",
                    n_jets=int(NN_structure["nJets_train"])
                    if "nJets_train" in NN_structure
                    and NN_structure["nJets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=NN_structure["batch_size"],
                    excluded_var=excluded_var,
                    sample_weights=NN_structure["use_sample_weights"],
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
            f"input file {train_config.train_file} is neither a .h5 file nor a"
            " directory with TF Record Files. You should check this."
        )

    umami, _ = Umami_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        njet_features=metadata["n_jet_features"],
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = NN_structure["epochs"]

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    if "LRR" in NN_structure and NN_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**NN_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Set ModelCheckpoint as callback
    umami_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_mChkPt)

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        class_labels=NN_structure["class_labels"],
        main_class=NN_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=WP,
        frac_dict=eval_params["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP=WP,
            n_jets=n_jets_val,
            dir_name=train_config.model_name,
        ),
    )

    # Append the callback
    callbacks.append(my_callback)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    history = umami.fit(
        train_dataset,
        epochs=nEpochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(
        #     [
        #         val_data_dict["X_valid_trk"],
        #         val_data_dict["X_valid"],
        #     ],
        #     val_data_dict["Y_valid"],
        # ),
        callbacks=callbacks,
        steps_per_epoch=int(NN_structure["nJets_train"]) / NN_structure["batch_size"]
        if "nJets_train" in NN_structure and NN_structure["nJets_train"] is not None
        else metadata["n_jets"] / NN_structure["batch_size"],
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
