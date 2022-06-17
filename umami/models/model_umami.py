#!/usr/bin/env python
"""Keras model of the UMAMI tagger."""
from umami.configuration import logger  # isort:skip
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
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.preprocessing_tools import GetVariableDict


def create_umami_model(
    train_config: object,
    input_shape: tuple,
    njet_features: int,
    continue_training: bool = False,
):
    """Keras model definition of UMAMI tagger.

    Parameters
    ----------
    train_config : object
        training config
    input_shape : tuple
        dataset input shape
    njet_features: int
        number of jet features
    continue_training : bool, optional
        Decide, if the training is continued using the latest
        model file, by default False

    Returns
    -------
    keras model
        UMAMI keras model
    int
        Number of epochs
    int
        Starting epoch number
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.NN_structure

    # Set NN options
    batch_norm = nn_structure["Batch_Normalisation"]
    dropout = nn_structure["dropout"]
    class_labels = nn_structure["class_labels"]

    # Check if a prepared model is used or not
    umami, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if umami is None:
        logger.info("No modelfile provided! Initialising a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(nn_structure["DIPS_ppm_units"]):

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
        f_net = utf.Sum(name="Sum")(tdd)

        for j, (f_nodes, drop_rate) in enumerate(
            zip(
                nn_structure["DIPS_dense_units"],
                [dropout] * len(nn_structure["DIPS_dense_units"][:-1]) + [0],
            )
        ):

            f_net = Dense(f_nodes, activation="linear", name=f"F{j}_Dense")(f_net)
            if batch_norm:
                f_net = BatchNormalization(name=f"F{j}_BatchNormalization")(f_net)
            if dropout != 0:
                f_net = Dropout(rate=drop_rate, name=f"F{j}_Dropout")(f_net)
            f_net = Activation(activations.relu, name=f"F{j}_ReLU")(f_net)

        dips_output = Dense(len(class_labels), activation="softmax", name="dips")(f_net)

        # Input layer
        jet_inputs = Input(shape=(njet_features,))

        # Adding the intermediate dense layers for DL1
        x_net = jet_inputs
        for unit in nn_structure["intermediate_units"]:
            x_net = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x_net)
            x_net = BatchNormalization()(x_net)
            x_net = Activation("relu")(x_net)

        # Concatenate the inputs
        x_net = Concatenate()([f_net, x_net])

        # Loop to initialise the hidden layers
        for unit in nn_structure["DL1_units"]:
            x_net = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x_net)
            x_net = BatchNormalization()(x_net)
            x_net = Activation("relu")(x_net)

        jet_output = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="umami",
        )(x_net)

        umami = Model(
            inputs=[trk_inputs, jet_inputs], outputs=[dips_output, jet_output]
        )

    if load_optimiser is False:
        # Set optimier and loss
        model_optimiser = Adam(learning_rate=nn_structure["lr"])
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={"dips": nn_structure["dips_loss_weight"], "umami": 1},
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    return umami, nn_structure["epochs"], init_epoch


def umami_tagger(args, train_config, preprocess_config):
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
    nn_structure = train_config.NN_structure
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation

    # Init a list for the callbacks
    callbacks = []

    # Set the tracks collection name
    tracks_name = train_config.tracks_name

    # Get needed variable from the train config
    working_point = (
        float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
    )
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
        with h5py.File(train_config.train_file, "r") as h5_file:
            (
                metadata["n_jets"],
                metadata["n_trks"],
                metadata["n_trk_features"],
            ) = h5_file[f"X_{tracks_name}_train"].shape
            _, metadata["n_dim"] = h5_file["Y_train"].shape
            _, metadata["n_jet_features"] = h5_file["X_train"].shape
            if exclude is not None:
                metadata["n_jet_features"] -= len(excluded_var)
            logger.debug(
                f"Input shape of jet training set: {metadata['n_jet_features']}"
            )

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
                    n_jets=int(nn_structure["nJets_train"])
                    if "nJets_train" in nn_structure
                    and nn_structure["nJets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
                    excluded_var=excluded_var,
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
            f"input file {train_config.train_file} is neither a .h5 file nor a"
            " directory with TF Record Files. You should check this."
        )

    umami_model, _, init_epoch = create_umami_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        njet_features=metadata["n_jet_features"],
        continue_training=train_config.continue_training,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = nn_structure["epochs"]

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    if "LRR" in nn_structure and nn_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Set ModelCheckpoint as callback
    umami_model_checkpoint = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_model_checkpoint)

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        class_labels=nn_structure["class_labels"],
        main_class=nn_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=working_point,
        frac_dict=eval_params["frac_values"],
        n_jets=n_jets_val,
        continue_training=train_config.continue_training,
        batch_size=val_params["val_batch_size"],
        use_lrr=nn_structure["LRR"] if "LRR" in nn_structure else False,
    )

    # Append the callback
    callbacks.append(my_callback)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    umami_model.fit(
        train_dataset,
        epochs=n_epochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(
        #     [
        #         val_data_dict["X_valid_trk"],
        #         val_data_dict["X_valid"],
        #     ],
        #     val_data_dict["Y_valid"],
        # ),
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure["nJets_train"]) / nn_structure["batch_size"]
        if "nJets_train" in nn_structure and nn_structure["nJets_train"] is not None
        else metadata["n_jets"] / nn_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
