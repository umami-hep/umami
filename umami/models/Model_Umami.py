#!/usr/bin/env python
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
import yaml
from tensorflow.keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Masking,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.tools import yaml_loader


def Umami_model(train_config=None, input_shape=None, njet_features=None):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        umami = load_model(
            train_config.model_file, {"Sum": utf.Sum}, compile=False
        )

    else:
        logger.info("No modelfile provided! Initialize a new one!")

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
                tdd = TimeDistributed(
                    Dropout(rate=dropout), name=f"Phi{i}_Dropout"
                )(tdd)

            tdd = TimeDistributed(
                Activation(activations.relu), name=f"Phi{i}_ReLU"
            )(tdd)

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

        dips_output = Dense(
            len(class_labels), activation="softmax", name="dips"
        )(F)

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

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    # Set optimier and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    umami.compile(
        loss="categorical_crossentropy",
        loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return umami, NN_structure["epochs"]


def Umami(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation

    val_data_dict = None
    if Val_params["n_jets"] > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=int(Val_params["n_jets"]),
        )

    # Load the excluded variables from train_config
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    else:
        exclude = None

    # Load variable config
    with open(train_config.var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Get excluded variables
    _, _, excluded_var = utt.get_jet_feature_indices(
        variable_config["train_variables"], exclude
    )

    if ".h5" in train_config.train_file:
        with h5py.File(train_config.train_file, "r") as f:
            nJets, nTrks, nFeatures = f["X_trk_train"].shape
            nJets, nDim = f["Y_train"].shape
            nJets, njet_features = f["X_train"].shape
    elif os.path.isdir(train_config.train_file):
        train_file_names = os.listdir(train_config.train_file)
        for train_file_name in train_file_names:
            if not (".tfrecord" in train_file_name) and not (
                train_file_name == "metadata.json"
            ):
                raise ValueError(
                    f"input file {train_config.train_file} is neither a .h5 file nor a directory with TF Record Files. You should check this."
                )
        if "metadata.json" not in train_file_names:
            raise KeyError("No metadata file in directory.")
        try:
            nfiles = train_config.config["nfiles"]
        except KeyError:
            logger.warning(
                "no number of files to be loaded in parallel defined. Set to 5"
            )
            nfiles = 5
        tfrecord_reader = utf.TFRecordReader(
            train_config.train_file, NN_structure["batch_size"], nfiles
        )
        train_dataset = tfrecord_reader.load_Dataset()
        metadata_name = (train_config.train_file + "/metadata.json").replace(
            "//", "/"
        )
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)
            nJets = metadata["nJets"]
            nTrks = metadata["nTrks"]
            nFeatures = metadata["nFeatures"]
            njet_features = metadata["njet_features"]
            nDim = metadata["nDim"]
    else:
        raise ValueError(
            f"input file {train_config.train_file} is neither a .h5 file nor a directory with TF Record Files. You should check this."
        )

    if NN_structure["nJets_train"] is not None:
        nJets = int(NN_structure["nJets_train"])

    logger.info(f"nJets: {nJets}, nTrks: {nTrks}")
    logger.info(f"nFeatures: {nFeatures}, njet_features: {njet_features}")

    umami, _ = Umami_model(
        train_config=train_config,
        input_shape=(nTrks, nFeatures),
        njet_features=njet_features,
    )

    if ".h5" in train_config.train_file:
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.umami_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    X_trk_Name="X_trk_train",
                    Y_Name="Y_train",
                    n_jets=nJets,
                    batch_size=NN_structure["batch_size"],
                    excluded_var=excluded_var,
                ),
                output_types=(
                    {"input_1": tf.float32, "input_2": tf.float32},
                    tf.float32,
                ),
                output_shapes=(
                    {
                        "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                        "input_2": tf.TensorShape([None, njet_features]),
                    },
                    tf.TensorShape([None, nDim]),
                ),
            )
            .repeat()
            .prefetch(3)
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

    # Set ModelCheckpoint as callback
    umami_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files"
        + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
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

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    history = umami.fit(
        train_dataset,
        epochs=nEpochs,
        validation_data=(
            [
                val_data_dict["X_valid_trk"],
                val_data_dict["X_valid"],
            ],
            val_data_dict["Y_valid"],
        ),
        callbacks=[umami_mChkPt, reduce_lr, my_callback],
        steps_per_epoch=nJets / NN_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
    )

    # Dump dict into json
    logger.info(
        f"Dumping history file to {train_config.model_name}/history.json"
    )

    # Make the history dict the same shape as the dict from the callbacks
    hist_dict = utt.prepare_history_dict(history.history)

    # Dump history file to json
    with open(f"{train_config.model_name}/history.json", "w") as outfile:
        json.dump(hist_dict, outfile, indent=4)


def UmamiZeuthen(args, train_config, preprocess_config):
    if is_qsub_available():
        args.model_name = train_config.model_name
        args.umami = True
        submit_zeuthen(args)
    else:
        logger.warning(
            "No Zeuthen batch system found, training locally instead."
        )
        Umami(args, train_config, preprocess_config)
