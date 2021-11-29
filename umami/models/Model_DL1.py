from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.tools import yaml_loader

os.environ["KERAS_BACKEND"] = "tensorflow"


def DL1_model(train_config, input_shape):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    # Load model from file if defined
    if train_config.model_file is not None:
        logger.info(f"Loading model from: {train_config.model_file}")
        model = load_model(train_config.model_file, compile=False)

    else:
        # Define input
        inputs = Input(shape=input_shape)

        # Define layers
        x = inputs
        for i, unit in enumerate(NN_structure["dense_sizes"]):
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x)

            # Add Batch Normalization if True
            if batch_norm:
                x = BatchNormalization()(x)

            # Add dropout if != 0
            if dropout != 0:
                x = Dropout(NN_structure["dropout_rate"][i])(x)

            # Define activation for the layer
            x = Activation(NN_structure["activations"][i])(x)

        predictions = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
        )(x)
        model = Model(inputs=inputs, outputs=predictions)

    # Print DL1 model summary when log level lower or equal INFO level
    if logger.level <= 20:
        model.summary()

    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return model, NN_structure["epochs"]


def TrainLargeFile(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation

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

    # Get the shapes for training
    with h5py.File(train_config.train_file, "r") as f:
        nJets, nFeatures = f["X_train"].shape
        nJets, nDim = f["Y_train"].shape
        if excluded_var is not None:
            nFeatures -= len(excluded_var)

        if NN_structure["nJets_train"] is not None:
            nJets = int(NN_structure["nJets_train"])

    # Print how much jets are used
    logger.info(f"Number of Jets used for training: {nJets}")

    # Build train_datasets for training
    train_dataset = (
        tf.data.Dataset.from_generator(
            utf.dl1_generator(
                train_file_path=train_config.train_file,
                X_Name="X_train",
                Y_Name="Y_train",
                n_jets=nJets,
                batch_size=NN_structure["batch_size"],
                excluded_var=excluded_var,
            ),
            (tf.float32, tf.float32),
            (
                tf.TensorShape([None, nFeatures]),
                tf.TensorShape([None, nDim]),
            ),
        )
        .repeat()
        .prefetch(3)
    )

    # Load model and epochs
    model, epochs = DL1_model(
        train_config=train_config, input_shape=(nFeatures,)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    dl1_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files"
        + "/model_epoch{epoch:03d}.h5",
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
        val_data_dict = utt.load_validation_data_dl1(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=int(Val_params["n_jets"]),
        )

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
    history = model.fit(
        x=train_dataset,
        epochs=nEpochs,
        validation_data=(val_data_dict["X_valid"], val_data_dict["Y_valid"]),
        callbacks=[dl1_mChkPt, reduce_lr, my_callback],
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

    logger.info(f"Models saved {train_config.model_name}")


def TrainLargeFileZeuthen(args, train_config, preprocess_config):
    if is_qsub_available():
        args.model_name = train_config.model_name
        args.dl1 = True
        submit_zeuthen(args)
    else:
        logger.warning(
            "No Zeuthen batch system found, training locally instead."
        )
        TrainLargeFile(args, train_config, preprocess_config)
