"""Keras model of the DL1 tagger."""
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
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
from umami.preprocessing_tools import GetVariableDict

os.environ["KERAS_BACKEND"] = "tensorflow"


def DL1_model(train_config, input_shape, feature_connect_indices=None):
    """
    Constructs or loads the DL1 model

    Parameters
    ----------
    train_config : dict
        Training configuration with NN_structure sub-dict
        giving the structure of the NN.
    input_shape : tuple
        Size of the input: (nFeatures,).

    Returns
    -------
    model: keras tensorflow model.
    NN_structure["epochs"]: number of epochs to be trained
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

    # Load model from file if defined
    if train_config.model_file is not None:
        logger.info(f"Loading model from: {train_config.model_file}")
        model = load_model(train_config.model_file, compile=load_optimiser)

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

        if feature_connect_indices is not None:
            x = tf.keras.layers.concatenate(
                [x, tf.gather(inputs, feature_connect_indices, axis=1)], 1
            )

        predictions = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
        )(x)
        model = Model(inputs=inputs, outputs=predictions)

        # Compile model with given optimiser
        model_optimiser = Adam(learning_rate=NN_structure["lr"])
        model.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print DL1 model summary when log level lower or equal INFO level
    if logger.level <= 20:
        model.summary()

    return model, NN_structure["epochs"]


def TrainLargeFile(args, train_config, preprocess_config):
    """Training handling of DL1 tagger.

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

    # Load the excluded variables from train_config
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    else:
        exclude = None

    # Load variable config
    variable_config = GetVariableDict(train_config.var_dict)

    # Get excluded variables
    variables, _, excluded_var = utt.get_jet_feature_indices(
        variable_config["train_variables"], exclude
    )

    # Get variables to bring back in last layer
    feature_connect_indices = None
    if "repeat_end" in NN_structure and NN_structure["repeat_end"] is not None:
        repeat_end = NN_structure["repeat_end"]
        logger.info(f"Repeating the following variables in the last layer {repeat_end}")
        feature_connect_indices = utt.get_jet_feature_position(repeat_end, variables)

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

    # pass correct tensor types/shapes depending on using sample weights
    if NN_structure["use_sample_weights"]:
        tensor_types = (tf.float32, tf.float32, tf.float32)
        tensor_shapes = (
            tf.TensorShape([None, nFeatures]),
            tf.TensorShape([None, nDim]),
            tf.TensorShape([None]),
        )
    else:
        tensor_types = (tf.float32, tf.float32)
        tensor_shapes = (
            tf.TensorShape([None, nFeatures]),
            tf.TensorShape([None, nDim]),
        )
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
                sample_weights=NN_structure["use_sample_weights"],
            ),
            tensor_types,
            tensor_shapes,
        )
        .repeat()
        .prefetch(3)
    )

    # Load model and epochs
    model, epochs = DL1_model(
        train_config=train_config,
        input_shape=(nFeatures,),
        feature_connect_indices=feature_connect_indices,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    dl1_mChkPt = ModelCheckpoint(
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
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(val_data_dict["X_valid"], val_data_dict["Y_valid"]),
        callbacks=[dl1_mChkPt, reduce_lr, my_callback],
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

    logger.info(f"Models saved {train_config.model_name}")
