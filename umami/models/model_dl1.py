"""Keras model of the DL1 tagger."""
from umami.configuration import logger  # isort:skip
import os

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.preprocessing_tools import GetVariableDict

os.environ["KERAS_BACKEND"] = "tensorflow"


def create_dl1_model(
    train_config: object,
    input_shape: tuple,
    feature_connect_indices: list = None,
    continue_training: bool = False,
):
    """
    Constructs or loads the DL1 model

    Parameters
    ----------
    train_config : object
        Training configuration with NN_structure sub-dict
        giving the structure of the NN.
    input_shape : tuple
        Size of the input: (nFeatures,).
    feature_connect_indices : list
        List with features that are feeded in another time.
    continue_training : bool, optional
        Decide, if the training is continued using the latest
        model file, by default False

    Returns
    -------
    model : keras model
        Keras model.
    NN_structure["epochs"] :
        number of epochs to be trained
    init_epoch : int
        Starting epoch number
    """

    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    # Check if a prepared model is used or not
    model, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if model is None:
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

    if load_optimiser is False:
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

    return model, NN_structure["epochs"], init_epoch


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

    # Get needed variable from the train config
    WP = float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
    n_jets_val = (
        int(val_params["n_jets"])
        if "n_jets" in val_params
        else int(eval_params["n_jets"])
    )

    # Load the excluded variables from train_config
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]
        logger.debug("Exclude option specified with values %s.", exclude)

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
    if "repeat_end" in nn_structure and nn_structure["repeat_end"] is not None:
        repeat_end = nn_structure["repeat_end"]
        logger.info(
            "Repeating the following variables in the last layer %s", repeat_end
        )
        feature_connect_indices = utt.get_jet_feature_position(repeat_end, variables)

    if ".h5" in train_config.train_file:
        # Init a metadata dict
        metadata = {}

        # Get the shapes for training
        with h5py.File(train_config.train_file, "r") as f:
            metadata["n_jets"], metadata["n_dim"] = f["Y_train"].shape
            _, metadata["n_jet_features"] = f["X_train"].shape
            if exclude is not None:
                metadata["n_jet_features"] -= len(excluded_var)
            logger.debug("Input shape of training set: %s", metadata["n_jet_features"])

        if nn_structure["use_sample_weights"]:
            tensor_types = (tf.float32, tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, metadata["n_jet_features"]]),
                tf.TensorShape([None, metadata["n_dim"]]),
                tf.TensorShape([None]),
            )
        else:
            tensor_types = (tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, metadata["n_jet_features"]]),
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        # Build train_datasets for training
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.dl1_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    Y_Name="Y_train",
                    n_jets=int(nn_structure["n_jets_train"])
                    if "n_jets_train" in nn_structure
                    and nn_structure["n_jets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
                    excluded_var=excluded_var,
                    sample_weights=nn_structure["use_sample_weights"],
                ),
                tensor_types,
                tensor_shapes,
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

    # Load model and epochs
    dl1_model, epochs, init_epoch = create_dl1_model(
        train_config=train_config,
        input_shape=(metadata["n_jet_features"],),
        feature_connect_indices=feature_connect_indices,
        continue_training=train_config.continue_training,
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
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(dl1_mChkPt)

    if "LRR" in nn_structure and nn_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Load validation data for callback
    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data_dl1(
            train_config=train_config,
            preprocess_config=preprocess_config,
            n_jets=n_jets_val,
        )

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallback(
        model_name=train_config.model_name,
        class_labels=nn_structure["class_labels"],
        main_class=nn_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=WP,
        frac_dict=eval_params["frac_values"],
        n_jets=n_jets_val,
        continue_training=train_config.continue_training,
        batch_size=val_params["val_batch_size"],
        use_lrr=nn_structure["LRR"] if "LRR" in nn_structure else False,
    )

    # Append the callback
    callbacks.append(my_callback)

    logger.info("Start training")
    dl1_model.fit(
        x=train_dataset,
        epochs=nEpochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(val_data_dict["X_valid"], val_data_dict["Y_valid"]),
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure["n_jets_train"]) / nn_structure["batch_size"]
        if "n_jets_train" in nn_structure and nn_structure["n_jets_train"] is not None
        else metadata["n_jets"] / nn_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
