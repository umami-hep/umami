"""Keras model of the DL1 tagger."""
from umami.configuration import logger  # isort:skip
import os

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
from umami.preprocessing_tools import get_variable_dict

os.environ["KERAS_BACKEND"] = "tensorflow"


def create_dl1_model(
    train_config: object,
    input_shape: tuple,
    feature_connect_indices: list = None,
):
    """
    Constructs or loads the DL1 model

    Parameters
    ----------
    train_config : object
        Training configuration with nn_structure sub-dict
        giving the structure of the NN.
    input_shape : tuple
        Size of the input: (nFeatures,).
    feature_connect_indices : list
        List with features that are feeded in another time.

    Returns
    -------
    model : keras model
        Keras model.
    nn_structure.epochs :
        number of epochs to be trained
    init_epoch : int
        Starting epoch number
    """

    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure

    # Set NN options
    batch_norm = nn_structure.batch_normalisation
    class_labels = nn_structure.class_labels
    # Read dropout rates, set to zero if not specified
    dropout_rates = utt.get_dropout_rates("dropout_rate", "dense_sizes", nn_structure)

    # Check if a prepared model is used or not
    model, init_epoch, load_optimiser = utf.prepare_model(train_config=train_config)

    if model is None:
        # Define input
        inputs = Input(shape=input_shape)

        # Define layers
        layer = inputs
        for i, unit in enumerate(nn_structure.dense_sizes):
            layer = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(layer)

            # Add Batch Normalization if True
            if batch_norm:
                layer = BatchNormalization()(layer)

            # Add dropout layer if dropout rate is non-zero for this layer
            if dropout_rates[i] != 0:
                layer = Dropout(dropout_rates[i])(layer)

            # Define activation for the layer
            layer = Activation(nn_structure.activations[i])(layer)

        if feature_connect_indices is not None:
            layer = tf.keras.layers.concatenate(
                [layer, tf.gather(inputs, feature_connect_indices, axis=1)], 1
            )

        predictions = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
        )(layer)
        model = Model(inputs=inputs, outputs=predictions)

    if load_optimiser is False:
        # Compile model with given optimiser
        model_optimiser = Adam(learning_rate=nn_structure.learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print DL1 model summary when log level lower or equal INFO level
    if logger.level <= 20:
        model.summary()

    return model, nn_structure.epochs, init_epoch


def train_dl1(args, train_config):
    """Training handling of DL1 tagger.

    Parameters
    ----------
    args : parser args
        Arguments from command line parser
    train_config : object
        training configuration

    Raises
    ------
    ValueError
        If input is neither a h5 nor a directory.
    """

    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure
    val_params = train_config.validation_settings
    eval_params = train_config.evaluation_settings

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    working_point = val_params.working_point
    n_jets_val = val_params.n_jets

    # Load the excluded variables from train_config
    exclude = train_config.general.exclude
    logger.debug("Exclude option specified with values %s.", exclude)

    # Load variable config
    variable_config = get_variable_dict(train_config.general.var_dict)

    # Get excluded variables
    variables, _, excluded_var = utt.get_jet_feature_indices(
        variable_config["train_variables"], exclude
    )

    # Get variables to bring back in last layer
    feature_connect_indices = None
    if nn_structure.repeat_end is not None:
        repeat_end = nn_structure.repeat_end
        logger.info(
            "Repeating the following variables in the last layer %s", repeat_end
        )
        feature_connect_indices = utt.get_jet_feature_position(repeat_end, variables)

    if ".h5" in train_config.general.train_file:
        # Init a metadata dict
        metadata = {}

        generator_args = {
            "train_file_path": train_config.general.train_file,
            "x_name": "jets/inputs",
            "y_name": "jets/labels_one_hot",
            "n_jets": None,
            "batch_size": nn_structure.batch_size,
            "excluded_var": excluded_var,
            "sample_weights": nn_structure.use_sample_weights,
            "config_file": train_config.general.preprocess_config,
            "tracks_name": train_config.general.tracks_name,
        }

        # Get a small generator for metadata
        generator = utf.get_generator(
            "Dl1",
            generator_args,
            train_config.general.train_data_structure,
            small=True,
        )

        # Get the shapes for training from generator
        metadata["n_jets"] = generator.get_n_jets()
        generator_args["n_jets"] = (
            int(nn_structure.n_jets_train)
            if nn_structure.n_jets_train is not None
            else metadata["n_jets"]
        )
        metadata["n_dim"] = generator.get_n_dim()
        metadata["n_jet_features"] = generator.get_n_jet_features()
        logger.debug("Input shape of training set: %s", metadata["n_jet_features"])

        # make correct shapes for the tensors
        if nn_structure.use_sample_weights:
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

        # Get a full generator for training
        generator = utf.get_generator(
            "Dl1",
            generator_args,
            train_config.general.train_data_structure,
            small=False,
        )

        # Build train_datasets for training from generator
        train_dataset = (
            tf.data.Dataset.from_generator(
                generator,
                output_types=tensor_types,
                output_shapes=tensor_shapes,
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    elif os.path.isdir(train_config.general.train_file):
        train_dataset, metadata = utf.load_tfrecords_train_dataset(
            train_config=train_config
        )

    else:
        raise ValueError(
            f"input file {train_config.general.train_file} is neither a .h5 file nor a"
            " directory with TF Record Files. You should check this."
        )

    # Load model and epochs
    dl1_model, epochs, init_epoch = create_dl1_model(
        train_config=train_config,
        input_shape=(metadata["n_jet_features"],),
        feature_connect_indices=feature_connect_indices,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = epochs

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    # Set ModelCheckpoint as callback
    dl1_m_chkpt = ModelCheckpoint(
        f"{train_config.general.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure.batch_size,
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(dl1_m_chkpt)

    if nn_structure.lrr:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.get_learning_rate_reducer(nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Load validation data for callback
    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data(
            train_config=train_config,
            n_jets=n_jets_val,
            convert_to_tensor=True,
        )

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallback(
        model_name=train_config.general.model_name,
        class_labels=nn_structure.class_labels,
        main_class=nn_structure.main_class,
        val_data_dict=val_data_dict,
        target_beff=working_point,
        frac_dict=eval_params.frac_values,
        n_jets=n_jets_val,
        continue_training=train_config.general.continue_training,
        batch_size=val_params.val_batch_size,
        use_lrr=nn_structure.lrr,
    )

    # Append the callback
    callbacks.append(my_callback)

    logger.info("Start training")
    dl1_model.fit(
        x=train_dataset,
        epochs=n_epochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(val_data_dict["X_valid"], val_data_dict["Y_valid"]),
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure.n_jets_train) / nn_structure.batch_size
        if nn_structure.n_jets_train is not None
        else metadata["n_jets"] / nn_structure.batch_size,
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
