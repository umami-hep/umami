"""Keras model of the DIPS tagger."""
from umami.configuration import logger  # isort:skip
import os

import h5py
import tensorflow as tf
from tensorflow.keras import activations  # pylint: disable=import-error
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    BatchNormalization,
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


def create_dips_model(
    train_config: object,
    input_shape: tuple,
    continue_training: bool = False,
):
    """Keras model definition of DIPS.

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
        Dips keras model
    int
        Number of epochs
    int
        Starting epoch number

    Raises
    ------
    ValueError
        If the deprecated global `dropout` parameter is used.
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure

    # Set NN options
    batch_norm = nn_structure["batch_normalisation"]
    class_labels = nn_structure["class_labels"]

    dropout_rates_phi = utt.get_dropout_rates(
        "dropout_rate_phi", "ppm_sizes", nn_structure
    )
    dropout_rates_f = utt.get_dropout_rates("dropout_rate", "dense_sizes", nn_structure)
    if "dropout" in nn_structure:
        raise ValueError(
            "The global parameter `dropout` is deprecated. Please specify a "
            "dropout rate for each layer."
        )
    # Check if a prepared model is used or not
    dips, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if dips is None:
        logger.info("No modelfile provided! Initialising a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, (phi_nodes, dropout_rate_phi) in enumerate(
            zip(nn_structure["ppm_sizes"], dropout_rates_phi)
        ):

            tdd = TimeDistributed(
                Dense(phi_nodes, activation="linear"), name=f"Phi{i}_Dense"
            )(tdd)

            if batch_norm:
                tdd = TimeDistributed(
                    BatchNormalization(), name=f"Phi{i}_BatchNormalization"
                )(tdd)

            if dropout_rate_phi != 0:
                tdd = TimeDistributed(
                    Dropout(rate=dropout_rate_phi), name=f"Phi{i}_Dropout"
                )(tdd)

            tdd = TimeDistributed(Activation(activations.relu), name=f"Phi{i}_ReLU")(
                tdd
            )

        # This is where the magic happens... sum up the track features!
        f_layer = utf.Sum(name="Sum")(tdd)

        # Define the main dips structure
        for j, (f_nodes, dropout_rate_f) in enumerate(
            zip(nn_structure["dense_sizes"], dropout_rates_f)
        ):

            f_layer = Dense(f_nodes, activation="linear", name=f"F{j}_Dense")(f_layer)
            if batch_norm:
                f_layer = BatchNormalization(name=f"F{j}_BatchNormalization")(f_layer)
            if dropout_rate_f != 0:
                f_layer = Dropout(rate=dropout_rate_f, name=f"F{j}_Dropout")(f_layer)
            f_layer = Activation(activations.relu, name=f"F{j}_ReLU")(f_layer)

        # Set output and activation function
        output = Dense(len(class_labels), activation="softmax", name="Jet_class")(
            f_layer
        )
        dips = Model(inputs=trk_inputs, outputs=output)

    if load_optimiser is False:
        # Set optimier and loss
        model_optimiser = Adam(learning_rate=nn_structure["lr"])
        dips.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print Dips model summary when log level lower or equal INFO level
    if logger.level <= 20:
        dips.summary()

    return dips, nn_structure["epochs"], init_epoch


def train_dips(args, train_config):
    """Training handling of DIPS tagger.

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
    tracks_name = train_config.tracks_name

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    working_point = (
        float(val_params["working_point"])
        if "working_point" in val_params
        else float(eval_params["working_point"])
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
        with h5py.File(train_config.train_file, "r") as f_train:
            (
                metadata["n_jets"],
                metadata["n_trks"],
                metadata["n_trk_features"],
            ) = f_train[f"{tracks_name}/inputs"].shape
            _, metadata["n_dim"] = f_train["jets/labels_one_hot"].shape

        if nn_structure["use_sample_weights"]:
            tensor_types = (tf.float32, tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, metadata["n_trks"], metadata["n_trk_features"]]),
                tf.TensorShape([None, metadata["n_dim"]]),
                tf.TensorShape([None]),
            )

        else:
            tensor_types = (tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, metadata["n_trks"], metadata["n_trk_features"]]),
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        # Get training set from generator
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.DipsGenerator(
                    train_file_path=train_config.train_file,
                    x_trk_name=f"{tracks_name}/inputs",
                    y_name="jets/labels_one_hot",
                    n_jets=int(nn_structure["n_jets_train"])
                    if "n_jets_train" in nn_structure
                    and nn_structure["n_jets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
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
            f"input file {train_config.train_file} is neither a .h5 file nor "
            "a directory with TF Record Files. You should check this."
        )

    # Init dips model
    dips_model, epochs, init_epoch = create_dips_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        continue_training=train_config.continue_training,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = epochs

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    # Set ModelCheckpoint as callback
    dips_m_chkpt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(dips_m_chkpt)

    if "lrr" in nn_structure and nn_structure["lrr"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.get_learning_rate_reducer(**nn_structure)

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
        model_name=train_config.model_name,
        class_labels=nn_structure["class_labels"],
        main_class=nn_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=working_point,
        frac_dict=eval_params["frac_values"],
        n_jets=n_jets_val,
        continue_training=train_config.continue_training,
        batch_size=val_params["val_batch_size"],
        use_lrr=nn_structure["lrr"] if "lrr" in nn_structure else False,
    )

    # Append the callback
    callbacks.append(my_callback)

    logger.info("Start training")
    dips_model.fit(
        train_dataset,
        epochs=n_epochs,
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
