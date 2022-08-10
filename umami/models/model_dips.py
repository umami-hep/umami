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
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

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
        for i, phi_nodes in enumerate(NN_structure["ppm_sizes"]):

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

        # Define the main dips structure
        for j, (F_nodes, p) in enumerate(
            zip(
                NN_structure["dense_sizes"],
                [dropout] * len(NN_structure["dense_sizes"][:-1]) + [0],
            )
        ):

            F = Dense(F_nodes, activation="linear", name=f"F{j}_Dense")(F)
            if batch_norm:
                F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
            if dropout != 0:
                F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
            F = Activation(activations.relu, name=f"F{j}_ReLU")(F)

        # Set output and activation function
        output = Dense(len(class_labels), activation="softmax", name="Jet_class")(F)
        dips = Model(inputs=trk_inputs, outputs=output)

    if load_optimiser is False:
        # Set optimier and loss
        model_optimiser = Adam(learning_rate=NN_structure["lr"])
        dips.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print Dips model summary when log level lower or equal INFO level
    if logger.level <= 20:
        dips.summary()

    return dips, NN_structure["epochs"], init_epoch


def Dips(args, train_config):
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
    nn_structure = train_config.NN_structure
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation
    tracks_name = train_config.tracks_name

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    WP = float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
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
                utf.dips_generator(
                    train_file_path=train_config.train_file,
                    X_trk_Name=f"X_{tracks_name}_train",
                    Y_Name="Y_train",
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
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    dips_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(dips_mChkPt)

    if "LRR" in nn_structure and nn_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Load validation data for callback
    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data_dips(
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
    dips_model.fit(
        train_dataset,
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
