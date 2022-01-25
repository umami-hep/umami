"""Keras model of the DIPS tagger."""
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.callbacks import (
    ModelCheckpoint,  # pylint: disable=no-name-in-module
)
from tensorflow.keras.layers import (  # pylint: disable=no-name-in-module
    Activation,
    BatchNormalization,
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


def Dips_model(train_config=None, input_shape=None):
    """Keras model definition of DIPS.

    Parameters
    ----------
    train_config : object, optional
        training config, by default None
    input_shape : tuple, optional
        dataset input shape, by default None

    Returns
    -------
    keras model
        Dips keras model
    int
        number of epochs
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        dips = load_model(train_config.model_file, {"Sum": utf.Sum}, compile=False)

    else:
        logger.info("No modelfile provided! Initialize a new one!")

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

    # Print Dips model summary when log level lower or equal INFO level
    if logger.level <= 20:
        dips.summary()

    # Set optimier and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    dips.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return dips, NN_structure["epochs"]


def Dips(args, train_config, preprocess_config):
    """Training handling of DIPS tagger.

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
        if inputs are neither in .h5 or directory with tfrecords files given
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation

    if ".h5" in train_config.train_file:
        # Get the shapes for training
        with h5py.File(train_config.train_file, "r") as f:
            nJets, nTrks, nFeatures = f["X_trk_train"].shape
            nJets, nDim = f["Y_train"].shape

            if NN_structure["nJets_train"] is not None:
                nJets = int(NN_structure["nJets_train"])

        if NN_structure["use_sample_weights"]:
            tensor_types = (tf.float32, tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, nTrks, nFeatures]),
                tf.TensorShape([None, nDim]),
                tf.TensorShape([None]),
            )

        else:
            tensor_types = (tf.float32, tf.float32)
            tensor_shapes = (
                tf.TensorShape([None, nTrks, nFeatures]),
                tf.TensorShape([None, nDim]),
            )

        # Get training set from generator
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.dips_generator(
                    train_file_path=train_config.train_file,
                    X_trk_Name="X_trk_train",
                    Y_Name="Y_train",
                    n_jets=nJets,
                    batch_size=NN_structure["batch_size"],
                    sample_weights=NN_structure["use_sample_weights"],
                ),
                tensor_types,
                tensor_shapes,
            )
            .repeat()
            .prefetch(3)
        )

    elif os.path.isdir(train_config.train_file):

        # Get the files in dir
        train_file_names = os.listdir(train_config.train_file)

        # Loop over files in dir
        for train_file_name in train_file_names:

            # Check if file is tfrecords or .h5
            if not (".tfrecord" in train_file_name) and not (
                train_file_name == "metadata.json"
            ):
                raise ValueError(
                    f"input file {train_config.train_file} is neither a "
                    ".h5 file nor a directory with TF Record Files. "
                    "You should check this."
                )

        # Check if train file is in metadata
        if "metadata.json" not in train_file_names:
            raise KeyError("No metadata file in directory.")

        # Check if nfiles is given. Otherwise set to 5
        try:
            nfiles = train_config.config["nfiles"]

        except KeyError:
            logger.warning(
                "no number of files to be loaded in parallel defined. Set to 5"
            )
            nfiles = 5

        # Get the tfrecords
        tfrecord_reader = utf.TFRecordReader(
            train_config.train_file, NN_structure["batch_size"], nfiles
        )

        # Load the dataset from reader
        train_dataset = tfrecord_reader.load_Dataset()

        # Get the metadata name
        metadata_name = (train_config.train_file + "/metadata.json").replace("//", "/")

        # Load metadata in file
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)
            nJets = metadata["nJets"]
            nTrks = metadata["nTrks"]
            nFeatures = metadata["nFeatures"]
            nDim = metadata["nDim"]
    else:
        raise ValueError(
            f"input file {train_config.train_file} is neither a .h5 file nor "
            "a directory with TF Record Files. You should check this."
        )

    # Print how much jets are used
    logger.info(f"Number of Jets used for training: {nJets}")

    # Init dips model
    dips, epochs = Dips_model(train_config=train_config, input_shape=(nTrks, nFeatures))

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
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    if "LRR" in NN_structure and NN_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**NN_structure)

    # Load validation data for callback
    val_data_dict = None
    if Val_params["n_jets"] > 0:
        val_data_dict = utt.load_validation_data_dips(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=int(Val_params["n_jets"]),
            convert_to_tensor=True,
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
    history = dips.fit(
        train_dataset,
        epochs=nEpochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=(val_data_dict["X_valid"], val_data_dict["Y_valid"]),
        callbacks=[dips_mChkPt, reduce_lr, my_callback],
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
