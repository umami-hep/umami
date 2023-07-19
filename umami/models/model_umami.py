"""Keras model of the UMAMI tagger."""
from umami.configuration import logger  # isort:skip
import os

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
from umami.preprocessing_tools import get_variable_dict


def create_umami_model(
    train_config: object,
    input_shape: tuple,
    njet_features: int,
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
    nn_structure = train_config.nn_structure

    # Set NN options
    batch_norm = nn_structure.batch_normalisation
    dropout_rates_phi = utt.get_dropout_rates(
        "dropout_rate_phi", "dips_ppm_units", nn_structure
    )
    dropout_rates_f = utt.get_dropout_rates(
        "dropout_rate_f", "dips_dense_units", nn_structure
    )
    dropout_rates_dl1 = utt.get_dropout_rates("dropout_rate", "dl1_units", nn_structure)
    class_labels = nn_structure.class_labels

    # Check if a prepared model is used or not
    umami, init_epoch, load_optimiser = utf.prepare_model(train_config=train_config)

    if umami is None:
        logger.info("No modelfile provided! Initialising a new one!")

        # Set the track input
        trk_inputs = Input(shape=input_shape)

        # Masking the missing tracks
        masked_inputs = Masking(mask_value=0)(trk_inputs)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, (phi_nodes, dropout_rate_phi) in enumerate(
            zip(nn_structure.dips_ppm_units, dropout_rates_phi)
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
        f_net = utf.Sum(name="Sum")(tdd)

        # Define the main dips structure
        for j, (f_nodes, dropout_rate_f) in enumerate(
            zip(nn_structure.dips_dense_units, dropout_rates_f)
        ):
            f_net = Dense(f_nodes, activation="linear", name=f"F{j}_Dense")(f_net)
            if batch_norm:
                f_net = BatchNormalization(name=f"F{j}_BatchNormalization")(f_net)
            if dropout_rate_f != 0:
                f_net = Dropout(rate=dropout_rate_f, name=f"F{j}_Dropout")(f_net)
            f_net = Activation(activations.relu, name=f"F{j}_ReLU")(f_net)

        dips_output = Dense(len(class_labels), activation="softmax", name="dips")(f_net)

        # Input layer
        jet_inputs = Input(shape=(njet_features,))

        # Adding the intermediate dense layers for DL1
        x_net = jet_inputs
        for unit in nn_structure.intermediate_units:
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
        for i, (unit, dropout_rate_dl1) in enumerate(
            zip(nn_structure.dl1_units, dropout_rates_dl1)
        ):
            x_net = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(x_net)

            # Add Batch Normalization if True
            if batch_norm:
                x_net = BatchNormalization()(x_net)

            # Add dropout layer if dropout rate is non-zero for this layer
            if dropout_rate_dl1 != 0:
                x_net = Dropout(dropout_rate_dl1)(x_net)

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
        model_optimiser = Adam(learning_rate=nn_structure.learning_rate)
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={"dips": nn_structure.dips_loss_weight, "umami": 1},
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    return umami, nn_structure.epochs, init_epoch


def train_umami(args, train_config):
    """Training handling of UMAMI tagger.

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

    # Set the tracks collection name
    tracks_name = train_config.general.tracks_name

    # Get needed variable from the train config
    working_point = val_params.working_point
    n_jets_val = val_params.n_jets

    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data(
            train_config=train_config,
            n_jets=n_jets_val,
            convert_to_tensor=True,
        )

    # Load the excluded variables from train_config
    exclude = train_config.general.exclude
    logger.debug("Exclude option specified with values %s.", exclude)

    # Load variable config
    variable_config = get_variable_dict(train_config.general.var_dict)

    # Get excluded variables
    _, _, excluded_var = utt.get_jet_feature_indices(
        variable_config["train_variables"], exclude
    )

    if ".h5" in train_config.general.train_file:
        # Init a metadata dict
        metadata = {}

        generator_args = {
            "train_file_path": train_config.general.train_file,
            "x_name": "jets/inputs",
            "x_trk_name": f"{tracks_name}/inputs",
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
            "Umami",
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
        metadata["n_trk_features"] = generator.get_n_trk_features()
        metadata["n_trks"] = generator.get_n_trks()
        logger.debug("Input shape of training set: %s", metadata["n_jet_features"])

        if nn_structure.use_sample_weights:
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

        generator = utf.get_generator(
            "Umami",
            generator_args,
            train_config.general.train_data_structure,
            small=False,
        )

        # Get training set from generator
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

    umami_model, _, init_epoch = create_umami_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        njet_features=metadata["n_jet_features"],
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = nn_structure.epochs

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    if nn_structure.lrr:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.get_learning_rate_reducer(nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Set ModelCheckpoint as callback
    umami_model_checkpoint = ModelCheckpoint(
        f"{train_config.general.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure.batch_size,
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_model_checkpoint)

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
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
        steps_per_epoch=int(nn_structure.n_jets_train) / nn_structure.batch_size
        if nn_structure.n_jets_train is not None
        else metadata["n_jets"] / nn_structure.batch_size,
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
