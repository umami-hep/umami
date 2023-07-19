"""Keras model of the CADS tagger."""
from umami.configuration import logger  # isort:skip
import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt


def create_cads_model(
    train_config: object,
    input_shape: tuple,
):
    """Keras model definition of CADS.

    Parameters
    ----------
    train_config : object
        training config
    input_shape : tuple
        dataset input shape

    Returns
    -------
    keras model
        CADS keras model
    int
        Number of epochs
    int
        Starting epoch number
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure

    # Check if a prepared model is used or not
    cads, init_epoch, load_optimiser = utf.prepare_model(train_config=train_config)

    if cads is None:
        # Init a new cads/dips attention model
        cads = utf.deepsets_model(
            repeat_input_shape=input_shape,
            num_conditions=nn_structure.n_conditions,
            num_set_features=nn_structure.ppm_sizes[-1],
            sets_nodes=nn_structure.ppm_sizes[:-1],
            classif_nodes=nn_structure.dense_sizes,
            classif_output=len(nn_structure.class_labels),
            pooling=nn_structure.pooling,
            attention_nodes=nn_structure.attention_sizes,
            condition_sets=nn_structure.ppm_condition,
            condition_attention=nn_structure.attention_condition,
            condition_classifier=nn_structure.dense_condition,
            shortcut_inputs=False,
            sets_batch_norm=nn_structure.batch_normalisation,
            classif_batch_norm=nn_structure.batch_normalisation,
            activation="relu",
            attention_softmax=False,
        )

    if load_optimiser is False:
        # Set optimiser and loss
        model_optimiser = Adam(learning_rate=nn_structure.learning_rate)
        cads.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimiser,
            metrics=["accuracy"],
        )

    # Print CADS model summary when log level lower or equal INFO level
    if logger.level <= 20:
        cads.summary()

    return cads, nn_structure.epochs, init_epoch


def train_cads(args, train_config):
    """
    Training handling of CADS.

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
    tracks_name = train_config.general.tracks_name

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    working_point = val_params.working_point
    n_jets_val = val_params.n_jets

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
            "n_conds": nn_structure.n_conditions,
            "chunk_size": int(1e6),
            "sample_weights": nn_structure.use_sample_weights,
            "config_file": train_config.general.preprocess_config,
            "tracks_name": train_config.general.tracks_name,
        }

        # Get a small generator for metadata
        generator = utf.get_generator(
            "CADS",
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
                    "input_2": tf.TensorShape([None, nn_structure.n_conditions]),
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
                    "input_2": tf.TensorShape([None, nn_structure.n_conditions]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        # Get a full generator for training
        generator = utf.get_generator(
            "CADS",
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
            f"input file {train_config.general.train_file} is neither a .h5 file nor "
            "a directory with TF Record Files. You should check this."
        )

    # Init CADS model
    cads_model, epochs, init_epoch = create_cads_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = epochs

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    # Set ModelCheckpoint as callback
    cads_model_checkpoint = ModelCheckpoint(
        f"{train_config.general.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure.batch_size,
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(cads_model_checkpoint)

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
            jets_var_list=["absEta_btagJes", "pt_btagJes"],
            n_cond=nn_structure.n_conditions,
        )

        # TODO: Add a representative validation dataset for training (shown in
        # stdout)
        # Create the validation data tuple for the fit function
        # validation_data = (
        #     [
        #         val_data_dict["X_valid_trk"],
        #         val_data_dict["X_valid"],
        #     ],
        #     val_data_dict["Y_valid"],
        # )

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
    cads_model.fit(
        train_dataset,
        epochs=n_epochs,
        # TODO: Add a representative validation dataset for training (shown in stdout)
        # validation_data=validation_data,
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure.n_jets_train) / nn_structure.batch_size
        if nn_structure.n_jets_train is not None
        else metadata["n_jets"] / nn_structure.batch_size,
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
