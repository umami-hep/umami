"""Keras model of the UMAMI with conditional attention tagger."""
from umami.configuration import logger  # isort:skip
import os

import h5py
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.tools import yaml_loader


def create_umami_cond_att_model(
    train_config: object,
    input_shape: tuple,
    njet_features: int,
    continue_training: bool = False,
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
    continue_training : bool, optional
        Decide, if the training is continued using the latest
        model file, by default False

    Returns
    -------
    keras model
        UMAMI with conditional attention keras model
    int
        number of epochs
    int
        Starting epoch number
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure

    # Check if a prepared model is used or not
    umami, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if umami is None:
        logger.info("No modelfile provided! Initialize a new one!")

        umami = utf.deepsets_model_umami(
            trk_input_shape=input_shape,
            jet_input_shape=njet_features,
            num_conditions=nn_structure["n_conditions"],
            num_set_features=nn_structure["dips_ppm_units"][-1],
            dips_sets_nodes=nn_structure["dips_ppm_units"][:-1],
            f_classif_nodes=nn_structure["dips_dense_units"],
            classif_output=len(nn_structure["class_labels"]),
            intermediate_units=nn_structure["intermediate_units"],
            dl1_units=nn_structure["dl1_units"],
            pooling="attention",
            attention_nodes=nn_structure["attention_units"],
            condition_sets=nn_structure["dips_ppm_condition"],
            condition_attention=nn_structure["attention_condition"],
            condition_classifier=nn_structure["dense_condition"],
            shortcut_inputs=False,
            sets_batch_norm=nn_structure["batch_normalisation"],
            classif_batch_norm=nn_structure["batch_normalisation"],
            activation="relu",
            attention_softmax=False,
        )

    if load_optimiser is False:
        # Set optimier and loss
        model_optimizer = Adam(learning_rate=nn_structure["lr"])
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={"dips": nn_structure["dips_loss_weight"], "umami": 1},
            optimizer=model_optimizer,
            metrics=["accuracy"],
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    return umami, nn_structure["epochs"], init_epoch


def train_umami_cond_att(args, train_config):
    """Training handling of UMAMI with conditional attention tagger.

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
            _, metadata["n_jet_features"] = f_train["jets/inputs"].shape
            if exclude is not None:
                metadata["n_jet_features"] -= len(excluded_var)
            logger.debug(
                "Input shape of jet training set: %s", metadata["n_jet_features"]
            )

        if nn_structure["use_sample_weights"]:
            tensor_types = (
                {"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32},
                tf.float32,
                tf.float32,
            )
            tensor_shapes = (
                {
                    "input_1": tf.TensorShape(
                        [None, metadata["n_trks"], metadata["n_trk_features"]]
                    ),
                    "input_2": tf.TensorShape([None, nn_structure["n_conditions"]]),
                    "input_3": tf.TensorShape([None, metadata["n_jet_features"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
                tf.TensorShape([None]),
            )

        else:
            tensor_types = (
                {"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32},
                tf.int64,
            )
            tensor_shapes = (
                {
                    "input_1": tf.TensorShape(
                        [None, metadata["n_trks"], metadata["n_trk_features"]]
                    ),
                    "input_2": tf.TensorShape([None, nn_structure["n_conditions"]]),
                    "input_3": tf.TensorShape([None, metadata["n_jet_features"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.UmamiConditionGenerator(
                    train_file_path=train_config.train_file,
                    x_name="jets/inputs",
                    x_trk_name=f"{tracks_name}/inputs",
                    y_name="jets/labels_one_hot",
                    n_jets=int(nn_structure["n_jets_train"])
                    if "n_jets_train" in nn_structure
                    and nn_structure["n_jets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
                    n_conds=nn_structure["n_conditions"],
                    chunk_size=int(1e6),
                    excluded_var=excluded_var,
                    sample_weights=nn_structure["use_sample_weights"],
                ),
                output_types=tensor_types,
                output_shapes=tensor_shapes,
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

    umami_cond_att_model, _, init_epoch = create_umami_cond_att_model(
        train_config=train_config,
        input_shape=(metadata["n_trks"], metadata["n_trk_features"]),
        njet_features=metadata["n_jet_features"],
        continue_training=train_config.continue_training,
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        n_epochs = nn_structure["epochs"]

    # If not, use epochs from config file
    else:
        n_epochs = args.epochs

    # Set ModelCheckpoint as callback
    umami_m_chk_pt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_m_chk_pt)

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
            n_cond=nn_structure["n_conditions"],
        )

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
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

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    umami_cond_att_model.fit(
        train_dataset,
        epochs=n_epochs,
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure["n_jets_train"]) / nn_structure["batch_size"]
        if "n_jets_train" in nn_structure and nn_structure["n_jets_train"] is not None
        else metadata["n_jets"] / nn_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
