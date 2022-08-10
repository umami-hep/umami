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
    NN_structure = train_config.NN_structure

    # Check if a prepared model is used or not
    umami, init_epoch, load_optimiser = utf.prepare_model(
        train_config=train_config,
        continue_training=continue_training,
    )

    if umami is None:
        logger.info("No modelfile provided! Initialize a new one!")

        umami = utf.Deepsets_model_umami(
            trk_input_shape=input_shape,
            jet_input_shape=njet_features,
            num_conditions=NN_structure["N_Conditions"],
            num_set_features=NN_structure["DIPS_ppm_units"][-1],
            DIPS_sets_nodes=NN_structure["DIPS_ppm_units"][:-1],
            F_classif_nodes=NN_structure["DIPS_dense_units"],
            classif_output=len(NN_structure["class_labels"]),
            intermediate_units=NN_structure["intermediate_units"],
            DL1_units=NN_structure["DL1_units"],
            pooling="attention",
            attention_nodes=NN_structure["attention_units"],
            condition_sets=NN_structure["DIPS_ppm_condition"],
            condition_attention=NN_structure["attention_condition"],
            condition_classifier=NN_structure["dense_condition"],
            shortcut_inputs=False,
            sets_batch_norm=NN_structure["Batch_Normalisation"],
            classif_batch_norm=NN_structure["Batch_Normalisation"],
            activation="relu",
            attention_softmax=False,
        )

    if load_optimiser is False:
        # Set optimier and loss
        model_optimizer = Adam(learning_rate=NN_structure["lr"])
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
            optimizer=model_optimizer,
            metrics=["accuracy"],
        )

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    return umami, NN_structure["epochs"], init_epoch


def UmamiCondAtt(args, train_config):
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
        with h5py.File(train_config.train_file, "r") as f:
            metadata["n_jets"], metadata["n_trks"], metadata["n_trk_features"] = f[
                f"X_{tracks_name}_train"
            ].shape
            _, metadata["n_dim"] = f["Y_train"].shape
            _, metadata["n_jet_features"] = f["X_train"].shape
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
                    "input_2": tf.TensorShape([None, nn_structure["N_Conditions"]]),
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
                    "input_2": tf.TensorShape([None, nn_structure["N_Conditions"]]),
                    "input_3": tf.TensorShape([None, metadata["n_jet_features"]]),
                },
                tf.TensorShape([None, metadata["n_dim"]]),
            )

        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.umami_condition_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    X_trk_Name=f"X_{tracks_name}_train",
                    Y_Name="Y_train",
                    n_jets=int(nn_structure["n_jets_train"])
                    if "n_jets_train" in nn_structure
                    and nn_structure["n_jets_train"] is not None
                    else metadata["n_jets"],
                    batch_size=nn_structure["batch_size"],
                    nConds=nn_structure["N_Conditions"],
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
        nEpochs = nn_structure["epochs"]

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    umami_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=nn_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_mChkPt)

    if "LRR" in nn_structure and nn_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**nn_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Load validation data for callback
    val_data_dict = None
    if n_jets_val > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config=train_config,
            n_jets=n_jets_val,
            convert_to_tensor=True,
            nCond=nn_structure["N_Conditions"],
        )

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
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

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    umami_cond_att_model.fit(
        train_dataset,
        epochs=nEpochs,
        callbacks=callbacks,
        steps_per_epoch=int(nn_structure["n_jets_train"]) / nn_structure["batch_size"]
        if "n_jets_train" in nn_structure and nn_structure["n_jets_train"] is not None
        else metadata["n_jets"] / nn_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
        initial_epoch=init_epoch,
    )
