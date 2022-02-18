#!/usr/bin/env python
"""Keras model of the UMAMI with conditional attention tagger."""
from umami.configuration import logger  # isort:skip
import json
import os

import h5py
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.models import load_model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

import umami.tf_tools as utf
import umami.train_tools as utt

# from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.tools import yaml_loader


def Umami_model(train_config=None, input_shape=None, njet_features=None):
    """Keras model definition of UMAMI tagger.

    Parameters
    ----------
    train_config : object, optional
        training config, by default None
    input_shape : tuple, optional
        dataset input shape, by default None
    njet_features: int, optional
        number of jet features, by default None

    Returns
    -------
    keras model
        UMAMI with conditional attention keras model
    int
        number of epochs
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    if train_config.model_file is not None:
        # Load DIPS model from file
        logger.info(f"Loading model from: {train_config.model_file}")
        custom_obj = {
            "Sum": utf.Sum,
            "Attention": utf.Attention,
            "DeepSet": utf.DeepSet,
            "AttentionPooling": utf.AttentionPooling,
            "DenseNet": utf.DenseNet,
            "ConditionalAttention": utf.ConditionalAttention,
            "ConditionalDeepset": utf.ConditionalDeepSet,
        }

        umami = load_model(train_config.model_file, custom_obj, compile=False)

    else:
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

    # Print Umami model summary when log level lower or equal INFO level
    if logger.level <= 20:
        umami.summary()

    # Set optimier and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    umami.compile(
        loss="categorical_crossentropy",
        loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return umami, NN_structure["epochs"]


def UmamiCondAtt(args, train_config, preprocess_config):
    """Training handling of UMAMI with conditional attention tagger.

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
    KeyError
        if no metadata file in directory
    """
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    val_params = train_config.Validation_metrics_settings
    eval_params = train_config.Eval_parameters_validation

    # Init a list for the callbacks
    callbacks = []

    # Get needed variable from the train config
    WP = float(val_params["WP"]) if "WP" in val_params else float(eval_params["WP"])
    n_jets = (
        int(val_params["n_jets"])
        if "n_jets" in val_params
        else int(eval_params["n_jets"])
    )

    if "N_Conditions" in NN_structure:
        nCond = NN_structure["N_Conditions"]
    else:
        nCond = None

    val_data_dict = None
    if n_jets > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=n_jets,
            convert_to_tensor=True,
            nCond=nCond,
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
        with h5py.File(train_config.train_file, "r") as f:
            nJets, nTrks, nFeatures = f["X_trk_train"].shape
            nJets, nDim = f["Y_train"].shape
            nJets, njet_features = f["X_train"].shape
    elif os.path.isdir(train_config.train_file):
        train_file_names = os.listdir(train_config.train_file)
        for train_file_name in train_file_names:
            if not (".tfrecord" in train_file_name) and not (
                train_file_name == "metadata.json"
            ):
                raise ValueError(
                    f"input file {train_config.train_file} is neither a .h5 file nor a"
                    " directory with TF Record Files. You should check this."
                )
        if "metadata.json" not in train_file_names:
            raise KeyError("No metadata file in directory.")
        try:
            nfiles = train_config.config["nfiles"]
        except KeyError:
            logger.warning(
                "no number of files to be loaded in parallel defined. Set to 5"
            )
            nfiles = 5
        sample_weights = NN_structure["use_sample_weights"]
        tfrecord_reader = utf.TFRecordReader(
            train_config.train_file,
            NN_structure["batch_size"],
            nfiles,
            sample_weights,
            nCond,
        )
        train_dataset = tfrecord_reader.load_Dataset()
        metadata_name = (train_config.train_file + "/metadata.json").replace("//", "/")
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)
            nJets = metadata["nJets"]
            nTrks = metadata["nTrks"]
            nFeatures = metadata["nFeatures"]
            njet_features = metadata["njet_features"]
            nDim = metadata["nDim"]
    else:
        raise ValueError(
            f"input file {train_config.train_file} is neither a .h5 file nor a"
            " directory with TF Record Files. You should check this."
        )

    if NN_structure["nJets_train"] is not None:
        nJets = int(NN_structure["nJets_train"])

    logger.info(f"nJets: {nJets}, nTrks: {nTrks}")
    logger.info(f"nFeatures: {nFeatures}, njet_features: {njet_features}")

    umami, _ = Umami_model(
        train_config=train_config,
        input_shape=(nTrks, nFeatures),
        njet_features=njet_features,
    )

    if NN_structure["use_sample_weights"]:
        tensor_types = (
            {"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32},
            tf.float32,
            tf.float32,
        )
        tensor_shapes = (
            {
                "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                "input_2": tf.TensorShape([None, nCond]),
                "input_3": tf.TensorShape([None, njet_features]),
            },
            tf.TensorShape([None, nDim]),
            tf.TensorShape([None]),
        )
    else:
        tensor_types = (
            {"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32},
            tf.int64,
        )
        tensor_shapes = (
            {
                "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                "input_2": tf.TensorShape([None, nCond]),
                "input_3": tf.TensorShape([None, njet_features]),
            },
            tf.TensorShape([None, nDim]),
        )

    if ".h5" in train_config.train_file:
        train_dataset = (
            tf.data.Dataset.from_generator(
                utf.umami_condition_generator(
                    train_file_path=train_config.train_file,
                    X_Name="X_train",
                    X_trk_Name="X_trk_train",
                    Y_Name="Y_train",
                    n_jets=nJets,
                    batch_size=NN_structure["batch_size"],
                    nConds=nCond,
                    chunk_size=int(1e6),
                    excluded_var=excluded_var,
                    sample_weights=NN_structure["use_sample_weights"],
                ),
                output_types=tensor_types,
                output_shapes=tensor_shapes,
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = NN_structure["epochs"]

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    if "LRR" in NN_structure and NN_structure["LRR"] is True:
        # Define LearningRate Reducer as Callback
        reduce_lr = utf.GetLRReducer(**NN_structure)

        # Append the callback
        callbacks.append(reduce_lr)

    # Set ModelCheckpoint as callback
    umami_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}/model_files" + "/model_epoch{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    # Append the callback
    callbacks.append(umami_mChkPt)

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        class_labels=NN_structure["class_labels"],
        main_class=NN_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=WP,
        frac_dict=eval_params["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP=WP,
            n_jets=n_jets,
            dir_name=train_config.model_name,
        ),
    )

    # Append the callback
    callbacks.append(my_callback)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    history = umami.fit(
        train_dataset,
        epochs=nEpochs,
        callbacks=callbacks,
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
