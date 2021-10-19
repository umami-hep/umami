#!/usr/bin/env python
from umami.configuration import logger  # isort:skip
import argparse
import json

import h5py
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.preprocessing_tools import Configuration
from umami.tools import yaml_loader


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing command line" "options."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Name of the training config file",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number\
        of training epochs.",
    )

    parser.add_argument(
        "-z",
        "--zeuthen",
        action="store_true",
        help="Train on Zeuthen with GPU support",
    )

    # TODO: implementng vr_overlap
    parser.add_argument(
        "--vr_overlap",
        action="store_true",
        help="""Option to
                        enable vr overlap removall for validation sets.""",
    )
    parser.add_argument(
        "-o",
        "--overwrite_config",
        action="store_true",
        help="Overwrite the configs files saved in metadata folder",
    )
    args = parser.parse_args()
    return args


def Umami(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    val_data_dict = None
    if train_config.Eval_parameters_validation["n_jets"] > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config,
            preprocess_config,
            train_config.Eval_parameters_validation["n_jets"],
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

    # Use the number of jets set in the config file for training
    NN_structure = train_config.NN_structure

    with h5py.File(train_config.train_file, "r") as f:
        nJets, nTrks, nFeatures = f["X_trk_train"].shape
        nJets, nDim = f["Y_train"].shape
        nJets, njet_features = f["X_train"].shape

    logger.info(f"nJets: {nJets}, nTrks: {nTrks}")
    logger.info(f"nFeatures: {nFeatures}, njet_features: {njet_features}")

    umami, _ = utf.Umami_model(
        train_config=train_config,
        input_shape=(nTrks, nFeatures),
        njet_features=njet_features,
    )

    train_dataset = (
        tf.data.Dataset.from_generator(
            utf.umami_generator(
                train_file_path=train_config.train_file,
                X_Name="X_train",
                X_trk_Name="X_trk_train",
                Y_Name="Y_train",
                n_jets=NN_structure["nJets_train"],
                batch_size=train_config.NN_structure["batch_size"],
                excluded_var=excluded_var,
            ),
            output_types=(
                {"input_1": tf.float32, "input_2": tf.float32},
                tf.float32,
            ),
            output_shapes=(
                {
                    "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                    "input_2": tf.TensorShape([None, njet_features]),
                },
                tf.TensorShape([None, nDim]),
            ),
        )
        .repeat()
        .prefetch(3)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = NN_structure["epochs"]

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Define LearningRate Reducer as Callback
    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.8,
        patience=3,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_learning_rate=0.000001,
    )

    # Set ModelCheckpoint as callback
    umami_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}" + "/umami_model_{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    # Init the Umami callback
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        class_labels=train_config.NN_structure["class_labels"],
        main_class=train_config.NN_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=train_config.Eval_parameters_validation["WP"],
        frac_dict=train_config.Eval_parameters_validation["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP=train_config.Eval_parameters_validation["WP"],
            n_jets=train_config.Eval_parameters_validation["n_jets"],
            dir_name=train_config.model_name,
        ),
    )

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    logger.info("Start training")
    history = umami.fit(
        train_dataset,
        epochs=nEpochs,
        callbacks=[umami_mChkPt, reduce_lr, my_callback],
        steps_per_epoch=nJets / train_config.NN_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
    )

    # Dump dict into json
    logger.info(
        f"Dumping history file to {train_config.model_name}/history.json"
    )

    # Make the history dict the same shape as the dict from the callbacks
    hist_dict = utt.prepare_history_dict(history.history)

    # Dump history file to json
    with open(f"{train_config.model_name}/history.json", "w") as outfile:
        json.dump(hist_dict, outfile, indent=4)


def UmamiZeuthen(args, train_config, preprocess_config):
    if is_qsub_available():
        args.model_name = train_config.model_name
        args.umami = True
        submit_zeuthen(args)
    else:
        logger.warning(
            "No Zeuthen batch system found, training locally instead."
        )
        Umami(args, train_config, preprocess_config)


if __name__ == "__main__":
    args = GetParser()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)

    utt.create_metadata_folder(
        train_config_path=args.config_file,
        var_dict_path=train_config.var_dict,
        model_name=train_config.model_name,
        preprocess_config_path=train_config.preprocess_config,
        overwrite_config=True if args.overwrite_config else False,
    )

    if args.zeuthen:
        UmamiZeuthen(args, train_config, preprocess_config)
    else:
        Umami(args, train_config, preprocess_config)
