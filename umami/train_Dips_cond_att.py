from umami.configuration import logger  # isort:skip
import argparse
import json

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import umami.tf_tools as utf
import umami.train_tools as utt
from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.preprocessing_tools import Configuration


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing command line options."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Name of the training config file",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, help="Number of training epochs."
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
        help="Option to enable vr overlap removall for validation sets.",
    )

    parser.add_argument(
        "-o",
        "--overwrite_config",
        action="store_true",
        help="Overwrite the configs files saved in metadata folder",
    )

    args = parser.parse_args()
    return args


def Dips_model(train_config=None, input_shape=None):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]

    dips = utf.Deepsets_model(
        repeat_input_shape=input_shape,
        num_conditions=NN_structure["N_Conditions"],
        num_set_features=NN_structure["ppm_sizes"][-1],
        sets_nodes=NN_structure["ppm_sizes"][:-1],
        classif_nodes=NN_structure["dense_sizes"],
        classif_output=len(NN_structure["class_labels"]),
        pooling="attention",
        attention_nodes=NN_structure["attention_sizes"],
        condition_sets=NN_structure["ppm_condition"],
        condition_attention=NN_structure["attention_condition"],
        condition_classifier=NN_structure["dense_condition"],
        shortcut_inputs=False,
        sets_batch_norm=batch_norm,
        classif_batch_norm=batch_norm,
        activation="relu",
        attention_softmax=False,
    )

    # Print Dips model summary when log level lower or equal INFO level
    if logger.level <= 20:
        dips.summary()

    # Set optimiser and loss
    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    dips.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return dips, NN_structure["epochs"]


def Dips(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation

    # Get the shapes for training
    with h5py.File(train_config.train_file, "r") as f:
        nJets, nTrks, nFeatures = f["X_trk_train"].shape
        nJets, nDim = f["Y_train"].shape

        if NN_structure["nJets_train"] is not None:
            nJets = int(NN_structure["nJets_train"])

    # Print how much jets are used
    logger.info(f"Number of Jets used for training: {nJets}")

    # Init dips model
    dips, epochs = Dips_model(
        train_config=train_config, input_shape=(nTrks, nFeatures)
    )

    # Get training set from generator
    train_dataset = (
        tf.data.Dataset.from_generator(
            utf.dips_condition_generator(
                train_file_path=train_config.train_file,
                X_Name="X_train",
                X_trk_Name="X_trk_train",
                Y_Name="Y_train",
                n_jets=nJets,
                batch_size=NN_structure["batch_size"],
                nConds=NN_structure["N_Conditions"],
                chunk_size=int(1e6),
            ),
            output_types=(
                {"input_1": tf.float32, "input_2": tf.float32},
                tf.float32,
            ),
            output_shapes=(
                {
                    "input_1": tf.TensorShape([None, nTrks, nFeatures]),
                    "input_2": tf.TensorShape(
                        [None, NN_structure["N_Conditions"]]
                    ),
                },
                tf.TensorShape([None, nDim]),
            ),
        )
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    dips_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}" + "/dips_model_{epoch:03d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=NN_structure["batch_size"],
        save_weights_only=False,
    )

    # Set ReduceLROnPlateau as callback
    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.8,
        patience=3,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=0.000001,
    )

    # Load validation data for callback
    val_data_dict = None
    if Val_params["n_jets"] > 0:
        val_data_dict = utt.load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=int(Val_params["n_jets"]),
            convert_to_tensor=True,
            jets_var_list=["absEta_btagJes", "pt_btagJes"],
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
        validation_data=(
            [
                val_data_dict["X_valid_trk"],
                val_data_dict["X_valid"],
            ],
            val_data_dict["Y_valid"],
        ),
        callbacks=[dips_mChkPt, reduce_lr, my_callback],
        steps_per_epoch=nJets / NN_structure["batch_size"],
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


def DipsZeuthen(args, train_config, preprocess_config):
    if is_qsub_available():
        args.model_name = train_config.model_name
        args.dips = True
        submit_zeuthen(args)
    else:
        logger.warning(
            "No Zeuthen batch system found, training locally instead."
        )
        Dips(args, train_config, preprocess_config)


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

    # Start the real training
    if args.zeuthen:
        DipsZeuthen(args, train_config, preprocess_config)

    else:
        Dips(args, train_config, preprocess_config)
