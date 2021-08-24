from umami.configuration import logger  # isort:skip
import argparse
import os

import h5py
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import umami.train_tools as utt
from umami.institutes.utils import is_qsub_available, submit_zeuthen
from umami.preprocessing_tools import Configuration
from umami.tools import yaml_loader

os.environ["KERAS_BACKEND"] = "tensorflow"


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
        default=300,
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


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TODO: add gpu support


class generator:
    # How many jets should be loaded into memory
    chunk_size = 1e6

    def __init__(
        self, train_file_path, X_Name, Y_Name, n_jets, batch_size, excluded_var
    ):
        self.train_file_path = train_file_path
        self.X_Name = X_Name
        self.Y_Name = Y_Name
        self.batch_size = batch_size
        self.excluded_var = excluded_var
        self.n_jets = len(self.y) if n_jets is None else int(n_jets)
        self.length = int(self.n_jets / self.batch_size)
        self.step_size = self.batch_size * int(
            generator.chunk_size / self.batch_size
        )

    def load_in_memory(self, part=0):
        logger.info(
            f"\nloading in memory {part + 1}/{1 + self.n_jets // self.step_size}"
        )

        # Load the
        with open(h5py.File(self.train_file_path, "r")) as f:
            self.x_in_mem = f[self.X_Name][
                self.step_size * part : self.step_size * (part + 1)
            ]
            self.y_in_mem = f[self.Y_Name][
                self.step_size * part : self.step_size * (part + 1)
            ]

        # Exclude variables if needed
        self.x_in_mem = (
            np.delete(self.x_in_mem, self.excluded_var, 1)
            if self.excluded_var is not None
            else self.x_in_mem
        )

    def __call__(self):
        self.load_in_memory()
        n = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * n:
                self.load_in_memory(n)
                n += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            small_step += 1
            yield (batch_x, batch_y)


def DL1_model(train_config, input_shape):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    class_labels = NN_structure["class_labels"]

    # Load model from file if defined
    if train_config.model_file is not None:
        logger.info(f"Loading model from: {train_config.model_file}")
        model = load_model(train_config.model_file, compile=False)

    else:
        # Define input
        inputs = Input(shape=input_shape)

        # Define layers
        for i, unit in enumerate(NN_structure["units"]):
            x = Dense(
                units=unit,
                activation="linear",
                kernel_initializer="glorot_uniform",
            )(inputs)

            # Add Batch Normalization if True
            if batch_norm:
                x = BatchNormalization()(x)

            # Add dropout if != 0
            if dropout != 0:
                x = Dropout(NN_structure["dropout_rate"][i])(x)

            # Define activation for the layer
            x = Activation(NN_structure["activations"][i])(x)

        predictions = Dense(
            units=len(class_labels),
            activation="softmax",
            kernel_initializer="glorot_uniform",
        )(x)
        model = Model(inputs=inputs, outputs=predictions)

    # Print DL1 model summary when log level lower or equal INFO level
    if logger.level <= 20:
        model.summary()

    model_optimizer = Adam(learning_rate=NN_structure["lr"])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )
    return model, NN_structure["epochs"]


def TrainLargeFile(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure
    Val_params = train_config.Eval_parameters_validation

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

    # Get the shapes for training
    with open(h5py.File(train_config.train_file, "r")) as f:
        nJets, nFeatures = f["X_train"].shape
        nJets, nDim = f["Y_train"].shape

    # Print how much jets are used
    logger.info(f"Number of Jets used for training: {nJets}")

    # Build train_datasets for training
    train_dataset = (
        tf.data.Dataset.from_generator(
            generator(
                train_file_path=train_config.train_file,
                X_Name="X_train",
                Y_Name="Y_train",
                n_jets=NN_structure["nJets_train"],
                batch_size=NN_structure["batch_size"],
                excluded_var=excluded_var,
            ),
            (tf.float32, tf.float32),
            (
                tf.TensorShape([None, nFeatures]),
                tf.TensorShape([None, nDim]),
            ),
        )
        .repeat()
        .prefetch(3)
    )

    # Load the validation tracks
    X_valid, Y_valid = utt.GetTestSample(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=NN_structure["class_labels"],
        nJets=int(Val_params["n_jets"]),
        exclude=exclude,
    )

    # Load the add_files if defined
    if train_config.add_validation_file is not None:
        X_valid_add, Y_valid_add = utt.GetTestSample(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=NN_structure["class_labels"],
            nJets=int(Val_params["n_jets"]),
            exclude=exclude,
        )

    else:
        X_valid_add, Y_valid_add = None, None

    # Forming a dict for Callback
    val_data_dict = {
        "X_valid": X_valid,
        "Y_valid": Y_valid,
        "X_valid_add": X_valid_add,
        "Y_valid_add": Y_valid_add,
    }

    # Load model and epochs
    model, epochs = DL1_model(
        train_config=train_config, input_shape=(X_valid.shape[1],)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.8,
        patience=3,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=0.000001,
    )

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallback(
        model_name=train_config.model_name,
        class_labels=NN_structure["class_labels"],
        main_class=NN_structure["main_class"],
        val_data_dict=val_data_dict,
        target_beff=Val_params["WP_b"],
        frac_dict=Val_params["frac_values"],
        dict_file_name=utt.get_validation_dict_name(
            WP_b=Val_params["WP_b"],
            n_jets=Val_params["n_jets"],
            dir_name=train_config.model_name,
        ),
    )

    logger.info("Start training")
    model.fit(
        x=train_dataset,
        epochs=nEpochs,
        callbacks=[reduce_lr, my_callback],
        steps_per_epoch=nJets / NN_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
    )

    logger.info(f"Models saved {train_config.model_name}")


def TrainLargeFileZeuthen(args, train_config, preprocess_config):
    if is_qsub_available():
        args.model_name = train_config.model_name
        args.dl1 = True
        submit_zeuthen(args)
    else:
        logger.warning(
            "No Zeuthen batch system found, training locally instead."
        )
        TrainLargeFile(args, train_config, preprocess_config)


if __name__ == "__main__":
    args = GetParser()
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
        TrainLargeFileZeuthen(args, train_config, preprocess_config)
    else:
        TrainLargeFile(args, train_config, preprocess_config)
