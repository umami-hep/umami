#!/usr/bin/env python
import argparse
import os

import h5py
import tensorflow as tf
from keras import activations, layers
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Masking,
    TimeDistributed,
)
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import umami.train_tools as utt
from umami.preprocessing_tools import Configuration

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


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
    # TODO: implementng vr_overlap
    parser.add_argument(
        "--vr_overlap",
        action="store_true",
        help="""Option to
                        enable vr overlap removall for validation sets.""",
    )
    args = parser.parse_args()
    return args


class generator:

    # How many jets should be loaded into memory
    chunk_size = 5e5

    def __init__(self, X, X_trk, Y, batch_size):
        self.x = X
        self.x_trk = X_trk
        self.y = Y
        self.batch_size = batch_size
        self.n_jets = len(self.y)
        self.length = int(self.n_jets / self.batch_size)
        self.step_size = self.batch_size * int(
            generator.chunk_size / self.batch_size
        )

    def load_in_memory(self, part=0):
        print(
            "\nloading in memory",
            part + 1,
            "/",
            1 + self.n_jets // self.step_size,
        )
        self.x_in_mem = self.x[
            self.step_size * part : self.step_size * (part + 1)
        ]
        self.x_trk_in_mem = self.x_trk[
            self.step_size * part : self.step_size * (part + 1)
        ]
        self.y_in_mem = self.y[
            self.step_size * part : self.step_size * (part + 1)
        ]

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
            batch_x_trk = self.x_trk_in_mem[
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
            yield {"input_1": batch_x_trk, "input_2": batch_x}, batch_y


def Umami_model(train_config=None, input_shape=None, njet_features=None):
    batch_norm = True
    dropout = 0
    nClasses = 3

    NN_structure = train_config.NN_structure

    trk_inputs = Input(shape=input_shape)

    masked_inputs = Masking(mask_value=0)(trk_inputs)
    tdd = masked_inputs

    for i, phi_nodes in enumerate(NN_structure["DIPS_ppm_units"]):
        tdd = TimeDistributed(
            Dense(phi_nodes, activation="linear"), name=f"Phi{i}_Dense"
        )(tdd)
        if batch_norm:
            tdd = TimeDistributed(
                BatchNormalization(), name=f"Phi{i}_BatchNormalization"
            )(tdd)
        if dropout != 0:
            tdd = TimeDistributed(
                Dropout(rate=dropout), name=f"Phi{i}_Dropout"
            )(tdd)
        tdd = TimeDistributed(
            layers.Activation(activations.relu), name=f"Phi{i}_ReLU"
        )(tdd)

    # This is where the magic happens... sum up the track features!
    F = utt.Sum(name="Sum")(tdd)

    for j, (F_nodes, p) in enumerate(
        zip(
            NN_structure["DIPS_dense_units"],
            [dropout] * len(NN_structure["DIPS_dense_units"][:-1]) + [0],
        )
    ):

        F = Dense(F_nodes, activation="linear", name=f"F{j}_Dense")(F)
        if batch_norm:
            F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
        if dropout != 0:
            F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
        F = layers.Activation(activations.relu, name=f"F{j}_ReLU")(F)

    dips_output = Dense(nClasses, activation="softmax", name="dips")(F)

    # Input layer
    jet_inputs = Input(shape=(njet_features,))

    # adding the intermediate dense layers for DL1
    x = jet_inputs
    for unit in NN_structure["intermediate_units"]:
        x = Dense(
            units=unit,
            activation="linear",
            kernel_initializer="glorot_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    # Concatenate the inputs
    x = layers.concatenate([F, x])

    # loop to initialise the hidden layers
    for unit in NN_structure["DL1_units"]:
        x = Dense(
            units=unit,
            activation="linear",
            kernel_initializer="glorot_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    jet_output = Dense(
        units=nClasses,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        name="umami",
    )(x)

    umami = Model(
        inputs=[trk_inputs, jet_inputs], outputs=[dips_output, jet_output]
    )
    umami.summary()

    model_optimizer = Adam(lr=NN_structure["lr"])
    umami.compile(
        loss="categorical_crossentropy",
        loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
        optimizer=model_optimizer,
        metrics=["accuracy"],
    )

    return umami


def Umami(args, train_config, preprocess_config):
    val_data_dict = None
    if train_config.Eval_parameters_validation["n_jets"] > 0:
        val_data_dict = utt.load_validation_data(
            train_config,
            preprocess_config,
            train_config.Eval_parameters_validation["n_jets"],
        )

    exclude = []
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    file = h5py.File(train_config.train_file, "r")
    X_trk_train = file["X_trk_train"]
    X_train = file["X_train"][:, utt.get_jet_feature_indicies(exclude=exclude)]
    Y_train = file["Y_train"]
    nJets, nTrks, nFeatures = X_trk_train.shape
    nJets, nDim = Y_train.shape
    njet_features = X_train.shape[1]
    print(f"nJets: {nJets}, nTrks: {nTrks}")
    print(f"nFeatures: {nFeatures}, njet_features: {njet_features}")
    if "model_file" in train_config.config:
        print(f"Loading model from: {train_config.config['model_file']}")
        umami = load_model(
            train_config.config["model_file"], {"Sum": utt.Sum}, compile=False
        )
        NN_structure = train_config.NN_structure
        model_optimizer = Adam(lr=NN_structure["lr"])
        umami.compile(
            loss="categorical_crossentropy",
            loss_weights={
                "dips": NN_structure["dips_loss_weight"],
                "umami": 1,
            },
            optimizer=model_optimizer,
            metrics=["accuracy"],
        )
    else:
        umami = Umami_model(
            train_config=train_config,
            input_shape=(nTrks, nFeatures),
            njet_features=njet_features,
        )

    train_dataset = (
        tf.data.Dataset.from_generator(
            generator(
                X_train,
                X_trk_train,
                Y_train,
                train_config.NN_structure["batch_size"],
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
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        val_data_dict=val_data_dict,
        target_beff=train_config.Eval_parameters_validation["WP_b"],
        charm_fraction=train_config.Eval_parameters_validation["fc_value"],
        dict_file_name=utt.get_validation_dict_name(
            WP_b=train_config.Eval_parameters_validation["WP_b"],
            fc_value=train_config.Eval_parameters_validation["fc_value"],
            n_jets=train_config.Eval_parameters_validation["n_jets"],
            dir_name=train_config.model_name,
        ),
    )

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    print("Start training")
    umami.fit(
        train_dataset,
        epochs=nEpochs,
        callbacks=[reduce_lr, my_callback],
        steps_per_epoch=nJets / train_config.NN_structure["batch_size"],
        use_multiprocessing=True,
        workers=8,
    )


if __name__ == "__main__":
    args = GetParser()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    Umami(args, train_config, preprocess_config)
