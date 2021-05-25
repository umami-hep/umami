import argparse

import h5py
import tensorflow as tf
from tensorflow.keras import activations, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    Masking,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope

import umami.train_tools as utt
from umami.preprocessing_tools import Configuration
from umami.train_tools import Sum

# from plottingFunctions import sigBkgEff


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

    # TODO: implementng vr_overlap
    parser.add_argument(
        "--vr_overlap",
        action="store_true",
        help="Option to enable vr overlap removall for validation sets.",
    )

    args = parser.parse_args()
    return args


class generator:
    # How many jets should be loaded into memory
    chunk_size = 1e6

    def __init__(self, X, Y, batch_size):
        self.x = X
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
            batch_y = self.y_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            small_step += 1
            yield (batch_x, batch_y)


def Dips_model(train_config=None, input_shape=None):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Set NN options
    batch_norm = NN_structure["Batch_Normalisation"]
    dropout = NN_structure["dropout"]
    nClasses = NN_structure["nClasses"]

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
            tdd = TimeDistributed(
                Dropout(rate=dropout), name=f"Phi{i}_Dropout"
            )(tdd)

        tdd = TimeDistributed(
            layers.Activation(activations.relu), name=f"Phi{i}_ReLU"
        )(tdd)

    # This is where the magic happens... sum up the track features!
    F = Sum(name="Sum")(tdd)

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
        F = layers.Activation(activations.relu, name=f"F{j}_ReLU")(F)

    # Set output and activation function
    output = Dense(nClasses, activation="softmax", name="Jet_class")(F)
    dips = Model(inputs=trk_inputs, outputs=output)

    # Print Dips model summary
    dips.summary()

    # Set optimier and loss
    model_optimizer = Adam(lr=NN_structure["lr"])
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

    # Load the validation tracks
    X_valid, Y_valid = utt.GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        nJets=int(Val_params["n_jets"]),
    )

    # Load the extra validation tracks if defined.
    # If not, set it to none
    if train_config.add_validation_file is not None:
        X_valid_add, Y_valid_add = utt.GetTestSampleTrks(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            nJets=int(Val_params["n_jets"]),
        )

    else:
        X_valid_add = None
        Y_valid_add = None

    # Load the training file
    print("Load training data tracks")
    file = h5py.File(train_config.train_file, "r")
    X_train = file["X_trk_train"]
    Y_train = file["Y_train"]

    # Use the number of jets set in the config file for training
    if NN_structure["nJets_train"] is not None:
        X_train = X_train[: int(NN_structure["nJets_train"])]
        Y_train = Y_train[: int(NN_structure["nJets_train"])]

    # Get the shapes for training
    nJets, nTrks, nFeatures = X_train.shape
    nJets, nDim = Y_train.shape

    # Print how much jets are used
    print(f"Number of Jets used for training: {nJets}")

    if "model_file" in train_config.config:
        # Load DIPS model from file
        print(f"Loading model from: {train_config['model_file']}")
        with CustomObjectScope({"Sum": Sum}):
            dips = load_model(train_config["model_file"])

        # Load epoch from train_config
        epochs = train_config.NN_structure["epochs"]

    else:
        # Init dips model
        dips, epochs = Dips_model(
            train_config=train_config, input_shape=(nTrks, nFeatures)
        )

    # Get training set from generator
    train_dataset = (
        tf.data.Dataset.from_generator(
            generator(
                X_train, Y_train, train_config.NN_structure["batch_size"]
            ),
            (tf.float32, tf.float32),
            (
                tf.TensorShape([None, nTrks, nFeatures]),
                tf.TensorShape([None, nDim]),
            ),
        )
        .repeat()
        .prefetch(3)
    )

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set ModelCheckpoint as callback
    dips_mChkPt = ModelCheckpoint(
        f"{train_config.model_name}" + "/dips_model_{epoch:02d}.h5",
        monitor="val_loss",
        verbose=True,
        save_best_only=False,
        validation_batch_size=train_config.NN_structure["batch_size"],
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

    # Forming a dict for Callback
    val_data_dict = {
        "X_valid": X_valid,
        "Y_valid": Y_valid,
        "X_valid_add": X_valid_add,
        "Y_valid_add": Y_valid_add,
    }

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallbackDips(
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

    print("Start training")
    dips.fit(
        train_dataset,
        epochs=nEpochs,
        validation_data=(X_valid, Y_valid),
        callbacks=[dips_mChkPt, reduce_lr, my_callback],
        # callbacks=[reduce_lr, my_callback],
        # callbacks=[my_callback],
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
    Dips(args, train_config, preprocess_config)
