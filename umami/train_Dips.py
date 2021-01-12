import h5py
import argparse

import numpy as np
import tensorflow as tf

from keras.layers import BatchNormalization, TimeDistributed, Dropout
from keras.layers import Dense, Input, Masking
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from keras import activations
from tensorflow.keras.callbacks import ReduceLROnPlateau

import umami.train_tools as utt
from umami.train_tools import Sum
from umami.preprocessing_tools import Configuration
# from plottingFunctions import sigBkgEff


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Name of the training config file")
    parser.add_argument('-e', '--epochs', type=int, help="Number\
        of trainng epochs.")
    # TODO: implementng vr_overlap
    parser.add_argument('--vr_overlap', action='store_true', help='''Option to
                        enable vr overlap removall for validation sets.''')
    parser.add_argument('-p', '--performance_check', action='store_true',
                        help="Performs performance check - can be run during"
                        " training")
    args = parser.parse_args()
    return args


class generator:
    def __init__(self, X, Y, batch_size):
        self.x = X
        self.y = Y
        self.batch_size = batch_size

    def __call__(self):
        length = int(np.ceil(len(self.x) / float(self.batch_size)))
        for idx in range(length):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
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

        tdd = TimeDistributed(Dense(phi_nodes, activation='linear'),
                              name=f"Phi{i}_Dense")(tdd)
        if batch_norm:
            tdd = TimeDistributed(BatchNormalization(),
                                  name=f"Phi{i}_BatchNormalization")(tdd)
        if dropout != 0:
            tdd = TimeDistributed(Dropout(rate=dropout),
                                  name=f"Phi{i}_Dropout")(tdd)
        tdd = TimeDistributed(layers.Activation(
            activations.relu), name=f"Phi{i}_ReLU")(tdd)

    # This is where the magic happens... sum up the track features!
    F = Sum(name="Sum")(tdd)

    # Define the main dips structure
    for j, (F_nodes, p) in enumerate(
        zip(
            NN_structure["dense_sizes"],
            [dropout] * len(NN_structure["dense_sizes"][:-1]) + [0],
        )
    ):

        F = Dense(F_nodes, activation='linear', name=f"F{j}_Dense")(F)
        if batch_norm:
            F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
        if dropout != 0:
            F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
        F = layers.Activation(activations.relu, name=f"F{j}_ReLU")(F)

    # Set output and activation function
    output = Dense(nClasses, activation='softmax', name="Jet_class")(F)
    dips = Model(inputs=trk_inputs, outputs=output)

    # dips.summary()
    # Set optimier and loss
    model_optimizer = Adam(lr=NN_structure["lr"])
    dips.compile(
        loss='categorical_crossentropy',
        optimizer=model_optimizer,
        metrics=['accuracy']
    )
    return dips, NN_structure["batch_size"], NN_structure["epochs"]


def Dips(args, train_config, preprocess_config):
    # Load NN Structure and training parameter from file
    NN_structure = train_config.NN_structure

    # Load the validation tracks
    X_valid, Y_valid = utt.GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        nJets=int(NN_structure["nJets_val"])
    )

    # Load the extra validation tracks if defined.
    # If not, set it to none
    if train_config.add_validation_file is not None:
        X_valid_add, Y_valid_add = utt.GetTestSampleTrks(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            nJets=int(NN_structure["nJets_val"])
        )

    else:
        X_valid_add = None
        Y_valid_add = None

    # Load the training file
    print("Load training data tracks")
    file = h5py.File(train_config.train_file, 'r')
    X_train = file['X_trk_train']
    Y_train = file['Y_train']

    # Use the number of jets set in the config file for training
    X_train = X_train[:int(NN_structure["nJets_train"])]
    Y_train = Y_train[:int(NN_structure["nJets_train"])]

    # Get the shapes for training
    nJets, nTrks, nFeatures = X_train.shape
    nJets, nDim = Y_train.shape

    # Print how much jets are used
    print(f"Number of Jets used for training: {nJets}")

    # Init dips model
    dips, batch_size, epochs = Dips_model(
        train_config=train_config,
        input_shape=(nTrks, nFeatures)
    )

    # Get training set from generator
    train_dataset = tf.data.Dataset.from_generator(
        generator(X_train, Y_train, batch_size),
        (tf.float32, tf.float32),
        (tf.TensorShape([None, nTrks, nFeatures]),
         tf.TensorShape([None, nDim]))
    ).repeat()

    # Check if epochs is set via argparser or not
    if args.epochs is None:
        nEpochs = epochs

    # If not, use epochs from config file
    else:
        nEpochs = args.epochs

    # Set EarlyStopping as callback
    earlyStop = EarlyStopping(
        monitor='val_loss', verbose=True, patience=10
    )

    # Set ModelCheckpoint as callback
    dips_mChkPt = ModelCheckpoint(
        'dips/dips_model_{epoch:02d}.h5',
        monitor='val_loss',
        verbose=True,
        save_best_only=False,
        validation_batch_size=batch_size,
        save_weights_only=False
    )

    # Set ReduceLROnPlateau as callback
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.8,
        patience=3,
        verbose=1, mode='auto',
        cooldown=5, min_lr=0.000001
    )

    # Set my_callback as callback. Writes history information
    # to json file.
    my_callback = utt.MyCallback(
        model_name=train_config.model_name,
        X_valid=X_valid,
        Y_valid=Y_valid,
        X_valid_add=X_valid_add,
        Y_valid_add=Y_valid_add
    )

    print("Start training")

    dips.fit(
        train_dataset,
        epochs=nEpochs,
        validation_data=(X_valid, Y_valid),
        callbacks=[earlyStop, dips_mChkPt, reduce_lr, my_callback],
        # callbacks=[reduce_lr, my_callback],
        # callbacks=[reduce_lr],
        steps_per_epoch=len(Y_train) / batch_size,
        use_multiprocessing=True,
        workers=8
    )


if __name__ == '__main__':
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    if args.performance_check:
        utt.RunPerformanceCheck(train_config, compare_tagger=True,
                                tagger_comp_var=["rnnip_pu", "rnnip_pc",
                                                 "rnnip_pb"],
                                comp_tagger_name="RNNIP")
    else:
        Dips(args, train_config, preprocess_config)
