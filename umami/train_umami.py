import h5py
import argparse

import numpy as np
import tensorflow as tf

from keras.layers import BatchNormalization, TimeDistributed, Dropout
from keras.layers import Dense, Input, Masking
from keras.models import Model
from keras.optimizers import Adam
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
    parser.add_argument('-e', '--epochs', default=300, type=int, help="Number\
        of training epochs.")
    # TODO: implementng vr_overlap
    parser.add_argument('--vr_overlap', action='store_true', help='''Option to
                        enable vr overlap removall for validation sets.''')
    parser.add_argument('-p', '--performance_check', action='store_true',
                        help="Performs performance check - can be run during"
                        " training")
    args = parser.parse_args()
    return args


class generator:
    def __init__(self, X, X_trk, Y, batch_size):
        self.x = X
        self.x_trk = X_trk
        self.y = Y
        self.batch_size = batch_size

    def __call__(self):
        length = int(np.ceil(len(self.x) / float(self.batch_size)))
        for idx in range(length):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x_trk = self.x_trk[idx * self.batch_size:(idx + 1) *
                                     self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            # yield (batch_x, batch_x_trk, batch_y)
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

    for j, (F_nodes, p) in enumerate(
        zip(
            NN_structure["DIPS_dense_units"],
            [dropout] * len(NN_structure["DIPS_dense_units"][:-1]) + [0],
        )
    ):

        F = Dense(F_nodes, activation='linear', name=f"F{j}_Dense")(F)
        if batch_norm:
            F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
        if dropout != 0:
            F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
        F = layers.Activation(activations.relu, name=f"F{j}_ReLU")(F)

    dips_output = Dense(nClasses, activation='softmax', name="dips")(F)

    # Input layer
    jet_inputs = Input(shape=(njet_features,))

    # adding the intermediate dense layers for DL1
    x = jet_inputs
    for unit in NN_structure["intermediate_units"]:
        x = Dense(units=unit, activation="linear",
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Concatenate the inputs
    x = layers.concatenate([F, x])

    # loop to initialise the hidden layers
    for unit in NN_structure["DL1_units"]:
        x = Dense(units=unit, activation="linear",
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    jet_output = Dense(units=nClasses, activation='softmax',
                       kernel_initializer='glorot_uniform', name="umami")(x)

    umami = Model(inputs=[trk_inputs, jet_inputs],
                  outputs=[dips_output, jet_output])
    umami.summary()

    model_optimizer = Adam(lr=NN_structure["lr"])
    umami.compile(
        loss='categorical_crossentropy',
        loss_weights={"dips": NN_structure["dips_loss_weight"], "umami": 1},
        optimizer=model_optimizer,
        metrics=['accuracy']
    )

    # dips.summary()
    return umami, NN_structure["batch_size"]


def Umami(args, train_config, preprocess_config):
    X_valid_trk, Y_valid_trk = utt.GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config)

    X_valid, Y_valid = utt.GetTestSample(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config)

    assert np.equal(Y_valid, Y_valid_trk).all()

    file = h5py.File(train_config.train_file, 'r')
    X_trk_train = file['X_trk_train']
    X_train = file['X_train']
    Y_train = file['Y_train']
    nJets, nTrks, nFeatures = X_trk_train.shape
    nJets, nDim = Y_train.shape
    njet_features = X_train.shape[1]
    print(f"nJets: {nJets}, nTrks: {nTrks}")
    print(f"nFeatures: {nFeatures}, njet_features: {njet_features}")
    umami, batch_size = Umami_model(train_config=train_config,
                                    input_shape=(nTrks, nFeatures),
                                    njet_features=njet_features)

    train_dataset = tf.data.Dataset.from_generator(
        generator(X_train, X_trk_train, Y_train, batch_size),
        output_types=({"input_1": tf.float32, "input_2": tf.float32},
                      tf.float32),
        output_shapes=({"input_1": tf.TensorShape([None, nTrks, nFeatures]),
                        "input_2": tf.TensorShape([None, njet_features])},
                       tf.TensorShape([None, nDim]))
    ).repeat()

    nEpochs = args.epochs

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3,
                                  verbose=1, mode='auto',
                                  cooldown=5, min_lr=0.000001)
    my_callback = utt.MyCallbackUmami(
        model_name=train_config.model_name,
        X_valid=X_valid,
        X_valid_trk=X_valid_trk,
        Y_valid=Y_valid
    )

    print("Start training")
    umami.fit(train_dataset,
              epochs=nEpochs,
              #  validation_data=(X_valid, Y_valid),
              #  callbacks=[earlyStop, dips_mChkPt, reduce_lr],
              #  callbacks=[reduce_lr],
              callbacks=[reduce_lr, my_callback],
              steps_per_epoch=len(Y_train) / batch_size,
              use_multiprocessing=True,
              workers=8
              )
# dips_hist = dips.fit(train_dataset,
    #                      epochs=nEpochs,
    #                     #  validation_data=(X_valid, Y_valid),
    #                      #  callbacks=[earlyStop, dips_mChkPt, reduce_lr],
    #                      callbacks=[reduce_lr, my_callback],
    #                      steps_per_epoch=len(Y_train) / batch_size,
    #                      use_multiprocessing=True,
    #                      workers=8
    #                      )

    # epochs = np.arange(1, len(dips_hist.history['loss'])+1)

    # plt.plot(epochs, dips_hist.history['loss'], label='training')
    # # plt.plot(epochs, dips_hist.history['val_loss'], label='validation')
    # plt.xlabel('epochs', fontsize=14)
    # plt.ylabel('cross-entropy loss', fontsize=14)
    # plt.legend()
    # plt.title('DIPS')

    # plt.savefig('dips/plots/dips-loss.pdf', transparent=True)


if __name__ == '__main__':
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)
    if args.performance_check:
        utt.RunPerformanceCheckUmami(train_config, compare_tagger=True,
                                     tagger_comp_var=["rnnip_pu", "rnnip_pc",
                                                      "rnnip_pb"],
                                     comp_tagger_name="RNNIP")
        utt.RunPerformanceCheckUmami(train_config, compare_tagger=True)
    else:
        Umami(args, train_config, preprocess_config)
