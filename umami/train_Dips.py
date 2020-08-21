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
    batch_norm = True
    dropout = 0
    nClasses = 3

    batch_size = 256
    # batch_size = 15600
    # batch_size = 65600

    ppm_sizes_int = [100, 100, 128]
    dense_sizes_int = [100, 100, 100]

    trk_inputs = Input(shape=input_shape)

    masked_inputs = Masking(mask_value=0)(trk_inputs)
    tdd = masked_inputs

    for i, phi_nodes in enumerate(ppm_sizes_int):

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

    for j, (F_nodes, p) in enumerate(zip(dense_sizes_int, [dropout] *
                                         len(dense_sizes_int[:-1])+[0])):

        F = Dense(F_nodes, activation='linear', name=f"F{j}_Dense")(F)
        if batch_norm:
            F = BatchNormalization(name=f"F{j}_BatchNormalization")(F)
        if dropout != 0:
            F = Dropout(rate=p, name=f"F{j}_Dropout")(F)
        F = layers.Activation(activations.relu, name=f"F{j}_ReLU")(F)

    output = Dense(nClasses, activation='softmax', name="Jet_class")(F)
    dips = Model(inputs=trk_inputs, outputs=output)

    # dips.summary()
    model_optimizer = Adam(lr=0.01)
    dips.compile(loss='categorical_crossentropy', optimizer=model_optimizer,
                 #  optimizer='adam',
                 metrics=['accuracy'])
    return dips, batch_size


def Dips(args, train_config, preprocess_config):
    X_valid, Y_valid = utt.GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config)

    file = h5py.File(train_config.train_file, 'r')
    Ntrain = 6000000
    # Ntrain = 600000
    X_train = file['X_trk_train'][:Ntrain]
    Y_train = file['Y_train'][:Ntrain]
    nJets, nTrks, nFeatures = X_train.shape
    nJets, nDim = Y_train.shape

    dips, batch_size = Dips_model(input_shape=(nTrks, nFeatures))

    train_dataset = tf.data.Dataset.from_generator(
        generator(X_train, Y_train, batch_size),
        (tf.float32, tf.float32),
        (tf.TensorShape([None, nTrks, nFeatures]),
         tf.TensorShape([None, nDim]))
    ).repeat()

    nEpochs = args.epochs

    # earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=10)

    # dips_mChkPt = ModelCheckpoint('dips/dips_model_{epoch:02d}.h5',
    #                               monitor='val_loss',
    #                               verbose=True,
    #                               save_best_only=False,
    #                               validation_batch_size=15000,
    #                               save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3,
                                  verbose=1, mode='auto',
                                  cooldown=5, min_lr=0.000001)
    my_callback = utt.MyCallback(
        model_name="dips",
        X_valid=X_valid,
        Y_valid=Y_valid
        )

    print("Start training")
    dips.fit(train_dataset,
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
    if args. performance_check:
        utt.RunPerformanceCheck(train_config, compare_tagger=True,
                                tagger_comp_var=["rnnip_pu", "rnnip_pc",
                                                 "rnnip_pb"],
                                comp_tagger_name="RNNIP")
    else:
        Dips(args, train_config, preprocess_config)
