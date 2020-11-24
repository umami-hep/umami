import h5py
import argparse
import os
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from umami.preprocessing_tools import Configuration
import umami.train_tools as utt
os.environ['KERAS_BACKEND'] = 'tensorflow'


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


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TODO: add gpu support

def NN_model(train_config, input_shape):
    NN_config = train_config.NN_structure
    inputs = Input(shape=input_shape)
    x = inputs
    for i, unit in enumerate(NN_config["units"]):
        x = Dense(units=unit, activation="linear",
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation(NN_config['activations'][i])(x)
        if "dropout_rate" in NN_config:
            x = Dropout(NN_config["dropout_rate"][i])(x)
    predictions = Dense(units=3, activation='softmax',
                        kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    model_optimizer = Adam(lr=NN_config["lr"])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=model_optimizer,
        metrics=['accuracy']
    )
    return model, NN_config["batch_size"]


def TrainLargeFile(args, train_config, preprocess_config):
    print("Loading validation data (training data will be loaded per batch)")
    X_valid, Y_valid = utt.GetTestSample(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config)

    X_valid_add, Y_valid_add = None, None
    if train_config.add_validation_file is not None:
        X_valid_add, Y_valid_add = utt.GetTestSample(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config)
        assert X_valid.shape[1] == X_valid_add.shape[1]

    model, batch_size = NN_model(train_config, (X_valid.shape[1],))
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3,
                                  verbose=1, mode='auto',
                                  cooldown=5, min_lr=0.000001)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
    #                               min_lr=0.00001)
    my_callback = utt.MyCallback(model_name=train_config.model_name,
                                 X_valid=X_valid,
                                 Y_valid=Y_valid,
                                 X_valid_add=X_valid_add,
                                 Y_valid_add=Y_valid_add)
    callbacks = [reduce_lr, my_callback]
    file = h5py.File(train_config.train_file, 'r')
    X_train = file['X_train']
    Y_train = file['Y_train']

    # create the training datasets
    # examples taken from https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/  # noqa
    dx_train = tf.data.Dataset.from_tensor_slices(X_train)
    dy_train = tf.data.Dataset.from_tensor_slices(Y_train)
    # zip the x and y training data together and batch etc.
    train_dataset = tf.data.Dataset.zip(
        (dx_train, dy_train)).repeat().batch(batch_size)
    model.fit(x=train_dataset,
              epochs=args.epochs,
              callbacks=callbacks,
              steps_per_epoch=len(Y_train) / batch_size,
              use_multiprocessing=True,
              workers=8
              )

    print("Models saved:", train_config.model_name)


if __name__ == '__main__':
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)

    if args. performance_check:
        utt.RunPerformanceCheck(train_config)
    else:
        TrainLargeFile(args, train_config, preprocess_config)