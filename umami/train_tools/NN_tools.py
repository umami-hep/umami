import os
import json
import h5py
import yaml
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Layer
from tensorflow.keras.callbacks import Callback
from umami.tools import yaml_loader
from umami.preprocessing_tools import GetBinaryLabels, Gen_default_dict


def GetRejection(y_pred, y_true, target_beff=0.77, cfrac=0.018):
    """Calculates the c and light rejection for specific WP and c-fraction."""
    b_index, c_index, u_index = 2, 1, 0
    y_true = np.argmax(y_true, axis=1)
    b_jets = y_pred[y_true == b_index]
    c_jets = y_pred[y_true == c_index]
    u_jets = y_pred[y_true == u_index]

    add_small = 1e-10
    bscores = np.log(
        (b_jets[:, b_index] + add_small) /
        (cfrac * b_jets[:, c_index] + (1 - cfrac)
         * b_jets[:, u_index] + add_small)
        )
    cutvalue = np.percentile(bscores, 100.0 * (1.0 - target_beff))

    c_eff = len(c_jets[np.log(
        (c_jets[:, b_index] + add_small) /
        (cfrac * c_jets[:, c_index] + (1 - cfrac) * c_jets[:, u_index] +
         add_small)) > cutvalue]) / float(len(c_jets) + add_small)
    u_eff = len(u_jets[np.log(
        (u_jets[:, b_index] + add_small) /
        (cfrac * u_jets[:, c_index] + (1 - cfrac) * u_jets[:, u_index] +
         add_small)) > cutvalue]) / float(len(u_jets) + add_small)

    if c_eff == 0:
        c_eff = -1
    if u_eff == 0:
        u_eff = -1
    return 1. / c_eff, 1. / u_eff


class MyCallback(Callback):
    def __init__(self, X_valid=0, Y_valid=0, log_file=None, verbose=False,
                 model_name='test', X_valid_add=None, Y_valid_add=None):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_valid_add = X_valid_add
        self.Y_valid_add = Y_valid_add
        self.result = []
        self.log = open(log_file, 'w') if log_file else None
        self.verbose = verbose
        self.model_name = model_name
        os.system("mkdir -p %s" % self.model_name)
        self.dict_list = []

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('%s/model_epoch%i.h5' % (self.model_name, epoch))
        val_loss, val_acc = self.model.evaluate(self.X_valid,
                                                self.Y_valid,
                                                batch_size=5000)
        y_pred = self.model.predict(self.X_valid, batch_size=5000)
        c_rej, u_rej = GetRejection(y_pred, self.Y_valid)
        print("c-rej:", c_rej, "u-rej:", u_rej)
        add_loss, add_acc, c_rej_add, u_rej_add = None, None, None, None
        if self.X_valid_add is not None:
            add_loss, add_acc = self.model.evaluate(self.X_valid_add,
                                                    self.Y_valid_add,
                                                    batch_size=5000)
            y_pred_add = self.model.predict(self.X_valid_add, batch_size=5000)
            c_rej_add, u_rej_add = GetRejection(y_pred_add, self.Y_valid_add)
        dict_epoch = {
            "epoch": epoch,
            "loss": logs['loss'],
            "acc": logs['accuracy'],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "c_rej": c_rej,
            "u_rej": u_rej,
            "val_loss_add": add_loss if add_loss else None,
            "val_accuracy_add": add_acc if add_acc else None,
            "c_rej_add": c_rej_add if c_rej_add else None,
            "u_rej_add": u_rej_add if u_rej_add else None
        }

        self.dict_list.append(dict_epoch)
        with open('%s/DictFile.json' % self.model_name, 'w') as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


class MyCallbackUmami(Callback):
    def __init__(
        self,
        X_valid=0,
        Y_valid=0,
        log_file=None,
        verbose=False,
        model_name="test",
        X_valid_add=None,
        Y_valid_add=None,
        X_valid_trk=None,
        X_valid_trk_add=None,
    ):
        self.X_valid = X_valid
        self.X_valid_trk = X_valid_trk
        self.Y_valid = Y_valid
        self.X_valid_add = X_valid_add
        self.X_valid_trk_add = X_valid_trk_add
        self.Y_valid_add = Y_valid_add
        self.result = []
        self.log = open(log_file, "w") if log_file else None
        self.verbose = verbose
        self.model_name = model_name
        os.system("mkdir -p %s" % self.model_name)
        self.dict_list = []

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"{self.model_name}/model_epoch{epoch}.h5")
        (
            loss,
            dips_loss,
            umami_loss,
            dips_accuracy,
            umami_accuracy,
        ) = self.model.evaluate(
            [self.X_valid_trk, self.X_valid],
            self.Y_valid,
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )
        # loss: - dips_loss: - umami_loss: - dips_accuracy:  - umami_accuracy:
        y_pred_dips, y_pred_umami = self.model.predict(
            [self.X_valid_trk, self.X_valid],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )
        c_rej_dips, u_rej_dips = GetRejection(y_pred_dips, self.Y_valid)
        c_rej_umami, u_rej_umami = GetRejection(y_pred_umami, self.Y_valid)
        print("Dips:", "c-rej:", c_rej_dips, "u-rej:", u_rej_dips)
        print("Umami:", "c-rej:", c_rej_umami, "u-rej:", u_rej_umami)
        (
            loss_add,
            dips_loss_add,
            umami_loss_add,
            dips_accuracy_add,
            umami_accuracy_add,
            c_rej_dips_add,
            u_rej_dips_add,
            c_rej_umami_add,
            u_rej_umami_add,
        ) = (None, None, None, None, None, None, None, None, None)
        if self.X_valid_add is not None:
            (
                loss_add,
                dips_loss_add,
                umami_loss_add,
                dips_accuracy_add,
                umami_accuracy_add,
            ) = self.model.evaluate(
                [self.X_valid_trk_add, self.X_valid_add],
                self.Y_valid_add,
                batch_size=5000,
                use_multiprocessing=True,
                workers=8,
                verbose=0,
            )
            y_pred_dips_add, y_pred_umami_add = self.model.predict(
                [self.X_valid_trk_add, self.X_valid_add],
                batch_size=5000,
                use_multiprocessing=True,
                workers=8,
                verbose=0,
            )
            c_rej_dips_add, u_rej_dips_add = GetRejection(
                y_pred_dips_add, self.Y_valid_add
            )
            c_rej_umami_add, u_rej_umami_add = GetRejection(
                y_pred_umami_add, self.Y_valid_add
            )
        dict_epoch = {
            "epoch": epoch,
            "loss": logs["loss"],
            "dips_loss": logs["dips_loss"],
            "umami_loss": logs["umami_loss"],
            "dips_acc": logs["dips_accuracy"],
            "umami_acc": logs["umami_accuracy"],
            "val_loss": loss,
            "dips_val_loss": dips_loss,
            "umami_val_loss": umami_loss,
            "dips_val_acc": dips_accuracy,
            "umami_val_acc": umami_accuracy,
            "val_loss_add": loss_add,
            "dips_val_loss_add": dips_loss_add,
            "umami_val_loss_add": umami_loss_add,
            "dips_val_acc_add": dips_accuracy_add,
            "umami_val_acc_add": umami_accuracy_add,
            "c_rej_dips": c_rej_dips,
            "c_rej_umami": c_rej_umami,
            "u_rej_dips": u_rej_dips,
            "u_rej_umami": u_rej_umami,
            "c_rej_dips_add": c_rej_dips_add,
            "c_rej_umami_add": c_rej_umami_add,
            "u_rej_dips_add": u_rej_dips_add,
            "u_rej_umami_add": u_rej_umami_add,
        }

        self.dict_list.append(dict_epoch)
        with open(f"{self.model_name}/DictFile.json", "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


def GetTestSample(input_file, var_dict, preprocess_config,
                  nJets=int(3e5)):
    """
        Apply the scaling and shifting to dataset using numpy
    """

    jets = pd.DataFrame(h5py.File(input_file, 'r')['/jets'][:nJets])
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    jets.query(f"{variable_config['label']} <= 5", inplace=True)
    labels = GetBinaryLabels(jets[variable_config['label']].values)
    variables = variable_config["train_variables"]
    jets = jets[variables]
    jets = jets.replace([np.inf, -np.inf], np.nan)
    with open(preprocess_config.dict_file, 'r') as infile:
        scale_dict = json.load(infile)['jets']
    print("Replacing default values.")
    default_dict = Gen_default_dict(scale_dict)
    jets = jets.fillna(default_dict)
    print("Applying scaling and shifting.")
    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        else:
            jets[elem['name']] -= elem['shift']
            jets[elem['name']] /= elem['scale']
    return jets.values, labels


def GetTestSampleTrks(input_file, var_dict, preprocess_config,
                      nJets=int(3e5)):
    """
        Apply the scaling and shifting to dataset using numpy
    """
    print("Loading validation data tracks")
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    labels = h5py.File(input_file, 'r')['/jets'][:nJets][
        variable_config['label']]
    indices_toremove = np.where(labels > 5)[0]
    labels = np.delete(labels, indices_toremove, 0)

    labels = GetBinaryLabels(labels)

    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trkVars = noNormVars + logNormVars + jointNormVars

    trks = np.asarray(h5py.File(input_file, 'r')['/tracks'][:nJets])
    trks = np.delete(trks, indices_toremove, 0)

    with open(preprocess_config.dict_file, 'r') as infile:
        scale_dict = json.load(infile)['tracks']

    var_arr_list = []
    trk_mask = ~np.isnan(trks["ptfrac"])
    for var in trkVars:
        if var in logNormVars:
            x = np.log(trks[var])
        else:
            x = trks[var]
        if var in logNormVars:
            x -= scale_dict[var]["shift"]
            x /= scale_dict[var]["scale"]
        elif var in jointNormVars:
            x = np.where(trk_mask, x - scale_dict[var]["shift"], x)
            x = np.where(trk_mask, x / scale_dict[var]["scale"], x)
        var_arr_list.append(np.nan_to_num(x))

    return np.stack(var_arr_list, axis=-1), labels


def GetTestFile(file: str, var_dict: str, preprocess_config: dict, nJets: int):
    X_trk, Y_trk = GetTestSampleTrks(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config, nJets=nJets)

    X, Y = GetTestSample(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config, nJets=nJets)

    assert np.equal(Y, Y_trk).all()

    return X, X_trk, Y


class Sum(Layer):
    """
    Simple sum layer.
    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on their
    own.

    Author: Dan Guest
    https://github.com/dguest/flow-network/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:, :, None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
