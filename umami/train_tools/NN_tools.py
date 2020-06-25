import os
import json
import h5py
import yaml
import numpy as np
import pandas as pd
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


def GetTestSample(input_file, var_dict, preprocess_config):
    """
        Apply the scaling and shifting to dataset using numpy
    """

    jets = pd.DataFrame(h5py.File(input_file, 'r')['/jets'][:])
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
