import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from keras import backend as K
from keras.layers import Layer
from keras.models import load_model
from tensorflow.keras.callbacks import Callback

from umami.preprocessing_tools import Gen_default_dict, GetBinaryLabels
from umami.tools import yaml_loader


def get_validation_dict_name(WP_b, fc_value, n_jets, dir_name, plot_label=""):
    return os.path.join(
        dir_name,
        f"validation_WP{str(WP_b).replace('.','p')}_fc{str(fc_value).replace('.','p')}_{n_jets}jets_Dict.json",
    )


def get_parameters_from_validation_dict_name(dict_name):
    sp = dict_name.split("/")[-1].split("_")
    parameters = {}
    parameters["WP_b"] = float(sp[1].replace("WP", "").replace("p", "."))
    parameters["fc_value"] = float(sp[2].replace("fc", "").replace("p", "."))
    parameters["n_jets"] = int(sp[3].replace("jets", ""))
    parameters["dir_name"] = str(Path(dict_name).parent)
    if get_validation_dict_name(**parameters) != dict_name:
        raise Exception(
            f"Can't infer parameters correctly for {dict_name}. Parameters: {parameters}"
        )
    return parameters


def GetRejection(y_pred, y_true, target_beff=0.77, cfrac=0.018):
    """Calculates the c and light rejection for specific WP and c-fraction."""
    b_index, c_index, u_index = 2, 1, 0
    y_true = np.argmax(y_true, axis=1)
    b_jets = y_pred[y_true == b_index]
    c_jets = y_pred[y_true == c_index]
    u_jets = y_pred[y_true == u_index]

    add_small = 1e-10
    bscores = np.log(
        (b_jets[:, b_index] + add_small)
        / (
            cfrac * b_jets[:, c_index]
            + (1 - cfrac) * b_jets[:, u_index]
            + add_small
        )
    )
    cutvalue = np.percentile(bscores, 100.0 * (1.0 - target_beff))

    c_eff = len(
        c_jets[
            np.log(
                (c_jets[:, b_index] + add_small)
                / (
                    cfrac * c_jets[:, c_index]
                    + (1 - cfrac) * c_jets[:, u_index]
                    + add_small
                )
            )
            > cutvalue
        ]
    ) / float(len(c_jets) + add_small)
    u_eff = len(
        u_jets[
            np.log(
                (u_jets[:, b_index] + add_small)
                / (
                    cfrac * u_jets[:, c_index]
                    + (1 - cfrac) * u_jets[:, u_index]
                    + add_small
                )
            )
            > cutvalue
        ]
    ) / float(len(u_jets) + add_small)

    if c_eff == 0:
        c_eff = -1
    if u_eff == 0:
        u_eff = -1
    return 1.0 / c_eff, 1.0 / u_eff


class MyCallback(Callback):
    def __init__(
        self,
        X_valid=0,
        Y_valid=0,
        log_file=None,
        verbose=False,
        model_name="test",
        X_valid_add=None,
        Y_valid_add=None,
    ):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_valid_add = X_valid_add
        self.Y_valid_add = Y_valid_add
        self.result = []
        self.log = open(log_file, "w") if log_file else None
        self.verbose = verbose
        self.model_name = model_name
        setup_output_directory(self.model_name)
        self.dict_list = []

    def on_epoch_end(self, epoch, logs=None):
        self.model.save("%s/model_epoch%i.h5" % (self.model_name, epoch))
        val_loss, val_acc = self.model.evaluate(
            self.X_valid, self.Y_valid, batch_size=5000
        )
        y_pred = self.model.predict(self.X_valid, batch_size=5000)
        c_rej, u_rej = GetRejection(y_pred, self.Y_valid)
        print("c-rej:", c_rej, "u-rej:", u_rej)
        add_loss, add_acc, c_rej_add, u_rej_add = None, None, None, None
        if self.X_valid_add is not None:
            add_loss, add_acc = self.model.evaluate(
                self.X_valid_add, self.Y_valid_add, batch_size=5000
            )
            y_pred_add = self.model.predict(self.X_valid_add, batch_size=5000)
            c_rej_add, u_rej_add = GetRejection(y_pred_add, self.Y_valid_add)
        dict_epoch = {
            "epoch": epoch,
            "loss": logs["loss"],
            "acc": logs["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "c_rej": c_rej,
            "u_rej": u_rej,
            "val_loss_add": add_loss if add_loss else None,
            "val_accuracy_add": add_acc if add_acc else None,
            "c_rej_add": c_rej_add if c_rej_add else None,
            "u_rej_add": u_rej_add if u_rej_add else None,
        }

        self.dict_list.append(dict_epoch)
        with open("%s/DictFile.json" % self.model_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


def setup_output_directory(dir_name):
    outdir = Path(dir_name)
    if outdir.is_dir():
        print("Removing model*.5 and *.json files.")
        for model_file in outdir.glob("model*.h5"):
            model_file.unlink()
        for model_file in outdir.glob("*.json"):
            model_file.unlink()
    elif outdir.is_file():
        raise Exception(
            f"{dir_name} is the output directory name but it already exists as a file!"
        )
    else:
        outdir.mkdir()


class MyCallbackUmami(Callback):
    def __init__(
        self,
        val_data_dict=None,
        log_file=None,
        verbose=False,
        model_name="test",
        target_beff=0.77,
        charm_fraction=0.018,
        dict_file_name="DictFile.json",
    ):
        self.val_data_dict = val_data_dict
        self.target_beff = target_beff
        self.charm_fraction = charm_fraction
        self.result = []
        self.log = open(log_file, "w") if log_file else None
        self.verbose = verbose
        self.model_name = model_name
        setup_output_directory(self.model_name)
        self.dict_list = []
        self.dict_file_name = dict_file_name

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"{self.model_name}/model_epoch{epoch}.h5")
        dict_epoch = {
            "epoch": epoch,
            "learning_rate": logs["lr"].item(),
            "loss": logs["loss"],
            "dips_loss": logs["dips_loss"],
            "umami_loss": logs["umami_loss"],
            "dips_acc": logs["dips_accuracy"],
            "umami_acc": logs["umami_accuracy"],
        }
        if self.val_data_dict:
            result_dict = evaluate_model(
                self.model,
                self.val_data_dict,
                self.target_beff,
                self.charm_fraction,
            )
            # Once we use python >=3.9 (see https://www.python.org/dev/peps/pep-0584/#specification) switch to the following: dict_epoch |= result_dict
            dict_epoch = {**dict_epoch, **result_dict}

        self.dict_list.append(dict_epoch)
        with open(self.dict_file_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


def get_jet_feature_indicies(exclude: list = []):
    indicies = list(range(41))
    algo_indicies = {
        "SV1": range(8, 17),
        "IP2D": range(17, 21),
        "IP3D": range(21, 25),
    }
    for exclude_this in exclude:
        if exclude_this in algo_indicies:
            print(
                f"INFO: Excluding {exclude_this} from the jet feature inputs!"
            )
            for index in algo_indicies[exclude_this]:
                indicies.remove(index)
        else:
            print(f"ERROR: Can't find {exclude_this} in {algo_indicies}!")
            exit(1)
    return indicies


def GetTestSample(
    input_file, var_dict, preprocess_config, nJets=int(3e5), exclude=[]
):
    """
    Apply the scaling and shifting to dataset using numpy
    """

    jets = pd.DataFrame(h5py.File(input_file, "r")["/jets"][:nJets])
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    jets.query(f"{variable_config['label']} <= 5", inplace=True)
    labels = GetBinaryLabels(jets[variable_config["label"]].values)
    variables = variable_config["train_variables"]
    jets = jets[variables]
    jets = jets.replace([np.inf, -np.inf], np.nan)
    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["jets"]
    print("Replacing default values.")
    default_dict = Gen_default_dict(scale_dict)
    jets = jets.fillna(default_dict)
    print("Applying scaling and shifting.")
    for elem in scale_dict:
        if elem["name"] not in variables:
            print(elem["name"], "in scale dict but not in variable config.")
            continue
        if "isDefaults" in elem["name"]:
            continue
        else:
            jets[elem["name"]] -= elem["shift"]
            jets[elem["name"]] /= elem["scale"]
    return jets.values[:, get_jet_feature_indicies(exclude)], labels


def GetTestSampleTrks(input_file, var_dict, preprocess_config, nJets=int(3e5)):
    """
    Apply the scaling and shifting to dataset using numpy
    """
    print("Loading validation data tracks")
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)
    labels = h5py.File(input_file, "r")["/jets"][:nJets][
        variable_config["label"]
    ]
    indices_toremove = np.where(labels > 5)[0]
    labels = np.delete(labels, indices_toremove, 0)

    labels = GetBinaryLabels(labels)

    noNormVars = variable_config["track_train_variables"]["noNormVars"]
    logNormVars = variable_config["track_train_variables"]["logNormVars"]
    jointNormVars = variable_config["track_train_variables"]["jointNormVars"]
    trkVars = noNormVars + logNormVars + jointNormVars

    trks = np.asarray(h5py.File(input_file, "r")["/tracks"][:nJets])
    trks = np.delete(trks, indices_toremove, 0)

    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["tracks"]

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


def load_validation_data(train_config, preprocess_config, nJets: int):
    exclude = []
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]
    val_data_dict = {}
    (
        val_data_dict["X_valid"],
        val_data_dict["X_valid_trk"],
        val_data_dict["Y_valid"],
    ) = GetTestFile(
        train_config.validation_file,
        train_config.var_dict,
        preprocess_config,
        nJets=nJets,
        exclude=exclude,
    )
    (
        val_data_dict["X_valid_add"],
        val_data_dict["Y_valid_add"],
        val_data_dict["X_valid_trk_add"],
    ) = (None, None, None)
    if train_config.add_validation_file is not None:
        (
            val_data_dict["X_valid_add"],
            val_data_dict["X_valid_trk_add"],
            val_data_dict["Y_valid_add"],
        ) = GetTestFile(
            train_config.add_validation_file,
            train_config.var_dict,
            preprocess_config,
            nJets=nJets,
            exclude=exclude,
        )
        assert (
            val_data_dict["X_valid"].shape[1]
            == val_data_dict["X_valid_add"].shape[1]
        )
    return val_data_dict


def GetTestFile(
    file: str,
    var_dict: str,
    preprocess_config: dict,
    nJets: int,
    exclude: list,
):
    X_trk, Y_trk = GetTestSampleTrks(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        nJets=nJets,
    )

    X, Y = GetTestSample(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        nJets=nJets,
        exclude=exclude,
    )

    assert np.equal(Y, Y_trk).all()

    return X, X_trk, Y


def evaluate_model(model, data_dict, target_beff=0.77, cfrac=0.018):
    (
        loss,
        dips_loss,
        umami_loss,
        dips_accuracy,
        umami_accuracy,
    ) = model.evaluate(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        data_dict["Y_valid"],
        batch_size=5000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )
    # loss: - dips_loss: - umami_loss: - dips_accuracy:  - umami_accuracy:
    y_pred_dips, y_pred_umami = model.predict(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        batch_size=5000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )
    c_rej_dips, u_rej_dips = GetRejection(
        y_pred_dips, data_dict["Y_valid"], target_beff, cfrac
    )
    c_rej_umami, u_rej_umami = GetRejection(
        y_pred_umami, data_dict["Y_valid"], target_beff, cfrac
    )
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
    if data_dict["X_valid_add"] is not None:
        (
            loss_add,
            dips_loss_add,
            umami_loss_add,
            dips_accuracy_add,
            umami_accuracy_add,
        ) = model.evaluate(
            [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]],
            data_dict["Y_valid_add"],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )
        y_pred_dips_add, y_pred_umami_add = model.predict(
            [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )
        c_rej_dips_add, u_rej_dips_add = GetRejection(
            y_pred_dips_add, data_dict["Y_valid_add"], target_beff, cfrac
        )
        c_rej_umami_add, u_rej_umami_add = GetRejection(
            y_pred_umami_add, data_dict["Y_valid_add"], target_beff, cfrac
        )
    result_dict = {
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
    return result_dict


def calc_validation_metrics(
    train_config,
    preprocess_config,
    target_beff=0.77,
    cfrac=0.018,
    nJets=300000,
):
    val_data_dict = load_validation_data(
        train_config, preprocess_config, nJets
    )
    training_output = [
        os.path.join(train_config.model_name, f)
        for f in os.listdir(train_config.model_name)
        if "model" in f
    ]
    with open(
        get_validation_dict_name(
            **train_config.Eval_parameters_validation,
            dir_name=train_config.model_name,
        ),
        "r",
    ) as training_out_json:
        training_output_list = json.load(training_out_json)

    results = []
    for n, model_file in enumerate(training_output):
        print(f"Working on {n+1}/{len(training_output)} input files")
        result_dict = {}
        epoch = int(
            model_file[model_file.find("epoch") + 5 : model_file.find(".h5")]
        )
        for train_epoch in training_output_list:
            if epoch == train_epoch["epoch"]:
                result_dict = train_epoch
        umami = load_model(model_file, {"Sum": Sum})
        val_result_dict = evaluate_model(
            umami, val_data_dict, target_beff, cfrac
        )
        for k, v in val_result_dict.items():
            result_dict[k] = v
        results.append(result_dict)
        del umami

    results = sorted(results, key=lambda x: x["epoch"])

    output_file_path = get_validation_dict_name(
        target_beff, cfrac, nJets, train_config.model_name
    )
    with open(output_file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    return output_file_path


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
