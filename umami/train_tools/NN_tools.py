from umami.configuration import logger  # isort:skip
import json
import os
import re
from glob import glob
from pathlib import Path
from shutil import copyfile

import h5py
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

from umami.preprocessing_tools import Configuration as Preprocess_Configuration
from umami.preprocessing_tools import Gen_default_dict, GetBinaryLabels
from umami.tools import replaceLineInFile, yaml_loader
from umami.train_tools import Configuration as Train_Configuration


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_validation_dict_name(WP_b, fc_value, n_jets, dir_name):
    return os.path.join(
        dir_name,
        f"validation_WP{str(WP_b).replace('.','p')}_fc{str(fc_value).replace('.','p')}_{int(n_jets)}jets_Dict.json",
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


def create_metadata_folder(
    train_config_path,
    model_name,
    preprocess_config_path,
    overwrite_config=False,
):
    # Check if model path already existing
    # If not, make it
    os.makedirs(os.path.join(model_name, "metadata"), exist_ok=True)

    # Get Variable dict
    train_config = Train_Configuration(train_config_path)
    var_dict_path = train_config.var_dict

    # Get scale dict
    preprocess_config = Preprocess_Configuration(preprocess_config_path)
    scale_dict_path = preprocess_config.dict_file

    # Copy files to metadata folder if not existing
    for file_path in [
        train_config_path,
        preprocess_config_path,
        var_dict_path,
        scale_dict_path,
    ]:
        if overwrite_config is True:
            logger.info(f"Copy {file_path} to metadata folder!")
            copyfile(
                file_path,
                os.path.join(
                    model_name, "metadata", os.path.basename(file_path)
                ),
            )

            # Change the paths for the preprocess config and var dict in the train_config
            if file_path == train_config_path:
                metadata_preprocess_config_path = os.path.join(
                    os.getcwd(),
                    model_name,
                    "metadata",
                    os.path.basename(preprocess_config_path),
                )

                metadata_var_dict_path = os.path.join(
                    os.getcwd(),
                    model_name,
                    "metadata",
                    os.path.basename(var_dict_path),
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    "preprocess_config:",
                    f"preprocess_config: {metadata_preprocess_config_path}",
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    "var_dict:",
                    f"var_dict: {metadata_var_dict_path}",
                )

            if file_path == preprocess_config_path:
                metadata_scale_dict_path = os.path.join(
                    os.getcwd(),
                    model_name,
                    "metadata",
                    os.path.basename(scale_dict_path),
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    "dict_file:",
                    f"dict_file: {metadata_scale_dict_path}",
                )

        elif not os.path.isfile(
            os.path.join(model_name, "metadata", os.path.basename(file_path))
        ):
            logger.info(f"Copy {file_path} to metadata folder!")
            copyfile(
                file_path,
                os.path.join(
                    model_name, "metadata", os.path.basename(file_path)
                ),
            )

            # Change the paths for the preprocess config and var dict in the train_config
            if file_path == train_config_path:
                metadata_preprocess_config_path = os.path.join(
                    model_name,
                    "metadata",
                    os.path.basename(preprocess_config_path),
                )

                metadata_var_dict_path = os.path.join(
                    model_name, "metadata", os.path.basename(var_dict_path)
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    "preprocess_config:",
                    f"  preprocess_config: {metadata_preprocess_config_path}",
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    "var_dict:",
                    f"  var_dict: {metadata_var_dict_path}",
                )


def GetRejection(
    y_pred,
    y_true,
    target_eff=0.77,
    frac=0.018,
    d_type="b",
    taufrac=None,
    use_taus=False,
):
    """
    Calculates the rejections for specific WP and fraction.
    Can perform b or c rejection depending on the value of
    - d_type = "b" or "c"
    If doing b (c) rejection, frac corresponds to fc (fb).
    """
    tau_index, b_index, c_index, u_index = 3, 2, 1, 0
    y_true = np.argmax(y_true, axis=1)
    b_jets = y_pred[y_true == b_index]
    c_jets = y_pred[y_true == c_index]
    u_jets = y_pred[y_true == u_index]

    bool_b = False
    main_flavour = c_jets
    second_flavour = b_jets
    if d_type == "b":
        bool_b = True
        main_flavour = b_jets
        second_flavour = c_jets

    ufrac = 1 - frac
    if use_taus:
        tau_jets = y_pred[y_true == tau_index]
        if bool_b:
            index_list = [b_index, c_index, u_index, tau_index]
        else:
            index_list = [c_index, b_index, u_index, tau_index]
        if taufrac is None:
            taufrac = 1 - frac
        else:
            ufrac = 1 - frac - taufrac
    else:
        if bool_b:
            index_list = [b_index, c_index, u_index]
        else:
            index_list = [c_index, b_index, u_index]
        taufrac = None

    add_small = 1e-10

    def frac_template(jet_flav, index=index_list):
        """
        Jet flav is the jet flavour used.
        Index is a list with either:
        - ["b_index", "c_index", "u_index"]
        - ["b_index", "c_index", "u_index", "tau_index"]
        - ["c_index", "b_index", "u_index"]
        - ["c_index", "b_index", "u_index", "tau_index"]
        First two for b tagging, second two for c tagging
        """
        if len(index) == 4:
            frac_value = np.log(
                (jet_flav[:, index[0]] + add_small)
                / (
                    frac * jet_flav[:, index[1]]
                    + ufrac * jet_flav[:, index[2]]
                    + taufrac * jet_flav[:, index[3]]
                    + add_small
                )
            )
        else:
            frac_value = np.log(
                (jet_flav[:, index[0]] + add_small)
                / (
                    frac * jet_flav[:, index[1]]
                    + ufrac * jet_flav[:, index[2]]
                    + add_small
                )
            )
        return frac_value

    disc_score = frac_template(main_flavour)
    cutvalue = np.percentile(disc_score, 100.0 * (1.0 - target_eff))
    # Starting rejection:
    # Secondary flavour (c if b_type == "b", b otherwise)
    second_eff = len(
        second_flavour[frac_template(second_flavour) > cutvalue]
    ) / float(len(second_flavour) + add_small)
    # Light
    light_eff = len(u_jets[frac_template(u_jets) > cutvalue]) / float(
        len(u_jets) + add_small
    )
    # Taus
    if use_taus:
        tau_eff = len(tau_jets[frac_template(tau_jets) > cutvalue]) / float(
            len(tau_jets) + add_small
        )

    if second_eff == 0:
        second_eff = -1
    if light_eff == 0:
        light_eff = -1
    if use_taus and tau_eff == 0:
        tau_eff = -1

    if use_taus:
        return 1.0 / second_eff, 1.0 / light_eff, 1.0 / tau_eff, cutvalue
    else:
        return 1.0 / second_eff, 1.0 / light_eff, cutvalue


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
        include_taus=False,
        eval_config=None,
    ):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_valid_add = X_valid_add
        self.Y_valid_add = Y_valid_add
        self.result = []
        self.include_taus = include_taus
        self.log = open(log_file, "w") if log_file else None
        self.verbose = verbose
        self.model_name = model_name
        setup_output_directory(self.model_name)
        self.dict_list = []
        if eval_config is not None:
            if (
                "fc_value" in eval_config
                and eval_config["fc_value"] is not None
            ):
                self.fc_value = eval_config["fc_value"]
            else:
                self.fc_value = 0.018
            if (
                "fb_value" in eval_config
                and eval_config["fb_value"] is not None
            ):
                self.fb_value = eval_config["fb_value"]
            else:
                self.fb_value = 0.2
            if include_taus:
                if "ftauforb_value" in eval_config:
                    self.ftauforb_value = eval_config["ftauforb_value"]
                else:
                    self.ftauforb_value = None
                if "ftauforc_value" in eval_config:
                    self.ftauforc_value = eval_config["ftauforc_value"]
                else:
                    self.ftauforc_value = None
        else:
            self.fc_value = 0.018
            self.fb_value = 0.2
            if include_taus:
                self.ftauforb_value = None
                self.ftauforc_value = None

    def on_epoch_end(self, epoch, logs=None):
        self.model.save("%s/model_epoch%i.h5" % (self.model_name, epoch))
        val_loss, val_acc = self.model.evaluate(
            self.X_valid, self.Y_valid, batch_size=5000
        )
        y_pred = self.model.predict(self.X_valid, batch_size=5000)
        if self.include_taus:
            c_rej, u_rej, tau_rej, disc_cut = GetRejection(
                y_pred,
                self.Y_valid,
                frac=self.fc_value,
                taufrac=self.ftauforb_value,
                use_taus=self.include_taus,
            )
            logger.info(
                f"For b: c-rej: {c_rej} u-rej: {u_rej} tau-rej: {tau_rej}"
            )
            b_rejC, u_rejC, tau_rejC, disc_cutC = GetRejection(
                y_pred,
                self.Y_valid,
                d_type="c",
                frac=self.fb_value,
                taufrac=self.ftauforc_value,
                use_taus=self.include_taus,
            )
            logger.info(
                f"For c: b-rej: {b_rejC} u-rej: {u_rejC} tau-rej: {tau_rejC}"
            )
        else:
            c_rej, u_rej, disc_cut = GetRejection(
                y_pred,
                self.Y_valid,
                frac=self.fc_value,
            )
            logger.info(f"For b: c-rej: {c_rej} u-rej: {u_rej}")
            b_rejC, u_rejC, disc_cutC = GetRejection(
                y_pred,
                self.Y_valid,
                d_type="c",
                frac=self.fb_value,
            )
            logger.info(f"For c: b-rej: {b_rejC} u-rej: {u_rejC}")
        add_loss, add_acc, c_rej_add, u_rej_add = None, None, None, None
        b_rejC_add, u_rejC_add = None, None
        if self.include_taus:
            tau_rej_add, tau_rejC_add = None, None

        if self.X_valid_add is not None:
            add_loss, add_acc = self.model.evaluate(
                self.X_valid_add, self.Y_valid_add, batch_size=5000
            )
            y_pred_add = self.model.predict(self.X_valid_add, batch_size=5000)

            if self.include_taus:
                c_rej_add, u_rej_add, tau_rej_add, disc_cut_add = GetRejection(
                    y_pred_add,
                    self.Y_valid_add,
                    frac=self.fc_value,
                    taufrac=self.ftauforb_value,
                    use_taus=self.include_taus,
                )
                (
                    b_rejC_add,
                    u_rejC_add,
                    tau_rejC_add,
                    disc_cutC_add,
                ) = GetRejection(
                    y_pred_add,
                    self.Y_valid_add,
                    d_type="c",
                    frac=self.fb_value,
                    taufrac=self.ftauforc_value,
                    use_taus=self.include_taus,
                )
            else:
                c_rej_add, u_rej_add, disc_cut_add = GetRejection(
                    y_pred_add, self.Y_valid_add, frac=self.fc_value
                )
                b_rejC_add, u_rejC_add, disc_cutC_add = GetRejection(
                    y_pred_add,
                    self.Y_valid_add,
                    d_type="c",
                    frac=self.fb_value,
                )
        dict_epoch = {
            "epoch": epoch,
            "loss": logs["loss"],
            "acc": logs["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "c_rej": c_rej,
            "u_rej": u_rej,
            "disc_cut": disc_cut,
            "b_rejC": b_rejC,
            "u_rejC": u_rejC,
            "disc_cutC": disc_cutC,
            "val_loss_add": add_loss if add_loss else None,
            "val_acc_add": add_acc if add_acc else None,
            "c_rej_add": c_rej_add if c_rej_add else None,
            "u_rej_add": u_rej_add if u_rej_add else None,
            "disc_cut_add": disc_cut_add if disc_cut_add else None,
            "b_rejC_add": b_rejC_add if b_rejC_add else None,
            "u_rejC_add": u_rejC_add if u_rejC_add else None,
            "disc_cutC_add": disc_cutC_add if disc_cutC_add else None,
        }
        if self.include_taus:
            dict_epoch["tau_rej"] = tau_rej
            dict_epoch["tau_rej_add"] = tau_rej_add
            dict_epoch["tau_rejC"] = tau_rejC
            dict_epoch["tau_rejC_add"] = tau_rejC_add

        self.dict_list.append(dict_epoch)
        with open("%s/DictFile.json" % self.model_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


def setup_output_directory(dir_name):
    outdir = Path(dir_name)
    if outdir.is_dir():
        logger.info("Removing model*.h5 and *.json files.")
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


class MyCallbackDips(Callback):
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
            "acc": logs["accuracy"],
        }
        if self.val_data_dict:
            result_dict = evaluate_model_dips(
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


def filter_taus(train_set, test_set):
    """
    Small code to filter taus away from the training dataset
    """
    if test_set.shape[1] > 3:
        # Use test set, which has the label, to remove those corresponding to taus in both sets
        # Taus are fourth column (index 3), since label is 15 (after u, c, and b)
        tau_indices = np.where(test_set[:, 3] == 1)[0]
        train_set = np.delete(train_set, tau_indices, axis=0)
        test_set = np.delete(test_set, tau_indices, axis=0)
        test_set = np.delete(test_set, 3, axis=1)  # delete tau label column
    else:
        logger.info(
            f"There does not seem to be any tau data, shape of test is: {test_set.shape[1]}"
        )
    return (train_set, test_set)


def get_jet_feature_indices(variable_header: dict, exclude=None):
    """
    Deletes from the jet samples the keys listed in exclude
    Example of algorithm keys: SV1 or JetFitter
    Works for both sub-aglorithm and variables
    """
    excluded_variables = []
    all_variables = [i for j in variable_header for i in variable_header[j]]
    if exclude is None:
        return all_variables, excluded_variables, None
    missing_header = []
    for exclude_this in exclude:
        if exclude_this in variable_header:
            excluded_variables.extend(variable_header[exclude_this])
            variable_header.pop(exclude_this, None)
        else:
            missing_header.append(exclude_this)
    variables = [i for j in variable_header for i in variable_header[j]]
    # If elements in exclude are not headers, check if they aren't variables
    for exclude_that in missing_header:
        if exclude_that in variables:
            excluded_variables.append(exclude_that)
            variables.remove(exclude_that)
        else:
            logger.info(f"Variables to exclude not found: {exclude_that}")
    # Get the index of the excluded variables for training
    excluded_var_indices = [
        i for i, excl in enumerate(all_variables) if excl in excluded_variables
    ]
    # set to None if the list of excluded variables is empty
    excluded_var_indices = (
        None if len(excluded_var_indices) == 0 else excluded_var_indices
    )
    logger.debug(f"variables: {variables}")
    logger.debug(f"excluded_variables: {excluded_variables}")
    logger.debug(f"excluded_var_indices: {excluded_var_indices}")
    return variables, excluded_variables, excluded_var_indices


def GetTestSample(
    input_file,
    var_dict,
    preprocess_config,
    nJets=int(3e5),
    exclude=[],
    use_taus=False,
):
    """
    Apply the scaling and shifting to dataset using numpy
    """

    # Get the paths of the input file as list
    # In case there are multiple files (Wildcard etc.)
    filepaths = glob(input_file)

    # Load variables
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Load scale dict
    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["jets"]

    # Define a counter for the number of jets already loaded
    nJets_counter = 0

    # Iterate over the list of input files
    for j, file in enumerate(sorted(filepaths, key=natural_keys)):
        logger.info(f"Input file is {file}")

        # Load dataframe from file
        jets = pd.DataFrame(h5py.File(file, "r")["/jets"][:nJets])

        # Selected the wanted flavours from the files
        if use_taus is True:
            jets.query(
                f"{variable_config['label']} in [0, 4, 5, 15]", inplace=True
            )
        else:
            jets.query(
                f"{variable_config['label']} in [0, 4, 5]", inplace=True
            )

        # Get binary labels for training
        labels = GetBinaryLabels(jets[variable_config["label"]].values)

        # Retrieve variables and the excluded variables from the config
        variables, excluded_variables, _ = get_jet_feature_indices(
            variable_config["train_variables"], exclude
        )

        # Select only wanted variables
        jets = jets[variables]

        # Replace inf with nans
        jets = jets.replace([np.inf, -np.inf], np.nan)

        logger.info("Replacing default values.")
        default_dict = Gen_default_dict(scale_dict)
        jets = jets.fillna(default_dict)

        logger.info("Applying scaling and shifting.")
        scale_dict_variables = []
        for elem in scale_dict:
            scale_dict_variables.append(elem["name"])
            if elem["name"] not in variables:
                if elem["name"] in excluded_variables:
                    logger.info(
                        f"{elem['name']} has been excluded from variable config (is in scale dict)."
                    )
                else:
                    logger.warning(
                        f"{elem['name']} in scale dict but not in variable config."
                    )
                continue
            if "isDefaults" in elem["name"]:
                continue
            else:
                jets[elem["name"]] -= elem["shift"]
                jets[elem["name"]] /= elem["scale"]
        if not set(variables).issubset(scale_dict_variables):
            raise KeyError(
                f"Requested {(set(variables).difference(scale_dict_variables))} which are not in scale dict."
            )

        # If not the first file processed, append to the global one
        if j == 0:
            all_jets = jets
            all_labels = labels

        # if the first file processed, set as global one
        else:
            all_jets = all_jets.append(jets, ignore_index=True)
            all_labels = all_labels.append(labels, ignore_index=True)

        nJets_counter += len(all_jets)

        # Stop loading if enough jets are loaded
        if nJets_counter >= nJets:
            break

    # Check if enough jets are loaded
    if nJets_counter < nJets:
        logger.warning(
            f"Requested {nJets} but only {nJets_counter} could be loaded!"
        )

    else:
        logger.info(f"Loaded {nJets_counter} jets!")

    return all_jets, all_labels


def GetTestSampleTrks(
    input_file,
    var_dict,
    preprocess_config,
    nJets=int(3e5),
    use_taus=False,
):
    """
    Apply the scaling and shifting to dataset using numpy
    """

    # Get the paths of the input file as list
    # In case there are multiple files (Wildcard etc.)
    filepaths = glob(input_file)

    # Load variables
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Load scale dict for the tracks
    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["tracks"]

    # Define a counter for the number of jets already loaded
    nJets_counter = 0

    # Iterate over the list of input files
    for j, file in enumerate(sorted(filepaths, key=natural_keys)):
        logger.info(f"Loading validation data tracks from file {file}")

        labels = h5py.File(input_file, "r")["/jets"][:nJets][
            variable_config["label"]
        ]
        indices_toremove = np.where(labels > 5)[0]
        labels = np.delete(labels, indices_toremove, 0)

        labels = GetBinaryLabels(labels)

        # Retrieve variables from config
        noNormVars = variable_config["track_train_variables"]["noNormVars"]
        logNormVars = variable_config["track_train_variables"]["logNormVars"]
        jointNormVars = variable_config["track_train_variables"][
            "jointNormVars"
        ]
        trkVars = noNormVars + logNormVars + jointNormVars

        # Load the tracks from file
        trks = np.asarray(h5py.File(input_file, "r")["/tracks"][:nJets])
        trks = np.delete(trks, indices_toremove, 0)

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

        # If not the first file processed, append to the global one
        if j == 0:
            all_trks = np.stack(var_arr_list, axis=-1)
            all_labels = labels

        # if the first file processed, set as global one
        else:
            np.append(all_trks, np.stack(var_arr_list, axis=-1))
            np.append(all_labels, labels)

        # Stop loading if enough jets are loaded
        if nJets_counter >= nJets:
            break

    # Check if enough jets are loaded
    if nJets_counter < nJets:
        logger.warning(
            f"Requested {nJets} but only {nJets_counter} could be loaded!"
        )

    else:
        logger.info(f"Loaded {nJets_counter} jets!")

    return all_trks, all_labels


def load_validation_data(train_config, preprocess_config, nJets: int):
    exclude = None
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


def load_validation_data_dips(train_config, preprocess_config, nJets: int):
    val_data_dict = {}
    (val_data_dict["X_valid"], val_data_dict["Y_valid"],) = GetTestSampleTrks(
        train_config.validation_file,
        train_config.var_dict,
        preprocess_config,
        nJets=nJets,
    )
    (
        val_data_dict["X_valid_add"],
        val_data_dict["Y_valid_add"],
    ) = (None, None)
    if train_config.add_validation_file is not None:
        (
            val_data_dict["X_valid_add"],
            val_data_dict["Y_valid_add"],
        ) = GetTestSampleTrks(
            train_config.add_validation_file,
            train_config.var_dict,
            preprocess_config,
            nJets=nJets,
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
        nJets=int(nJets),
    )

    X, Y = GetTestSample(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        nJets=int(nJets),
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
    c_rej_dips, u_rej_dips, disc_cut_dips = GetRejection(
        y_pred_dips, data_dict["Y_valid"], target_beff, cfrac
    )
    c_rej_umami, u_rej_umami, disc_cut_umami = GetRejection(
        y_pred_umami, data_dict["Y_valid"], target_beff, cfrac
    )
    logger.info(f"Dips: c-rej: {c_rej_dips} u-rej: {u_rej_dips}")
    logger.info(f"Umami: c-rej: {c_rej_umami} u-rej: {u_rej_umami}")
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
        c_rej_dips_add, u_rej_dips_add, disc_cut_dips_add = GetRejection(
            y_pred_dips_add, data_dict["Y_valid_add"], target_beff, cfrac
        )
        c_rej_umami_add, u_rej_umami_add, disc_cut_umami_add = GetRejection(
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
        "disc_cut_dips": disc_cut_dips,
        "disc_cut_umami": disc_cut_umami,
        "c_rej_dips_add": c_rej_dips_add,
        "c_rej_umami_add": c_rej_umami_add,
        "u_rej_dips_add": u_rej_dips_add,
        "u_rej_umami_add": u_rej_umami_add,
        "disc_cut_dips_add": disc_cut_dips_add,
        "disc_cut_umami_add": disc_cut_umami_add,
    }
    return result_dict


def evaluate_model_dips(model, data_dict, target_beff=0.77, cfrac=0.018):
    loss, accuracy = model.evaluate(
        data_dict["X_valid"],
        data_dict["Y_valid"],
        batch_size=5000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    y_pred_dips = model.predict(
        data_dict["X_valid"],
        batch_size=5000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    c_rej, u_rej, disc_cut = GetRejection(
        y_pred_dips, data_dict["Y_valid"], target_beff, cfrac
    )

    logger.info(f"Dips: c-rej: {c_rej} u-rej: {u_rej}")

    (
        loss_add,
        accuracy_add,
        c_rej_add,
        u_rej_add,
    ) = (None, None, None, None)

    if data_dict["X_valid_add"] is not None:
        loss_add, accuracy_add = model.evaluate(
            data_dict["X_valid_add"],
            data_dict["Y_valid_add"],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        y_pred_add = model.predict(
            data_dict["X_valid_add"],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        c_rej_add, u_rej_add, disc_cut_add = GetRejection(
            y_pred_add, data_dict["Y_valid_add"], target_beff, cfrac
        )

    result_dict = {
        "val_loss": loss,
        "val_acc": accuracy,
        "val_loss_add": loss_add,
        "val_acc_add": accuracy_add,
        "c_rej": c_rej,
        "u_rej": u_rej,
        "c_rej_add": c_rej_add,
        "u_rej_add": u_rej_add,
        "disc_cut": disc_cut,
        "disc_cut_add": disc_cut_add,
    }
    return result_dict


def calc_validation_metrics(
    train_config,
    preprocess_config,
    target_beff=0.77,
    cfrac=0.018,
    nJets=300000,
):
    Eval_parameters = train_config.Eval_parameters_validation

    val_data_dict = load_validation_data(
        train_config, preprocess_config, nJets
    )
    training_output = [
        os.path.join(train_config.model_name, f)
        for f in os.listdir(train_config.model_name)
        if "model" in f
    ]
    try:
        with open(
            get_validation_dict_name(
                WP_b=Eval_parameters["WP_b"],
                fc_value=Eval_parameters["fc_value"],
                n_jets=Eval_parameters["n_jets"],
                dir_name=train_config.model_name,
            ),
            "r",
        ) as training_out_json:
            training_output_list = json.load(training_out_json)

    except FileNotFoundError:
        training_output_list = [
            {"epoch": n} for n in range(train_config.NN_structure["epochs"])
        ]

    results = []
    for n, model_file in enumerate(training_output):
        logger.info(f"Working on {n+1}/{len(training_output)} input files")
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


def calc_validation_metrics_dips(
    train_config,
    preprocess_config,
    target_beff=0.77,
    cfrac=0.018,
    nJets=300000,
):
    Eval_parameters = train_config.Eval_parameters_validation

    val_data_dict = load_validation_data_dips(
        train_config, preprocess_config, nJets
    )
    training_output = [
        os.path.join(train_config.model_name, f)
        for f in os.listdir(train_config.model_name)
        if "model_epoch" in f
    ]
    try:
        with open(
            get_validation_dict_name(
                WP_b=Eval_parameters["WP_b"],
                fc_value=Eval_parameters["fc_value"],
                n_jets=Eval_parameters["n_jets"],
                dir_name=train_config.model_name,
            ),
            "r",
        ) as training_out_json:
            training_output_list = json.load(training_out_json)

    except FileNotFoundError:
        training_output_list = [
            {"epoch": n} for n in range(train_config.NN_structure["epochs"])
        ]

    results = []
    for n, model_file in enumerate(sorted(training_output, key=natural_keys)):
        logger.info(f"Working on {n+1}/{len(training_output)} input files")
        result_dict = {}
        epoch = int(
            model_file[
                model_file.rfind("model_epoch") + 11 : model_file.find(".h5")
            ]
        )
        for train_epoch in training_output_list:
            if epoch == train_epoch["epoch"]:
                result_dict = train_epoch

        dips = load_model(model_file, {"Sum": Sum})
        val_result_dict = evaluate_model_dips(
            dips, val_data_dict, target_beff, cfrac
        )
        for k, v in val_result_dict.items():
            result_dict[k] = v
        results.append(result_dict)
        del dips

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
