from umami.configuration import global_config, logger  # isort:skip
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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_validation_dict_name(WP_b, n_jets, dir_name):
    return os.path.join(
        dir_name,
        f"validation_WP{str(WP_b).replace('.','p')}_{int(n_jets)}jets_Dict.json",
    )


def get_class_label_ids(class_labels):
    """
    This function retrieves the flavour ids of the class_labels provided
    and returns them as a list.
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new list

    for counter, class_label in enumerate(class_labels):
        if counter == 0:
            id_list = np.asarray(
                flavour_categories[class_label]["label_value"]
            )

        else:
            id_list = np.append(
                id_list,
                np.asarray(flavour_categories[class_label]["label_value"]),
            )

    # Flatten the list if needed and return it
    return id_list.tolist()


def get_class_label_variables(class_labels):
    """
    This function returns a list of the label variables used for the
    provided class_labels.
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new lists
    label_var_list = []
    flatten_class_labels = []

    for class_label in class_labels:
        if type(flavour_categories[class_label]["label_value"]) is list:
            for i in range(
                len(flavour_categories[class_label]["label_value"])
            ):
                label_var_list.append(
                    flavour_categories[class_label]["label_var"]
                )
                flatten_class_labels.append(class_label)

        else:
            label_var_list.append(flavour_categories[class_label]["label_var"])
            flatten_class_labels.append(class_label)

    # Flatten the lists if needed
    label_var_list = np.asarray(label_var_list).flatten().tolist()
    flatten_class_labels = np.asarray(flatten_class_labels).flatten().tolist()

    return label_var_list, flatten_class_labels


def get_class_prob_var_names(tagger_name, class_labels):
    """
    This function returns a list of the probability variable names used for the
    provided class_labels.
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new list
    prob_var_list = []

    # Append the prob var names to new list
    for class_label in class_labels:
        prob_var_list.append(
            tagger_name
            + "_"
            + flavour_categories[class_label]["prob_var_name"]
        )

    # Return list of prob var names in correct order
    return prob_var_list


def get_parameters_from_validation_dict_name(dict_name):
    sp = dict_name.split("/")[-1].split("_")
    parameters = {}
    parameters["WP_b"] = float(sp[1].replace("WP", "").replace("p", "."))
    parameters["n_jets"] = int(sp[2].replace("jets", ""))
    parameters["dir_name"] = str(Path(dict_name).parent)
    if get_validation_dict_name(**parameters) != dict_name:
        raise Exception(
            f"Can't infer parameters correctly for {dict_name}. Parameters: {parameters}"
        )
    return parameters


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


def create_metadata_folder(
    train_config_path,
    var_dict_path,
    model_name,
    preprocess_config_path,
    overwrite_config=False,
):
    # Check if model path already existing
    # If not, make it
    os.makedirs(os.path.join(model_name, "metadata"), exist_ok=True)

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


def CalcDiscValues(
    jets_dict: dict,
    index_dict: dict,
    main_class: str,
    frac_dict: dict,
    rej_class=None,
):
    """
    Calculate the disc value based on the flavours used.

    Input:
    - jets_dict: A dict with the class_labels and their jets.
    - index_dict: A dict with the class_labels and their respective indices.
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_dict: A dict with the respective fractions for each class provided
            except main_class

    Output:
    - disc_score: Tagging discriminator score for the main flavour
    """

    # Set the rejection class for rejection calculation
    if rej_class is None:
        rej_class = main_class

    # Init denominator of disc_score and add_small
    denominator = 0
    add_small = 1e-10

    # Get class_labels list without main class
    class_labels_wo_main = list(jets_dict.keys())
    class_labels_wo_main.remove(main_class)

    # Calculate counter of disc_score
    counter = jets_dict[rej_class][:, index_dict[main_class]] + add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += (
            frac_dict[class_label]
            * jets_dict[rej_class][:, index_dict[class_label]]
        )
    denominator += add_small

    # Calculate final disc_score and return it
    return np.log(counter / denominator)


def GetRejection(
    y_pred,
    y_true,
    class_labels: list,
    main_class: str,
    frac_dict: dict = {"cjets": 0.018, "ujets": 0.982},
    target_eff=0.77,
):
    """
    Calculates the rejections for a specific WP for all provided
    classes except the discriminant class (main_class).

    Input:
    - y_pred: The prediction output of the NN
    - y_true: The true class of the jets
    - class_labels: A list of the class_labels which are used
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_dict: A dict with the respective fractions for each class provided
            except main_class
    - target_eff: WP which is used for discriminant calculation.

    Output:
    - Rejection_Dict: Dict of the rejections. The keys of the dict
                      are the provided class_labels without main_class
    """

    # Init new dict for jets and indices
    jets_dict = {}
    index_dict = {}
    rej_dict = {}

    # Get max value of y_true
    y_true = np.argmax(y_true, axis=1)

    # Iterate over the different class_labels and select their respective jets
    for class_counter, class_label in enumerate(class_labels):
        jets_dict.update({f"{class_label}": y_pred[y_true == class_counter]})
        index_dict.update({f"{class_label}": class_counter})

    # Calculate disc score
    disc_scores = CalcDiscValues(
        jets_dict=jets_dict,
        index_dict=index_dict,
        main_class=main_class,
        frac_dict=frac_dict,
    )

    # Calculate cutvalue on the discriminant depending of the WP
    cutvalue = np.percentile(disc_scores, 100.0 * (1.0 - target_eff))

    # Get all non-main flavours
    class_labels_wo_main = class_labels
    class_labels_wo_main.remove(main_class)

    # Calculate efficiencies
    for iter_main_class in class_labels_wo_main:
        rej_dict[f"{iter_main_class}_rej"] = (
            len(
                jets_dict[iter_main_class][
                    CalcDiscValues(
                        jets_dict=jets_dict,
                        index_dict=index_dict,
                        main_class=main_class,
                        frac_dict=frac_dict,
                        rej_class=iter_main_class,
                    )
                    > cutvalue
                ]
            )
            / (len(jets_dict[iter_main_class]) + 1e-10)
        )

    return rej_dict, cutvalue


class MyCallback(Callback):
    def __init__(
        self,
        class_labels: list,
        main_class: str,
        val_data_dict=None,
        log_file=None,
        verbose=False,
        model_name="test",
        target_beff=0.77,
        frac_dict={
            "cjets": 0.018,
            "ujets": 0.982,
        },
        dict_file_name="DictFile.json",
    ):
        self.class_labels = class_labels
        self.main_class = main_class
        self.val_data_dict = val_data_dict
        self.target_beff = target_beff
        self.frac_dict = frac_dict
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
            result_dict = evaluate_model(
                model=self.model,
                data_dict=self.val_data_dict,
                class_labels=self.class_labels,
                main_class=self.main_class,
                target_beff=self.target_beff,
                frac_dict=self.frac_dict,
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
        class_labels: list,
        main_class: str,
        val_data_dict=None,
        log_file=None,
        verbose=False,
        model_name="test",
        target_beff=0.77,
        frac_dict={
            "cjets": 0.018,
            "ujets": 0.982,
        },
        dict_file_name="DictFile.json",
    ):
        self.val_data_dict = val_data_dict
        self.target_beff = target_beff
        self.frac_dict = frac_dict
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
            result_dict = evaluate_model_umami(
                model=self.model,
                data_dict=self.val_data_dict,
                class_labels=self.class_labels,
                main_class=self.main_class,
                target_beff=self.target_beff,
                frac_dict=self.frac_dict,
            )
            # Once we use python >=3.9 (see https://www.python.org/dev/peps/pep-0584/#specification) switch to the following: dict_epoch |= result_dict
            dict_epoch = {**dict_epoch, **result_dict}

        self.dict_list.append(dict_epoch)
        with open(self.dict_file_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)

    def on_train_end(self, logs=None):
        if self.log:
            self.log.close()


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
    class_labels,
    nJets=int(3e5),
    exclude=[],
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

        # Get class_labels variables etc. from global config
        class_ids = get_class_label_ids(class_labels)
        class_label_vars, flatten_class_labels = get_class_label_variables(
            class_labels
        )

        # Init new column for string labels
        jets["Umami_string_labels"] = np.zeros_like(jets[class_label_vars[0]])
        jets["Umami_labels"] = np.zeros_like(jets[class_label_vars[0]])

        # Change type of column to string
        jets = jets.astype({"Umami_string_labels": "str"})

        # Iterate over the classes and add the correct labels to Umami columns
        for class_id, class_label_var, class_label in zip(
            class_ids, class_label_vars, flatten_class_labels
        ):
            indices_tochange = np.where(
                jets[class_label_var].values == class_id
            )

            # Add a string description which this class is
            jets["Umami_string_labels"].values[indices_tochange] = class_label

            # Add the right column label to class
            jets["Umami_labels"].values[indices_tochange] = class_labels.index(
                class_label
            )

        # Get the indices of the jets that are not used
        indices_toremove = np.where(jets["Umami_string_labels"] == "0")[0]

        # Remove all unused jets
        jets = jets.drop(indices_toremove)

        # Binarize Labels
        labels = GetBinaryLabels(jets["Umami_labels"].values)

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

        # Adding the loaded jets to counter
        nJets_counter += len(all_labels)

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

    print(all_jets.shape)
    return all_jets, all_labels


def GetTestSampleTrks(
    input_file,
    var_dict,
    preprocess_config,
    class_labels,
    nJets=int(3e5),
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

        # Get class_labels variables etc. from global config
        class_ids = get_class_label_ids(class_labels)
        class_label_vars, flatten_class_labels = get_class_label_variables(
            class_labels
        )

        # Load the used label variables from file
        with h5py.File(input_file, "r") as jets:
            for iterator, iter_class_var in enumerate(
                list(dict.fromkeys(class_label_vars))
            ):
                if iterator == 0:
                    labels = pd.DataFrame(
                        jets["/jets"][iter_class_var], columns=[iter_class_var]
                    )

                else:
                    labels[iter_class_var] = jets["/jets"][iter_class_var]

        # Use only the amout of jets requested
        labels = labels[:nJets]

        # Init new column for string labels
        labels["Umami_string_labels"] = np.zeros_like(
            labels[class_label_vars[0]]
        )
        labels["Umami_labels"] = np.zeros_like(labels[class_label_vars[0]])

        # Change type of column to string
        labels = labels.astype({"Umami_string_labels": "str"})

        # Iterate over the classes and add the correct labels to Umami columns
        for (class_id, class_label_var, class_label) in zip(
            class_ids, class_label_vars, flatten_class_labels
        ):
            indices_tochange = np.where(
                labels[class_label_var].values == class_id
            )[0]

            # Add a string description which this class is
            labels["Umami_string_labels"].values[
                indices_tochange
            ] = class_label

            # Add the right column label to class
            labels["Umami_labels"].values[
                indices_tochange
            ] = class_labels.index(class_label)

        # Get the indices of the jets that are not used
        indices_toremove = np.where(labels["Umami_string_labels"] == "0")[0]

        # Remove unused jets from labels
        labels = labels.drop(indices_toremove)

        # Binarize the labels
        binary_labels = GetBinaryLabels(labels["Umami_labels"].values)

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
            all_labels = binary_labels

        # if the first file processed, set as global one
        else:
            np.append(all_trks, np.stack(var_arr_list, axis=-1))
            np.append(all_labels, binary_labels)

        # Adding the loaded jets to counter
        nJets_counter += len(all_labels)

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


def load_validation_data_umami(train_config, preprocess_config, nJets: int):
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]
    val_data_dict = {}
    (
        val_data_dict["X_valid"],
        val_data_dict["X_valid_trk"],
        val_data_dict["Y_valid"],
    ) = GetTestFile(
        file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=train_config.NN_structure["class_labels"],
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
            file=train_config.validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=train_config.NN_structure["class_labels"],
            nJets=nJets,
            exclude=exclude,
        )
        assert (
            val_data_dict["X_valid"].shape[1]
            == val_data_dict["X_valid_add"].shape[1]
        )
    return val_data_dict


def load_validation_data_dips(train_config, preprocess_config, nJets: int):
    """
    Load the validation data for DIPS.

    Input:
    - train_config: Loaded train_config object.
    - train_config: Loaded preprocess_config object.
    - nJets: Number of jets to load.

    Output:
    - val_data_dict: Dict with the validation data
    """

    val_data_dict = {}
    (val_data_dict["X_valid"], val_data_dict["Y_valid"],) = GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=train_config.NN_structure["class_labels"],
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
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=train_config.NN_structure["class_labels"],
            nJets=nJets,
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
    class_labels: list,
    nJets: int,
    exclude: list,
):
    """
    Load the training jets and tracks.

    Input:
    - file: Filepath to the file where the data are loaded from.
    - var_dict: Path to the dict with the variables which are used.
    - preprocess_config: The loaded preprocess config object.
    - class_labels: List of classes used for training of the model.
    - nJets: Number of jets used for evaluation.
    - exclude: List of variables that are to be excluded.

    Output:
    - Returns the X, X_trk and Y for training/evaluation.
    """

    X_trk, Y_trk = GetTestSampleTrks(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=int(nJets),
    )

    X, Y = GetTestSample(
        input_file=file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=int(nJets),
        exclude=exclude,
    )

    assert np.equal(Y, Y_trk).all()

    return X, X_trk, Y


def evaluate_model_umami(
    model,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    target_beff: float = 0.77,
    frac_dict: dict = {"cjets": 0.018, "ujets": 0.982},
):
    """
    Evaluate the UMAMI model on the data provided.

    Input:
    - model: Loaded UMAMI model for evaluation.
    - data_dict: Dict with the loaded data which are to be evaluated.
    - class_labels: List of classes used for training of the model.
    - main_class: Main class which is to be tagged.
    - target_beff: Working Point which is to be used for evaluation.
    - frac_dict: Dict with the fractions of the non-main classes.
                 Sum needs to be one!

    Output:
    - Dict with validation metrics/rejections.
    """

    # Calculate accuracy andloss of UMAMI and Dips part
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

    # Evaluate with the model for predictions
    y_pred_dips, y_pred_umami = model.predict(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        batch_size=5000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    # Get rejections for DIPS and UMAMI
    rej_dict_dips, disc_cut_dips = GetRejection(
        y_pred=y_pred_dips,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
        target_eff=target_beff,
    )
    rej_dict_umami, disc_cut_umami = GetRejection(
        y_pred=y_pred_umami,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
        target_eff=target_beff,
    )

    # Evaluate Models on add_files if given
    (
        loss_add,
        dips_loss_add,
        umami_loss_add,
        dips_accuracy_add,
        umami_accuracy_add,
        rej_dict_umami_add,
        rej_dict_dips_add,
        disc_cut_umami_add,
        disc_cut_dips_add,
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

        # Evaluate with the model for predictions
        y_pred_dips_add, y_pred_umami_add = model.predict(
            [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]],
            batch_size=5000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Get rejections for DIPS and UMAMI
        rej_dict_dips_add, disc_cut_dips_add = GetRejection(
            y_pred=y_pred_dips_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict,
            target_eff=target_beff,
        )
        rej_dict_umami_add, disc_cut_umami_add = GetRejection(
            y_pred=y_pred_umami_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict,
            target_eff=target_beff,
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
        "disc_cut_dips": disc_cut_dips,
        "disc_cut_umami": disc_cut_umami,
        "disc_cut_dips_add": disc_cut_dips_add,
        "disc_cut_umami_add": disc_cut_umami_add,
    }

    # Write results in one dict
    result_dict.update(
        {f"{key}_rej_umami": rej_dict_umami[key] for key in rej_dict_umami}
    )
    result_dict.update(
        {f"{key}_rej_dips": rej_dict_dips[key] for key in rej_dict_dips}
    )
    result_dict.upadte(
        {
            f"{key}_rej_umami_add": rej_dict_umami_add[key]
            for key in rej_dict_umami_add
        }
    )
    result_dict.upadte(
        {
            f"{key}_rej_dips_add": rej_dict_dips_add[key]
            for key in rej_dict_dips_add
        }
    )
    return result_dict


def evaluate_model(
    model,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    target_beff: float = 0.77,
    frac_dict: dict = {"cjets": 0.018, "ujets": 0.982},
):
    """
    Evaluate the DIPS/DL1 model on the data provided.

    Input:
    - model: Loaded DIPS/DL1 model for evaluation.
    - data_dict: Dict with the loaded data which are to be evaluated.
    - class_labels: List of classes used for training of the model.
    - main_class: Main class which is to be tagged.
    - target_beff: Working Point which is to be used for evaluation.
    - frac_dict: Dict with the fractions of the non-main classes.
                 Sum needs to be one!

    Output:
    - Dict with validation metrics/rejections.
    """

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

    rej_dict, disc_cut = GetRejection(
        y_pred=y_pred_dips,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
        target_eff=target_beff,
    )

    (
        loss_add,
        accuracy_add,
        rej_dict_add,
        disc_cut_add,
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

        rej_dict_add, disc_cut_add = GetRejection(
            y_pred=y_pred_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict,
            target_eff=target_beff,
        )

    result_dict = {
        "val_loss": loss,
        "val_acc": accuracy,
        "val_loss_add": loss_add,
        "val_acc_add": accuracy_add,
        "disc_cut": disc_cut,
        "disc_cut_add": disc_cut_add,
    }

    # Write results in one dict
    result_dict.update({f"{key}_rej": rej_dict[key] for key in rej_dict})
    result_dict.upadte(
        {f"{key}_rej_add": rej_dict_add[key] for key in rej_dict_add}
    )

    # Return finished dict
    return result_dict


def calc_validation_metrics(
    train_config,
    preprocess_config,
    frac_dict: dict,
    tagger: str,
    target_beff=0.77,
    nJets=int(3e5),
):
    """
    Calculates the validation metrics and rejections for each epoch UMAMI
    and dump it into a json.

    Input:
    - train_config: The loaded train config object
    - preprocess_config: The loaded preprocess config object
    - frac_dict: Dict with the fractions for discriminant calculation
    - target_beff: Target efficiency for main_class
    - nJets: Number of jets used for evaluation

    Output:
    - Json file with validation metrics and rejections for each epoch
    """

    # Get evaluation parameters and NN structure from train config
    Eval_parameters = train_config.Eval_parameters_validation
    NN_structure = train_config.NN_structure

    # Make a list with the model epochs saves
    training_output = [
        os.path.join(train_config.model_name, f)
        for f in os.listdir(train_config.model_name)
        if "model_epoch" in f
    ]

    # Open the json file and load the training out
    with open(
        get_validation_dict_name(
            WP_b=Eval_parameters["WP_b"],
            n_jets=Eval_parameters["n_jets"],
            dir_name=train_config.model_name,
        ),
        "r",
    ) as training_out_json:
        training_output_list = json.load(training_out_json)

    # Init a results list
    results = []

    # Loop over the different model savepoints at each epoch
    for n, model_file in enumerate(training_output):
        logger.info(f"Working on {n+1}/{len(training_output)} input files")

        # Init results dict to save to
        result_dict = {}

        # Get the epoch number from the .h5 file
        epoch = int(
            model_file[
                model_file.rfind("model_epoch")
                + len("model_epoch") : model_file.find(".h5")
            ]
        )

        # Load the epoch from json and add it to dict
        for train_epoch in training_output_list:
            if epoch == train_epoch["epoch"]:
                result_dict = train_epoch

        if tagger == "umami":
            # Load UMAMI model
            umami = load_model(model_file, {"Sum": Sum})

            # Evaluate Umami model
            val_result_dict = evaluate_model_umami(
                model=umami,
                data_dict=load_validation_data_umami(
                    train_config, preprocess_config, nJets
                ),
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=frac_dict,
            )

            # Delete model
            del umami

        elif tagger == "dl1":
            # Load DL1 model
            dl1 = load_model(model_file)

            # Evaluate DL1 model
            val_result_dict = evaluate_model(
                model=dl1,
                data_dict=load_validation_data_umami(
                    train_config, preprocess_config, nJets
                ),
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=frac_dict,
            )

            # Delete model
            del dl1

        elif tagger == "dips":
            # Load DIPS model
            dips = load_model(model_file, {"Sum": Sum})

            # Validate dips
            val_result_dict = evaluate_model(
                model=dips,
                data_dict=load_validation_data_dips(
                    train_config, preprocess_config, nJets
                ),
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=frac_dict,
            )

            # Delete model
            del dips

        # Save results in dict
        for k, v in val_result_dict.items():
            result_dict[k] = v
        results.append(result_dict)

    # Sort the results after epoch
    results = sorted(results, key=lambda x: x["epoch"])

    # Get validation dict name
    output_file_path = get_validation_dict_name(
        target_beff, nJets, train_config.model_name
    )

    # Dump dict into json
    with open(output_file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    # Return Validation dict name
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
