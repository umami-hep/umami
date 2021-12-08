from umami.configuration import global_config, logger  # isort:skip
import copy
import json
import os
import re
from glob import glob
from pathlib import Path
from shutil import copyfile

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.tf_tools as utf
from umami.preprocessing_tools import Configuration as Preprocess_Configuration
from umami.preprocessing_tools import (
    Gen_default_dict,
    GetBinaryLabels,
    GetSampleCuts,
    apply_scaling_trks,
)
from umami.tools import replaceLineInFile, yaml_loader


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_epoch_from_string(string):
    m = re.search("model_epoch(.+?).h5", string)
    return m.group(1)


def get_validation_dict_name(WP, n_jets, dir_name):
    return os.path.join(
        dir_name,
        f"validation_WP{str(WP).replace('.','p')}_{int(n_jets)}jets_Dict.json",
    )


def GetModelPath(model_name, epoch: int) -> str:
    """
    Get the path where the model will be saved/is saved.

    Input:
    - model_name: Name of the model that is to be saved/loaded.
    - epoch: The epoch which is to be saved/loaded

    Output:
    - model_path: Path to the model file of the specified epoch.
    """

    # Get path
    model_path = f"{model_name}/model_files/model_epoch{epoch:03d}.h5"

    # Get logger output for debugging
    logger.debug(f"Treating model {model_path}")

    # Return path
    return model_path


def get_variable_cuts(
    Eval_parameters: dict,
    file: str,
):
    """
    Get the variable cuts from the Eval parameters if there, else return None

    Input:
    - Eval_parameters: Loaded Eval_parameters from the train_config as dict
    - file: Filetype or naming of the cuts you want to load (e.g validation_file)

    Output:
    - cut_vars_dict: Dict with the variables and their cuts.
    """

    if (
        ("variable_cuts" in Eval_parameters)
        and (Eval_parameters["variable_cuts"] is not None)
        and (file in Eval_parameters["variable_cuts"])
    ):
        return Eval_parameters["variable_cuts"][file]

    else:
        return None


def prepare_history_dict(hist_dict: dict):
    """
    Make the history dict from keras the same shape as the one from the
    Callbacks
    """

    # Init a new list
    history_dict_list = []

    # Iterate over the epochs
    for epoch_counter in range(len(hist_dict["loss"])):

        # Init a temporary dict for the epoch
        tmp_dict = {"epoch": epoch_counter}

        # Add the metrics from this epoch to the dict
        for metric in hist_dict:
            tmp_dict[metric] = float(hist_dict[metric][epoch_counter])

        # Append dict to list
        history_dict_list.append(tmp_dict)

    # Return dict
    return history_dict_list


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
    parameters["WP"] = float(sp[1].replace("WP", "").replace("p", "."))
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

    # Create directory for models
    os.makedirs(os.path.join(model_name, "model_files"), exist_ok=True)

    # Get scale dict
    preprocess_config = Preprocess_Configuration(preprocess_config_path)
    scale_dict_path = preprocess_config.dict_file
    preprocess_parameters_path = preprocess_config.ParameterConfigPath

    # Copy files to metadata folder if not existing
    for file_path in [
        train_config_path,
        preprocess_config_path,
        var_dict_path,
        scale_dict_path,
        preprocess_parameters_path,
    ]:
        if (overwrite_config is True) or not os.path.isfile(
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

            elif file_path == preprocess_parameters_path:
                metadata_scale_dict_path = os.path.join(
                    os.getcwd(),
                    model_name,
                    "metadata",
                    os.path.basename(scale_dict_path),
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
                    ".dict_file: &dict_file",
                    f".dict_file: &dict_file {metadata_scale_dict_path}",
                )

                replaceLineInFile(
                    os.path.join(
                        model_name, "metadata", os.path.basename(file_path)
                    ),
                    ".var_file: &var_file",
                    f".var_file: &var_file {metadata_var_dict_path}",
                )


def LoadJetsFromFile(
    filepath: str,
    class_labels: list,
    nJets: int,
    variables: list = None,
    cut_vars_dict: dict = None,
    print_logger: bool = True,
    chunk_size: int = 1e6,
):
    """
    Load jets from file. Only jets from classes in class_labels are returned.

    Input:
    - filepath: Path to the .h5 file with the jets.
    - class_labels: List of class labels which are used.
    - nJets: Number of jets to load.
    - variables: Variables which are loaded.
    - cut_vars_dict: Variable cuts that are applied when loading the jets.
    - print_logger: Decide if the number of jets loaded from the file is printed.

    Output:
    - Jets: The jets as numpy ndarray
    - Umami_labels: The internal class label for each jet. Corresponds with the
                    index of the class label in class_labels.
    """
    # Make sure the nJets argument is an integer
    nJets = int(nJets)
    chunk_size = int(chunk_size)

    # Check if the chunk size is smaller than nJets, if yes change it
    chunk_size = chunk_size if chunk_size >= nJets else nJets

    if type(filepath) is str:
        # Get the paths of the files as a iterable list
        filepaths = glob(filepath)

    elif type(filepath) is list:
        filepaths = filepath

    else:
        raise KeyError(
            f"Given filepath is {type(filepath)}, not a string or a list!"
        )

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {filepath}!
            Check the filepath in your train_config file!
            """
        )

    # Get class_labels variables etc. from global config
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(
        class_labels
    )

    # Load variables for cuts if given
    if variables and cut_vars_dict:
        # Deepcopy the variables list
        variables_list = copy.deepcopy(variables)

        # Add the class label variables to the variables list
        variables_list += class_label_vars

        # Iterate over the cuts and get the needed variables
        for variable in cut_vars_dict:
            variables_list += list(variable.keys())

        # Ensure each variable is only once in the list
        variables_list = list(set(variables_list))

    # Init a counter for the number of loaded jets
    nJets_counter = 0

    for j, file in enumerate(sorted(filepaths, key=natural_keys)):

        # Get the number of available jets in the file
        nJets_infile = len(h5py.File(file, "r")["/jets"])

        # Check how many times we need to iterate over the file
        # to get all jets
        n_chunks = int(np.ceil(nJets_infile / chunk_size))

        # Iterate over the file
        for infile_counter in range(n_chunks):
            if variables:
                jets = pd.DataFrame(
                    h5py.File(file, "r")["/jets"].fields(variables_list)[
                        infile_counter
                        * chunk_size : (infile_counter + 1)
                        * chunk_size
                    ]
                )

            else:
                jets = pd.DataFrame(
                    h5py.File(file, "r")["/jets"][
                        infile_counter
                        * chunk_size : (infile_counter + 1)
                        * chunk_size
                    ]
                )

            # Init new column for string labels
            jets["Umami_string_labels"] = np.zeros_like(
                jets[class_label_vars[0]]
            )
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
                jets["Umami_string_labels"].values[
                    indices_tochange
                ] = class_label

                # Add the right column label to class
                jets["Umami_labels"].values[
                    indices_tochange
                ] = class_labels.index(class_label)

            # Define the conditions to remove
            toremove_conditions = jets["Umami_string_labels"] == "0"

            # Get the indices of the jets that are not used
            indices_toremove = np.where(
                toremove_conditions == True  # noqa: E712
            )[0]

            if cut_vars_dict:
                # Apply cuts and get a list of which jets to remove
                indices_toremove_cuts = GetSampleCuts(
                    jets=jets, cuts=cut_vars_dict
                )

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(
                            indices_toremove.tolist()
                            + indices_toremove_cuts.tolist()
                        )
                    )
                )

            # Remove all unused jets
            jets = jets.drop(indices_toremove)

            # If not the first file processed, append to the global one
            if j == 0 and infile_counter == 0:
                all_jets = jets
                all_labels = jets["Umami_labels"].values

            # if the first file processed, set as global one
            else:
                all_jets = all_jets.append(jets, ignore_index=True)
                all_labels = np.append(all_labels, jets["Umami_labels"].values)

            # Adding the loaded jets to counter
            nJets_counter += len(jets)

            # Break the loop inside the file if enough jets are loaded
            if nJets_counter >= nJets:
                break

        # Break the loop over the files if enough jets are loaded
        if nJets_counter >= nJets:
            break

    if print_logger:
        # Check if enough jets are loaded
        if nJets_counter < nJets:
            logger.warning(
                f"Requested {nJets} but only {nJets_counter} could be loaded!"
            )

        else:
            logger.info(f"Loaded {nJets} jets!")

    # Return the jets and internal labels
    return all_jets[:nJets], all_labels[:nJets]


def LoadTrksFromFile(
    filepath: str,
    class_labels: list,
    nJets: int,
    cut_vars_dict: dict = None,
    print_logger: bool = True,
    chunk_size: int = 1e6,
):
    """
    Load tracks from file. Only jets from classes in class_labels are returned.

    Input:
    - filepath: Path to the .h5 file with the jets.
    - class_labels: List of class labels which are used.
    - nJets: Number of jets to load.
    - cut_vars_dict: Variable cuts that are applied when loading the jets.
    - print_logger: Decide if the number of jets loaded from the file is printed.

    Output:
    - Trks: The tracks of the jets as numpy ndarray
    - Umami_labels: The internal class label for each jet. Corresponds with the
                    index of the class label in class_labels.
    """
    # Make sure the nJets argument is an integer
    nJets = int(nJets)
    chunk_size = int(chunk_size)

    # Check if the chunk size is small than nJets, if yes change it
    chunk_size = chunk_size if chunk_size >= nJets else nJets

    if type(filepath) is str:
        # Get the paths of the files as a iterable list
        filepaths = glob(filepath)

    elif type(filepath) is list:
        filepaths = filepath

    else:
        raise KeyError(
            f"Given filepath is {type(filepath)}, not a string or a list!"
        )

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {filepath}!
            Check the filepath in your train_config file!
            """
        )

    # Get class_labels variables etc. from global config
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(
        class_labels
    )

    # Define the labels which are needed
    jet_vars_to_load = list(dict.fromkeys(class_label_vars))

    # Load variables for cuts if given
    if cut_vars_dict:

        # Iterate over the cuts and get the needed variables
        for variable in cut_vars_dict:
            jet_vars_to_load += list(variable.keys())

        # Ensure each variable is only once in the list
        jet_vars_to_load = list(set(jet_vars_to_load))

    # Init a counter for the number of loaded jets
    nJets_counter = 0

    # Iterate over the files
    for j, file in enumerate(sorted(filepaths, key=natural_keys)):

        # Get the number of available jets in the file
        nJets_infile = len(h5py.File(file, "r")["/jets"])

        # Check how many times we need to iterate over the file
        # to get all jets
        n_chunks = int(np.ceil(nJets_infile / chunk_size))

        # Iterate over the file
        for infile_counter in range(n_chunks):

            # Load the used label variables from file
            with h5py.File(file, "r") as jets:
                for iterator, iter_class_var in enumerate(jet_vars_to_load):
                    if iterator == 0:
                        labels = pd.DataFrame(
                            jets["/jets"][iter_class_var][
                                infile_counter
                                * chunk_size : (infile_counter + 1)
                                * chunk_size
                            ],
                            columns=[iter_class_var],
                        )

                    else:
                        labels[iter_class_var] = jets["/jets"][iter_class_var][
                            infile_counter
                            * chunk_size : (infile_counter + 1)
                            * chunk_size
                        ]

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

            # Define the conditions to remove
            toremove_conditions = labels["Umami_string_labels"] == "0"

            # Get the indices of the jets that are not used
            indices_toremove = np.where(
                toremove_conditions == True  # noqa: E712
            )[0]

            if cut_vars_dict:
                # Apply cuts and get a list of which jets to remove
                indices_toremove_cuts = GetSampleCuts(
                    jets=labels, cuts=cut_vars_dict
                )

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(
                            indices_toremove.tolist()
                            + indices_toremove_cuts.tolist()
                        )
                    )
                )

            # Remove unused jets from labels
            labels = labels.drop(indices_toremove)
            Umami_labels = labels["Umami_labels"].values

            # Load tracks and delete unused classes
            trks = np.delete(
                arr=np.asarray(
                    h5py.File(file, "r")["/tracks"][
                        infile_counter
                        * chunk_size : (infile_counter + 1)
                        * chunk_size
                    ]
                ),
                obj=indices_toremove,
                axis=0,
            )

            # If not the first file processed, append to the global one
            if j == 0 and infile_counter == 0:
                all_trks = trks
                all_labels = Umami_labels

            # if the first file processed, set as global one
            else:
                all_trks = np.append(all_trks, trks, axis=0)
                all_labels = np.append(all_labels, Umami_labels)

            # Adding the loaded jets to counter
            nJets_counter += len(trks)

            # Break the loop inside the file if enough jets are loaded
            if nJets_counter >= nJets:
                break

        # Break the loop over the files if enough jets are loaded
        if nJets_counter >= nJets:
            break

    if print_logger:
        # Check if enough jets are loaded
        if nJets_counter < nJets:
            logger.warning(
                f"Requested {nJets} but only {nJets_counter} could be loaded!"
            )

        else:
            logger.info(f"Loaded {nJets} jets!")

    # Return Trks and labels
    return all_trks[:nJets], all_labels[:nJets]


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


def GetScore(
    y_pred,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
):
    """
    Calculates the scores for the provided jets.

    Input:
    - y_pred: The prediction output of the NN
    - class_labels: A list of the class_labels which are used
    - main_class: The main discriminant class. For b-tagging obviously "bjets"
    - frac_dict: A dict with the respective fractions for each class provided
                 except main_class

    Output:
    - Discriminant Score for the jets provided.
    """

    # Init index dict
    index_dict = {}

    # Get Index of main class
    for class_label in class_labels:
        index_dict[f"{class_label}"] = class_labels.index(class_label)

    # Init denominator of disc_score and add_small
    denominator = 0
    add_small = 1e-10

    # Get class_labels list without main class
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)

    # Calculate counter of disc_score
    counter = y_pred[:, index_dict[main_class]] + add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += (
            frac_dict[class_label] * y_pred[:, index_dict[class_label]]
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
    y_true = np.argmax(y_true, axis=1) if len(y_true.shape) == 2 else y_true

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
        rej_class=None,
    )

    # Calculate cutvalue on the discriminant depending of the WP
    cutvalue = np.percentile(disc_scores, 100.0 * (1.0 - target_eff))

    # Get all non-main flavours
    class_labels_wo_main = copy.deepcopy(class_labels)
    class_labels_wo_main.remove(main_class)

    # Calculate efficiencies
    for iter_main_class in class_labels_wo_main:
        try:
            rej_dict[f"{iter_main_class}_rej"] = 1 / (
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

        except ZeroDivisionError:
            logger.error(
                f"""
                Not enough jets for rejection calculation of class {iter_main_class} for {target_eff} efficiency!\n
                Maybe loosen the eff_min to fix it or give more jets!
                """
            )
            raise ZeroDivisionError(
                "Not enough jets for rejection calculation!"
            )

    return rej_dict, cutvalue


class CallbackBase(Callback):
    """Base class for the callbacks of the different models.

    This class provides the base functionalites for the different
    callbacks needed for the models that are available.
    """

    def __init__(
        self,
        class_labels: list,
        main_class: str,
        val_data_dict: dict = None,
        model_name: str = "test",
        target_beff: float = 0.77,
        frac_dict: dict = {
            "cjets": 0.018,
            "ujets": 0.982,
        },
        dict_file_name: str = "DictFile.json",
    ):
        """Init the parameters needed for the callback

        Parameters
        ----------
        class_labels : list
            List of class labels used in training (ORDER MATTERS!).
        main_class : str
            Name of the main class which is used. For b-tagging
            obviously `bjets`.
        val_data_dict : dict
            Dict with the loaded validation data. These are loaded
            using the `load_validation_data_*` functions.
        model_name : str
            Name of the model used to evaluate. This is important
            for the path where the results are of the callback are saved.
        target_beff : float
            Float value between 0 and 1 for which main class efficiency
            the rejections are calculated.
        frac_dict : dict
            Dict with the fraction values for the non-main classes. The
            values need to add up to 1.
        dict_file_name : str
            Name of the file where the dict with the results of the callback
            are saved.
        """

        # Add parameters to as attributes
        self.class_labels = class_labels
        self.main_class = main_class
        self.val_data_dict = val_data_dict
        self.target_beff = target_beff
        self.frac_dict = frac_dict
        self.model_name = model_name
        self.dict_file_name = dict_file_name

        # Init a list for the result dicts for each epoch
        self.dict_list = []

        # Init the directory and clean it from previous training
        setup_output_directory(self.model_name)


class MyCallback(CallbackBase):
    """Callback class for the standard taggers

    This class is the callback for the standard taggers. Only one
    output (not like the umami tagger) is given.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Get the needed metrics at epoch end and calculate rest.

        This method saves the training metrics at the end of the
        epoch and also calculates the validation metrics and
        the rejections for each non-main class for given
        efficiency and fraction values. Those are also saved.

        Parameters
        ----------
        epoch : int
            Number of the epoch which just finished and is now
            evaluated and saved.
        logs : dict
            Dict with the training metrics of the just finished
            epoch.
        """

        # Define a dict with the epoch and the training metrics
        dict_epoch = {
            "epoch": epoch,
            "learning_rate": logs["lr"].item(),
            "loss": logs["loss"],
            "accuracy": logs["accuracy"],
        }

        # If val data is given, calculate validaton metrics and rejections
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

        # Append the dict to the list
        self.dict_list.append(dict_epoch)

        # Dump the list in json file
        with open(self.dict_file_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)


class MyCallbackUmami(CallbackBase):
    """Callback class for the umami tagger

    This class is the callback for the umami tagger. Due to the
    two outputs of the tagger, we need special metrics etc.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Get the needed metrics at epoch end and calculate rest.

        This method saves the training metrics at the end of the
        epoch and also calculates the validation metrics and
        the rejections for each non-main class for given
        efficiency and fraction values. Those are also saved.

        Parameters
        ----------
        epoch : int
            Number of the epoch which just finished and is now
            evaluated and saved.
        logs : dict
            Dict with the training metrics of the just finished
            epoch.
        """

        # Define a dict with the epoch and the training metrics
        dict_epoch = {
            "epoch": epoch,
            "learning_rate": logs["lr"].item(),
            "loss": logs["loss"],
            "dips_loss": logs["dips_loss"],
            "umami_loss": logs["umami_loss"],
            "dips_accuracy": logs["dips_accuracy"],
            "umami_accuracy": logs["umami_accuracy"],
        }

        # If val data is given, calculate validaton metrics and rejections
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

        # Append the dict to the list
        self.dict_list.append(dict_epoch)

        # Dump the list in json file
        with open(self.dict_file_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)


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
            logger.warning(f"Variables to exclude not found: {exclude_that}")
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
    input_file: str,
    var_dict: str,
    preprocess_config,
    class_labels: list,
    nJets: int = int(3e5),
    exclude: list = None,
    cut_vars_dict: dict = None,
    jet_variables: list = None,
    print_logger: bool = True,
):
    """
    Load the jet variables from the validation/evaluation files and apply scaling/shifting.

    Input:
    - input_file: Path to the file which is to be loaded.
    - var_dict: Variable dict with the wanted jet variables inside.
    - preprocess_config: Loaded preprocessing config that was used.
    - class_labels: List of classes used for training of the model.
    - nJets: Number of jets that should be loaded.
    - exclude: List of variables that are not loaded.
    - cut_vars_dict: Dict with the cuts that should be applied.
    - jet_variables: List of variables that are used.
    - print_logger: Decide, if the logger info is printed or not.

    Output:
    - all_jets: X values of the jets for the NN's.
    - all_labels: Y values ready to be used in the NN's.
    """

    # Assert that the jet variables and exlude are not called at the same time
    if jet_variables and exclude:
        raise ValueError(
            "You can't set exclude and jet_variables. Choose one!"
        )

    # Adding class_labels check between preprocess_config and given labels
    assert (
        preprocess_config.preparation["class_labels"] == class_labels
    ), "class_labels from preprocessing_config and from train_config are different! They need to be the same!"

    # Get the paths of the input file as list
    # In case there are multiple files (Wildcard etc.)
    filepaths = glob(input_file)

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {input_file}!
            Check the filepath in your train_config file!
            """
        )

    # Load variables
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Load scale dict
    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["jets"]

    jets, Umami_labels = LoadJetsFromFile(
        filepath=filepaths,
        class_labels=class_labels,
        nJets=nJets,
        cut_vars_dict=cut_vars_dict,
        variables=jet_variables,
        print_logger=False,
    )

    # Binarize Labels
    labels = GetBinaryLabels(Umami_labels)

    # Check if jet_variables is defined
    if jet_variables:
        # Retrieve the defined variables
        variables = jet_variables
        excluded_variables = []

    else:
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
            if print_logger:
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

    # Return jets and labels
    return jets, labels


def GetTestSampleTrks(
    input_file: str,
    var_dict: str,
    preprocess_config,
    class_labels: list,
    nJets: int = int(3e5),
    cut_vars_dict: dict = None,
    print_logger: bool = True,
):
    """
    Apply the scaling and shifting to dataset using numpy
    """

    # Adding class_labels check between preprocess_config and given labels
    assert (
        preprocess_config.preparation["class_labels"] == class_labels
    ), "class_labels from preprocessing_config and from train_config are different! They need to be the same!"

    # making sure the nJets aregument is an integer
    nJets = int(nJets)
    # Get the paths of the input file as list
    # In case there are multiple files (Wildcard etc.)
    filepaths = glob(input_file)

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {input_file}!
            Check the filepath in your train_config file!
            """
        )

    # Load variables
    with open(var_dict, "r") as conf:
        variable_config = yaml.load(conf, Loader=yaml_loader)

    # Load scale dict for the tracks
    with open(preprocess_config.dict_file, "r") as infile:
        scale_dict = json.load(infile)["tracks"]

    trks, labels = LoadTrksFromFile(
        filepath=filepaths,
        class_labels=class_labels,
        nJets=nJets,
        cut_vars_dict=cut_vars_dict,
        print_logger=False,
    )

    # Binarize the labels
    binary_labels = GetBinaryLabels(labels)

    # Apply scaling to the tracks
    trks, _ = apply_scaling_trks(
        trks=trks,
        variable_config=variable_config,
        scale_dict=scale_dict,
    )

    return trks, binary_labels


def load_validation_data_umami(
    train_config,
    preprocess_config,
    nJets: int,
    jets_var_list: list = [],
    convert_to_tensor: bool = False,
):
    """
    Load the validation data for UMAMI.

    Input:
    - train_config: Loaded train_config object.
    - train_config: Loaded preprocess_config object.
    - nJets: Number of jets to load.

    Output:
    - val_data_dict: Dict with the validation data
    """

    # Define NN_Structure and the Eval params
    NN_structure = train_config.NN_structure
    Eval_parameters = train_config.Eval_parameters_validation

    # Get the cut vars dict if defined
    cut_vars_dict = get_variable_cuts(
        Eval_parameters=Eval_parameters,
        file="validation_file",
    )

    # Check for excluded variables
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Init a new dict for the loaded val data
    val_data_dict = {}
    (X_valid, X_valid_trk, Y_valid,) = GetTestFile(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=NN_structure["class_labels"],
        nJets=nJets,
        exclude=exclude,
        jet_variables=jets_var_list,
        cut_vars_dict=cut_vars_dict,
    )

    if convert_to_tensor:
        # Transform to tf.tensors and add to val_dict
        val_data_dict["X_valid"] = tf.convert_to_tensor(
            X_valid, dtype=tf.float64
        )
        val_data_dict["X_valid_trk"] = tf.convert_to_tensor(
            X_valid_trk, dtype=tf.float64
        )
        val_data_dict["Y_valid"] = tf.convert_to_tensor(
            Y_valid, dtype=tf.int64
        )

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["X_valid_trk"] = X_valid_trk
        val_data_dict["Y_valid"] = Y_valid

    # Set placeholder Nones for the add_files
    (
        X_valid_add,
        X_valid_trk_add,
        Y_valid_add,
    ) = (None, None, None)

    # Check if add_files are defined and load them
    if train_config.add_validation_file is not None:
        # Get cut vars dict for add_validation file
        cut_vars_dict_add = get_variable_cuts(
            Eval_parameters=Eval_parameters,
            file="add_validation_file",
        )

        (X_valid_add, X_valid_trk_add, Y_valid_add,) = GetTestFile(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=NN_structure["class_labels"],
            nJets=nJets,
            exclude=exclude,
            jet_variables=jets_var_list,
            cut_vars_dict=cut_vars_dict_add,
        )

        if len(jets_var_list) != 0:
            X_valid_add = X_valid_add[jets_var_list]

        if convert_to_tensor:
            # Transform to tf.tensors and add to val_dict
            val_data_dict["X_valid_add"] = tf.convert_to_tensor(
                X_valid_add, dtype=tf.float64
            )
            val_data_dict["X_valid_trk_add"] = tf.convert_to_tensor(
                X_valid_trk_add, dtype=tf.float64
            )
            val_data_dict["Y_valid_add"] = tf.convert_to_tensor(
                Y_valid_add, dtype=tf.int64
            )

        else:
            val_data_dict["X_valid_add"] = X_valid_add
            val_data_dict["X_valid_trk_add"] = X_valid_trk_add
            val_data_dict["Y_valid_add"] = Y_valid_add

        # Assert a correct shape
        assert (
            val_data_dict["X_valid"].shape[1]
            == val_data_dict["X_valid_add"].shape[1]
        ), "validation_file and add_validation_file have different amounts of variables!"

    # Return the val data dict
    return val_data_dict


def load_validation_data_dl1(
    train_config,
    preprocess_config,
    nJets: int,
    convert_to_tensor: bool = False,
):
    """
    Load the validation data for DL1.

    Input:
    - train_config: Loaded train_config object.
    - train_config: Loaded preprocess_config object.
    - nJets: Number of jets to load.

    Output:
    - val_data_dict: Dict with the validation data
    """

    # Define NN_Structure and the Eval params
    NN_structure = train_config.NN_structure
    Eval_parameters = train_config.Eval_parameters_validation

    # Get cut vars dict for add_validation file
    cut_vars_dict = get_variable_cuts(
        Eval_parameters=Eval_parameters,
        file="validation_file",
    )

    # Ensure the nJets is an int
    nJets = int(nJets)

    # Check for excluded variables
    exclude = None
    if "exclude" in train_config.config:
        exclude = train_config.config["exclude"]

    # Init a new dict for the loaded val data
    val_data_dict = {}

    # Load the validation data
    (X_valid, Y_valid,) = GetTestSample(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=NN_structure["class_labels"],
        nJets=nJets,
        exclude=exclude,
        cut_vars_dict=cut_vars_dict,
    )

    if convert_to_tensor:
        # Transform to tf.tensors and add to val_dict
        val_data_dict["X_valid"] = tf.convert_to_tensor(
            X_valid, dtype=tf.float64
        )
        val_data_dict["Y_valid"] = tf.convert_to_tensor(
            Y_valid, dtype=tf.int64
        )

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["Y_valid"] = Y_valid

    # Set placeholder Nones for the add_files
    (
        X_valid_add,
        Y_valid_add,
    ) = (None, None)

    # Check if add_files are defined and load them
    if train_config.add_validation_file is not None:

        # Get cut vars dict for add_validation file
        cut_vars_dict_add = get_variable_cuts(
            Eval_parameters=Eval_parameters,
            file="add_validation_file",
        )

        (X_valid_add, Y_valid_add,) = GetTestSample(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=NN_structure["class_labels"],
            nJets=nJets,
            exclude=exclude,
            cut_vars_dict=cut_vars_dict_add,
        )

        if convert_to_tensor:
            # Transform to tf.tensors and add to val_dict
            val_data_dict["X_valid_add"] = tf.convert_to_tensor(
                X_valid_add, dtype=tf.float64
            )
            val_data_dict["Y_valid_add"] = tf.convert_to_tensor(
                Y_valid_add, dtype=tf.int64
            )

        else:
            val_data_dict["X_valid_add"] = X_valid_add
            val_data_dict["Y_valid_add"] = Y_valid_add

        # Assert a correct shape
        assert (
            val_data_dict["X_valid"].shape[1]
            == val_data_dict["X_valid_add"].shape[1]
        ), "validation_file and add_validation_file have different amounts of variables!"

    # Return the val data dict
    return val_data_dict


def load_validation_data_dips(
    train_config,
    preprocess_config,
    nJets: int,
    convert_to_tensor: bool = False,
):
    """
    Load the validation data for DIPS.

    Input:
    - train_config: Loaded train_config object.
    - train_config: Loaded preprocess_config object.
    - nJets: Number of jets to load.

    Output:
    - val_data_dict: Dict with the validation data
    """

    # Define NN_Structure and the Eval params
    NN_structure = train_config.NN_structure
    Eval_parameters = train_config.Eval_parameters_validation

    # Get cut vars dict for add_validation file
    cut_vars_dict = get_variable_cuts(
        Eval_parameters=Eval_parameters,
        file="validation_file",
    )

    val_data_dict = {}
    (X_valid, Y_valid,) = GetTestSampleTrks(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config,
        class_labels=NN_structure["class_labels"],
        nJets=nJets,
        cut_vars_dict=cut_vars_dict,
    )

    if convert_to_tensor:
        # Transform to tf.tensors and add to val_dict
        val_data_dict["X_valid"] = tf.convert_to_tensor(
            X_valid, dtype=tf.float64
        )
        val_data_dict["Y_valid"] = tf.convert_to_tensor(
            Y_valid, dtype=tf.int64
        )

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["Y_valid"] = Y_valid

    # Set placeholder Nones for the add_files
    (
        X_valid_add,
        Y_valid_add,
    ) = (None, None)

    # Check if add_files are defined and load them
    if train_config.add_validation_file is not None:

        # Get cut vars dict for add_validation file
        cut_vars_dict_add = get_variable_cuts(
            Eval_parameters=Eval_parameters,
            file="add_validation_file",
        )

        (X_valid_add, Y_valid_add,) = GetTestSampleTrks(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config,
            class_labels=NN_structure["class_labels"],
            nJets=nJets,
            cut_vars_dict=cut_vars_dict_add,
        )

        if convert_to_tensor:
            # Transform to tf.tensors and add to val_dict
            val_data_dict["X_valid_add"] = tf.convert_to_tensor(
                X_valid_add, dtype=tf.float64
            )
            val_data_dict["Y_valid_add"] = tf.convert_to_tensor(
                Y_valid_add, dtype=tf.int64
            )

        else:
            val_data_dict["X_valid_add"] = X_valid_add
            val_data_dict["Y_valid_add"] = Y_valid_add

        # Assert a correct shape
        assert (
            val_data_dict["X_valid"].shape[1]
            == val_data_dict["X_valid_add"].shape[1]
        ), "validation_file and add_validation_file have different amounts of variables!"

    # Return the val data dict
    return val_data_dict


def GetTestFile(
    input_file: str,
    var_dict: str,
    preprocess_config: dict,
    class_labels: list,
    nJets: int,
    exclude: list = None,
    cut_vars_dict: dict = None,
    jet_variables: list = None,
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
    - cut_vars_dict: Dict with the variable cuts which should be applied.
    - jet_variables: Jet variables that are to be loaded.

    Output:
    - Returns the X, X_trk and Y for training/evaluation.
    """

    X_trk, Y_trk = GetTestSampleTrks(
        input_file=input_file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=int(nJets),
        cut_vars_dict=cut_vars_dict,
        print_logger=False,
    )

    X, Y = GetTestSample(
        input_file=input_file,
        var_dict=var_dict,
        preprocess_config=preprocess_config,
        class_labels=class_labels,
        nJets=int(nJets),
        exclude=exclude,
        cut_vars_dict=cut_vars_dict,
        jet_variables=jet_variables,
        print_logger=True,
    )

    assert np.equal(Y, Y_trk).all()

    return X, X_trk, Y


def evaluate_model_umami(
    model,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    target_beff: float = 0.77,
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
        batch_size=5_000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    # Evaluate with the model for predictions
    y_pred_dips, y_pred_umami = model.predict(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        batch_size=5_000,
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
        frac_dict=frac_dict["dips"],
        target_eff=target_beff,
    )
    rej_dict_umami, disc_cut_umami = GetRejection(
        y_pred=y_pred_umami,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict["umami"],
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
            batch_size=5_000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Evaluate with the model for predictions
        y_pred_dips_add, y_pred_umami_add = model.predict(
            [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]],
            batch_size=5_000,
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
            frac_dict=frac_dict["dips"],
            target_eff=target_beff,
        )
        rej_dict_umami_add, disc_cut_umami_add = GetRejection(
            y_pred=y_pred_umami_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict["umami"],
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
        {f"{key}_umami": rej_dict_umami[key] for key in rej_dict_umami}
    )
    result_dict.update(
        {f"{key}_dips": rej_dict_dips[key] for key in rej_dict_dips}
    )
    result_dict.update(
        {
            f"{key}_umami_add": rej_dict_umami_add[key]
            for key in rej_dict_umami_add
        }
    )
    result_dict.update(
        {
            f"{key}_dips_add": rej_dict_dips_add[key]
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

    # Check which input data need to be used
    if "X_valid_trk" in data_dict and "X_valid" in data_dict:
        x = [data_dict["X_valid_trk"], data_dict["X_valid"]]

    elif "X_valid_trk" in data_dict and "X_valid" not in data_dict:
        x = data_dict["X_valid_trk"]

    else:
        x = data_dict["X_valid"]

    loss, accuracy = model.evaluate(
        x=x,
        y=data_dict["Y_valid"],
        batch_size=15_000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    y_pred_dips = model.predict(
        x=x,
        batch_size=15_000,
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
        if "X_valid_trk_add" in data_dict and "X_valid_add" in data_dict:
            x_add = [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]]

        elif "X_valid_trk_add" in data_dict and "X_valid_add" not in data_dict:
            x_add = data_dict["X_valid_trk_add"]

        else:
            x_add = data_dict["X_valid_add"]

        loss_add, accuracy_add = model.evaluate(
            x=x_add,
            y=data_dict["Y_valid_add"],
            batch_size=15_000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        y_pred_add = model.predict(
            x=x_add,
            batch_size=15_000,
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
    result_dict.update({f"{key}": rej_dict[key] for key in rej_dict})
    result_dict.update(
        {f"{key}_add": rej_dict_add[key] for key in rej_dict_add}
    )

    # Return finished dict
    return result_dict


def calc_validation_metrics(
    train_config,
    preprocess_config,
    tagger: str,
    target_beff: float = 0.77,
    nJets: int = int(3e5),
    model_string: str = "model_epoch",
):
    """
    Calculates the validation metrics and rejections for each epoch
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
    Second_model_string = (
        "dips_model_" if model_string == "model_epoch" else "model_epoch"
    )

    # Make a list with the model epochs saves
    training_output = [
        os.path.join(f"{train_config.model_name}/model_files/", f)
        for f in os.listdir(f"{train_config.model_name}/model_files/")
        if model_string in f
    ]

    if len(training_output) == 0:
        logger.warning(
            f"{model_string} models used but not found! Using {Second_model_string}"
        )

        # Set new model string
        model_string = Second_model_string

        # Make a list with the model epochs saves with second model name string
        training_output = [
            os.path.join(f"{train_config.model_name}/model_files/", f)
            for f in os.listdir(f"{train_config.model_name}/model_files/")
            if model_string in f
        ]

    # Open the json file and load the training out
    try:
        with open(
            get_validation_dict_name(
                WP=Eval_parameters["WP"],
                n_jets=Eval_parameters["n_jets"],
                dir_name=train_config.model_name,
            ),
            "r",
        ) as training_out_json:
            training_output_list = json.load(training_out_json)

    except FileNotFoundError:
        logger.info(
            "No callback json file with validation metrics found! Make new one"
        )
        training_output_list = [
            {"epoch": n} for n in range(train_config.NN_structure["epochs"])
        ]

    # Init a results list
    results = []

    # Check tagger and load the correct val data
    if tagger == "umami":
        data_dict = load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=nJets,
            convert_to_tensor=False,
        )

    elif tagger == "dl1":
        data_dict = load_validation_data_dl1(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=nJets,
            convert_to_tensor=False,
        )

    elif tagger == "dips":
        data_dict = load_validation_data_dips(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=nJets,
            convert_to_tensor=False,
        )

    elif tagger == "dips_cond_att":
        data_dict = load_validation_data_umami(
            train_config=train_config,
            preprocess_config=preprocess_config,
            nJets=nJets,
            jets_var_list=[
                global_config.etavariable,
                global_config.pTvariable,
            ],
            convert_to_tensor=False,
        )

    else:
        raise ValueError(f"Tagger {tagger} is not supported!")

    # Loop over the different model savepoints at each epoch
    for n, model_file in enumerate(sorted(training_output, key=natural_keys)):
        logger.info(f"Working on {n+1}/{len(training_output)} input files")

        # Init results dict to save to
        result_dict = {}

        # Get the epoch number from the .h5 file
        try:
            epoch = int(
                model_file[
                    model_file.rfind(f"{model_string}")
                    + len(f"{model_string}") : model_file.find(".h5")
                ]
            )

        except ValueError:
            raise ValueError(
                f"Epoch could not be extracted from {model_string}!"
            )

        # Load the epoch from json and add it to dict
        for train_epoch in training_output_list:
            if epoch == train_epoch["epoch"]:
                result_dict = train_epoch

        # Ensure the epoch is in the dict
        result_dict["epoch"] = epoch

        if tagger == "umami":
            # Load UMAMI model
            umami = load_model(model_file, {"Sum": utf.Sum})

            # Evaluate Umami model
            val_result_dict = evaluate_model_umami(
                model=umami,
                data_dict=data_dict,
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=Eval_parameters["frac_values"],
            )

            # Delete model
            del umami

        elif tagger == "dl1":
            # Load DL1 model
            dl1 = load_model(model_file)

            # Evaluate DL1 model
            val_result_dict = evaluate_model(
                model=dl1,
                data_dict=data_dict,
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=Eval_parameters["frac_values"],
            )

            # Delete model
            del dl1

        elif tagger == "dips":
            # Load DIPS model
            with CustomObjectScope({"Sum": utf.Sum}):
                dips = load_model(model_file)

            # Validate dips
            val_result_dict = evaluate_model(
                model=dips,
                data_dict=data_dict,
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=Eval_parameters["frac_values"],
            )

            # Delete model
            del dips

        elif tagger == "dips_cond_att":
            # Load DIPS Conditional Attention model
            with CustomObjectScope(
                {
                    "Sum": utf.Sum,
                    "Attention": utf.Attention,
                    "DeepSet": utf.DeepSet,
                    "AttentionPooling": utf.AttentionPooling,
                    "DenseNet": utf.DenseNet,
                    "ConditionalAttention": utf.ConditionalAttention,
                    "ConditionalDeepSet": utf.ConditionalDeepSet,
                }
            ):
                dips_cond_add = load_model(model_file)

            # Validate dips
            val_result_dict = evaluate_model(
                model=dips_cond_add,
                data_dict=data_dict,
                class_labels=NN_structure["class_labels"],
                main_class=NN_structure["main_class"],
                target_beff=target_beff,
                frac_dict=Eval_parameters["frac_values"],
            )

            # Delete model
            del dips_cond_add

        else:
            raise ValueError(f"Tagger {tagger} is not supported!")

        # Save results in dict
        for k, v in val_result_dict.items():
            result_dict[k] = v

        # Append results dict to list
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
