"""Helper functions for training tools."""
from umami.configuration import global_config, logger  # isort:skip
import json
import os
import re
from glob import glob
from pathlib import Path
from shutil import copyfile

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import umami.metrics as umt
import umami.tf_tools as utf
from umami.data_tools import LoadJetsFromFile, LoadTrksFromFile
from umami.preprocessing_tools import Configuration as Preprocess_Configuration
from umami.preprocessing_tools import (
    Gen_default_dict,
    GetBinaryLabels,
    apply_scaling_trks,
)
from umami.tools import natural_keys, replaceLineInFile, yaml_loader


def get_unique_identifiers(keys: list, prefix: str) -> list:
    """Helper function which takes a list of strings, searches them for a given prefix
    of the form "prefix_<identifier>" and returns the remaining part of the matching
    strings

    Parameters
    ----------
    keys : list
        List of strings which are searched for the given prefix
    prefix : str
        Prefix to search for in the provided strings

    Returns
    -------
    identifiers : list
        Sorted list of the unique identifiers that could be found.
    """

    identifiers = list(
        {key.replace(prefix + "_", "") for key in keys if key.startswith(prefix)}
    )

    return sorted(identifiers)


def get_epoch_from_string(string):
    """
    Get the epoch from the model file string.

    Parameters
    ----------
    string : str
        Path of the model file.

    Returns
    -------
    epoch : int
        Epoch of the model file.
    """

    m = re.search("model_epoch(.+?).h5", string)
    return m.group(1)


def get_validation_dict_name(WP: float, n_jets: int, dir_name: str) -> str:
    """
    Get the validation dict name based on WP, number of jets and dir_name.

    Parameters
    ----------
    WP : float
        Working point that was used to calculate validation dict.
    n_jets : int
        Number of jets that was used to calculate validation dict.
    dir_name : str
        Path of the directory where the validation dict is saved.

    Returns
    -------
    validation_dict_path : str
        Path of the validation dict.
    """

    # Get the path of the validation dict
    validation_dict_path = os.path.join(
        dir_name,
        f"validation_WP{str(WP).replace('.','p')}_{int(n_jets)}jets_Dict.json",
    )

    return validation_dict_path


def GetModelPath(model_name: str, epoch: int) -> str:
    """
    Get the path where the model will be saved/is saved.

    Parameters
    ----------
    model_name : str
        Name of the model that is to be saved/loaded.
    epoch : int
        The epoch which is to be saved/loaded

    Returns
    -------
    model_path : str
        Path to the model file of the specified epoch.
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
) -> dict:
    """
    Get the variable cuts from the Eval parameters if there, else return None

    Parameters
    ----------
    Eval_parameters : dict
        Loaded Eval_parameters from the train_config as dict
    file : str
        Filetype or naming of the cuts you want to load (e.g validation_file)

    Returns
    -------
    cut_vars_dict : dict
        Dict with the variables and their cuts.
    """

    if (
        ("variable_cuts" in Eval_parameters)
        and (Eval_parameters["variable_cuts"] is not None)
        and (file in Eval_parameters["variable_cuts"])
    ):
        return Eval_parameters["variable_cuts"][file]

    return None


def prepare_history_dict(hist_dict: dict) -> list:
    """
    Make the history dict from keras the same shape as the one from the callbacks.

    Parameters
    ----------
    hist_dict : dict
        Dict with the history inside.

    Returns
    -------
    history_dict_list : list
        Reshaped history dict as list. Same shape as the one from the callbacks
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


def get_parameters_from_validation_dict_name(dict_name: str) -> dict:
    """
    Get the parameters used to calculate the validation dict from the
    validation dict name.

    Parameters
    ----------
    dict_name : str
        Name of the validation dict.

    Returns
    -------
    parameters : dict
        Dict with the parameters (WP, n_jets, dir_name) used to calculate
        the validation dict.

    Raises
    ------
    Exception
        If the name of the validation dict could be rebuild from the
        extracted parameters.
    """

    # Split the path and only get the dict name
    sp = dict_name.split("/")[-1].split("_")

    # Init a new dict for the parameters
    parameters = {}

    # Get the parameters from the name and add them to the dict
    parameters["WP"] = float(sp[1].replace("WP", "").replace("p", "."))
    parameters["n_jets"] = int(sp[2].replace("jets", ""))
    parameters["dir_name"] = str(Path(dict_name).parent)

    # Check if the values are correct extracted. Try to build the name
    # from the parameters and check if they are identical.
    if get_validation_dict_name(**parameters) != dict_name:
        raise Exception(
            f"Can't infer parameters correctly for {dict_name}. Parameters:"
            f" {parameters}"
        )

    # Return the parameters
    return parameters


def setup_output_directory(dir_name: str) -> None:
    """
    Check the output directory path and init/clean it.

    Parameters
    ----------
    dir_name : str
        Path of the output directory.

    Raises
    ------
    Exception
        If the dir_name is an existing file.
    """

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
    train_config_path: str,
    var_dict_path: str,
    model_name: str,
    preprocess_config_path: str,
    overwrite_config: bool = False,
) -> None:
    """
    Create a metadata folder in the new model_name dir and
    copy all configs there and change the paths inside the
    configs to the new metadata directory path.

    Parameters
    ----------
    train_config_path : str
        Path to the train config that is used.
    var_dict_path : str
        Path to the variable dict that is used.
    model_name : str
        Model name that is used.
    preprocess_config_path : str
        Path to the preprocessing config that is used.
    overwrite_config : bool
        If configs already in metadata folder, overwrite
        them or not.
    """

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
                os.path.join(model_name, "metadata", os.path.basename(file_path)),
            )

            # Change the paths for the preprocess config and var dict in the
            # train_config
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
                    os.path.join(model_name, "metadata", os.path.basename(file_path)),
                    "preprocess_config:",
                    f"preprocess_config: {metadata_preprocess_config_path}",
                )

                replaceLineInFile(
                    os.path.join(model_name, "metadata", os.path.basename(file_path)),
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
                    os.path.join(model_name, "metadata", os.path.basename(file_path)),
                    ".dict_file: &dict_file",
                    f".dict_file: &dict_file {metadata_scale_dict_path}",
                )

                replaceLineInFile(
                    os.path.join(model_name, "metadata", os.path.basename(file_path)),
                    ".var_file: &var_file",
                    f".var_file: &var_file {metadata_var_dict_path}",
                )


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
        frac_dict: dict = None,
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
        super().__init__()

        # Add parameters to as attributes
        self.class_labels = class_labels
        self.main_class = main_class
        self.val_data_dict = val_data_dict
        self.target_beff = target_beff
        self.frac_dict = (
            {
                "cjets": 0.018,
                "ujets": 0.982,
            }
            if frac_dict is None
            else frac_dict
        )
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

            # Once we use python >=3.9
            # (see https://www.python.org/dev/peps/pep-0584/#specification)
            #  switch to the following: dict_epoch |= result_dict
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

            # Once we use python >=3.9
            # (see https://www.python.org/dev/peps/pep-0584/#specification)
            # switch to the following: dict_epoch |= result_dict
            dict_epoch = {**dict_epoch, **result_dict}

        # Append the dict to the list
        self.dict_list.append(dict_epoch)

        # Dump the list in json file
        with open(self.dict_file_name, "w") as outfile:
            json.dump(self.dict_list, outfile, indent=4)


def get_jet_feature_indices(variable_header: dict, exclude: list = None):
    """
    Deletes from the jet samples the keys listed in exclude.

    Parameters
    ----------
    variable_header : dict
        List with the variables.
    exclude : list
        List with the variables that are to be excluded.

    Returns
    -------
    variables : list
        List with the new variables without the excluded ones.
    excluded_variables : list
        List of the excluded variables.
    excluded_var_indices : list
        List of the indicies of the excluded variables.
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


def get_jet_feature_position(variable_list, column_names):
    """
    Return the index position of the variables listed in variable_list within
    the column_names list.
    WARNING: should match the column order of the training data!
    """
    position_list = []
    for variable in variable_list:
        try:
            index_pos = column_names.index(variable)
            position_list.append(index_pos)
        except ValueError as no_var_err:
            raise ValueError(
                f"Variable {variable} to fast forward not found in set!"
            ) from no_var_err
    return position_list


def GetTestSample(
    input_file: str,
    var_dict: str,
    preprocess_config: object,
    class_labels: list,
    nJets: int = int(3e5),
    exclude: list = None,
    cut_vars_dict: dict = None,
    jet_variables: list = None,
    print_logger: bool = True,
):
    """
    Load the jet variables and labels. Scale the jet variables for validation
    use in the NN's.

    Parameters
    ----------
    input_file : str
        Path to the file which is to be loaded.
    var_dict : str
        Variable dict with the wanted jet variables inside.
    preprocess_config : object
        Loaded preprocessing config that was used.
    class_labels : list
        List of classes used for training of the model.
    nJets : int
        Number of jets that should be loaded.
    exclude : list
        List of variables that are not loaded.
    cut_vars_dict : dict
        Dict with the cuts that should be applied.
    jet_variables : list
        List of variables that are used.
    print_logger : bool
        Decide, if the logger info is printed or not.

    Returns
    -------
    jets : numpy.ndarray
        X values of the jets ready to be used in the NN's.
    labels : numpy.ndarray
        Y values ready to be used in the NN's.

    Raises
    ------
    ValueError
        If jet_variables and exclude are used at the same time.
    RuntimeError
        If no file could be found in the given filepath.
    KeyError
        If variable is used which is not in the scale dict.
    """

    # Assert that the jet variables and exlude are not called at the same time
    if jet_variables and exclude:
        raise ValueError("You can't set exclude and jet_variables. Choose one!")

    # Adding class_labels check between preprocess_config and given labels
    # Try/Except here for backward compatibility
    try:
        assert preprocess_config.sampling["class_labels"] == class_labels, (
            "class_labels from preprocessing_config and from train_config are"
            " different! They need to be the same!"
        )

    except (AttributeError, KeyError):
        logger.warning(
            "Deprecation Warning: class_labels are given in preparation"
            " and not in sampling block! Consider moving this to"
            " the sampling block in your config!"
        )
        assert preprocess_config.preparation["class_labels"] == class_labels, (
            "class_labels from preprocessing_config and from train_config are"
            " different! They need to be the same!"
        )

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
                        f"{elem['name']} has been excluded from variable"
                        " config (is in scale dict)."
                    )
                else:
                    logger.warning(
                        f"{elem['name']} in scale dict but not in variable config."
                    )
            continue
        if "isDefaults" in elem["name"]:
            continue
        jets[elem["name"]] -= elem["shift"]
        jets[elem["name"]] /= elem["scale"]
    if not set(variables).issubset(scale_dict_variables):
        raise KeyError(
            f"Requested {(set(variables).difference(scale_dict_variables))}"
            " which are not in scale dict."
        )

    # Return jets and labels
    return jets, labels


def GetTestSampleTrks(
    input_file: str,
    var_dict: str,
    preprocess_config: object,
    class_labels: list,
    nJets: int = int(3e5),
    cut_vars_dict: dict = None,
    print_logger: bool = False,
):
    """
    Load the track variables and labels. Scale the track variables for validation
    use in the NN's.

    Parameters
    ----------
    input_file : str
        Path to the file which is to be loaded.
    var_dict : str
        Variable dict with the wanted track variables inside.
    preprocess_config : object
        Loaded preprocessing config that was used.
    class_labels : list
        List of classes used for training of the model.
    nJets : int
        Number of jets that should be loaded.
    cut_vars_dict : dict
        Dict with the cuts that should be applied.
    print_logger : bool
        Decide, if the logger info is printed or not.

    Returns
    -------
    trks : numpy.ndarray
        X values of the tracks ready to be used in the NN's.
    binary_labels : numpy.ndarray
        Y values ready to be used in the NN's.

    Raises
    ------
    RuntimeError
        If no file could be found in the given filepath.
    """

    # Adding class_labels check between preprocess_config and given labels
    # Try/Except here for backward compatibility
    try:
        assert preprocess_config.sampling["class_labels"] == class_labels, (
            "class_labels from preprocessing_config and from train_config are"
            " different! They need to be the same!"
        )

    except (AttributeError, KeyError):
        logger.warning(
            "Deprecation Warning: class_labels are given in preparation"
            " and not in sampling block! Consider moving this to"
            " the sampling block in your config!"
        )
        assert preprocess_config.preparation["class_labels"] == class_labels, (
            "class_labels from preprocessing_config and from train_config are"
            " different! They need to be the same!"
        )

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
        print_logger=print_logger,
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
    train_config: object,
    preprocess_config: object,
    nJets: int,
    jets_var_list: list = None,
    convert_to_tensor: bool = False,
) -> dict:
    """
    Load the validation data for UMAMI.

    Parameters
    ----------
    train_config : object
        Loaded train_config object.
    preprocess_config : object
        Loaded preprocess_config object.
    nJets : int
        Number of jets to load.
    jets_var_list : list
        List with jet variables that are to be loaded.
    convert_to_tensor : bool
        Decide, if the validation data are converted to
        tensorflow tensors to avoid memory leaks.

    Returns
    -------
    val_data_dict : dict
        Dict with the validation data.
    """
    if jets_var_list is None:
        jets_var_list = []
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
        val_data_dict["X_valid"] = tf.convert_to_tensor(X_valid, dtype=tf.float64)
        val_data_dict["X_valid_trk"] = tf.convert_to_tensor(
            X_valid_trk, dtype=tf.float64
        )
        val_data_dict["Y_valid"] = tf.convert_to_tensor(Y_valid, dtype=tf.int64)

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["X_valid_trk"] = X_valid_trk
        val_data_dict["Y_valid"] = Y_valid

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
            val_data_dict["X_valid"].shape[1] == val_data_dict["X_valid_add"].shape[1]
        ), (
            "validation_file and add_validation_file have different amounts of"
            " variables!"
        )

    else:
        val_data_dict["X_valid_add"] = None
        val_data_dict["X_valid_trk_add"] = None
        val_data_dict["Y_valid_add"] = None

    # Return the val data dict
    return val_data_dict


def load_validation_data_dl1(
    train_config: object,
    preprocess_config: object,
    nJets: int,
    convert_to_tensor: bool = False,
) -> dict:
    """
    Load the validation data for DL1.

    Parameters
    ----------
    train_config : object
        Loaded train_config object.
    preprocess_config : object
        Loaded preprocess_config object.
    nJets : int
        Number of jets to load.
    convert_to_tensor : bool
        Decide, if the validation data are converted to
        tensorflow tensors to avoid memory leaks.

    Returns
    -------
    val_data_dict : dict
        Dict with the validation data.
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
        val_data_dict["X_valid"] = tf.convert_to_tensor(X_valid, dtype=tf.float64)
        val_data_dict["Y_valid"] = tf.convert_to_tensor(Y_valid, dtype=tf.int64)

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["Y_valid"] = Y_valid

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
            val_data_dict["X_valid"].shape[1] == val_data_dict["X_valid_add"].shape[1]
        ), (
            "validation_file and add_validation_file have different amounts of"
            " variables!"
        )

    else:
        val_data_dict["X_valid_add"] = None
        val_data_dict["Y_valid_add"] = None

    # Return the val data dict
    return val_data_dict


def load_validation_data_dips(
    train_config: object,
    preprocess_config: object,
    nJets: int,
    convert_to_tensor: bool = False,
) -> dict:
    """
    Load the validation data for DIPS.

    Parameters
    ----------
    train_config : object
        Loaded train_config object.
    preprocess_config : object
        Loaded preprocess_config object.
    nJets : int
        Number of jets to load.
    convert_to_tensor : bool
        Decide, if the validation data are converted to
        tensorflow tensors to avoid memory leaks.

    Returns
    -------
    val_data_dict : dict
        Dict with the validation data.
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
        val_data_dict["X_valid"] = tf.convert_to_tensor(X_valid, dtype=tf.float64)
        val_data_dict["Y_valid"] = tf.convert_to_tensor(Y_valid, dtype=tf.int64)

    else:
        val_data_dict["X_valid"] = X_valid
        val_data_dict["Y_valid"] = Y_valid

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
            val_data_dict["X_valid"].shape[1] == val_data_dict["X_valid_add"].shape[1]
        ), (
            "validation_file and add_validation_file have different amounts of"
            " variables!"
        )

    else:
        val_data_dict["X_valid_add"] = None
        val_data_dict["Y_valid_add"] = None

    # Return the val data dict
    return val_data_dict


def GetTestFile(
    input_file: str,
    var_dict: str,
    preprocess_config: object,
    class_labels: list,
    nJets: int,
    exclude: list = None,
    cut_vars_dict: dict = None,
    jet_variables: list = None,
    print_logger: bool = True,
):
    """
    Load the jet and track variables and labels. Scale the jet
    and track variables for validation use in the NN's.

    Parameters
    ----------
    input_file : str
        Path to the file which is to be loaded.
    var_dict : str
        Variable dict with the wanted jet variables inside.
    preprocess_config : object
        Loaded preprocessing config that was used.
    class_labels : list
        List of classes used for training of the model.
    nJets : int
        Number of jets that should be loaded.
    exclude : list
        List of variables that are not loaded.
    cut_vars_dict : dict
        Dict with the cuts that should be applied.
    jet_variables : list
        List of variables that are used.
    print_logger : bool
        Decide, if the logger info is printed or not.

    Returns
    -------
    X : numpy.ndarray
        X values of the jets ready to be used in the NN's.
    X_trk : numpy.ndarray
        X values of the tracks ready to be used in the NN's.
    Y : numpy.ndarray
        Y values ready to be used in the NN's.
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
        print_logger=print_logger,
    )

    assert np.equal(Y, Y_trk).all()

    return X, X_trk, Y


def evaluate_model_umami(
    model: object,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
    target_beff: float = 0.77,
) -> dict:
    """
    Evaluate the UMAMI model on the data provided.

    Parameters
    ----------
    model : object
        Loaded UMAMI model for evaluation.
    data_dict : dict
        Dict with the loaded data which are to be evaluated.
    class_labels : list
        List of classes used for training of the model.
    main_class : str
        Main class which is to be tagged.
    target_beff : float
        Working Point which is to be used for evaluation.
    frac_dict : dict
        Dict with the fractions of the non-main classes.
        Sum needs to be one!

    Returns
    -------
    result_dict : dict
        Dict with validation metrics/rejections.
    """

    # Calculate accuracy andloss of UMAMI and Dips part
    (loss, dips_loss, umami_loss, dips_accuracy, umami_accuracy,) = model.evaluate(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        data_dict["Y_valid"],
        batch_size=15_000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    # Evaluate with the model for predictions
    y_pred_dips, y_pred_umami = model.predict(
        [data_dict["X_valid_trk"], data_dict["X_valid"]],
        batch_size=15_000,
        use_multiprocessing=True,
        workers=8,
        verbose=0,
    )

    # Get rejections for DIPS and UMAMI
    rej_dict_dips, disc_cut_dips = umt.GetRejection(
        y_pred=y_pred_dips,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict["dips"],
        target_eff=target_beff,
    )
    rej_dict_umami, disc_cut_umami = umt.GetRejection(
        y_pred=y_pred_umami,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict["umami"],
        target_eff=target_beff,
    )

    # Write metrics to results dict
    result_dict = {
        "val_loss": loss,
        "dips_val_loss": dips_loss,
        "umami_val_loss": umami_loss,
        "dips_val_acc": dips_accuracy,
        "umami_val_acc": umami_accuracy,
        "disc_cut_dips": disc_cut_dips,
        "disc_cut_umami": disc_cut_umami,
    }

    # Write rejections to the results dict
    # TODO Change this in python 3.9
    result_dict.update({f"{key}_umami": elem for key, elem in rej_dict_umami.items()})
    result_dict.update({f"{key}_dips": elem for key, elem in rej_dict_dips.items()})

    # Evaluate Models on add_files if given
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
            batch_size=15_000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Evaluate with the model for predictions
        y_pred_dips_add, y_pred_umami_add = model.predict(
            [data_dict["X_valid_trk_add"], data_dict["X_valid_add"]],
            batch_size=15_000,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Get rejections for DIPS and UMAMI
        rej_dict_dips_add, disc_cut_dips_add = umt.GetRejection(
            y_pred=y_pred_dips_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict["dips"],
            target_eff=target_beff,
        )
        rej_dict_umami_add, disc_cut_umami_add = umt.GetRejection(
            y_pred=y_pred_umami_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict["umami"],
            target_eff=target_beff,
        )

        # Add metrics to results dict
        result_dict.update(
            {
                "val_loss_add": loss_add,
                "dips_val_loss_add": dips_loss_add,
                "umami_val_loss_add": umami_loss_add,
                "dips_val_acc_add": dips_accuracy_add,
                "umami_val_acc_add": umami_accuracy_add,
                "disc_cut_dips_add": disc_cut_dips_add,
                "disc_cut_umami_add": disc_cut_umami_add,
            }
        )

        # Write rejections to the results dict
        # TODO Change this in python 3.9
        result_dict.update(
            {f"{key}_umami_add": elem for key, elem in rej_dict_umami_add.items()}
        )
        result_dict.update(
            {f"{key}_dips_add": elem for key, elem in rej_dict_dips_add.items()}
        )

    return result_dict


def evaluate_model(
    model: object,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    target_beff: float = 0.77,
    frac_dict: dict = None,
) -> dict:
    """
    Evaluate the DIPS/DL1 model on the data provided.

    Parameters
    ----------
    model : object
        Loaded UMAMI model for evaluation.
    data_dict : dict
        Dict with the loaded data which are to be evaluated.
    class_labels : list
        List of classes used for training of the model.
    main_class : str
        Main class which is to be tagged.
    target_beff : float
        Working Point which is to be used for evaluation.
    frac_dict : dict
        Dict with the fractions of the non-main classes.
        Sum needs to be one!

    Returns
    -------
    result_dict : dict
        Dict with validation metrics/rejections.
    """
    if frac_dict is None:
        frac_dict = {"cjets": 0.018, "ujets": 0.982}
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

    rej_dict, disc_cut = umt.GetRejection(
        y_pred=y_pred_dips,
        y_true=data_dict["Y_valid"],
        class_labels=class_labels,
        main_class=main_class,
        frac_dict=frac_dict,
        target_eff=target_beff,
    )

    # Adding the results to result_dict
    result_dict = {
        "val_loss": loss,
        "val_acc": accuracy,
        "disc_cut": disc_cut,
    }

    # Write rejection in results dict
    # TODO Change this in python 3.9
    result_dict.update({f"{key}": elem for key, elem in rej_dict.items()})

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

        rej_dict_add, disc_cut_add = umt.GetRejection(
            y_pred=y_pred_add,
            y_true=data_dict["Y_valid_add"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict,
            target_eff=target_beff,
        )

        # Adding metrics to results dict
        # TODO Change this in python 3.9
        result_dict.update(
            {
                "val_loss_add": loss_add,
                "val_acc_add": accuracy_add,
                "disc_cut_add": disc_cut_add,
            }
        )

        # Write the rejection values to the results dict
        # TODO Change this in python 3.9
        result_dict.update({f"{key}_add": elem for key, elem in rej_dict_add.items()})

    # Return finished dict
    return result_dict


def calc_validation_metrics(
    train_config: object,
    preprocess_config: object,
    tagger: str,
    target_beff: float = 0.77,
    nJets: int = int(3e5),
    model_string: str = "model_epoch",
) -> str:
    """
    Calculates the validation metrics and rejections for each epoch
    and dump it into a json.

    Parameters
    ----------
    train_config : object
        The loaded train config object.
    preprocess_config : object
        The loaded preprocess config object.
    tagger : str
        Name of the tagger that is used to calcualte metrics.
    target_beff : float
        Working point that is to be used.
    nJets : int
        Number of jets to use for calculation.
    model_string : str
        Name of the model files.

    Returns
    -------
    output_file_path
        Path to the validation dict where the results are saved in.

    Raises
    ------
    ValueError
        If "tagger" is not dips, dl1, umami or dips_cond_att.
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
        logger.info("No callback json file with validation metrics found! Make new one")
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

        except ValueError as val_error:
            raise ValueError(
                f"Epoch could not be extracted from {model_string}!"
            ) from val_error

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
