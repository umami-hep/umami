"""Helper functions for training tools."""
from umami.configuration import global_config, logger  # isort:skip
import gc
import json
import os
import re
from glob import glob
from pathlib import Path
from shutil import copyfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error
from tensorflow.keras.callbacks import Callback  # pylint: disable=import-error
from tensorflow.keras.models import load_model  # pylint: disable=import-error
from tensorflow.keras.utils import CustomObjectScope  # pylint: disable=import-error

import umami.metrics as umt
import umami.tf_tools as utf
from umami.data_tools import LoadJetsFromFile, LoadTrksFromFile
from umami.preprocessing_tools import (
    PreprocessConfiguration,
    apply_scaling_jets,
    apply_scaling_trks,
    binarise_jet_labels,
    get_scale_dict,
    get_variable_dict,
)
from umami.tools import natural_keys, replace_line_in_file


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

    epoch = re.search("model_epoch(.+?).h5", string)
    return epoch.group(1)


def get_metrics_file_name(
    working_point: float,
    n_jets: int,
    dir_name: str,
):
    """
    Get the validation dict name based on WP, number of jets and dir_name.

    Parameters
    ----------
    working_point : float
        Working point that was used to calculate validation dict.
    n_jets : int
        Number of jets that was used to calculate validation dict.
    dir_name : str
        Path of the directory where the validation dict is saved.

    Returns
    -------
    train_dict_path : str
        Path of the training dict.
    validation_dict_path : str
        Path of the validation dict.
    """

    # Get the path to the train metrics dict
    train_dict_path = os.path.join(dir_name, "train_metrics_dict.json")

    if n_jets and n_jets > 0:
        # Get the path of the validation dict
        validation_dict_path = os.path.join(
            dir_name,
            f"validation_WP{str(working_point).replace('.','p')}_"
            + f"{int(n_jets)}jets_Dict.json",
        )

    else:
        validation_dict_path = None

    return train_dict_path, validation_dict_path


def get_model_path(model_name: str, epoch: int) -> str:
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
    logger.debug("Treating model %s", model_path)

    # Return path
    return model_path


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
    parameters["working_point"] = float(sp[1].replace("WP", "").replace("p", "."))
    parameters["n_jets"] = int(sp[2].replace("jets", ""))
    parameters["dir_name"] = str(Path(dict_name).parent)

    # Check if the values are correctly extracted. Try to build the name
    # from the parameters and check if they are identical.
    _, val_dict_name = get_metrics_file_name(
        working_point=parameters["working_point"],
        n_jets=parameters["n_jets"],
        dir_name=parameters["dir_name"],
    )
    if val_dict_name != dict_name:
        raise Exception(
            f"Can't infer parameters correctly for {dict_name}. Parameters:"
            f" {parameters}"
        )

    # Return the parameters
    return parameters


def setup_output_directory(
    dir_name: str,
    continue_training: bool = False,
) -> None:
    """
    Check the output directory path and init/clean it.

    Parameters
    ----------
    dir_name : str
        Path of the output directory.
    continue_training : bool
        Decide, if the training is continued (True) and the files in the
        folder are not erased or a new training is started (False) and
        the old model files and validation json files are erased.

    Raises
    ------
    FileExistsError
        If the dir_name is an existing file.
    """

    outdir = Path(dir_name)
    if outdir.is_dir() and not continue_training:
        logger.info("Removing model*.h5 and *.json files.")
        for model_file in outdir.glob("model_files/model_epoch*.h5"):
            model_file.unlink()
        for model_file in outdir.glob("*.json"):
            model_file.unlink()
    elif outdir.is_dir() and continue_training:
        logger.info("Continue training. Old model files will not be erased.")
    elif outdir.is_file():
        raise FileExistsError(
            f"{dir_name} is the output directory name but it already exists as a file!"
        )
    else:
        outdir.mkdir()


def create_metadata_folder(
    train_config_path: str,
    var_dict_path: str,
    model_name: str,
    preprocess_config_path: str,
    model_file_path: str = None,
    overwrite_config: bool = False,
) -> None:
    """
    Create a metadata folder in the new model_name dir and
    copy all configs there and change the paths inside the
    configs to the new metadata directory path.
    The following configs are copied:
    preprocessing config, variable dict, scale dict, training config and
    model_file

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
    model_file_path : str
        Path to a model to start from (the model given in model_file).
    overwrite_config : bool
        If configs already in metadata folder, overwrite
        them or not.
    """

    # Check if model path already existing
    # If not, make it
    meta_data_folder = Path(model_name) / "metadata"
    logger.info("Creating meta data folder %s", meta_data_folder)
    meta_data_folder.mkdir(parents=True, exist_ok=True)

    # Create directory for models
    model_files_folder = Path(model_name) / "model_files"
    logger.info("Creating models folder %s", model_files_folder)
    model_files_folder.mkdir(parents=True, exist_ok=True)

    # Get scale dict and copy it to metadata
    preprocess_config = PreprocessConfiguration(preprocess_config_path)
    metadata_preprocess_config_path = (
        meta_data_folder / preprocess_config.yaml_config.name
    )
    if (overwrite_config is True) or not metadata_preprocess_config_path.is_file():
        preprocess_config.copy_to_out_dir(suffix="", out_dir=meta_data_folder)

    # Copy files to metadata folder if not existing
    for file_path in [
        train_config_path,
        var_dict_path,
        model_file_path,
    ]:
        if file_path is None:
            continue
        new_file_path = meta_data_folder / Path(file_path).name
        if (overwrite_config is False) and new_file_path.is_file():
            continue
        logger.info("Copy %s to metadata folder!", file_path)
        copyfile(file_path, new_file_path)

        # Change the paths for the preprocess config and var dict in the
        # train_config
        if file_path == train_config_path:

            replace_line_in_file(
                new_file_path,
                "preprocess_config:",
                f"preprocess_config: {metadata_preprocess_config_path}",
            )

            if model_file_path:
                metadata_model_file_path = meta_data_folder / Path(model_file_path).name

                replace_line_in_file(
                    new_file_path,
                    "model_file:",
                    f"model_file: {metadata_model_file_path}",
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
        model_name: str,
        use_lrr: bool,
        n_jets: int = None,
        val_data_dict: dict = None,
        target_beff: float = 0.77,
        frac_dict: dict = None,
        continue_training: bool = False,
        batch_size: int = 15_000,
    ):
        """Init the parameters needed for the callback

        Parameters
        ----------
        class_labels : list
            List of class labels used in training (ORDER MATTERS!).
        main_class : str
            Name of the main class which is used. For b-tagging
            obviously `bjets`.
        model_name : str
            Name of the model used to evaluate. This is important
            for the path where the results are of the callback are saved.
        n_jets : int, optional
            Number of jets used in the validation. By default None
        val_data_dict : dict, optional
            Dict with the loaded validation data. These are loaded
            using the `load_validation_data` functions. By default
            None.
        target_beff : float
            Float value between 0 and 1 for which main class efficiency
            the rejections are calculated.
        frac_dict : dict
            Dict with the fraction values for the non-main classes. The
            values need to add up to 1.
        use_lrr : bool
            Info if a learning rate reducer is used in the training.
        continue_training : bool, optional
            Decide, if the this is a continuation of an already existing training
            or not, by default False.
        batch_size : int, optional
            Set the batch size used for validation.
        """
        super().__init__()

        # Add parameters to as attributes
        self.class_labels = class_labels
        self.main_class = main_class
        self.n_jets = n_jets
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
        self.continue_training = continue_training
        self.batch_size = batch_size
        self.use_lrr = use_lrr

        # Get the names for the train/val metrics files
        (
            self.train_metrics_file_name,
            self.val_metrics_file_name,
        ) = get_metrics_file_name(
            working_point=self.target_beff,
            n_jets=self.n_jets,
            dir_name=self.model_name,
        )

        # Init a list for the result dicts for each epoch
        if self.continue_training:
            try:
                with open(self.train_metrics_file_name, "r") as file:
                    self.train_metrics_list = json.loads(file.read())

            except FileNotFoundError:
                logger.warning(
                    "No train metrics file found named %s! Init a new one!",
                    self.train_metrics_file_name,
                )
                self.train_metrics_list = []

            if self.val_data_dict is not None and self.n_jets > 0:
                try:
                    with open(self.val_metrics_file_name, "r") as file:
                        self.val_metrics_list = json.loads(file.read())

                except FileNotFoundError:
                    logger.warning(
                        "No validation metrics file found named %s! Init a new one!",
                        self.val_metrics_file_name,
                    )
                    self.val_metrics_list = []

            else:
                self.val_metrics_list = []

        else:
            self.train_metrics_list = []
            self.val_metrics_list = []

        # Init the directory and clean it from previous training
        setup_output_directory(
            dir_name=self.model_name,
            continue_training=self.continue_training,
        )


class MyCallback(CallbackBase):
    """Callback class for the standard taggers

    This class is the callback for the standard taggers. Only one
    output (not like the umami tagger) is given.
    """

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict = None,
    ) -> None:
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
        train_metrics_dict = {
            "epoch": epoch + 1,
            "loss": logs["loss"],
            "accuracy": logs["accuracy"],
        }

        # Write out learning rate if a lrr is used
        if self.use_lrr:
            train_metrics_dict["learning_rate"] = logs["lr"].item()

        # Append the dict to the list
        self.train_metrics_list.append(train_metrics_dict)

        # Dump the list in json file
        with open(self.train_metrics_file_name, "w") as train_outfile:
            json.dump(self.train_metrics_list, train_outfile, indent=4)

        # If val data is given, calculate validaton metrics and rejections
        if self.val_data_dict:
            val_metrics_dict = evaluate_model(
                model=self.model,
                data_dict=self.val_data_dict,
                class_labels=self.class_labels,
                main_class=self.main_class,
                target_beff=self.target_beff,
                frac_dict=self.frac_dict,
                batch_size=self.batch_size,
            )

            # Add the epoch to val_metrics dict
            val_metrics_dict["epoch"] = epoch + 1

            # Append the dict to the list
            self.val_metrics_list.append(val_metrics_dict)

            # Dump the list in json file
            with open(self.val_metrics_file_name, "w") as val_outfile:
                json.dump(self.val_metrics_list, val_outfile, indent=4)

        # Cleaning up the memory used by evaluate_model
        gc.collect()
        K.clear_session()


class MyCallbackUmami(CallbackBase):
    """Callback class for the umami tagger

    This class is the callback for the umami tagger. Due to the
    two outputs of the tagger, we need special metrics etc.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
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
        train_metrics_dict = {
            "epoch": epoch + 1,
            "loss": logs["loss"],
            "loss_dips": logs["dips_loss"],
            "loss_umami": logs["umami_loss"],
            "accuracy_dips": logs["dips_accuracy"],
            "accuracy_umami": logs["umami_accuracy"],
        }

        # Write out learning rate if a lrr is used
        if self.use_lrr:
            train_metrics_dict["learning_rate"] = logs["lr"].item()

        # Append the dict to the list
        self.train_metrics_list.append(train_metrics_dict)

        # Dump the list in json file
        with open(self.train_metrics_file_name, "w") as train_outfile:
            json.dump(self.train_metrics_list, train_outfile, indent=4)

        # If val data is given, calculate validaton metrics and rejections
        if self.val_data_dict:
            val_metrics_dict = evaluate_model_umami(
                model=self.model,
                data_dict=self.val_data_dict,
                class_labels=self.class_labels,
                main_class=self.main_class,
                target_beff=self.target_beff,
                frac_dict=self.frac_dict,
                batch_size=self.batch_size,
            )

            # Add the epoch to val_metrics dict
            val_metrics_dict["epoch"] = epoch + 1

            # Append the dict to the list
            self.val_metrics_list.append(val_metrics_dict)

            # Dump the list in json file
            with open(self.val_metrics_file_name, "w") as val_outfile:
                json.dump(self.val_metrics_list, val_outfile, indent=4)

        # Cleaning up the memory used by evaluate_model
        gc.collect()
        K.clear_session()


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
            logger.warning("Variables to exclude not found: %s", exclude_that)
    # Get the index of the excluded variables for training
    excluded_var_indices = [
        i for i, excl in enumerate(all_variables) if excl in excluded_variables
    ]
    # set to None if the list of excluded variables is empty
    excluded_var_indices = (
        None if len(excluded_var_indices) == 0 else excluded_var_indices
    )
    logger.debug("variables: %s", variables)
    logger.debug("excluded_variables: %s", excluded_variables)
    logger.debug("excluded_var_indices: %s", excluded_var_indices)
    return variables, excluded_variables, excluded_var_indices


def get_jet_feature_position(
    variable_list: list,
    column_names: list,
) -> list:
    """
     Return the index position of the variables listed in variable_list within
    the column_names list.
    WARNING: should match the column order of the training data!

    Parameters
    ----------
    variable_list : list
        List with the variables
    column_names : list
        List with the names of the columns

    Returns
    -------
    list
        List with the positions of the columns

    Raises
    ------
    ValueError
        If the variable is not in the set.
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


def get_test_sample(
    input_file: str,
    var_dict: str,
    scale_dict: str,
    class_labels: list,
    n_jets: int = int(3e5),
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
        Path to the variable dict or the loaded variable dict directly.
    scale_dict : str
        Path to the scale dict or the loaded scale dict directly.
    class_labels : list
        List of classes used for training of the model.
    n_jets : int
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
        If no input file could be found in the given filepath.
    """

    # Assert that the jet variables and exlude are not called at the same time
    if jet_variables and exclude:
        raise ValueError("You can't set exclude and jet_variables. Choose one!")

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

    # Load scale and variables dict
    var_dict_loaded = get_variable_dict(var_dict)
    scale_dict_loaded = get_scale_dict(file_path=scale_dict, dict_key="jets")

    jets, Umami_labels = LoadJetsFromFile(
        filepath=filepaths,
        class_labels=class_labels,
        n_jets=n_jets,
        cut_vars_dict=cut_vars_dict,
        variables=jet_variables,
        print_logger=print_logger,
    )

    # Binarize Labels
    labels = binarise_jet_labels(
        labels=Umami_labels,
        internal_labels=list(range(len(class_labels))),
    )

    # Check if jet_variables is defined
    if jet_variables:
        # Retrieve the defined variables
        variables = jet_variables
        excluded_variables = []

    else:
        # Retrieve variables and the excluded variables from the config
        variables, excluded_variables, _ = get_jet_feature_indices(
            var_dict_loaded["train_variables"], exclude
        )

    logger.debug("Excluded variables: %s", excluded_variables)

    jets = apply_scaling_jets(
        jets=jets,
        variables_list=variables,
        scale_dict=scale_dict_loaded,
    )

    # Return jets and labels
    return jets, labels


def get_test_sample_trks(
    input_file: str,
    var_dict: str,
    scale_dict: str,
    class_labels: list,
    tracks_name: str,
    n_jets: int = int(3e5),
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
        Path to the variable dict or the loaded variable dict directly.
    scale_dict : str
        Path to the scale dict or the loaded scale dict directly.
    class_labels : list
        List of classes used for training of the model.
    tracks_name : str
        Name of tracks collection to use.
    n_jets : int
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
        If no input file could be found in the given filepath.
    """

    # making sure the n_jets argument is an integer
    n_jets = int(n_jets)

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

    # Load scale and variables dict
    var_dict_loaded = get_variable_dict(var_dict)
    scale_dict_loaded = get_scale_dict(file_path=scale_dict, dict_key=str(tracks_name))

    trks, labels = LoadTrksFromFile(
        filepath=filepaths,
        class_labels=class_labels,
        n_jets=n_jets,
        tracks_name=tracks_name,
        cut_vars_dict=cut_vars_dict,
        print_logger=print_logger,
    )

    # Binarize the labels
    binary_labels = binarise_jet_labels(
        labels=labels,
        internal_labels=list(range(len(class_labels))),
    )

    # Apply scaling to the tracks
    trks, _ = apply_scaling_trks(
        trks=trks,
        variable_config=var_dict_loaded,
        scale_dict=scale_dict_loaded,
        tracks_name=tracks_name,
    )

    return trks, binary_labels


def load_validation_data(
    train_config: object,
    n_jets: int,
    jets_var_list: list = None,
    convert_to_tensor: bool = False,
    nCond: int = None,
) -> dict:
    """
    Load the validation data for UMAMI.

    Parameters
    ----------
    train_config : object
        Loaded train_config object.
    n_jets : int
        Number of jets to load.
    jets_var_list : list
        List with jet variables that are to be loaded.
    convert_to_tensor : bool
        Decide, if the validation data are converted to
        tensorflow tensors to avoid memory leaks.
    nCond: int
        Number of addittional variables used for attention

    Returns
    -------
    val_data_dict : dict
        Dict with the validation data.

    Raises
    ------
    ValueError
        If the given tagger is not supported.
    """
    if jets_var_list is None:
        jets_var_list = []

    # Get class labels and tagger name
    class_labels = train_config.nn_structure["class_labels"]
    tagger = train_config.nn_structure["tagger"]

    # Init a new dict for the loaded val data
    val_data_dict = {}
    val_files = train_config.validation_files

    # Set the tracks collection name
    if hasattr(train_config, "tracks_name"):
        tracks_name = train_config.tracks_name
        logger.debug("Using tracks_name value '%s' for validation", tracks_name)

    else:
        tracks_name = None

    # Check for excluded variables
    exclude = train_config.config.get("exclude", None)

    # Loop over the different validation files
    for val_file_identifier, val_file_config in val_files.items():
        logger.info("Loading validation file %s", val_file_identifier)
        # Get the cut vars dict if defined
        cut_vars_dict = (
            val_file_config["variable_cuts"]
            if "variable_cuts" in val_file_config
            else None
        )

        # Set default to None and change it if used
        X_valid, X_valid_trk, Y_valid = None, None, None

        # Check which tagger is used and load the neede data
        if tagger.casefold() in ("dips", "dips_attention"):
            (X_valid, Y_valid,) = get_test_sample_trks(
                input_file=val_file_config["path"],
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=class_labels,
                tracks_name=tracks_name,
                n_jets=n_jets,
                cut_vars_dict=cut_vars_dict,
            )

        elif tagger.casefold() in ("dl1"):
            (X_valid, Y_valid,) = get_test_sample(
                input_file=val_file_config["path"],
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=class_labels,
                n_jets=n_jets,
                exclude=exclude,
                cut_vars_dict=cut_vars_dict,
            )

        elif tagger.casefold() in ("umami", "umami_cond_att", "cads"):
            (X_valid, X_valid_trk, Y_valid,) = get_test_file(
                input_file=val_file_config["path"],
                var_dict=train_config.var_dict,
                scale_dict=train_config.preprocess_config.dict_file,
                class_labels=class_labels,
                tracks_name=tracks_name,
                n_jets=n_jets,
                exclude=exclude,
                jet_variables=jets_var_list,
                cut_vars_dict=cut_vars_dict,
            )

        else:
            raise ValueError(f"The defined tagger {tagger} is not supported!")

        if X_valid is not None:
            val_data_dict[f"X_valid_{val_file_identifier}"] = (
                tf.convert_to_tensor(X_valid, dtype=tf.float32)
                if convert_to_tensor
                else X_valid
            )

        if X_valid_trk is not None:
            val_data_dict[f"X_valid_trk_{val_file_identifier}"] = (
                tf.convert_to_tensor(X_valid_trk, dtype=tf.float32)
                if convert_to_tensor
                else X_valid_trk
            )

        if Y_valid is not None:
            val_data_dict[f"Y_valid_{val_file_identifier}"] = (
                tf.convert_to_tensor(Y_valid, dtype=tf.uint8)
                if convert_to_tensor
                else Y_valid
            )

        if nCond is not None:
            val_data_dict[f"X_valid_addvars_{val_file_identifier}"] = (
                tf.convert_to_tensor(X_valid.iloc[:, :nCond], dtype=tf.float32)
                if convert_to_tensor
                else X_valid.iloc[:, :nCond]
            )

    # Return the val data dict
    return val_data_dict


def get_test_file(
    input_file: str,
    var_dict: str,
    scale_dict: str,
    class_labels: list,
    tracks_name: str,
    n_jets: int,
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
        Path to the variable dict or the loaded variable dict directly.
    scale_dict : str
        Path to the scale dict or the loaded scale dict directly.
    class_labels : list
        List of classes used for training of the model.
    tracks_name : str
        Name of the tracks collection to use.
    n_jets : int
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

    X_trk, Y_trk = get_test_sample_trks(
        input_file=input_file,
        var_dict=var_dict,
        scale_dict=scale_dict,
        class_labels=class_labels,
        tracks_name=tracks_name,
        n_jets=int(n_jets),
        cut_vars_dict=cut_vars_dict,
        print_logger=False,
    )

    X, Y = get_test_sample(
        input_file=input_file,
        var_dict=var_dict,
        scale_dict=scale_dict,
        class_labels=class_labels,
        n_jets=int(n_jets),
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
    batch_size: int = 15_000,
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
    frac_dict : dict
        Dict with the fractions of the non-main classes.
        Sum needs to be one!
    target_beff : float, optional
        Working Point which is to be used for evaluation.
        By default 0.77
    batch_size : int, optional
        Number of jets used for validation. By default
        15_000

    Returns
    -------
    result_dict : dict
        Dict with validation metrics/rejections.
    """

    validation_file_identifiers = get_unique_identifiers(
        keys=list(data_dict.keys()), prefix="Y_valid"
    )

    if len(validation_file_identifiers) == 0:
        logger.warning("Didn't find any validation file identifiers.")

    result_dict = {}

    # loop over validation files and load X_valid, X_valid_trk, Y_valid for each file
    for val_file_identifier in validation_file_identifiers:
        # Check which input data need to be used
        # Calculate accuracy andloss of UMAMI and Dips part
        if f"X_valid_addvars_{val_file_identifier}" in data_dict:
            x = [
                data_dict[f"X_valid_trk_{val_file_identifier}"],
                data_dict[f"X_valid_addvars_{val_file_identifier}"],
                data_dict[f"X_valid_{val_file_identifier}"],
            ]
        else:
            x = [
                data_dict[f"X_valid_trk_{val_file_identifier}"],
                data_dict[f"X_valid_{val_file_identifier}"],
            ]
        (loss, dips_loss, umami_loss, dips_accuracy, umami_accuracy,) = model.evaluate(
            x,
            data_dict[f"Y_valid_{val_file_identifier}"],
            batch_size=batch_size,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Evaluate with the model for predictions
        y_pred_dips, y_pred_umami = model.predict(
            x,
            batch_size=batch_size,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        # Get rejections for DIPS and UMAMI
        rej_dict_dips, disc_cut_dips = umt.get_rejection(
            y_pred=y_pred_dips,
            y_true=data_dict[f"Y_valid_{val_file_identifier}"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict["dips"],
            target_eff=target_beff,
            unique_identifier=val_file_identifier,
            subtagger="dips",
        )
        rej_dict_umami, disc_cut_umami = umt.get_rejection(
            y_pred=y_pred_umami,
            y_true=data_dict[f"Y_valid_{val_file_identifier}"],
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict["umami"],
            target_eff=target_beff,
            unique_identifier=val_file_identifier,
            subtagger="umami",
        )

        # Write metrics to results dict
        # TODO Change this in python 3.9
        result_dict.update(
            {
                f"val_loss_{val_file_identifier}": loss,
                f"val_loss_dips_{val_file_identifier}": dips_loss,
                f"val_loss_umami_{val_file_identifier}": umami_loss,
                f"val_acc_dips_{val_file_identifier}": dips_accuracy,
                f"val_acc_umami_{val_file_identifier}": umami_accuracy,
                f"disc_cut_dips_{val_file_identifier}": disc_cut_dips,
                f"disc_cut_umami_{val_file_identifier}": disc_cut_umami,
            }
        )

        # Write rejections to the results dict
        # TODO Change this in python 3.9
        result_dict.update(rej_dict_umami)
        result_dict.update(rej_dict_dips)

    return result_dict


def evaluate_model(
    model: object,
    data_dict: dict,
    class_labels: list,
    main_class: str,
    target_beff: float = 0.77,
    frac_dict: dict = None,
    batch_size: int = 15_000,
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
    batch_size : int, optional
        Number of jets used for validation. By default
        15_000

    Returns
    -------
    result_dict : dict
        Dict with validation metrics/rejections.
    """

    validation_file_identifiers = get_unique_identifiers(
        keys=list(data_dict.keys()), prefix="Y_valid"
    )

    if len(validation_file_identifiers) == 0:
        logger.warning("Didn't find any validation file identifiers.")

    result_dict = {}
    # loop over validation files and load X_valid, Y_valid for each file
    for val_file_identifier in validation_file_identifiers:
        # Check which input data need to be used
        if (
            f"X_valid_trk_{val_file_identifier}" in data_dict
            and f"X_valid_{val_file_identifier}" in data_dict
        ):
            x = [
                data_dict[f"X_valid_trk_{val_file_identifier}"],
                data_dict[f"X_valid_{val_file_identifier}"],
            ]

        elif (
            f"X_valid_trk_{val_file_identifier}" in data_dict
            and f"X_valid_{val_file_identifier}" not in data_dict
        ):
            x = data_dict[f"X_valid_trk_{val_file_identifier}"]

        else:
            x = data_dict[f"X_valid_{val_file_identifier}"]

        loss, accuracy = model.evaluate(
            x=x,
            y=data_dict[f"Y_valid_{val_file_identifier}"],
            batch_size=batch_size,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        y_pred_dips = model.predict(
            x=x,
            batch_size=batch_size,
            use_multiprocessing=True,
            workers=8,
            verbose=0,
        )

        rej_dict, disc_cut = umt.get_rejection(
            y_pred=y_pred_dips,
            y_true=data_dict[f"Y_valid_{val_file_identifier}"],
            unique_identifier=val_file_identifier,
            class_labels=class_labels,
            main_class=main_class,
            frac_dict=frac_dict,
            target_eff=target_beff,
        )

        # Adding the results to result_dict
        result_dict.update(
            {
                f"val_loss_{val_file_identifier}": loss,
                f"val_acc_{val_file_identifier}": accuracy,
                f"disc_cut_{val_file_identifier}": disc_cut,
            }
        )

        # Write the rejection values to the results dict
        # TODO Change this in python 3.9
        result_dict.update(
            {f"{key}": rej_dict[key] for key in rej_dict}  # pylint: disable=C0206
        )

    # Return finished dict
    return result_dict


def calc_validation_metrics(
    train_config: object,
    tagger: str,
    target_beff: float = 0.77,
    n_jets: int = int(3e5),
    model_string: str = "model_epoch",
) -> str:
    """
    Calculates the validation metrics and rejections for each epoch
    and dump it into a json.

    Parameters
    ----------
    train_config : object
        The loaded train config object.
    tagger : str
        Name of the tagger that is used to calcualte metrics.
    target_beff : float
        Working point that is to be used.
    n_jets : int
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
        If "tagger" is not dips, dl1, umami or cads.
    """

    # Get evaluation parameters and NN structure from train config
    eval_parameters = train_config.evaluation_settings
    val_params = train_config.validation_settings
    nn_structure = train_config.nn_structure
    second_model_string = (
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
            "%s models used but not found! Using %s", model_string, second_model_string
        )

        # Set new model string
        model_string = second_model_string

        # Make a list with the model epochs saves with second model name string
        training_output = [
            os.path.join(f"{train_config.model_name}/model_files/", f)
            for f in os.listdir(f"{train_config.model_name}/model_files/")
            if model_string in f
        ]

    # Open the json file and load the training out
    try:

        # Get val dict file name
        _, val_output_file_path = get_metrics_file_name(
            working_point=eval_parameters["working_point"],
            n_jets=eval_parameters["n_jets"],
            dir_name=train_config.model_name,
        )

        with open(val_output_file_path, "r") as training_out_json:
            training_output_list = json.load(training_out_json)

    except FileNotFoundError:
        logger.info("No callback json file with validation metrics found! Make new one")
        training_output_list = [
            {"epoch": n} for n in range(train_config.nn_structure["epochs"])
        ]

    # Init a results list
    results = []

    # Check tagger and load the correct val data
    data_dict = load_validation_data(
        train_config=train_config,
        n_jets=n_jets,
        convert_to_tensor=False,
        jets_var_list=[
            global_config.etavariable,
            global_config.pTvariable,
        ]
        if tagger.casefold() == "cads"
        else None,
    )

    # Loop over the different model savepoints at each epoch
    for n, model_file in enumerate(sorted(training_output, key=natural_keys)):
        logger.info("Working on %i/%i input files", n + 1, len(training_output))

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

        if tagger.casefold() == "umami":
            # Load UMAMI model
            umami = load_model(model_file, {"Sum": utf.Sum})

            # Evaluate Umami model
            val_result_dict = evaluate_model_umami(
                model=umami,
                data_dict=data_dict,
                class_labels=nn_structure["class_labels"],
                main_class=nn_structure["main_class"],
                target_beff=target_beff,
                frac_dict=eval_parameters["frac_values"],
                batch_size=val_params["val_batch_size"],
            )

            # Delete model
            del umami

        elif tagger.casefold() == "dl1":
            # Load DL1 model
            dl1 = load_model(model_file)

            # Evaluate DL1 model
            val_result_dict = evaluate_model(
                model=dl1,
                data_dict=data_dict,
                class_labels=nn_structure["class_labels"],
                main_class=nn_structure["main_class"],
                target_beff=target_beff,
                frac_dict=eval_parameters["frac_values"],
                batch_size=val_params["val_batch_size"],
            )

            # Delete model
            del dl1

        elif tagger.casefold() == "dips":
            # Load DIPS model
            with CustomObjectScope({"Sum": utf.Sum}):
                dips = load_model(model_file)

            # Validate dips
            val_result_dict = evaluate_model(
                model=dips,
                data_dict=data_dict,
                class_labels=nn_structure["class_labels"],
                main_class=nn_structure["main_class"],
                target_beff=target_beff,
                frac_dict=eval_parameters["frac_values"],
                batch_size=val_params["val_batch_size"],
            )

            # Delete model
            del dips

        elif tagger.casefold() in ("cads", "dips_attention"):
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
                cads = load_model(model_file)

            # Validate dips
            val_result_dict = evaluate_model(
                model=cads,
                data_dict=data_dict,
                class_labels=nn_structure["class_labels"],
                main_class=nn_structure["main_class"],
                target_beff=target_beff,
                frac_dict=eval_parameters["frac_values"],
                batch_size=val_params["val_batch_size"],
            )

            # Delete model
            del cads

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
    _, val_output_file_path = get_metrics_file_name(
        working_point=target_beff,
        n_jets=n_jets,
        dir_name=train_config.model_name,
    )

    # Dump dict into json
    with open(val_output_file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    # Return Validation dict name
    return val_output_file_path
