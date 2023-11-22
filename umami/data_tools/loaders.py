"""Provides functions for loading datasets from files."""
from copy import deepcopy
from glob import glob

import h5py
import numpy as np
import pandas as pd

from umami.configuration import global_config, logger
from umami.data_tools.cuts import get_sample_cuts
from umami.helper_tools import get_class_label_variables
from umami.tools import natural_keys


def load_jets_from_file(
    filepath: str,
    class_labels: list,
    n_jets: int = None,
    variables: list = None,
    cut_vars_dict: dict = None,
    print_logger: bool = True,
    chunk_size: int = 1e6,
    indices_to_load: tuple = None,
):
    """
    Load jets from file. Only jets from classes in class_labels are returned.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file with the jets.
    class_labels : list
        List of class labels which are used.
    n_jets : int
        Number of jets to load.
    variables : list
        Variables which are loaded.
    cut_vars_dict : dict
        Variable cuts that are applied when loading the jets.
    print_logger : bool
        Decide if the number of jets loaded from the file is printed.
    chunk_size : int
        Chunk size how much jets are loaded in on go.
    indices_to_load : int, optional
        Load the given indices, by default None

    Returns
    -------
    all_jets : pandas.DataFrame
        The jets as numpy ndarray
    all_labels : numpy.ndarray
        The internal class label for each jet. Corresponds with the
        index of the class label in class_labels.

    Raises
    ------
    ValueError
        If neither n_jets nor indices_to_load is given
    ValueError
        If both n_jets and indices_to_load is given
    KeyError
        If filepath is not a list or a string
    RuntimeError
        If no files could be found in filepath
    """

    # Check that either n_jets or indices_to_load is given
    if n_jets is None and indices_to_load is None:
        raise ValueError("Neither n_jets nor indices_to_load were given!")

    if n_jets is not None and indices_to_load is not None:
        raise ValueError("You can't give both n_jets and indices_to_load!")

    if n_jets:
        # Make sure the n_jets argument is an integer
        n_jets = int(n_jets)
        chunk_size = int(chunk_size)

        # Check if the chunk size is smaller than n_jets, if yes change it
        chunk_size = chunk_size if chunk_size >= n_jets else n_jets

    if isinstance(filepath, str):
        # Get the paths of the files as a iterable list
        filepaths = glob(filepath)

    elif isinstance(filepath, list):
        filepaths = filepath

    else:
        raise KeyError(f"Given filepath is {type(filepath)}, not a string or a list!")

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {filepath}!
            Check the filepath in your train_config file!
            """
        )

    # Get class_labels variables etc. from global config
    class_label_vars = get_class_label_variables(class_labels)

    if variables:
        # Get list with all available variables
        avai_var_list = list(h5py.File(filepaths[0], "r")["/jets"].dtype.fields.keys())

        # Make a copy of the variables list to loop over it
        variables_list = deepcopy(variables)

        # Add the class label variables to the variables list
        variables_list += class_label_vars

        # Check if all variables from variable_list are available
        for var in variables:
            if var not in avai_var_list:
                variables_list.remove(var)

                # Check for logger
                if print_logger:
                    logger.warning("%s not available in files!", var)

        # Load variables for cuts if given
        if cut_vars_dict:
            # Iterate over the cuts and get the needed variables
            for variable in cut_vars_dict:
                variables_list += list(variable.keys())

        # Ensure each variable is only once in the list
        variables_list = list(set(variables_list))

    # Init a counter for the number of loaded jets
    n_jets_counter = 0

    for j, file in enumerate(sorted(filepaths, key=natural_keys)):
        # Get the number of available jets in the file
        n_jets_infile = len(h5py.File(file, "r")["/jets"])

        # Check how many times we need to iterate over the file
        # to get all jets. If indices to load is used, this is
        # zero because one specific is loaded
        if not indices_to_load:
            loop_list = range(int(np.ceil(n_jets_infile / chunk_size)))

        else:
            loop_list = [0]

        # Iterate over the file
        for infile_counter in loop_list:
            if not indices_to_load:
                lower_index_counter = infile_counter * chunk_size
                upper_index_counter = (infile_counter + 1) * chunk_size

            else:
                lower_index_counter = indices_to_load[0]
                upper_index_counter = indices_to_load[1]

            if variables:
                jets = pd.DataFrame(
                    h5py.File(file, "r")["/jets"].fields(variables_list)[
                        lower_index_counter:upper_index_counter
                    ]
                )

            else:
                jets = pd.DataFrame(
                    h5py.File(file, "r")["/jets"][
                        lower_index_counter:upper_index_counter
                    ]
                )

            # Init new column for string labels
            jets["Umami_string_labels"] = np.zeros_like(jets[class_label_vars[0]])
            jets["Umami_labels"] = np.zeros_like(jets[class_label_vars[0]])

            # Change type of column to strings
            jets = jets.astype({"Umami_string_labels": "str"})

            flavour_categories = global_config.flavour_categories
            for class_label in class_labels:
                cuts = flavour_categories[class_label]["cuts"]
                indices_tochange = np.ones_like(jets["Umami_labels"]).astype(bool)
                indices_toremove = get_sample_cuts(jets=jets, cuts=cuts)
                indices_tochange[indices_toremove] = False

                jets["Umami_string_labels"].values[indices_tochange] = class_label

                # Add the right column label to class
                jets["Umami_labels"].values[indices_tochange] = class_labels.index(
                    class_label
                )

            # Iterate over the classes and add the correct labels to Umami columns

            # Define the conditions to remove
            toremove_conditions = jets["Umami_string_labels"] == "0"

            # Get the indices of the jets that are not used
            indices_toremove = np.where(
                toremove_conditions == True  # pylint: disable=C0121 # noqa: E712
            )[0]

            if cut_vars_dict:
                # Apply cuts and get a list of which jets to remove
                indices_toremove_cuts = get_sample_cuts(jets=jets, cuts=cut_vars_dict)

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(indices_toremove.tolist() + indices_toremove_cuts.tolist())
                    )
                )

            # Remove all unused jets
            if len(indices_toremove) != 0:
                jets = jets.drop(indices_toremove)

            # Remove the string labels
            jets = jets.drop(columns=["Umami_string_labels"])

            # If not the first file processed, append to the global one
            if j == 0 and infile_counter == 0:
                all_jets = jets
                all_labels = jets["Umami_labels"].values

            # if the first file processed, set as global one
            else:
                all_jets = all_jets.append(jets, ignore_index=True)
                all_labels = np.append(all_labels, jets["Umami_labels"].values)

            # Adding the loaded jets to counter
            n_jets_counter += len(jets)

            # Break the loop inside the file if enough jets are loaded
            if indices_to_load is None:
                if n_jets_counter >= n_jets:
                    break

        # Break the loop over the files if enough jets are loaded
        if indices_to_load is None:
            if n_jets_counter >= n_jets:
                break

    if print_logger:
        # Check if enough jets are loaded
        if indices_to_load is None:
            if n_jets_counter < n_jets:
                logger.warning(
                    "Requested %i but only %i could be loaded!",
                    n_jets,
                    n_jets_counter,
                )

            else:
                logger.info("Loaded %i jets!", n_jets)

        else:
            logger.info("Loaded jets for given indices!")

    # Return the jets and internal labels
    return all_jets[:n_jets], all_labels[:n_jets]


def load_trks_from_file(
    filepath: str,
    class_labels: list,
    n_jets: int = None,
    tracks_name: str = "tracks",
    cut_vars_dict: dict = None,
    print_logger: bool = True,
    chunk_size: int = 1e6,
    indices_to_load: tuple = None,
):
    """
    Load tracks from file. Only jets from classes in class_labels are returned.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file with the jets.
    class_labels : list
        List of class labels which are used.
    n_jets : int
        Number of jets to load.
    tracks_name : str
        Name of the tracks collection to load
    cut_vars_dict : dict
        Variable cuts that are applied when loading the jets.
    print_logger : bool
        Decide if the number of jets loaded from the file is printed.
    chunk_size : int
        Chunk size how much jets are loaded in on go.
    indices_to_load : int, optional
        Load the given indices, by default None

    Returns
    -------
    all_trks : numpy.ndarray
        The tracks of the jets as numpy ndarray
    all_labels : numpy.ndarray
        The internal class label for each jet. Corresponds with the
        index of the class label in class_labels.

    Raises
    ------
    ValueError
        If neither n_jets nor indices_to_load is given
    ValueError
        If both n_jets and indices_to_load is given
    KeyError
        If filepath is not a list or a string
    RuntimeError
        If no files could be found in filepath
    """

    # Check that either n_jets or indices_to_load is given
    if n_jets is None and indices_to_load is None:
        raise ValueError("Neither n_jets nor indices_to_load were given!")

    if n_jets is not None and indices_to_load is not None:
        raise ValueError("You can't give both n_jets and indices_to_load!")

    if n_jets:
        # Make sure the n_jets argument is an integer
        n_jets = int(n_jets)
        chunk_size = int(chunk_size)

        # Check if the chunk size is smaller than n_jets, if yes change it
        chunk_size = chunk_size if chunk_size >= n_jets else n_jets

    if isinstance(filepath, str):
        # Get the paths of the files as a iterable list
        filepaths = glob(filepath)

    elif isinstance(filepath, list):
        filepaths = filepath

    else:
        raise KeyError(f"Given filepath is {type(filepath)}, not a string or a list!")

    # Check if filepaths is empty
    if len(filepaths) == 0:
        raise RuntimeError(
            f"""
            No file found in path {filepath}!
            Check the filepath in your train_config file!
            """
        )

    # Get class_labels variables etc. from global config
    class_label_vars = get_class_label_variables(class_labels)

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
    n_jets_counter = 0

    # Iterate over the files
    for j, file in enumerate(sorted(filepaths, key=natural_keys)):
        # Get the number of available jets in the file
        n_jets_infile = len(h5py.File(file, "r")["/jets"])

        # Check how many times we need to iterate over the file
        # to get all jets. If indices to load is used, this is
        # zero because one specific is loaded
        if not indices_to_load:
            loop_list = range(int(np.ceil(n_jets_infile / chunk_size)))

        else:
            loop_list = [0]

        # Iterate over the file
        for infile_counter in loop_list:
            if not indices_to_load:
                lower_index_counter = infile_counter * chunk_size
                upper_index_counter = (infile_counter + 1) * chunk_size

            else:
                lower_index_counter = indices_to_load[0]
                upper_index_counter = indices_to_load[1]

            # Load the used label variables from file
            with h5py.File(file, "r") as jets:
                for iterator, iter_class_var in enumerate(jet_vars_to_load):
                    if iterator == 0:
                        labels = pd.DataFrame(
                            jets["/jets"][iter_class_var][
                                lower_index_counter:upper_index_counter
                            ],
                            columns=[iter_class_var],
                        )

                    else:
                        labels[iter_class_var] = jets["/jets"][iter_class_var][
                            lower_index_counter:upper_index_counter
                        ]

            # Init new column for string labels
            labels["Umami_string_labels"] = np.zeros_like(labels[class_label_vars[0]])
            labels["Umami_labels"] = np.zeros_like(labels[class_label_vars[0]])

            # Change type of column to string
            labels = labels.astype({"Umami_string_labels": "str"})

            flavour_categories = global_config.flavour_categories

            # Iterate over the classes and add the correct labels to Umami columns
            for class_label in class_labels:
                cuts = flavour_categories[class_label]["cuts"]
                indices_tochange = np.ones_like(labels["Umami_labels"]).astype(bool)
                indices_toremove = get_sample_cuts(jets=labels, cuts=cuts)
                indices_tochange[indices_toremove] = False

                # Add a string description which this class is
                labels["Umami_string_labels"].values[indices_tochange] = class_label

                # Add the right column label to class
                labels["Umami_labels"].values[indices_tochange] = class_labels.index(
                    class_label
                )

            # Define the conditions to remove
            toremove_conditions = labels["Umami_string_labels"] == "0"

            # Get the indices of the jets that are not used
            indices_toremove = np.where(
                toremove_conditions == True  # pylint: disable=C0121 # noqa: E712
            )[0]

            if cut_vars_dict:
                # Apply cuts and get a list of which jets to remove
                indices_toremove_cuts = get_sample_cuts(jets=labels, cuts=cut_vars_dict)

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(indices_toremove.tolist() + indices_toremove_cuts.tolist())
                    )
                )

            # Load tracks
            trks = np.asarray(
                h5py.File(file, "r")[f"/{tracks_name}"][
                    lower_index_counter:upper_index_counter
                ]
            )

            if len(indices_toremove) != 0:
                # Remove unused jets from labels
                labels = labels.drop(indices_toremove)

                # Delete unused classes and cutted tracks
                trks = np.delete(
                    arr=trks,
                    obj=indices_toremove,
                    axis=0,
                )

            # If not the first file processed, append to the global one
            if j == 0 and infile_counter == 0:
                all_trks = trks
                all_labels = labels["Umami_labels"].values

            # if the first file processed, set as global one
            else:
                all_trks = np.append(all_trks, trks, axis=0)
                all_labels = np.append(all_labels, labels["Umami_labels"].values)

            # Adding the loaded jets to counter
            n_jets_counter += len(trks)

            # Break the loop inside the file if enough jets are loaded
            if indices_to_load is None:
                if n_jets_counter >= n_jets:
                    break

        # Break the loop over the files if enough jets are loaded
        if indices_to_load is None:
            if n_jets_counter >= n_jets:
                break

    if print_logger:
        # Check if enough jets are loaded
        if indices_to_load is None:
            if n_jets_counter < n_jets:
                logger.warning(
                    "Requested %i but only %i could be loaded!",
                    n_jets,
                    n_jets_counter,
                )

            else:
                logger.info("Loaded %i jets!", n_jets)

        else:
            logger.info("Loaded jets for given indices!")

    # Return Trks and labels
    return all_trks[:n_jets], all_labels[:n_jets]
