"""Provides functions for loading datasets from files."""
from copy import deepcopy
from glob import glob

import h5py
import numpy as np
import pandas as pd

from umami.classification_tools import get_class_label_ids, get_class_label_variables
from umami.configuration import logger
from umami.data_tools.Cuts import GetSampleCuts
from umami.tools import natural_keys


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

    Parameters
    ----------
    filepath : str
        Path to the .h5 file with the jets.
    class_labels : list
        List of class labels which are used.
    nJets : int
        Number of jets to load.
    variables : list
        Variables which are loaded.
    cut_vars_dict : dict
        Variable cuts that are applied when loading the jets.
    print_logger : bool
        Decide if the number of jets loaded from the file is printed.
    chunk_size : int
        Chunk size how much jets are loaded in on go.

    Returns
    -------
    all_jets : pandas.DataFrame
        The jets as numpy ndarray
    all_labels : numpy.ndarray
        The internal class label for each jet. Corresponds with the
        index of the class label in class_labels.

    Raises
    ------
    KeyError
        If filepath is not a list or a string
    RuntimeError
        If no files could be found in filepath
    """

    # Make sure the nJets argument is an integer
    nJets = int(nJets)
    chunk_size = int(chunk_size)

    # Check if the chunk size is smaller than nJets, if yes change it
    chunk_size = chunk_size if chunk_size >= nJets else nJets

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
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(class_labels)

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
                    logger.warning(f"{var} not available in files!")

        # Load variables for cuts if given
        if cut_vars_dict:

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
                        infile_counter * chunk_size : (infile_counter + 1) * chunk_size
                    ]
                )

            else:
                jets = pd.DataFrame(
                    h5py.File(file, "r")["/jets"][
                        infile_counter * chunk_size : (infile_counter + 1) * chunk_size
                    ]
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
                indices_tochange = np.where(jets[class_label_var].values == class_id)

                # Add a string description which this class is
                jets["Umami_string_labels"].values[indices_tochange] = class_label

                # Add the right column label to class
                jets["Umami_labels"].values[indices_tochange] = class_labels.index(
                    class_label
                )

            # Define the conditions to remove
            toremove_conditions = jets["Umami_string_labels"] == "0"

            # Get the indices of the jets that are not used
            indices_toremove = np.where(
                toremove_conditions == True  # pylint: disable=C0121 # noqa: E712
            )[0]

            if cut_vars_dict:
                # Apply cuts and get a list of which jets to remove
                indices_toremove_cuts = GetSampleCuts(jets=jets, cuts=cut_vars_dict)

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(indices_toremove.tolist() + indices_toremove_cuts.tolist())
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

    Parameters
    ----------
    filepath : str
        Path to the .h5 file with the jets.
    class_labels : list
        List of class labels which are used.
    nJets : int
        Number of jets to load.
    cut_vars_dict : dict
        Variable cuts that are applied when loading the jets.
    print_logger : bool
        Decide if the number of jets loaded from the file is printed.
    chunk_size : int
        Chunk size how much jets are loaded in on go.

    Returns
    -------
    all_trks : pandas.DataFrame
        The tracks of the jets as numpy ndarray
    all_labels : numpy.ndarray
        The internal class label for each jet. Corresponds with the
        index of the class label in class_labels.

    Raises
    ------
    KeyError
        If filepath is not a list or a string
    RuntimeError
        If no files could be found in filepath
    """

    # Make sure the nJets argument is an integer
    nJets = int(nJets)
    chunk_size = int(chunk_size)

    # Check if the chunk size is small than nJets, if yes change it
    chunk_size = chunk_size if chunk_size >= nJets else nJets

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
    class_ids = get_class_label_ids(class_labels)
    class_label_vars, flatten_class_labels = get_class_label_variables(class_labels)

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
            labels["Umami_string_labels"] = np.zeros_like(labels[class_label_vars[0]])
            labels["Umami_labels"] = np.zeros_like(labels[class_label_vars[0]])

            # Change type of column to string
            labels = labels.astype({"Umami_string_labels": "str"})

            # Iterate over the classes and add the correct labels to Umami columns
            for (class_id, class_label_var, class_label) in zip(
                class_ids, class_label_vars, flatten_class_labels
            ):
                indices_tochange = np.where(labels[class_label_var].values == class_id)[
                    0
                ]

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
                indices_toremove_cuts = GetSampleCuts(jets=labels, cuts=cut_vars_dict)

                # Combine the indicies to remove lists
                indices_toremove = np.asarray(
                    list(
                        set(indices_toremove.tolist() + indices_toremove_cuts.tolist())
                    )
                )

            # Remove unused jets from labels
            labels = labels.drop(indices_toremove)
            Umami_labels = labels["Umami_labels"].values

            # Load tracks and delete unused classes
            trks = np.delete(
                arr=np.asarray(
                    h5py.File(file, "r")["/tracks"][
                        infile_counter * chunk_size : (infile_counter + 1) * chunk_size
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
