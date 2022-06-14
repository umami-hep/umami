"""Provides methods for classification of events by flavour."""
from umami.configuration import global_config  # isort:skip
import numpy as np


def get_class_label_ids(class_labels: list) -> list:
    """
    Retrieves the flavour ids of the class_labels provided
    and returns them as a list.

    Parameters
    ----------
    class_labels : list
        List with the class labels.

    Returns
    -------
    id_list : list
        List of the class label ids.
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new list

    for counter, class_label in enumerate(class_labels):
        if counter == 0:
            id_list = np.asarray(flavour_categories[class_label]["label_value"])

        else:
            id_list = np.append(
                id_list,
                np.asarray(flavour_categories[class_label]["label_value"]),
            )

    # Flatten the list if needed and return it
    id_list = id_list.tolist()

    # If only one flavour is given, an int would be returned. This is prevented
    # with this check (so that all loops work properly)
    if isinstance(id_list, int):
        id_list = [id_list]

    return id_list


def get_class_label_variables(class_labels: list):
    """
    Returns a list of the label variables used for the provided class_labels.

    Parameters
    ----------
    class_labels : list
        List with the class labels.

    Returns
    -------
    label_var_list : list
        List with the truth label variables needed for the classes.
    flatten_class_labels : list
        Same shape as label_var_list. List with class labels.
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new lists
    label_var_list = []
    flatten_class_labels = []

    for class_label in class_labels:

        # Check if multiple label values are defined for that flavour
        if isinstance(flavour_categories[class_label]["label_value"], list):

            # If x ids are defined, loop over them and add the
            # truth variable x times to the label_var_list
            n_repeat = len(flavour_categories[class_label]["label_value"])
            # Append the truth variable to the label_var_list
            label_var_list += [flavour_categories[class_label]["label_var"]] * n_repeat
            # Add the class_label to the flatten class list
            flatten_class_labels += [class_label] * n_repeat

        else:
            # Add the label variable and class label to list
            label_var_list.append(flavour_categories[class_label]["label_var"])
            flatten_class_labels.append(class_label)

    # Flatten the lists if needed
    label_var_list = np.asarray(label_var_list).flatten().tolist()
    flatten_class_labels = np.asarray(flatten_class_labels).flatten().tolist()

    return label_var_list, flatten_class_labels


def get_class_prob_var_names(tagger_name: str, class_labels: list):
    """
    Returns a list of the probability variable names used for the
    provided class_labels.

    Parameters
    ----------
    tagger_name : str
        Name of the tagger that is used e.g. dips20210729.
    class_labels : list
        List with the class labels.

    Returns
    -------
    prob_var_list : list
        List with the tagger_name and probabilites merged e.g.
        ["dips20210729_pb", "dips20210729_pc", "dips20210729_pu"].
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new list
    prob_var_list = []

    # Append the prob var names to new list
    for class_label in class_labels:
        prob_var_list.append(
            tagger_name + "_" + flavour_categories[class_label]["prob_var_name"]
        )

    # Return list of prob var names in correct order
    return prob_var_list
