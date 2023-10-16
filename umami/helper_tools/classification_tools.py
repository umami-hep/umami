"""Provides methods for classification of events by flavour."""

from umami.configuration import global_config
from umami.tools.tools import flatten_list


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
    """

    # Get the global_config
    flavour_categories = global_config.flavour_categories

    # Init new lists
    label_var_list = []

    for class_label in class_labels:
        for cut_list in flavour_categories[class_label]["cuts"]:
            for cut in cut_list:
                label_var_list.append(cut)

    # Flatten the lists if needed
    label_var_list = flatten_list(label_var_list)
    return label_var_list


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
