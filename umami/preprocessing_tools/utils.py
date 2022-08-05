"""Collection of utility functions for preprocessing tools."""
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import label_binarize

from umami.configuration import logger
from umami.tools import yaml_loader


def get_variable_dict(yaml_file: str) -> dict:
    """
    Reads yaml_file containig the variables and exports
    them to a dict.

    Parameters
    ----------
    yaml_file : str
        Input yaml file containing trainig variables

    Returns
    -------
    out_dict : dict
        Dictionary containing training variables
    """
    with open(yaml_file, "r") as conf:
        in_dict = yaml.load(conf, Loader=yaml_loader)
        out_dict = in_dict.copy()

    if "track_train_variables" in out_dict.keys():
        if (
            "noNormVars" in out_dict["track_train_variables"]
            or "logNormVars" in out_dict["track_train_variables"]
            or "jointNormVas" in out_dict["track_train_variables"]
        ):
            del out_dict["track_train_variables"]
            out_dict["track_train_variables"] = {}
            out_dict["track_train_variables"]["tracks"] = in_dict[
                "track_train_variables"
            ]
            logger.warning(
                "'track_train_varibles' should be a nested dictionary. Default tracks"
                "name 'tracks' being used"
            )

    return out_dict


def binarise_jet_labels(
    labels: pd.DataFrame,
    internal_labels: list,
    column: str = "label",
) -> np.ndarray:
    """
    Transforms labels to binary labels

    Parameters
    ----------
    labels : pd.DataFrame or np.ndarray
        Dataframe or array with the labels inside.
    internal_labels : list
        List with the used labels.
    column : str, optional
        Column name of the labels if pd.DataFrame is given,
        by default "label"

    Returns
    -------
    np.ndarray
        containing binary label with shape (len(labels), n_classes)

    Raises
    ------
    TypeError
        If given labels are neither pd.DataFrame nor np.ndarray
    ValueError
        If the given labels are empty
    """

    if isinstance(labels, pd.DataFrame):
        labels = np.array(labels[column].values)

    elif not isinstance(labels, np.ndarray):
        raise TypeError(
            f"Given type {type(labels)} is not supported. Only np.ndarray and"
            " pd.DataFrame"
        )

    if len(labels) == 0:
        raise ValueError("Given labels are empty!")

    for unique_label in np.unique(labels):
        if unique_label not in internal_labels:
            raise ValueError(
                "Given internal labels list does not contain all labels"
                " available in the labels!"
            )

    # Workaround to prevent 1d labels if only two classes are given in class_labels
    internal_labels.append(-1)

    # One hot encode the labels
    labels = label_binarize(
        y=labels,
        classes=internal_labels,
    )[:, :-1]

    # Remove the workaround
    internal_labels.pop()

    return labels
