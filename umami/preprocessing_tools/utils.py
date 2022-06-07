"""Collection of utility functions for preprocessing tools."""

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer

from umami.configuration import logger
from umami.tools import yaml_loader


def GetVariableDict(yaml_file: str) -> dict:
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


def GetBinaryLabels(
    df: pd.DataFrame,
    column: str = "label",
) -> np.ndarray:
    """
    Transforms labels to binary labels

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the labels inside.
    column : str, optional
        Label name to be used to binarise, by default "label"

    Returns
    -------
    np.ndarray
        containing binary label with shape (len(df), n_classes)
    """

    lb = LabelBinarizer()
    if isinstance(df, np.ndarray):
        return lb.fit_transform(df)

    labels = np.array(df[column].values)
    return lb.fit_transform(labels)
