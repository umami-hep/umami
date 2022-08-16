"""Helper functions for plotting"""
import numpy as np

from umami.configuration import logger, global_config  # isort:skip


def retrieve_truth_label_var_value(class_labels: list) -> tuple:
    """
    Retrieve the truth label variable to use and their values for a given
    list of flavours.

    Parameters
    ----------
    class_labels : list
        List with the classes to retrieve. Like ["bjets", "cjets", "ujets"]

    Returns
    -------
    tuple
        The label dict and the label variabel dict. The first contains the
        value of the given flavour and the latter contain the truth label
        variable which is to use for this specific flavour. Both are dicts.
        For each flavour given in class_labels, a key and a value are given
        in both dicts.
    """

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Get dict with the labels for the flavour
    label_dict = {
        iter_flav: flav_cat[iter_flav]["label_value"] for iter_flav in class_labels
    }

    # Get a dict which truth variable is used for the specified flavour
    label_var_dict = {
        iter_flav: flav_cat[iter_flav]["label_var"] for iter_flav in class_labels
    }

    return (label_dict, label_var_dict)


def translate_kwargs(kwargs):
    """Maintaining backwards compatibility for the kwargs and the new plot_base syntax.

    Parameters
    ----------
    kwargs : dict
        dictionary with kwargs

    Returns
    -------
    dict
        kwargs compatible with new naming.
    """
    mapping = {
        "ApplyATLASStyle": "apply_atlas_style",
        "AtlasTag": "atlas_first_tag",
        "Bin_Width_y_axis": "bin_width_in_ylabel",
        "labelFontSize": "fontsize",
        "legcols": "leg_ncol",
        "legFontSize": "leg_fontsize",
        "loc_legend": "leg_loc",
        "legend_loc": "leg_loc",
        "Log": "logy",
        "n_Leading": "n_leading",
        "ncol": "leg_ncol",
        "normalise": "norm",
        "ratio_cut": ["ymin_ratio_1", "ymax_ratio_1"],
        "Ratio_Cut": ["ymin_ratio_1", "ymax_ratio_1"],
        "SecondTag": "atlas_second_tag",
        "set_logy": "logy",
        "UseAtlasTag": "use_atlas_tag",
        "yAxisIncrease": "y_scale",
    }
    deprecated_args = ["yAxisAtlasTag"]
    for key, elem in mapping.items():
        if key in kwargs:
            # if old naming is used, translate to new naming
            logger.debug("Mapping keyword argument %s -> %s", key, elem)
            if isinstance(elem, str):
                # print warning if old AND new convention are used
                if elem in kwargs:
                    logger.warning(
                        "You specified two keyword arguments which mean the same: "
                        "%s, %s --> using the new naming convention %s",
                        key,
                        elem,
                        elem,
                    )
                else:
                    kwargs[elem] = kwargs[key]

            elif isinstance(elem, list):
                for i, key_new in enumerate(elem):
                    kwargs[key_new] = kwargs[key][i]
            kwargs.pop(key)

    # Remove deprecated arguments from kwargs
    for dep_key in deprecated_args:
        if dep_key in kwargs:
            logger.warning(
                "You specified the argument %s, which is no longer"
                " supported and will be ignored.",
                dep_key,
            )
            kwargs.pop(dep_key)
    return kwargs


def translate_binning(
    binning,
    variable_name: str = None,
):
    """Helper function to translate binning used in some configs to an integer that
    represents the number of bins or an array representing the bin edges

    Parameters
    ----------
    binning : int, list or None
        Binning
    variable_name : str, optional
        Name of the variable. If provided, and the name contains "number", the binning
        will be created such that integer numbers are at the center of the bins,
        by default None

    Returns
    -------
    int or np.ndarray
        Number of bins or array of bin edges

    Raises
    ------
    ValueError
        If unsupported type is provided
    """
    if isinstance(binning, list):
        if len(binning) != 3:
            raise ValueError(
                "The list given for binning has to be of length 3, representing "
                "[x_min, x_max, bin_width]"
            )
        bins = np.arange(binning[0], binning[1], binning[2])
        if variable_name is not None:
            if variable_name.startswith("number"):
                bins = np.arange(binning[0] - 0.5, binning[1] - 0.5, binning[2])

    # If int, set to the given numbers
    elif isinstance(binning, int):
        bins = binning

    # If None, set number of bins to 100
    elif binning is None:
        bins = 100
    else:
        raise ValueError(f"Type {type(binning)} is not supported!")

    return bins
