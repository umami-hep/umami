"""Module to define sample cuts."""
import operator
import re
from functools import reduce

import numpy as np


def GetSampleCuts(jets: np.ndarray, cuts: list) -> np.ndarray:
    """
    Given an array of jets and a list of cuts, the function provides a list of
    indices which are removed by applying the cuts.
    Users can define the cuts either via the variable name and logical operators:
    ==, !=, >, >=, <, <= or using the dedicated modulo operator.

    The latter is defined as follows:
    mod_[N]_[operator] with
    - [N] denoting "modulo N "
    - [operator] denoting operator used for comparison to condition

    Parameters
    ----------
    jets : np.ndarray
        Array of jets which need to pass certain cuts.
    cuts : list
        List from config file which contains dict objects for individual cuts.

    Returns
    -------
    indices_to_remove : np.ndarray
        Numpy array of indices to be removed by the cuts

    Raises
    ------
    KeyError
        If the cut object in the list is not a dict with one entry.
    RuntimeError
        If the modulo is incorrectly used.
    RuntimeError
        If the modulo is incorrectly used. Operation is not supported.
    KeyError
        If unsupported operator is provided.
    """

    # define operator dict to be able to call them via string from config
    inverted_ops = {
        "==": operator.ne,
        "!=": operator.eq,
        ">": operator.le,
        ">=": operator.lt,
        "<": operator.ge,
        "<=": operator.gt,
    }
    cut_rejections = []

    # flatten list of cuts in case these cuts are provided as lists inside of a list
    clean_list = []
    for sublist in cuts:
        if isinstance(sublist, list):
            for item in sublist:
                clean_list.append(item)
        else:
            clean_list.append(sublist)
    cuts = clean_list

    for cut_entry in cuts:
        if cut_entry is None:
            continue

        # expect a dictionary with only one entry
        cut = list(cut_entry.keys())
        if len(cut) != 1:
            raise KeyError(
                "The cut object is expected to be a dictionary with one entry."
            )
        cut = cut[0]
        properties = cut_entry[cut]
        op = properties["operator"]
        cond = properties["condition"]
        NaNCheck = properties.get("NaNcheck", False)
        # modulo operation: assume structure mod_[N]_[operator]
        # - [N] denoting "modulo N "
        # - [operator] denoting operator used for comparison to condition
        if "mod_" in op:
            try:
                found = re.search(r"mod_(\d+?)_([=!><]+)", op)
                modulo = int(found.group(1))
                op = found.group(2)
            except AttributeError as Error:
                raise RuntimeError(
                    "Incorrect use of modulo cut for sample:                  "
                    "   specify in config as mod_N_op                     with"
                    " N as an integer and                     op the operator"
                    " used for testing the condition."
                ) from Error
            except KeyError as Error:
                raise RuntimeError(
                    "Incorrect use of modulo cut for sample:                 "
                    "    only supported operators 'op' in mod_N_op are:      "
                    f"               {list(inverted_ops.keys())}."
                ) from Error
            cut_rejection = inverted_ops[op]((jets[cut] % modulo), cond)
        else:
            if op in list(inverted_ops.keys()):  # pylint: disable=C0201:
                if isinstance(cond, list):
                    indices = [inverted_ops[op](jets[cut], cond_i) for cond_i in cond]
                    cut_rejection = reduce(operator.and_, indices)
                else:
                    cond = float(cond)
                    cut_rejection = inverted_ops[op](jets[cut], cond)
                    if NaNCheck:
                        cut_rejection = cut_rejection & (jets[cut] == jets[cut])
            else:
                raise KeyError(
                    "Only supported operators are:                    "
                    f" {list(inverted_ops.keys())}."
                )
        cut_rejections.append(cut_rejection)

    indices_to_remove = np.where(reduce(operator.or_, cut_rejections, False))[0]
    del cut_rejections

    return indices_to_remove


def GetCategoryCuts(label_var: str, label_value: float) -> list:
    """
    This function returns the cut object for the categories used in the
    preprocessing.

    Parameters
    ----------
    label_var : str
        Name of the variable.
    label_value : float, int, list
        Value for the cut of the variable.

    Returns
    -------
    list
        List with the cut objects inside.

    Raises
    ------
    ValueError
        If label_value is not a float, int or a list.
    """

    cut_object = []
    if isinstance(label_value, (float, int, list)):
        cut_object.append(
            {
                label_var: {
                    "operator": "==",
                    "condition": label_value,
                }
            }
        )
    else:
        raise ValueError(
            "The 'label_value' in the global config has a not allowed type."
            " Should be either an integer or a list of integers."
        )

    return cut_object
