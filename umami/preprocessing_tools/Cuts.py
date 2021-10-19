import operator
import re
from functools import reduce

import numpy as np


def GetSampleCuts(jets, cuts):
    """Given an array of jets and a list of cuts, the function provides a list of indices
       which are removed by applying the cuts.
       Users can define the cuts either via the variable name and logical operators:
       ==, !=, >, >=, <, <= or using the dedicated modulo operator.

       The latter is defined as follows:
       mod_[N]_[operator] with
       - [N] denoting "modulo N "
       - [operator] denoting operator used for comparison to condition

    Args:
        jets (array of jets): array of jets which need to pass certain cuts
        cuts (list): list from config file which contains dict objects for individual cuts

    Returns:
        indices_to_remove: numpy array of indices to be removed by the cuts
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

    for cut_entry in cuts:
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
            except AttributeError:
                raise RuntimeError(
                    "Incorrect use of modulo cut for sample: \
                    specify in config as mod_N_op \
                    with N as an integer and \
                    op the operator used for testing the condition."
                )
            except KeyError:
                raise RuntimeError(
                    f"Incorrect use of modulo cut for sample: \
                    only supported operators 'op' in mod_N_op are: \
                    {list(inverted_ops.keys())}."
                )
            cut_rejection = inverted_ops[op]((jets[cut] % modulo), cond)
        else:
            if op in list(inverted_ops.keys()):
                if type(cond) is list:
                    indices = [
                        inverted_ops[op](jets[cut], cond_i) for cond_i in cond
                    ]
                    cut_rejection = reduce(operator.and_, indices)
                else:
                    cond = float(cond)
                    cut_rejection = inverted_ops[op](jets[cut], cond)
                    if NaNCheck:
                        cut_rejection = cut_rejection & (
                            jets[cut] == jets[cut]
                        )
            else:
                raise KeyError(
                    f"Only supported operators are: \
                    {list(inverted_ops.keys())}."
                )
        cut_rejections.append(cut_rejection)

    indices_to_remove = np.where(reduce(operator.or_, cut_rejections, False))[
        0
    ]
    del cut_rejections

    return indices_to_remove


def GetCategoryCuts(label_var, label_value):
    """
    This function returns the cut object for the categories used in the
    preprocessing.
    """
    cut_object = []
    if (
        type(label_value) is int
        or type(label_value) is float
        or type(label_value) is list
    ):
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
            "The 'label_value' in the global config has a not allowed type. Should be either an integer or a list of integers."
        )

    return cut_object
