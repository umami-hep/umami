"""Helper tools for helper_tools."""

import itertools

import h5py


def compare_h5_files_variables(*h5_files, key):
    """Compare variable contant of several hdf5 files.

    Parameters
    ----------
    *h5_files : str
        name of hdf5 files to be compared
    key : str
        hdf5 dataset key

    Returns
    -------
    list
        list of variables common to all provided files
    list
        list of variables not common in provided files

    Raises
    ------
    ValueError
        if no input files provided.
    """

    # return None if no positional arguments given
    if not h5_files:
        raise ValueError("No input files provided.")
    files_vars = []
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f_h5:
            files_vars.append(set(f_h5[key][:1].dtype.names))

    # get all possible combinations of input variable sets
    combinations = list(itertools.combinations(files_vars, 2))

    non_common_vars = []
    for elem in combinations:
        non_common_var = elem[0].symmetric_difference(elem[1])
        non_common_vars += list(non_common_var)

    non_common_vars = list(set(non_common_vars))

    common_vars_list = files_vars[0]
    for elem in non_common_vars:
        if elem in common_vars_list:
            common_vars_list.remove(elem)
    common_vars = list(set(common_vars_list))

    return common_vars, non_common_vars
