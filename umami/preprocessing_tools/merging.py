"""
Helper functions to merge hdf5 (big) files
"""
import h5py


def check_size(data) -> int:
    """
    Check if #entries is the same for all keys and return it

    Parameters
    ----------
    data : dict
        Array with the entries inside.

    Returns
    -------
    int
        Number of entries

    Raises
    ------
    ValueError
        If not all datasets have the same number of entries.
    """

    sizes = [d.shape[0] for d in data.values()]

    if max(sizes) != min(sizes):
        raise ValueError(
            "Each dataset within a file must have the same number of entries!"
        )

    return sizes[0]


def check_keys(data1, data2) -> bool:
    """
    Check it both files have the same datasets.
    Return True if both have the same datasets or
    raise a ValueError otherwise.

    Parameters
    ----------
    data1 : dict
        current data dictionary
    data2 : dict
        data dictionary to be added

    Returns
    -------
    bool
        True if everything works.

    Raises
    ------
    ValueError
        If Files have different datasets!
    """
    if data1.keys() != data2.keys():
        raise ValueError("Files have different datasets!")
    return True


def check_shapes(data1, data2):
    """
    Check if shapes of datasets are the same.
    Return True if both datasets have the same shapes or
    raise a ValueError otherwise.

    Parameters
    ----------
    data1 : dict
        current data dictionary
    data2 : dict
        data dictionary to be added

    Returns
    -------
    bool
        If shapes are the same, return True, else False

    Raises
    ------
    ValueError
        If shapes are different.
    """
    # datasets must have equal keys, otherwise comparison does not make sense
    check_keys(data1, data2)
    for key in data1.keys():
        if data1[key].shape[1:] != data2[key].shape[1:]:
            raise ValueError(f"Different shapes for dataset: {key}. ")
    return True


def get_size(filelist: list):
    """
    Get total size of datasets; return size and ranges per file.

    Parameters
    ----------
    filelist : list
        List of input files

    Returns
    -------
    total_size : list
        Size of the files.
    ranges : list
        Ranges of the files.
    """

    total_size = 0
    ranges = {}

    for file in filelist:
        data = h5py.File(file, "r")
        size = check_size(data)
        ranges[file] = [total_size, total_size + size]
        total_size = total_size + size
        data.close()

    return total_size, ranges


def create_datasets(output, source: dict, size):
    """
    Prepare datasets for merged file based on dictionary.

    Parameters
    ----------
    output : h5py File
        output merged hdf5 file
    source : dict
        dict with arrays to save per key or path to one input hdf5 file or
        one input hdf5 file
    size : int
        total number of entries per dataset
    """

    # check if 'source' is a dict, otherwise assume it is a path to a hdf5 file
    close_file = False
    if not isinstance(source, dict) and isinstance(source, str):
        source = h5py.File(source, "r")
        close_file = True

    for key in source:
        shape = list(source[key].shape)
        shape[0] = size
        output.create_dataset(key, shape, dtype=source[key].dtype, compression="gzip")
    if close_file:
        source.close()


def add_data(source, output, data_range):
    """
    Add content of "source" to "output" hdf5 file.

    Parameters
    ----------
    source : str, dict
        input hdf5 file path / input hdf5 file / dictionary
    output : h5py File
        output hdf5 file
    data_range : list
        where to save data in output arrays
    """
    # check if 'source' is a dict, otherwise assume it is a path to a hdf5 file
    close_file = False
    if not isinstance(source, dict) and isinstance(source, str):
        source = h5py.File(source, "r")
        close_file = True
    check_keys(source, output)
    check_shapes(source, output)
    for key in source:
        output[key][data_range[0] : data_range[1]] = source[key]
    if close_file:
        source.close()
