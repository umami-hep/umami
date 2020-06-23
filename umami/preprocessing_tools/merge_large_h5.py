"""
Helper functions to merge hdf5 (big) files
"""
import h5py


def check_size(data):
    """Check if #entries is the same for all keys and return it

    Keyword arguments:
    data -- data dictionary
    """

    sizes = [d.shape[0] for d in data.values()]

    if max(sizes) != min(sizes):
        raise ValueError("Each dataset within a file must have the "
                         "same number of entries!")

    return sizes[0]


def get_size(filelist):

    """Get total size of datasets; return size and ranges per file.

    Keyword arguments:
    filelist -- the list of input files
    """

    total_size = 0
    ranges = {}

    for f in filelist:
        data = h5py.File(f, 'r')
        size = check_size(data)
        ranges[f] = [total_size, total_size + size]
        total_size = total_size + size
        data.close()

    return total_size, ranges


def create_datasets(output, source, size):

    """Prepare datasets for merged file based on dictionary.

    Keyword argument:
    output -- output merged hdf5 file
    source -- dict with arrays to save per key
    size -- total number of entries per dataset
    """

    for key in source:
        shape = list(source[key].shape)
        shape[0] = size
        output.create_dataset(key, shape, dtype=source[key].dtype,
                              compression='gzip')
