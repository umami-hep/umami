"""
Helper functions to creating hybrid hdf5 samples from ttbar and Zprime ntuples
"""
import h5py
import numpy as np

from umami.configuration import logger
from umami.preprocessing_tools import GetSampleCuts


def get_jets(
    filename,
    n_jets,
    sample_type,
    sample_category=None,
    write_tracks=False,
    tracks_dset_name=None,
    cuts=None,
    extended_labelling=False,
):
    """Helper function to extract jet and track information from a h5 ntuple.
    The function can be used to create hybrid samples from ttbar
    and Zprime input ntuples. The hybrid samples are created from jets in
    the ttbar sample for b-jet pt below a certain pt threshold and from
    jets in the Zprime for b-jet pt above a certain pt threshold.

    :param filename: path to the h5 ntuple
    :param n_jets: number of jets to be extracted from the ntuple
    :param sample_type: type of the sample (can be either 'ttbar' or 'zprime')
    :param sample_category: relevant for ttbar sample, type of jet flavour
                            (can be 'bjets', 'cjets', 'ujets', or 'taujets')
    :param write_tracks: also store track information (true/false)
    :param cuts: specify a dictionary of cuts to be applied for selecting jets
                 for a respective sample (e.g. event number parity, jet flavor)
                 (no cut applied, if not specified)
    :param extended_labelling: use extended flavour labelling (true/false)
    :returns: (jets, tracks), where jets is a numpy array of jets.
              Similarly, tracks is a numpy array of tracks but is only created
              if the write_tracks parameter is set to True.
    """
    logger.info("Opening file " + filename)

    data_set = h5py.File(filename, "r")
    jets = data_set["jets"]
    logger.info(f"Total number of jets in file: {jets.size}")
    if write_tracks:
        tracks = data_set[tracks_dset_name]
        logger.debug(f"Tracks dataset: {tracks_dset_name}")
        logger.info(f"Total number of tracks in file: {tracks.size}")

    indices_to_remove = GetSampleCuts(jets, cuts, extended_labelling)
    jets = np.delete(jets, indices_to_remove)[:n_jets]
    jets = jets[:n_jets]
    if write_tracks:
        tracks = np.delete(tracks, indices_to_remove, axis=0)[:n_jets]
        tracks = tracks[:n_jets]
        return jets, tracks
    else:
        return jets, None
