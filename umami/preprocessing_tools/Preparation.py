"""
Helper functions to creating hybrid hdf5 samples from ttbar and Zprime ntuples
"""
import logging

import h5py
import numpy as np


def get_jets(
    filename,
    n_jets,
    sample_type,
    sample_category=None,
    eventnumber_parity=None,
    write_tracks=False,
    pt_cut=None,
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
    :param eventnumber_parity: choose to select only jets with associated
                               event number which is either 'even' or 'odd',
                               if not specified select all jets
    :param write_tracks: also store track information (true/false)
    :param pt_cut: specify pt cut used as threshold in hybrid sample creation
                   from ttbar and Zprime samples
                   (no cut applied, if not specified)
    :returns: (jets, tracks), where jets is a numpy array of jets.
              Similarly, tracks is a numpy array of tracks but is only created
              if the write_tracks parameter is set to True.
    """
    b_pdgid = 5

    logging.debug("Opening file " + filename)
    data_set = h5py.File(filename, "r")
    jets = data_set["jets"]
    logging.debug(f"Total number of jets in file: {jets.size}")
    if write_tracks:
        tracks = data_set["tracks"]
        logging.debug(f"Total number of tracks in file: {tracks.size}")

    # define event number parity rejection
    # (for splitting data in training and testing sets)
    if eventnumber_parity == "even":
        parity_rejection = (jets["eventNumber"] % 2) == 1
    elif eventnumber_parity == "odd":
        parity_rejection = (jets["eventNumber"] % 2) == 0
    else:
        parity_rejection = False

    # define category rejection (only relevant for ttbar sample)
    if sample_type == "ttbar" and sample_category == "bjets":
        category_rejection = jets["HadronConeExclTruthLabelID"] != b_pdgid
    elif sample_type == "ttbar" and sample_category == "cjets":
        category_rejection = jets["HadronConeExclTruthLabelID"] != 4
    elif sample_type == "ttbar" and sample_category == "ujets":
        category_rejection = jets["HadronConeExclTruthLabelID"] != 0
    elif sample_type == "ttbar" and sample_category == "taujets":
        category_rejection = jets["HadronConeExclTruthLabelID"] != 15
    else:
        category_rejection = False

    # define pt cut rejection
    if sample_type == "ttbar" and pt_cut:
        pt_cut_rejection = (
            (abs(jets["HadronConeExclTruthLabelID"]) == b_pdgid)
            & (jets["GhostBHadronsFinalPt"] > pt_cut)
        ) | (
            (abs(jets["HadronConeExclTruthLabelID"]) != b_pdgid)
            & (jets["pt_btagJes"] > pt_cut)
        )
    elif sample_type == "zprime" and pt_cut:
        pt_cut_rejection = (
            (abs(jets["HadronConeExclTruthLabelID"]) == b_pdgid)
            & (jets["GhostBHadronsFinalPt"] < pt_cut)
        ) | (
            (abs(jets["HadronConeExclTruthLabelID"]) != b_pdgid)
            & (jets["pt_btagJes"] < pt_cut)
        )
    else:
        pt_cut_rejection = False

    indices_to_remove = np.where(
        parity_rejection | category_rejection | pt_cut_rejection
    )[0]

    del parity_rejection
    del category_rejection
    del pt_cut_rejection
    jets = np.delete(jets, indices_to_remove)[:n_jets]
    jets = jets[:n_jets]
    if write_tracks:
        tracks = np.delete(tracks, indices_to_remove, axis=0)[:n_jets]
        tracks = tracks[:n_jets]
        return jets, tracks
    else:
        return jets, None
