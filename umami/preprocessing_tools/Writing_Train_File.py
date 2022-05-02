"""Module handling training file writing to disk."""
import json
import os
import pickle

import h5py
import numpy as np
from numpy.lib.recfunctions import repack_fields, structured_to_unstructured
from scipy.stats import binned_statistic_2d

from umami.configuration import logger
from umami.preprocessing_tools import GetVariableDict, preprocessing_plots


class TrainSampleWriter:
    """Class to write training files to disk."""

    def __init__(
        self,
        config: object,
        compression: str = None,
    ) -> None:
        """
        Init the needed configs and variables

        Parameters
        ----------
        config : object
            Loaded config file for the preprocessing.
        compression : str, optional
            Type of compression which should be used, by default None
        """

        self.config = config
        self.bool_use_tracks = config.sampling["options"]["save_tracks"]
        self.tracks_names = self.config.sampling["options"]["tracks_names"]
        self.compression = compression
        self.precision = config.config["precision"]
        self.rnd_seed = 42
        self.variable_config = GetVariableDict(config.var_file)

    def load_generator(
        self,
        input_file: str,
        index: list,
        nJets: int,
        chunk_size: int = 100_000,
    ):
        """
        Set up a generator who loads the scaled file and save it in the format for
        training.

        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        index : list
            List with the indicies.
        nJets : int
            Number of jets used.
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted
            per step, by default 100_000

        Yields
        ------
        jets : np.ndarray
            Yielded jets
        tracks : np.ndarray
            Yielded tracks
        labels : np.ndarray
            Yielded labels
        tracks_labels : np.ndarray
            Yielded track labels
        flavour : np.ndarray
            Yielded flavours
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0

            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunk_size)
                end_ind = min(end_ind, nJets)

                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:

                # Retrieve the slice of indices randomly selected from whole file
                indices_selected = index[index_tuple[0] : index_tuple[1]]
                # Need to sort the indices
                indices_selected = np.sort(indices_selected).astype(int)

                # Load jets
                jets = f["/jets"][indices_selected]
                labels = f["/labels"][indices_selected]
                flavour = f["/flavour"][indices_selected]

                # shuffling the chunk now (prior step still has ordered chunks)
                rng_index = np.arange(len(jets))
                rng = np.random.default_rng(seed=self.rnd_seed)
                rng.shuffle(rng_index)
                jets = jets[rng_index]
                labels = labels[rng_index]
                flavour = flavour[rng_index]

                if self.bool_use_tracks is False:
                    yield jets, labels, flavour

                elif self.bool_use_tracks is True:
                    tracks, track_labels = [], []

                    # Loop over track selections
                    for tracks_name in self.tracks_names:

                        # Load tracks
                        trks = np.asarray(
                            h5py.File(input_file, "r")[f"/{tracks_name}"][
                                indices_selected
                            ],
                            dtype=self.precision,
                        )
                        trks = trks[rng_index]
                        if f"{tracks_name}_labels" in f.keys():
                            trk_labels = np.asarray(
                                h5py.File(input_file, "r")[f"/{tracks_name}_labels"][
                                    indices_selected
                                ]
                            )
                            trk_labels = trk_labels[rng_index]
                        else:
                            trk_labels = None
                        tracks.append(trks)
                        track_labels.append(trk_labels)

                    yield jets, tracks, labels, track_labels, flavour

    def better_shuffling(
        self,
        thearray: np.ndarray,
        nJets: int,
        slice_size: int = int(1e4),
    ) -> np.ndarray:
        """
        Shuffles the index list with fixed slices.

        Parameters
        ----------
        thearray : np.ndarray
            Input array with the values to shuffle.
        nJets : int
            Number of jets in the array
        slice_size : int, optional
            How much values are shuffeld at one, by default int(1e4)

        Returns
        -------
        np.ndarray
            Shuffeld input array.
        """

        missing = slice_size - nJets % slice_size
        adding = np.asarray([np.nan] * missing)
        thearray = np.concatenate([thearray, adding])
        thearray = thearray.reshape((-1, slice_size))
        rng = np.random.default_rng(seed=self.rnd_seed)
        rng.shuffle(thearray)
        thearray = thearray.reshape((-1))

        # Remove the nans that were introduced and return
        return thearray[~np.isnan(thearray)]

    def WriteTrainSample(
        self,
        input_file: str = None,
        output_file: str = None,
        chunk_size: int = 100_000,
    ) -> None:
        """
        Write the training file.

        Parameters
        ----------
        input_file : str, optional
            File with scaled/shifted jets. Default is name from
            config + resampled_scaled, by default None
        output_file : str, optional
            Name of the output file. Default is name from
            config + resampled_scaled_shuffled., by default None
        chunk_size : int, optional
            The number of jets which are loaded and written per step,
            by default 100_000
        """

        # Get the input files for writing/merging
        if input_file is None:
            input_file = self.config.GetFileName(option="resampled_scaled")

        # Define outfile name
        if output_file is None:
            out_file = self.config.GetFileName(option="resampled_scaled_shuffled")
        if self.config.sampling["options"]["bool_attach_sample_weights"]:
            file_name = (
                self.config.config["parameters"]["sample_path"] + "/flavour_weights"
            )
            with open(file_name, "rb") as file:
                weights_dict = pickle.load(file)

        # Extract the correct variables
        variables_header_jets = self.variable_config["train_variables"]
        jets_variables = [
            i for j in variables_header_jets for i in variables_header_jets[j]
        ]

        # Get the max length of the input file
        n_jets = len(h5py.File(input_file, "r")["/jets"])

        # Get the number of chunks that need to be processed
        n_chunks = int(np.ceil(n_jets / chunk_size))

        # Create an absolute index list for the file and shuffle it
        absolute_index = np.arange(n_jets)
        absolute_index = self.better_shuffling(absolute_index, n_jets)

        load_generator = self.load_generator(
            input_file=input_file,
            index=absolute_index,
            nJets=n_jets,
            chunk_size=chunk_size,
        )

        logger.info(f"Saving final train files to {out_file}")
        logger.info(f"Using precision: {self.precision}")
        logger.info(f"Using compression: {self.compression}")

        with h5py.File(out_file, "w") as h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            jet_idx = 0

            while chunk_counter <= n_chunks:
                logger.info(f"Writing chunk {chunk_counter+1} of {n_chunks+1}.")
                try:
                    # Load jets from file
                    if self.bool_use_tracks is False:
                        jets, labels, flavour = next(load_generator)

                    else:
                        (jets, tracks, labels, track_labels, flavour) = next(
                            load_generator
                        )

                    # final absolute jet index of this chunk
                    jet_idx_end = jet_idx + len(jets)

                    if self.config.sampling["options"]["bool_attach_sample_weights"]:
                        self.calculateWeights(weights_dict, jets, labels)

                    weights = jets["weight"]

                    # Reform jets to unstructured arrays
                    jets = repack_fields(jets[jets_variables])
                    jets = structured_to_unstructured(jets)

                    if chunk_counter == 0:
                        h5file.create_dataset(
                            "X_train",
                            compression=None,
                            dtype=self.precision,
                            shape=(n_jets, jets.shape[1]),
                        )

                        h5file.create_dataset(
                            "Y_train",
                            compression=None,
                            dtype=np.uint8,
                            shape=(n_jets, labels.shape[1]),
                        )

                        h5file.create_dataset(
                            "flavour",
                            compression=None,
                            dtype=np.uint8,
                            shape=(n_jets,),
                        )

                        h5file.create_dataset(
                            "weight",
                            compression=None,
                            dtype=self.precision,
                            shape=(n_jets,),
                        )

                    if chunk_counter == 0 and self.bool_use_tracks is True:
                        for i, tracks_name in enumerate(self.tracks_names):
                            chunks = (
                                (1,) + tracks[i].shape[1:] if self.compression else None
                            )
                            h5file.create_dataset(
                                f"X_{tracks_name}_train",
                                compression=self.compression,
                                chunks=chunks,
                                dtype=self.precision,
                                shape=(
                                    n_jets,
                                    tracks[i].shape[1],
                                    tracks[i].shape[2],
                                ),
                            )
                            if track_labels[i] is not None:
                                h5file.create_dataset(
                                    f"Y_{tracks_name}_train",
                                    compression=None,
                                    dtype=np.int8,
                                    shape=(
                                        n_jets,
                                        track_labels[i].shape[1],
                                        track_labels[i].shape[2],
                                    ),
                                )

                    # Jet inputs
                    h5file["X_train"][jet_idx:jet_idx_end] = jets

                    # One-hot flavour labels
                    h5file["Y_train"][jet_idx:jet_idx_end] = labels

                    # flavour int
                    h5file["flavour"][jet_idx:jet_idx_end] = flavour

                    # Weights
                    h5file["weight"][jet_idx:jet_idx_end] = weights

                    # Appending tracks if used
                    if self.bool_use_tracks is True:

                        # Loop over tracks selections
                        for i, tracks_name in enumerate(self.tracks_names):
                            # Track inputs
                            h5file[f"X_{tracks_name}_train"][
                                jet_idx:jet_idx_end
                            ] = tracks[i]

                            if track_labels[i] is not None:
                                # Track labels
                                h5file[f"Y_{tracks_name}_train"][
                                    jet_idx:jet_idx_end
                                ] = track_labels[i]

                except StopIteration:
                    break

                # increment counters
                chunk_counter += 1
                jet_idx = jet_idx_end

        # Plot the variables from the output file of the resampling process
        if (
            "njets_to_plot" in self.config.sampling["options"]
            and self.config.sampling["options"]["njets_to_plot"]
        ):
            preprocessing_plots(
                sample=self.config.GetFileName(option="resampled_scaled_shuffled"),
                var_dict=self.variable_config,
                class_labels=self.config.sampling["class_labels"],
                plots_dir=os.path.join(
                    self.config.config["parameters"]["file_path"],
                    "plots/resampling_scaled_shuffled/",
                ),
                track_collection_list=self.config.sampling["options"]["tracks_names"]
                if "tracks_names" in self.config.sampling["options"]
                and "save_tracks" in self.config.sampling["options"]
                and self.config.sampling["options"]["save_tracks"] is True
                else None,
                nJets=self.config.sampling["options"]["njets_to_plot"],
            )

    def calculateWeights(
        self,
        weights_dict: dict,
        jets: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Finds the according weight for the jet, with the weights calculated
        from the GetFlavorWeights method. Writes it onto the jets["weight"].

        Parameters
        ---------
        weights_dict : dict
            weights_dict per flavor and some additional info written into a
            pickle file at /hybrids/flavour_weights

            - 'bjets', etc.
            - 'bins_x' : pt bins
            - 'bins_y' : eta bins
            - 'bin_indices_flat' : flattened indices of the bins in the histogram
            - 'label_map' : {0: 'ujets', 1: 'cjets', 2: 'bjets'}

        jets : np.ndarray
            Containing values of jet variables
        labels : np.ndarray
            Binarized truth value of flavor for jet with shape
            (nJets x (nFlavor x 1))
        """
        # scale to original values for binning
        with open(self.config.dict_file, "r") as infile:
            jets_scale_dict = json.load(infile)
        for elem in jets_scale_dict["jets"]:
            if elem["name"] == "pt_btagJes":
                jets[elem["name"]] *= elem["scale"]
                jets[elem["name"]] += elem["shift"]
            if elem["name"] == "absEta_btagJes":
                jets[elem["name"]] *= elem["scale"]
                jets[elem["name"]] += elem["shift"]

        # Get binnumber of jet from 2D pt,eta grid
        _, _, _, binnumbers = binned_statistic_2d(
            x=jets["pt_btagJes"],
            y=jets["absEta_btagJes"],
            values=jets["pt_btagJes"],
            statistic="count",
            bins=[
                weights_dict["bins_x"],
                weights_dict["bins_y"],
            ],
        )

        # transfrom labels into "bjets", "cjets"...
        label_keys = [weights_dict["label_map"][np.argmax(label)] for label in labels]
        for i, binnumber in enumerate(binnumbers):
            # look where in flattened 2D bin array this binnumber is
            index = np.where(weights_dict["bin_indices_flat"] == binnumber)
            # extract weight with flavour key and index
            weight = weights_dict[label_keys[i]][index]
            # if its out of the defined bin bounds, default to 1
            if not weight:
                weight = 1
            jets["weight"][i] = weight
