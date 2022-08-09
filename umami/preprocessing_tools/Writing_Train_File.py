"""Module handling training file writing to disk."""
import json
import os
import pickle

import h5py
import numpy as np
from numpy.lib.recfunctions import (
    append_fields,
    repack_fields,
    structured_to_unstructured,
)
from scipy.stats import binned_statistic_2d

from umami.configuration import logger
from umami.plotting_tools import preprocessing_plots
from umami.preprocessing_tools import get_variable_dict
from umami.preprocessing_tools.Scaling import apply_scaling_jets, apply_scaling_trks


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

        self.save_tracks = (
            config.sampling["options"]["save_tracks"]
            if "save_tracks" in config.sampling["options"].keys()
            else False
        )
        self.save_track_labels = (
            config.sampling["options"]["save_track_labels"]
            if "save_track_labels" in config.sampling["options"].keys()
            else False
        )
        self.track_label_variables = (
            config.sampling["options"]["track_truth_variables"]
            if "track_truth_variables" in config.sampling["options"].keys()
            else None
        )
        self.class_labels = config.sampling["class_labels"]
        self.tracks_names = config.sampling["options"]["tracks_names"]
        self.compression = compression
        self.precision = config.config["precision"]
        self.rnd_seed = 42
        self.variable_config = get_variable_dict(config.var_file)
        self.scale_dict = config.dict_file
        self.sampling_options = config.sampling["options"]

        # Adding the full config to retrieve the correct paths
        self.config = config

    def load_scaled_generator(
        self,
        input_file: str,
        index: list,
        n_jets: int,
        jets_scale_dict: dict,
        tracks_scale_dict: dict = None,
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
        n_jets : int
            Number of jets used.
        jets_scale_dict : dict
            Scale dict of the jet variables with the values inside.
        tracks_scale_dict : dict, optional
            Scale dict of the track variables., by default None
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
            while start_ind < n_jets:
                end_ind = int(start_ind + chunk_size)
                end_ind = min(end_ind, n_jets)

                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:

                # Retrieve the slice of indices randomly selected from whole file
                indices_selected = index[index_tuple[0] : index_tuple[1]]

                # Need to sort the indices
                indices_selected = np.sort(indices_selected).astype(int)

                # Get the jet variables from the variable config
                variables_header_jets = self.variable_config["train_variables"]
                jets_variables = [
                    i for j in variables_header_jets for i in variables_header_jets[j]
                ]

                # Check if weights are available in the resampled file
                if (
                    "weight" in list(f["/jets"].dtype.fields.keys())
                    and "weight" not in jets_variables
                ):
                    jets_variables += ["weight"]

                # Load jets
                jets = f["/jets"].fields(jets_variables)[indices_selected]
                labels = f["/labels"][indices_selected]

                # keep the jet flavour
                flavour = f["/jets"].fields([self.variable_config["label"]])[
                    indices_selected
                ]

                # If no weights are available, init ones as weights
                if "weight" not in jets_variables:
                    length = n_jets if n_jets < chunk_size else len(jets)
                    jets = append_fields(
                        jets, "weight", np.ones(int(length)), dtypes="<i8"
                    )

                # shuffling the chunk now (prior step still has ordered chunks)
                rng_index = np.arange(len(jets))
                rng = np.random.default_rng(seed=self.rnd_seed)
                rng.shuffle(rng_index)
                jets = jets[rng_index]
                labels = labels[rng_index]
                flavour = flavour[rng_index]

                # Apply the scaling for the jet variables
                jets = apply_scaling_jets(
                    jets=jets,
                    variables_list=jets_variables,
                    scale_dict=jets_scale_dict,
                )

                if self.save_tracks is False:
                    yield jets, labels, flavour

                elif self.save_tracks is True:
                    tracks, track_labels = [], []

                    # Loop over track selections
                    for tracks_name in self.tracks_names:

                        # Get the tracks scale dict
                        trk_scale_dict = tracks_scale_dict[tracks_name]

                        # Load tracks
                        trks = np.asarray(
                            h5py.File(input_file, "r")[f"/{tracks_name}"][
                                indices_selected
                            ]
                        )
                        trks = trks[rng_index]

                        # Apply scaling to the tracks
                        trks, trk_labels = apply_scaling_trks(
                            trks=trks,
                            variable_config=self.variable_config,
                            scale_dict=trk_scale_dict,
                            tracks_name=tracks_name,
                            save_track_labels=self.save_track_labels,
                            track_label_variables=self.track_label_variables,
                        )

                        tracks.append(trks)
                        track_labels.append(trk_labels)

                    yield jets, tracks, labels, track_labels, flavour

    def better_shuffling(
        self,
        thearray: np.ndarray,
        n_jets: int,
        slice_size: int = int(1e4),
    ) -> np.ndarray:
        """
        Shuffles the index list with fixed slices.

        Parameters
        ----------
        thearray : np.ndarray
            Input array with the values to shuffle.
        n_jets : int
            Number of jets in the array
        slice_size : int, optional
            How much values are shuffeld at one, by default int(1e4)

        Returns
        -------
        np.ndarray
            Shuffeld input array.
        """

        missing = slice_size - n_jets % slice_size
        adding = np.asarray([np.nan] * missing)
        thearray = np.concatenate([thearray, adding])
        thearray = thearray.reshape((-1, slice_size))
        rng = np.random.default_rng(seed=self.rnd_seed)
        rng.shuffle(thearray)
        thearray = thearray.reshape((-1))

        # Remove the nans that were introduced and return
        return thearray[~np.isnan(thearray)]

    def write_train_sample(
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
            input_file = self.config.get_file_name(option="resampled")

        # Define outfile name
        if output_file is None:
            out_file = self.config.get_file_name(option="resampled_scaled_shuffled")

        if self.sampling_options["bool_attach_sample_weights"]:
            file_name = (
                self.config.config["parameters"]["sample_path"] + "/flavour_weights"
            )
            with open(file_name, "rb") as file:
                weights_dict = pickle.load(file)

        # Get scale dict
        with open(self.scale_dict, "r") as infile:
            jets_scale_dict = json.load(infile)["jets"]

        # Check if tracks are used
        if self.save_tracks:
            tracks_scale_dict = {}

            # Get the scale dict for tracks
            with open(self.scale_dict, "r") as infile:
                full_scale_dict = json.load(infile)
                for tracks_name in self.tracks_names:
                    tracks_scale_dict[tracks_name] = full_scale_dict[f"{tracks_name}"]

        else:
            tracks_scale_dict = None

        # Extract the correct variables (without weights)
        jets_variables = [
            var
            for var_group in self.variable_config["train_variables"]
            for var in self.variable_config["train_variables"][var_group]
        ]

        # Get the max length of the input file
        n_jets = len(h5py.File(input_file, "r")["/jets"])

        # Get the number of chunks that need to be processed
        n_chunks = int(np.ceil(n_jets / chunk_size))

        # Create an absolute index list for the file and shuffle it
        absolute_index = np.arange(n_jets)
        absolute_index = self.better_shuffling(absolute_index, n_jets)

        # Init the generator which loads/shuffles/scales the jet/track variables
        load_generator = self.load_scaled_generator(
            input_file=input_file,
            index=absolute_index,
            n_jets=n_jets,
            jets_scale_dict=jets_scale_dict,
            tracks_scale_dict=tracks_scale_dict,
            chunk_size=chunk_size,
        )

        logger.info("Saving final train files to %s", out_file)
        logger.info("Using precision: %s", self.precision)
        logger.info("Using compression: %s", self.compression)

        with h5py.File(out_file, "w") as h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            jet_idx = 0

            while chunk_counter <= n_chunks:
                logger.info("Writing chunk %i of %i", chunk_counter + 1, n_chunks + 1)
                try:
                    # Load jets from file
                    if self.save_tracks is False:
                        jets, labels, flavour = next(load_generator)

                    else:
                        (
                            jets,
                            tracks,
                            labels,
                            track_labels,
                            flavour,
                        ) = next(load_generator)

                    # final absolute jet index of this chunk
                    jet_idx_end = jet_idx + len(jets)

                    if self.sampling_options["bool_attach_sample_weights"]:
                        self.calculate_weights(weights_dict, jets, labels)

                    weights = jets["weight"]

                    # Reform jets to unstructured arrays
                    jets = repack_fields(jets[jets_variables])
                    jets = structured_to_unstructured(jets)

                    # Reform the flavour to unstructured array
                    flavour = repack_fields(flavour[self.variable_config["label"]])

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

                        # Writing jet variables as attributes to X_train
                        h5file["X_train"].attrs["jet_variables"] = jets_variables
                        h5file["Y_train"].attrs["label_classes"] = self.class_labels
                        h5file["flavour"].attrs["label_classes"] = self.class_labels

                    if chunk_counter == 0 and self.save_tracks is True:
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

                            if self.save_track_labels:
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

                                # Add the track truth variables as attribute to the
                                # track labels dataset
                                h5file[f"Y_{tracks_name}_train"].attrs[
                                    tracks_name + "_truth_variables"
                                ] = self.track_label_variables

                            # Extract the track variables used for training from the
                            # var config and write them as attribute to the tracks
                            # dataset
                            h5file[f"X_{tracks_name}_train"].attrs[
                                tracks_name + "_variables"
                            ] = [
                                trk_var
                                for trk_header in self.variable_config[
                                    "track_train_variables"
                                ][tracks_name]
                                for trk_var in self.variable_config[
                                    "track_train_variables"
                                ][tracks_name][trk_header]
                            ]

                    # Jet inputs
                    h5file["X_train"][jet_idx:jet_idx_end] = jets

                    # One-hot flavour labels
                    h5file["Y_train"][jet_idx:jet_idx_end] = labels

                    # flavour int
                    h5file["flavour"][jet_idx:jet_idx_end] = flavour

                    # Weights
                    h5file["weight"][jet_idx:jet_idx_end] = weights

                    # Appending tracks if used
                    if self.save_tracks is True:

                        # Loop over tracks selections
                        for i, tracks_name in enumerate(self.tracks_names):
                            # Track inputs
                            h5file[f"X_{tracks_name}_train"][
                                jet_idx:jet_idx_end
                            ] = tracks[i]

                            if self.save_track_labels:
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
            "n_jets_to_plot" in self.sampling_options
            and self.sampling_options["n_jets_to_plot"]
        ):
            logger.info("Plotting prepared training dataset distributions...")
            preprocessing_plots(
                sample=self.config.get_file_name(option="resampled_scaled_shuffled"),
                var_dict=self.variable_config,
                class_labels=self.config.sampling["class_labels"],
                plots_dir=os.path.join(
                    self.config.config["parameters"]["file_path"],
                    "plots/resampling_scaled_shuffled/",
                ),
                track_collection_list=self.sampling_options["tracks_names"]
                if "tracks_names" in self.sampling_options
                and "save_tracks" in self.sampling_options
                and self.sampling_options["save_tracks"] is True
                else None,
                n_jets=self.sampling_options["n_jets_to_plot"],
                atlas_second_tag=self.config.plot_sample_label,
                logy=True,
                ylabel="Normalised number of jets",
            )

    def calculate_weights(
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
            (n_jets x (nFlavor x 1))
        """
        # scale to original values for binning
        with open(self.scale_dict, "r") as infile:
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
