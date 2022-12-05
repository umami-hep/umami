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
from umami.preprocessing_tools import binarise_jet_labels, get_variable_dict
from umami.preprocessing_tools.scaling import (
    apply_scaling_jets,
    apply_scaling_trks,
    as_full,
)


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
        self.jet_vars = sum(self.variable_config["train_variables"].values(), [])
        self.scale_dict = config.dict_file
        self.sampling_options = config.sampling["options"]
        self.validation = config.sampling["use_validation_samples"]

        # Adding the full config to retrieve the correct paths
        self.config = config

        # if set to true, use all jet vars
        self.concat_jet_tracks = config.config.get("concat_jet_tracks", False)
        if isinstance(self.concat_jet_tracks, bool) and self.concat_jet_tracks:
            self.concat_jet_tracks = self.jet_vars
        if self.concat_jet_tracks:
            assert all(i in self.jet_vars for i in self.concat_jet_tracks)

        # some variables which get defined later on
        self.h5file = None
        self.jet_group = None
        self.track_groups = None

    def load_scaled_generator(
        self,
        input_file: str,
        index: list,
        n_jets: int,
        scale_dict: dict,
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
        scale_dict : dict
            Scale dict of the jet and track variables.
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
        valid : np.ndarray
            Yielded valid flag
        flavour : np.ndarray
            Yielded flavours
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as in_file:

            # Get the indices
            start_ind = 0
            tupled_indices = []

            while start_ind < n_jets:
                # Calculate end index of the chunk
                end_ind = int(start_ind + chunk_size)

                # Check if end index is bigger than n_jets
                end_ind = min(end_ind, n_jets)

                # Append to list
                tupled_indices.append((start_ind, end_ind))

                # Set the new start index
                start_ind = end_ind

            for index_tuple in tupled_indices:

                # Retrieve the slice of indices randomly selected from whole file
                indices_selected = index[index_tuple[0] : index_tuple[1]]

                # Need to sort the indices
                indices_selected = np.sort(indices_selected).astype(int)

                # Check if weights are available in the resampled file
                if (
                    "weight" in list(in_file["/jets"].dtype.fields.keys())
                    and "weight" not in self.jet_vars
                ):
                    self.jet_vars += ["weight"]

                # Load jets
                jets = in_file["/jets"].fields(self.jet_vars)[indices_selected]
                labels = in_file["/labels"][indices_selected]
                label_classes = list(range(len(self.class_labels)))
                labels_one_hot = binarise_jet_labels(
                    labels=labels, internal_labels=label_classes
                )

                # Loop over the columns and change all floats to full precision
                for iter_var in self.jet_vars:
                    if jets[iter_var].dtype.kind == "f":
                        jets[iter_var] = jets[iter_var].astype(np.float32)

                # If no weights are available, init ones as weights
                if "weight" not in self.jet_vars:
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
                labels_one_hot = labels_one_hot[rng_index]

                # Apply the scaling for the jet variables
                jets = apply_scaling_jets(
                    jets=jets,
                    variables_list=self.jet_vars,
                    scale_dict=scale_dict["jets"],
                )

                if self.save_tracks is False:
                    tracks, valid, track_labels = None, None, None
                else:
                    tracks, valid, track_labels = [], [], []

                    # Loop over track selections
                    for tracks_name in self.tracks_names:

                        # Retrieving the dtypes of the variables to load
                        to_load_dtype = [
                            (n, as_full(x))
                            for n, x in in_file[f"/{tracks_name}"].dtype.descr
                        ]

                        # Get the tracks scale dict
                        trk_scale_dict = scale_dict[tracks_name]

                        # Load tracks
                        trks = np.asarray(
                            in_file[f"/{tracks_name}"][indices_selected],
                            dtype=to_load_dtype,
                        )
                        trks = trks[rng_index]

                        # Apply scaling to the tracks
                        trks, valid_flag, trk_labels = apply_scaling_trks(
                            trks=trks,
                            variable_config=self.variable_config,
                            scale_dict=trk_scale_dict,
                            tracks_name=tracks_name,
                            save_track_labels=self.save_track_labels,
                            track_label_variables=self.track_label_variables,
                        )

                        tracks.append(trks)
                        valid.append(valid_flag)
                        track_labels.append(trk_labels)

                yield jets, tracks, labels, labels_one_hot, track_labels, valid

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

        # Get the input files for writing/merging, checking for validation is required
        if input_file is None:
            input_file = self.config.get_file_name(
                option="resampled", use_val=self.validation
            )

        # Define outfile name
        if output_file is None:
            out_file = self.config.get_file_name(
                option="resampled_scaled_shuffled", use_val=self.validation
            )

        weights_dict = None
        if self.sampling_options["bool_attach_sample_weights"]:
            file_name = (
                self.config.config["parameters"]["sample_path"] + "/flavour_weights"
            )
            with open(file_name, "rb") as file:
                weights_dict = pickle.load(file)

        # Get scale dict
        with open(self.scale_dict, "r") as infile:
            scale_dict = json.load(infile)

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
            scale_dict=scale_dict,
            chunk_size=chunk_size,
        )

        logger.info("Saving final train files to %s", out_file)
        logger.info("Using precision: %s", self.precision)
        logger.info("Using compression: %s", self.compression)

        with h5py.File(out_file, "w") as self.h5file:

            # Set up chunk counter and start looping
            chunk_counter = 0
            jet_idx = 0

            while chunk_counter <= n_chunks:
                logger.info("Writing chunk %i of %i", chunk_counter + 1, n_chunks + 1)

                try:
                    jet_idx = self.save_chunk(
                        load_generator,
                        chunk_counter,
                        jet_idx,
                        n_jets,
                        weights_dict,
                    )
                except StopIteration:
                    break

                # increment counters
                chunk_counter += 1

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
        for varname, scale_dict in jets_scale_dict["jets"].items():
            if varname == "pt_btagJes":
                jets[varname] *= scale_dict["scale"]
                jets[varname] += scale_dict["shift"]
            if varname == "absEta_btagJes":
                jets[varname] *= scale_dict["scale"]
                jets[varname] += scale_dict["shift"]

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

    def init_jet_datasets(
        self,
        n_jets: int,
        jets: np.ndarray,
        labels_one_hot: np.ndarray,
    ):
        """
        Create jet datasets

        Parameters
        ----------
        n_jets : int
            total number of jets that will be written
        jets : np.ndarray
            jet input feature array
        labels_one_hot : np.ndarray
            jet label array, one hot

        Returns
        -------
        h5py.Group
            h5 group containing jet datasets
        """

        jet_group = self.h5file.create_group("jets")

        jet_group.create_dataset(
            "inputs",
            compression=None,
            dtype=self.precision,
            shape=(n_jets, jets.shape[1]),
        )

        jet_group.create_dataset(
            "labels",
            compression=None,
            dtype=np.uint8,
            shape=(n_jets,),
        )

        jet_group.create_dataset(
            "labels_one_hot",
            compression=None,
            dtype=np.uint8,
            shape=(n_jets, labels_one_hot.shape[1]),
        )

        jet_group.create_dataset(
            "weight",
            compression=None,
            dtype=self.precision,
            shape=(n_jets,),
        )

        # Writing jet variables as attributes to the jet group datasets
        jet_group["inputs"].attrs["jet_variables"] = self.jet_vars
        jet_group["labels_one_hot"].attrs["label_classes"] = self.class_labels
        jet_group["labels"].attrs["label_classes"] = self.class_labels

        return jet_group

    def init_track_datasets(
        self,
        n_jets: int,
        tracks: list,
        valid: list,
        labels: list,
    ):
        """
        Create track-like datasets

        Parameters
        ----------
        n_jets : int
            total number of jets that will be written
        tracks : list
            list of track arrays
        valid : list
            list of track valid arrays
        labels : list
            list of track label arrays

        Returns
        -------
        list
            list of track h5 groups
        """

        track_groups = []

        # for each track-like group
        for i, tracks_name in enumerate(self.tracks_names):

            trk_i = tracks[i]

            # create a group
            track_group = self.h5file.create_group(tracks_name)
            track_groups.append(track_group)

            tracks_shape = (n_jets, trk_i.shape[1], trk_i.shape[2])
            if self.concat_jet_tracks:
                tracks_shape = (
                    n_jets,
                    tracks_shape[1],
                    trk_i.shape[2] + len(self.concat_jet_tracks),
                )

            # chunking is jet-wise for optimal random access
            chunks = (1,) + trk_i.shape[1:] if self.compression else None

            track_group.create_dataset(
                "inputs",
                compression=self.compression,
                chunks=chunks,
                dtype=self.precision,
                shape=tracks_shape,
            )

            track_group.create_dataset(
                "valid", compression=None, dtype=bool, shape=(n_jets, valid[i].shape[1])
            )

            if self.save_track_labels:

                for label in self.track_label_variables:
                    track_group.create_dataset(
                        f"labels/{label}",
                        chunks=(1, trk_i.shape[1]),
                        dtype=np.int8,
                        shape=(n_jets, labels[i].shape[1]),
                    )

            # save track variables as attribute
            track_vars = sum(
                self.variable_config["track_train_variables"][tracks_name].values(), []
            )
            if self.concat_jet_tracks:
                track_vars = [f"jet_{v}" for v in self.concat_jet_tracks] + track_vars
            track_group["inputs"].attrs["variables"] = track_vars

        return track_groups

    def save_chunk(
        self,
        load_generator,
        chunk_counter: int,
        jet_idx: int,
        n_jets: int,
        weights_dict: dict,
    ):
        """Save a single chunk of ready to train data to file

        Parameters
        ----------
        load_generator : Generator
            Yields data
        chunk_counter : int
            Index of the current chunk being written
        jet_idx : int
            Start index for the current chunk
        n_jets : int
            Total number of jets to write
        weights_dict : dict
            Jet weight dictionary

        Returns
        -------
        int
            Stop index for the current chunk
        """

        # Load data
        (
            jets,
            tracks,
            labels,
            labels_one_hot,
            track_labels,
            valid,
        ) = next(load_generator)

        # final absolute jet index of this chunk
        jet_idx_end = jet_idx + len(jets)

        if self.sampling_options["bool_attach_sample_weights"]:
            self.calculate_weights(weights_dict, jets, labels)

        weights = jets["weight"]

        # Reform jets to unstructured arrays
        jets = repack_fields(jets[self.jet_vars])
        jets = structured_to_unstructured(jets)

        if chunk_counter == 0:
            self.jet_group = self.init_jet_datasets(
                n_jets,
                jets,
                labels_one_hot,
            )

        if chunk_counter == 0 and self.save_tracks:
            self.track_groups = self.init_track_datasets(
                n_jets,
                tracks,
                valid,
                track_labels,
            )

        # write jets
        self.jet_group["inputs"][jet_idx:jet_idx_end] = jets
        self.jet_group["labels"][jet_idx:jet_idx_end] = labels
        self.jet_group["labels_one_hot"][jet_idx:jet_idx_end] = labels_one_hot
        self.jet_group["weight"][jet_idx:jet_idx_end] = weights

        # write tracks
        if self.save_tracks is True:

            # Loop over tracks selections
            for i, track_group in enumerate(self.track_groups):
                track_i = tracks[i]

                # concatenate jet and track inputs
                if self.concat_jet_tracks:
                    idxs = [self.jet_vars.index(v) for v in self.concat_jet_tracks]
                    rep_jets = np.repeat(jets[:, None, idxs], track_i.shape[1], axis=1)
                    track_i = np.concatenate([rep_jets, track_i], axis=2)
                    track_i[~valid[i]] = 0

                # Track inputs
                track_group["inputs"][jet_idx:jet_idx_end] = track_i

                # Valid flag
                track_group["valid"][jet_idx:jet_idx_end] = valid[i]

                # Track labels
                if self.save_track_labels:
                    for idx, label in enumerate(self.track_label_variables):
                        track_group[f"labels/{label}"][
                            jet_idx:jet_idx_end
                        ] = track_labels[i][..., idx]

        return jet_idx_end