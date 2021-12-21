import h5py
import numpy as np
import yaml
import pickle
from numpy.lib.recfunctions import repack_fields, structured_to_unstructured
from scipy.stats import binned_statistic_2d
import json

from umami.configuration import logger
from umami.tools import yaml_loader


class TrainSampleWriter:
    def __init__(self, config, compression=None) -> None:
        """
        Init the needed configs and variables

        Input:
        - config: Loaded config file for the preprocessing.
        - compression: Type of compression which should be used.
                       Default: None
        """
        self.config = config
        self.bool_use_tracks = config.sampling["options"]["save_tracks"]
        self.compression = compression
        self.precision = config.config["precision"]
        self.rnd_seed = 42

        with open(config.var_file, "r") as conf:
            self.variable_config = yaml.load(conf, Loader=yaml_loader)

    def load_generator(
        self, input_file: str, index: int, nJets: int, chunkSize: int = 100_000
    ):
        """
        Set up a generator who loads the scaled file and save it in the format for
        training.

        Input:
        - input_file: File which is to be scaled.
        - nJets: Number of jets used.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Yield: The yielded jets/tracks and labels loaded from file
        """

        # Open the file and load the jets
        with h5py.File(input_file, "r") as f:

            # Get the indices
            start_ind = 0

            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunkSize)
                if end_ind > nJets:
                    end_ind = nJets

                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunkSize)

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
                    # Load tracks
                    trks = np.asarray(
                        h5py.File(input_file, "r")["/tracks"][indices_selected],
                        dtype=self.precision,
                    )
                    trks = trks[rng_index]
                    if "track_labels" in f.keys():
                        track_labels = np.asarray(
                            h5py.File(input_file, "r")["/track_labels"][
                                indices_selected
                            ]
                        )
                        track_labels = track_labels[rng_index]
                    else:
                        track_labels = None

                    yield jets, trks, labels, track_labels, flavour

    def better_shuffling(self, thearray, nJets, slice_size=int(1e4)):
        """
        Shuffles the index list with fixed slices.
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
        chunkSize: int = 100_000,
    ):
        """
        Input:
        - input_file: File with scaled/shifted jets. Default is name from
                        config + resampled_scaled
        - output_file: Name of the output file. Default is name from
                        config + resampled_scaled_shuffled.
        - chunkSize: The number of jets which are loaded and scaled/shifted per step.

        Output:
        - Train File: File which ready for training with the NN's.
        """

        # Get the input files for writing/merging
        if input_file is None:
            input_file = self.config.GetFileName(option="resampled_scaled")

        # Define outfile name
        if output_file is None:
            out_file = self.config.GetFileName(
                option="resampled_scaled_shuffled"
            )
        if self.config.sampling["options"]["bool_attach_sample_weights"]:
            file_name = (
                self.config.config["parameters"]["sample_path"]
                + "/flavour_weights"
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
        n_chunks = int(np.ceil(n_jets / chunkSize))

        # Create an absolute index list for the file and shuffle it
        absolute_index = np.arange(n_jets)
        absolute_index = self.better_shuffling(absolute_index, n_jets)

        load_generator = self.load_generator(
            input_file=input_file,
            index=absolute_index,
            nJets=n_jets,
            chunkSize=chunkSize,
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

                    if self.config.sampling["options"][
                        "bool_attach_sample_weights"
                    ]:
                        self.calculateWeights(weights_dict, jets, labels)

                    weights = jets["weight"]

                    # Reform jets to unstructured arrays
                    jets = repack_fields(jets[jets_variables])
                    jets = structured_to_unstructured(jets)

                    if chunk_counter == 0:
                        h5file.create_dataset(
                            "X_train",
                            compression=self.compression,
                            dtype=self.precision,
                            shape=(n_jets, jets.shape[1]),
                        )

                        h5file.create_dataset(
                            "Y_train",
                            compression=self.compression,
                            dtype=np.uint8,
                            shape=(n_jets, labels.shape[1]),
                        )

                        h5file.create_dataset(
                            "flavour",
                            compression=self.compression,
                            dtype=np.uint8,
                            shape=(n_jets,),
                        )

                        h5file.create_dataset(
                            "weight",
                            compression=self.compression,
                            dtype=self.precision,
                            shape=(n_jets,),
                        )

                        if self.bool_use_tracks is True:
                            h5file.create_dataset(
                                "X_trk_train",
                                compression=self.compression,
                                dtype=self.precision,
                                shape=(
                                    n_jets,
                                    tracks.shape[1],
                                    tracks.shape[2],
                                ),
                            )
                            if track_labels is not None:
                                h5file.create_dataset(
                                    "Y_trk_train",
                                    compression=self.compression,
                                    dtype=np.int8,
                                    shape=(
                                        n_jets,
                                        track_labels.shape[1],
                                        track_labels.shape[2],
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

                        # Track inputs
                        h5file["X_trk_train"][jet_idx:jet_idx_end] = tracks

                        if track_labels is not None:
                            # Track labels
                            h5file["Y_trk_train"][jet_idx:jet_idx_end] = track_labels

                except StopIteration:
                    break

                # increment counters
                chunk_counter += 1
                jet_idx = jet_idx_end

    def calculateWeights(self, weights_dict, jets, labels):
        """
        Finds the according weight for the jet, with the weights calculated
        from the GetFlavorWeights method. Writes it onto the jets["weight"].

        Parameters
        ---------
        weights_dict : dict of callables
            weights_dict per flavor and some additional info written into a
            pickle file at /hybrids/flavour_weights

            - 'bjets', etc.
            - 'bins_x' : pt bins
            - 'bins_y' : eta bins
            - 'bin_indices_flat' : flattened indices of the bins in the histogram
            - 'label_map' : {0: 'ujets', 1: 'cjets', 2: 'bjets'}

        jets : arraylike
            Containing values of jet variables
        labels : arraylike (nJets x (nFlavor x 1))
            Binarized truth value of flavor for jet.
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
        label_keys = [
            weights_dict["label_map"][np.argmax(label)] for label in labels
        ]
        for i, binnumber in enumerate(binnumbers):
            # look where in flattened 2D bin array this binnumber is
            index = np.where(weights_dict["bin_indices_flat"] == binnumber)
            # extract weight with flavour key and index
            weight = weights_dict[label_keys[i]][index]
            # if its out of the defined bin bounds, default to 1
            if not weight:
                weight = 1
            jets["weight"][i] = weight
