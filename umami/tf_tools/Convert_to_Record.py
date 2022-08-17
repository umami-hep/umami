"""Module converting h5 to tf records."""
from umami.configuration import logger  # isort:skip

import json
import os

import h5py
import tensorflow as tf
import tqdm


class h5_to_tf_record_converter:
    """h5 converter to tf records."""

    def __init__(self, config):
        self.config = config
        self.path_h5 = self.config.get_file_name(option="resampled_scaled_shuffled")
        try:
            self.chunk_size = int(config.convert_to_tfrecord["chunk_size"])
            logger.info("Save %i entries in one file", self.chunk_size)

        except (AttributeError, KeyError, ValueError) as chunk_size_no_int:
            try:
                self.chunk_size = config.convert_to_tfrecord["chunk_size"]
                if not isinstance(self.chunk_size, int):
                    raise KeyError from chunk_size_no_int
                logger.info("Save %i entries in one file", self.chunk_size)

            except KeyError:
                logger.warning(
                    "Chunk size for conversion into tf records not set in config"
                    "file. Set to 5000"
                )
                self.chunk_size = 5_000

        self.tracks_name = (
            config.sampling["options"]["tracks_names"]
            if "tracks_names" in config.sampling["options"]
            else None
        )

        self.save_track_labels = (
            config.sampling["options"]["save_track_labels"]
            if "save_track_labels" in config.sampling["options"]
            else False
        )

        self.n_add_vars = (
            config.convert_to_tfrecord["N_add_vars"]
            if "N_add_vars" in config.convert_to_tfrecord
            else None
        )

    def load_h5File_Train(self):
        """
        load the numbers of entries given by the chunk size for the jets,
        tracks and labels from train file.

        Yields
        ------
        X_jets : array_like
            Training jets
        X_trks : array_like
            Training tracks
        Y_jets : array_like
            Training jet labels
        Y_trks : array_like
            Training track labels
        Weights : array_like
            Training weights
        X_Add_Vars : array_like
            Conditional variables for CADS and Umami Cond Att.
        """

        # Open the h5 output file
        with h5py.File(self.path_h5, "r") as hFile:

            # Get the number of jets in the file
            length_dataset = len(hFile["X_train"])
            logger.info(
                "Total length of the dataset is %i. Load %i samples at a time",
                length_dataset,
                self.chunk_size,
            )

            # Get the number of loads that needs to be done
            total_loads = int(length_dataset / self.chunk_size)

            # Ensure that the loads are enough
            if length_dataset % self.chunk_size != 0:
                total_loads += 1

            logger.info("Total number of loading steps is %i", total_loads)
            for i in tqdm.tqdm(range(total_loads)):

                # Get start and end chunk index
                start = i * self.chunk_size
                end = (i + 1) * self.chunk_size

                # Get the jets
                X_jets = hFile["X_train"][start:end]

                # Get the labels
                Y_jets = hFile["Y_train"][start:end]

                # Get the weights
                Weights = hFile["weight"][start:end]

                if self.tracks_name is not None:
                    # Get a list with all tracks inside
                    X_trks = {
                        track_name: hFile[f"X_{track_name}_train"][start:end]
                        for track_name in self.tracks_name
                    }

                    if self.save_track_labels:
                        Y_trks = {
                            track_name: hFile[f"Y_{track_name}_train"][start:end]
                            for track_name in self.tracks_name
                        }

                    else:
                        Y_trks = None

                else:
                    X_trks = None
                    Y_trks = None

                # Check if conditional jet parameters are used or not
                if self.n_add_vars is not None:
                    X_Add_Vars = hFile["X_train"][start:end, : self.n_add_vars]

                else:
                    X_Add_Vars = None

                # Yield the chunk
                yield X_jets, X_trks, Y_jets, Y_trks, Weights, X_Add_Vars

    def save_parameters(self, record_dir):
        """
        write metadata into metadata.json and save it with tf record files

        Parameters
        ----------
        record_dir : str
            directory where metadata should be saved
        """

        # Open h5 file
        with h5py.File(self.path_h5) as h5file:

            # Init a data dict
            data = {}

            # Get dimensional values of the jets
            data["n_jets"] = len(h5file["X_train"])
            data["n_jet_features"] = len(h5file["X_train"][0])

            # Get the dimensional values of the labels
            data["n_dim"] = len(h5file["Y_train"][0])

            # Get the dimensional values of the tracks and save them for each track
            # collection in a dict
            if self.tracks_name is not None:
                data["n_trks"] = {
                    track_name: len(h5file[f"X_{track_name}_train"][0])
                    for track_name in self.tracks_name
                }
                data["n_trk_features"] = {
                    track_name: len(h5file[f"X_{track_name}_train"][0][0])
                    for track_name in self.tracks_name
                }

                if self.save_track_labels:
                    data["n_trks_labels"] = {
                        track_name: len(h5file[f"Y_{track_name}_train"][0])
                        for track_name in self.tracks_name
                    }
                    data["n_trks_classes"] = {
                        track_name: len(h5file[f"Y_{track_name}_train"][0][0])
                        for track_name in self.tracks_name
                    }

                else:
                    data["n_trks_labels"] = None
                    data["n_trks_classes"] = None

            else:
                data["n_trks"] = None
                data["n_trk_features"] = None
                data["n_trks_labels"] = None
                data["n_trks_classes"] = None

            # Get the dimensional values of the conditional variables
            if self.n_add_vars is not None:
                data["n_add_vars"] = self.n_add_vars

            else:
                data["n_add_vars"] = None

        # Get filepath for the metadata file
        metadata_filename = record_dir + "/metadata.json"

        # Write the metadata (dim. values) to file
        with open(metadata_filename, "w") as metadata:
            logger.info("Writing metadata to %s", metadata_filename)
            json.dump(data, metadata)

    def write_tfrecord(self):
        """
        write inputs and labels of train file into a TFRecord
        """

        # Get the path to h5 file and make a dir with that name
        record_dir = self.path_h5.replace(".h5", "")
        os.makedirs(record_dir, exist_ok=True)

        # Get filename
        tf_filename_start = record_dir.split("/")[-1]
        n = 0

        # Iterate over chunks
        for (
            X_jets,
            X_trks,
            Y_jets,
            Y_trks,
            Weights,
            X_Add_Vars,
        ) in self.load_h5File_Train():
            n += 1

            # Get filename of the chunk
            filename = (
                record_dir
                + "/"
                + tf_filename_start
                + "_"
                + str(n).zfill(4)
                + ".tfrecord"
            )

            with tf.io.TFRecordWriter(filename) as file_writer:
                for iterator, _ in enumerate(X_jets):

                    # Get record bytes example
                    record_bytes = tf.train.Example()

                    # Add jets
                    record_bytes.features.feature["X_jets"].float_list.value.extend(
                        X_jets[iterator].reshape(-1)
                    )

                    # Add labels
                    record_bytes.features.feature["Y_jets"].int64_list.value.extend(
                        Y_jets[iterator]
                    )

                    # Add weights
                    record_bytes.features.feature["Weights"].float_list.value.extend(
                        Weights[iterator].reshape(-1)
                    )

                    if self.tracks_name is not None:
                        # Add track collections
                        for key, item in X_trks.items():
                            record_bytes.features.feature[
                                f"X_{key}_train"
                            ].float_list.value.extend(item[iterator].reshape(-1))

                        if self.save_track_labels:
                            for key, item in Y_trks.items():
                                record_bytes.features.feature[
                                    f"Y_{key}_train"
                                ].int64_list.value.extend(item[iterator].reshape(-1))

                    # Add conditional variables if used
                    if self.n_add_vars is not None:
                        record_bytes.features.feature[
                            "X_Add_Vars"
                        ].float_list.value.extend(X_Add_Vars[iterator].reshape(-1))

                    # Write to file
                    file_writer.write(record_bytes.SerializeToString())
                logger.info("Data written in %s", filename)
        self.save_parameters(record_dir=record_dir)
