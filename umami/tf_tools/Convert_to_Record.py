"""Module converting h5 to tf records."""
from umami.configuration import logger  # isort:skip

import json
import os

import h5py
import tensorflow as tf
import tqdm


class h5toTFRecordConverter:
    """h5 converter to tf records."""

    def __init__(self, config):
        self.config = config
        self.path_h5 = self.config.GetFileName(option="resampled_scaled_shuffled")
        try:
            self.chunk_size = int(config.convert_to_tfrecord["chunk_size"])
            logger.info(f"Save {self.chunk_size} entries in one file")

        except (AttributeError, KeyError, ValueError) as chunk_size_no_int:
            try:
                self.chunk_size = config.preparation["convert"]["chunk_size"]
                if not isinstance(self.chunk_size, int):
                    raise KeyError from chunk_size_no_int
                logger.info(f"Save {self.chunk_size} entries in one file")

            except KeyError:
                logger.warning(
                    "Chunk size for conversion into tf records not set in config"
                    "file. Set to 5000"
                )
                self.chunk_size = 5_000
        # TODO: adding possibility to use more than first element of 'tracks_names'
        # only first element of the tracks_names list get converted only
        self.tracks_name = config.sampling["options"]["tracks_names"][0]
        logger.warning(
            "Only first element of `track_names` from the config file is converted to "
            f"tf records {self.tracks_name}. In case you want to convert another track "
            "collection, please adapt your config file."
        )
        if "N_add_vars" in config.convert_to_tfrecord:
            self.n_add_vars = config.convert_to_tfrecord["N_add_vars"]
        else:
            self.n_add_vars = None

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
        Y : array_like
            Training labels
        Weights : array_like
            Training weights
        """

        with h5py.File(self.path_h5, "r") as hFile:
            length_dataset = len(hFile["X_train"])
            logger.info(
                f"Total length of the dataset is {length_dataset}. Load"
                f" {self.chunk_size} samples at a time"
            )
            total_loads = int(length_dataset / self.chunk_size)
            if length_dataset % self.chunk_size != 0:
                total_loads += 1
            logger.info(f"Total number of loading steps is {total_loads}")
            for i in tqdm.tqdm(range(total_loads)):
                start = i * self.chunk_size
                end = (i + 1) * self.chunk_size
                X_jets = hFile["X_train"][start:end]
                X_trks = hFile[f"X_{self.tracks_name}_train"][start:end]
                Y = hFile["Y_train"][start:end]
                Weights = hFile["weight"][start:end]
                if self.n_add_vars is not None:
                    X_Add_Vars = hFile["X_train"][start:end, : self.n_add_vars]
                else:
                    X_Add_Vars = [None] * (end - start)
                yield X_jets, X_trks, Y, Weights, X_Add_Vars

    def save_parameters(self, record_dir):
        """
        write metadata into metadata.json and save it with tf record files

        Parameters
        ----------
        record_dir : str
            directory where metadata should be saved
        """
        with h5py.File(self.path_h5) as h5file:
            nJets = len(h5file["X_train"])
            njet_feature = len(h5file["X_train"][0])
            nTrks = len(h5file[f"X_{self.tracks_name}_train"][0])
            nFeatures = len(h5file[f"X_{self.tracks_name}_train"][0][0])
            nDim = len(h5file["Y_train"][0])
            n_add_vars = self.n_add_vars
            data = {
                "nJets": nJets,
                "njet_features": njet_feature,
                "nTrks": nTrks,
                "nFeatures": nFeatures,
                "nDim": nDim,
                "nadd_vars": n_add_vars,
            }
        metadata_filename = record_dir + "/metadata.json"
        with open(metadata_filename, "w") as metadata:
            logger.info(f"Writing metadata to {metadata_filename}")
            json.dump(data, metadata)

    def write_tfrecord(self):
        """
        write inputs and labels of train file into a TFRecord
        """
        record_dir = self.path_h5.replace(".h5", "")
        os.makedirs(record_dir, exist_ok=True)
        tf_filename_start = record_dir.split("/")[-1]
        n = 0
        for X_jets, X_trks, Y, Weights, X_Add_Vars in self.load_h5File_Train():
            n += 1
            filename = (
                record_dir
                + "/"
                + tf_filename_start
                + "_"
                + str(n).zfill(4)
                + ".tfrecord"
            )
            with tf.io.TFRecordWriter(filename) as file_writer:
                for (x_jets, x_trks, y, weight, x_add_vars) in zip(
                    X_jets, X_trks, Y, Weights, X_Add_Vars
                ):
                    record_bytes = tf.train.Example()
                    record_bytes.features.feature["X_jets"].float_list.value.extend(
                        x_jets.reshape(-1)
                    )
                    record_bytes.features.feature["X_trks"].float_list.value.extend(
                        x_trks.reshape(-1)
                    )
                    record_bytes.features.feature["Y"].int64_list.value.extend(y)
                    record_bytes.features.feature["Weights"].float_list.value.extend(
                        weight.reshape(-1)
                    )
                    if x_add_vars is not None:
                        record_bytes.features.feature[
                            "X_Add_Vars"
                        ].float_list.value.extend(x_add_vars.reshape(-1))
                    file_writer.write(record_bytes.SerializeToString())
                logger.info(f"Data written in {filename}")
        self.save_parameters(record_dir=record_dir)
