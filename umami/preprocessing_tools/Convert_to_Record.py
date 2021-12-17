from umami.configuration import logger  # isort:skip

import json
import os

import h5py
import tensorflow as tf
import tqdm


class h5toTFRecordConverter:
    def __init__(self, config):
        self.config = config
        self.path_h5 = self.config.GetFileName(option="resampled_scaled_shuffled")
        try:
            self.chunk_size = config.preparation["convert"]["chunk_size"]
            logger.info(f"Save {self.chunk_size} entries in one file")
        except KeyError:
            logger.warning(
                "Chunk size for conversion into tf records not set in config"
                "file. Set to 5000"
            )
            self.chunk_size = 5_000

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
                X_trks = hFile["X_trk_train"][start:end]
                Y = hFile["Y_train"][start:end]
                Weights = hFile["weight"][start:end]
                yield X_jets, X_trks, Y, Weights

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
            nTrks = len(h5file["X_trk_train"][0])
            nFeatures = len(h5file["X_trk_train"][0][0])
            nDim = len(h5file["Y_train"][0])
            data = {
                "nJets": nJets,
                "njet_features": njet_feature,
                "nTrks": nTrks,
                "nFeatures": nFeatures,
                "nDim": nDim,
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
        for X_jets, X_trks, Y, Weights in self.load_h5File_Train():
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
                for (x_jets, x_trks, y, weight) in zip(X_jets, X_trks, Y, Weights):
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
                    file_writer.write(record_bytes.SerializeToString())
                logger.info(f"Data written in {filename}")
        self.save_parameters(record_dir=record_dir)
