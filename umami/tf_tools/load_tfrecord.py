"""Reader for tf records datasets."""
import json

import tensorflow as tf


class TFRecordReader:
    """Reader for tf records datasets."""

    def __init__(self, path, batch_size, nfiles, sample_weights=False, n_cond=None):
        """
        Reads the tf records dataset.

        Parameters
        ----------
        path : str
            path where TFRecord is saved
        batch_size : int
            size of batches for the training
        nfiles : int
            number of tf record files loaded in parallel
        sample_weights : bool
            decide wether or not the sample weights should
            be returned
        n_cond : int
            number of additional variables used for attention
        """
        self.path = path
        self.batch_size = batch_size
        self.nfiles = nfiles
        self.sample_weights = sample_weights
        self.n_cond = n_cond

    def load_Dataset(self):
        """
        Load TFRecord and create Dataset for training

        Returns
        -------
        tf_Dataset
        """
        data_files = tf.io.gfile.glob((self.path + "/*.tfrecord").replace("//", "/"))
        Dataset_shards = tf.data.Dataset.from_tensor_slices([data_files])
        Dataset_shards.shuffle(tf.cast(tf.shape(data_files)[0], tf.int64))
        tf_Dataset = Dataset_shards.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=self.nfiles,
        )
        tf_Dataset = (
            tf_Dataset.shuffle(self.batch_size * 10)
            .batch(self.batch_size)
            .map(
                self.decode_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .repeat()
            .prefetch(3)
        )
        return tf_Dataset

    def decode_fn(self, record_bytes):
        """
        Convert serialised Dataset to dictionary and return inputs and labels

        Parameters
        ----------
        record_bytes : serialised object
            serialised Dataset

        Returns
        -------
        inputs : dict
            Dictionary of tf_data of jet and track inputs
        labels : tf_data
            tf data stream of labels

        """
        metadata_name = (self.path + "/metadata.json").replace("//", "/")
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)
        shapes = {
            "shape_Xjets": [metadata["njet_features"]],
            "shape_Xtrks": [metadata["nTrks"], metadata["nFeatures"]],
            "shape_Y": [metadata["nDim"]],
        }
        features = {
            "X_jets": tf.io.FixedLenFeature(
                shape=shapes["shape_Xjets"], dtype=tf.float32
            ),
            "X_trks": tf.io.FixedLenFeature(
                shape=shapes["shape_Xtrks"], dtype=tf.float32
            ),
            "Y": tf.io.FixedLenFeature(shape=shapes["shape_Y"], dtype=tf.int64),
            "Weights": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
        }
        if self.n_cond is not None:
            shapes["shape_Add_Vars"] = [metadata["nadd_vars"]]
            features["X_Add_Vars"] = tf.io.FixedLenFeature(
                shape=shapes["shape_Add_Vars"], dtype=tf.float32
            )

        parse_ex = tf.io.parse_example(record_bytes, features)  # pylint: disable=E1120

        # return the jet inputs and labels
        if self.n_cond is not None:
            input_dir = {
                "input_1": parse_ex["X_trks"],
                "input_2": parse_ex["X_Add_Vars"][:, : self.n_cond],
                "input_3": parse_ex["X_jets"],
            }
        else:
            input_dir = {
                "input_1": parse_ex["X_trks"],
                "input_2": parse_ex["X_jets"],
            }

        if self.sample_weights:
            return input_dir, parse_ex["Y"], parse_ex["Weights"]

        return input_dir, parse_ex["Y"]
