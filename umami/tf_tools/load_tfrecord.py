"""Reader for tf records datasets."""
import json
import os

import tensorflow as tf

from umami.configuration import logger


def load_tfrecords_train_dataset(
    train_config: object,
):
    """
    Load the train dataset from tfrecords files.

    Parameters
    ----------
    train_config : object
        Loaded train config.

    Returns
    -------
    train_dataset : tfrecord.Dataset
        Loaded train dataset from tfrecords.
    metadata : dict
        Dict with the metadata infos of the train dataset.

    Raises
    ------
    ValueError
        If one of the given input files is not a tfrecords file.
    KeyError
        If no metadata file could be found in tfrecords directory.
    """
    # Load NN Structure and training parameter from file
    nn_structure = train_config.nn_structure
    tracks_name = train_config.tracks_name

    # Get the files in dir
    train_file_names = os.listdir(train_config.train_file)

    # Loop over files in dir
    for train_file_name in train_file_names:

        # Check if file is tfrecords or .h5
        if not (".tfrecord" in train_file_name) and not (
            train_file_name == "metadata.json"
        ):
            raise ValueError(
                f"Input file {train_config.train_file} is neither a "
                ".h5 file nor a directory with TF Record Files. "
                "You should check this."
            )

    # Check if train file is in metadata
    if "metadata.json" not in train_file_names:
        raise KeyError("No metadata file in directory.")

    # Check if nfiles is given. Otherwise set to 5
    try:
        nfiles = nn_structure["nfiles_tfrecord"]

    except KeyError:
        nfiles = 5
        logger.warning(
            "No number of files to be loaded in parallel defined. Set to %i", nfiles
        )

    # Get the tfrecords
    tfrecord_reader = TFRecordReader(
        path=train_config.train_file,
        batch_size=nn_structure["batch_size"],
        nfiles=nfiles,
        tagger_name=nn_structure["tagger"],
        tracks_name=tracks_name,
        use_track_labels=nn_structure["attention_aux_task"]
        if "attention_aux_task" in nn_structure
        else False,
        n_cond=nn_structure["n_conditions"] if "n_conditions" in nn_structure else None,
    )

    # Load the dataset from reader
    train_dataset = tfrecord_reader.load_dataset()

    # Get the metadata name
    metadata_name = (train_config.train_file + "/metadata.json").replace("//", "/")

    # Load metadata in file
    with open(metadata_name, "r") as metadata_file:
        metadata = json.load(metadata_file)
        metadata["n_trks"] = metadata["n_trks"][tracks_name]
        metadata["n_trk_features"] = metadata["n_trk_features"][tracks_name]
        metadata["n_trks_labels"] = metadata["n_trks_labels"][tracks_name]
        metadata["n_trks_classes"] = metadata["n_trks_classes"][tracks_name]

    return train_dataset, metadata


class TFRecordReader:
    """Reader for tf records datasets."""

    def __init__(
        self,
        path: str,
        batch_size: int,
        nfiles: int,
        tagger_name: str,
        tracks_name: str = None,
        use_track_labels: bool = False,
        sample_weights: bool = False,
        n_cond: int = None,
    ):
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
        tagger_name : str
            Name of the tagger that is used
        tracks_name : str, optional
            Name of the track collection that is loaded,
            by default None
        use_track_labels : bool, optional
            Decide if you want to use track labels, by default
            False.
        sample_weights : bool, optional
            decide wether or not the sample weights should
            be returned, by default False
        n_cond : int, optional
            number of additional variables used for attention,
            by default None
        """
        self.path = path
        self.batch_size = batch_size
        self.nfiles = nfiles
        self.tagger_name = tagger_name
        self.tracks_name = tracks_name
        self.use_track_labels = use_track_labels
        self.sample_weights = sample_weights
        self.n_cond = n_cond

    def load_dataset(self):
        """
        Load TFRecord and create Dataset for training

        Returns
        -------
        tf_Dataset
        """
        data_files = tf.io.gfile.glob((self.path + "/*.tfrecord").replace("//", "/"))
        dataset_shards = tf.data.Dataset.from_tensor_slices([data_files])
        dataset_shards.shuffle(tf.cast(tf.shape(data_files)[0], tf.int64))
        tf_dataset = dataset_shards.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=self.nfiles,
        )
        tf_dataset = (
            tf_dataset.shuffle(self.batch_size * 10)
            .batch(self.batch_size)
            .map(
                self.decode_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .repeat()
            .prefetch(3)
        )
        return tf_dataset

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

        Raises
        ------
        KeyError
            If given track selection not in metadata.
        KeyError
            If no conditional info is found in metadata.
        ValueError
            If tagger type is not supported.
        """

        # Get metadata file and load it
        metadata_name = (self.path + "/metadata.json").replace("//", "/")
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # Set output shapes
        shapes = {}
        features = {}

        if self.tagger_name.casefold() in ("dl1", "umami", "umami_cond_att"):
            shapes["shape_Xjets"] = [metadata["n_jet_features"]]
            features["X_jets"] = tf.io.FixedLenFeature(
                shape=shapes["shape_Xjets"], dtype=tf.float32
            )

        # Set track shape
        if self.tagger_name.casefold() in (
            "dips",
            "dips_attention",
            "cads",
            "umami",
            "umami_cond_att",
        ):
            try:
                shapes[f"shape_X_{self.tracks_name}_train"] = [
                    metadata["n_trks"][self.tracks_name],
                    metadata["n_trk_features"][self.tracks_name],
                ]
                features[f"{self.tracks_name}/inputs"] = tf.io.FixedLenFeature(
                    shape=shapes[f"shape_X_{self.tracks_name}_train"], dtype=tf.float32
                )

                if self.use_track_labels:
                    shapes[f"shape_Y_{self.tracks_name}_train"] = [
                        metadata["n_trks_labels"][self.tracks_name],
                        metadata["n_trks_classes"][self.tracks_name],
                    ]
                    features[f"{self.tracks_name}/labels"] = tf.io.FixedLenFeature(
                        shape=shapes[f"shape_Y_{self.tracks_name}_train"],
                        dtype=tf.int64,
                    )

            except KeyError as error:
                raise KeyError(
                    f"Track collection {self.tracks_name} not in metadata file!"
                ) from error

        # Set label shape
        shapes["shape_Y_jets"] = [metadata["n_dim"]]
        features["Y_jets"] = tf.io.FixedLenFeature(
            shape=shapes["shape_Y_jets"], dtype=tf.int64
        )

        # Set weights shape
        features["Weights"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)

        # Set conditional variables shape
        if self.n_cond is not None:
            try:
                shapes["shape_Add_Vars"] = [metadata["n_add_vars"]]
                features["X_Add_Vars"] = tf.io.FixedLenFeature(
                    shape=shapes["shape_Add_Vars"], dtype=tf.float32
                )

            except KeyError as error:
                raise KeyError(
                    "No conditional information saved in tfrecords metadata file!"
                ) from error

        # Get the parser
        parse_ex = tf.io.parse_example(record_bytes, features)  # pylint: disable=E1120

        # return the jet inputs and labels
        if self.tagger_name.casefold() == "dl1":
            input_dir = {"input_1": parse_ex["X_jets"]}

        elif self.tagger_name.casefold() in ("dips", "dips_attention", "cads"):
            input_dir = {
                "input_1": parse_ex[f"{self.tracks_name}/inputs"],
            }

            if self.n_cond is not None:
                input_dir["input_2"] = parse_ex["X_Add_Vars"][:, : self.n_cond]

        elif self.tagger_name.casefold() in ("umami", "umami_cond_att"):
            if self.n_cond is not None:
                input_dir = {
                    "input_1": parse_ex[f"{self.tracks_name}/inputs"],
                    "input_2": parse_ex["X_Add_Vars"][:, : self.n_cond],
                    "input_3": parse_ex["X_jets"],
                }

            else:
                input_dir = {
                    "input_1": parse_ex[f"{self.tracks_name}/inputs"],
                    "input_2": parse_ex["X_jets"],
                }

        else:
            raise ValueError(f"Tagger '{self.tagger_name}' is not supported!")

        if self.use_track_labels:
            if self.sample_weights:
                return (
                    input_dir,
                    parse_ex["Y_jets"],
                    parse_ex[f"{self.tracks_name}/labels"],
                    parse_ex["Weights"],
                )

            return (
                input_dir,
                parse_ex["Y_jets"],
                parse_ex[f"{self.tracks_name}/labels"],
            )

        if self.sample_weights:
            return input_dir, parse_ex["Y_jets"], parse_ex["Weights"]

        return input_dir, parse_ex["Y_jets"]
