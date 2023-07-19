"""Data generator module to handle reading of training datasets."""
from umami.configuration import logger  # isort:skip
import h5py
import numpy as np


class ModelGenerator:  # pylint: disable=too-few-public-methods
    """Base class for the generators of the datasets for the models.

    This class provides the base functionalites for the different
    models to load the dataset.
    """

    def __init__(
        self,
        train_file_path: str,
        y_name: str,
        n_jets: int,
        batch_size: int,
        sample_weights: bool,
        chunk_size: int = 1e5,
        x_name: str = None,
        x_trk_name: str = None,
        excluded_var: list = None,
        n_conds: int = None,
        print_logger: bool = False,
    ):
        """Init the parameters needed for the generators.

        Parameters
        ----------
        train_file_path : str
            Path to the train file that is to be used.
        y_name : str
            Name of the truth info inside the train file.
        n_jets : int
            Number of jets that is to be used for training.
        batch_size : int
            Batch size for the training.
        sample_weights : bool
            Decide, if you want to use sample weights. Those
            need to be processed in the preprocessing. Otherwise
            the values are ones.
        chunk_size : int
            Chunk size for loading the training jets.
        x_name : str
            Name of the jet variables inside the train file.
        x_trk_name : str
            Name of the track variables inside the train file.
        excluded_var : list
            List with excluded variables. Only available for
            DL1 training.
        n_conds : int
            Number of conditions used for training of CADS.
        print_logger : bool
            Decide, if the logger outputs are printed or not.

        Raises
        ------
        ValueError
            If neither x_name or x_trk_name is given.
        """

        self.train_file_path = train_file_path
        self.x_name = x_name
        self.x_trk_name = x_trk_name
        self.y_name = y_name
        self.batch_size = batch_size
        self.excluded_var = excluded_var
        self.n_conds = n_conds
        self.chunk_size = chunk_size
        self.print_logger = print_logger
        self.sample_weights = sample_weights
        if n_jets is not None:
            self.n_jets = int(n_jets)
        else:
            if x_name is not None:
                with h5py.File(self.train_file_path, "r") as f_train:
                    self.n_jets = int(len(f_train[self.x_name]))

            elif x_trk_name is not None:
                with h5py.File(self.train_file_path, "r") as f_train:
                    self.n_jets = int(len(f_train[self.x_trk_name]))

            else:
                raise ValueError(
                    "You need to give either X_Name or X_Trk_Name to the generator!"
                )
        self.length = int(self.n_jets / self.batch_size)
        self.step_size = self.batch_size * int(self.chunk_size / self.batch_size)
        self.x_in_mem = None
        self.weight_in_mem = None
        self.x_trk_in_mem = None
        self.y_in_mem = None

    def load_in_memory(self, load_jets: bool, load_tracks: bool, part: int = 0):
        """
        Load the jets or tracks or both step by step in memory.

        Parameters
        ----------
        load_jets : bool
            Define, if jets are loaded or not.
        load_tracks : bool
            Define, if tracks are loaded or not.
        part : int
            Part of the data which is to be loaded.

        Raises
        ------
        ValueError
            If X_Name or X_trk_Name are not given when requesting jets or
            tracks, respectively.
        """

        if self.print_logger is True:
            logger.info(
                "\nloading in memory %i/%i", part + 1, 1 + self.n_jets // self.step_size
            )

        # Check that the correct X_Name and X_trk_Name is given
        if load_jets is True and self.x_name is None:
            raise ValueError(
                "X_Name needs to be given when jet features are to be loaded!"
            )

        if load_tracks is True and self.x_trk_name is None:
            raise ValueError(
                "X_trk_Name needs to be given when track features are to be loaded!"
            )

        # Open train file
        with h5py.File(self.train_file_path, "r") as f_train:
            # Load jets if wanted
            if load_jets:
                self.x_in_mem = f_train[self.x_name][
                    self.step_size * part : self.step_size * (part + 1)
                ]

                # Exclude variables if needed
                self.x_in_mem = (
                    np.delete(self.x_in_mem, self.excluded_var, 1)
                    if self.excluded_var is not None
                    else self.x_in_mem
                )

            if self.sample_weights:
                # load weights
                self.weight_in_mem = f_train["jets/weight"][
                    self.step_size * part : self.step_size * (part + 1)
                ]

            # Load tracks if wanted
            if load_tracks:
                self.x_trk_in_mem = f_train[self.x_trk_name][
                    self.step_size * part : self.step_size * (part + 1)
                ]

            # Load truth labels
            self.y_in_mem = f_train[self.y_name][
                self.step_size * part : self.step_size * (part + 1)
            ]

    def get_n_jets(self):
        """
        Get the number of jets.

        Returns
        -------
        int
            The number of jets.
        """
        return self.n_jets

    def get_n_dim(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of dimensions
        of the labels. It does not load jets or tracks.

        Returns
        -------
        int
            The number of dimensions of the labels.
        """
        self.load_in_memory(load_jets=False, load_tracks=False, part=0)
        return self.y_in_mem.shape[1]

    def get_n_jet_features(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of features
        for the jets. It does not load tracks.

        Returns
        -------
        int
            The number of jet features.
        """
        self.load_in_memory(load_jets=True, load_tracks=False, part=0)
        return self.x_in_mem.shape[1]

    def get_n_trk_features(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of features
        for the tracks. It does not load jets.

        Returns
        -------
        int
            The number of track features.
        """
        self.load_in_memory(load_jets=False, load_tracks=True, part=0)
        return self.x_trk_in_mem.shape[2]

    def get_n_trks(self):
        """
        This function loads the necessary data into memory
        and retrieves the number of tracks.
        It does not load jets.

        Returns
        -------
        int
            The number of tracks.
        """
        self.load_in_memory(load_jets=False, load_tracks=True, part=0)
        return self.x_trk_in_mem.shape[1]


class DipsGenerator(ModelGenerator):
    """Generator class for DIPS.

    This class provides the a generator that loads the training dataset
    for DIPS.
    """

    def __call__(self):
        """
        Load the first chunk in memory and yield the full dataset.

        Yields
        ------
        (batch_x_trk, batch_y) : tuple
            Yielded chunks of the training dataset.
        """

        self.load_in_memory(part=0, load_jets=False, load_tracks=True)
        iteration = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * iteration:
                self.load_in_memory(part=iteration, load_jets=False, load_tracks=True)
                iteration += 1
                small_step = 0
            batch_x_trk = self.x_trk_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            if self.sample_weights:
                batch_sample_weight = self.weight_in_mem[
                    small_step * self.batch_size : (1 + small_step) * self.batch_size
                ]
                small_step += 1
                yield (batch_x_trk, batch_y, batch_sample_weight)
            else:
                small_step += 1
                yield (batch_x_trk, batch_y)


class Dl1Generator(ModelGenerator):
    """Generator class for DL1*.

    This class provides the a generator that loads the training dataset
    for DL1*.
    """

    def __call__(self):
        """
        Load the first chunk in memory and yield the full dataset.

        Yields
        ------
        (batch_x_trk, batch_y) : tuple
            Yielded chunks of the training dataset.
        """

        self.load_in_memory(part=0, load_jets=True, load_tracks=False)
        iteration = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * iteration:
                self.load_in_memory(part=iteration, load_jets=True, load_tracks=False)
                iteration += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            if self.sample_weights:
                batch_sample_weight = self.weight_in_mem[
                    small_step * self.batch_size : (1 + small_step) * self.batch_size
                ]
                small_step += 1
                yield (batch_x, batch_y, batch_sample_weight)
            else:
                small_step += 1
                yield (batch_x, batch_y)


class UmamiGenerator(ModelGenerator):
    """Generator class for UMAMI.

    This class provides the a generator that loads the training dataset
    for UMAMI.
    """

    def __call__(self):
        """
        Load the first chunk in memory and yield the full dataset.

        Yields
        ------
        (batch_x_trk, batch_y) : tuple
            Yielded chunks of the training dataset.
        """

        self.load_in_memory(part=0, load_jets=True, load_tracks=True)
        iteration = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * iteration:
                self.load_in_memory(part=iteration, load_jets=True, load_tracks=True)
                iteration += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_x_trk = self.x_trk_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            small_step += 1
            yield {"input_1": batch_x_trk, "input_2": batch_x}, batch_y


class CadsGenerator(ModelGenerator):
    """Generator class for CADS.

    This class provides the a generator that loads the training dataset
    for CADS.
    """

    def __call__(self):
        """
        Load the first chunk in memory and yield the full dataset.

        Yields
        ------
        (batch_x_trk, batch_y) : tuple
            Yielded chunks of the training dataset.
        """

        self.load_in_memory(part=0, load_jets=True, load_tracks=True)
        iteration = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * iteration:
                self.load_in_memory(part=iteration, load_jets=True, load_tracks=True)
                iteration += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size,
                : self.n_conds,
            ]
            batch_x_trk = self.x_trk_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            small_step += 1
            yield {"input_1": batch_x_trk, "input_2": batch_x}, batch_y


class UmamiConditionGenerator(ModelGenerator):
    """Generator class for UMAMI with conditional attention.

    This class provides the a generator that loads the training dataset
    for UMAMI with conditional attention.
    """

    def __call__(self):
        self.load_in_memory(part=0, load_jets=True, load_tracks=True)
        iteration = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * iteration:
                self.load_in_memory(part=iteration, load_jets=True, load_tracks=True)
                iteration += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_x_cond = self.x_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size,
                : self.n_conds,
            ]
            batch_x_trk = self.x_trk_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step * self.batch_size : (1 + small_step) * self.batch_size
            ]
            small_step += 1
            yield {
                "input_1": batch_x_trk,
                "input_2": batch_x_cond,
                "input_3": batch_x,
            }, batch_y
