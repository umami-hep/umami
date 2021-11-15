from umami.configuration import logger  # isort:skip
import h5py
import numpy as np


class Model_Generator(object):
    def __init__(
        self,
        train_file_path: str,
        Y_Name: str,
        n_jets: int,
        batch_size: int,
        chunk_size: int = 1e5,
        X_Name: str = None,
        X_trk_Name: str = None,
        excluded_var: list = None,
    ):
        self.train_file_path = train_file_path
        self.X_Name = X_Name
        self.X_trk_Name = X_trk_Name
        self.Y_Name = Y_Name
        self.batch_size = batch_size
        self.excluded_var = excluded_var
        self.chunk_size = chunk_size
        if n_jets is not None:
            self.n_jets = int(n_jets)
        else:
            if X_Name is not None:
                with h5py.File(self.train_file_path, "r") as f:
                    self.n_jets = int(len(f[self.X_Name]))

            elif X_trk_Name is not None:
                with h5py.File(self.train_file_path, "r") as f:
                    self.n_jets = int(len(f[self.X_trk_Name]))

            else:
                raise ValueError(
                    "You need to give either X_Name or X_Trk_Name to the generator!"
                )
        self.length = int(self.n_jets / self.batch_size)
        self.step_size = self.batch_size * int(
            self.chunk_size / self.batch_size
        )

    def load_in_memory(
        self, load_jets: bool, load_tracks: bool, part: int = 0
    ):
        """
        Load the jets or tracks or both step by step in memory.

        Input:
        - load_jets, bool: Define, if jets are loaded or not.
        - load_tracks, bool: Define, if tracks are loaded or not.
        - part, int: Part of the data which is to be loaded.

        Output:
        - Loads the part of data to memory.
        """

        logger.info(
            f"\nloading in memory {part + 1}/{1 + self.n_jets // self.step_size}"
        )

        # Check that the correct X_Name and X_trk_Name is given
        if load_jets is True and self.X_Name is None:
            raise ValueError(
                "X_Name needs to be given when jet features are to be loaded!"
            )

        elif load_tracks is True and self.X_trk_Name is None:
            raise ValueError(
                "X_trk_Name needs to be given when track features are to be loaded!"
            )

        # Open train file
        with h5py.File(self.train_file_path, "r") as f:
            # Load jets if wanted
            if load_jets:
                self.x_in_mem = f[self.X_Name][
                    self.step_size * part : self.step_size * (part + 1)
                ]

                # Exclude variables if needed
                self.x_in_mem = (
                    np.delete(self.x_in_mem, self.excluded_var, 1)
                    if self.excluded_var is not None
                    else self.x_in_mem
                )

            # Load tracks if wanted
            if load_tracks:
                self.x_trk_in_mem = f[self.X_trk_Name][
                    self.step_size * part : self.step_size * (part + 1)
                ]

            # Load truth labels
            self.y_in_mem = f[self.Y_Name][
                self.step_size * part : self.step_size * (part + 1)
            ]


class dips_generator(Model_Generator):
    def __call__(self):
        self.load_in_memory(part=0, load_jets=False, load_tracks=True)
        n = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * n:
                self.load_in_memory(part=n, load_jets=False, load_tracks=True)
                n += 1
                small_step = 0
            batch_x_trk = self.x_trk_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            small_step += 1
            yield (batch_x_trk, batch_y)


class dl1_generator(Model_Generator):
    def __call__(self):
        self.load_in_memory(part=0, load_jets=True, load_tracks=False)
        n = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * n:
                self.load_in_memory(part=n, load_jets=True, load_tracks=False)
                n += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            small_step += 1
            yield (batch_x, batch_y)


class umami_generator(Model_Generator):
    def __call__(self):
        self.load_in_memory(part=0, load_jets=True, load_tracks=True)
        n = 1
        small_step = 0
        for idx in range(self.length):
            if (idx + 1) * self.batch_size > self.step_size * n:
                self.load_in_memory(part=n, load_jets=True, load_tracks=True)
                n += 1
                small_step = 0
            batch_x = self.x_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            batch_x_trk = self.x_trk_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            batch_y = self.y_in_mem[
                small_step
                * self.batch_size : (1 + small_step)
                * self.batch_size
            ]
            small_step += 1
            yield {"input_1": batch_x_trk, "input_2": batch_x}, batch_y
