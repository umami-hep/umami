"""Helper functions to creating hybrid hdf5 samples from ttbar and Zprime ntuples."""
import h5py
import numpy as np
from tqdm import tqdm

from umami.configuration import global_config, logger
from umami.data_tools import get_category_cuts, get_sample_cuts


class PrepareSamples:
    """
    This class is preparing the samples for further processing defined in the
    configuration file:
        - extracts the selected jets (applying cuts: flavour, pT etc.)
        - writes these iteratively to h5 output files

    This class will take the information provided in the `samples` block in
    the preprocessing config.
    """

    def __init__(self, args, config) -> None:
        """Preparation of h5 samples.

        Parameters
        ----------
        args : parse_args
            command line arguments
        config : config class
            preprocessing configuration class containing all info about preprocessing
        """
        self.config = config
        self.__setup(args)
        self.rnd_seed = 42

    def __setup(self, args):
        """Setting up preparation class

        Parameters
        ----------
        args : parse_args
            command line arguments

        Raises
        ------
        KeyError
            if no samples defined in preprocessing config
        KeyError
            if specified sample not in preprocessing configuration
        KeyError
            if requested sample category not defined in global config
        """
        # check if sample is provided, otherwise exit
        if not args.sample:
            raise KeyError("Please provide --sample to prepare hybrid samples")
        # load list of samples
        self.sample = self.config.preparation.get_sample(args.sample)

        cuts = self.sample.cuts
        if self.sample.category == "inclusive":
            self.cuts = cuts
        else:
            try:
                category_setup = global_config.flavour_categories[self.sample.category]
            except KeyError as error:
                raise KeyError(
                    f"Requested sample category {self.sample.category} not"
                    " defined in global config."
                ) from error

            # retrieving the cuts for the category selection
            category_cuts = get_category_cuts(
                category_setup.get("label_var"),
                category_setup.get("label_value"),
                category_setup.get("operator"),
            )
            self.cuts = cuts + category_cuts

        # Check if tracks are used
        self.save_tracks = self.config.sampling["options"]["save_tracks"]
        self.tracks_names = self.config.sampling["options"]["tracks_names"]

        self.sample.output_name.parent.mkdir(parents=True, exist_ok=True)

        # bookkeeping variables for running over the ntuples
        self.jets_loaded = 0
        self.create_file = True
        self.shuffle_array = args.shuffle_array

    def get_batches_per_file(self, filename: str):
        """
        Split the file into batches to avoid that the loaded data is too large.

        Parameters
        ----------
        filename : str
            name of file to be split in batches

        Returns
        -------
        str
            filename
        list
            tuples of start and end index of batch

        """
        with h5py.File(filename, "r") as data_set:
            # get total number of jets in file
            total_n_jets = len(data_set["jets"])
            logger.debug("Total number of jets in file: %i", total_n_jets)
            # first tuple is given by (0, batch_size)
            start_batch = 0
            end_batch = self.config.preparation.batch_size
            indices_batches = [(start_batch, end_batch)]
            # get remaining tuples of indices defining the batches
            while end_batch <= total_n_jets:
                start_batch += self.config.preparation.batch_size
                end_batch = start_batch + self.config.preparation.batch_size
                indices_batches.append((start_batch, end_batch))
        return (filename, indices_batches)

    def jets_generator(self, files_in_batches: list) -> tuple:
        """
        Helper function to extract jet and track information from a h5 ntuple.

        Parameters
        ----------
        files_in_batches : list
            tuples of filename and tuple of start and end index of batch

        Yields
        -------
        numpy.ndarray
            jets
        numpy.ndarray
            tracks if `self.save_tracks` is set to True
        """
        for filename, batches in files_in_batches:
            if self.sample.n_jets <= 0:
                break
            logger.debug("Opening file %s.", filename)
            with h5py.File(filename, "r") as data_set:
                for batch in batches:
                    # load jets in batches
                    jets = data_set["jets"][batch[0] : batch[1]]
                    indices_to_remove = get_sample_cuts(jets, self.cuts)
                    jets = np.delete(jets, indices_to_remove)
                    # if tracks should be saved, also load them in batches
                    # TODO: update when changing to python 3.9
                    if self.save_tracks:
                        tracks = {}
                        for tracks_name in self.tracks_names:
                            trk = data_set[tracks_name][batch[0] : batch[1]]
                            trk = np.delete(trk, indices_to_remove, axis=0)
                            tracks.update({tracks_name: trk})
                    else:
                        tracks = None
                    yield (jets, tracks)

    def run(self):
        """Run over Ntuples to extract jets (and potentially also tracks)."""
        logger.info(
            "Preparing ntuples for %s %s...", self.sample.type, self.sample.category
        )

        pbar = tqdm(total=self.sample.n_jets)
        # get list of batches for each file
        files_in_batches = map(
            self.get_batches_per_file,
            self.config.preparation.get_input_files(self.sample.type),
        )
        # loop over batches for all files and load the batches separately
        n_jets_check = self.sample.n_jets
        displayed_writing_output = True
        for jets, tracks in self.jets_generator(files_in_batches):
            if jets.shape[0] == 0:
                continue
            pbar.update(jets.size)
            self.jets_loaded += jets.size
            self.sample.n_jets -= jets.size

            if self.shuffle_array:
                pbar.write("Shuffling array")

                # Init a index list
                rng_index = np.arange(len(jets))

                # Shuffle the index list
                rng = np.random.default_rng(seed=self.rnd_seed)
                rng.shuffle(rng_index)

                # Shuffle jets (and tracks)
                jets = jets[rng_index]
                if self.save_tracks:
                    for tracks_name in self.tracks_names:
                        tracks[tracks_name] = tracks[tracks_name][rng_index]

            if self.create_file:
                self.create_file = False  # pylint: disable=W0201:
                # write to file by creating dataset
                pbar.write(f"Creating output file: {self.sample.output_name}")
                with h5py.File(self.sample.output_name, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
                    if self.save_tracks:
                        for tracks_name in self.tracks_names:
                            out_file.create_dataset(
                                tracks_name,
                                data=tracks[tracks_name],
                                compression="gzip",
                                chunks=True,
                                maxshape=(None, tracks[tracks_name].shape[1]),
                            )
            else:
                # appending to existing dataset
                if displayed_writing_output:
                    pbar.write(f"Writing to output file: {self.sample.output_name}")
                with h5py.File(self.sample.output_name, "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    if self.save_tracks:
                        for tracks_name in self.tracks_names:
                            out_file[tracks_name].resize(
                                (
                                    out_file[tracks_name].shape[0]
                                    + tracks[tracks_name].shape[0]
                                ),
                                axis=0,
                            )
                            out_file[tracks_name][
                                -tracks[tracks_name].shape[0] :
                            ] = tracks[tracks_name]
                displayed_writing_output = False
            if self.sample.n_jets <= 0:
                break
        pbar.close()
        if self.sample.n_jets > 0:
            logger.warning(
                "Not as many jets selected as defined in config file. Only"
                " %i jets selected instead of %i",
                self.jets_loaded,
                n_jets_check,
            )
