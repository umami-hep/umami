"""
Helper functions to creating hybrid hdf5 samples from ttbar and Zprime ntuples
"""
import os
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm

from umami.configuration import global_config, logger
from umami.data_tools import GetCategoryCuts, GetSampleCuts


def GetPreparationSamplePath(sample):
    """
    Retrieves the output sample path of the samples defined in the
    'samples' block in the preprocessing config.
    """
    return os.path.join(sample.get("f_output")["path"], sample.get("f_output")["file"])


class PrepareSamples:
    """
    This class is preparing the samples for further processing defined in the
    configuration file:
        - extracts the selected jets (applying cuts: flavour, pT etc)
        - writes these iteratively to h5 output files

    This class will take the information provided in the `samples` block in
    the preprocessing config.
    """

    def __init__(self, args, config) -> None:
        self.config = config
        self.__setup(args)
        self.rnd_seed = 42

    def __setup(self, args):
        # check if sample is provided, otherwise exit
        if not args.sample:
            raise KeyError("Please provide --sample to prepare hybrid samples")
        # load list of samples
        samples = self.config.preparation["samples"]
        try:
            sample = samples[args.sample]
        except KeyError as Error:
            raise KeyError(f'sample "{args.sample}" not in config file!') from Error

        self.sample_type = sample.get("type")
        self.sample_category = sample.get("category")
        cuts = sample.get("cuts", None)
        if self.sample_category == "inclusive":
            self.cuts = cuts
        else:
            try:
                category_setup = global_config.flavour_categories[self.sample_category]
            except KeyError as Error:
                raise KeyError(
                    f"Requested sample category {self.sample_category} not"
                    " defined in global config."
                ) from Error

            # retrieving the cuts for the category selection
            category_cuts = GetCategoryCuts(
                category_setup["label_var"], category_setup["label_value"]
            )
            self.cuts = cuts + category_cuts
        self.n_jets_to_get = int(sample.get("n_jets", 0))

        # Check if tracks are used
        self.save_tracks = self.config.sampling["options"]["save_tracks"]

        # Check for tracks name. If not there, use default
        if (
            "tracks_name" in self.config.sampling["options"]
            and self.config.sampling["options"]["tracks_name"] is not None
        ):
            self.tracks_name = self.config.sampling["options"]["tracks_name"]

        else:
            self.tracks_name = "tracks"

        output_path = sample.get("f_output")["path"]
        self.output_file = os.path.join(output_path, sample.get("f_output")["file"])
        # bookkeeping variables for running over the ntuples
        self.jets_loaded = 0
        self.create_file = True
        self.shuffle_array = args.shuffle_array
        # set up ntuples
        ntuples = self.config.preparation["ntuples"]
        ntuple_path = ntuples.get(sample["type"])["path"]
        ntuple_file_pattern = ntuples.get(sample["type"])["file_pattern"]
        self.ntuples = glob(os.path.join(ntuple_path, ntuple_file_pattern))
        # ensure output path exists
        os.makedirs(output_path, exist_ok=True)
        # get size of batches
        if "batchsize" not in self.config.preparation:
            logger.warning("no batch size given. Batch size set to 1,000,000")
            self.batch_size = 1_000_000
        else:
            self.batch_size = self.config.preparation["batchsize"]

    def GetBatchesPerFile(self, filename: str):
        """
        Split the file into batches to avoid that the loaded
        data is too large
        """
        batch_size = self.batch_size
        with h5py.File(filename, "r") as data_set:
            # get total number of jets in file
            total_n_jets = len(data_set["jets"])
            logger.debug(f"Total number of jets in file: {total_n_jets}")
            # first tuple is given by (0, batch_size)
            start_batch = 0
            end_batch = batch_size
            indices_batches = [(start_batch, end_batch)]
            # get remaining tuples of indices defining the batches
            while end_batch <= total_n_jets:
                start_batch += batch_size
                end_batch = start_batch + batch_size
                indices_batches.append((start_batch, end_batch))
        return (filename, indices_batches)

    def jets_generator(self, files_in_batches):
        """Helper function to extract jet and track information from a h5 ntuple.

        :param filename: path to the h5 ntuple
        :returns: generates (jets, tracks), where jets is a numpy array of jets with
                  the size of one batch.
                Similarly, tracks is a numpy array of tracks but is only created
                if `self.save_tracks` is set to True.
        """
        for filename, batches in files_in_batches:
            if self.n_jets_to_get <= 0:
                break
            with h5py.File(filename, "r") as data_set:
                for batch in batches:
                    # load jets in batches
                    jets = data_set["jets"][batch[0] : batch[1]]
                    indices_to_remove = GetSampleCuts(jets, self.cuts)
                    jets = np.delete(jets, indices_to_remove)
                    # if tracks should be saved, also load them in batches
                    if self.save_tracks:
                        tracks = data_set[self.tracks_name][batch[0] : batch[1]]
                        tracks = np.delete(tracks, indices_to_remove, axis=0)
                    else:
                        tracks = None
                    yield (jets, tracks)

    def Run(self):
        """Run over Ntuples to extract jets (and potentially also tracks)"""
        logger.info(
            f"Preparing ntuples for {self.sample_type} {self.sample_category}..."
        )

        pbar = tqdm(total=self.n_jets_to_get)
        # get list of batches for each file
        files_in_batches = map(self.GetBatchesPerFile, self.ntuples)
        # loop over batches for all files and load the batches separately
        n_jets_check = self.n_jets_to_get
        for jets, tracks in self.jets_generator(files_in_batches):
            if jets.shape[0] == 0:
                continue
            pbar.update(jets.size)
            self.jets_loaded += jets.size
            self.n_jets_to_get -= jets.size

            if self.shuffle_array:
                pbar.write("Shuffling array")
                rng = np.random.default_rng(seed=self.rnd_seed)
                rng.shuffle(jets)
                if self.save_tracks:
                    rng = np.random.default_rng(seed=self.rnd_seed)
                    rng.shuffle(tracks)

            if self.create_file:
                self.create_file = False  # pylint: disable=W0201:
                # write to file by creating dataset
                pbar.write("Creating output file: " + self.output_file)
                with h5py.File(self.output_file, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
                    if self.save_tracks:
                        out_file.create_dataset(
                            "tracks",
                            data=tracks,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, tracks.shape[1]),
                        )
            else:
                # appending to existing dataset
                pbar.write("Writing to output file: " + self.output_file)
                with h5py.File(self.output_file, "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    if self.save_tracks:
                        out_file["tracks"].resize(
                            (out_file["tracks"].shape[0] + tracks.shape[0]),
                            axis=0,
                        )
                        out_file["tracks"][-tracks.shape[0] :] = tracks

            if self.n_jets_to_get <= 0:
                break
        pbar.close()
        if self.n_jets_to_get > 0:
            logger.warning(
                "Not as many jets selected as defined in config file. Only"
                f" {self.jets_loaded} jets selected instead of {n_jets_check}"
            )
