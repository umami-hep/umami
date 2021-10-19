"""
Helper functions to creating hybrid hdf5 samples from ttbar and Zprime ntuples
"""
import os
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm

from umami.configuration import global_config, logger
from umami.preprocessing_tools import GetCategoryCuts, GetSampleCuts


def GetPreparationSamplePath(sample):
    """
    Retrieves the output sample path of the samples defined in the
    'samples' block in the preprocessing config.
    """
    return os.path.join(
        sample.get("f_output")["path"], sample.get("f_output")["file"]
    )


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

    def __setup(self, args):
        # check if sample is provided, otherwise exit
        if not args.sample:
            raise KeyError("Please provide --sample to prepare hybrid samples")
        # load list of samples
        samples = self.config.preparation["samples"]
        try:
            sample = samples[args.sample]
        except KeyError:
            raise KeyError(f'sample "{args.sample}" not in config file!')

        self.sample_type = sample.get("type")
        self.sample_category = sample.get("category")
        cuts = sample.get("cuts", None)
        if self.sample_category == "inclusive":
            self.cuts = cuts
        else:
            try:
                category_setup = global_config.flavour_categories[
                    self.sample_category
                ]
            except KeyError:
                raise KeyError(
                    f"Requested sample category {self.sample_category} not defined in global config."
                )

            # retrieving the cuts for the category selection
            category_cuts = GetCategoryCuts(
                category_setup["label_var"], category_setup["label_value"]
            )
            self.cuts = cuts + category_cuts
        self.n_jets_to_get = int(sample.get("n_jets", 0))
        self.save_tracks = args.tracks
        self.tracks_name = args.tracks_name
        output_path = sample.get("f_output")["path"]
        self.output_file = os.path.join(
            output_path, sample.get("f_output")["file"]
        )
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

    def get_jets(self, filename):
        """Helper function to extract jet and track information from a h5 ntuple.

        :param filename: path to the h5 ntuple
        :returns: (jets, tracks), where jets is a numpy array of jets.
                Similarly, tracks is a numpy array of tracks but is only created
                if `self.save_tracks` is set to True.
        """
        data_set = h5py.File(filename, "r")
        jets = data_set["jets"]
        logger.debug(f"Total number of jets in file: {jets.size}")
        if self.save_tracks:
            tracks = data_set[self.tracks_name]
            logger.debug(f"Tracks dataset: {self.tracks_name}")
            logger.debug(f"Total number of tracks in file: {tracks.size}")

        indices_to_remove = GetSampleCuts(jets, self.cuts)
        jets = np.delete(jets, indices_to_remove)[: self.n_jets_to_get]
        jets = jets[: self.n_jets_to_get]
        if self.save_tracks:
            tracks = np.delete(tracks, indices_to_remove, axis=0)[
                : self.n_jets_to_get
            ]
            tracks = tracks[: self.n_jets_to_get]
            return jets, tracks
        else:
            return jets, None

    def Run(self):
        """Run over Ntuples to extract jets (and potentially also tracks)"""
        logger.info("Processing ntuples...")
        pbar = tqdm(total=self.n_jets_to_get)
        for i, filename in enumerate(self.ntuples):
            if self.n_jets_to_get <= 0:
                break

            jets, tracks = self.get_jets(filename)
            pbar.update(jets.size)
            self.jets_loaded += jets.size
            self.n_jets_to_get -= jets.size

            if self.shuffle_array:
                pbar.write("Shuffling array")
                rng = np.random.default_rng(seed=42)
                rng.shuffle(jets)
                if self.save_tracks:
                    rng.shuffle(tracks)

            if self.create_file:
                self.create_file = False
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
                f"Not enough selected jets from files, only {self.jets_loaded}"
            )
