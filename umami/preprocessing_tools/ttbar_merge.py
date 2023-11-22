"""Helper functions for merging single leptonic and dileptonic ttbar samples."""
import os
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from umami.configuration import Configuration, logger


class MergeConfig(Configuration):
    """Merge config class."""

    def __init__(self, yaml_config: str) -> None:
        """Init the MergeConfig class.

        Parameters
        ----------
        yaml_config : str
            Path to yaml config file.
        """
        super().__init__(yaml_config)
        self.yaml_default_config = (
            Path(os.path.dirname(__file__))
            / "configs/preprocessing_default_config.yaml"
        )
        self.load_config_file()


class TTbarMerge:
    """
    This class merges the single and dilepton ttbar samples in the required
    ratio to match Run-2 MC non-allhadronic ttbar sample.
    """

    def __init__(self, config: object) -> None:
        self.config = config
        self.ratio = config.config.get("ratio", 5.0)
        self.out_size = config.config.get("out_size", 100_000)
        self.compression = config.config.get("compression", None)

        self.index_dir = config.config["index_dir"]
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)

        self.out_dir = config.config["out_dir"]
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # self.save_tracks = config.sampling["options"].get("save_tracks", False)
        self.save_tracks = config.config.get("save_tracks", False)
        if self.save_tracks:
            self.tracks_names = config.config[
                "tracks_names"
            ]  # self.config.sampling["options"]["tracks_names"]

        self.seed = config.config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

    def get_indices(self) -> None:
        """
        Get indices for dilepton sample to match ratio to single lepton sample.
        Indices are saved to file for later use.

        Raises
        ------
        ValueError
            If no dilepton sample found.
        """

        logger.info("Getting Sample Files")
        files_single = self.get_input_files("single_lepton")
        files_dilepton = self.get_input_files("dilepton")

        logger.info("Getting event lists")
        events_single, jets_single = event_list(files_single)
        events_dilepton, _ = event_list(files_dilepton)

        num_dilepton = int(len(events_single) / self.ratio)

        if num_dilepton > len(events_dilepton):
            raise ValueError(
                f"Requested {num_dilepton} events,"
                f"but only {len(events_dilepton)} available."
            )

        # get random subset of dilepton events
        events_dilepton_sample = self.rng.choice(
            events_dilepton, num_dilepton, replace=False
        )

        # get indices of jets from dilepton events
        logger.info("Getting indices of dilepton events subset")
        dilepton_indices = event_indices(files_dilepton, events_dilepton_sample)

        # should return all indices as a subsample is not used for single lepton events
        single_indices = event_indices(files_single, events_single)

        # get total number of jets to be written, all jets from single lepton sample and
        # subset of jets from dilepton sample
        jets_dilepton_sample = sum([len(i) for i in dilepton_indices])
        n_jets = jets_single + jets_dilepton_sample
        logger.info("Total number of jets: %s", n_jets)

        # number of output files is number of jets divided by output size
        n_files = int(np.ceil(n_jets / self.out_size))

        # split each element of index lists into n_files chunks
        for i in range(len(dilepton_indices)):  # pylint: disable=C0200
            dilepton_indices[i] = np.array_split(dilepton_indices[i], n_files)
            single_indices[i] = np.array_split(single_indices[i], n_files)

        # write a yaml file with indices for each output file
        logger.info("Writing index dictionaries")
        for i in tqdm(range(n_files)):
            # dilepton indices has shape (n_input_files, n_output_files) (index as i,j)
            # loop over n_output_files and write dict with indices for each file

            single_dict = {
                str(files_single[j]): single_indices[j][i].tolist()
                for j in range(len(files_single))
            }
            dilepton_dict = {
                str(files_dilepton[j]): dilepton_indices[j][i].tolist()
                for j in range(len(files_dilepton))
            }

            # merge the two dictionaries
            index_dict = {**single_dict, **dilepton_dict}

            # save in index directory specified in config
            index_file = Path(self.index_dir) / f"ttbar_merge_{i}.yaml"
            with open(index_file, "w") as f_index:
                yaml.dump(index_dict, f_index)

    def merge(self, file_range: list, index_dir: str) -> None:
        """
        Merge single and dilepton ttbar samples.

        Parameters
        ----------
        file_range : list
            List of output files to merge, allows splitting across multiple jobs.
        index_dir : str
            Directory containing index file dictionaries.

         Raises
        ------
        ValueError
            If the file range passed via --file_range does not
            consist of two arguments
        """

        if index_dir is None:
            index_dir = self.index_dir

        # load index files from index directory or subset if file_range is specified
        index_files = list(Path(index_dir).rglob("ttbar_merge_*.yaml"))
        if file_range is not None:
            if len(file_range) != 2:
                raise ValueError(
                    'Please pass the file range as "--filerange <file number start>'
                    ' <file number end>". The passed argument for the file range is'
                    f" {file_range}."
                )
            logger.info("Using files %s to %s", file_range[0], int(file_range[1]) - 1)
            index_files = index_files[int(file_range[0]) : int(file_range[1])]
        else:
            logger.warning(
                "You did not specify a file range for the merging procedure.Consider"
                ' using the option "--filerange <file number start> <file number end>"'
                " to split the merging across multiple jobs. The <file number end> is"
                " exclusive."
            )

        # loop over index files loading the indices and merging the samples
        logger.info("Merging samples into output files")
        for index_file in tqdm(index_files):
            with open(index_file, "r") as f_index:
                index_dict = yaml.load(f_index, Loader=yaml.FullLoader)

            # get the output file name
            output_file = Path(self.out_dir) / index_file.name.replace("yaml", "h5")

            self.write_file(index_dict, output_file)

    def write_file(self, index_dict: dict, output_file: str) -> None:
        """
        Write merged output file from passed index dictionary.

        Parameters
        ----------
        index_dict : dict
            Dictionary of indices to be used from each input file.
        output_file : str
            Name of output file.
        """

        first_file = True
        for input_file, indices in index_dict.items():
            for jets, tracks in self.load_jets_generator(
                input_file, save_tracks=self.save_tracks, indices=indices
            ):
                if first_file:
                    first_file = False
                    with h5py.File(output_file, "w") as h5file:
                        if self.save_tracks:
                            for idx, track_name in enumerate(self.tracks_names):
                                h5file.create_dataset(
                                    track_name,
                                    data=tracks[idx],
                                    compression=self.compression,
                                    maxshape=(
                                        self.out_size,
                                        tracks[idx].shape[1],
                                    ),
                                )
                        h5file.create_dataset(
                            "jets",
                            data=jets,
                            compression=self.compression,
                            maxshape=(self.out_size,),
                        )
                else:
                    with h5py.File(output_file, "a") as h5file:
                        if self.save_tracks:
                            for idx, track_name in enumerate(self.tracks_names):
                                h5file[track_name].resize(
                                    (
                                        h5file[track_name].shape[0]
                                        + tracks[idx].shape[0]
                                    ),
                                    axis=0,
                                )
                                h5file[track_name][-tracks[idx].shape[0] :] = tracks[
                                    idx
                                ]

                        h5file["jets"].resize(
                            (h5file["jets"].shape[0] + jets.shape[0]), axis=0
                        )
                        h5file["jets"][-jets.shape[0] :] = jets

    def get_input_files(self, channel: str) -> list:
        """
        Get input files for a given channel.

        Parameters
        ----------
        channel : str
            Channel to get input files for - single or dilepton.

        Returns
        -------
        list
            List of input files.

        Raises
        ------
        ValueError
            If no input files found for given channel.
        KeyError
            If no channel specified doesn't exist in config.
        """
        try:
            input_channel = self.config.config[channel]
        except KeyError as error:
            raise KeyError(f"No input files for channel {channel}") from error

        input_path = Path(input_channel["path"])
        file_list = list(input_path.rglob(input_channel["file_pattern"]))

        if len(file_list) == 0:
            raise ValueError(f"No files found for {channel} channel.")

        return file_list

    def load_jets_generator(
        self,
        input_file: str,
        chunk_size: int = 100_000,
        indices: list = None,
        save_tracks: bool = False,
    ) -> np.ndarray:
        """
        Yield jets (and tracks) from input file in batches
        with option to select events by index.

        Parameters
        ----------
        input_file : str
            Path to input file.
        chunk_size : int
            Number of jets or tracks to load at a time.
        indices : list
            Indices of jets or tracks to load.
        save_tracks : bool
            Whether to load tracks as well as jets, by default false.

        Yields
        ------
        np.ndarray
            numpy arrays of jets and tracks if save_tracks is true.
        """

        with h5py.File(input_file, "r") as data_set:
            if indices is not None:
                # if single index is requested from a file
                if isinstance(indices, int):
                    max_events = 1
                    indices = [indices]
                else:
                    max_events = len(indices)
            else:
                max_events = data_set["jets"].shape[0]

            for start in range(0, max_events, chunk_size):
                end = min(start + chunk_size, max_events)

                if indices is not None:
                    if save_tracks:
                        track_list = []
                        for track in self.tracks_names:
                            track_list.append(data_set[track][indices[start:end]])
                        yield data_set["jets"][indices[start:end]], track_list
                    else:
                        yield data_set["jets"][indices[start:end]], None
                else:
                    if save_tracks:
                        track_list = []
                        for track in self.tracks_names:
                            track_list.append(data_set[track][start:end])
                        yield data_set["jets"][start:end], track_list
                    else:
                        yield data_set["jets"][start:end], None


def event_indices(input_file_list: list, event_numbers: np.ndarray) -> list:
    """
    Get indices for each input file for jets
    from selected subsample of dilepton events.

    Parameters
    ----------
    input_file_list : list
        List of input files.
    event_numbers : np.ndarray
        Array of event numbers.

    Returns
    -------
    list
        List of numpy arrays of indices for each input file.
    """

    indices = []

    for input_file in input_file_list:
        with h5py.File(input_file, "r") as data_set:
            events_list = data_set["jets"].fields(["eventNumber"])[:]
            indices.append(
                np.where(np.isin(events_list.astype(int), event_numbers.astype(int)))[0]
            )

    return indices


def event_list(input_file_list: list) -> tuple:
    """
    Get list of unique event numbers from input files and number of jets in sample.

    Parameters
    ----------
    input_file_list : list
        List of input files.

    Returns
    -------
    np.ndarray
        Array of unique event numbers.
    int
        Number of jets in sample.
    """
    events_list = []
    n_jets = 0

    for input_file in input_file_list:
        with h5py.File(input_file, "r") as data_set:
            events_list.append(data_set["jets"].fields(["eventNumber"])[:])
            n_jets += len(data_set["jets"])

    # flatten list of numpy arrays into single numpy array
    events_list = np.concatenate(events_list, axis=0)
    event_set = np.unique(events_list)

    return event_set, n_jets
