"""Resampling base module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import itertools
import os
from collections import Counter
from json import JSONEncoder

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from umami.configuration import logger
from umami.data_tools import compare_h5_files_variables
from umami.preprocessing_tools.Preparation import GetPreparationSamplePath


def SamplingGenerator(
    file: str,
    indices: np.ndarray,
    label: int,
    label_classes: list,
    use_tracks: bool = False,
    tracks_names: list = None,
    chunk_size: int = 10_000,
    seed: int = 42,
    duplicate: bool = False,
):
    """
    Generator to iterate over datasets based on given indices.

    This method also implements fancy indexing for H5 files by
    separating a list of indices that may contain duplicates
    into lists of uniques indices. The splitting consists in:
        - 1st list: all indices repeated at least 1 time (all).
        - 2nd list: all indices repeated at least 2 times.
        - 3rd list: ............................. 3 ......
        ...

    Parameters
    ----------
    file : str
        the path to the h5 file to read
    indices : list or numpy.array
        the indices of entries to read
    label : int
        the label of the jets being read
    label_classes : list
        the combined labelling scheme
    use_tracks : bool
        whether to store tracks, by default False
    tracks_names : list
        list of tracks collection names to use, by default None
    chunk_size : int
        the size of each chunk (last chunk might at most be twice this size),
        by default 10000
    seed : int
        random seed to use, by default 42
    duplicate : bool
        whether the reading should assume duplicates are present.
        DO NOT USE IF NO DUPLICATES ARE EXPECTED!, by default False

    Yields
    -------
    numpy.ndarray
        jets
    numpy.ndarray
        tracks, if `use_tracks` is True
    numpy.ndarray
        labels
    """
    tracks_names = tracks_names or ["tracks"]
    with h5py.File(file, "r") as f:
        start_ind = 0
        end_ind = int(start_ind + chunk_size)
        # create indices and then shuffle those to be used in the loop below
        tupled_indices = []
        while end_ind <= len(indices) or start_ind == 0:
            if end_ind + chunk_size > len(indices):
                # Missing less then a chunk, joining to last chunk
                end_ind = len(indices)
            tupled_indices.append((start_ind, end_ind))
            start_ind = end_ind
            end_ind = int(start_ind + chunk_size)
        rng = np.random.default_rng(seed=seed)
        tupled_indices = rng.choice(tupled_indices, len(tupled_indices), replace=False)
        for index_tuple in tupled_indices:
            loading_indices = indices[index_tuple[0] : index_tuple[1]]
            label_classes.append(-1)
            labels = label_binarize(
                (np.ones(index_tuple[1] - index_tuple[0]) * label),
                classes=label_classes,
            )[:, :-1]
            label_classes = label_classes[:-1]
            if duplicate and quick_check_duplicates(loading_indices):
                # Duplicate indices, fancy indexing of H5 not working, manual approach.
                list_loading_indices = []
                counting = Counter(loading_indices)
                for index in counting:
                    number_occ = counting[index]
                    while len(list_loading_indices) < number_occ:
                        list_loading_indices.append([])
                    for i in range(number_occ):
                        list_loading_indices[i].append(index)
                jet_ls = []
                track_ls = {elem: [] for elem in tracks_names}
                for i, sublist_loading_indices in enumerate(list_loading_indices):
                    jet_ls.append(f["jets"][sublist_loading_indices])
                    if use_tracks:
                        for tracks_name in tracks_names:
                            track_ls[tracks_name].append(
                                f[tracks_name][sublist_loading_indices]
                            )
                jets = np.concatenate(jet_ls)
                if use_tracks:
                    tracks = [
                        np.concatenate(track_ls[tracks_name])
                        for tracks_name in tracks_names
                    ]
                    yield jets, tracks, labels
                else:
                    yield jets, labels
            else:
                # No duplicate indices, fancy indexing of H5 working.
                if use_tracks:
                    tracks = [
                        f[tracks_name][loading_indices] for tracks_name in tracks_names
                    ]
                    yield f["jets"][loading_indices], tracks, labels
                else:
                    yield f["jets"][loading_indices], labels


def read_dataframe_repetition(
    file_df, loading_indices, duplicate, use_tracks, tracks_names="tracks"
):
    """
    Implements a fancier reading of H5 dataframe (allowing repeated indices).
    Designed to read a h5 file with jets (and tracks if use_track is true).

    Parameters
    ----------
    file_df : file
        file containing datasets
    loading_indices : list
        indices to load
    duplicate : bool
        whether the reading should assume duplicates are present.
        DO NOT USE IF NO DUPLICATES ARE EXPECTED!, by default False
    use_tracks : bool
        whether to store tracks, by default False
    tracks_names : list
        list of tracks collection names to use, by default "tracks"

    Returns
    -------
    numpy.ndarray
        jets
    numpy.ndarray
        tracks if `use_tracks` is True
    """
    if duplicate and quick_check_duplicates(loading_indices):
        # Duplicate indices, fancy indexing of H5 not working, manual approach.
        list_loading_indices = []
        counting = Counter(loading_indices)
        for index in counting:
            number_occ = counting[index]
            while len(list_loading_indices) < number_occ:
                list_loading_indices.append([])
            for i in range(number_occ):
                list_loading_indices[i].append(index)
        jet_ls = []
        track_ls = {elem: [] for elem in tracks_names}
        for i, sublist_loading_indices in enumerate(list_loading_indices):
            jet_ls.append(file_df["jets"][sublist_loading_indices])
            if use_tracks:
                for tracks_name in tracks_names:
                    track_ls[tracks_name].append(
                        file_df[tracks_name][sublist_loading_indices]
                    )

        jets = np.concatenate(jet_ls)
        if use_tracks:
            tracks = [
                np.concatenate(track_ls[tracks_name]) for tracks_name in tracks_names
            ]
            return jets, tracks
        return jets

    # No duplicate indices, fancy indexing of H5 working.
    if use_tracks:
        tracks = [file_df[tracks_name][loading_indices] for tracks_name in tracks_names]
        return (
            file_df["jets"][loading_indices],
            tracks,
        )
    return file_df["jets"][loading_indices]


class JsonNumpyEncoder(JSONEncoder):
    """This functions converts the numpy type to a json compatible format.

    Parameters
    ----------
    JSONEncoder : class
        base class from json package
    """

    def default(self, o):
        """overwriting default function of JSONEncoder class

        Parameters
        ----------
        o : numpy integer, float or ndarray
            objects from json loader

        Returns
        -------
        class
            modified JSONEncoder class
        """
        # TODO: change this when using python 3.10
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def CorrectFractions(
    N_jets: list,
    target_fractions: list,
    class_names: list = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Corrects the fractions of N classes

    Parameters
    ----------
    N_jets : list
        List actual number of available jets per class
    target_fractions : list
        List of the target fraction per class
    class_names : list, optional
        List with the class names, by default None
    verbose : bool, optional
        Decide, if more detailed output is logged, by default True

    Returns
    -------
    nJets_to_keep : np.ndarray
        Array of N_jets to keep per class

    Raises
    ------
    ValueError
        If not all N_jets entries are bigger than 0.
    ValueError
        If the 'target_fractions' don't add up to one.
    """

    if not np.all(N_jets):
        raise ValueError("Each N_jets entry needs to be >0.")
    assert len(N_jets) == len(target_fractions)
    if not np.isclose(np.sum(target_fractions), 1):
        raise ValueError("The 'target_fractions' have to sum up to 1.")

    df = pd.DataFrame(
        {
            "N_jets": N_jets,
            "target_fractions": target_fractions,
            "original_order": range(len(N_jets)),
            "target_N_jets": N_jets,
        }
    )
    if class_names is not None:
        assert len(N_jets) == len(class_names)
        df["class_names"] = class_names
    df.sort_values("target_fractions", ascending=False, inplace=True, ignore_index=True)

    # start with the class with the highest fraction and use it as reference
    for i in range(1, len(df)):
        # check next highest target fraction
        relative_fraction = (
            df.iloc[0]["target_N_jets"]
            / df.iloc[i]["N_jets"]
            / df.iloc[0]["target_fractions"]
            * df.iloc[i]["target_fractions"]
        )
        if relative_fraction == 1:
            continue
        if relative_fraction < 1:
            # need to correct now the fractions of the class with the smaller fraction
            # calculate how much jets need to be subtracted
            x = df["N_jets"][i] - (
                df["target_N_jets"][0]
                * df["target_fractions"][i]
                / df["target_fractions"][0]
            )
            df.at[i, "target_N_jets"] -= x

        else:
            # correct the higher fraction one
            x = df["N_jets"][0] - (
                df["target_N_jets"][i]
                * df["target_fractions"][0]
                / df["target_fractions"][i]
            )
            target_N_jets_reference = df["target_N_jets"][0] - x
            # adapt the fractions of all already corrected one
            target_N_jets_reference_fraction = (
                target_N_jets_reference / df["target_N_jets"][0]
            )
            df.loc[: i - 1, "target_N_jets"] = (
                df.loc[: i - 1, "target_N_jets"] * target_N_jets_reference_fraction
            ).astype(int)

    # print some information
    df.sort_values("original_order", inplace=True)
    if verbose:
        for i in range(len(df)):
            entry = df.iloc[i]
            if class_names is None:
                logger.info(
                    f"class {i}: selected"
                    f" {entry['target_N_jets']}/{entry['N_jets']} jets per"
                    " class giving the requested fraction of"
                    f" {entry['target_fractions']}"
                )
            else:
                logger.info(
                    f"{entry['class_names']}: selected"
                    f" {entry['target_N_jets']}/{entry['N_jets']} jets per"
                    " class giving the requested fraction of"
                    f" {entry['target_fractions']}"
                )
    return df["target_N_jets"].astype(int).values


def quick_check_duplicates(X: list) -> bool:
    """
    This performs a quick duplicate check in list X.
    If a duplicate is found, returns directly True.

    Parameters
    ----------
    X : list
        List with entries.

    Returns
    -------
    duplicate_is_there : bool
        Return True if one element is double in the list,
        False if not.
    """

    set_X = set()
    for item in X:
        if item in set_X:
            return True
        set_X.add(item)
    return False


def CalculateBinning(bins: list) -> np.ndarray:
    """
    Calculate and return the bin egdes for the provided
    bins.

    Parameters
    ----------
    bins : list
        Is either a list containing the np.linspace arguments,
        or a list of them

    Returns
    -------
    bin_edges : np.ndarray
        Array with the bin edges
    """
    if any(isinstance(i, list) for i in bins):
        return np.concatenate([np.linspace(*elem) for elem in bins])
    return np.linspace(*bins)


class Resampling:
    """
    Base class for all resampling methods in umami.
    """

    def __init__(self, config: object) -> None:
        """
        Initalise the Resampling class and all needed configs.

        Parameters
        ----------
        config : object
            umami.preprocessing_tools.Configuration object which stores
            the config for the preprocessing
        """

        # Get options as attributes of self
        self.config = config
        self.options = config.sampling.get("options")
        self.preparation_samples = config.preparation.get("samples")
        # filling self.[var_x var_y bins_x bins_y]
        self._GetBinning()
        self.rnd_seed = 42
        self.jets_key = "jets"
        self.save_tracks = (
            self.options["save_tracks"]
            if "save_tracks" in self.options.keys()
            else False
        )
        self.tracks_names = self.config.sampling["options"]["tracks_names"]

        # Get path attributes
        self.outfile_name = self.config.GetFileName(option="resampled")
        self.outfile_path = self.config.config["parameters"]["sample_path"]
        self.resampled_path = self.config.config["parameters"]["file_path"]

        # Check if the directory for the outfile is existing
        out_dir = os.path.dirname(self.outfile_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Check if the directory for the resampled, scaled files
        # (normally preprocessed/) exists
        if os.path.dirname(self.resampled_path):
            os.makedirs(os.path.dirname(self.resampled_path), exist_ok=True)

        # Get class labels from sampling/preparation.
        # Try/Except here for backward compatibility
        try:
            self.class_labels_map = {
                label: label_id
                for label_id, label in enumerate(config.sampling["class_labels"])
            }

        except KeyError:
            self.class_labels_map = {
                label: label_id
                for label_id, label in enumerate(config.preparation["class_labels"])
            }
            logger.warning(
                "Deprecation Warning: class_labels are given in preparation"
                " and not in sampling block! Consider moving this to"
                " the sampling block in your config!"
            )

    def _GetBinning(self):
        """
        Retrieves the binning and the corresponding variables which are used
        for the resampling. Saves the bins and variables to class variables

        Raises
        ------
        ValueError
            If more than two resampling variables are given.
        """

        # Get the sampling variables
        sampling_variables = self.options.get("sampling_variables")

        # Check that not more than two variables are given
        if len(sampling_variables) != 2:
            raise ValueError("Resampling is so far only supporting 2 variables.")

        # Get the variable names as list
        variables = [list(elem.keys())[0] for elem in sampling_variables]

        # Get the two variables from the list
        self.var_x = variables[0]
        self.var_y = variables[1]

        # Calculate the binning of the variables with the provided info about
        # the binning
        logger.info(f"Using {variables[0]} and {variables[1]} for resampling.")
        self.bins_x = CalculateBinning(sampling_variables[0][self.var_x]["bins"])
        self.bins_y = CalculateBinning(sampling_variables[1][self.var_y]["bins"])

        # Get number of bins
        self.nbins = np.array([len(self.bins_x), len(self.bins_y)])

    def GetBins(self, x: np.ndarray, y: np.ndarray):
        """
        Calculates the bin statistics for a 2D histogram. This post might be
        helpful to understand the flattened bin numbering:
        https://stackoverflow.com/questions/63275441/can-i-get-binned-statistic-2d-to-return-bin-numbers-for-only-bins-in-range

        Parameters
        ----------
        x : np.ndarray
            Array with values from variable x.
        y : np.ndarray
            Array with values from variable y and same length as x

        Returns
        -------
        binnumber : np.ndarray
            Array with bin number of each jet with same length as x and y
        bins_indices_flat : np.ndarray
            Array with flat bin numbers mapped from 2D with length nBins
        statistic : np.ndarray
            Array with counts per bin, length nBins
        """

        # Assert same shape of x and y
        assert len(x) == len(y)

        # Get the statistic and binnumbers for the provided binning
        statistic, _, _, binnumber = binned_statistic_2d(
            x=x,
            y=y,
            values=x,
            statistic="count",
            bins=[self.bins_x, self.bins_y],
        )

        # Get the flat bin indices for 2d
        bins_indices_flat_2d = np.indices(self.nbins - 1) + 1

        # Get the flat bin indices for 1d
        bins_indices_flat = np.ravel_multi_index(
            bins_indices_flat_2d, self.nbins + 1
        ).flatten()

        # Return the binnumer, the flat bin indicies and the flatten statistic
        return binnumber, bins_indices_flat, statistic.flatten()

    def ResamplingGenerator(
        self,
        file: str,
        indices: list,
        label: int,
        label_classes: list,
        variables: dict,
        use_tracks: bool = False,
        tracks_names: list = None,
        chunk_size: int = 10000,
        seed: int = 42,
    ):
        """Generator for resampling

        Parameters
        ----------
        file : str
            path to h5 file
        indices : list
            list of indices which need to be loaded
        label : int
            flavour label
        label_classes : list
            list with all label classes used, info is necessary for the binarisation
        variables : dict
            variables per dataset which should be used
        use_tracks : bool, optional
            writing out tracks, by default False
        tracks_names : list, optional
            list containing the tracks collection names to write
        chunk_size : int, optional
            size of loaded chunks, by default 10_000
        seed : int, optional
            random seed, by default 42

        Yields
        -------
        numpy.ndarray
            jets
        numpy.ndarray
            tracks if `use_tracks` is True
        numpy.ndarray
            binarised labels
        """

        # Get the tracks name. If not provided, use default
        tracks_names = tracks_names or ["tracks"]

        # Open the h5 file
        with h5py.File(file, "r") as f:

            # Set the start and end index for the given chunk
            start_ind = 0
            end_ind = int(start_ind + chunk_size)

            # create indices and then shuffle those to be used in the loop below
            tupled_indices = []
            while end_ind <= len(indices) or start_ind == 0:
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            # Get a random generator with specified seed
            rng = np.random.default_rng(seed=seed)

            # Mix the chunks
            tupled_indices = rng.choice(
                tupled_indices, len(tupled_indices), replace=False
            )

            # Iterate over the indicies
            for index_tuple in tupled_indices:
                loading_indices = indices[index_tuple[0] : index_tuple[1]]
                label_classes.append(-1)
                # One hot encode the labels
                labels = label_binarize(
                    (np.ones(index_tuple[1] - index_tuple[0]) * label),
                    classes=label_classes,
                )[:, :-1]
                label_classes = label_classes[:-1]
                # Yield the jets and labels
                # If tracks are used, also yield the tracks
                if use_tracks:
                    tracks = [
                        f[tracks_name].fields(variables[tracks_name])[loading_indices]
                        for tracks_name in tracks_names
                    ]
                    yield f["jets"].fields(variables["jets"])[
                        loading_indices
                    ], tracks, labels
                else:
                    yield f["jets"].fields(variables["jets"])[loading_indices], labels

    def WriteFile(self, indices: dict, chunk_size: int = 10_000):
        """
        Takes the indices as input calculated in the GetIndices function and
        reads them in and writes them to disk.
        Writes the selected jets from the samples to disk

        Parameters
        ----------
        indices : dict
            Dict of indices as returned by the GetIndices function.
        chunk_size : int, optional
            Size of loaded chunks, by default 10_000

        Raises
        ------
        TypeError
            If concatenated samples have different shape.
        TypeError
            If the used samples don't have the same content of variables.
        """

        # reading chunks of each sample in here
        # adding already here a column with labels
        sample_lengths = [len(indices[sample]) for sample in indices]
        max_sample = np.amax(sample_lengths)
        n_chunks = round(max_sample / chunk_size + 0.5)
        chunk_sizes = np.asarray(sample_lengths) / n_chunks

        # check if all specified samples have the same variables
        sample_paths = list(self.sample_file_map.values())
        common_vars = {}
        for dataset in ["jets"] + self.tracks_names if self.save_tracks else ["jets"]:
            common_vars_i, diff_vars = compare_h5_files_variables(
                *sample_paths, key=dataset
            )
            common_vars[dataset] = common_vars_i
            logger.debug(f"Common vars in {dataset}: {common_vars_i}")
            logger.debug(f"Diff vars in {dataset}: {diff_vars}")
            if diff_vars:
                logger.warning(
                    f"The {dataset} in your specified samples don't have the same "
                    f" variables. The following variables are different: {diff_vars}"
                )
                logger.warning("These variables are ignored in all further steps.")

        generators = [
            self.ResamplingGenerator(
                file=self.sample_file_map[sample],
                indices=indices[sample],
                chunk_size=chunk_sizes[i],
                label=self.class_labels_map[
                    self.preparation_samples[sample]["category"]
                ],
                label_classes=list(range(len(self.class_labels_map))),
                variables=common_vars,
                use_tracks=self.save_tracks,
                tracks_names=self.tracks_names,
                seed=self.rnd_seed + i,
            )
            for i, sample in enumerate(indices)
        ]
        create_file = True
        chunk_counter = 0
        logger.info(f"Writing to file {self.outfile_name}")
        pbar = tqdm(total=np.sum(sample_lengths))
        while chunk_counter < n_chunks + 1:
            for i, _ in enumerate(indices):
                try:
                    if self.save_tracks:
                        if i == 0:
                            jets, tracks, labels = next(generators[i])
                        else:
                            try:
                                jets_i, tracks_i, labels_i = next(generators[i])
                                labels = np.concatenate([labels, labels_i])
                                jets = np.concatenate([jets, jets_i])
                                tracks = np.concatenate([tracks, tracks_i], axis=1)
                            except TypeError as invalid_type:
                                if str(invalid_type) == "invalid type promotion":
                                    raise TypeError(
                                        "It seems that the samples you are "
                                        "using are not compatible with each other. "
                                        "Check that they contain all the same "
                                        "variables."
                                    ) from invalid_type
                                raise TypeError(str(invalid_type)) from invalid_type

                    else:
                        if i == 0:
                            jets, labels = next(generators[i])
                        else:
                            jets_i, labels_i = next(generators[i])
                            labels = np.concatenate([labels, labels_i])
                            jets = np.concatenate([jets, jets_i])
                except StopIteration:
                    if i <= len(indices) - 1:
                        continue
                    break
            pbar.update(jets.size)

            # Init a index list
            rng_index = np.arange(len(jets))

            # Shuffle the index list
            rng = np.random.default_rng(seed=self.rnd_seed)
            rng.shuffle(rng_index)

            # Shuffle the jets, labels (and tracks)
            jets = jets[rng_index]
            labels = labels[rng_index]
            if self.save_tracks:
                tracks = [trk[rng_index] for trk in tracks]

            if create_file:
                create_file = False
                # write to file by creating dataset
                with h5py.File(self.outfile_name, "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
                    out_file.create_dataset(
                        "labels",
                        data=labels,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, labels.shape[1]),
                    )
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file.create_dataset(
                                tracks_name,
                                data=tracks[i],
                                compression="gzip",
                                chunks=True,
                                maxshape=(None, tracks[i].shape[1]),
                            )
            else:
                # appending to existing dataset
                with h5py.File(self.outfile_name, "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    out_file["labels"].resize(
                        (out_file["labels"].shape[0] + labels.shape[0]),
                        axis=0,
                    )
                    out_file["labels"][-labels.shape[0] :] = labels
                    if self.save_tracks:
                        for i, tracks_name in enumerate(self.tracks_names):
                            out_file[tracks_name].resize(
                                (out_file[tracks_name].shape[0] + tracks[i].shape[0]),
                                axis=0,
                            )
                            out_file[tracks_name][-tracks[i].shape[0] :] = tracks[i]
            chunk_counter += 1
        pbar.close()


class ResamplingTools(Resampling):
    """Helper class for resampling."""

    def InitialiseSamples(
        self,
        n_jets: int = None,
    ) -> None:
        """
        At this point the arrays of the 2 variables are loaded which are used
        for the sampling and saved into class variables.

        Parameters
        ----------
        n_jets : int, optional
            If the custom_njets_initial are not set, use this value to decide
            how much jets are loaded from each sample. By default None

        Raises
        ------
        KeyError
            If the samples are not correctly specified.
        """

        self.samples = {}
        try:
            samples = self.options["samples"]
        except KeyError as error:
            raise KeyError(
                "You chose the 'count', 'weight' or 'importance_no_replace' option "
                "for the sampling but didn't provide the samples to use. "
                "Please specify them in the configuration file!"
            ) from error

        # list of sample classes, bjets, cjets, etc
        valid_class_categories = self.GetValidClassCategories(samples)
        self.class_categories = valid_class_categories[next(iter(samples.keys()))]
        # map of sample categories and indexes as IDs
        self.sample_categories = {
            elem: i for i, elem in enumerate(list(valid_class_categories.keys()))
        }
        # map of sample categories
        self.sample_map = {elem: {} for elem in list(valid_class_categories.keys())}
        self.sample_file_map = {}
        for sample_category in self.sample_categories:
            sample_id = self.sample_categories[sample_category]
            self.samples[sample_category] = []
            for sample in samples[sample_category]:
                preparation_sample = self.preparation_samples.get(sample)
                preparation_sample_path = GetPreparationSamplePath(preparation_sample)
                self.sample_file_map[sample] = preparation_sample_path
                logger.info(
                    f"Loading sampling variables from {preparation_sample_path}"
                )
                with h5py.File(preparation_sample_path, "r") as f:

                    # Check for custom initial jets
                    if (
                        "custom_njets_initial" in self.options
                        and self.options["custom_njets_initial"] is not None
                        and sample in list(self.options["custom_njets_initial"])
                    ):
                        nJets_initial = int(
                            self.options["custom_njets_initial"][sample]
                        )
                        logger.debug(
                            f"Using custom_njets_initial for {sample} of "
                            f"{nJets_initial} from config"
                        )

                    # Check if the parameter is given in init (for pdf sampling)
                    elif n_jets is not None:
                        nJets_initial = n_jets

                    else:
                        nJets_initial = None

                    jets_x = np.asarray(f["jets"].fields(self.var_x)[:nJets_initial])
                    jets_y = np.asarray(f["jets"].fields(self.var_y)[:nJets_initial])
                logger.info(
                    f"Loaded {len(jets_x)}"
                    f" {preparation_sample.get('category')} jets from"
                    f" {sample}."
                )
                # construct a flat array with 5 columns:
                # x, y, index, sample_id, sample_class
                sample_vector = np.asarray(
                    [
                        jets_x,
                        jets_y,
                        range(len(jets_x)),
                        np.ones(len(jets_x)) * sample_id,
                        np.ones(len(jets_x)) * self.sample_categories[sample_category],
                    ]
                ).T
                self.samples[sample_category].append(
                    {
                        "file": preparation_sample_path,
                        "sample_vector": sample_vector,
                        "category": preparation_sample.get("category"),
                        "sample": sample,
                        "sample_id": sample_id,
                    }
                )
                self.sample_map[sample_category] = {
                    **self.sample_map[sample_category],
                    preparation_sample.get("category"): sample,
                }

    def GetValidClassCategories(self, samples: dict):
        """
        Helper function to check sample categories requested in resampling were
        also defined in the sample preparation step. Returns sample classes.

        Parameters
        ----------
        samples : dict
            Dict wih the samples

        Returns
        -------
        check_consistency : dict
            Dict with the consistency check results.

        Raises
        ------
        KeyError
            If the sample requested is not in the preparation block.
        RuntimeError
            If your specified samples in the sampling block don't have
            same samples in each category.
        """
        # ttbar or zprime are the categories, samples are bjets, cjets, etc.
        categories = samples.keys()
        check_consistency = {}
        for category in categories:
            check_consistency[category] = []
            for sample in samples[category]:
                preparation_sample = self.preparation_samples.get(sample)
                if preparation_sample is None:
                    raise KeyError(
                        f"'{sample}' was requested in sampling/samples block, "
                        "however, it was not defined in preparation/samples in"
                        "the preprocessing config file!"
                    )
                check_consistency[category].append(preparation_sample["category"])
        combs = list(itertools.combinations(check_consistency.keys(), 2))
        combs_check = [
            sorted(check_consistency[elem[0]]) == sorted(check_consistency[elem[1]])
            for elem in combs
        ]
        if not all(combs_check):
            raise RuntimeError(
                "Your specified samples in the sampling/samples "
                "block need to have the same samples in each sample category."
            )
        return check_consistency

    def ConcatenateSamples(self):
        """
        Takes initialized object from InitialiseSamples() and concatenates
        samples with the same category into dict which contains the
        samplevector: array(sample_size x 5)
        with pt, eta, jet_count, sample_id (ttbar:0, zprime:1), sample_class

        Returns
        -------
        self.concat_samples = {
            "bjets": {"jets": array(sample_size x 5)},
            "cjets": {"jets": array(sample_size x 5)},
            ...
        }

        """

        concat_samples = {elem: {"jets": None} for elem in self.class_categories}

        for sample_category in self.samples:  # pylint: disable=C0206
            for sample in self.samples[sample_category]:
                if concat_samples[sample["category"]]["jets"] is None:
                    concat_samples[sample["category"]]["jets"] = sample["sample_vector"]

                else:
                    concat_samples[sample["category"]]["jets"] = np.concatenate(
                        [
                            concat_samples[sample["category"]]["jets"],
                            sample["sample_vector"],
                        ]
                    )
        self.concat_samples = concat_samples
        return concat_samples

    def GetPtEtaBinStatistics(self):
        """Retrieve pt and eta bin statistics."""
        # calculate the 2D bin statistics for each sample and add it to
        # concat_samples dict with keys 'binnumbers','bin_indices_flat', 'stat'
        for class_category in self.class_categories:
            binnumbers, ind, stat = self.GetBins(
                self.concat_samples[class_category]["jets"][:, 0],
                self.concat_samples[class_category]["jets"][:, 1],
            )
            self.concat_samples[class_category]["binnumbers"] = binnumbers
            self.concat_samples[class_category]["stat"] = stat
            self.bin_indices_flat = ind
