import itertools
import os
import pickle
import warnings
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from umami.configuration import global_config, logger
from umami.preprocessing_tools import GetPreparationSamplePath

from .utils import ResamplingPlots


def SamplingGenerator(
    file: str,
    indices,
    label: int,
    label_classes: list,
    use_tracks: bool = False,
    chunk_size: int = 10000,
    seed=23,
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
    file: str, the path to the h5 file to read
    indices: list or numpy array, the indices of entries to read
    label: int, the label of the jets being read
    label_classes: list or numpy array, the combined labelling scheme
    use_tracks: bool, whether to store tracks
    chunk_size: int, the size of each chunk
                (last chunk might at most be twice this size)
    seed: int, random seed to use
    duplicate: bool, whether the reading should assume duplicates are
                present. DO NOT USE IF NO DUPLICATES ARE EXPECTED.

    Returns
    -------
    An iterator to read the jets datasets (+ labels + tracks)

    Yields
    -------
    jets, (tracks), labels arrays.

    """
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
        tupled_indices = rng.choice(
            tupled_indices, len(tupled_indices), replace=False
        )
        for index_tuple in tupled_indices:
            loading_indices = indices[index_tuple[0] : index_tuple[1]]
            labels = label_binarize(
                (np.ones(index_tuple[1] - index_tuple[0]) * label),
                classes=label_classes,
            )
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
                jet_ls, track_ls = [], []
                for i, sublist_loading_indices in enumerate(
                    list_loading_indices
                ):
                    jet_ls.append(f["jets"][sublist_loading_indices])
                    if use_tracks:
                        track_ls.append(f["tracks"][sublist_loading_indices])
                jets = np.concatenate(jet_ls)
                if use_tracks:
                    tracks = np.concatenate(track_ls)
                    yield jets, tracks, labels
                else:
                    yield jets, labels
            else:
                # No duplicate indices, fancy indexing of H5 working.
                if use_tracks:
                    yield f["jets"][loading_indices], f["tracks"][
                        loading_indices
                    ], labels
                else:
                    yield f["jets"][loading_indices], labels


def read_dataframe_repetition(file_df, loading_indices, duplicate, use_tracks):
    """
    Implements a fancier reading of H5 dataframe (allowing repeated indices).
    Designed to read a h5 file with jets (and tracks if use_track is true).
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
        jet_ls, track_ls = [], []
        for i, sublist_loading_indices in enumerate(list_loading_indices):
            jet_ls.append(file_df["jets"][sublist_loading_indices])
            if use_tracks:
                track_ls.append(file_df["tracks"][sublist_loading_indices])
        jets = np.concatenate(jet_ls)
        if use_tracks:
            tracks = np.concatenate(track_ls)
            return jets, tracks
        else:
            return jets
    else:
        # No duplicate indices, fancy indexing of H5 working.
        if use_tracks:
            return (
                file_df["jets"][loading_indices],
                file_df["tracks"][loading_indices],
            )
        else:
            return file_df["jets"][loading_indices]


def CorrectFractions(
    N_jets: list,
    target_fractions: list,
    class_names: list = None,
    verbose: bool = True,
):
    """
    Corrects the fractions of N classes

    Parameters
    ----------
    N_jets: list actual number of available jets per class
    target_fractions: list of the target fraction per class
    Returns
    -------
    target_N_jets: list of N_jets to keep per class
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
    df.sort_values(
        "target_fractions", ascending=False, inplace=True, ignore_index=True
    )

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
        elif relative_fraction < 1:
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
            df.loc[: i - 1, "target_N_jets"] = int(
                df.loc[: i - 1, "target_N_jets"]
                * target_N_jets_reference_fraction
            )

    # print some information
    df.sort_values("original_order", inplace=True)
    if verbose:
        for i in range(len(df)):
            entry = df.iloc[i]
            if class_names is None:
                logger.info(
                    f"class {i}: selected {entry['target_N_jets']}/{entry['N_jets']} jets per class giving the requested fraction of {entry['target_fractions']}"
                )
            else:
                logger.info(
                    f"{entry['class_names']}: selected {entry['target_N_jets']}/{entry['N_jets']} jets per class giving the requested fraction of {entry['target_fractions']}"
                )
    return df["target_N_jets"].astype(int).values


def quick_check_duplicates(X):
    """
    This performs a quick duplicate check in list X.
    If a duplicate is found, returns directly True.
    """
    set_X = set()
    for item in X:
        if item in set_X:
            return True
        set_X.add(item)
    return False


def CalculateBinning(bins: list):
    """
    Parameters
    ----------
    bins: is either a list containing the np.linspace arguments,
          or a list of them
    Returns
    -------
    bin edges
    """
    if any(isinstance(i, list) for i in bins):
        return np.concatenate([np.linspace(*elem) for elem in bins])
    return np.linspace(*bins)


class Resampling(object):
    """
    Base class for all resampling methods in umami.
    """

    def __init__(self, config) -> None:
        """
        Parameters
        ----------
        sampling_options: dict with the sampling options from the preprocessing
                          config file
        preparation_samples: dict with the preparation options from the
                             preprocessing config file
        Returns
        -------
        """
        self.config = config
        self.options = config.sampling.get("options")
        self.preparation_samples = config.preparation.get("samples")
        # filling self.[var_x var_y bins_x bins_y]
        self._GetBinning()
        self.rnd_seed = 42
        self.jets_key = "jets"
        self.class_labels_map = {
            label: label_id
            for label_id, label in enumerate(
                config.preparation["class_labels"]
            )
        }
        self.save_tracks = (
            self.options["save_tracks"]
            if "save_tracks" in self.options.keys()
            else False
        )
        self.outfile_name = self.config.GetFileName(option="resampled")
        self.outfile_path = self.config.config["parameters"]["sample_path"]

    def _GetBinning(self):
        """
        Retrieves the binning and the corresponding variables which are used
        for the resampling.
        Parameters
        ----------
        Returns
        -------
        saves the bins and variables to class variables
        """
        sampling_variables = self.options.get("sampling_variables")
        if len(sampling_variables) != 2:
            raise ValueError(
                "Resampling is so far only supporting 2 variables."
            )
        vars = [list(elem.keys())[0] for elem in sampling_variables]
        self.var_x = vars[0]
        self.var_y = vars[1]
        logger.info(f"Using {vars[0]} and {vars[1]} for resampling.")
        self.bins_x = CalculateBinning(
            sampling_variables[0][self.var_x]["bins"]
        )
        self.bins_y = CalculateBinning(
            sampling_variables[1][self.var_y]["bins"]
        )
        self.nbins = np.array([len(self.bins_x), len(self.bins_y)])

    def GetBins(self, x, y):
        """
        Parameters
        ----------
        x: numpy array with values from variable x
        y: numpy array with values from variable y and same length as x
        Returns
        -------
        binnumber: numpy array with bin number of each jet with same length
                   as x and y
        bins_indices_flat: numpy array with flat bin numbers mapped from 2D
                           with length nBins
        statistic: numpy array with counts per bin, legth nBins
        """
        assert len(x) == len(y)
        statistic, _, _, binnumber = binned_statistic_2d(
            x=x,
            y=y,
            values=x,
            statistic="count",
            bins=[self.bins_x, self.bins_y],
        )

        bins_indices_flat_2d = np.indices(self.nbins - 1) + 1
        bins_indices_flat = np.ravel_multi_index(
            bins_indices_flat_2d, self.nbins + 1
        ).flatten()
        return binnumber, bins_indices_flat, statistic.flatten()

    def ResamplingGenerator(
        self,
        file: str,
        indices: list,
        label: int,
        label_classes: list,
        use_tracks: bool = False,
        chunk_size: int = 10000,
        seed=23,
    ):
        raise NotImplementedError

    def WriteFile(self, indices: dict, chunk_size: int = 4000):
        """
        Takes the indices as input calculated in the GetIndices function and
        reads them in and writes them to disk.

        Parameters
        ----------
        indices: dict of indices as returned by the GetIndices function
        Returns
        -------
        Writes the selected jets from the samples to disk
        """
        # reading chunks of each sample in here
        # adding already here a column with labels
        sample_lengths = [len(indices[sample]) for sample in indices]
        max_sample = np.amax(sample_lengths)
        n_chunks = round(max_sample / chunk_size + 0.5)
        chunk_sizes = np.asarray(sample_lengths) / n_chunks

        generators = [
            self.ResamplingGenerator(
                self.sample_file_map[sample],
                indices[sample],
                chunk_size=chunk_sizes[i],
                label=self.class_labels_map[
                    self.preparation_samples[sample]["category"]
                ],
                label_classes=list(range(len(self.class_labels_map))),
                use_tracks=self.save_tracks,
                seed=23 + i,
            )
            for i, sample in enumerate(indices)
        ]
        create_file = True
        chunk_counter = 0
        logger.info(f"Writing to file {self.outfile_name}")
        pbar = tqdm(total=np.sum(sample_lengths))
        while chunk_counter < n_chunks + 1:
            for i, sample in enumerate(indices):
                try:
                    if self.save_tracks:
                        if i == 0:
                            jets, tracks, labels = next(generators[i])
                        else:
                            jets_i, tracks_i, labels_i = next(generators[i])
                            labels = np.concatenate([labels, labels_i])
                            jets = np.concatenate([jets, jets_i])
                            tracks = np.concatenate([tracks, tracks_i])
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
                    else:
                        break
            pbar.update(jets.size)
            rng = np.random.default_rng(seed=self.rnd_seed)
            rng.shuffle(jets)
            rng.shuffle(labels)
            if self.save_tracks:
                rng.shuffle(tracks)

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
                        out_file.create_dataset(
                            "tracks",
                            data=tracks,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, tracks.shape[1]),
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
                        out_file["tracks"].resize(
                            (out_file["tracks"].shape[0] + tracks.shape[0]),
                            axis=0,
                        )
                        out_file["tracks"][-tracks.shape[0] :] = tracks
            chunk_counter += 1
        pbar.close()


class PDFSampling(Resampling):
    """
    An importance sampling approach using ratios between distributions to sample
    and a target as importance weights.
    """

    def __init__(self, config) -> None:
        super(PDFSampling, self).__init__(config)
        self.inter_func_dict = {}
        self._ratio_dict = {}
        self._bin_edges_dict = {}
        # Setting some limits: important for good spline approximation
        sampling_var = self.options.get("sampling_variables")
        bin_info = []
        ranges_info = []
        extreme_ranges = []
        for samp_var in sampling_var:
            for var_ind, var in enumerate(list(samp_var.keys())):
                while len(bin_info) < len(samp_var[var]["bins"]):
                    bin_info.append([])
                    ranges_info.append([])
                themin, themax = 1e8, 0
                for i, bin_cat in enumerate(samp_var[var]["bins"]):
                    bin_info[i].append(bin_cat[2])
                    ranges_info[i].append((bin_cat[0], bin_cat[1]))
                    if bin_cat[0] < themin:
                        themin = bin_cat[0]
                    if bin_cat[1] > themax:
                        themax = bin_cat[1]
                extreme_ranges.append([themin, themax])
        self.limit = {
            "bins": bin_info,
            "ranges": ranges_info,
            "extreme_ranges": extreme_ranges,
        }
        self.number_to_sample = {}
        self.iterations = self.options.get("iterations")

    @property
    def Ratio(self):
        return self._ratio_dict

    @property
    def Inter_Func_Dict(self):
        return self.inter_func_dict

    def Load_Samples_Generator(self, sample_category, sample_id, chunk_size):
        sample, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        in_file = GetPreparationSamplePath(preparation_sample)
        samples = {}
        with h5py.File(in_file, "r") as f:
            Njets_initial = None
            if (
                "custom_njets_initial" in self.options
                and self.options["custom_njets_initial"] is not None
                and sample in list(self.options["custom_njets_initial"])
            ):
                Njets_initial = int(
                    self.options["custom_njets_initial"][sample]
                )
            else:
                Njets_initial = len(f["jets"])

            start_ind = 0
            end_ind = int(start_ind + chunk_size)
            # create indices and then shuffle those to be used in the loop below
            tupled_indices = []
            while end_ind <= Njets_initial or start_ind == 0:
                if end_ind + chunk_size > Njets_initial:
                    # Missing less then a chunk, joining to last chunk
                    end_ind = Njets_initial
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:
                loading_indices = np.arange(index_tuple[0], index_tuple[1])
                jets_x = np.asarray(f["jets"][self.var_x])[loading_indices]
                jets_y = np.asarray(f["jets"][self.var_y])[loading_indices]
                sample_vector = np.asarray(
                    [
                        jets_x,
                        jets_y,
                        range(len(jets_x)),
                        np.ones(len(jets_x)) * sample_id,
                        np.ones(len(jets_x))
                        * self.sample_categories[sample_category],
                    ]
                ).T
                samples = {
                    "file": in_file,
                    "sample_vector": sample_vector,
                    "category": preparation_sample.get("category"),
                }
                yield sample, samples, index_tuple[1] != Njets_initial

    def Load_Index_Generator(self, in_file, chunk_size):
        with h5py.File(in_file, "r") as f:
            Nindices = len(f["jets"])
            start_ind = 0
            end_ind = int(start_ind + chunk_size)
            # create indices and then shuffle those to be used in the loop below
            tupled_indices = []
            while end_ind <= Nindices or start_ind == 0:
                if end_ind + chunk_size > Nindices:
                    # Missing less then a chunk, joining to last chunk
                    end_ind = Nindices
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:
                loading_indices = np.arange(index_tuple[0], index_tuple[1])
                indices = np.asarray(f["jets"])[loading_indices]
                yield indices, index_tuple[1] != Nindices

    def Load_Samples(self, sample_category, sample_id):
        sample, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        in_file = GetPreparationSamplePath(preparation_sample)
        samples = {}
        with h5py.File(in_file, "r") as f:
            Njets_initial = None
            if (
                "custom_njets_initial" in self.options
                and self.options["custom_njets_initial"] is not None
                and sample in list(self.options["custom_njets_initial"])
            ):
                Njets_initial = int(
                    self.options["custom_njets_initial"][sample]
                )
            jets_x = np.asarray(f["jets"][self.var_x])[:Njets_initial]
            jets_y = np.asarray(f["jets"][self.var_y])[:Njets_initial]
            logger.info(
                f"Loaded {len(jets_x)} {preparation_sample.get('category')} jets from {sample}."
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

        samples = {
            "file": in_file,
            "sample_vector": sample_vector,
            "category": preparation_sample.get("category"),
        }
        return sample, samples

    def File_to_histogram(
        self,
        sample_category,
        category_ind,
        sample_id,
        iterator=False,
        chunk_size=1e6,
    ):
        available_numbers = 0
        if iterator:
            generator = self.Load_Samples_Generator(
                sample_category=sample_category,
                sample_id=sample_id,
                chunk_size=chunk_size,
            )

            load_chunk = True
            chunk_counter = 0
            while load_chunk:
                try:
                    _, target_dist, load_more = next(generator)
                except StopIteration:
                    break
                load_chunk = load_more
                if chunk_counter == 0:
                    h_target, x_bin_edges, y_bin_edges = np.histogram2d(
                        target_dist["sample_vector"][:, 0],
                        target_dist["sample_vector"][:, 1],
                        bins=self.limit["bins"][category_ind],
                        range=self.limit["ranges"][category_ind],
                    )
                else:
                    new_hist, _, _ = np.histogram2d(
                        target_dist["sample_vector"][:, 0],
                        target_dist["sample_vector"][:, 1],
                        bins=self.limit["bins"][category_ind],
                        range=self.limit["ranges"][category_ind],
                    )
                    h_target += new_hist

                available_numbers += len(target_dist["sample_vector"])
                chunk_counter += 1
        else:
            _, target_dist = self.Load_Samples(sample_category, sample_id)
            available_numbers = len(target_dist["sample_vector"])
            h_target, x_bin_edges, y_bin_edges = np.histogram2d(
                target_dist["sample_vector"][:, 0],
                target_dist["sample_vector"][:, 1],
                bins=self.limit["bins"][category_ind],
                range=self.limit["ranges"][category_ind],
            )

        _, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        return_dict = {
            "hist": h_target,
            "xbins": x_bin_edges,
            "ybins": y_bin_edges,
            "available_numbers": available_numbers,
            "category": preparation_sample.get("category"),
        }
        if iterator:
            return return_dict
        else:
            return_dict["target_dist"] = target_dist
            return return_dict

    def Initialise_Flavour_Samples(self):
        """
        Initialising input files: this one just creates the map. (basd on UnderSampling one).
        Parameters
        ----------

        Returns
        -------
        At this point the arrays of the 2 variables are loaded which are used
        for the sampling and saved into class variables.
        """

        try:
            samples = self.options["samples"]
        except KeyError:
            raise KeyError(
                "You chose the 'pdf' option for the sampling but didn't"
                "provide the samples to use. Please specify them in the"
                "configuration file!"
            )
        # saving a list of sample categories with associated IDs
        self.sample_categories = {
            elem: i for i, elem in enumerate(list(samples.keys()))
        }
        self.CheckSampleConsistency(samples)

        self.upsampling_max_rate = {}
        self.sample_map = {elem: {} for elem in list(samples.keys())}
        self.sample_file_map = {}
        self.max_upsampling = {}

        sample_id = 0
        for sample_category in self.sample_categories:
            self.sample_file_map[sample_category] = {}
            for sample_id, sample in enumerate(samples[sample_category]):
                preparation_sample = self.preparation_samples.get(sample)
                self.sample_file_map[sample_category][sample_id] = (
                    sample,
                    preparation_sample,
                )
                # If max upsampling ratio set, need to save the number of jets available
                if (
                    "max_upsampling_ratio" in self.options
                    and self.options["max_upsampling_ratio"] is not None
                    and sample in list(self.options["max_upsampling_ratio"])
                ):
                    max_upsampling = float(
                        self.options["max_upsampling_ratio"][sample]
                    )
                    in_file = GetPreparationSamplePath(preparation_sample)
                    with h5py.File(in_file, "r") as f:
                        num_available = len(f["jets"])
                    self.max_upsampling[sample] = (
                        max_upsampling,
                        num_available,
                    )

    def CheckSampleConsistency(self, samples):
        """
        Helper function to check if each sample category has the same amount
        of samples with same category (e.g. Z' and ttbar both have b, c & light)
        """
        check_consistency = {elem: [] for elem in self.sample_categories}
        for category in self.sample_categories:
            for sample in samples[category]:
                preparation_sample = self.preparation_samples.get(sample)
                if preparation_sample is None:
                    raise KeyError(
                        f"'{sample}' was requested in sampling/samples block,"
                        "however, it is not defined in preparation/samples in"
                        "the preprocessing config file"
                    )
                check_consistency[category].append(
                    preparation_sample["category"]
                )
        combs = list(itertools.combinations(check_consistency.keys(), 2))
        combs_check = [
            sorted(check_consistency[elem[0]])
            == sorted(check_consistency[elem[1]])
            for elem in combs
        ]
        if not all(combs_check):
            raise RuntimeError(
                "Your specified samples in the sampling/samples "
                "block need to have the same category in each sample category."
            )
        self.class_categories = check_consistency[
            next(iter(self.sample_categories))
        ]

    def CalculatePDF(
        self,
        store_key,
        x_y_original=None,
        x_y_target=None,
        target_hist=None,
        original_hist=None,
        target_bins=None,
        bins=[100, 9],
        limits=None,
    ):
        """
        Calculates the histograms of the input data and uses them to
        calculate the PDF Ratio. Works either on dataframe or pre-made histograms
        CalculatePDFRatio is invoked here.

        Inputs:
        x_y_target: A 2D tuple of the target datapoints of x and y.
        x_y_original: A 2D tuple of the to resample datapoints of x and y.

        Or, x_y_target and/or x_y_original can be replaced by:
        target_hist: hist for the target
        original_hist: hist for the original flavour
        If using target_hist, need to define target_bins, a tuple with (binx, biny).

        store_key: key of the interpolation function to be added
                   to self.inter_func_dict (and self._ratio_dict)
        bins: This can be all possible binning inputs as for numpy
              histogram2d. Not used if hist are passed instead of arrays.
        limits: limits for the binning. Not used if hist are passed instead of arrays.

        Output:
        Provides the PDF interpolation function which is used for sampling (entry in a dict).
        It is a property of the class.
        """

        # Calculate the corresponding histograms
        if target_hist is not None:
            if target_bins is not None:
                h_target, self._x_bin_edges, self._y_bin_edges = (
                    target_hist,
                    target_bins[0],
                    target_bins[1],
                )
            else:
                raise ValueError(
                    "Feeding a histogram but not the bins in PDF calculation."
                )
        elif x_y_target is not None:
            h_target, self._x_bin_edges, self._y_bin_edges = np.histogram2d(
                x_y_target[:, 0], x_y_target[:, 1], bins, range=limits
            )
        else:
            raise ValueError(
                f"Improper target input for PDF calculation of {store_key}."
            )

        if original_hist is not None:
            h_original = original_hist
        elif x_y_original is not None:
            h_original, _, _ = np.histogram2d(
                x_y_original[:, 0],
                x_y_original[:, 1],
                bins=[self._x_bin_edges, self._y_bin_edges],
            )
        else:
            raise ValueError(
                f"Improper original flavour input for PDF calculation of {store_key}."
            )

        # Calculate the PDF Ratio
        self.CalculatePDFRatio(
            store_key,
            h_target,
            h_original,
            self._x_bin_edges,
            self._y_bin_edges,
        )

    def CalculatePDFRatio(
        self, store_key, h_target, h_original, x_bin_edges, y_bin_edges
    ):
        """
        Receives the histograms of the target and original data, the bins
        and a max ratio value. Latter is optional.

        Inputs:
        h_target: Output of numpy histogram2D for the target datapoints
        h_original: Output of numpy histogram2D for the original datapoints
        store_key: key of the interpolation function to be added
                   to self.inter_func_dict (and self._ratio_dict)
        bins: The bin edges of the binning used for the numpy histogram2D.
              This is also returned from numpy histgram2D

        Output:
        Provides the PDF interpolation function which is used for sampling.
        This can be returned with Inter_Func_Dict. It is a property of the class.
        """

        # Transform bin edges to bin centres
        self.x_bins = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        self.y_bins = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

        # Calculating the ratio of the reference distribution w.r.t. the target distribution
        ratio = np.divide(
            h_target,
            h_original,
            out=np.zeros(
                h_original.shape,
                dtype=float,
            ),
            where=(h_original != 0),
        )

        self._ratio_dict[store_key] = ratio
        self._bin_edges_dict[store_key] = (
            x_bin_edges[1:-1],
            y_bin_edges[1:-1],
        )

        self.inter_func_dict[store_key] = RectBivariateSpline(
            self.x_bins,
            self.y_bins,
            self._ratio_dict[store_key],
            kx=3,
            ky=3,
            bbox=[
                x_bin_edges[0],
                x_bin_edges[-1],
                y_bin_edges[0],
                y_bin_edges[-1],
            ],
        )

    def save(self, file_name: str, overwrite: bool = False):
        """
        Save the interpolation function to file

        Input:
        file_name: Path where the pickle file is saved.

        Output:
        Pickle file of the PDF interpolation function.
        """

        if self.inter_func is not None:
            if os.path.isfile(file_name) is True:
                if overwrite is True:
                    logger.warning(
                        "File already exists at given path! Overwrite existing file!"
                    )

                    # Dump function into pickle file
                    with open(file_name, "wb") as file:
                        pickle.dump(self.inter_func, file)

                else:
                    logger.warning(
                        "File already exists at given path! PDF interpolation function not saved!"
                    )

            else:
                # Dump function into pickle file
                with open(file_name, "wb") as file:
                    pickle.dump(self.inter_func, file)

        else:
            raise ValueError("Interpolation function not calculated/given!")

    def load(self, file_name: str):
        """
        Load the interpolation function from file.

        Input:
        file_name: Path where the pickle file is saved.

        Output:
        PDF interpolation function of the pickle file is added as property
        to the class.
        """

        with open(file_name, "rb") as file:
            self.inter_func = pickle.load(file)

    def inMemoryResample(
        self, x_values, y_values, size, store_key, replacement=True
    ):
        """
        Resample all of the datapoints at once. Requirement for that
        is that all datapoints fit in the RAM.

        Input:
        x_values: x values of the datapoints which are to be resampled from (i.e pT)
        y_values: y values of the datapoints which are to be resampled from (i.e eta)
        size: Number of jets which are resampled.
        store_key: key of the interpolation function to be added
                   to self.inter_func_dict (and self._ratio_dict)

        Output:
        Resampled jets
        """
        if replacement is False:
            logger.info("PDF sampling without replacement for given set!")
        if type(x_values) == float or type(x_values) == int:
            x_values = np.asarray([x_values])

        if type(y_values) == float or type(y_values) == int:
            y_values = np.asarray([y_values])

        # Check for sizes of x_values and y_values
        if len(y_values) != len(x_values):
            raise ValueError("x_values and y_values need to have same size!")

        # Evaluate the datapoints with the PDF function
        r_resamp = self.Return_unnormalised_PDF_weights(
            x_values, y_values, store_key
        )

        # Normalise the datapoints for sampling
        r_resamp = r_resamp / np.sum(r_resamp)
        """
        # When debugging, this snippet plots the
        # weights and interpolated values vs pT.

        import matplotlib as mtp
        import matplotlib.pyplot as plt

        n, _ = np.histogram(x_values, bins=self._x_bin_edges)
        sy, _ = np.histogram(x_values, bins=self._x_bin_edges, weights=r_resamp)
        mean = np.divide(
            sy,
            n,
            out=np.zeros(
                sy.shape,
                dtype=float,
            ),
            where=(n != 0),
        )
        fig = plt.figure()
        plt.plot(self.x_bins, mean, label=f"effective weights {store_key}")
        plt.legend(loc="best", ncol=1, fontsize=7)
        plt.xlabel('pT (MeV)')
        plt.ylabel('Weights')
        plt.tight_layout()
        plt.savefig(f"{store_key}_weights.pdf")

        fig = plt.figure()
        plt.plot(self.x_bins, n, label=f"Distribution {store_key}")
        plt.legend(loc="best", ncol=1, fontsize=7)
        plt.xlabel('pT (MeV)')
        plt.ylabel('Weights')
        plt.tight_layout()
        plt.savefig(f"{store_key}_distribution.pdf")
        """

        # Resample the datapoints based on their PDF Ratio value
        sampled_indices = self.Resample_chunk(r_resamp, size, replacement)

        return sampled_indices

    def Return_unnormalised_PDF_weights(self, x_values, y_values, store_key):
        r_resamp = self.inter_func_dict[store_key].ev(x_values, y_values)

        # Neutralise all datapoints where the ratio is less than 0
        indices = np.where(r_resamp < 0)[0]
        r_resamp[indices] = 0
        return r_resamp

    def Resample_chunk(self, r_resamp, size, replacement=True):
        sampled_indices = np.random.default_rng().choice(
            len(r_resamp), p=r_resamp, size=size, replace=replacement
        )
        return sampled_indices

    def Resample_Iterator(
        self,
        sample_category,
        sample_id,
        save_name,
        sample_name,
        chunk_size=1e6,
    ):
        """
        Resample with the data not completely stored in memory.
        Will load the jets in chunks, computing first the sum of PDF
        weights and then sampling with replacement based on the normalised
        weights.
        """
        _, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        store_key = sample_category + "_" + preparation_sample.get("category")

        # First pass over data to get sum of weights
        load_chunk = True
        generator = self.Load_Samples_Generator(
            sample_category=sample_category,
            sample_id=sample_id,
            chunk_size=chunk_size,
        )
        sum_of_weights = 0
        while load_chunk:
            try:
                _, target_dist, load_more = next(generator)
            except StopIteration:
                break
            load_chunk = load_more
            weights = self.Return_unnormalised_PDF_weights(
                target_dist["sample_vector"][:, 0],
                target_dist["sample_vector"][:, 1],
                store_key=store_key,
            )
            sum_of_weights += np.sum(weights)

        # second pass over data, normalising weights and sampling
        load_chunk = True
        generator = self.Load_Samples_Generator(
            sample_category=sample_category,
            sample_id=sample_id,
            chunk_size=chunk_size,
        )
        create_file = True
        sampled_jets = 0
        pbar = tqdm(total=self.number_to_sample[sample_name])
        while load_chunk:
            try:
                _, target_dist, load_more = next(generator)
            except StopIteration:
                break
            load_chunk = load_more
            weights = self.Return_unnormalised_PDF_weights(
                target_dist["sample_vector"][:, 0],
                target_dist["sample_vector"][:, 1],
                store_key=store_key,
            )
            chunk_weights = np.sum(weights)
            weights = weights / sum_of_weights
            to_sample = self.number_to_sample[sample_name] / chunk_weights
            if not load_chunk:
                # last chunk
                to_sample = self.number_to_sample[sample_name] - sampled_jets

            selected_ind = self.Resample_chunk(weights, size=round(to_sample))
            # Important that the indices from the generator are not random!
            # Otherwise this is very inefficient
            selected_indices = np.sort(selected_ind).astype(int)
            sampled_jets += len(selected_indices)
            pbar.update(selected_indices.size)
            if create_file:
                create_file = False
                with h5py.File(save_name, "w") as f:
                    f.create_dataset(
                        "jets",
                        data=selected_indices,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
            else:
                with h5py.File(save_name, "a") as f:
                    f["jets"].resize(
                        (f["jets"].shape[0] + selected_indices.shape[0]),
                        axis=0,
                    )
                    f["jets"][-selected_indices.shape[0] :] = selected_indices
        pbar.close()

    def Save_partial_iterator(
        self,
        sample_category,
        category_id,
        sample_id,
        selected_indices,
        chunk_size: int = 1e6,
    ):
        """
        Save the selected data to an output file with an iterative approach
        (generator) for writing only, writing in chunk of size chunk_size.
        The file is read in one go.
        """
        sample, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        in_file = GetPreparationSamplePath(preparation_sample)

        sample_lengths = len(selected_indices)
        max_sample = np.amax(sample_lengths)
        n_chunks = round(max_sample / chunk_size + 0.5)
        chunk_sizes = sample_lengths / n_chunks

        generators = SamplingGenerator(
            in_file,
            selected_indices,
            chunk_size=chunk_sizes,
            label=self.class_labels_map[preparation_sample["category"]],
            label_classes=list(range(len(self.class_labels_map))),
            use_tracks=self.save_tracks,
            seed=23,
            duplicate=True,
        )

        create_file = True
        chunk_counter = 0
        save_name = os.path.join(
            self.outfile_path,
            "PDF_sampling",
            self.options["samples"][sample_category][sample_id]
            + "_selected.h5",
        )

        logger.info(f"Writing to file {save_name}.")

        pbar = tqdm(total=np.sum(sample_lengths))
        while chunk_counter < n_chunks + 1:
            try:
                if self.save_tracks:
                    jets, tracks, labels = next(generators)
                else:
                    jets, labels = next(generators)
            except StopIteration:
                break
            pbar.update(jets.size)
            rng = np.random.default_rng(seed=42)
            rng.shuffle(jets)
            rng.shuffle(labels)
            if self.save_tracks:
                rng.shuffle(tracks)

            if create_file:
                create_file = False
                # write to file by creating dataset
                with h5py.File(save_name, "w") as out_file:
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
                        out_file.create_dataset(
                            "tracks",
                            data=tracks,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, tracks.shape[1]),
                        )
            else:
                # appending to existing dataset
                with h5py.File(save_name, "a") as out_file:
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
                        out_file["tracks"].resize(
                            (out_file["tracks"].shape[0] + tracks.shape[0]),
                            axis=0,
                        )
                        out_file["tracks"][-tracks.shape[0] :] = tracks
            chunk_counter += 1
        pbar.close()

    def Save_complete_iterator(
        self, sample_category, category_id, sample_id, chunk_size: int = 1e4
    ):
        """
        Save the selected data to an output file with an iterative approach
        (generator) for both writing and reading, in chunk of size chunk_size.
        """
        sample_name = self.options["samples"][sample_category][sample_id]
        sample, preparation_sample = self.sample_file_map[sample_category][
            sample_id
        ]
        in_file = GetPreparationSamplePath(preparation_sample)
        index_file = os.path.join(
            self.outfile_path,
            "PDF_sampling",
            self.options["samples"][sample_category][sample_id]
            + "_indices.h5",
        )
        index_generator = self.Load_Index_Generator(index_file, chunk_size)
        label = self.class_labels_map[preparation_sample["category"]]
        label_classes = list(range(len(self.class_labels_map)))
        use_tracks = self.save_tracks
        duplicate = True

        save_name = os.path.join(
            self.outfile_path,
            "PDF_sampling",
            self.options["samples"][sample_category][sample_id]
            + "_selected.h5",
        )
        logger.info(f"Writing to file {save_name}.")
        load_chunk = True
        create_file = True
        pbar = tqdm(total=self.number_to_sample[sample_name])
        while load_chunk:
            try:
                indices, load_more = next(index_generator)
            except StopIteration:
                break
            load_chunk = load_more

            labels = label_binarize(
                (np.ones(len(indices)) * label),
                classes=label_classes,
            )
            with h5py.File(in_file, "r") as file_df:
                if use_tracks:
                    jets, tracks = read_dataframe_repetition(
                        file_df,
                        loading_indices=indices,
                        duplicate=duplicate,
                        use_tracks=use_tracks,
                    )
                else:
                    jets = read_dataframe_repetition(
                        file_df,
                        loading_indices=indices,
                        duplicate=duplicate,
                        use_tracks=use_tracks,
                    )
            pbar.update(jets.size)
            rng = np.random.default_rng(seed=42)
            rng.shuffle(jets)
            rng.shuffle(labels)
            if self.save_tracks:
                rng.shuffle(tracks)

            if create_file:
                create_file = False
                # write to file by creating dataset
                with h5py.File(save_name, "w") as out_file:
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
                        out_file.create_dataset(
                            "tracks",
                            data=tracks,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, tracks.shape[1]),
                        )
            else:
                # appending to existing dataset
                with h5py.File(save_name, "a") as out_file:
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
                        out_file["tracks"].resize(
                            (out_file["tracks"].shape[0] + tracks.shape[0]),
                            axis=0,
                        )
                        out_file["tracks"][-tracks.shape[0] :] = tracks
        pbar.close()

    def Generate_Target_PDF(self, iterator=True):
        """
        This method creates the target distribution (seperated) and store the associated
        histogram in memory (use for sampling) as well as the target numbers.

        Parameters
        ----------
        iterator: bool, whether to use the iterator approach or
                        load the whole sample in memory.

        Returns
        -------
        Save to memory the target histogram, binning info, and target numbers.
        """

        logger.info("Generating target PDF.")
        self.target_fractions = []
        available_numbers = []
        self.target_histo = {}

        for cat_ind, sample_category in enumerate(self.options["samples"]):
            reading_dict = self.File_to_histogram(
                sample_category=sample_category,
                category_ind=cat_ind,
                sample_id=0,
                iterator=iterator,
                chunk_size=1e6,
            )
            self.target_histo[sample_category] = {
                "hist": reading_dict["hist"],
                "xbins": reading_dict["xbins"],
                "ybins": reading_dict["ybins"],
            }
            available_numbers.append(reading_dict["available_numbers"])
            self.target_fractions.append(
                self.options["fractions"][sample_category]
            )

        # Correct target numbers
        njets_asked = self.options["njets"]
        target_numbers_corr = CorrectFractions(
            N_jets=available_numbers,
            target_fractions=self.target_fractions,
            verbose=False,
        )
        if njets_asked == -1:
            logger.info("Maximising number of jets to target distribution.")
        else:
            logger.info(f"Requesting {njets_asked} in total from target.")
            total_corr = sum(target_numbers_corr)
            if total_corr < njets_asked:
                logger.info("Requesting more jets from target than available.")
                logger.info("PDF sampling will thus upsample target too.")
                ratio = float(njets_asked / total_corr)
                target_numbers_corr = [
                    int(num * ratio) for num in target_numbers_corr
                ]
            elif total_corr > njets_asked:
                ratio = float(njets_asked / total_corr)
                target_numbers_corr = [
                    int(num * ratio) for num in target_numbers_corr
                ]
        self.target_number = {
            list(self.options["samples"].keys())[ind]: target
            for ind, target in enumerate(target_numbers_corr)
        }
        for cat_ind, sample_category in enumerate(self.options["samples"]):
            logger.info(
                f"target - category {sample_category}: selected {self.target_number[sample_category]}/{available_numbers[cat_ind]} jets, giving the requested fraction of {self.target_fractions[cat_ind]}"
            )

        # For each flavour type, set the target numbers, respecting flavour ratio and upsampling max ratio (if given)
        for sample_id, sample in enumerate(
            self.options["samples"][list(self.sample_categories.keys())[0]]
        ):
            flavour_names = []
            for cat_ind, sample_category in enumerate(self.options["samples"]):
                flavour_name = self.sample_file_map[sample_category][
                    sample_id
                ][0]
                flavour_names.append(flavour_name)
            if any(
                flavour_name in self.max_upsampling
                for flavour_name in flavour_names
            ):
                asked_num = []
                for cat_ind, flavour_name in enumerate(flavour_names):
                    num_asked = self.target_number[
                        list(self.options["samples"])[cat_ind]
                    ]
                    if flavour_name in self.max_upsampling:
                        upsampling, num_av = self.max_upsampling[flavour_name]
                        upsampling_asked = float(num_asked) / num_av
                        if upsampling < upsampling_asked:
                            num_corrected = int(num_av * upsampling)
                            asked_num.append(num_corrected)
                            logger.warning(
                                f"Upsampling ratio demanded to {flavour_name} is {upsampling_asked}, over limit of {upsampling}."
                            )
                            logger.warning(
                                f"Number of {flavour_name} demanded will therefore be limited."
                            )
                        else:
                            logger.info(
                                f"Upsampling ratio demanded to {flavour_name} is {upsampling_asked}, below limit of {upsampling}."
                            )
                            asked_num.append(num_asked)
                    else:
                        logger.info(
                            f"No maximum upsampling ratio demanded to {flavour_name}."
                        )
                        asked_num.append(num_asked)
                asked_num_corr = CorrectFractions(
                    N_jets=asked_num,
                    target_fractions=self.target_fractions,
                    verbose=False,
                )
                for flavour_name, num in zip(flavour_names, asked_num_corr):
                    self.number_to_sample[flavour_name] = int(num)
                    logger.info(
                        f"For {flavour_name}, demanding {int(num)} jets."
                    )
            else:
                for sample_category, flavour_name in zip(
                    self.options["samples"], flavour_names
                ):
                    self.number_to_sample[flavour_name] = self.target_number[
                        sample_category
                    ]
                    logger.info(
                        f"For {flavour_name}, demanding {self.target_number[sample_category]} jets."
                    )

    def Generate_Flavour_PDF(
        self,
        sample_category: str,
        category_id: int,
        sample_id: int,
        iterator: bool = True,
    ):
        """
        This method:
            - create the flavour distribution (also seperated),
            - produce the PDF between the flavour and the target.

        Parameters
        ----------
        sample_category: str, the name of the category study.
        category_id: int, the location of the category in the list.
        sample_id: int, the location of the sample flavour in the category dict.
        iterator: bool, whether to use the iterator approach or
                        load the whole sample in memory.

        Returns
        -------
        Add a dictionary object to the class pointing to the interpolation functions (also saves them).
        Returns None or dataframe of flavour (if iterator or not) and the histogram of the flavour.

        """

        reading_dict = self.File_to_histogram(
            sample_category=sample_category,
            category_ind=category_id,
            sample_id=sample_id,
            iterator=iterator,
            chunk_size=1e6,
        )
        logger.info(
            f"Computing PDF in {sample_category} for the {reading_dict['category']}."
        )
        self.CalculatePDF(
            store_key=f"{sample_category}_{reading_dict['category']}",
            target_hist=self.target_histo[sample_category]["hist"],
            original_hist=reading_dict["hist"],
            target_bins=(
                self.target_histo[sample_category]["xbins"],
                self.target_histo[sample_category]["ybins"],
            ),
            bins=self.limit["bins"][category_id],
            limits=self.limit["ranges"][category_id],
        )

        if iterator:
            return None, reading_dict["hist"]
        else:
            return reading_dict["target_dist"], reading_dict["hist"]

    def Sample_Flavour(
        self,
        sample_category: str,
        category_id: int,
        sample_id: int,
        iterator: bool = True,
        flavour_distribution=None,
    ):
        """
        This method:
            - samples the required amount based on PDF and fractions
            - storing the indices selected to memory.
        Parameters
        ----------
        sample_category: str, the name of the category study.
        category_id: int, the location of the category in the list.
        sample_id: int, the location of the sample flavour in the category dict.
        iterator: bool, whether to use the iterator approach or load the whole sample
                        in memory.
        flavour_distribution: None or numpy array, the loaded data (for the flavour).
                              If it is None, an iterator method is used.

        Returns
        -------
        Returns (and stores to memory, if iterator is false) the selected indices
        for the flavour studied.
        """

        sample_name = self.options["samples"][sample_category][sample_id]
        save_name = os.path.join(
            self.outfile_path,
            "PDF_sampling",
            self.options["samples"][sample_category][sample_id]
            + "_indices.h5",
        )
        if not os.path.exists(os.path.join(self.outfile_path, "PDF_sampling")):
            os.mkdir(os.path.join(self.outfile_path, "PDF_sampling"))
        logger.info(f"Selecting indices for {sample_name}.")
        logger.info(f"Saving indices to: {save_name}")
        if flavour_distribution is None or iterator:
            logger.info("Using iterating approach.")
            self.Resample_Iterator(
                sample_category=sample_category,
                sample_id=sample_id,
                save_name=save_name,
                sample_name=sample_name,
            )
            return None

        else:
            logger.info("Using in-memory approach.")
            print(
                "Store Key",
                sample_category + "_" + flavour_distribution["category"],
            )
            selected_ind = self.inMemoryResample(
                flavour_distribution["sample_vector"][:, 0],
                flavour_distribution["sample_vector"][:, 1],
                size=self.number_to_sample[sample_name],
                store_key=sample_category
                + "_"
                + flavour_distribution["category"],
            )
            with h5py.File(save_name, "w") as f:
                selected_indices = np.sort(selected_ind).astype(int)
                f.create_dataset(
                    "jets",
                    data=selected_indices,
                    compression="gzip",
                )

            return selected_indices

    def Save_Flavour(
        self,
        sample_category: str,
        category_id: int,
        sample_id: int,
        selected_indices: dict = None,
        chunk_size: int = 1e3,
        iterator: bool = True,
    ):
        """
        This method stores the selected date to memory (based on given indices).

        Parameters
        ----------
        sample_category: str, the name of the category study.
        category_id: int, the location of the category in the list.
        sample_id: int, the location of the sample flavour in the category dict.
        iterator: bool, whether to use the iterator approach or load the whole sample
                        in memory.
        chunk_size: int, the size of the chunks (the last chunk may be at most 2 * chunk_size).
        selected_indices: None or array of int: the indices to resample. If None, iterator approach used.

        Returns
        -------
        Stores to memory the selected data
        """

        logger.info(
            f"Sampling {self.options['samples'][sample_category][sample_id]} for {sample_category}."
        )

        if iterator or selected_indices is None:
            logger.info("Using complete iterating approach for saving.")
            self.Save_complete_iterator(
                sample_category,
                category_id,
                sample_id,
                chunk_size,
            )

        else:
            logger.info(
                "Using partial iterating approach for saving (indices in memory)."
            )
            self.Save_partial_iterator(
                sample_category,
                category_id,
                sample_id,
                selected_indices,
                chunk_size,
            )

    def Combine_Flavours(self, chunk_size: int = 1e6):
        """
        This method loads the stored flavour resampled and combines them
        iteratively into a single file.

        Parameters
        ----------

        Returns
        -------
        Stores to memory the combined selected data
        into a single file.
        """
        # self.iterations
        output_name = self.config.GetFileName(option="resampled")
        logger.info("Combining all the flavours into a single file.")
        logger.info(f"Storing to {self.outfile_path}")
        logger.info(f"Storing to {output_name}")

        create_file = True
        for sample_id, sample in enumerate(
            self.options["samples"][list(self.sample_categories.keys())[0]]
        ):
            for cat_ind, sample_category in enumerate(self.options["samples"]):
                load_name = os.path.join(
                    self.outfile_path,
                    "PDF_sampling",
                    self.options["samples"][sample_category][sample_id]
                    + "_selected.h5",
                )
                with h5py.File(load_name, "r") as f:
                    logger.info(
                        f"Working on {sample_category} {self.sample_file_map[sample_category][sample_id][1].get('category')}."
                    )
                    total_size_file = len(f["jets"])
                    load_per_iteration = chunk_size
                    number_of_chunks = round(
                        total_size_file / load_per_iteration + 0.5
                    )
                    start_ind = 0
                    chunk_number = 0
                    pbar = tqdm(total=np.sum(total_size_file))
                    while chunk_number < number_of_chunks:
                        end_ind = int(start_ind + load_per_iteration)
                        if chunk_number == number_of_chunks - 1:
                            end_ind = int(total_size_file)
                        jets = f["jets"][start_ind:end_ind]
                        labels = f["labels"][start_ind:end_ind]
                        if self.save_tracks:
                            tracks = f["tracks"][start_ind:end_ind]
                        start_ind = end_ind

                        pbar.update(jets.size)
                        if create_file:
                            create_file = False
                            with h5py.File(output_name, "w") as out_file:
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
                                    out_file.create_dataset(
                                        "tracks",
                                        data=tracks,
                                        compression="gzip",
                                        chunks=True,
                                        maxshape=(None, tracks.shape[1]),
                                    )
                        else:
                            with h5py.File(output_name, "a") as out_file:
                                out_file["jets"].resize(
                                    (
                                        out_file["jets"].shape[0]
                                        + jets.shape[0]
                                    ),
                                    axis=0,
                                )
                                out_file["jets"][-jets.shape[0] :] = jets
                                out_file["labels"].resize(
                                    (
                                        out_file["labels"].shape[0]
                                        + labels.shape[0]
                                    ),
                                    axis=0,
                                )
                                out_file["labels"][-labels.shape[0] :] = labels
                                if self.save_tracks:
                                    out_file["tracks"].resize(
                                        (
                                            out_file["tracks"].shape[0]
                                            + tracks.shape[0]
                                        ),
                                        axis=0,
                                    )
                                    out_file["tracks"][
                                        -tracks.shape[0] :
                                    ] = tracks
                        chunk_number += 1
                    pbar.close()

    def Make_plots(self, binning=[200, 20]):
        """
        Produce plots of the variables used in resampling (before and after preprocessing).
        """
        ranges = self.limit["extreme_ranges"]

        logger.info("Making plots of variables before sampling.")
        histo_before_dict = {}
        for sample_id, sample in enumerate(
            self.options["samples"][list(self.sample_categories.keys())[0]]
        ):
            for cat_ind, sample_category in enumerate(self.options["samples"]):
                _, flavour_distribution = self.Load_Samples(
                    sample_category, sample_id
                )
                flavour_name = self.sample_file_map[sample_category][
                    sample_id
                ][1].get("category")
                if sample_id == 0 and cat_ind == 0:
                    thehist, binx, biny = np.histogram2d(
                        flavour_distribution["sample_vector"][:, 0],
                        flavour_distribution["sample_vector"][:, 1],
                        bins=binning,
                        range=ranges,
                    )
                else:
                    thehist, _, _ = np.histogram2d(
                        flavour_distribution["sample_vector"][:, 0],
                        flavour_distribution["sample_vector"][:, 1],
                        bins=(binx, biny),
                    )
                if cat_ind == 0:
                    histo_before_dict[flavour_name] = thehist
                else:
                    histo_before_dict[flavour_name] = np.add(
                        histo_before_dict[flavour_name], thehist
                    )
        plot_name_clean = self.config.GetFileName(
            extension="",
            option="pt_eta-before_sampling",
            custom_path="plots/",
        )
        ResamplingPlots(
            concat_samples=histo_before_dict,
            positions_x_y=[0, 1],
            variable_names=[self.var_x, self.var_y],
            plot_base_name=plot_name_clean,
            binning={self.var_x: binx, self.var_y: biny},
            Log=True,
            hist_input=True,
        )

        logger.info("Making plots of variables after sampling.")
        histo_after_dict = {}
        for sample_id, sample in enumerate(
            self.options["samples"][list(self.sample_categories.keys())[0]]
        ):
            for cat_ind, sample_category in enumerate(self.options["samples"]):
                load_name = os.path.join(
                    self.outfile_path,
                    "PDF_sampling",
                    self.options["samples"][sample_category][sample_id]
                    + "_selected.h5",
                )
                with h5py.File(load_name, "r") as f:
                    flavour_name = self.sample_file_map[sample_category][
                        sample_id
                    ][1].get("category")
                    logger.info(
                        f"Loading {len(f['jets'])} jets for {flavour_name} from {load_name}."
                    )
                    jets_x = np.asarray(f["jets"][self.var_x])
                    jets_y = np.asarray(f["jets"][self.var_y])
                    thehist, binx, biny = np.histogram2d(
                        jets_x,
                        jets_y,
                        bins=(binx, biny),
                    )
                if cat_ind == 0:
                    histo_after_dict[flavour_name] = thehist
                else:
                    histo_after_dict[flavour_name] = np.add(
                        histo_after_dict[flavour_name], thehist
                    )
        plot_name_clean = self.config.GetFileName(
            extension="",
            option="pt_eta-after_sampling",
            custom_path="plots/",
        )
        ResamplingPlots(
            concat_samples=histo_after_dict,
            positions_x_y=[0, 1],
            variable_names=[self.var_x, self.var_y],
            plot_base_name=plot_name_clean,
            binning={self.var_x: binx, self.var_y: biny},
            Log=True,
            hist_input=True,
        )

    def Run(self):
        self.Initialise_Flavour_Samples()
        logger.info("Starting PDFsampling.")
        # Whether to use iterator approach or in-memory (one file at a time).
        iterator = True

        # Retrieve the PDF between target and all distribution
        self.Generate_Target_PDF(iterator=iterator)

        for sample_id, sample in enumerate(
            self.options["samples"][list(self.sample_categories.keys())[0]]
        ):
            for cat_ind, sample_category in enumerate(self.options["samples"]):
                # First step: generate the PDF of the Flavour to Target.
                flavour_dist, _ = self.Generate_Flavour_PDF(
                    sample_category=sample_category,
                    category_id=cat_ind,
                    sample_id=sample_id,
                    iterator=iterator,
                )

                # Second step: use the flavour pdf to select indices (target included)
                selected_indices = self.Sample_Flavour(
                    sample_category=sample_category,
                    category_id=cat_ind,
                    sample_id=sample_id,
                    flavour_distribution=flavour_dist,
                    iterator=iterator,
                )

                # Third step: use the selected indices to save the thus sampled file
                self.Save_Flavour(
                    sample_category=sample_category,
                    category_id=cat_ind,
                    sample_id=sample_id,
                    selected_indices=selected_indices,
                    iterator=iterator,
                    chunk_size=1e3,
                )

        self.Make_plots()

        # Now that everything is saved, load each file and concatenate them into a single large file
        self.Combine_Flavours()

        logger.info("PDFsampling finished.")


class UnderSampling(Resampling):
    def InitialiseSamples(self):
        """
        Initialising input files.
        Parameters
        ----------

        Returns
        -------
        At this point the arrays of the 2 variables are loaded which are used
        for the sampling and saved into class variables.
        """
        self.samples = {}
        try:
            samples = self.options["samples"]
        except KeyError:
            raise KeyError(
                "You chose the 'count' or 'probability_ratio' option "
                "for the sampling but didn't provide the samples to use. "
                "Please specify them in the configuration file!"
            )

        # list of sample classes, bjets, cjets, etc
        valid_class_categories = self.GetValidClassCategories(samples)
        self.class_categories = valid_class_categories[
            next(iter(samples.keys()))
        ]
        # map of sample categories and indexes as IDs
        self.sample_categories = {
            elem: i
            for i, elem in enumerate(list(valid_class_categories.keys()))
        }
        # map of sample categories
        self.sample_map = {
            elem: {} for elem in list(valid_class_categories.keys())
        }
        self.sample_file_map = {}
        for sample_category in self.sample_categories:
            sample_id = self.sample_categories[sample_category]
            self.samples[sample_category] = []
            for sample in samples[sample_category]:
                preparation_sample = self.preparation_samples.get(sample)
                preparation_sample_path = GetPreparationSamplePath(
                    preparation_sample
                )
                self.sample_file_map[sample] = preparation_sample_path
                logger.info(
                    f"Loading sampling variables from {preparation_sample_path}"
                )
                with h5py.File(preparation_sample_path, "r") as f:
                    nJets_initial = None
                    if (
                        "custom_njets_initial" in self.options
                        and self.options["custom_njets_initial"] is not None
                        and sample
                        in list(self.options["custom_njets_initial"])
                    ):
                        nJets_initial = int(
                            self.options["custom_njets_initial"][sample]
                        )
                    jets_x = np.asarray(f["jets"][self.var_x])[:nJets_initial]
                    jets_y = np.asarray(f["jets"][self.var_y])[:nJets_initial]
                logger.info(
                    f"Loaded {len(jets_x)} {preparation_sample.get('category')} jets from {sample}."
                )
                # construct a flat array with 5 columns:
                # x, y, index, sample_id, sample_class
                sample_vector = np.asarray(
                    [
                        jets_x,
                        jets_y,
                        range(len(jets_x)),
                        np.ones(len(jets_x)) * sample_id,
                        np.ones(len(jets_x))
                        * self.sample_categories[sample_category],
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

    def GetValidClassCategories(self, samples):
        """helper function to check sample categories requested in resampling were
        also defined in the sample preparation step. Returns sample classes."""
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
                check_consistency[category].append(
                    preparation_sample["category"]
                )
        combs = list(itertools.combinations(check_consistency.keys(), 2))
        combs_check = [
            sorted(check_consistency[elem[0]])
            == sorted(check_consistency[elem[1]])
            for elem in combs
        ]
        if not all(combs_check):
            raise RuntimeError(
                "Your specified samples in the sampling/samples "
                "block need to have the same samples in each sample category."
            )
        return check_consistency

    def GetIndices(self):
        """
        Applies the UnderSampling to the given arrays.

        Parameters
        ----------

        Returns
        -------
        Returns the indices for the jets to be used separately for each
        category and sample.

        """
        # concatenate samples with the same category
        # no fraction between different samples is ensured at this stage,
        # this will be done at the end
        concat_samples = {
            elem: {"jets": None} for elem in self.class_categories
        }
        for sample_category in self.samples:
            for sample in self.samples[sample_category]:
                if concat_samples[sample["category"]]["jets"] is None:
                    concat_samples[sample["category"]]["jets"] = sample[
                        "sample_vector"
                    ]
                else:
                    concat_samples[sample["category"]][
                        "jets"
                    ] = np.concatenate(
                        [
                            concat_samples[sample["category"]]["jets"],
                            sample["sample_vector"],
                        ]
                    )

        # calculate the 2D bin statistics for each sample
        for class_category in self.class_categories:
            binnumbers, ind, stat = self.GetBins(
                concat_samples[class_category]["jets"][:, 0],
                concat_samples[class_category]["jets"][:, 1],
            )
            concat_samples[class_category]["binnumbers"] = binnumbers
            concat_samples[class_category]["stat"] = stat

            self.bins_indices_flat = ind

        self.concat_samples = concat_samples

        min_count_per_bin = np.amin(
            [concat_samples[sample]["stat"] for sample in concat_samples],
            axis=0,
        )

        rng = np.random.default_rng(seed=self.rnd_seed)
        indices_to_keep = {elem: [] for elem in self.class_categories}
        # retrieve the indices of the jets to keep
        for bin_i, count in zip(self.bins_indices_flat, min_count_per_bin):
            for class_category in self.class_categories:
                indices_to_keep[class_category].append(
                    rng.choice(
                        np.nonzero(
                            concat_samples[class_category]["binnumbers"]
                            == bin_i
                        )[0],
                        int(count),
                        replace=False,
                    )
                )
        # join the indices into one array per class category
        for class_category in self.class_categories:
            indices_to_keep[class_category] = np.concatenate(
                indices_to_keep[class_category]
            )
        # make sure the sample fractions are correct
        # since undersampling is applied at this stage, it is enough to
        # calculate the fractions for one flavour class
        # there might be some fluctuations but that should be negligible
        # TODO: could be done for each of them
        reference_class_category = self.class_categories[0]
        target_fractions = []
        N_jets = []
        sample_categories = concat_samples[reference_class_category]["jets"][
            indices_to_keep[reference_class_category], 4
        ]
        for sample_category in self.sample_categories:
            target_fractions.append(self.options["fractions"][sample_category])
            N_jets.append(
                len(
                    np.nonzero(
                        sample_categories
                        == self.sample_categories[sample_category]
                    )[0]
                )
            )
        target_N_jets = CorrectFractions(
            N_jets=N_jets,
            target_fractions=target_fractions,
            class_names=list(self.sample_categories.keys()),
        )
        target_N_jets = {
            list(self.sample_categories.keys())[i]: target_N_jets[i]
            for i in range(len(target_N_jets))
        }
        for class_category in self.class_categories:
            sample_categories = concat_samples[class_category]["jets"][
                indices_to_keep[class_category], 4
            ]
            indices_to_keep_tmp = np.asarray([])
            for sample_category in self.sample_categories:
                location = np.nonzero(
                    sample_categories
                    == self.sample_categories[sample_category]
                )[0]
                indices_to_keep_tmp = np.append(
                    indices_to_keep_tmp,
                    rng.choice(
                        indices_to_keep[class_category][location],
                        size=target_N_jets[sample_category]
                        if target_N_jets[sample_category]
                        <= len(indices_to_keep[class_category][location])
                        else len(indices_to_keep[class_category][location]),
                        replace=False,
                    ),
                )
            indices_to_keep[class_category] = indices_to_keep_tmp.astype(int)

        # check if more jets are available as requested in the config file
        # we assume that all classes have the same number of jets now
        if len(indices_to_keep[reference_class_category]) > self.options[
            "njets"
        ] // len(self.class_categories):
            size_per_class = self.options["njets"] // len(
                self.class_categories
            )
            for class_category in self.class_categories:
                indices_to_keep[class_category] = rng.choice(
                    indices_to_keep[class_category],
                    size=int(size_per_class)
                    if size_per_class <= len(indices_to_keep[class_category])
                    else len(indices_to_keep[class_category]),
                    replace=False,
                )
        else:
            size_per_class = len(indices_to_keep[reference_class_category])
            size_total = len(indices_to_keep[reference_class_category]) * len(
                self.class_categories
            )
            logger.warning(
                f"You asked for {self.options['njets']:.0f} jets, however, "
                f"only {size_total} are available."
            )

        # get indices per single sample
        self.indices_to_keep = {}
        self.x_y_after_sampling = {}
        size_total = 0
        with h5py.File(self.options["intermediate_index_file"], "w") as f:
            for class_category in self.class_categories:
                self.x_y_after_sampling[class_category] = concat_samples[
                    class_category
                ]["jets"][indices_to_keep[class_category], :2]
                sample_categories = concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 4
                ]
                sample_indices = concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 2
                ]
                for sample_category in self.sample_categories:
                    sample_name = self.sample_map[sample_category][
                        class_category
                    ]
                    self.indices_to_keep[sample_name] = np.sort(
                        sample_indices[
                            sample_categories
                            == self.sample_categories[sample_category]
                        ]
                    ).astype(int)
                    f.create_dataset(
                        sample_name,
                        data=self.indices_to_keep[sample_name],
                        compression="gzip",
                    )
                    sample_size = len(self.indices_to_keep[sample_name])
                    size_total += sample_size
                    logger.info(
                        f"Using {sample_size} jets from {sample_name}."
                    )
        logger.info(f"Using in total {size_total} jets.")
        return self.indices_to_keep

    def ResamplingGenerator(
        self,
        file: str,
        indices: list,
        label: int,
        label_classes: list,
        use_tracks: bool = False,
        chunk_size: int = 10000,
        seed=23,
    ):
        with h5py.File(file, "r") as f:
            start_ind = 0
            end_ind = int(start_ind + chunk_size)
            # create indices and then shuffle those to be used in the loop below
            tupled_indices = []
            while end_ind <= len(indices) or start_ind == 0:
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)
            rng = np.random.default_rng(seed=seed)
            tupled_indices = rng.choice(
                tupled_indices, len(tupled_indices), replace=False
            )

            for index_tuple in tupled_indices:
                loading_indices = indices[index_tuple[0] : index_tuple[1]]
                labels = label_binarize(
                    (np.ones(index_tuple[1] - index_tuple[0]) * label),
                    classes=label_classes,
                )
                if use_tracks:
                    yield f["jets"][loading_indices], f["tracks"][
                        loading_indices
                    ], labels
                else:
                    yield f["jets"][loading_indices], labels

    def Run(self):
        logger.info("Starting undersampling.")
        self.InitialiseSamples()
        self.GetIndices()

        logger.info("Plotting distributions before undersampling.")
        plot_name_clean = self.config.GetFileName(
            extension="", option="pt_eta-wider_bin", custom_path="plots/"
        )
        ResamplingPlots(
            concat_samples=self.concat_samples,
            positions_x_y=[0, 1],
            variable_names=["pT", "abseta"],
            plot_base_name=plot_name_clean,
            binning={"pT": 200, "abseta": 20},
            Log=True,
        )

        logger.info("Plotting distributions after undersampling.")
        plot_name_clean = self.config.GetFileName(
            extension="",
            option="downsampled-pt_eta-wider_bins",
            custom_path="plots/",
        )
        ResamplingPlots(
            concat_samples=self.x_y_after_sampling,
            positions_x_y=[0, 1],
            variable_names=["pT", "abseta"],
            plot_base_name=plot_name_clean,
            binning={
                "pT": 200,
                "abseta": 20,
            },
            Log=True,
            after_sampling=True,
        )
        logger.info(f"Saving plots as {plot_name_clean}(pT|abseta).pdf")

        # Write file to disk
        self.WriteFile(self.indices_to_keep)

    # TODO: write the following functions also for flexible amount of classes
    # plot_name = self.config.GetFileName(
    #     x + 1,
    #     option="downsampled-pt_eta",
    #     extension=".pdf",
    #     custom_path="plots/",
    # )
    # upt.MakePlots(
    #     bjets,
    #     cjets,
    #     ujets,
    #     taujets,
    #     plot_name=plot_name,
    #     binning={
    #         global_config.pTvariable: downs.pt_bins,
    #         global_config.etavariable: downs.eta_bins,
    #     },
    # )
    # plot_name = self.config.GetFileName(
    #     x + 1,
    #     extension=".pdf",
    #     option="downsampled-pt_eta-wider_bins",
    #     custom_path="plots/",
    # )
    # upt.MakePlots(
    #     bjets,
    #     cjets,
    #     ujets,
    #     taujets,
    #     plot_name=plot_name,
    #     binning={
    #         global_config.pTvariable: 200,
    #         global_config.etavariable: 20,
    #     },
    # )


class UnderSamplingProp(object):
    """
    Alternative to the UnderSampling approach, this implements a
    proportional sampler to prepare the training dataset. It makes sure
    that in each pT/eta bin each category has the same ratio of jets.
    This is especially suited if not enough statistics is available for
    some of the labels.
    For example, in bin X, if 1% of b, 2% of c, 3 % of l jets are found,
    sampler will take 1% of all b, 1% of all c and 1% of all l in the bin.
    """

    def __init__(self, bjets, cjets, ujets, taujets=None):
        super(UnderSamplingProp, self).__init__()
        self.bjets = bjets
        self.cjets = cjets
        self.ujets = ujets
        self.taujets = taujets
        self.bool_taujets = taujets is not None
        self.pt_bins = np.concatenate(
            (np.linspace(0, 600000, 351), np.linspace(650000, 6000000, 84))
        )
        self.eta_bins = np.linspace(0, 2.5, 10)
        self.nbins = np.array([len(self.pt_bins), len(self.eta_bins)])
        self.pT_var_name = global_config.pTvariable
        self.eta_var_name = global_config.etavariable
        self.rnd_seed = 42

    def GetIndices(self):
        """
        Applies the weighted UnderSampling to the given arrays.
        Returns the indices for the jets to be used separately for b, c and
        light jets (as well as taus, optionally).
        """
        binnumbers_b, ind_b, stat_b, total_b = self.GetBins(self.bjets)
        binnumbers_c, _, stat_c, total_c = self.GetBins(self.cjets)
        binnumbers_u, _, stat_u, total_u = self.GetBins(self.ujets)
        if self.bool_taujets:
            binnumbers_tau, _, stat_tau, total_tau = self.GetBins(self.taujets)
            min_weight_per_bin = np.amin(
                [stat_b, stat_c, stat_u, stat_tau], axis=0
            )
        else:
            min_weight_per_bin = np.amin([stat_b, stat_c, stat_u], axis=0)

        bjet_indices = []
        cjet_indices = []
        ujet_indices = []
        taujet_indices = []

        for elem, weight in zip(ind_b, min_weight_per_bin):
            np.random.seed(self.rnd_seed)
            bjet_indices.append(
                np.random.choice(
                    np.where(binnumbers_b == elem)[0],
                    int(weight * total_b),
                    replace=False,
                )
            )
            np.random.seed(self.rnd_seed)
            cjet_indices.append(
                np.random.choice(
                    np.where(binnumbers_c == elem)[0],
                    int(weight * total_c),
                    replace=False,
                )
            )
            np.random.seed(self.rnd_seed)
            ujet_indices.append(
                np.random.choice(
                    np.where(binnumbers_u == elem)[0],
                    int(weight * total_u),
                    replace=False,
                )
            )
            if self.bool_taujets:
                np.random.seed(self.rnd_seed)
                taujet_indices.append(
                    np.random.choice(
                        np.where(binnumbers_tau == elem)[0],
                        int(weight * total_tau),
                        replace=False,
                    )
                )
        if self.bool_taujets:
            sorted_taujets = np.sort(np.concatenate(taujet_indices))
        else:
            sorted_taujets = None

        return (
            np.sort(np.concatenate(bjet_indices)),
            np.sort(np.concatenate(cjet_indices)),
            np.sort(np.concatenate(ujet_indices)),
            sorted_taujets,
        )

    def GetBins(self, df):
        statistic, xedges, yedges, binnumber = binned_statistic_2d(
            x=df[self.pT_var_name],
            y=df[self.eta_var_name],
            values=df[self.pT_var_name],
            statistic="count",
            bins=[self.pt_bins, self.eta_bins],
        )

        bins_indices_flat_2d = np.indices(self.nbins - 1) + 1
        bins_indices_flat = np.ravel_multi_index(
            bins_indices_flat_2d, self.nbins + 1
        ).flatten()

        total_count = df.shape[0]
        weighted_flatten_statistic = statistic.flatten() / total_count

        return (
            binnumber,
            bins_indices_flat,
            weighted_flatten_statistic,
            total_count,
        )


class ProbabilityRatioUnderSampling(UnderSampling):
    """
    The ProbabilityRatioUnderSampling is used to prepare the training dataset.
    It makes sure that all flavor fractions are equal and the flavours distributions
    have the same shape as the target distribution.
    This is an alternative to the class UnderSampling, with the difference that
    it always ensures the target distribution is the target distribution,
    regardless of pre-sampling flavor fractions and low statistics.
    Does not work well with taus as of now.
    """

    def GetIndices(self):
        """
        Applies the UnderSampling to the given arrays.

        Parameters
        ----------

        Returns
        -------
        Returns the indices for the jets to be used separately for each
        category and sample.

        """
        try:
            target_distribution = self.options["target_distribution"]
        except KeyError:
            raise ValueError(
                "Resampling method probabilty_ratio requires a target distribution class "
                "in the options block of the configuration file (i.e. bjets, cjets, ujets)."
            )

        concat_samples = {
            elem: {"jets": None} for elem in self.class_categories
        }
        for sample_category in self.samples:
            for sample in self.samples[sample_category]:
                if concat_samples[sample["category"]]["jets"] is None:
                    concat_samples[sample["category"]]["jets"] = sample[
                        "sample_vector"
                    ]
                else:
                    concat_samples[sample["category"]][
                        "jets"
                    ] = np.concatenate(
                        [
                            concat_samples[sample["category"]]["jets"],
                            sample["sample_vector"],
                        ]
                    )

        # calculate the 2D bin statistics for each sample
        bin_indices_flat = []
        for class_category in self.class_categories:
            binnumbers, ind, stat = self.GetBins(
                concat_samples[class_category]["jets"][:, 0],
                concat_samples[class_category]["jets"][:, 1],
            )
            concat_samples[class_category]["binnumbers"] = binnumbers
            concat_samples[class_category]["stat"] = stat
            bin_indices_flat = ind

        self.concat_samples = concat_samples

        min_count_per_bin = np.amin(
            [concat_samples[sample]["stat"] for sample in concat_samples],
            axis=0,
        )
        max_probability_ratio = self.GetMaxProbabilityRatio(
            {
                sample: concat_samples[sample]["stat"]
                for sample in concat_samples
            },
            target_distribution,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_ratios = np.nan_to_num(
                concat_samples[target_distribution]["stat"]
                / min_count_per_bin
                / max_probability_ratio
            )

        rng = np.random.default_rng(seed=self.rnd_seed)
        indices_to_keep = {elem: [] for elem in self.class_categories}
        # retrieve the indices of the jets to keep
        for bin_i, count, prob_ratio in zip(
            bin_indices_flat, min_count_per_bin, prob_ratios
        ):
            for class_category in self.class_categories:
                indices_to_keep[class_category].append(
                    rng.choice(
                        np.nonzero(
                            concat_samples[class_category]["binnumbers"]
                            == bin_i
                        )[0],
                        int(count * prob_ratio),
                        replace=False,
                    )
                )
        # join the indices into one array per class category
        for class_category in self.class_categories:
            indices_to_keep[class_category] = np.concatenate(
                indices_to_keep[class_category]
            )

        # check if more jets are available as requested in the config file
        # we assume that all classes have the same number of jets now
        if len(indices_to_keep[target_distribution]) >= self.options[
            "njets"
        ] // len(self.class_categories):
            size_per_class = self.options["njets"] // len(
                self.class_categories
            )
            for class_category in self.class_categories:
                indices_to_keep[class_category] = rng.choice(
                    indices_to_keep[class_category],
                    size=int(size_per_class)
                    if size_per_class <= len(indices_to_keep[class_category])
                    else len(indices_to_keep[class_category]),
                    replace=False,
                )
        else:
            size_per_class = len(indices_to_keep[target_distribution])
            size_total = len(indices_to_keep[target_distribution]) * len(
                self.class_categories
            )
            logger.warning(
                f"You asked for {self.options['njets']:.0f} jets, however, "
                f"only {size_total} are available."
            )
        # get indices per single sample
        self.indices_to_keep = {}
        self.x_y_after_sampling = {}
        size_total = 0
        with h5py.File(self.options["intermediate_index_file"], "w") as f:
            for class_category in self.class_categories:
                self.x_y_after_sampling[class_category] = concat_samples[
                    class_category
                ]["jets"][indices_to_keep[class_category], :2]
                sample_categories = concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 4
                ]
                sample_indices = concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 2
                ]
                for sample_category in self.sample_categories:
                    sample_name = self.sample_map[sample_category][
                        class_category
                    ]
                    self.indices_to_keep[sample_name] = np.sort(
                        sample_indices[
                            sample_categories
                            == self.sample_categories[sample_category]
                        ]
                    ).astype(int)
                    f.create_dataset(
                        sample_name,
                        data=self.indices_to_keep[sample_name],
                        compression="gzip",
                    )
                    sample_size = len(self.indices_to_keep[sample_name])
                    size_total += sample_size
                    logger.info(
                        f"Using {sample_size} jets from {sample_name}."
                    )
        logger.info(f"Using in total {size_total} jets.")
        return self.indices_to_keep

    def GetMaxProbabilityRatio(self, stats: dict, target_distribution: str):
        """
        Computes probability ratios against the target distribution and retuns the max.
        This is used to scale the other distributions, i.e. cjets, ujets, taujets, so that
        they are alwasy above the target distribution and with the same shape as the
        target distribution.

        Parameters
        ----------
        stats: Dictionary of stats such as bin count for different jet flavours
        target_distribution: the target distribution, i.e. bjets, to compute
        probability ratios against

        Returns
        -------
        The max probability ratio.

        """
        stat_target = stats.get(target_distribution)
        if stat_target is None:
            raise ValueError(
                "Target distribution class does not exist in your sample classes."
            )
        logger.info(f"target_distribution, {target_distribution}")
        probability_ratios = []
        for sample in stats:
            if sample != target_distribution:
                with np.errstate(
                    divide="ignore", invalid="ignore"
                ), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ratios = stat_target.astype(float) / stats[sample]
                    max_ratio = np.nanmax(ratios)
                    logger.info(f"max_ratio for {sample}, {max_ratio}")
                    probability_ratios.append(max_ratio)

        max_probability_ratio = np.amax(
            probability_ratios, where=~np.isnan(probability_ratios), initial=-1
        )
        return max_probability_ratio


def GetScales(vec, w, varname, custom_defaults_vars):
    """
    Calculates the weighted average and std for vector vec and weight w.
    """
    if np.sum(w) == 0:
        raise ValueError("Sum of weights has to be >0.")
    # find NaN values
    nans = np.isnan(vec)
    # check if variable has predefined default value
    if varname in custom_defaults_vars:
        default = custom_defaults_vars[varname]
    # NaN values are not considered in calculation for average
    else:
        w_without_nan = w[~nans]
        vec_without_nan = vec[~nans]
        default = np.ma.average(vec_without_nan, weights=w_without_nan)
    # replace NaN values with default values
    vec[nans] = default
    average = np.ma.average(vec, weights=w)
    std = np.sqrt(np.average((vec - average) ** 2, weights=w))
    return varname, average, std, default


def dict_in(varname, average, std, default):
    """
    Creates dictionary entry containing scale and shift parameters.
    """
    return {
        "name": varname,
        "shift": average,
        "scale": std,
        "default": default,
    }


def Gen_default_dict(scale_dict):
    """
    Generates default value dictionary from scale/shift dictionary.
    """
    default_dict = {}
    for elem in scale_dict:
        if "isDefaults" in elem["name"]:
            continue
        default_dict[elem["name"]] = elem["default"]
    return default_dict
