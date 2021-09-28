import itertools
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from umami.configuration import global_config, logger
from umami.preprocessing_tools import GetPreparationSamplePath

from .utils import ResamplingPlots


def CorrectFractions(
    N_jets: list, target_fractions: list, class_names: list = None
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
        self.outfile_name = self.config.GetFileName()

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


class PDFResampling(Resampling):
    def Run(self):
        pass


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
