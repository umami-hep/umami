"""Importance sampling without replacement module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import os

import h5py
import numpy as np

from umami.configuration import logger
from umami.preprocessing_tools.resampling.resampling_base import ResamplingTools
from umami.preprocessing_tools.utils import (
    GetVariableDict,
    ResamplingPlots,
    generate_process_tag,
    preprocessing_plots,
)


class UnderSamplingNoReplace(ResamplingTools):
    """
    The UnderSamplingNoReplace is used to prepare the training dataset.
    It makes sure that all flavour fractions are equal and the flavour distributions
    have the same shape as the target distribution.
    This is an alternative to the UnderSampling class, with the difference that
    it ensures that the predefined target distribution is always the final target
    distribution, regardless of pre-sampling flavour fractions and low statistics.
    This method also ensures that the final fractions are equal.
    Does not work well with taus as of now.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.x_y_after_sampling = None
        self.indices_to_keep = None

    def GetIndices(self) -> dict:
        """
        Applies the undersampling to the given arrays.

        Returns
        -------
        np.ndarray
            Indices for the jets to be used separately for each
            category and sample.

        Raises
        ------
        ValueError
            If no target is given.
        """

        try:
            target_distribution = self.options["target_distribution"]
        except KeyError as Error:
            raise ValueError(
                "Resampling method importance_no_replace requires a target"
                " distribution class in the options block of the configuration"
                " file (i.e. bjets, cjets, ujets)."
            ) from Error

        # Concatenate the samples with the same category into dict which contains the
        # samplevector: array(sample_size x 5)
        # with pt, eta, jet_count, sample_id (ttbar:0, zprime:1), sample_class
        self.ConcatenateSamples()

        # calculate the 2D bin statistics for each sample
        self.GetPtEtaBinStatistics()

        # create dictionary of just the samples' 2D stats
        stats = {
            distribution: self.concat_samples[distribution]["stat"]
            for distribution in self.concat_samples
        }

        # Get the sampling probabilities relative to the target.
        # We can do this for all the flavours since the sampling probabilities
        # for the target with respect to the target distribution is just 1
        sampling_probabilities = self.GetSamplingProbabilities(
            target_distribution,
            stats,
        )

        # Scale stats by the max sampling probability to ensure the
        # the distributions gets sacled below the target distribution
        stats = {
            sample: stats[sample] / np.max(sampling_probabilities[sample])
            for sample in stats
        }

        # Multiply scaled stats by the sampling probabilities which results
        # in the stats having the same shape as the target distribution.
        stats = {
            sample: stats[sample] * sampling_probabilities[sample] for sample in stats
        }

        # Get the minimum count per pT/eta bin of the scaled and weighted stats
        min_count_per_bin = np.amin([stats[sample] for sample in stats], axis=0)

        # retrieve the indices of the jets to keep
        indices_to_keep = {elem: [] for elem in self.class_categories}
        rng = np.random.default_rng(seed=self.rnd_seed)
        for bin_i, min_count in zip(self.bin_indices_flat, min_count_per_bin):
            for class_category in self.class_categories:
                indices_to_keep[class_category].append(
                    rng.choice(
                        np.nonzero(
                            self.concat_samples[class_category]["binnumbers"] == bin_i
                        )[0],
                        int(min_count),
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
        njets_requested = self.options.get("njets", None)
        if njets_requested == -1 or njets_requested is None:
            logger.info("Maximising number of jets to target distribution.")
        else:
            logger.info(f"Requesting {njets_requested} in total.")
            njets_per_class = njets_requested // len(self.class_categories)
            if len(indices_to_keep[target_distribution]) >= njets_per_class:
                for class_category in self.class_categories:
                    indices_to_keep[class_category] = rng.choice(
                        indices_to_keep[class_category],
                        size=int(njets_per_class)
                        if njets_per_class <= len(indices_to_keep[class_category])
                        else len(indices_to_keep[class_category]),
                        replace=False,
                    )
            else:
                njets_per_class = len(indices_to_keep[target_distribution])
                size_total = njets_per_class * len(self.class_categories)
                logger.warning(
                    f"You asked for {njets_requested:.0f} jets, however, "
                    f"only {size_total} are available."
                )

        # get indices per single sample
        self.indices_to_keep = {}
        self.x_y_after_sampling = {}
        size_total = 0
        with h5py.File(self.options["intermediate_index_file"], "w") as f:
            for class_category in self.class_categories:
                self.x_y_after_sampling[class_category] = self.concat_samples[
                    class_category
                ]["jets"][indices_to_keep[class_category], :2]
                sample_categories = self.concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 4
                ]
                sample_indices = self.concat_samples[class_category]["jets"][
                    indices_to_keep[class_category], 2
                ]
                for sample_category in self.sample_categories:
                    sample_name = self.sample_map[sample_category][class_category]
                    self.indices_to_keep[sample_name] = np.sort(
                        sample_indices[
                            sample_categories == self.sample_categories[sample_category]
                        ]
                    ).astype(int)
                    f.create_dataset(
                        sample_name,
                        data=self.indices_to_keep[sample_name],
                        compression="gzip",
                    )
                    sample_size = len(self.indices_to_keep[sample_name])
                    size_total += sample_size
                    logger.info(f"Using {sample_size} jets from {sample_name}.")
        logger.info(f"Using in total {size_total} jets.")
        return self.indices_to_keep

    def GetSamplingProbability(  # pylint: disable=no-self-use
        self,
        target_stat: np.ndarray,
        original_stat: np.ndarray,
    ) -> dict:
        """
        Computes probability ratios against the target distribution.

        Parameters
        ----------
        target_stat : np.ndarray
            Target distribution or histogram, i.e. bjets histo, to compute
            probability ratios against.
        original_stat : np.ndarray
            Original distribution or histogram, i.e. cjets histo, to
            scale using target_stat.

        Returns
        -------
        dict
            A dictionary of the sampling probabilities for each flavour.
        """

        ratios = np.divide(
            target_stat,
            original_stat,
            out=np.zeros(
                original_stat.shape,
                dtype=float,
            ),
            where=(original_stat != 0),
        )
        return ratios

    def GetSamplingProbabilities(
        self,
        target_distribution: str = "bjets",
        stats: dict = None,
    ) -> dict:
        """
        Computes probability ratios against the target distribution for each flavour.
        The probabiliy sampling ensures the resulting flavour fractions are the same
        and distributions shapes are the same.

        Parameters
        ----------
        target_distribution : str, optional
            Target distribution, i.e. bjets, to compute
            probability ratios against, by default "bjets"
        stats : dict, optional
            Dictionary of stats such as bin count for different jet flavours,
            by default None

        Returns
        -------
        dict
            A dictionary of the sampling probabilities for each flavour.

        Raises
        ------
        ValueError
            If target distribution class does not exist in your sample classes.
        """

        if stats is None:
            stats = {
                "bjets": np.ndarray,
                "cjets": np.ndarray,
                "ujets": np.ndarray,
                "taujets": np.ndarray,
            }
        if target_distribution is None:
            raise ValueError(
                "Target distribution class does not exist in your sample classes."
            )
        target_stat = stats.get(target_distribution)

        sampling_probabilities = {}
        # For empty stats set the sampling probability to zero
        for flav in stats:
            sampling_probabilities[flav] = self.GetSamplingProbability(
                target_stat, stats[flav]
            )

        return sampling_probabilities

    def Run(self):
        """Run function executing full chain."""
        logger.info("Starting undersampling.")
        self.InitialiseSamples()
        self.GetIndices()

        logger.info("Plotting distributions before undersampling.")

        # Check if the directory for the plots exists
        plot_dir_path = os.path.join(
            self.resampled_path,
            "plots/",
        )
        os.makedirs(plot_dir_path, exist_ok=True)

        plot_name_clean = self.config.GetFileName(
            extension="",
            option="pt_eta-wider_bin_",
            custom_path=plot_dir_path,
        )
        ResamplingPlots(
            concat_samples=self.concat_samples,
            positions_x_y=[0, 1],
            variable_names=["pT", "abseta"],
            plot_base_name=plot_name_clean,
            normalised=False,
            binning={"pT": 200, "abseta": 20},
            Log=True,
            second_tag=generate_process_tag(self.config.preparation["ntuples"].keys()),
        )

        logger.info("Plotting distributions after undersampling.")
        plot_name_clean = self.config.GetFileName(
            extension="",
            option="downsampled-pt_eta-wider_bins_",
            custom_path=plot_dir_path,
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
            second_tag=generate_process_tag(self.config.preparation["ntuples"].keys()),
        )
        logger.info(f"Saving plots as {plot_name_clean}(pT|abseta).pdf")

        # Write file to disk
        self.WriteFile(self.indices_to_keep)

        # Plot the variables from the output file of the resampling process
        if "njets_to_plot" in self.options and self.options["njets_to_plot"]:
            preprocessing_plots(
                sample=self.config.GetFileName(option="resampled"),
                var_dict=GetVariableDict(self.config.var_file),
                class_labels=self.config.sampling["class_labels"],
                plots_dir=os.path.join(
                    self.resampled_path,
                    "plots/resampling/",
                ),
                track_collection_list=self.options["tracks_names"]
                if "tracks_names" in self.options
                and "save_tracks" in self.options
                and self.options["save_tracks"] is True
                else None,
                nJets=self.options["njets_to_plot"],
            )
