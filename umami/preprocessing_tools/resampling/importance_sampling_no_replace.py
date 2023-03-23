"""Importance sampling without replacement module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import numpy as np

from umami.configuration import logger
from umami.preprocessing_tools.resampling.count_sampling import SimpleSamplingBase


class UnderSamplingNoReplace(SimpleSamplingBase):
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

    def get_indices(self) -> dict:
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
            target_distribution = self.options.target_distribution
        except KeyError as error:
            raise ValueError(
                "Resampling method importance_no_replace requires a target"
                " distribution class in the options block of the configuration"
                " file (i.e. bjets, cjets, ujets)."
            ) from error

        # Concatenate the samples with the same category into dict which contains the
        # samplevector: array(sample_size x 5)
        # with pt, eta, jet_count, sample_id (ttbar:0, zprime:1), sample_class
        self.concatenate_samples()

        # calculate the 2D bin statistics for each sample
        self.get_pt_eta_bin_statistics()

        # create dictionary of just the samples' 2D stats
        stats = {
            distribution: self.concat_samples[distribution]["stat"]
            for distribution in self.concat_samples
        }

        # Get the sampling probabilities relative to the target.
        # We can do this for all the flavours since the sampling probabilities
        # for the target with respect to the target distribution is just 1
        sampling_probabilities = self.get_sampling_probabilities(
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
        n_jets_requested = (
            self.options.n_jets
            if not self.use_validation_samples
            else self.options.n_jets_validation
        )
        if n_jets_requested == -1 or n_jets_requested is None:
            logger.info("Maximising number of jets to target distribution.")
        else:
            logger.info("Requesting %i in total.", n_jets_requested)
            n_jets_per_class = n_jets_requested // len(self.class_categories)
            if len(indices_to_keep[target_distribution]) >= n_jets_per_class:
                for class_category in self.class_categories:
                    indices_to_keep[class_category] = rng.choice(
                        indices_to_keep[class_category],
                        size=int(n_jets_per_class)
                        if n_jets_per_class <= len(indices_to_keep[class_category])
                        else len(indices_to_keep[class_category]),
                        replace=False,
                    )
            else:
                n_jets_per_class = len(indices_to_keep[target_distribution])
                size_total = n_jets_per_class * len(self.class_categories)
                logger.warning(
                    "You asked for %i jets, however, only %i are available.",
                    n_jets_requested,
                    size_total,
                )

        # get indices per single sample
        self.indices_to_keep = self._indices_concat_to_per_sample(indices_to_keep)
        self._save_indices()
        return self.indices_to_keep

    def get_sampling_probability(  # pylint: disable=no-self-use
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

    def get_sampling_probabilities(
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
            sampling_probabilities[flav] = self.get_sampling_probability(
                target_stat, stats[flav]
            )

        return sampling_probabilities
