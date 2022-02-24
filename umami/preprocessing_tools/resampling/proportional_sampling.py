"""Proportional sampling module handling data preprocessing."""
# pylint: disable=attribute-defined-outside-init,no-self-use
import numpy as np
from scipy.stats import binned_statistic_2d

from umami.configuration import global_config


class UnderSamplingProp:
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
        super().__init__()
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

    def GetIndices(self) -> tuple:
        """
        Applies the weighted UnderSampling to the given arrays.
        Returns the

        Returns
        -------
        tuple
            Indices for the jets to be used separately for b, c and
            light jets (as well as taus, optionally).
        """

        binnumbers_b, ind_b, stat_b, total_b = self.GetBins(self.bjets)
        binnumbers_c, _, stat_c, total_c = self.GetBins(self.cjets)
        binnumbers_u, _, stat_u, total_u = self.GetBins(self.ujets)
        if self.bool_taujets:
            binnumbers_tau, _, stat_tau, total_tau = self.GetBins(self.taujets)
            min_weight_per_bin = np.amin([stat_b, stat_c, stat_u, stat_tau], axis=0)
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

    def GetBins(self, df: dict):
        """
        Retrieving bins.

        Parameters
        ----------
        df : dict
            Dict with the pT and eta variable inside.

        Returns
        -------
        tuple
            The binnumber, bin indicies flat, weighted flatten statistics
            and the total count.
        """

        statistic, binnumber = binned_statistic_2d(
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
