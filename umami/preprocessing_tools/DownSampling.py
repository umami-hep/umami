import numpy as np
# import yaml
# import os
# import warnings
from scipy.stats import binned_statistic_2d

# from umami.tools import yaml_loader
# import h5py
# from numpy.lib.recfunctions import repack_fields
# import json
# import yaml
# from keras.utils import np_utils


class DownSampling(object):
    """The DownSampling is used to prepare the training dataset. It makes sure
    that in each pT/eta bin the same amount of jets are filled."""

    def __init__(self, bjets, cjets, ujets):
        super(DownSampling, self).__init__()
        self.bjets = bjets
        self.cjets = cjets
        self.ujets = ujets
        self.pt_bins = np.concatenate((np.linspace(0, 600000, 351),
                                       np.linspace(650000, 6000000, 84)))
        self.eta_bins = np.linspace(0, 2.5, 10)
        self.nbins = np.array([len(self.pt_bins), len(self.eta_bins)])
        self.pT_var_name = 'pt_uncalib'
        self.eta_var_name = 'abs_eta_uncalib'

    def GetIndices(self):
        """Applies the DownSampling to the given arrays.
        Returns the indices for the jets to be used separately for b,c and
        light jets."""
        binnumbers_b, ind_b, stat_b = self.GetBins(self.bjets)
        binnumbers_c, _, stat_c = self.GetBins(self.cjets)
        binnumbers_u, _, stat_u = self.GetBins(self.ujets)
        min_count_per_bin = np.amin([stat_b, stat_c, stat_u], axis=0)

        bjet_indices = []
        cjet_indices = []
        ujet_indices = []
        
        for elem, count in zip(ind_b, min_count_per_bin):
            bjet_indices.append(np.where(binnumbers_b == elem)[0][:int(count)])
            cjet_indices.append(np.where(binnumbers_c == elem)[0][:int(count)])
            ujet_indices.append(np.where(binnumbers_u == elem)[0][:int(count)])

        return np.sort(np.concatenate(bjet_indices)),\
            np.sort(np.concatenate(cjet_indices)),\
            np.sort(np.concatenate(ujet_indices))

    def GetBins(self, df):
        statistic, xedges, yedges, binnumber = binned_statistic_2d(
            x=df[self.pT_var_name],
            y=df[self.eta_var_name],
            values=df[self.pT_var_name],
            statistic='count', bins=[self.pt_bins, self.eta_bins])

        bins_indices_flat_2d = np.indices(self.nbins - 1) + 1
        bins_indices_flat = np.ravel_multi_index(
            bins_indices_flat_2d, self.nbins + 1).flatten()

        return binnumber, bins_indices_flat, statistic.flatten()


def GetNJetsPerIteration(config):
    if config.iterations == 0:
        raise ValueError("The iterations have to be >=1 and not 0.")
    if config.ttbar_frac > 0.:
        nZ = (int(config.njets) * 3 * (1 / config.ttbar_frac - 1)
              ) // config.iterations
        ncjets = int(2.3 * config.njets) // config.iterations
        nujets = int(2.7 * config.njets) // config.iterations
        njets = int(config.njets) // config.iterations
    else:
        nZ = int(config.njets) // config.iterations
        ncjets = nujets = njets = 0

    N_list = []
    for x in range(config.iterations + 1):
        N_dict = {"nZ": int(nZ * x),
                  "nbjets": int(njets * x),
                  "ncjets": int(ncjets * x),
                  "nujets": int(nujets * x)
                  }
        N_list.append(N_dict)
    return N_list
