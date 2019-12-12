import numpy as np

# import h5py
# from numpy.lib.recfunctions import repack_fields
# import pandas as pd
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
        self.pT_var_name = 'pt_uncalib'
        self.eta_var_name = 'abs_eta_uncalib'

    def Apply(self):
        """Applies the DownSampling to the given arrays.
        Returns the indices for the jets to be used separately for b,c and
        light jets."""
        histvals_b, _, _ = np.histogram2d(self.bjets[self.eta_var_name],
                                          self.bjets[self.pT_var_name],
                                          [self.eta_bins, self.pt_bins])
        histvals_c, _, _ = np.histogram2d(self.cjets[self.eta_var_name],
                                          self.cjets[self.pT_var_name],
                                          [self.eta_bins, self.pt_bins])
        histvals_u, _, _ = np.histogram2d(self.ujets[self.eta_var_name],
                                          self.ujets[self.pT_var_name],
                                          [self.eta_bins, self.pt_bins])

        b_locations_pt = np.digitize(self.bjets[self.pT_var_name],
                                     self.pt_bins) - 1
        b_locations_eta = np.digitize(self.bjets[self.eta_var_name],
                                      self.eta_bins) - 1
        b_locations = zip(b_locations_pt, b_locations_eta)
        b_locations = list(b_locations)

        c_locations_pt = np.digitize(self.cjets[self.pT_var_name],
                                     self.pt_bins) - 1
        c_locations_eta = np.digitize(self.cjets[self.eta_var_name],
                                      self.eta_bins) - 1
        c_locations = zip(c_locations_pt, c_locations_eta)
        c_locations = list(c_locations)

        u_locations_pt = np.digitize(self.ujets[self.pT_var_name],
                                     self.pt_bins) - 1
        u_locations_eta = np.digitize(self.ujets[self.eta_var_name],
                                      self.eta_bins) - 1
        u_locations = zip(u_locations_pt, u_locations_eta)
        u_locations = list(u_locations)

        c_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        b_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        u_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        print('Grouping the bins')
        for i, x in enumerate(c_locations):
            c_loc_indices[x].append(i)

        for i, x in enumerate(b_locations):
            b_loc_indices[x].append(i)

        for i, x in enumerate(u_locations):
            u_loc_indices[x].append(i)

        cjet_indices = []
        bjet_indices = []
        ujet_indices = []
        print('Matching the bins for all flavours')

        for pt_bin_i in range(len(self.pt_bins) - 1):
            for eta_bin_i in range(len(self.eta_bins) - 1):
                loc = (pt_bin_i, eta_bin_i)

                nbjets = int(histvals_b[eta_bin_i][pt_bin_i])
                ncjets = int(histvals_c[eta_bin_i][pt_bin_i])
                nujets = int(histvals_u[eta_bin_i][pt_bin_i])

                njets = min([nbjets, ncjets, nujets])
                c_indices_for_bin = c_loc_indices[loc][0:njets]
                b_indices_for_bin = b_loc_indices[loc][0:njets]
                u_indices_for_bin = u_loc_indices[loc][0:njets]
                cjet_indices += c_indices_for_bin
                bjet_indices += b_indices_for_bin
                ujet_indices += u_indices_for_bin

        cjet_indices.sort()
        bjet_indices.sort()
        ujet_indices.sort()

        return np.array(bjet_indices), np.array(cjet_indices),\
            np.array(ujet_indices)
