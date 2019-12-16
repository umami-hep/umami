import numpy as np
import yaml
import os
import warnings
from umami.tools import yaml_loader
# import h5py
# from numpy.lib.recfunctions import repack_fields
# import json
# import yaml
# from keras.utils import np_utils


class DownSampling(object):
    """The DownSampling is used to prepare the training dataset. It makes sure
    that in each pT/eta bin the same amount of jets are filled."""

    def __init__(self, bjets, cjets, ujets, run_immediatly=True):
        super(DownSampling, self).__init__()
        self.bjets = bjets
        self.cjets = cjets
        self.ujets = ujets
        self.pt_bins = np.concatenate((np.linspace(0, 600000, 351),
                                       np.linspace(650000, 6000000, 84)))
        self.eta_bins = np.linspace(0, 2.5, 10)
        self.pT_var_name = 'pt_uncalib'
        self.eta_var_name = 'abs_eta_uncalib'
        if run_immediatly:
            self.GetIndices()

    def GetIndices(self):
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

        b_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        c_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        u_loc_indices = {(pti, etai): [] for pti, _ in
                         enumerate(self.pt_bins[::-1]) for etai, _ in
                         enumerate(self.eta_bins[::-1])}
        print('Grouping the bins')
        ignored_over_underflow = False
        for i, x in enumerate(b_locations):
            if x not in b_loc_indices:
                ignored_over_underflow = True
                continue
            b_loc_indices[x].append(i)

        for i, x in enumerate(c_locations):
            if x not in c_loc_indices:
                ignored_over_underflow = True
                continue
            c_loc_indices[x].append(i)

        for i, x in enumerate(u_locations):
            if x not in u_loc_indices:
                ignored_over_underflow = True
                continue
            u_loc_indices[x].append(i)
        if ignored_over_underflow:
            print("# WARNING: You have jets in your sample which are not in",
                  "the provided bins.")

        bjet_indices = []
        cjet_indices = []
        ujet_indices = []
        print('Matching the bins for all flavours')

        for pt_bin_i in range(len(self.pt_bins) - 1):
            for eta_bin_i in range(len(self.eta_bins) - 1):
                loc = (pt_bin_i, eta_bin_i)

                nbjets = int(histvals_b[eta_bin_i][pt_bin_i])
                ncjets = int(histvals_c[eta_bin_i][pt_bin_i])
                nujets = int(histvals_u[eta_bin_i][pt_bin_i])

                njets = min([nbjets, ncjets, nujets])
                b_indices_for_bin = b_loc_indices[loc][0:njets]
                c_indices_for_bin = c_loc_indices[loc][0:njets]
                u_indices_for_bin = u_loc_indices[loc][0:njets]
                bjet_indices += b_indices_for_bin
                cjet_indices += c_indices_for_bin
                ujet_indices += u_indices_for_bin

        bjet_indices.sort()
        cjet_indices.sort()
        ujet_indices.sort()

        return np.array(bjet_indices), np.array(cjet_indices),\
            np.array(ujet_indices)


class Configuration(object):
    """docstring for Configuration."""

    def __init__(self, yaml_config=None):
        super(Configuration, self).__init__()
        self.yaml_config = yaml_config
        self.yaml_default_config = "configs/preprocessing_default_config.yaml"
        self.LoadConfigFiles()
        self.GetConfiguration()

    def LoadConfigFiles(self):
        self.yaml_default_config = os.path.join(os.path.dirname(__file__),
                                                self.yaml_default_config)
        with open(self.yaml_default_config, "r") as conf:
            self.default_config = yaml.load(conf, Loader=yaml_loader)
        print("Using config file", self.yaml_config)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf, Loader=yaml_loader)

    def GetConfiguration(self):
        for elem in self.default_config:
            if elem in self.config:
                if type(self.config[elem]) is dict and "f_" in elem:
                    if 'file' not in self.config[elem]:
                        raise KeyError("You need to specify the 'file' for"
                                       f"{elem} in your config file!")
                    if self.config[elem]['file'] is None:
                        raise KeyError("You need to specify the 'file' for"
                                       f" {elem} in your config file!")
                    if 'path' in self.config[elem]:
                        setattr(self, elem,
                                os.path.join(self.config[elem]['path'],
                                             self.config[elem]['file'])
                                )
                    else:
                        setattr(self, elem, self.config[elem]['file'])

                else:
                    setattr(self, elem, self.config[elem])
            elif self.default_config[elem] is None:
                raise KeyError(f"You need to specify {elem} in your"
                               "config file!")
            else:
                warnings.warn(f"setting {elem} to default value "
                              f"{self.default_config[elem]}")
                setattr(self, elem, self.default_config[elem])


def GetNJetsPerIteration(config):
    print(type(config.ttbar_frac))
    print(type(config.njets))
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
        N_dict = {"nZ": nZ * x,
                  "nbjets": njets * x,
                  "ncjets": ncjets * x,
                  "nujets": nujets * x
                  }
        N_list.append(N_dict)
    return N_list
