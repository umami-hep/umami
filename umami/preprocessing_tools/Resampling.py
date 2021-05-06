import numpy as np
from scipy.stats import binned_statistic_2d


class UnderSampling(object):
    """
    The DownSampling is used to prepare the training dataset. It makes sure
    that in each pT/eta bin the same amount of jets are filled.
    """

    def __init__(self, bjets, cjets, ujets, tjets=None):
        super(UnderSampling, self).__init__()
        self.bjets = bjets
        self.cjets = cjets
        self.ujets = ujets
        self.tjets = tjets
        self.bool_tjets = tjets is not None
        self.pt_bins = np.concatenate(
            (np.linspace(0, 600000, 351), np.linspace(650000, 6000000, 84))
        )
        self.eta_bins = np.linspace(0, 2.5, 10)
        self.nbins = np.array([len(self.pt_bins), len(self.eta_bins)])
        self.pT_var_name = "pt_btagJes"
        self.eta_var_name = "absEta_btagJes"
        self.rnd_seed = 42

    def GetIndices(self):
        """
        Applies the UnderSampling to the given arrays.
        Returns the indices for the jets to be used separately for b, c and
        light jets (as well as taus, optionally).
        """
        binnumbers_b, ind_b, stat_b = self.GetBins(self.bjets)
        binnumbers_c, _, stat_c = self.GetBins(self.cjets)
        binnumbers_u, _, stat_u = self.GetBins(self.ujets)
        if self.bool_tjets:
            binnumbers_t, _, stat_t = self.GetBins(self.tjets)
            min_count_per_bin = np.amin(
                [stat_b, stat_c, stat_u, stat_t], axis=0
            )
        else:
            min_count_per_bin = np.amin([stat_b, stat_c, stat_u], axis=0)

        bjet_indices = []
        cjet_indices = []
        ujet_indices = []
        tjet_indices = []

        for elem, count in zip(ind_b, min_count_per_bin):
            np.random.seed(self.rnd_seed)
            bjet_indices.append(
                np.random.choice(
                    np.where(binnumbers_b == elem)[0],
                    int(count),
                    replace=False,
                )
            )
            np.random.seed(self.rnd_seed)
            cjet_indices.append(
                np.random.choice(
                    np.where(binnumbers_c == elem)[0],
                    int(count),
                    replace=False,
                )
            )
            np.random.seed(self.rnd_seed)
            ujet_indices.append(
                np.random.choice(
                    np.where(binnumbers_u == elem)[0],
                    int(count),
                    replace=False,
                )
            )
            if self.bool_tjets:
                np.random.seed(self.rnd_seed)
                tjet_indices.append(
                    np.random.choice(
                        np.where(binnumbers_t == elem)[0],
                        int(count),
                        replace=False,
                    )
                )
        if self.bool_tjets:
            sorted_tjets = np.sort(np.concatenate(tjet_indices))
        else:
            sorted_tjets = None

        return (
            np.sort(np.concatenate(bjet_indices)),
            np.sort(np.concatenate(cjet_indices)),
            np.sort(np.concatenate(ujet_indices)),
            sorted_tjets,
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

        return binnumber, bins_indices_flat, statistic.flatten()


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
        self.pT_var_name = "pt_btagJes"
        self.eta_var_name = "absEta_btagJes"
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


class Weighting2D(object):
    """Alternatively to the UnderSampling approach, the 2D weighting can be
    used to prepare the training dataset. It makes sure
    that in each pT/eta bin each category has the same weight.
    This is especially suited if not enough statistics is available.
    """

    def __init__(self, bjets, cjets, ujets):
        super(Weighting2D, self).__init__()
        self.bjets = bjets
        self.cjets = cjets
        self.ujets = ujets
        self.pt_bins = np.concatenate(
            (np.linspace(0, 600000, 351), np.linspace(650000, 6000000, 84))
        )
        self.eta_bins = np.linspace(0, 2.5, 10)
        self.pt_bins = np.linspace(0, 6000000, 3)
        self.eta_bins = np.linspace(0, 2.5, 3)
        self.nbins = np.array([len(self.pt_bins), len(self.eta_bins)])
        self.pT_var_name = "pt_btagJes"
        self.eta_var_name = "absEta_btagJes"
        self.rnd_seed = 42

    def GetWeights(self):
        """"Retrieves the weights for the sample."""
        binnumbers_b, ind_b, stat_b = self.GetBins(self.bjets)
        binnumbers_c, _, stat_c = self.GetBins(self.cjets)
        binnumbers_u, _, stat_u = self.GetBins(self.ujets)

        # Using the b-jet distribution as reference
        # print(self.bjets)
        print(stat_b)
        # bin_weights_u = np.divide(stat_u, stat_b)
        bin_weights_c = np.divide(stat_c, stat_b)
        # bin_weights_b = np.ones(len(stat_b))
        print(bin_weights_c)
        # for elem, count in zip(ind_b, min_count_per_bin):
        #     np.random.seed(self.rnd_seed)
        #     bjet_indices.append(np.random.choice(np.where(
        #  binnumbers_b == elem)
        #                         [0], int(count), replace=False))
        #     np.random.seed(self.rnd_seed)
        #     cjet_indices.append(np.random.choice(np.where(
        # binnumbers_c == elem)
        #                         [0], int(count), replace=False))
        #     np.random.seed(self.rnd_seed)
        #     ujet_indices.append(np.random.choice(np.where(
        # binnumbers_u == elem)
        #                         [0], int(count), replace=False))

        # return np.sort(np.concatenate(bjet_indices)),\
        #     np.sort(np.concatenate(cjet_indices)),\
        #     np.sort(np.concatenate(ujet_indices))

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

        return binnumber, bins_indices_flat, statistic.flatten()


def GetNJetsPerIteration(config, total_number_of_taus=0):
    take_taus = config.bool_process_taus
    if config.iterations == 0:
        raise ValueError("The iterations have to be >=1 and not 0.")
    if config.ttbar_frac > 0.0:
        nZ = (
            int(config.njets) * 3 * (1 / config.ttbar_frac - 1)
        ) // config.iterations
        ncjets = int(2.3 * config.njets) // config.iterations
        nujets = int(2.7 * config.njets) // config.iterations
        njets = int(config.njets) // config.iterations
        if take_taus:
            # Equal number of taus per iteration
            ntaujets = int(total_number_of_taus) // config.iterations
    else:
        nZ = int(config.njets) // config.iterations
        ncjets = nujets = njets = 0
        if take_taus:
            ntaujets = 0

    N_list = []
    for x in range(config.iterations + 1):
        N_dict = {
            "nZ": int(nZ * x),
            "nbjets": int(njets * x),
            "ncjets": int(ncjets * x),
            "nujets": int(nujets * x),
        }
        if take_taus:
            N_dict["ntaujets"] = int(ntaujets * x)
        N_list.append(N_dict)
    return N_list


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


def EnforceFraction(
    sample, ttbar_frac, statistics_dict, label, tolerance=0.01
):
    """
    If the ttbar fraction obtained is off from the one expected (= ttbar_frac) by more
    than tolerance, further downsamples to reach the expected fraction.

    Requires a statistics_dict like the one produced RunStatSamples.
    The key in the dict corresponding to the sample must be stored in label.
    """
    down_sample = False
    n_selected = None
    np.random.seed(42)
    (ttbar_frac_achieved, ntt, nZ) = statistics_dict[label]
    if ttbar_frac_achieved < (ttbar_frac - tolerance):  # Too much Z'
        nZ_required = int(ntt / ttbar_frac - ntt)
        if nZ_required > nZ:
            print(
                "Error, requiring {} Z while only {} available".format(
                    nZ_required, nZ
                )
            )
        n_selected = np.random.choice(
            np.where(sample["category"] == 0)[0],
            nZ_required,
            replace=False,
        )
        n_selected = np.concatenate(
            [n_selected, np.where(sample["category"] == 1)[0]]
        )
        down_sample = True
    elif ttbar_frac_achieved > (ttbar_frac + tolerance):  # Too much ttbar
        ntt_required = int(ttbar_frac / (1 - ttbar_frac) * nZ)
        if ntt_required > ntt:
            print(
                "Error, requiring {} tt while only {} available".format(
                    ntt_required, ntt
                )
            )
        n_selected = np.random.choice(
            np.where(sample["category"] == 1)[0],
            ntt_required,
            replace=False,
        )
        n_selected = np.concatenate(
            [n_selected, np.where(sample["category"] == 0)[0]]
        )
        down_sample = True
    if down_sample:
        sample = sample[n_selected]
        nX_tt = len(sample[sample["category"] == 1])
        nXjets = len(sample)
        ttfrac_X = float(nX_tt) / nXjets
        print(
            "Further downsampled! {} {} jets: {} ttbar (frac: {}) | {} Z'-ext (frac: {})".format(
                nXjets,
                label,
                nX_tt,
                round(ttfrac_X, 2),
                nXjets - nX_tt,
                round(1 - ttfrac_X, 2),
            )
        )
    return sample, n_selected


def RunStatSamples(bjets, cjets, ujets, taujets=None):
    """
    Looks at the content of the samples and computes ttbar fraction
    Returns a dictionary with labels for keys ("b", "c", ...) and value
    being tuple (ttbar fraction, number of ttbar, number of Z').
    """
    nb_tt = len(bjets[bjets["category"] == 1])
    nc_tt = len(cjets[cjets["category"] == 1])
    nu_tt = len(ujets[ujets["category"] == 1])
    nbjets = len(bjets)
    ncjets = len(cjets)
    nujets = len(ujets)
    if taujets is not None:
        ntau_tt = len(taujets[taujets["category"] == 1])
        ntaujets = len(taujets)
        ttfrac = float(nb_tt + nc_tt + nu_tt + ntau_tt) / float(
            nbjets + ncjets + nujets + ntaujets
        )
    else:
        ttfrac = float(nb_tt + nc_tt + nu_tt) / float(nbjets + ncjets + nujets)
    print("ttbar fraction:", round(ttfrac, 2))
    ttfrac_b = float(nb_tt) / nbjets
    ttfrac_c = float(nc_tt) / ncjets
    ttfrac_u = float(nu_tt) / nujets
    if taujets is not None:
        ttfrac_tau = float(ntau_tt) / ntaujets
    print(
        "{} b jets: {} ttbar (frac: {}) | {} Z'-ext (frac: {})".format(
            nbjets,
            nb_tt,
            round(ttfrac_b, 2),
            nbjets - nb_tt,
            round(1 - ttfrac_b, 2),
        )
    )
    print(
        "{} c jets: {} ttbar (frac: {}) | {} Z'-ext (frac: {})".format(
            ncjets,
            nc_tt,
            round(ttfrac_c, 2),
            ncjets - nc_tt,
            round(1 - ttfrac_c, 2),
        )
    )
    print(
        "{} u jets: {} ttbar (frac: {}) | {} Z'-ext (frac: {})".format(
            nujets,
            nu_tt,
            round(ttfrac_u, 2),
            nujets - nu_tt,
            round(1 - ttfrac_u, 2),
        )
    )
    if taujets is not None:
        print(
            "{} tau jets: {} ttbar (frac: {}) | {} Z'-ext (frac: {})".format(
                ntaujets,
                ntau_tt,
                round(ttfrac_tau, 2),
                ntaujets - ntau_tt,
                round(1 - ttfrac_tau, 2),
            )
        )
    else:
        ttfrac_tau, ntau_tt, ntaujets = 0, 0, 0
    stat_dict = {
        "b": (ttfrac_b, nb_tt, nbjets - nb_tt),
        "c": (ttfrac_c, nc_tt, ncjets - nc_tt),
        "u": (ttfrac_u, nu_tt, nujets - nu_tt),
        "tau": (ttfrac_tau, ntau_tt, ntaujets - ntau_tt),
    }
    return stat_dict


def RunSampling(
    bjets,
    cjets,
    ujets,
    taujets,
    btrk,
    ctrk,
    utrk,
    tautrk,
    sampling_method="count",
    take_taus=False,
    tracks=False,
):
    """
    Runs the undersampling, with the sampling_method chosen.
    """
    if take_taus:
        if sampling_method == "weight":
            downs = UnderSamplingProp(bjets, cjets, ujets, taujets)
            b_indices, c_indices, u_indices, tau_indices = downs.GetIndices()
        elif sampling_method == "count":
            downs = UnderSampling(bjets, cjets, ujets, taujets)
            b_indices, c_indices, u_indices, tau_indices = downs.GetIndices()
        elif sampling_method == "count_bcl_weight_tau":
            downs = UnderSamplingProp(bjets, cjets, ujets, taujets)
            (
                b_indices0,
                c_indices0,
                u_indices0,
                tau_indices,
            ) = downs.GetIndices()
            bjets = bjets[b_indices0]
            cjets = cjets[c_indices0]
            ujets = ujets[u_indices0]
            downs = UnderSampling(bjets, cjets, ujets)
            b_indices, c_indices, u_indices, _ = downs.GetIndices()
        taujets = taujets[tau_indices]
    else:
        if sampling_method == "weight":
            downs = UnderSamplingProp(bjets, cjets, ujets)
        elif sampling_method == "count":
            downs = UnderSampling(bjets, cjets, ujets)
        b_indices, c_indices, u_indices, _ = downs.GetIndices()
        taujets = None

    bjets = bjets[b_indices]
    cjets = cjets[c_indices]
    ujets = ujets[u_indices]

    if tracks:
        if sampling_method == "count_bcl_weight_tau":
            # A prior step for the tracks for b/c/l jets
            btrk = btrk[b_indices0]
            ctrk = ctrk[c_indices0]
            utrk = utrk[u_indices0]
        btrk = btrk[b_indices]
        ctrk = ctrk[c_indices]
        utrk = utrk[u_indices]
        if take_taus:
            tautrk = tautrk[tau_indices]
    return bjets, cjets, ujets, taujets, btrk, ctrk, utrk, tautrk, downs
