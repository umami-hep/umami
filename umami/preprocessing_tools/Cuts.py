import operator

import numpy as np

from umami.configuration import global_config


def GetCuts(jets, config, sample="ttbar"):
    # define operator dict to be able to call them via string from config
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        ">=": operator.ge,
        ">": operator.gt,
    }
    indices_to_remove = []
    # General cuts as defined in config (to remove outliers)
    if config.Cuts is not None:
        for elem in config.Cuts:
            op_func = ops[config.Cuts[elem]["operator"]]
            if config.Cuts[elem]["NaNcheck"] is True:
                indices_i_to_remove = np.where(
                    np.logical_not(
                        op_func(jets[elem], config.Cuts[elem]["condition"])
                    )
                    & (jets[elem] == jets[elem])
                )[0]
            else:
                indices_i_to_remove = np.where(
                    np.logical_not(
                        op_func(jets[elem], config.Cuts[elem]["condition"])
                    )
                )[0]

            indices_to_remove.append(indices_i_to_remove)

    if config.pT_max is not False:
        indices_to_remove.append(
            np.where(jets[global_config.pTvariable] > config.pT_max)[0]
        )
    if sample == "ttbar":
        if config.bhad_pTcut is not None:
            indices_to_remove_bjets = np.where(
                (jets["HadronConeExclTruthLabelID"] == 5)
                & (jets["GhostBHadronsFinalPt"] > config.bhad_pTcut)
            )[0]
            indices_to_remove.append(indices_to_remove_bjets)

        if config.pTcut is not None:
            indices_to_remove_xjets = np.where(
                (jets["HadronConeExclTruthLabelID"] != 5)
                & (jets[global_config.pTvariable] > config.pTcut)
            )[0]
            indices_to_remove.append(indices_to_remove_xjets)

        return np.unique(np.concatenate(indices_to_remove))

    elif sample == "Zprime":
        if config.bhad_pTcut is not None:
            indices_to_remove_bjets = np.where(
                (jets["HadronConeExclTruthLabelID"] == 5)
                & (jets["GhostBHadronsFinalPt"] < config.bhad_pTcut)
            )[0]
            indices_to_remove.append(indices_to_remove_bjets)

        if config.pTcut is not None:
            indices_to_remove_xjets = np.where(
                (jets["HadronConeExclTruthLabelID"] != 5)
                & (jets[global_config.pTvariable] < config.pTcut)
            )[0]
            indices_to_remove.append(indices_to_remove_xjets)

        return np.unique(np.concatenate(indices_to_remove))

    else:
        print("Chose either 'ttbar' or 'Zprime' as argument for sample")
        return 1
