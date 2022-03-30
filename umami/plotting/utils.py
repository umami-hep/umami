"""Helper functions for the plotting API"""
from umami.configuration import logger  # isort:skip
import numpy as np
import pandas as pd
from scipy.special import softmax


def translate_kwargs(kwargs):
    """Maintaining backwards compatibility for the kwargs and the new plot_base syntax.

    Parameters
    ----------
    kwargs : dict
        dictionary with kwargs

    Returns
    -------
    dict
        kwargs compatible with new naming.
    """
    mapping = {
        "ApplyATLASStyle": "apply_atlas_style",
        "AtlasTag": "atlas_first_tag",
        "Bin_Width_y_axis": "bin_width_in_ylabel",
        "labelFontSize": "fontsize",
        "legcols": "leg_ncol",
        "legFontSize": "leg_fontsize",
        "loc_legend": "leg_loc",
        "Log": "logy",
        "n_Leading": "n_leading",
        "ncol": "leg_ncol",
        "nJets": "n_jets",
        "normalise": "norm",
        "Ratio_Cut": "ratio_cut",
        "SecondTag": "atlas_second_tag",
        "set_logy": "logy",
        "UseAtlasTag": "use_atlas_tag",
        "yAxisIncrease": "y_scale",
    }
    deprecated_args = ["yAxisAtlasTag"]
    for key, elem in mapping.items():
        if key in kwargs:
            logger.debug(f"Mapping from old naming: {elem}: {kwargs[key]}")
            if elem in kwargs:
                logger.warning(
                    "You specified two keyword arguments which mean the same: "
                    f"{key}, {elem} --> using the new naming convention {elem}"
                )
            else:
                kwargs[elem] = kwargs[key]
            kwargs.pop(key)

    # Remove deprecated arguments from kwargs
    for dep_key in deprecated_args:
        if dep_key in kwargs:
            logger.warning(
                f"You specified the argument {dep_key}, which is no longer"
                " supported and will be ignored."
            )
            kwargs.pop(dep_key)
    return kwargs


def set_xaxis_ticklabels_invisible(ax):
    """Helper function to set the ticklabels of the xaxis invisible

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis you want to modify
    """

    for label in ax.get_xticklabels():
        label.set_visible(False)


def get_dummy_multiclass_scores(
    size: int = 10_000, bjets_mean: float = 1.4, seed: int = 42
):
    """
    Generate dummy data representing output of 3 class classifier.
    Adapted to light-, c- and b-jets and values of `HadronConeExclTruthLabelID`.

    Parameters
    ----------
    size : int, optional
        size of dummy data. For each of the 3 classes, the same amount of is produces,
        by default 10_000
    bjets_mean : float, optional
        mean value of the b-jets 3D gaussian, the more away from 0, the better the
        b-tagging performance, by default 1.4
    seed : int, optional
        random seed for number generation, by default 42

    Returns
    -------
    np.ndarray
        output scores of the shape (size, 3)
    np.ndarray
        labels of shape (size,). The order of the output is light-jets, c-jets, b-jets

    """
    size_class = int(size / 3)
    rng = np.random.default_rng(seed=seed)
    ujets = softmax(rng.normal(loc=[-1, 0, 0], scale=1, size=(size_class, 3)), axis=1)
    cjets = softmax(rng.normal(loc=[0, 1, 0], scale=2.5, size=(size_class, 3)), axis=1)
    bjets = softmax(
        rng.normal(loc=[0, 0, bjets_mean], scale=2, size=(size_class, 3)), axis=1
    )
    output = np.concatenate((ujets, cjets, bjets))
    labels = np.concatenate(
        (np.zeros(size_class), np.ones(size_class) * 4, np.ones(size_class) * 5)
    )
    return output, labels


def get_dummy_2_taggers(
    size: int = 10_000, shuffle: bool = True, seed: int = 42, add_pt: bool = False
):
    """
    Wrapper function of `get_dummy_multiclass_scores` to generate classifier output
    for 2 taggers, in this case rnnip and dips as well as HadronConeExclTruthLabelID.


    Parameters
    ----------
    size : int, optional
        size of dummy data, by default 10_000
    shuffle : bool, optional
        if True shuffles the dummy data, by default True
    seed : int, optional
        random seed for number generation (will count +10 for second tagger),
        by default 42
    add_pt : bool, optional
        specify if pt column should be added as well, by default False

    Returns
    -------
    pd.DataFrame
        pandas DataFrame with columns
        [HadronConeExclTruthLabelID, rnnip_pu, rnnip_pc, rnnip_pb, dips_pu, dips_pc,
        dips_pb]
    """
    output_rnnip, labels = get_dummy_multiclass_scores(
        bjets_mean=0.9, size=size, seed=seed
    )
    df = pd.DataFrame(output_rnnip, columns=["rnnip_pu", "rnnip_pc", "rnnip_pb"])
    df["HadronConeExclTruthLabelID"] = labels
    output_dips, _ = get_dummy_multiclass_scores(
        bjets_mean=1.4, size=size, seed=seed + 10
    )
    df2 = pd.DataFrame(output_dips, columns=["dips_pu", "dips_pc", "dips_pb"])
    df = pd.concat([df, df2], axis=1)
    if add_pt:
        rng = np.random.default_rng(seed=seed)
        df["pt"] = rng.exponential(100_000, size=len(df))
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df
