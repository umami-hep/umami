import numpy as np


def hist_w_unc(a, bins, normed: bool = True):
    """
    Computes histogram and the associated statistical uncertainty.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins: int or sequence of scalars or str
        bins parameter from np.histogram
    normed: bool
        If True (default) the calculated histogram is normalised to an integral
        of 1.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges (length(hist)+1)
    hist : numpy array
        The values of the histogram. If normed is true (default), returns the
        normed counts per bin
    unc : numpy array
        Statistical uncertainty per bin.
        If normed is true (default), returns the normed values.
    band : numpy array
        lower uncertainty band location: hist - unc
        If normed is true (default), returns the normed values.
    """
    arr_length = len(a)

    # Calculate the counts and the bin edges
    counts, bin_edges = np.histogram(a, bins=bins)

    unc = np.sqrt(counts) / arr_length if normed else np.sqrt(counts)
    band = counts / arr_length - unc if normed else counts - unc
    hist = counts / arr_length if normed else counts

    return bin_edges, hist, unc, band


def step_divide(nominator, denominator, default=1.0):
    """
    Division using numpy divide function returning default value in cases where
    denoinator is 0.

    Parameters
    ----------
    nominator: array_like
        Nominator in the ratio calculation.
    denominator: array_like
        Denominator in the ratio calculation.
    default: float
        default value which is returned if denominator is 0.

    Returns
    -------
    ratio: array_like
    """
    ratio = np.divide(
        nominator,
        denominator,
        out=np.ones(
            nominator.shape,
            dtype=float,
        )
        * default,
        where=(denominator != 0),
    )
    return ratio


def hist_ratio(nominator, denominator, nominator_unc, denominator_unc):
    """
    This method calculates the ratio of the given bincounts and
    returns the input for a step function that plots the ratio.

    Parameters
    ----------
    nominator : array_like
        Nominator in the ratio calculation.
    denominator : array_like
        Denominator in the ratio calculation.
    nominator_unc : array_like
        Uncertainty of the nominator.
    denominator_unc : array_like
        Uncertainty of the denominator.


    Returns
    -------
    step_ratio : array_like
        Ratio returning 1 in case the denominator is 0.
    step_ratio_unc : array_like
        Stat. uncertainty of the step_ratio

    Raises
    ------
    AssertionError
        If inputs don't have the same shape.

    """
    if nominator.shape != denominator.shape:
        raise (AssertionError("Nominator and denominator don't have the same legth"))
    if nominator.shape != nominator_unc.shape:
        raise (AssertionError("Nominator and nominator_unc don't have the same legth"))
    if denominator.shape != denominator_unc.shape:
        raise (
            AssertionError("Denominator and denominator_unc don't have the same legth")
        )
    step_ratio = step_divide(nominator, denominator)
    # Add an extra bin in the beginning to have the same binning as the input
    # Otherwise, the ratio will not be exactly above each other (due to step)
    step_ratio = np.append(np.array([step_ratio[0]]), step_ratio)

    # Calculate rel uncertainties
    nominator_rel_unc = step_divide(nominator_unc, nominator, default=0)
    denominator_rel_unc = step_divide(denominator_unc, denominator, default=0)

    # Calculate rel uncertainty
    step_rel_unc = np.sqrt(nominator_rel_unc ** 2 + denominator_rel_unc ** 2)

    # Add the first value again (same reason as for the step calculation)
    step_rel_unc = np.append(np.array([step_rel_unc[0]]), step_rel_unc)

    # Calculate final uncertainty
    step_unc = step_ratio * step_rel_unc

    return step_ratio, step_unc
