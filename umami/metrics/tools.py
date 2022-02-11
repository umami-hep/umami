"""Tools for metrics module."""
import numpy as np


def eff_err(
    x: np.ndarray,
    N: int,
    suppress_zero_divison_error: bool = False,
) -> np.ndarray:
    """Calculate statistical efficiency uncertainty.

    Parameters
    ----------
    x : numpy.array
        efficiency values
    N : int
        number of used statistics to calculate efficiency
    suppress_zero_divison_error : bool
        not raising Error for zero division

    Returns
    -------
    numpy.array
        efficiency uncertainties

    Raises
    ------
    ValueError
        if N <=0

    Notes
    -----
    This method uses binomial errors as described in section 2.2 of
    https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
    """
    # TODO: suppress_zero_divison_error should not be necessary, but functions calling
    # eff_err seem to need this functionality - should be deprecated though.
    if np.any(N <= 0) and not suppress_zero_divison_error:
        raise ValueError(f"You passed as argument `N` {N} but it has to be larger 0.")
    return np.sqrt(x * (1 - x) / N)


def rej_err(
    x: np.ndarray,
    N: int,
) -> np.ndarray:
    """Calculate the rejection uncertainties.

    Parameters
    ----------
    x : numpy.array
        rejection values
    N : int
        number of used statistics to calculate rejection

    Returns
    -------
    numpy.array
        rejection uncertainties

    Raises
    ------
    ValueError
        if N <=0
    ValueError
        if any rejection value is 0

    Notes
    -----
    special case of `eff_err()`
    """
    if np.any(N <= 0):
        raise ValueError(f"You passed as argument `N` {N} but it has to be larger 0.")
    if np.any(x == 0):
        raise ValueError("One rejection value is 0, cannot calculate error.")
    return np.power(x, 2) * eff_err(1 / x, N)
