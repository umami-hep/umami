"""Tools for metrics module."""
import numpy as np

from umami.configuration import logger


def eff_err(
    x: np.ndarray,
    N: int,
    suppress_zero_divison_error: bool = False,
    norm: bool = False,
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
    norm : bool, optional
        if True, normed (relative) error is being calculated, by default False

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
    logger.debug("Calculating efficiency error.")
    logger.debug(f"x: {x}")
    logger.debug(f"N: {N}")
    logger.debug(f"suppress_zero_divison_error: {suppress_zero_divison_error}")
    logger.debug(f"norm: {norm}")
    # TODO: suppress_zero_divison_error should not be necessary, but functions calling
    # eff_err seem to need this functionality - should be deprecated though.
    if np.any(N <= 0) and not suppress_zero_divison_error:
        raise ValueError(f"You passed as argument `N` {N} but it has to be larger 0.")
    if norm:
        return np.sqrt(x * (1 - x) / N) / x
    return np.sqrt(x * (1 - x) / N)


def rej_err(
    x: np.ndarray,
    N: int,
    norm: bool = False,
) -> np.ndarray:
    """Calculate the rejection uncertainties.

    Parameters
    ----------
    x : numpy.array
        rejection values
    N : int
        number of used statistics to calculate rejection
    norm : bool, optional
        if True, normed (relative) error is being calculated, by default False

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
    logger.debug("Calculating rejection error.")
    logger.debug(f"x: {x}")
    logger.debug(f"N: {N}")
    logger.debug(f"norm: {norm}")
    if np.any(N <= 0):
        raise ValueError(f"You passed as argument `N` {N} but it has to be larger 0.")
    if np.any(x == 0):
        raise ValueError("One rejection value is 0, cannot calculate error.")
    if norm:
        return np.power(x, 2) * eff_err(1 / x, N) / x
    return np.power(x, 2) * eff_err(1 / x, N)
