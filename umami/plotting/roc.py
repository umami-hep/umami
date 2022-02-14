"""ROC curve functions."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import pchip

from umami.configuration import logger
from umami.metrics import rej_err


@dataclass
class plot_line_object:
    """Base data class defining properties of a plot object.

    Parameters
    ----------
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    colour : str, optional
        colour of the object, by default None
    label : str, optional
        label of object, by default None
    linestyle : str, optional
        linestyle following numpy style, by default None

    """

    xmin: float = None
    xmax: float = None
    colour: str = None
    label: str = None
    linestyle: str = None


class roc(plot_line_object):
    """
    ROC class storing info about curve and allows to calculate ratio w.r.t other roc.
    """

    def __init__(
        self,
        sig_eff: np.ndarray,
        bkg_rej: np.ndarray,
        n_test: int = None,
        rej_class: str = None,
        signal_class: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        sig_eff : np.array
            array of signal efficiencies
        bkg_rej : np.array
            array of background rejection
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None
        signal_class : str
            Signal class, e.g. for b-tagging "bjets", by default None
        rej_class : str
            Rejection class, e.g. for b-tagging anc charm rejection "cjets",
            by default None
        key : str
            identifier for roc curve e.g. tagger, by default None
        **kwargs : kwargs
            kwargs passed to `plot_object`

        Raises
        ------
        ValueError
            if `sig_eff` and `bkg_rej` have a different shape
        """
        super().__init__(**kwargs)
        if len(sig_eff) != len(bkg_rej):
            raise ValueError(
                f"The shape of `sig_eff` ({np.shape(sig_eff)}) and `bkg_rej` "
                f"({np.shape(bkg_rej)}) have to be identical."
            )
        self.sig_eff = sig_eff
        self.bkg_rej = bkg_rej
        self.n_test = None if n_test is None else int(n_test)
        self.signal_class = signal_class
        self.rej_class = rej_class
        self.key = key

    def binomial_error(self, norm: bool = False, n_test: int = None) -> np.ndarray:
        """Calculate binomial error of roc curve.

        Parameters
        ----------
        norm : bool
            if True calulate relative error, by default False
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None

        Returns
        -------
        numpy.array
            binomial error

        Raises
        ------
        ValueError
            if no `n_test` was provided
        """
        if n_test is None:
            n_test = self.n_test
        if n_test is None:
            raise ValueError("No `n_test` provided, cannot calculate binomial error!")
        binom_err = rej_err(self.bkg_rej[self.non_zero_mask], n_test)
        if norm:
            return binom_err / self.bkg_rej[self.non_zero_mask]
        return binom_err

    def divide(self, roc_comp, inverse: bool = False):
        """Calculate ratio between the roc curve and another roc.

        Parameters
        ----------
        roc_comp : roc class
            second roc curve to calculate ratio with
        inverse : bool
            if False the ratio is calculated `this_roc / roc_comp`,
            if True the inverse is calculated

        Returns
        -------
        np.array
            signal efficiency used for the ratio calculation which is the overlapping
            interval of the two roc curves
        np.array
            ratio
        np.array or None
            ratio_err if `n_test` was provided to class
        """
        # if same objects return array with value 1
        if np.array_equal(
            np.array([self.sig_eff, self.bkg_rej]),
            np.array([roc_comp.sig_eff, roc_comp.bkg_rej]),
        ):
            logger.debug("roc objects are identical -> ratio is 1.")
            ratio = np.ones(len(self.sig_eff))
            if self.n_test is None:
                return self.sig_eff, ratio, None
            ratio_err = self.binomial_error(norm=True)
            return self.sig_eff, ratio, ratio_err

        # get overlapping sig_eff interval of the two roc curves
        min_eff = max(self.sig_eff.min(), roc_comp.sig_eff.min())
        max_eff = min(self.sig_eff.max(), roc_comp.sig_eff.max())
        eff_mask = np.all([self.sig_eff >= min_eff, self.sig_eff <= max_eff], axis=0)
        ratio_sig_eff = self.sig_eff[eff_mask]

        # Ratio of interpolated rejection functions
        ratio = self.fct_inter(ratio_sig_eff) / roc_comp.fct_inter(ratio_sig_eff)
        if inverse:
            ratio = 1 / ratio
        if self.n_test is None:
            return ratio_sig_eff, ratio, None
        ratio_err = self.binomial_error(norm=True)
        return ratio_sig_eff, ratio, ratio_err[eff_mask]

    @property
    def fct_inter(self):
        """
        Interpolate the rejection function for better ratio calculation plotting etc.

        Returns
        -------
        pchip
            interpolation function
        """
        return pchip(self.sig_eff, self.bkg_rej)

    @property
    def non_zero_mask(self):
        """Masking points where rejection is 0 and no signal efficiency change present

        Returns
        -------
        numpy.array
            masked indices
        """
        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(self.sig_eff)))

        # Also mask the rejections that are 0
        nonzero = (self.bkg_rej != 0) & (dx > 0)
        if self.xmin is not None:
            nonzero = nonzero & (self.sig_eff >= self.xmin)
        if self.xmax is not None:
            nonzero = nonzero & (self.sig_eff <= self.xmax)
        return nonzero

    @property
    def non_zero(self):
        """Abstraction of `non_zero_mask`

        Returns
        -------
        numpy.array
            masked signal efficiency
        numpy.array
            masked background rejection
        """
        return self.sig_eff[self.non_zero_mask], self.bkg_rej[self.non_zero_mask]
