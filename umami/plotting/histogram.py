"""Histogram plot functions."""
import matplotlib as mtp
import numpy as np
import pandas as pd

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.configuration import global_config, logger
from umami.helper_tools import hist_ratio, hist_w_unc
from umami.plotting.plot_base import plot_base, plot_line_object


class histogram(plot_line_object):
    """
    histogram class storing info about histogram and allows to calculate ratio w.r.t
    other histograms.
    """

    def __init__(
        self,
        values: np.ndarray,
        flavour: str = None,
        histtype: str = "step",
        **kwargs,
    ) -> None:
        """Initialise properties of histogram curve object.

        Parameters
        ----------
        values : np.ndarray
            Input data for the histogram
        flavour: str, optional
            Jet flavour in case the histogram corresponds to one specific flavour. If
            this is specified, the correct colour will be extracted from the global
            config. Allowed values are the ones from the global config, i.e. "bjets",
            "cjets", "ujets", "bbjets", ..., by default None
        histtype: str, optional
            `histtype` parameter which is handed to matplotlib.hist() when plotting the
            histograms. Supported values are "bar", "barstacked", "step", "stepfilled".
            By default "step"
        **kwargs : kwargs
            kwargs passed to `plot_line_object`

        Raises
        ------
        ValueError
            If input data is not of type np.ndarray or list
        """
        super().__init__(**kwargs)

        if isinstance(values, (np.ndarray, list, pd.core.series.Series)):
            values = np.array(values)
            if len(values) == 0:
                logger.warning("Histogram is empty.")
        else:
            raise ValueError(
                "Invalid type of histogram input data. Allowed values are "
                "numpy.ndarray, list, pandas.core.series.Series"
            )

        self.values = values
        self.flavour = flavour
        self.histtype = histtype

        # Set histogram attributes to None. They will be defined when the histograms
        # are plotted
        self.bin_edges = None
        self.hist = None
        self.unc = None
        self.band = None
        self.key = None

        self.label_addition = kwargs["label"] if "label" in kwargs else ""
        # If flavour was specified, extract configuration from global config
        if self.flavour is not None:
            self.colour = global_config.flavour_categories[self.flavour]["colour"]
            logger.debug(f"Histogram colour was set to {self.colour}")

            self.label = (
                f"{global_config.flavour_categories[self.flavour]['legend_label']}"
                f" {self.label_addition}"
            )
            logger.debug(f"Histogram label was set to {self.label}")

    def divide(self, other):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : histogram class
            Second histogram object to calculate ratio with

        Returns
        -------
        np.ndarray
            ratio
        np.ndarray
            ratio error
        Raises
        ------
        ValueError
            If binning is not identical between 2 objects
        ValueError
            If hist attribute is not set for one of the two histograms
        ValueError
            If bin_edges attribute is not set for one of the two histograms
        """
        if self.bin_edges is None or other.bin_edges is None:
            raise ValueError(
                "Can't divide histograms since bin edges are not available "
                "for both histogram. Bins are filled when they are plotted."
            )

        if self.hist is None or other.hist is None:
            raise ValueError(
                "Can't divide histograms since bin counts are not available for both"
                "histograms. Bins are filled when they are plotted."
            )

        if not np.all(self.bin_edges == other.bin_edges):
            raise ValueError("The binning of the two given objects do not match.")

        # Bins where the reference histogram is empty/zero, are given a ratio of np.inf
        # which means that the ratio plot will not have any entry in these bins.
        ratio, ratio_unc = hist_ratio(
            numerator=self.hist,
            denominator=other.hist,
            numerator_unc=self.unc,
            denominator_unc=other.unc,
            step=False,
        )
        # To use the matplotlib.step() function later on, the first bin is duplicated
        ratio = np.append(np.array([ratio[0]]), ratio)
        ratio_unc = np.append(np.array([ratio_unc[0]]), ratio_unc)

        return (ratio, ratio_unc)


class histogram_plot(plot_base):
    """Histogram plot class"""

    def __init__(
        self,
        binning: np.ndarray = None,
        norm: bool = True,
        logy: bool = False,
        **kwargs,
    ) -> None:
        """histogram plot properties

        Parameters
        ----------
        binning : np.ndarray, optional
            Array that represents the bin edges of the histograms. If not provided, the
            bin edges will be calculated from the histogram stack of all histograms in
            the plot, ensuring that all data is shown in the plot. By default None.
        norm : bool, optional
            Specify if the histograms are normalised, this means that histograms are
            divided by the total numer of counts. Therefore, the sum of the bin counts
            is equal to one, but NOT the area under the curve, which would be
            sum(bin_counts * bin_width). By default True.
        logy : bool, optional
            Set log scale on y-axis, by default False.
        **kwargs : kwargs
            kwargs from `plot_base`

        Raises
        ------
        ValueError
            If n_ratio_panels > 1
        """

        super().__init__(**kwargs)
        self.logy = logy
        self.binning = binning
        self.norm = norm
        self.plot_objects = {}
        self.add_order = []
        self.ratios_objects = {}
        self.ratio_axes = {}
        self.reference_object = None
        if self.n_ratio_panels > 1:
            raise ValueError("Not more than one ratio panel supported.")
        self.initialise_figure(sub_plot_index=6)
        self.plotting_done = False

    def add(self, curve: object, key: str = None, reference: bool = False):
        """Adding histogram object to figure.

        Parameters
        ----------
        curve : histogram class
            histogram curve
        key : str, optional
            Unique identifier for histogram, by default None
        reference : bool, optional
            If this histogram is used as reference for ratio calculation, by default
            False

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        if key is None:
            key = len(self.plot_objects) + 1
        if key in self.plot_objects:
            raise KeyError(f"Duplicated key {key} already used for unique identifier.")

        # Add key to histogram object
        curve.key = key
        logger.info(f"Adding histogram {key}")

        # Set linestyle
        if curve.linestyle is None:
            curve.linestyle = "-"
        # Set colours
        if curve.colour is None:
            curve.colour = pas.get_good_colours()[len(self.plot_objects)]
        # Set alpha
        if curve.alpha is None:
            curve.alpha = 0.8
        # Set linewidth
        if curve.linewidth is None:
            curve.linewidth = 1.6

        self.plot_objects[key] = curve
        self.add_order.append(key)
        if reference is True:
            self.set_reference(key)

    def set_reference(self, key: str):
        """Setting the reference histogram curves used in the ratios

        Parameters
        ----------
        key : str
            unique identifier of histogram object
        """
        if self.reference_object is None:
            self.reference_object = [key]
            logger.info(f"Using '{key}' as reference histogram")
        else:
            self.reference_object.append(key)
            logger.warning(
                f"You specified another curve {key} as reference for ratio. "
                "Adding it to reference histograms. "
                f"New list of reference histograms: {self.reference_object}"
            )

    def plot(self, **kwargs):
        """Plotting curves. This also generates the bins of the histograms that are
        added to the plot. Plot objects are drawn in the same order as they were added
        to the plot.

        Parameters
        ----------
        **kwargs: kwargs
            kwargs passed to matplotlib.axes.Axes.hist()

         Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []

        # Calculate binning of stacked histograms to ensure all histograms fit
        # in plot
        if self.binning is None:
            logger.info("No binning set --> calculating binning of stacked histograms.")
            _, self.binning = np.histogram(
                np.hstack(elem.values for elem in self.plot_objects.values()), bins=40
            )

        # Loop over all plot objects and plot them
        for key in self.add_order:
            elem = self.plot_objects[key]

            elem.bin_edges, elem.hist, elem.unc, elem.band = hist_w_unc(
                elem.values,
                bins=self.binning,
                normed=self.norm,
            )

            # Plot histogram
            self.axis_top.hist(
                x=elem.bin_edges[:-1],
                bins=elem.bin_edges,
                weights=elem.hist,
                histtype=elem.histtype,
                color=elem.colour,
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
                **kwargs,
            )

            # Plot histogram uncertainty
            self.axis_top.hist(
                x=elem.bin_edges[:-1],
                bins=self.binning,
                bottom=elem.band,
                weights=elem.unc * 2,
                **global_config.hist_err_style,
            )

            plt_handles.append(
                mtp.lines.Line2D(
                    [],
                    [],
                    color=elem.colour,
                    label=elem.label,
                    linestyle=elem.linestyle,
                )
            )

        plt_handles.append(
            mtp.patches.Patch(label="stat. uncertainty", **global_config.hist_err_style)
        )
        self.plotting_done = True
        return plt_handles

    def get_reference_histo(self, histo):
        """Get reference histogram from list of references

        Parameters
        ----------
        histo : histogram class
            Histogram we want to calculate the ratio for

        Returns
        -------
        histogram class
            Reference histogram

        Raises
        ------
        ValueError
            If no reference histo was found or multiple matches.
        """

        matches = 0
        reference_histo = None

        for key in self.reference_object:
            reference_candidate = self.plot_objects[key]
            if histo.flavour is not None:
                if histo.flavour == reference_candidate.flavour:
                    matches += 1
                    reference_histo = reference_candidate
            else:
                matches += 1
                reference_histo = reference_candidate

        if matches != 1:
            raise ValueError("Found more than one matching reference candidate.")

        logger.debug(
            f"Reference histogram for '{histo.key}' is '{reference_histo.key}'"
        )

        return reference_histo

    def plot_ratios(self):
        """Plotting ratio histograms.

        Raises
        ------
        ValueError
            If no reference histogram is defined
        """
        if self.reference_object is None:
            raise ValueError("Please specify a reference curve.")

        for key in self.add_order:
            elem = self.plot_objects[key]

            if elem.bin_edges is None:
                raise ValueError(
                    "Bin edges of plot object not set. This is done in "
                    "histogram_plot.plot(), so it has to be called before "
                    "plot_ratios() is called."
                )

            (ratio, ratio_err) = elem.divide(self.get_reference_histo(elem))

            # Plot the ratio values with the step function
            self.axis_ratio_1.step(
                x=elem.bin_edges,
                y=ratio,
                color=elem.colour,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
            )

            # Plot the ratio uncertainty
            self.axis_ratio_1.fill_between(
                x=elem.bin_edges,
                y1=ratio - ratio_err,
                y2=ratio + ratio_err,
                step="pre",
                facecolor="none",
                edgecolor=global_config.hist_err_style["edgecolor"],
                linewidth=global_config.hist_err_style["linewidth"],
                hatch=global_config.hist_err_style["hatch"],
            )

    def draw(self, rlabel: str = "Ratio", labelpad: int = None):
        """Draw figure.

        Parameters
        ----------
        rlabel : str, optional
            label of ratio panel
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default "ratio"
        """
        plt_handles = self.plot()
        if self.n_ratio_panels > 0:
            self.plot_ratios()
        self.set_xlim(
            self.binning[0] if self.xmin is None else self.xmin,
            self.binning[-1] if self.xmax is None else self.xmax,
        )
        self.make_legend(plt_handles)

        self.set_title()
        self.set_logy()
        self.set_y_lim()
        self.set_xlabel()
        self.set_tick_params()
        self.set_ylabel(self.axis_top)

        if self.n_ratio_panels > 0:
            self.set_ylabel(
                self.axis_ratio_1,
                rlabel,
                align_right=False,
                labelpad=labelpad,
            )
        self.fig.tight_layout()

        if self.apply_atlas_style:
            self.atlasify()
