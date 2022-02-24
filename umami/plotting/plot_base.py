"""Plotting bases for specialised plotting."""
from dataclasses import dataclass

import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib import gridspec

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.configuration import logger
from umami.tools import applyATLASstyle


# TODO: enable `kw_only` when switching to Python 3.10
# @dataclass(kw_only=True)
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
    alpha : float, optional
       Value for visibility of the plot lines, by default None

    """

    xmin: float = None
    xmax: float = None
    colour: str = None
    label: str = None
    linestyle: str = None
    linewidth: str = None
    alpha: float = None


# TODO: enable `kw_only` when switching to Python 3.10
# @dataclass(kw_only=True)
@dataclass
class plot_object:
    """Data base class defining properties of a plot object.

    Parameters
    ----------
    title : str, optional
        Title of the plot, by default ""
    draw_errors : bool, optional
        Binominal errors on the lines, by default True
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    ymin : float, optional
        Minimum value of the y-axis, by default None
    ymax : float, optional
        Maximum value of the y-axis, by default None
    y_scale : float
        Scaling up the y axis, e.g. to fit the ATLAS Tag. Applied if ymax not defined,
        by default 1.3
    logy : bool, optional
        Set log of y-axis of main panel, by default True
    ylabel : int, optional
        label of the y-axis, by default None
    fontsize : int, optional
        used fontsize, by default 12
    n_ratio_panels : int, optional
        amount of ratio panels between 0 and 2, by default 2
    figsize : (float, float)
        tuple of figure size `(width, height)` in inches.
    dpi : int, optional
        dpi used for plotting, by default 400
    leg_fontsize : int, optional
        Fontsize of the legend, by default 10
    leg_loc : str, optional
        Position of the legend in the plot, by default "best"
    leg_ncol : int, optional
        number of legend columns, by default 1
    apply_atlas_style : bool
        Apply ATLAS style for matplotlib, by default True
    use_atlas_tag : bool
        Use the ATLAS Tag in the plots, by default True
    atlas_first_tag : str
        First row of the ATLAS Tag, by default "Internal Simulation"
    atlas_second_tag : str
        Second Row of the ATLAS Tag. No need to add WP or fc. It will
        be added automatically,
        by default,
        "$sqrt{s}=13$ TeV, PFlow Jets, $t bar{t}$ Test Sample, fc=0.018"
    atlas_xmin : float
        x position of ATLAS label, by default 0.04
    atlas_ymax : float
        y position of ATLAS label, by default 0.9
    atlas_fontsize : float
        fontsize of ATLAS label, by default 10
    """

    title: str = ""
    draw_errors: bool = True

    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None
    y_scale: float = 1.3
    logy: bool = True
    xlabel: str = None
    ylabel: str = None
    label_fontsize: int = 12
    fontsize: int = 10

    n_ratio_panels: int = 1

    figsize: tuple = None
    dpi: int = 400

    # legend settings
    leg_fontsize: int = None
    leg_loc: str = "best"
    leg_ncol: int = 1

    # defining ATLAS style and tags
    apply_atlas_style: bool = True
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Internal Simulation"
    atlas_second_tag: str = (
        "$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample, $f_{c}=0.018$"
    )
    atlas_xmin: float = 0.04
    atlas_ymax: float = 0.9
    atlas_fontsize: int = None

    def __post_init__(self):
        """Check for allowed values.

        Raises
        ------
        ValueError
            if n_ratio_panels not in [0, 1, 2]
        """
        self.__check_figsize()
        allowed_n_ratio_panels = [0, 1, 2]
        if self.n_ratio_panels not in allowed_n_ratio_panels:
            raise ValueError(
                f"{self.n_ratio_panels} not allwed value for `n_ratio_panels`. "
                f"Allowed are {allowed_n_ratio_panels}"
            )
        if self.leg_fontsize is None:
            self.leg_fontsize = self.fontsize
        if self.atlas_fontsize is None:
            self.atlas_fontsize = self.fontsize

    def __check_figsize(self):
        """Check `figsize`

        Raises
        ------
        ValueError
            if shape of `figsize` is not a tuple or list with length 2
        """
        if self.figsize is None:
            return
        if isinstance(self.figsize, list) and len(self.figsize) == 2:
            self.figsize = tuple(self.figsize)
        elif not isinstance(self.figsize, tuple) or len(self.figsize) != 2:
            raise ValueError(
                f"You passed `figsize` as {self.figsize} which is not allowed. "
                "Either a tuple or a list of size 2 is allowed"
            )


class plot_base(plot_object):
    """Base class for plotting"""

    def __init__(self, **kwargs) -> None:
        """Initialise class

        Parameters
        ----------
        **kwargs : kwargs
            kwargs from `plot_object`
        """
        super().__init__(**kwargs)
        if self.apply_atlas_style:
            self.initialise_atlas_style()
        self.axis_top = None
        self.axis_ratio_1 = None
        self.axis_ratio_2 = None
        self.ratio_y_labels = {}
        self.fig = None

    def initialise_figure(self, sub_plot_index: int = 5):
        """
        Initialising plt.figure for different scenarios depending how many ratio panels
        are requested.

        Parameters
        ----------
        sub_plot_index : int, optional
            indicates for the scenario with one ratio how large the upper and lower
            panels are, by default 5
        """
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            # no ratio panel
            self.fig = plt.figure(
                figsize=(8, 8) if self.figsize is None else self.figsize
            )
            self.axis_top = plt.gca()
        elif self.n_ratio_panels == 1:
            # 1 ratio panel
            self.fig = plt.figure(
                figsize=(9.352, 6.616) if self.figsize is None else self.figsize
            )

            gs = gridspec.GridSpec(8, 1, figure=self.fig)
            self.axis_top = self.fig.add_subplot(gs[:sub_plot_index, 0])
            self.axis_ratio_1 = self.fig.add_subplot(
                gs[sub_plot_index:, 0], sharex=self.axis_top
            )

        elif self.n_ratio_panels == 2:
            # 2 ratio panels
            self.fig = plt.figure(
                figsize=(8, 8) if self.figsize is None else self.figsize
            )

            # Define the grid of the subplots
            gs = gridspec.GridSpec(11, 1, figure=self.fig)
            self.axis_top = self.fig.add_subplot(gs[:5, 0])
            self.axis_ratio_1 = self.fig.add_subplot(gs[5:8, 0], sharex=self.axis_top)
            self.axis_ratio_2 = self.fig.add_subplot(gs[8:, 0], sharex=self.axis_top)

        if self.n_ratio_panels >= 1:
            plt.setp(self.axis_top.get_xticklabels(), visible=False)
        if self.n_ratio_panels >= 2:
            plt.setp(self.axis_ratio_1.get_xticklabels(), visible=False)

    def set_title(self, title: str = None, **kwargs):
        """Set title of top panel.

        Parameters
        ----------
        title : str, optional
            title of top panel, if None using the value form the class variables,
            by default None
        **kwargs : kwargs
            kwargs passed to `plt.axis.set_title()`
        """
        self.axis_top.set_title(self.title if title is None else title, **kwargs)

    def set_logy(self, force: bool = False):
        """Set log scale of y-axis of main panel.

        Parameters
        ----------
        force : bool, optional
            forcing logy even if class variable is False, by default False
        """
        if not self.logy and not force:
            return
        if not self.logy:
            logger.warning("Setting log of y-axis but `logy` flag was set to False.")
        self.axis_top.set_yscale("log")
        ymin, ymax = self.axis_top.get_ylim()
        self.y_scale = ymin * ((ymax / ymin) ** self.y_scale) / ymax

    def set_y_lim(self):
        """Set limits of y-axis."""
        ymin, ymax = self.axis_top.get_ylim()
        self.axis_top.set_ylim(
            ymin if self.ymin is None else self.ymin,
            ymax * self.y_scale if self.ymax is None else self.ymax,
        )

    def set_ylabel(self, ax, label: str = None, align_right: bool = True, **kwargs):
        """Set y-axis label.

        Parameters
        ----------
        ax : plt.axis
            matplotlib axis object
        label : str, optional
            x-axis label, by default None
        align_right : bool, optional
            alignment of y-axis label, by default True
        **kwargs, kwargs
            kwargs passed to `plt.axis.set_ylabel()`
        """
        label_options = {}
        if align_right:
            label_options = {
                "fontsize": self.label_fontsize,
                "horizontalalignment": "right",
                "y": 1.0,
            }
        else:
            label_options = {
                "fontsize": self.label_fontsize,
            }

        ax.set_ylabel(
            self.ylabel if label is None else label,
            **label_options,
            **kwargs,
        )

    def set_xlabel(self, label: str = None, **kwargs):
        """Set x-axis label.

        Parameters
        ----------
        label : str, optional
            x-axis label, by default None
        **kwargs : kwargs
            kwargs passed to `plt.axis.set_xlabel`
        """
        xlabel_args = {
            "xlabel": self.xlabel if label is None else label,
            "horizontalalignment": "right",
            "x": 1.0,
            "fontsize": self.label_fontsize,
        }
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            self.axis_top.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 1:
            self.axis_ratio_1.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 2:
            self.axis_ratio_2.set_xlabel(**xlabel_args, **kwargs)

    def set_tick_params(self, labelsize: int = None, **kwargs):
        """Set x-axis label.

        Parameters
        ----------
        labelsize : int, optional
            label size of x- and y- axis ticks, by default None
            if None then using global fontsize
        **kwargs : kwargs
            kwargs passed to `plt.axis.set_xlabel`
        """
        labelsize = self.fontsize if labelsize is None else labelsize
        self.axis_top.tick_params(axis="y", labelsize=labelsize, **kwargs)
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            self.axis_top.tick_params(axis="x", labelsize=labelsize, **kwargs)
        elif self.n_ratio_panels == 1:
            self.axis_ratio_1.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_1.tick_params(axis="x", labelsize=labelsize, **kwargs)
        elif self.n_ratio_panels == 2:
            self.axis_ratio_1.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_2.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_2.tick_params(axis="x", labelsize=labelsize, **kwargs)

    def set_xlim(self, xmin: float = None, xmax: float = None, **kwargs):
        """Set limits of x-axis

        Parameters
        ----------
        xmin : float, optional
            min of x-axis, by default None
        xmax : float, optional
            max of x-axis, by default None
        **kwargs : kwargs
            kwargs passed to `plt.axis.set_xlim`
        """
        self.axis_top.set_xlim(
            self.xmin if xmin is None else xmin,
            self.xmax if xmax is None else xmax,
            **kwargs,
        )

    def savefig(
        self,
        plot_name: str,
        transparent: bool = True,
        dpi: int = None,
        **kwargs,
    ):
        """Save plot to disk.

        Parameters
        ----------
        plot_name : str
            file name of the plot
        transparent : bool, optional
            if plot transparent, by default True
        dpi : int, optional
            dpi for plotting, by default 400
        **kwargs : kwargs
            kwargs passed to `plt.savefig`
        """
        logger.debug(f"Saving plot to {plot_name}")
        plt.savefig(
            plot_name,
            transparent=transparent,
            dpi=self.dpi if dpi is None else dpi,
            **kwargs,
        )

    @staticmethod
    def tight_layout(**kwargs):
        """abstract function of plt tight_layout

        Parameters
        ----------
        **kwargs: kwargs
            kwargs from `plt.tight_layout()`
        """
        plt.tight_layout(**kwargs)

    @staticmethod
    def clear():
        """resetting matplolib figure"""
        plt.close()
        plt.clf()

    def make_atlas_tag(self):
        """Drawing ATLAS tag."""
        pas.makeATLAStag(
            ax=self.axis_top,
            fig=self.fig,
            first_tag=self.atlas_first_tag,
            second_tag=self.atlas_second_tag,
            xmin=self.atlas_xmin,
            ymax=self.atlas_ymax,
            fontsize=self.atlas_fontsize,
        )

    def initialise_atlas_style(self, force: bool = False):
        """Initialising ATLAS style.

        Parameters
        ----------
        force : bool, optional
            force ATLAS style also if class variable is False, by default False
        """
        if self.apply_atlas_style or force:
            logger.info("Initialise ATLAS style.")
            applyATLASstyle(mtp)
            if force:
                logger.warning(
                    "Initilising ATLAS style even though `apply_atlas_style` is set to "
                    "False."
                )

    def make_legend(self, handles: list, labels: list = None, **kwargs):
        """Drawing legend on axis.

        Parameters
        ----------
        handles :  list
            list of matplotlib.lines.Line2D object returned when plotting
        labels : list, optional
            plot labels. If None, the labels are extracted from the `handles`.
            By default None
        **kwargs : kwargs
            kwargs which can be passed to matplotlib axis
        """
        self.axis_top.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles]
            if labels is None
            else labels,
            loc=self.leg_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
            **kwargs,
        )

    def set_ratio_label(self, ratio_panel: int, label: str):
        """Associate the rejection class to a ratio panel

        Parameters
        ----------
        ratio_panel : int
            ratio panel either 1 or 2
        label : str
            y-axis label of the ratio panel

        Raises
        ------
        ValueError
            if requested ratio panels and given ratio_panel do not match.
        """
        # TODO: could add possibility to specify ratio label as function of rej_class
        if self.n_ratio_panels < ratio_panel and ratio_panel not in [1, 2]:
            raise ValueError(
                "Requested ratio panels and given ratio_panel do not match."
            )
        self.ratio_y_labels[ratio_panel] = label
