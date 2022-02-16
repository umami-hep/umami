"""Plotting bases for specialised plotting."""
from dataclasses import dataclass

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.configuration import logger
from umami.tools import applyATLASstyle


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
    fontsize: int = 12

    n_ratio_panels: int = 2

    figsize: tuple = None
    dpi: int = 400

    # legend settings
    leg_fontsize: int = 10
    leg_loc: str = "best"
    leg_ncol: int = 1

    # defining ATLAS style and tags
    apply_atlas_style: bool = True
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Internal Simulation"
    atlas_second_tag: str = (
        "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample, fc=0.018"
    )
    atlas_xmin: float = 0.04
    atlas_ymax: float = 0.9
    atlas_fontsize: int = 10

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

    def initialise_figure(self):
        """
        Initialising plt.figure for different scenarios depending how many ratio panels
        are requested.
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
            self.axis_top = self.fig.add_subplot(gs[:5, 0])
            self.axis_ratio_1 = self.fig.add_subplot(gs[5:, 0], sharex=self.axis_top)

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
                "fontsize": self.fontsize,
                "horizontalalignment": "right",
                "y": 1.0,
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
        }
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            self.axis_top.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 1:
            self.axis_ratio_1.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 2:
            self.axis_ratio_2.set_xlabel(**xlabel_args, **kwargs)

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

    def make_legend(self, handles, **kwargs):
        """Drawing legend on axis.

        Parameters
        ----------
        handles : matplotlib.lines.Line2D
            Line2D object returned when plotting
        **kwargs : kwargs
            kwargs which can be passed to matplotlib axis
        """
        self.axis_top.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
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


class roc_plot(plot_base):
    """Roc plot class"""

    def __init__(self, **kwargs) -> None:
        """roc plot properties

        Parameters
        ----------
        **kwargs : kwargs
            kwargs from `plot_base`
        """
        super().__init__(**kwargs)
        self.test = ""
        self.rocs = {}
        self.roc_ratios = {}
        self.ratio_axes = {}
        self.rej_class_ls = {}
        self.label_colours = {}
        self.leg_rej_labels = {}
        self.reference_roc = None
        self.initialise_figure()
        self.eff_min, self.eff_max = (1, 0)
        # setting default linestyles if no linestyles provided
        # solid line and densed dotted dashed
        self.default_linestyles = ["-", (0, (3, 1, 1, 1))]

    def add_roc(self, roc: object, key: str = None, reference: bool = False):
        """Adding roc object to figure.

        Parameters
        ----------
        roc : roc class
            roc curve
        key : str, optional
            unique identifier for roc, by default None
        reference : bool, optional
            if roc is used as reference for ratio calculation, by default False

        Raises
        ------
        KeyError
            if unique identifier key is used twice
        """
        if key is None:
            key = len(self.rocs) + 1
        if key in self.rocs:
            raise KeyError(
                f"Duplicated key {key} already used for roc unique identifier."
            )

        self.rocs[key] = roc
        # set linestyle
        if roc.rej_class not in self.rej_class_ls:
            self.rej_class_ls[roc.rej_class] = (
                self.default_linestyles[len(self.rej_class_ls)]
                if roc.linestyle is None
                else roc.linestyle
            )
        elif (
            roc.linestyle != self.rej_class_ls[roc.rej_class]
            and roc.linestyle is not None
        ):
            logger.warning(
                "You specified a different linestyle for the same rejection class "
                f"{roc.rej_class}. Will keep the linestyle defined first."
            )
        if roc.linestyle is None:
            roc.linestyle = self.rej_class_ls[roc.rej_class]

        # set colours
        if roc.label not in self.label_colours:
            self.label_colours[roc.label] = (
                pas.get_good_colours()[len(self.label_colours)]
                if roc.colour is None
                else roc.colour
            )
        elif roc.colour != self.label_colours[roc.label] and roc.colour is not None:
            logger.warning(
                f"You specified a different colour for the same label {roc.label}. "
                "This will lead to a mismatch in the line colours and the legend."
            )
        if roc.colour is None:
            roc.colour = self.label_colours[roc.label]

        if reference:
            logger.debug(f"Setting roc {key} as reference for {roc.rej_class}.")
            self.set_roc_reference(key, roc.rej_class)

    def set_roc_reference(self, key: str, rej_class: str):
        """Setting the reference roc curves used in the ratios

        Parameters
        ----------
        key : str
            unique identifier of roc object
        rej_class : str
            rejection class encoded in roc curve

        Raises
        ------
        ValueError
            if more rejection classes are set than actual ratio panels available.
        """
        if self.reference_roc is None:
            self.reference_roc = {}
            self.reference_roc[rej_class] = key
        elif rej_class not in self.reference_roc:
            if len(self.reference_roc) >= self.n_ratio_panels:
                raise ValueError(
                    "You cannot set more rejection classes than available ratio panels."
                )
            self.reference_roc[rej_class] = key
        else:
            logger.warning(
                f"You specified a second roc curve {key} as reference for ratio. "
                f"Using it as new reference instead of {self.reference_roc[rej_class]}."
            )
            self.reference_roc[rej_class] = key

    def set_leg_rej_labels(self, rej_class: str, label: str):
        """Set legend label for rejection class

        Parameters
        ----------
        rej_class : str
            rejection class
        label : str
            label added in legend
        """
        self.leg_rej_labels[rej_class] = label

    def set_ratio_class(self, ratio_panel: int, rej_class: str, label: str):
        """Associate the rejection class to a ratio panel

        Parameters
        ----------
        ratio_panel : int
            ratio panel either 1 or 2
        rej_class : str
            rejeciton class associated to that panel
        label : str
            y-axis label of the ratio panel

        Raises
        ------
        ValueError
            if requested ratio panels and given ratio_panel do not match.
        """
        if self.n_ratio_panels < ratio_panel and ratio_panel not in [1, 2]:
            raise ValueError(
                "Requested ratio panels and given ratio_panel do not match."
            )
        self.ratio_axes[ratio_panel] = rej_class
        self.set_ratio_label(ratio_panel, label)

    def add_ratios(self):
        """Calculating ratios.

        Raises
        ------
        ValueError
            if number of reference rocs and ratio panels don't match
        ValueError
            if no ratio classes are set
        """
        if len(self.reference_roc) != self.n_ratio_panels:
            raise ValueError(
                f"{len(self.reference_roc)} reference rocs defined but requested "
                f"{self.n_ratio_panels} ratio panels."
            )
        if len(self.ratio_axes) != self.n_ratio_panels:
            raise ValueError(
                "Ratio classes not set, set them first with `set_ratio_class`."
            )
        self.plot_ratios(ax=self.axis_ratio_1, rej_class=self.ratio_axes[1])
        self.axis_ratio_1.grid()

        if self.n_ratio_panels == 2:
            self.plot_ratios(ax=self.axis_ratio_2, rej_class=self.ratio_axes[2])
            self.axis_ratio_2.grid()

    def get_xlim_auto(self):
        """Returns min and max efficiency values

        Returns
        -------
        float
            min and max efficiency values
        """

        for elem in self.rocs.values():
            self.eff_min = min(np.min(elem.sig_eff), self.eff_min)
            self.eff_max = max(np.max(elem.sig_eff), self.eff_min)

        return self.eff_min, self.eff_max

    def plot_ratios(self, ax: plt.axis, rej_class: str):
        """Plotting ratio curves

        Parameters
        ----------
        ax : plt.axis
            matplotlib axis object
        rej_class : str
            rejection class
        """
        for key, elem in self.rocs.items():
            if elem.rej_class != rej_class:
                continue
            ratio_sig_eff, ratio, ratio_err = elem.divide(
                self.rocs[self.reference_roc[rej_class]]
            )
            self.roc_ratios[key] = (ratio_sig_eff, ratio, ratio_err)
            ax.plot(
                ratio_sig_eff,
                ratio,
                color=elem.colour,
                linestyle=elem.linestyle,
                linewidth=1.6,
            )
            if ratio_err is not None:
                ax.fill_between(
                    ratio_sig_eff,
                    ratio - ratio_err,
                    ratio + ratio_err,
                    color=elem.colour,
                    alpha=0.3,
                    zorder=1,
                )

    def draw_wps(self, wps: list, same_height: bool = False, colour: str = "red"):
        """Drawing working points in plot

        Parameters
        ----------
        wps : list
            list of working points to draw
        same_height : bool, optional
            working point lines on same height, by default False
        colour : str, optional
            colour of the vertical line, by default "red"
        """
        for wp in wps:
            # Set y-point of the WP lines/text
            ytext = 0.65 if same_height else 1.25 - wp

            self.axis_top.axvline(
                x=wp,
                ymax=ytext,
                color=colour,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Set the number above the line
            self.axis_top.text(
                x=wp - 0.005,
                y=ytext + 0.005,
                s=f"{int(wp * 100)}%",
                transform=self.axis_top.get_xaxis_text1_transform(0)[0],
                fontsize=10,
            )

            if self.n_ratio_panels > 0:
                self.axis_ratio_1.axvline(
                    x=wp, color=colour, linestyle="dashed", linewidth=1.0
                )
            if self.n_ratio_panels == 2:
                self.axis_ratio_2.axvline(
                    x=wp, color=colour, linestyle="dashed", linewidth=1.0
                )

    def make_split_legend(self, handles):
        """Draw legend for the case of 2 ratios, splitting up legend into models and
        rejection class.

        Parameters
        ----------
        handles : list
            list of Line2D objects to extract info for legend

        Raises
        ------
        ValueError
            if not 2 ratios requested
        """

        if self.n_ratio_panels != 2:
            raise ValueError("For a split legend you need 2 ratio panels.")

        line_list_rej = []
        for elem in [self.ratio_axes[1], self.ratio_axes[2]]:
            line_list_rej.append(
                mtp.lines.Line2D(
                    [],
                    [],
                    color="k",
                    label=self.leg_rej_labels[elem],
                    linestyle=self.rej_class_ls[elem],
                )
            )

        legend_flavs = self.axis_top.legend(
            handles=line_list_rej,
            labels=[handle.get_label() for handle in line_list_rej],
            loc="upper center",
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
        )

        # Add the second legend to plot
        self.axis_top.add_artist(legend_flavs)

        # Get the labels for the legends
        labels_list = []
        lines_list = []

        for line in handles:
            if line.get_label() not in labels_list:
                labels_list.append(line.get_label())
                lines_list.append(line)

        # Define the legend
        self.axis_top.legend(
            handles=lines_list,
            labels=labels_list,
            loc=self.leg_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
        )

    def draw(
        self,
        rlabel: str = None,
        labelpad: int = None,
    ):
        """Draw ploting

        Parameters
        ----------
        rlabel : str or list, optional
            label of ratio panels if only 1 ratio is requested
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default None
        """
        plt_handles = self.plot_roc()
        xmin, xmax = self.get_xlim_auto()
        # if self.xmin is not None or self.xmax is not None:
        self.set_xlim(
            xmin if self.xmin is None else self.xmin,
            xmax if self.xmax is None else self.xmax,
        )
        self.add_ratios()
        self.axis_top.grid()
        self.set_title()
        self.set_logy()
        self.set_y_lim()
        self.set_xlabel()
        self.set_ylabel(self.axis_top)

        if self.n_ratio_panels > 0:
            if rlabel is not None:
                rlabel_1 = rlabel[0] if len(rlabel) > 0 else rlabel
            else:
                rlabel_1 = self.ratio_y_labels[1]
            self.set_ylabel(
                self.axis_ratio_1,
                rlabel_1,
                align_right=False,
                labelpad=labelpad,
            )
        if self.n_ratio_panels == 2:
            self.set_ylabel(
                self.axis_ratio_2,
                self.ratio_y_labels[2] if rlabel is None else rlabel[1],
                align_right=False,
                labelpad=labelpad,
            )
        if self.use_atlas_tag:
            self.make_atlas_tag()

        if self.n_ratio_panels < 2:
            self.make_legend(plt_handles)
        else:
            if not self.leg_rej_labels:
                self.leg_rej_labels[self.ratio_axes[1]] = self.ratio_axes[1]
                self.leg_rej_labels[self.ratio_axes[2]] = self.ratio_axes[2]

            self.make_split_legend(handles=plt_handles)
        self.tight_layout()

    def plot_roc(self, **kwargs):
        """Plotting roc curves

        Parameters
        ----------
        **kwargs: kwargs
            kwargs passed to plt.axis.plot

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []
        for key, elem in self.rocs.items():
            plt_handles = plt_handles + self.axis_top.plot(
                elem.sig_eff[elem.non_zero_mask],
                elem.bkg_rej[elem.non_zero_mask],
                linestyle=elem.linestyle,
                color=elem.colour,
                label=elem.label if elem is not None else key,
                zorder=2,
                **kwargs,
            )
        return plt_handles
