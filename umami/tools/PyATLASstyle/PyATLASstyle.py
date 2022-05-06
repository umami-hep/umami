"""
Implementation of ATLAS style conventions.
Adapted from https://github.com/rateixei/PyATLASstyle
"""
import os

from matplotlib import font_manager


def get_good_colours():
    """List of colours adequate for plotting

    Returns
    -------
    list
        list with colours
    """
    return ["#AA3377", "#228833", "#4477AA", "#CCBB44", "#EE6677", "#BBBBBB"]


def applyATLASstyle(mpl):
    """Adapting matplotlib style to resemble ATLAS style recommendations.

    Parameters
    ----------
    mpl : matplotlib
        matplotlib library
    """
    font_dir = os.path.abspath(__file__).replace("PyATLASstyle.py", "fonts/")
    font_dirs = [
        font_dir,
    ]

    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for f in font_files:
        font_manager.FontManager.addfont(font_manager.fontManager, path=f)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.antialiased"] = False
    mpl.rcParams["lines.linewidth"] = 2.5
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["xtick.major.size"] = 10
    mpl.rcParams["xtick.minor.size"] = 5
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["ytick.major.size"] = 10
    mpl.rcParams["ytick.minor.size"] = 5
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 3


def makeATLAStag(
    ax,
    fig,
    first_tag: str = "",
    second_tag: str = "",
    xmin: float = 0.04,
    ymax: float = 0.9,
    fontsize: int = 10,
):
    """Adding ATLAS tag to figure.

    Parameters
    ----------
    ax : plt.axis
        matplotlib.pyplot.axis object on which label should be added
    fig : plt.figure
        matplotlib.pyplot.figure object
    first_tag : str
        First row of the ATLAS Tag, by default ""
    second_tag : str
         Second Row of the ATLAS Tag, by default ""
    xmin : float
        x position of label, by default 0.04
    ymax : float
        y position of label, by default 0.9
    fontsize : int
        fontsize of label, by default 10
    """
    line_spacing = 0.6
    box0 = ax.text(
        xmin,
        ymax,
        "ATLAS",
        fontweight="bold",
        fontstyle="italic",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    box0_ext_tr = ax.transAxes.inverted().transform(
        box0.get_window_extent(renderer=fig.canvas.get_renderer())
    )
    box1 = ax.text(
        box0_ext_tr[1][0],
        ymax,
        " ",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    box1_ext_tr = ax.transAxes.inverted().transform(
        box1.get_window_extent(renderer=fig.canvas.get_renderer())
    )
    ax.text(
        box1_ext_tr[1][0],
        ymax,
        first_tag,
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    ax.text(
        xmin,
        ymax
        - (box0_ext_tr[1][1] - box0_ext_tr[0][1])
        * (line_spacing + len(second_tag.split("\n"))),
        second_tag,
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
