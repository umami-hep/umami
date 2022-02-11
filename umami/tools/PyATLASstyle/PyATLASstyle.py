"""
Implementation of ATLAS style conventions.
Adapted from https://github.com/rateixei/PyATLASstyle
"""
import os


def get_good_colors():
    """List of colours adequate for plotting

    Returns
    -------
    list
        list with colours
    """
    return ["#AA3377", "#228833", "#4477AA", "#CCBB44", "#EE6677", "#BBBBBB"]


def applyATLASstyle(mtp):
    """Adapting matplotlib style to resemble ATLAS style recommendations.

    Parameters
    ----------
    mtp : matplotlib
        matplotlib library
    """
    font_dir = os.path.abspath(__file__).replace("PyATLASstyle.py", "fonts/")
    font_dirs = [
        font_dir,
    ]

    import matplotlib.font_manager as font_manager

    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for f in font_files:
        font_manager.FontManager.addfont(font_manager.fontManager, path=f)
    mtp.rcParams["font.size"] = 10
    mtp.rcParams["legend.frameon"] = False
    mtp.rcParams["legend.fontsize"] = 10
    mtp.rcParams["lines.antialiased"] = False
    mtp.rcParams["lines.linewidth"] = 2.5
    mtp.rcParams["xtick.direction"] = "in"
    mtp.rcParams["xtick.top"] = True
    mtp.rcParams["xtick.minor.visible"] = True
    mtp.rcParams["xtick.major.size"] = 10
    mtp.rcParams["xtick.minor.size"] = 5
    mtp.rcParams["ytick.direction"] = "in"
    mtp.rcParams["ytick.right"] = True
    mtp.rcParams["ytick.minor.visible"] = True
    mtp.rcParams["ytick.major.size"] = 10
    mtp.rcParams["ytick.minor.size"] = 5
    mtp.rcParams["axes.unicode_minus"] = False
    mtp.rcParams["pdf.fonttype"] = 3


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
