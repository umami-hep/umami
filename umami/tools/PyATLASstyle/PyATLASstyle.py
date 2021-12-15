import os

# taken from https://github.com/rateixei/PyATLASstyle


def get_good_colors():
    return ["#AA3377", "#228833", "#4477AA", "#CCBB44", "#EE6677", "#BBBBBB"]


def applyATLASstyle(mtp):
    font_dir = os.path.abspath(__file__).replace("PyATLASstyle.py", "fonts/")
    font_dirs = [
        font_dir,
    ]

    import matplotlib.font_manager as font_manager

    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for f in font_files:
        font_manager.FontManager.addfont(font_manager.fontManager, path=f)
    # mtp.rcParams['font.family'] = 'Arial'
    # mtp.rcParams['font.family'] = 'TeX Gyre Heros'
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
    # mtp.rcParams['mathtext.fontset'] = 'custom'
    # mtp.rcParams['mathtext.it'] = 'Arial:italic'
    # mtp.rcParams['mathtext.bf'] = 'Arial:bold'
    # mtp.rcParams['mathtext.rm'] = 'Arial'
    # mtp.rcParams['mathtext.sf'] = 'Arial'
    # mtp.rcParams['mathtext.cal'] = 'Arial:italic'
    # mtp.rcParams['mathtext.tt'] = 'Arial'
    # mtp.rcParams['mathtext.it'] = 'TeX Gyre Heros:italic'
    # mtp.rcParams['mathtext.bf'] = 'TeX Gyre Heros:bold'
    # mtp.rcParams['mathtext.rm'] = 'TeX Gyre Heros'
    # mtp.rcParams['mathtext.sf'] = 'TeX Gyre Heros'
    # mtp.rcParams['mathtext.cal'] = 'TeX Gyre Heros:italic'
    # mtp.rcParams['mathtext.tt'] = 'TeX Gyre Heros'
    mtp.rcParams["axes.unicode_minus"] = False
    mtp.rcParams["pdf.fonttype"] = 3
    # mtp.rcParams["axes.axisbelow"] = False


def makeATLAStag(
    ax,
    fig,
    first_tag: str = "",
    second_tag: str = "",
    xmin: float = 0.04,
    ymax: float = 0.85,
    fontsize: int = 10,
):
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
