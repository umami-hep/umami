from scipy.interpolate import pchip
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def eff_err(x, N):
    return np.sqrt(x * (1 - x) / N)


def plotROCRatio(teffs, beffs, labels, title='', text='',
                 ylabel='Background rejection',
                 tag='', figDir='../figures', subDir='mc16d',
                 styles=None, colors=None, xmin=.6, ymax=1e3, legFontSize=10,
                 rrange=None, rlabel='Ratio', binomialErrors=False, nTest=0,
                 plot_name=None, alabel=None,
                 legcols=2, flipRatio=False, labelpad=None):
    '''

    Plot the ROC curves with binomial errors with the ratio plot in a subpanel
    underneath. This function all accepts the same inputs as plotROC, and the
    additional ones are listed below.

    Addtional Inputs:
    - rrange: The range on the y-axis for the ratio panel
    - rlabel: The label for the y-axis for the ratio panel
    - binomialErrors: whether to include binomial errors for the rejection
                      curves
    - nTest: A list of the same length as beffs, with the number of events used
            to calculate the background efficiencies.
            We need this To calculate the binomial errors on the background
            rejection,
            using the formula given by
            http://home.fnal.gov/~paterno/images/effic.pdf.
    '''
    # set ylabel
    if ylabel == "light":
        ylabel = 'Light-Flavour Jet Rejection ($1/\epsilon_{l}$)'  # noqa
    elif ylabel == "c":
        ylabel = '$c$-Jet Rejection ($1/\epsilon_{c}$)'  # noqa

    # The points that I'm going to c.f. the ratio over
#     xx = np.linspace(0.6,1,101)
    xx = np.linspace(xmin, 1, 101)

    if binomialErrors and nTest == 0:
        print("Error: Requested binomialErrors, but did not pass nTest.",
              "Will NOT plot rej errors.")
        binomialErrors = False

    if styles is None:
        styles = ['-' for i in teffs]
    if colors is None:
        colors = ['C{}'.format(i) for i in range(len(teffs))]

    if type(nTest) != list:
        nTest = [nTest] * len(teffs)

    # Define the figure with two subplots of unequal sizes
    # plt.rcParams['figure.constrained_layout.use'] = True
    fig = plt.figure(figsize=(8, 5.5), constrained_layout=True)
    gs = gridspec.GridSpec(8, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:5, 0])
    ax2 = fig.add_subplot(gs[5:, 0], sharex=ax1)
    custom_lines = []

    for i, (teff, beff, label, style, color, nte) in enumerate(
            zip(teffs, beffs, labels, styles, colors, nTest)):

        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(teff)))
        # Also mask the rejections that are 0
        nonzero = (beff != 0) & (dx > 0)
        x = teff[nonzero]
        y = np.divide(1, beff[nonzero])

        if binomialErrors:

            yerr = np.power(y, 2) * eff_err(beff[nonzero], nte)

            y1 = y - yerr
            y2 = y + yerr

            # ax1.fill_between(x,y1,y2, color=color,label=label,zorder=2,
            # alpha=0.5)
            ax1.fill_between(x, y1, y2, color=color, zorder=2)  # , alpha=0.5)
            # ax1.plot(x,y, style, color=color,label=label, linewidth=2)
            custom_lines.append(Line2D([], [], color=color, lw=2.5,
                                       label=label, ls='-'))
            # p.fill_between(x,y1,y2)
            # ax1.fill_between(x,y1,y2, color=color,label=label,
            # linestyle=style) #, alpha=0.5)

        else:
            ax1.semilogy(x, y, style, color=color, label=label)

        f = pchip(x, y)

        if i == 0:
            f0 = f
            # Add a line indicating where the ratio=1
            # ax2.plot([xmin,1],[1,1], style, color=color, linewidth=2.)

            if binomialErrors:
                # Use a grey contour to show our uncertainty in the value of 1
                # ax2.fill_between(x,1-yerr/y,1+yerr/y,color='grey',alpha=0.5)
                ax2.plot(xx, np.ones(len(xx)), style, color=color,
                         linewidth=1.6)
                ax2.fill_between(x, 1 - yerr / y, 1 + yerr / y, color=color,
                                 alpha=0.5, zorder=1)
                # y0 = y
                # yerr0 = yerr
        else:
            if flipRatio:
                # ax2.plot(xx,  f0(xx) / f(xx), style, color=color)
                ratio_i = f0(xx) / f(xx)
                ratio_ix = f0(x) / f(x)
            else:
                ratio_i = f(xx) / f0(xx)
                ratio_ix = f(x) / f0(x)

            ax2.plot(xx, ratio_i, style, color=color, linewidth=1.6)
            ax2.fill_between(x, ratio_ix - yerr/f(x), ratio_ix + yerr / f(x),
                             color=color, alpha=0.5, zorder=1)

    # Add axes, titels and the legend
    ax2.set_xlabel('$b$-efficiency', fontsize=12, horizontalalignment='right',
                   x=1.0)
    ax1.set_ylabel(ylabel, fontsize=12, horizontalalignment='right', y=1.0)
    ax1.set_yscale('log')
    ax2.set_ylabel(rlabel, labelpad=labelpad, fontsize=12)
    ax1.text(xmin, ymax, text, horizontalalignment='left',
             verticalalignment='bottom')
    ax1.set_title(title)
    ax2.grid()
    ax1.grid()
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Print label
    if alabel is not None:
        ax1.text(**alabel, transform=ax1.transAxes)

    # Set the axes to be the same as those used in the pub note
    ax1.set_xlim(xmin, 1)
    ax1.set_ylim(1, ymax)

    ax2.set_xlim(xmin, 1)
    if rrange is not None:
        ax2.set_ylim(rrange)
    # ax1.legend(loc='best',fontsize=legFontSize, ncol=legcols)#, title="DL1r")
    ax1.legend(handles=custom_lines, loc='best', fontsize=legFontSize,
               ncol=legcols)  # , title="DL1r")

    # plt.tight_layout()
    if len(tag) != 0:
        plt.savefig('{}/{}/rocRatio_{}.pdf'.format(figDir, subDir, tag),
                    bbox_inches='tight', transparent=True)
    if plot_name is not None:
        plt.savefig(plot_name, transparent=True)
    # plt.show()
