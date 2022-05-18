"""Plotting module."""
# flake8: noqa
# pylint: skip-file

# This implementation is just temporary to get puma das a replacement for the plotting
# API within umami
# later everything needs to be changed in umami to the new naming

from puma.histogram import Histogram as histogram
from puma.histogram import HistogramPlot as histogram_plot
from puma.plot_base import PlotBase as plot_base
from puma.plot_base import PlotLineObject as plot_line_object
from puma.plot_base import PlotObject as plot_object
from puma.roc import Roc as roc
from puma.roc import RocPlot as roc_plot
from puma.var_vs_eff import VarVsEff as var_vs_eff
from puma.var_vs_eff import VarVsEffPlot as var_vs_eff_plot

# from umami.plotting.histogram import histogram, histogram_plot
# from umami.plotting.plot_base import plot_base, plot_line_object, plot_object
# from umami.plotting.roc import roc, roc_plot
# from umami.plotting.var_vs_eff import var_vs_eff, var_vs_eff_plot
