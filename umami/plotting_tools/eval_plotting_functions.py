"""Collection of plotting function for ftag performance plots."""
# pylint: disable=consider-using-f-string, invalid-name
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from puma import (
    Histogram,
    HistogramPlot,
    Line2D,
    Line2DPlot,
    PlotBase,
    Roc,
    RocPlot,
    VarVsEff,
    VarVsEffPlot,
)

from umami.configuration import global_config, logger
from umami.data_tools import retrieve_cut_string
from umami.plotting_tools.utils import translate_kwargs


def plot_var_vs_eff(
    df_list: list,
    tagger_list: list,
    model_labels: list,
    plot_name: str,
    class_labels: list,
    main_class: str,
    flavour: str,
    variable: str = None,
    working_point: float = 0.77,
    disc_cut: float = None,
    fixed_eff_bin: bool = False,
    bin_edges: list = None,
    working_point_line: bool = False,
    grid: bool = False,
    colours: list = None,
    alpha: float = 0.8,
    linewidth: float = 1.6,
    **kwargs,
) -> None:
    """For a given list of models, plot the b-eff, l and c-rej as a function of jet pt.

    Parameters
    ----------
    df_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    model_labels : list
        List of the labels which are to be used in the plot.
    plot_name : str
        Path, Name and format of the resulting plot file.
    class_labels : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class (= signal class). For b-tagging obviously "bjets"
    flavour : str
        Class that is to be plotted. For all non-signal classes, this
        will be the rejection and for the signal class, this will be
        the efficiency.
    variable : str, optional
        The variable against which the efficiency/rejection is plotted.
        By default None, which will use the pT as variable in GeV.
    working_point : float, optional
        Working point which is to be used, by default 0.77.
    disc_cut : float, optional
        Set a discriminant cut value for all taggers/models.
    fixed_eff_bin : bool, optional
        Calculate the WP cut on the discriminant per bin, by default None.
    bin_edges : list, optional
        As the name says, the edges of the bins used. Will be set
        automatically, if None. By default None.
    working_point_line : bool, optional
        Print a WP line in the upper plot, by default False.
    grid : bool, optional
        Use a grid in the plots, by default False
    colours : list, optional
        Custom colour list for the different models, by default None
    alpha : float, optional
        Value for visibility of the plot lines, by default 0.8
    linewidth : float, optional
        Define the linewidth of the plotted lines, by default 1.6
    **kwargs : kwargs
        kwargs for `VarVsEffPlot` function

    Raises
    ------
    ValueError
        If deprecated options are given.
    """

    # Translate the kwargs to new naming scheme
    kwargs = translate_kwargs(kwargs)

    # Look for deprecated variables and throw error if given
    deprecated = {
        "SWP_Comparison",
        "SWP_label_list",
        "Passed",
        "binomialErrors",
        "frameon",
        "Ratio_Cut",
    } & set(kwargs.keys())
    if deprecated:
        raise ValueError(
            f"The options {list(deprecated)} are deprecated. "
            "Please use the plotting python API."
        )

    # Set default variable
    if variable is None:
        variable = "pt"

    # Set xlabel if not given
    if "xlabel" not in kwargs and variable == "pt":
        kwargs["xlabel"] = r"$p_T$ [GeV]"

    # Set bin edges if not given
    if bin_edges is None:
        bin_edges = [0, 20, 50, 90, 150, 300, 1000]

    # Get global config of the classes
    flav_cat = global_config.flavour_categories
    n_curves = len(tagger_list)
    if colours is None:
        colours = [f"C{i}" for i in range(n_curves)]

    # Get the indicies of the flavours
    index_dict = {f"{flavour}": i for i, flavour in enumerate(class_labels)}

    # check length
    # TODO: change in python 3.10 -> add to zip() with strict=True argument
    if not all(
        len(elem) == n_curves
        for elem in [
            df_list,
            model_labels,
            tagger_list,
            colours,
        ]
    ):
        raise ValueError("Passed lists do not have same length.")

    mode = "bkg_rej"
    y_label = f'{flav_cat[flavour]["legend_label"]} rejection'
    if main_class == flavour:
        mode = "sig_eff"
        y_label = f'{flav_cat[flavour]["legend_label"]} efficiency'

    plot_pt = VarVsEffPlot(
        mode=mode,
        ylabel=y_label,
        n_ratio_panels=1,
        **kwargs,
    )
    # Redefine Second Tag with inclusive or fixed tag
    if fixed_eff_bin:
        plot_pt.atlas_second_tag = (
            f"{plot_pt.atlas_second_tag}\nConstant "
            rf"$\epsilon_b$ = {int(working_point * 100)}% per bin"
        )
    else:
        plot_pt.atlas_second_tag = (
            f"{plot_pt.atlas_second_tag}\nInclusive "
            rf"$\epsilon_b$ = {int(working_point * 100)}%"
        )

    for i, (df_results, model_label, tagger, colour) in enumerate(
        zip(df_list, model_labels, tagger_list, colours)
    ):
        # Get jet pts
        jet_var = df_results[variable]

        # Make pT in GeV
        if variable == "pt":
            jet_var = jet_var / 1e3

        # Get truth labels
        is_signal = df_results["labels"] == index_dict[main_class]
        is_bkg = (
            df_results["labels"] == index_dict[flavour] if mode == "bkg_rej" else None
        )
        disc = df_results[f"disc_{tagger}"]
        plot_pt.add(
            VarVsEff(
                x_var_sig=jet_var[is_signal],
                disc_sig=disc[is_signal],
                x_var_bkg=jet_var[is_bkg] if mode == "bkg_rej" else None,
                disc_bkg=disc[is_bkg] if mode == "bkg_rej" else None,
                bins=bin_edges,
                working_point=working_point,
                disc_cut=disc_cut,
                flat_per_bin=fixed_eff_bin,
                label=model_label,
                colour=colour,
                alpha=alpha,
                linewidth=linewidth,
            ),
            reference=i == 0,
        )

    plot_pt.draw()
    # Set grid
    if grid is True:
        plot_pt.set_grid()
    # Set WP Line
    if working_point_line is True:
        plot_pt.draw_hline(working_point)
        if main_class != flavour:
            logger.warning(
                "You set `working_point_line` to True but you are not looking at the"
                " singal efficiency. It will probably not be visible on your plot."
            )
    plot_pt.savefig(plot_name)


def plot_roc(
    df_results_list: list,
    tagger_list: list,
    rej_class_list: list,
    labels: list,
    plot_name: str,
    main_class: str = "bjets",
    df_eff_key: str = "effs",
    draw_errors: bool = True,
    labelpad: int = None,
    working_points: list = None,
    same_height_working_points: bool = True,
    linestyles: list = None,
    colours: list = None,
    n_test: list = None,
    reference_ratio: list = None,
    **kwargs,
):
    """Plotting the rejection curve for a given background class
    for the given models/taggers against the main_class efficiency.
    A ratio plot (first given model is the reference) is plotted in
    a subpanel underneath.

    Parameters
    ----------
    df_results_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    rej_class_list : list
        List of the class rejection which is to be plotted for each model.
    labels : list
        List of labels for the given models.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    plot_name : str
        Path, Name and format of the resulting plot file.
    df_eff_key : str, optional
        Dict key under which the efficiencies of the main class are saved,
        by default "effs"
    draw_errors : bool, optional
        Binominal errors on the lines, by default True
    labelpad : int, optional
        Spacing in points from the axes bounding box including
        ticks and tick labels, by default None
    working_points : list, optional
        List of working points which are to be plotted as
        vertical lines in the plot, by default None
    same_height_working_points : bool, optional
        Decide, if all working points lines have the same height or
        not, by default True
    linestyles : list, optional
        List of linestyles to use for the given models, by default None
    colours : list, optional
        List of linecolors to use for the given models, by default None
    n_test : list, optional
        A list of the same length as rej_class_list, with the number of
        events used to calculate the background efficiencies.
        We need this To calculate the binomial errors on the background
        rejection, using the formula given by
        http://home.fnal.gov/~paterno/images/effic.pdf, by default 0
    reference_ratio : list, optional
        List of bools indicating which roc used as reference for ratio calculation,
        by default None
    **kwargs : kwargs
        kwargs passed to RocPlot

    Raises
    ------
    ValueError
        if n_test not int, float of given for each roc
    ValueError
        if lists don't have the same length
    """

    # Check for number of provided Rocs
    n_rocs = len(df_results_list)

    if "ratio_id" in kwargs:
        if reference_ratio is None and kwargs["ratio_id"] is not None:
            # if old keyword is used the syntax was also different
            # translating this now into the new syntax
            # the old syntax looked like
            # ratio_id = [0, 0, 1, 1]
            # rej_class_list = ['ujets', 'ujets', 'cjets', 'cjets']
            # tagger_list = ['RNNIP', 'DIPS', 'RNNIP', 'DIPS']
            # in that case the first entry was used for the upper ratio and the
            #  3rd entry for the 2nd ratio
            # in the new syntax this would mean
            # reference_ratio = [True, False, True, False]
            reference_ratio = []
            _tmp_ratio_id = []
            for elem in kwargs["ratio_id"]:
                reference_ratio.append(elem not in _tmp_ratio_id)
                _tmp_ratio_id.append(elem)
        kwargs.pop("ratio_id")
    kwargs = translate_kwargs(kwargs)

    # catching default value as in old implementation to maintain backwards
    # compatibility
    if reference_ratio is None:
        reference_ratio = []
        _tmp_ratio_id = []
        for elem in rej_class_list:
            reference_ratio.append(elem not in _tmp_ratio_id)
            _tmp_ratio_id.append(elem)

    # remnant of old implementation passing empty list as default
    if linestyles == []:
        linestyles = None

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Loop over the given rejection types and add them to a lists
    flav_list = list(OrderedDict.fromkeys(rej_class_list))

    # Check the number of flavours given
    if len(flav_list) == 1:
        # Set number of ratio panels
        n_ratio_panels = 1
        ylabel = f'{flav_cat[rej_class_list[0]]["legend_label"]} Rejection'

    elif len(flav_list) == 2:
        n_ratio_panels = 2
        ylabel = "Background rejection"

    else:
        raise ValueError(
            f"Can't plot {len(flav_list)} rejections! Only 1 or 2 is supported!"
        )

    # Append a linestyles for each model determined by the rejections
    # with solid lines or dashed dotted lines
    if linestyles is None:
        linestyles = [
            "-" if elem == flav_list[0] else (0, (3, 1, 1, 1))
            for elem in rej_class_list
        ]

    # Create list for the models
    model_list = list(OrderedDict.fromkeys(labels))

    # Fill in the colors for the models given
    if colours is None:
        model_colours = {model: f"C{i}" for i, model in enumerate(model_list)}
        colours = [model_colours[elem] for elem in labels]

    if draw_errors is True:
        # Check if n_test is provided in all samples
        if n_test is None:
            logger.error(
                "Requested binomialErrors, but no number of jets used for "
                "rejection calculation are given! "
                "Will NOT plot rej errors."
            )
            draw_errors = False

        elif isinstance(n_test, list):
            if len(n_test) != len(df_results_list):
                raise ValueError(
                    "The provided `n_test` do not have the same length as the "
                    "`df_results_list`."
                )

    # check length
    # TODO: change in python 3.10 -> add to zip() with strict=True argument
    if not all(
        len(elem) == n_rocs
        for elem in [
            df_results_list,
            tagger_list,
            rej_class_list,
            labels,
            linestyles,
            colours,
            n_test,
            reference_ratio,
        ]
    ):
        raise ValueError("Passed lists do not have same length.")

    roc_plot = RocPlot(
        n_ratio_panels=n_ratio_panels,
        ylabel=ylabel.capitalize(),
        xlabel=f'{flav_cat[main_class]["legend_label"]} efficiency'.capitalize(),
        **kwargs,
    )
    roc_plot.set_ratio_class(
        ratio_panel=1,
        rej_class=flav_list[0],
    )

    if n_ratio_panels > 1:
        roc_plot.set_ratio_class(
            ratio_panel=2,
            rej_class=flav_list[1],
        )

    if working_points is not None:
        roc_plot.draw_vlines(
            xs=working_points,
            same_height=same_height_working_points,
        )

    # Loop over the models with the different settings for each model
    for (
        df_results,
        tagger,
        rej_class,
        label,
        linestyle,
        colour,
        nte,
        ratio_ref,
    ) in zip(
        df_results_list,
        tagger_list,
        rej_class_list,
        labels,
        linestyles,
        colours,
        n_test,
        reference_ratio,
    ):
        roc_curve = Roc(
            df_results[df_eff_key],
            df_results[f"{tagger}_{rej_class}_rej"],
            n_test=nte,
            rej_class=rej_class,
            signal_class=main_class,
            label=label,
            colour=colour,
            linestyle=linestyle,
        )
        roc_plot.add_roc(roc_curve, reference=ratio_ref)

    # Draw and save the plot
    roc_plot.draw(labelpad=labelpad)
    roc_plot.savefig(plot_name)


def plot_saliency(
    maps_dict: dict,
    plot_name: str,
    target_eff: float,
    jet_flavour: str,
    PassBool: bool,
    nFixedTrks: int,
    cmap: str = "PiYG",
    **kwargs,
):
    """Plot the saliency map given in maps_dict.

    Parameters
    ----------
    maps_dict : dict
        Dict with the saliency values inside
    plot_name : str
        Path, Name and format of the resulting plot file.
    target_eff : float, optional
        Working point to use, by default 0.77
    jet_flavour : str, optional
        Class which is to be plotted, by default "bjets"
    PassBool : bool, optional
        Decide, if the jets need to pass (True) or fail (False)
        the working point cut, by default True
    nFixedTrks : int, optional
        Decide, how many tracks the jets need to have, by default 8
    cmap : str, optional
        Colour map to use for the saliency map. By defaulkt PiYG
    **kwargs
        Keyword arguments handed to the plotting API

    Raises
    ------
    ValueError
        If the number of variables in the given dict is not the same
        as the number of variables used to calculate the saliency map.
    """
    # Get list of variables used for saliency maps
    variable_list = maps_dict["Variables_list"]

    # Load gradient map from file
    gradient_map = maps_dict[f"{int(target_eff * 100)}_{jet_flavour}_{PassBool}"][
        :, :nFixedTrks
    ]

    # Check kwargs for fontsize
    kwargs["fontsize"] = (
        kwargs["fontsize"]
        if "fontsize" in kwargs and kwargs["fontsize"] is not None
        else 12
    )

    # Check kwargs for figsize
    kwargs["figsize"] = (
        kwargs["figsize"]
        if "figsize" in kwargs and kwargs["figsize"] is not None
        else (0.9 * nFixedTrks, 0.5 * len(variable_list))
    )

    # Set log of x and y axis to False
    kwargs["logx"] = False
    kwargs["logy"] = False

    # Check that the number of variables are similar in the map and the var_dict
    if len(variable_list) != gradient_map.shape[0]:
        raise ValueError(
            "Number of variables in variable list and in gradient dict are "
            "not the same!"
        )

    # Define the colour scales and maps for the flavours
    colour_scale = np.max(np.abs(gradient_map))

    # Init the saliency plot
    saliency_plot = PlotBase(**kwargs)
    saliency_plot.initialise_figure()

    # Set x- and y-ticks
    saliency_plot.axis_top.set_xticks(
        ticks=np.arange(nFixedTrks),
        labels=np.arange(1, nFixedTrks + 1),
        fontsize=kwargs["fontsize"],
    )
    saliency_plot.axis_top.set_yticks(
        ticks=np.arange(len(variable_list)),
        labels=variable_list,
        fontsize=kwargs["fontsize"],
    )

    # Plot saliency
    im = saliency_plot.axis_top.imshow(
        X=gradient_map,
        cmap=cmap,
        origin="lower",
        vmin=-colour_scale,
        vmax=colour_scale,
    )

    # Plot colorbar and set size to graph size
    divider = make_axes_locatable(saliency_plot.axis_top)
    cax = divider.append_axes("right", size=f"{int(100/nFixedTrks)}%", pad=0.05)
    colorbar = saliency_plot.fig.colorbar(im, cax=cax)
    colorbar.ax.set_title(
        r"$\frac{\mathrm{\partial} D_{b}}{\mathrm{\partial} x_{ik}}$",
        size=1.6 * kwargs["fontsize"],
    )

    # Set the fontsize of the colorbar yticks
    for t in colorbar.ax.get_yticklabels():
        t.set_fontsize(kwargs["fontsize"])

    saliency_plot.initialise_plot()
    saliency_plot.savefig(plot_name)


def plot_score(
    df_list: list,
    model_labels: list,
    tagger_list: list,
    class_labels_list: list,
    main_class: str,
    plot_name: str,
    working_points: list = None,
    same_height_working_points: bool = False,
    **kwargs,
) -> None:
    """Plot the tagging discriminant scores for the evalutated jets.

    Parameters
    ----------
    df_list : list
        List with the pandas DataFrames inside.
    model_labels : list
        List of labels for the given models
    tagger_list : list
        List of tagger names of the used taggers
    class_labels_list : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class. For b-tagging obviously "bjets"
    plot_name : str
        Path, Name and format of the resulting plot file.
    working_points : list, optional
        List of workings points which are plotted as vertical lines.
        By default, the working points for the first provided model is
        calculated and plotted, not for the rest! By default None
    same_height_working_points : bool, optional
        Decide, if all working points lines have the same height or
        not, by default True
    **kwargs : kwargs
        kwargs for `HistogramPlot` function
    """

    # Set number of ratio panels if not specified
    if "n_ratio_panels" not in kwargs:
        kwargs["n_ratio_panels"] = 1 if len(df_list) > 1 else 0

    # Translate the old keywords
    kwargs = translate_kwargs(kwargs)

    # Define a few linestyles which are good
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Get flavour categories from global config file
    # TODO Add get_good_linestyles
    flav_cat = global_config.flavour_categories

    # Get the truth value and the truth variable to use for the used flavours
    cut_strings = retrieve_cut_string(class_labels_list[0] + [main_class])

    # Init the Histogram plot object
    score_plot = HistogramPlot(**kwargs)

    # Set the xlabel
    if score_plot.xlabel is None:
        score_plot.xlabel = f'{flav_cat[main_class]["legend_label"]} discriminant'

    if working_points is not None:
        # Init a new list for the WPs
        working_point_xvalues = []
        working_point_label = []

        # Calculate x value of WP line
        for working_point in sorted(working_points, reverse=True):
            working_point_xvalues.append(
                np.percentile(
                    df_list[0].query(cut_strings[main_class])[f"disc_{tagger_list[0]}"],
                    (1 - working_point) * 100,
                )
            )
            working_point_label.append(f"{int(working_point * 100)} %")

        score_plot.draw_vlines(
            xs=working_point_xvalues,
            same_height=same_height_working_points,
            labels=working_point_label,
        )

    for model_counter, (
        df_results,
        linestyle,
        model_label,
        tagger,
        class_labels,
    ) in enumerate(
        zip(
            df_list,
            linestyles,
            model_labels,
            tagger_list,
            class_labels_list,
        )
    ):
        # Get the truth value and the truth variable to use for the used flavours
        cut_strings = retrieve_cut_string(class_labels)

        for iter_flavour in class_labels:
            score_plot.add(
                Histogram(
                    values=df_results.query(cut_strings[iter_flavour])[
                        f"disc_{tagger}"
                    ],
                    flavour=iter_flavour,
                    ratio_group=iter_flavour,
                    label=model_label if len(model_labels) > 1 else None,
                    linestyle=linestyle,
                ),
                reference=not bool(model_counter) if len(df_list) > 1 else False,
            )

    # Draw and save the plot
    score_plot.draw()
    score_plot.savefig(plot_name)


def plot_prob(
    df_list: list,
    model_labels: list,
    tagger_list: list,
    class_labels_list: list,
    flavour: str,
    plot_name: str,
    **kwargs,
) -> None:
    """
    Plot the probability output for the given flavour for
    one/multiple models with a ratio plot in a subpanel (only if
    multiple models are provided).

    Parameters
    ----------
    df_list : list
        List of pandas DataFrames with the probability values inside
    model_labels : list
        List of labels for the given models
    tagger_list : list
        List of tagger names
    class_labels_list : list
        List with the class_labels used in the different taggers
    flavour : str
        Probability of this flavour is plotted.
    plot_name : str
        Path, Name and format of the resulting plot file.
    **kwargs : kwargs
        kwargs for `HistogramPlot` function
    """

    # Set number of ratio panels if not specified
    if "n_ratio_panels" not in kwargs:
        kwargs["n_ratio_panels"] = 1 if len(df_list) > 1 else 0

    # Translate the old keywords
    kwargs = translate_kwargs(kwargs)

    # Define a few linestyles which are good
    # TODO Add get_good_linestyles
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # Get flavour categories from global config file
    flav_cat = global_config.flavour_categories

    # Init the histogram plot object
    prob_plot = HistogramPlot(**kwargs)

    # Set the xlabel
    if prob_plot.xlabel is None:
        prob_plot.xlabel = f'{flav_cat[flavour]["legend_label"]} probability'

    for model_counter, (
        df_results,
        linestyle,
        model_label,
        tagger,
        class_labels,
    ) in enumerate(
        zip(
            df_list,
            linestyles,
            model_labels,
            tagger_list,
            class_labels_list,
        )
    ):
        # Get the truth value and the truth variable to use for the used flavours
        cut_strings = retrieve_cut_string(class_labels)

        for iter_flavour in class_labels:
            prob_plot.add(
                Histogram(
                    values=df_results.query(cut_strings[iter_flavour])[
                        f'{tagger}_{flav_cat[flavour]["prob_var_name"]}'
                    ],
                    flavour=iter_flavour,
                    ratio_group=iter_flavour,
                    label=model_label if len(model_labels) > 1 else None,
                    linestyle=linestyle,
                ),
                reference=not bool(model_counter) if len(df_list) > 1 else False,
            )

    # Draw and save the plot
    prob_plot.draw()
    prob_plot.savefig(plot_name)


def plot_confusion_matrix(
    df_results: dict,
    tagger_name: str,
    class_labels: list,
    plot_name: str,
    colorbar: bool = True,
    show_absolute: bool = False,
    show_normed: bool = True,
    transparent: bool = False,
    dpi: int = 400,
) -> None:
    """Plotting the confusion matrix for a given tagger.

    Parameters
    ----------
    df_results : dict
        Loaded pandas dataframe from the evaluation file
    tagger_name : str
        Name of the tagger in the evaluation file
    class_labels : list
        List of class labels used
    plot_name : str
        Full path + name of the plot with the extension
    colorbar : bool, optional
        Decide, if colourbar is shown or not, by default True
    show_absolute : bool, optional
        Show the absolute, by default False
    show_normed : bool, optional
        Show the output normed, by default True
    transparent : bool, optional
        Decide, if the background is transparent or not, by default False
    dpi : int, optional
        Sets a DPI value for the plot that is produced (mainly for png),
        by default 400
    """

    # Get a list of the tagger prob variables
    prob_list = []
    for prob in class_labels:
        prob_list.append(
            f'{tagger_name}_{global_config.flavour_categories[prob]["prob_var_name"]}'
        )

    # Get the truth
    y_target = df_results["labels"]

    # Get a bool list which jets to keep (removing unused flavour jets)
    jets_to_keep = y_target <= len(class_labels) - 1

    # Get the probabilities of the tagger and select the highest
    y_predicted = np.argmax(df_results[prob_list].values, axis=1)

    # Remove unused flavour jets
    y_target = y_target[jets_to_keep]
    y_predicted = y_predicted[jets_to_keep]

    # Define the confusion matrix
    cm = confusion_matrix(y_target=y_target, y_predicted=y_predicted, binary=False)

    # Plot the colormap
    mlxtend_plot_cm(
        conf_mat=cm,
        colorbar=colorbar,
        show_absolute=show_absolute,
        show_normed=show_normed,
        class_names=class_labels,
    )

    # Set tight layout for the plot
    plt.tight_layout()

    # Save the plot to path
    plt.savefig(plot_name, transparent=transparent, dpi=dpi)
    plt.close()


def plot_fraction_contour(
    df_results_list: list,
    tagger_list: list,
    label_list: list,
    rejections_to_plot: list,
    plot_name: str,
    rejections_to_fix_list: list,
    marked_points_list: list = None,
    colour_list: list = None,
    linestyle_list: list = None,
    **kwargs,
):
    """Plot contour plots for the given taggers for two rejections.
    The rejections are calulated with different fractions. If more
    than two rejections are available, the others need to be set to
    a fixed value.

    Parameters
    ----------
    df_results_list : list
        List of dicts with the results of evaluate_model.py inside for
        the different models which are to be plotted.
    tagger_list : list
        List of the models/taggers that are to be plotted.
    label_list : list
        List with the labels for the given taggers.
    rejections_to_plot : list
        List with two elements. The elements are the names for the two
        rejections that are plotted against each other. For example,
        ["cjets", "ujets"].
    plot_name : str
        Path, Name and format of the resulting plot file.
    rejections_to_fix_list : list
        List of dicts with the extra rejections. If more than two rejections are
        available, you need to fix the other rejections to a specific
        value. The dict entry key is the name of the rejection, for
        example "bbjets", and the entry is the value that it is set to,
        for example 0.2.
    marked_points_list : list, optional
        List with marker dicts for each model provided. The dict contains
        the information needed to plot the marker, like the fraction values,
        which colour is used etc, by default None
    colour_list : list, optional
        List with colours for the given taggers, by default None
    linestyle_list : list, optional
        List with linestyles for the given taggers, by default None
    **kwargs : kwargs
        kwargs for `fraction_scan_plot` function

    Raises
    ------
    IndexError
        If the given number of tagger names, labels and data dicts are not
        the same.
    ValueError
        If the given marker values summed (together with the fixed rejection
        values) is not equal to one (or not on the plotted curve).
    """

    # Translate the old keywords
    kwargs = translate_kwargs(kwargs)

    # Make a list of None to be able to loop over it
    if marked_points_list is None:
        marked_points_list = [None] * len(df_results_list)

    if colour_list is None:
        colour_list = [None] * len(df_results_list)

    if linestyle_list is None:
        linestyle_list = [None] * len(df_results_list)

    # Get global config for flavours
    flav_cat = global_config.flavour_categories

    # Init new list for the fraction values
    fraction_list = []

    # Extract fraction steps from dict
    for _, dict_values in df_results_list[0].items():
        fraction_list.append(dict_values[rejections_to_plot[0]])

    # Remove all doubled items
    fraction_list = list(dict.fromkeys(fraction_list))

    # Init the fraction scan plot
    frac_plot = Line2DPlot(**kwargs)

    # Ensure that for each model, a tagger name and a label is provided and vice versa
    # TODO Change in Python 3.10 to strict=True in the zip() function which will ensure
    # same length
    if not all(
        len(lst) == len(df_results_list)
        for lst in [
            tagger_list,
            label_list,
            colour_list,
            linestyle_list,
            rejections_to_fix_list,
            marked_points_list,
        ]
    ):
        raise IndexError(
            "Not the same amount of Evaluation files, names and labels are given! "
            "Please check that!"
        )

    # Loop over the combinations of the models
    for (
        df_results,
        tagger,
        label,
        colour,
        linestyle,
        fixed_rejections,
        marked_point_dict,
    ) in zip(
        df_results_list,
        tagger_list,
        label_list,
        colour_list,
        linestyle_list,
        rejections_to_fix_list,
        marked_points_list,
    ):
        # Init a dict for the rejections with an empty list for each rejection
        df = {str(rejection): [] for rejection in rejections_to_plot}

        # Init x- and y positon for the mark
        plot_point_x = None
        plot_point_y = None

        # Loop over the fraction values
        for frac in fraction_list:
            # Loop over the entries in the provided results
            for dict_key, dict_values in df_results.items():
                # Init a rej_to_fix bool
                rej_to_fix_bool = True

                # Check if all not-plotted rejections have a fixed value given
                if fixed_rejections:
                    for (
                        rej_to_fix_key,
                        rej_to_fix_key_value,
                    ) in fixed_rejections.items():
                        if not dict_values[rej_to_fix_key] == rej_to_fix_key_value:
                            rej_to_fix_bool = False

                # Check that the correct combination of fraction value and
                # rejection is chosen
                if (
                    dict_key.startswith(f"{tagger}_")
                    and dict_values[rejections_to_plot[0]] == frac
                    and rej_to_fix_bool
                ):
                    for rejection in rejections_to_plot:
                        df[rejection].append(dict_values[f"{rejection}_rej"])

                    if (
                        marked_point_dict
                        and marked_point_dict[rejections_to_plot[0]]
                        == dict_values[rejections_to_plot[0]]
                    ):
                        plot_point_x = dict_values[f"{rejections_to_plot[0]}_rej"]
                        plot_point_y = dict_values[f"{rejections_to_plot[1]}_rej"]

        # Plot the contour
        frac_plot.add(
            Line2D(
                x_values=df[rejections_to_plot[0]],
                y_values=df[rejections_to_plot[1]],
                label=label,
                colour=colour,
                linestyle=linestyle,
            )
        )

        if marked_point_dict:
            if plot_point_x is None and plot_point_y is None:
                raise ValueError(
                    f"The given marker for {tagger} is not on the plotted line!"
                    " Please check that you have given correct position values!"
                    " The values of the marker point (and the ones in fixed_rejections"
                    " if defined) must add up to 1!"
                )

            # Build the correct marker for the point
            frac_label_x = flav_cat[rejections_to_plot[0]]["prob_var_name"]
            frac_x_value = marked_point_dict[rejections_to_plot[0]]
            frac_label_y = flav_cat[rejections_to_plot[1]]["prob_var_name"]
            frac_y_value = marked_point_dict[rejections_to_plot[1]]

            point_label = (
                rf"{label} $f_{{{frac_label_x}}} = {frac_x_value}$,"
                rf" $f_{{{frac_label_y}}} = {frac_y_value}$"
            )

            # Add the marker for the contour
            frac_plot.add(
                Line2D(
                    x_values=plot_point_x,
                    y_values=plot_point_y,
                    colour=marked_point_dict["colour"]
                    if "colour" in marked_point_dict
                    and marked_point_dict["colour"] is not None
                    else None,
                    marker=marked_point_dict["marker_style"]
                    if "marker_style" in marked_point_dict
                    and marked_point_dict["marker_style"] is not None
                    else None,
                    label=marked_point_dict["marker_label"]
                    if "marker_label" in marked_point_dict
                    and marked_point_dict["marker_label"] is not None
                    else point_label,
                    markersize=marked_point_dict["markersize"]
                    if "markersize" in marked_point_dict
                    and marked_point_dict["markersize"] is not None
                    else None,
                    markeredgewidth=marked_point_dict["markeredgewidth"]
                    if "markeredgewidth" in marked_point_dict
                    and marked_point_dict["markeredgewidth"] is not None
                    else None,
                ),
                is_marker=True,
            )

    # Set the xlabel
    if frac_plot.xlabel is None:
        frac_plot.xlabel = (
            flav_cat[rejections_to_plot[0]]["legend_label"].capitalize() + " rejection"
        )

    # Set the ylabel
    if frac_plot.ylabel is None:
        frac_plot.ylabel = (
            flav_cat[rejections_to_plot[1]]["legend_label"].capitalize() + " rejection"
        )

    # Draw and save the plot
    frac_plot.draw()
    frac_plot.savefig(plot_name)
