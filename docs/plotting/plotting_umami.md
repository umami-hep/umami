# Plotting the evaluation results
The evaluation results can be plotted using different functions. There is the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py), [plotting_epoch_performance](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_epoch_performance.py) and the [plot_input_variables.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plot_input_variables.py). Each plotting script is explained in its dedicated section.

## plotting_umami.py
The plotting_umami.py is used to plot the results of the evaluation script. Different plots can be produced with it which are fully customizable. All plots that are defined in the `plotting_umami_config_X.yaml`. The `X` defines the tagger here but its just a name. All config files are usable with the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script.

### Yaml Config File
**Important: The indentation in this .yaml is important due to the way the files are read by the script.**
A fully written one can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_umami_config_dips.yaml).

The config file starts with the `Eval_parameters`. Here the `Path_to_models_dir` is set, where the models are saved. Also the `model_name` and the `epoch` which is to be plotted is set. A boolean parameter can be set here to add the epoch to the end of the plot name. This is `epoch_to_name`. For example, this can look like this:

```yaml
Eval_parameters:
  Path_to_models_dir: /home/fr/fr_fr/fr_af1100/b-Tagging/packages/umami/umami
  model_name: dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full
  epoch: 59
  epoch_to_name: True
```

In the different available plots, there are options that are available in mostly all of them. So they will be explained next. For specific options, look at the comment in the section of the plot.

| Options | Explanation |
|---------|-------------|
| `Name_of_the_plot` | All plots start with no indentation and the name of plot. This will be the output name of the plot file and has no impact on the plot itself. |
| `type` | This option specifies the plot function that is used. |
| `data_set_name` | Decides which evaluated dataset (or file) is used. This `data_set_name` are set in the `train_config` yaml file which is used in the evaluation of the model. There the different files are getting their own `data_set_name` which needs to be the same as here! |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! (Possible entries are defined in the [global_config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/global_config.yaml))|
| `models_to_plot` | In the comparison plots, the models which are to be plotted needs to be defined in here. You can add as many models as you want. For example this can be used to plot the results of the different taggers in one plot (e.g. for score comparison or ROC curves). The different models can be assisted with `evaluation_file` to point to the results file you have created with `evaluate_model.py`. e.g.`evaluation_file: YOURMODEL/results/results-rej_per_eff-229.h5` |
| `plot_settings` | In this section, all optional plotting settings are defined. They don't need to be defined but you can. For the specific available options in each function, look in the corresponding section. |

For plotting, these different plots are available:

#### Confusion Matrix
Plot a confusion matrix. For example:

```yaml
confusion_matrix_Dips_ttbar:
  type: "confusion_matrix"
  data_set_name: "ttbar"
  class_labels: ["ujets", "cjets", "bjets"]
  plot_settings:
    colorbar: True
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `colorbar` | Bool | Optional | Define, if the colorbar on the side is shown or not. |

#### Probability
Plotting the DNN probability output for a specific class. For example:

```yaml
Dips_prob_pb:
  type: "probability"
  data_set_name: "ttbar"
  tagger_name: "dips"
  class_labels: ["ujets", "cjets", "bjets"]
  prob_class: "bjets"
  plot_settings:
    Log: True
    nBins: 50
    yAxisIncrease: 10
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample"
    yAxisAtlasTag: 0.9
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | String | Necessary | This gives the type of plot function used. Must be `"probability"` here. |
| `data_set_name` | String | Necessary | Name of the dataset that is used. This is set at the evaluation in the train config. |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `prob_class` | String | Necessary | Class of the to be plotted probability. |
| `ApplyAtlasStyle` | Bool | Optional | Set the plotting style to ATLAS (root like look). |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `nBins` | Int | Optional | Number of bins that are used. |
| `Log` | Bool | Optional | Decide if the y-axis plotted in log |
| `figsize` | List | Optional | List with the figure sizes. For example [5, 3] |
| `labelFontSize` | Int | Optional | Fontsize of the labels |
| `loc_legend` | String | Optional | Sets the position of the legend. Default is "best". |
| `ncol` | Int | Optional | Number of columns in the legend. |
| `x_label` | String | Optional | Set the x-axis label. Default is "DNN Output" |
| `yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
| `yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |

#### Probability Comparison
Plotting the DNN probability output for different models. For example:

```yaml
Dips_prob_comparison_pb:
  type: "probability_comparison"
  prob_class: "bjets"
  models_to_plot:
    dips_r22:
      data_set_name: "ttbar"
      label: "rnnip"
      tagger_name: "rnnip"
      class_labels: ["ujets", "cjets", "bjets"]
    dips_r21:
      data_set_name: "ttbar"
      label: "DIPS"
      tagger_name: "dips"
      class_labels: ["ujets", "cjets", "bjets"]
  plot_settings:
    nBins: 50
    set_logy: True
    yAxisIncrease: 1.5
    figsize: [8, 6]
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets"
    yAxisAtlasTag: 0.9
    Ratio_Cut: [0.5, 1.5]
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | String | Necessary | This gives the type of plot function used. Must be `"probability_comparison"` here. |
| `prob_class` | String | Necessary | Class of the to be plotted probability. |
| `dips_r21` | None | Necessary | Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything. |
| `data_set_name` | String | Necessary | Name of the dataset that is used. This is set at the evaluation in the train config. |
| `label` | String | Necessary | Label for the Legend in the plot. Will be "FLAVOUR-jet LABEL" |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `x_label` | String | Optional | Set the x-axis label. Default is "DNN Output" |
| `Log` | Bool | Optional | Decide if the y-axis plotted in log |
| `nBins` | Int | Optional | Number of bins that are used. |
| `yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional | A list of the width and hight of the plot. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
| `Ratio_Cut` | List | Optional | Two element list that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |

#### Scores
Plotting the b-tagging discriminant scores for the different jet flavors. For example:

```yaml
scores_Dips_ttbar:
  type: "scores"
  data_set_name: "ttbar"
  tagger_name: "dips"
  class_labels: ["ujets", "cjets", "bjets"]
  main_class: "bjets"
  plot_settings:
    WorkingPoints: [0.60, 0.70, 0.77, 0.85]
    nBins: 50
    yAxisIncrease: 1.3
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample"
    yAxisAtlasTag: 0.9
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `main_class` | String | Class which is to be tagged. |
| `WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
| `nBins` | Int | Optional | Number of bins that are used. |
| `yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### Scores Comparison
Plotting the b-tagging discriminant scores for the different jet flavors for different models in the same plot. For example:

```yaml
scores_Dips_ttbar_comparison:
  type: "scores_comparison"
  main_class: "bjets"
  models_to_plot:
    dips_r21:
      data_set_name: "ttbar"
      tagger_name: "dips"
      class_labels: ["ujets", "cjets", "bjets"]
      label: "$t\\bar{t}$"
    dips_r22:
      data_set_name: "ttbar"
      tagger_name: "dips"
      class_labels: ["ujets", "cjets", "bjets"]
      label: "$t\\bar{t} 2$"
  plot_settings:
    WorkingPoints: [0.60, 0.70, 0.77, 0.85]
    nBins: 50
    yAxisIncrease: 1.4
    figsize: [8, 6]
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets"
    yAxisAtlasTag: 0.9
    Ratio_Cut: [0.5, 1.5]
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `main_class` | String | Class which is to be tagged. |
| `dips_r21` | None | Necessary | Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything. |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `label` | String | Necessary | Label for the Legend in the plot. |
| `WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
| `nBins` | Int | Optional | Number of bins that are used. |
| `yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional | A list of the width and hight of the plot. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
| `Ratio_Cut` | List | Optional | Two element list that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |

#### ROC Curves
Plotting the ROC Curves of the rejection rates against the b-tagging efficiency. For example:

```yaml
Dips_light_flavour_ttbar:
  type: "ROC"
  main_class: "bjets"
  models_to_plot:
    dips_r21:
      data_set_name: "ttbar"
      label: "DIPS"
      tagger_name: "dips"
      rejection_class: "cjets"
  plot_settings:
    binomialErrors: True
    xmin: 0.5
    ymax: 1000000
    figsize: [7, 6]
    WorkingPoints: [0.60, 0.70, 0.77, 0.85]
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample, fc=0.018"
    yAxisAtlasTag: 0.9
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `main_class` | String | Class which is to be tagged. |
| `dips_r21` | None | Necessary | Name of the model which is to be plotted. Not affecting anything, just for you. You can change dips_r21 to anything. |
| `label` | String | Necessary | Label for the Legend in the plot. |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `rejection_class` | String | Necessary | Class which the main flavour is plotted against. |
| `colors` | String | Optional | Set color for the model. If None, the colors will be set automatically for all |
| `linestyle` | String | Optional | Set linestyle for the model. If None, the linestyles will be set automatically for all models (all the same). |
| `xlabel` | String | Optional | Set the xlabel.
| `ylabel` | String | Optional | Set the ylabel of the X-rejection. For example: 'c' will output `c-flavor rejection`. |
| `binomialErrors` | Bool | Optional | Plot binomial errors to plot. |
| `xmin` | Float | Optional | Set the minimum b efficiency in the plot (which is the xmin limit). |
| `ymax` | Float | Optional | The maximum y axis. |
| `WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional |A list of the width and hight of the plot.
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### Comparison ROC Curves (Double Rejection ROC)
Plotting the ROC Curves of two rejection rates against a efficiency. You need to define a model for each model/rejection pair. For example:

```yaml
Dips_Comparison_flavour_ttbar:
  type: "ROC_Comparison"
  models_to_plot:
    rnnip_u:
      data_set_name: "ttbar"
      label: "recomm. RNNIP"
      tagger_name: "rnnip"
      rejection_class: "ujets"
    rnnip_c:
      data_set_name: "ttbar"
      label: "recomm. RNNIP"
      tagger_name: "rnnip"
      rejection_class: "cjets"
    dips_r21_u:
      data_set_name: "ttbar"
      label: "DIPS"
      tagger_name: "dips"
      rejection_class: "ujets"
    dips_r21_c:
      data_set_name: "ttbar"
      label: "DIPS"
      tagger_name: "dips"
      rejection_class: "cjets"
  plot_settings:
    binomialErrors: True
    xmin: 0.5
    ymax: 1000000
    figsize: [9, 9]
    WorkingPoints: [0.60, 0.70, 0.77, 0.85]
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Validation Sample, fc=0.018"
    yAxisAtlasTag: 0.9
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `dips_r21` | None | Necessary | Name of the model which is to be plotted. Not affecting anything, just for you. You can change dips_r21 to anything. |
| `label` | String | Necessary | Label for the Legend in the plot. |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `rejection_class` | String | Necessary | Class which the main flavour is plotted against. |
| `xlabel` | String | Optional | Set the xlabel. |
| `binomialErrors` | Bool | Optional | Plot binomial errors to plot. |
| `xmin` | Float | Optional | Set the minimum b efficiency in the plot (which is the xmin limit). |
| `ymax` | Float | Optional | The maximum y axis. |
| `WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional |A list of the width and hight of the plot.
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### Saliency Maps
Plotting the Saliency Map of the model. For example:

```yaml
Dips_saliency_b_WP77_passed_ttbar:
  type: "saliency"
  data_set_name: "ttbar"
  plot_settings:
    title: "Saliency map for $b$ jets from \n $t\\bar{t}$ who passed WP = 77% \n with exactly 8 tracks"
    target_beff: 0.77
    jet_flavour: "cjets"
    PassBool: True
    FlipAxis: True
    UseAtlasTag: True # Enable/Disable AtlasTag
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets"
    yAxisAtlasTag: 0.925
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `title` | String | Necessary | Title which will be on top above the plot itself. |
| `target_beff` | Float | Necessary | The WP which needs to be passed/not passed. |
| `jet_flavour` | Int | Necessary | Class which is to be plotted. |
| `PassBool` | Bool | Necessary | Decide if the b-tagging discriminant of the jets, which will be used, needs to be above the WP cut value or not. |
| `FlipAxis` | Bool | Optional | If True, the y and x axis will be switched. Useful for presentation plots. True: landscape format. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### pT vs Efficiency
Plot the b efficiency/c-rejection/light-rejection against the pT. For example:

```yaml
Dips_pT_vs_beff:
  type: "pT_vs_eff"
  models_to_plot:
    dips:
      data_set_name: "ttbar"
      label: "DIPS"
      tagger_name: "dips"
  plot_settings:
    bin_edges: [20, 30, 40, 60, 85, 110, 140, 175, 250]
    flavour: "cjets"
    class_labels: ["ujets", "cjets", "bjets"]
    main_class: "bjets"
    WP: 0.77
    WP_Line: True
    binomialErrors: True
    Fixed_WP_Bin: False
    SWP_Comparison: False
    figsize: [7, 5]
    Log: False
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample"
    yAxisAtlasTag: 0.9
    yAxisIncrease: 1.3
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `label` | String | Necessary | Label for the Legend in the plot. |
| `tagger_name` | String | Necessary | Name of the tagger which is to be plotted. |
| `evaluation_file` | String | Optional | Add a path to a evaluation file here. This file will be used for the model instead of the one defined at the top. The given `data_set_name` must be in the file! |
| `SWP_label` | String | Optional | String label of the Same Working Point (SWP) method. All models with the same SWP label use the same Working point cut value. |
| `bin_edges` | List | Necessary | Setting the edges of the bins. Don't forget the first/last edge! |
| `flavour` | String | Necessary | Flavour class rejection which is to be plotted. |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `main_class` | String | Class which is to be tagged. |
| `WP` | Float | Necessary | Float of the working point that will be used. |
| `WP_line` | Float | Optional | Print a horizontal line at this value efficiency. |
| `binomialErrors` | Bool | Optional | Plot binomial errors to plot. |
| `Fixed_WP_Bin` | Bool | Optional | Calculate the WP cut on the discriminant per bin. |
| `SWP_Comparison` | Bool | Optional | Use the same cut value on the discriminant for all models with the same SWP_label. Not works with Fixed_WP_Bin True.
| `figsize` | List | Optional |A list of the width and height of the plot.
| `Log` | Bool | Optional | Decide if the y axis is plotted as logarithmic or not.
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. Don't add fc value here! Its automatically added also the WP. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `ymin` | Float | Optional | Set the y axis minimum. Leave empty (=None) for automatically set border. |
| `ymax` | Float | Optional | Set the y axis maximum. Leave empty (=None) for automatically set border. |
| `alpha` | Float | Optional | The Alpha value of the plots. |

#### Variable vs Efficiency
Plot the efficiencies of all flavours versus any variable (not just pT). The variables must be included in the results h5 files from the evaluation step.

```yaml
eff_vs_pt:
  type: "ROCvsVar"
  evaluation_file:
  data_set_name: "zpext_r21"
  data_set_for_cut_name: "ttbar_r21"
  recompute: False
  class_labels: ["ujets", "cjets", "bjets", "taujets"]
  main_class: "bjets"
  tagger_name: "DL1"
  frac_values: {
    "cjets": 0.018,
    "ujets": 0.882,
    "taujets": 0.1,
  }
  variable: pt
  flat_eff: True
  efficiency: 70
  cut_value:
  max_variable: 1500000
  min_variable: 10000
  nbin: 100
  var_bins: [20, 30, 40, 50, 75, 100, 150, 250]
  xticksval: [20, 50, 100, 150, 200, 250]
  xticks: ["", "$50$", "$100$", "$150$", "$200$", "$250$"]
  plot_settings:
    xlabel: "$p_T$ [GeV]"
    minor_ticks_frequency: 10
    UseAtlasTag: True
    AtlasTag: "Internal"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$"
    ThirdTag: "Flat efficiency DL1r"
    Log: True

```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|

| `data_set_name` | string | N | The dataset to use from the dataframe as specified in evaluation. |
| `data_set_for_cut_name` | string | Optional | The dataset to use to compute the cut_value defining the working point. |
| `recompute` | bool | Optional | Whether to recompute the score or load it (useful if different fractions). If recomputing, `flavour_fractions` must be defined. Will recompute by default. |
| `class_labels` | List | Necessary | List of class labels that were used in the preprocessing/training. They must be the same as in preprocessing! Order is important! |
| `main_class` | string | Necessary | The main class label to tag. Must be in `class_labels`. |
| `tagger_name` | string | Necessary | The name of the tagger to use (will be composed with probability of the flavour to retrieve the probability: e.g., "DL1" leads to "DL1_pb", "DL1_pc", ...). |
| `frac_values` | dictionary | Optional | The flavour fractions used. Only necessary if recomputing the scores. Added to the tag if not None. |
| `variable` |  String | Necessary | A variable contained in the h5 result file from `evaluate.py` (e.g., "pt"). <br /> To include any non-standard variable in this h5, include them in the list of the parameter `add_variables_eval` in the training configuration ([example](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml#L69)). <br /> Note! pt variable is automatically transformed in GeV (divide by 1000)! |
| `flat_eff` | bool | Optional | Whether to use a flat b-tag b-jet efficiency per variable bin or a global one. |
| `efficiency` | int | Optional | The desired b-jet b-tag efficiency in percent |
| `cut_value` | None or float | Optional | Enforce a specific cut value on the b-discriminant. Can be used to apply R21 cut values to R22. `flat_eff` must be False. |
| `max_variable` | float | Optional | The maximum value to be considered for variable in the binning <br /> Note! For pt, value of variable is in GeV. |
| `max_variable` | float | Optional | The minimum value to be considered for variable in the binning <br /> Note! For pt, value of variable is in GeV. |
| `nbin` | int | Optional | The number of bin to be considered for variable in the binning |
| `var_bins` | List of float | Optional | The bins to use for variable. Overrides the three parameters above |
| `xticksval` | List of float | Optional | Main ticks positions. <br /> Note! For pt, values are in GeV. |
| `xticks` | List of  String | Optional | The ticks to write. Requires `xticksval` to work. |
| `xlabel` |  String | Optional | To write as name of the x label |
| `minor_ticks_frequency` | Int | Optional | Frequency of the minor ticks to draw <br /> Note! For pt, values are in GeV. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line of text right below the 'ATLAS' and the AtlasTag. Don't add fc value nor efficiency here! They are automatically added to the third tag. |
| `ThirdTag` | String | Optional | Write this text on the upper left corner. Usually meant to indicate efficiency format (global or flat) and the tagger used (DIPS, DL1r, ...). The fc value and the b-jet efficiency are automatically added to this tag. |
| `Log` | bool | Optional | Whether to  put the y-axis in log-scale. |

#### Variable vs Efficiency Comparison
Plot the efficiencies of each flavours versus any variable (not just pT) for all listed models. The variables must be included in the results h5 files from the evaluation step.

```yaml
eff_vs_pt_small:
  type: "ROCvsVar_comparison"
  tagger_name: "DL1"
  frac_values: {
    "cjets": 0.018,
    "ujets": 0.882,
    "taujets": 0.1,
  }
  class_labels: ["ujets", "cjets", "bjets", "taujets"]
  main_class: "bjets"
  variable: pt
  models_to_plot:
    model1:
      data_set_name: "ttbar_r22"
      data_set_for_cut_name: "ttbar_r21"
      label: "Model 1"
    model2:
      evaluation_file: path_to_result_other.h5
      data_set_name: "ttbar_r22"
      data_set_for_cut_name: "ttbar_r21"
      tagger_name: "DL1"
      class_labels: ["ujets", "cjets", "bjets", "taujets"]
      label: "Model 2"
      cut_value: None
      frac_values: {
        "cjets": 0.018,
        "ujets": 0.882,
        "taujets": 0.1,
      }
  flat_eff: False
  efficiency: 70
  max_variable: 1500000
  min_variable: 10000
  nbin: 100
  var_bins: [20, 30, 40, 50, 75, 100, 150, 250]
  xticksval: [20, 50, 100, 150, 200, 250]
  xticks: ["", "$50$", "$100$", "$150$", "$200$", "$250$"]
  plot_settings:
    xlabel: "$p_T$ [GeV]"
    minor_ticks_frequency: 10
    UseAtlasTag: True
    AtlasTag: "Internal"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$"
    ThirdTag: "Flat efficiency DL1r"
    Log: True
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `tagger_name` | string | Necessary | The name of the tagger to use (will be composed with probability of the flavour to retrieve the probability: e.g., "DL1" leads to "DL1_pb", "DL1_pc", ...). If one is defined above the models, will serve as default. Redefining it for a specific model forces this model to use the specified tagger. |
| `data_set_name` | string | Necessary (for each model) | The dataset to use from the dataframe as specified in evaluation. |
| `data_set_for_cut_name` | string | Optional (for each model) | The dataset to use to compute the cut_value defining the working point. |
| `recompute` | bool | Optional | Whether to recompute the score or load it (useful if different fractions). If recomputing, `flavour_fractions` must be defined for the model considered. Setting recompute above models set it as default. Will recompute by default. |
| `class_labels` | List | Necessary (for each model or above all models) | List of class labels that were used in the preprocessing/training. They must be the same as in preprocessing! Order is important! |
| `main_class` | string | Necessary | The main class label to tag. Must be in `class_labels`. |
| `frac_values` | dictionary | Optional (for each model or above all models, then serving as default) | The flavour fractions used. Necessary if recomputing the scores for a tagger. If defined above model, will be added to the tag if not None.|
| `variable` |  String | Necessary | A variable contained in the h5 result file from `evaluate.py` (e.g., "pt"). <br /> To include any non-standard variable in this h5, include them in the list of the parameter `add_variables_eval` in the training configuration ([example](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml#L69)). <br /> Note! pt variable is automatically transformed in GeV (divide by 1000)! |
| `flat_eff` | bool | Optional | Whether to use a flat b-tag b-jet efficiency per variable bin or a global one. |
| `efficiency` | int | Optional | The desired b-jet b-tag efficiency in percent |
| `cut_value` | None or float | Optional (for each model) | Enforce a specific cut value on the b-discriminant. Can be used to apply R21 cut values to R22. `flat_eff` must be False. |
| `max_variable` | float | Optional | The maximum value to be considered for variable in the binning <br /> Note! For pt, value of variable is in GeV. |
| `max_variable` | float | Optional | The minimum value to be considered for variable in the binning <br /> Note! For pt, value of variable is in GeV. |
| `nbin` | int | Optional | The number of bin to be considered for variable in the binning |
| `var_bins` | List of float | Optional | The bins to use for variable. Overrides the three parameters above |
| `xticksval` | List of float | Optional | Main ticks positions. <br /> Note! For pt, values are in GeV. |
| `xticks` | List of  String | Optional | The ticks to write. Requires `xticksval` to work. |
| `xlabel` |  String | Optional | To write as name of the x label |
| `minor_ticks_frequency` | Int | Optional | Frequency of the minor ticks to draw <br /> Note! For pt, values are in GeV. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line of text right below the 'ATLAS' and the AtlasTag. Don't add fc value nor efficiency here! They are automatically added to the third tag. |
| `ThirdTag` | String | Optional | Write this text on the upper left corner. Usually meant to indicate efficiency format (global or flat) and the tagger used (DIPS, DL1r, ...). The fc value and the b-jet efficiency are automatically added to this tag. |
| `Log` | bool | Optional | Whether to  put the y-axis in log-scale. |

#### Scanning fractions - DEPRECATED
DEPRECATED: For DL1 with taus, the evaluation step of `evaluate.py` generated an extra h5 file giving the c/b, light, and tau rejection as a function of the c/b-fraction and the tau fraction (this evaluation is no longer performed). To produce the plot associated to this information (2d heatmap of rejection for the two flavour fractions), add (for example) this to the plotting config:

```yaml
plot_scan_frac_tau:
  type: "FracScan"
  evaluation_file:  # from evaluation
  data_set_name: "ttbar"
  label: "umami_taurej"
  xlabel: "fraction_taus"
  ylabel: "fraction_c"
  plot_settings:
    UseAtlasTag: True
    AtlasTag: "Internal Simulations"
    SecondTag: "DL1r, $\\sqrt{s}$ = 13 TeV, $t\bar{t}$"
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `label` | String | Necessary | Indicate the rejection to plot (u for light). Choose from: <br /> For b-tagging: `["umami_crej", "umami_urej", "umami_taurej"]`, <br /> For c-tagging: `["umami_brejC", "umami_urejC", "umami_taurejC"]` |
| `xlabel` | String | Optional | The label to use for the xscale of the plot (normally taus) |
| `ylabel` | String | Optional | The label to use for the yscale of the plot. <br /> Either `"fraction_c"`or `"fraction_b"` for c- and b-tagging |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. Don't add fc value here! Its automatically added also the WP. |

### Executing the Script
The script can be executed by using the following command:

```bash
plotting_umami.py -c ${EXAMPLES}/plotting_umami_config_dips.yaml -o dips_eval_plots
```

The `-o` option defines the name of the output directory. It will be added to the model folder where also the results are saved. Also you can set the output filetype by using the `-f` option. For example:

```bash
plotting_umami.py -c ${EXAMPLES}/plotting_umami_config_dips.yaml -o dips_eval_plots -f png
```

The output plots will be .png now. Standard is pdf.