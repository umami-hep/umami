# Plotting the evaluation results
The evaluation results can be plotted using different functions. There is the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py), [plotting_epoch_performance](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_epoch_performance.py) and the [plot_input_variables.py](). Each plotting script is explained in its dedicated section.

## plotting_umami.py
The plotting_umami.py is used to plot the results of the evaluation script. Different plots can be produced with it which are fully customizable. All plots that are defined in the `plotting_umami_config_X.yaml`. The `X` defines the tagger here but its just a name. All config files are usable with the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py) script.

### Yaml Config File
**Important: The indentation in this .yaml is important due to the way the files are read by the script.**   
A fully written one can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/examples/plotting_umami_config_dips.yaml).   

The config file starts with the `Eval_parameters`. Here the `Path_to_models_dir` is set, where the models are saved. Also the `model_name` and the `epoch` which is to be plotted is set. A boolean parameter can be set here to indicate whether to plot with taus included (requires to have train and preprocessed with taus). For example, this can look like this:

```yaml
Eval_parameters:
  Path_to_models_dir: /work/ws/nemo/fr_af1100-Training-Simulations-0/b-Tagging/packages/umami/umami
  model_name: dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full
  epoch: 59
  bool_use_taus: False
```

In the different available plots, there are options that are available in mostly all of them. So they will be explained next. For specific options, look at the comment in the section of the plot.   

| Options | Explanation |
|---------|-------------|
| `Name_of_the_plot` | All plots start with no indentation and the name of plot. This will be the output name of the plot file and has no impact on the plot itself. |
| `type` | This option specifies the plot function that is used. |
| `data_set_name` | Decides which evaluated dataset (or file) is used. This `data_set_name` are set in the `train_config` yaml file which is used in the evaluation of the model. There the different files are getting their own `data_set_name` which needs to be the same as here! |
| `models_to_plot` | In the comparison plots, the models which are to be plotted needs to be defined in here. You can add as many models as you want. For example this can be used to plot the results of the different taggers in one plot (e.g. for score comparison or ROC curves). The different models can be assisted with `evaluation_file` to point to the results file you have created with `evaluate_model.py`. e.g.`evaluation_file: YOURMODEL/results/results-rej_per_eff-229.h5` |
| `plot_settings` | In this section, all optional plotting settings are defined. They don't need to be defined but you can. For the specific available options in each function, look in the corresponding section. |

For plotting, these different plots are available:

#### Confusion Matrix
Plot a confusion matrix. For example:

```yaml
confusion_matrix_Dips_ttbar:
  type: "confusion_matrix"
  data_set_name: "ttbar"
  prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
```
| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `prediction_labels` | List |  Necessary |A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`. <br /> For DIPS: `["dips_pb", "dips_pc", "dips_pu"]` <br /> For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]` <br /> For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]` <br /> For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]` <br /> For the retrained DL1r (using `EvaluateModelDL1`): `["pb", "pc", "pu"]` <br /> If taus are included (only retrained DL1r so far): `["pb", "pc", "pu", "ptau"]` |

#### Scores
Plotting the b-tagging discriminant scores for the different jet flavors. For example:

```yaml
scores_Dips_ttbar:
  type: "scores"
  data_set_name: "ttbar"
  prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
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
| `prediction_labels` | List | Necessary | A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`. <br /> For DIPS: `["dips_pb", "dips_pc", "dips_pu"]` <br /> For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]` <br /> For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]` <br /> For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]` <br /> For the retrained DL1r (using `EvaluateModelDL1`): `["pb", "pc", "pu"]` <br /> If taus are included (only retrained DL1r so far): `["pb", "pc", "pu", "ptau"]` |
| `WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
|`nBins` | Int | Optional | Number of bins that are used. |
|`yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
|`UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
|`AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
|`SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
|`yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### Scores Comparison
Plotting the b-tagging discriminant scores for the different jet flavors for different models in the same plot. For example:

```yaml
scores_Dips_ttbar_comparison:
  type: "scores_comparison"
  data_set_name: "ttbar"
  prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
  models_to_plot:
    dips_r21:
      data_set_name: "ttbar"
      label: "R21"
    dips_r22:
      data_set_name: "ttbar_comparison"
      label: "R22"
  plot_settings:
    WorkingPoints: [0.60, 0.70, 0.77, 0.85]
    nBins: 50
    yAxisIncrease: 1.4
    figsize: [8, 6]
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample"
    yAxisAtlasTag: 0.9
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `prediction_labels` | List | Necessary | A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`. <br /> For DIPS: `["dips_pb", "dips_pc", "dips_pu"]` <br /> For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]` <br /> For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]` <br /> For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]` <br /> For the retrained DL1r (using `EvaluateModelDL1`): `["pb", "pc", "pu"]` <br /> If taus are included (only retrained DL1r so far): `["pb", "pc", "pu", "ptau"]` |
|`dips_r21` | None | Necessary | Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything. |
|`label` | String | Necessary | Label for the Legend in the plot. |
|`WorkingPoints` | List | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |
|`nBins` | Int | Optional | Number of bins that are used. |
|`yAxisIncrease` | Float | Optional | Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
|`figsize` | List | Optional | A list of the width and hight of the plot. |
|`UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
|`AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
|`SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
|`yAxisAtlasTag` | Float | Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### ROC Curves
Plotting the ROC Curves of the rejection rates against the b-tagging efficiency. For example:

```yaml
Dips_light_flavour_ttbar:
  type: "ROC"
  models_to_plot:
    rnnip_r21:
      data_set_name: "ttbar"
      label: "Recommended RNNIP"
      df_key: "rnnip_urej"
    dips_r21:
      data_set_name: "ttbar"
      label: "DIPS"
      df_key: "dips_urej"
  plot_settings:
    xlabel: "$b$-jet efficiency"
    ylabel: "light"
    binomialErrors: True
    xmin: 0.5
    ymax: 1000000
    colors:
    figsize: [7, 6]
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
| `df_key` | String | Necessary | Decide which rejection is plotted. The structure is like this: `model_Xrej`. The `X` defines the wanted rejection. `u` for light-, `c` for c-rejection, `t` for tau-rejection. Note: for DL1 only, the light-, tau-, and b- rejection from c-jets are also supported. The structure is then: `model_XrejC`, with `X` taking as value `u`, `t`, or `b`. |
| `xlabel` | String | Optional | Set the xlabel.
| `ylabel` | String | Optional | Set the ylabel of the X-rejection. For example: 'c' will output `c-flavor rejection`. |
| `binomialErrors` | Bool | Optional | Plot binomial errors to plot. |
| `xmin` | Float | Optional | Set the minimum b efficiency in the plot (which is the xmin limit). |
| `ymax` | Float | Optional | The maximum y axis. |
| `colors` | List | Optional | For each model in `models_to_plot`, there must be a color in this list. If leave it empty (=None), the colors will be set automatically. | 
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
  plot_settings:
    data_set_name: "ttbar"
    title: "Saliency map for $b$ jets from \n $t\\bar{t}$ who passed WP = 77% \n with exactly 8 tracks"
    target_beff: 0.77
    jet_flavour: 2
    PassBool: True
    FlipAxis: True
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets"
    yAxisAtlasTag: 0.925
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `title` | String | Necessary | Title which will be on top above the plot itself. |
| `target_beff` | Float | Necessary | The WP which needs to be passed/not passed. |
| `jet_flavour` | Int | Necessary | The jet flavor that will be plotted. Current possibilites: 2: b, 1: c, 0: light. |
| `PassBool` | Bool | Necessary | Decide if the b-tagging discriminant of the jets, which will be used, needs to be above the WP cut value or not. |
| `FlipAxis` | Bool | Optional | If True, the y and x axis will be switched. Usefull for presenation plots. True: landscape format. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |

#### pT vs Efficiency
Plot the b efficiency/c-rejection/light-rejection against the pT. For example:

```yaml
pT_vs_beff_zpext:
  type: "pT_vs_eff"
  models_to_plot:
    rnnip_r21:
      data_set_name: "zpext"
      label: "Recommended RNNIP"
      prediction_labels: ["rnnip_pb", "rnnip_pc", "rnnip_pu"]
      evaluation_file:
      SWP_label: "rnnip"
    dips_r21:
      data_set_name: "zpext"
      label: "DIPS"
      prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
      evaluation_file:
      SWP_label: "dips"
  plot_settings:
    bin_edges: [200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000]
    flavor: 2
    WP: 0.77
    WP_Line: True
    Fixed_WP_Bin: False
    binomialErrors: True
    Same_WP_Cut_Comparison: True
    figsize: [7, 5]
    Log: False
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow Jets,\n$Z'$ Test Sample"
    yAxisAtlasTag: 0.9
    yAxisIncrease: 1.3
    ymin:
    ymax:
    alpha: 0.7
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `dips_r21` | None | Necessary | Name of the model which is to be plotted. Don't affect anything. Just for you. You can change dips_r21 to anything. |
| `label` | String | Necessary | Label for the Legend in the plot. |
| `prediction_labels` | List |  Necessary |A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`. <br /> For DIPS: `["dips_pb", "dips_pc", "dips_pu"]` <br /> For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]` <br /> For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]` <br /> For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]` <br /> For the retrained DL1r (using `EvaluateModelDL1`): `["pb", "pc", "pu"]` <br /> If taus are included (only retrained DL1r so far): `["pb", "pc", "pu", "ptau"]` |
| `evaluation_file` | String | Optional | Add a path to a evaluation file here. This file will be used for the model instead of the one defined at the top. The given `data_set_name` must be in the file! |
| `SWP_label` | String | Optional | The Same WP Cut label. All models with the same `SWP_label` will get the same discriminant cut value. |
| `bin_edges` | List | Optional | The pT bin edges that should be used. Don't forget the starting and the ending edge! |
| `flavor` | Int | Optional | Decide which eff/rej will be plotted. 2: b, 1: c, 0: light. |
| `WP` | Float | Optional | Which Working Point is used. |
| `WP_Line` | Bool | Optional | Decide if a horizontal WP line at is added or not. (Only used for beff). |
| `Fixed_WP_Bin` | Bool | Optional | If True, the b-Tagging discriminant cut value for the given WP is not calculated over all bins but separately for each bin. |
| `binomialErrors` | Bool | Optional | Plot binomial errors to plot. |
| `Same_WP_Cut_Comparison` | Bool | Optional | For all models with the same `SWP_label`, one cut value for the b-tagging discriminant is used.
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
Plot the b-tag efficiency for b, c, and light jets (+ optionaly tau jets) versus a given variable (not just pT). The variables must be included in the results h5 files from the evaluation step.

```yaml
eff_vs_pt_:
  type: "ROCvsVar"
  data_set_name: "ttbar"
  flat_eff: True 
  efficiency: 70 
  fc: 0.018
  prediction_labels: ["dips_pb", "dips_pc", "dips_pu"] 
  variable: pt   
  max_variable: 150
  min_variable: 10 
  nbin: 100 
  var_bins: [20, 30, 40, 50, 75, 100, 150, 250]
  xticksval: [20, 50, 100, 150, 200, 250]
  xticks: ["", "$50$", "$100$", "$150$", "$200$", "$250$"]
  plot_settings:
    xlabel: "$p_T$ [GeV]"
    minor_ticks_frequency: 10
    UseAtlasTag: True
    AtlasTag: "Internal"
    SecondTag: "$\sqrt{s}$ = 13 TeV, $t\bar{t}$"
    ThirdTag: "Flat efficiency DL1r"
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `flat_eff` | bool | Necessary | Whether to use a flat b-tag b-jet efficiency per variable bin or a global one. |
| `efficiency` | int | Necessary | The desired b-jet b-tag efficiency in percent |
| `fc` | float | Optional | The fc value to use |
| `prediction_labels` | List | Necessary |A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`. <br /> For DIPS: `["dips_pb", "dips_pc", "dips_pu"]` <br /> For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]` <br /> For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]` <br /> For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]` <br /> For the retrained DL1r (using `EvaluateModelDL1`): `["pb", "pc", "pu"]` <br /> If taus are included (only retrained DL1r so far): `["pb", "pc", "pu", "ptau"]` |
| `variable` |  String | Necessary | A variable contained in the h5 result file from `evaluate.py` (e.g., "pt"). <br /> To include any non-standard variable in this h5, include them in the list of the parameter `add_variables_eval` in the training configuration ([example](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml#L69)). <br /> Note! pt variable is automatically transformed in GeV (divide by 1000)! |
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

#### Scanning fractions
For DL1 with taus, the evaluation step of `evaluate.py` generates an extra h5 file giving the c/b, light, and tau rejection as a function of the c/b-fraction and the tau fraction. To produce the plot associated to this information, add (for example) this to the plotting config:

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

The `-o` option defines the name of the output directory. It will be added to the model folder where also the results are saved.

## Plot Input Variables
The input variables for different files can also be plotted using the `plot_input_variables.py` script. Its also steered by a yaml file. An example for such a file can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml). The structure is close to the one from `plotting_umami` but still a little bit different.   

### Yaml File
The following options are available (and need to be set). At the top of the file the `Eval_parameters` need to be set. Here you can define the number of jets that are used and also the variable dict, where all the variables that are available are saved.

```yaml
Eval_parameters:
  # Number of jets which are used
  nJets: 3000000

  # Path to variable dict
  var_dict: /work/ws/nemo/fr_af1100-Training-Simulations-0/b-Tagging/packages/umami/umami/configs/Dips_Variables.yaml
```

#### Number of Tracks per Jet
The number of tracks per jet can be plotted for all different files. This can be given like this:

```yaml
nTracks_ttbar_loose:
  variables: "tracks"
  folder_to_save: input_vars_trks_ttbar_loose
  nTracks: True
  Datasets_to_plot:
    R21:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p3985/user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow_looser-track_selection.2020-07-01-T193555-R26654_output.h5/*
      label: "R21 Loose"
    R22:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p4441/user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4441.EMPFlow_loose.2021-04-20-T171733-R21211_output.h5/*
      label: "R22 Loose"
  plot_settings:
    Log: True
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3M Jets"
    yAxisAtlasTag: 0.925
    yAxisIncrease: 2
    figsize: [7, 5]
    Ratio_Cut: [0.5, 2]
  flavors:
    b: 5
    c: 4
    u: 0
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `nTracks_ttbar_loose` | String | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | String | Necessary | Must be set to "tracks" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | String | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. |
| `nTracks` | Bool | Necessary | MUST BE TRUE HERE! Decide if the Tracks per Jets are plotted or the input variable. |
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | String | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achive the number of jets given in the `Eval_parameters`. |
| `label` | String | Necessary | Plot label for the plot legend. |
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |
| `Log` | Bool | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional | Two element list that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | List | Optional | Two element list that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
| `flavors` | None | Necessary | Here starts the flavors that are about to be plotted. Each entry name, for example `b` is also the label which is to be added to the plot legend. The number gives the particle ID of the wanted particle. |

#### Input Variables Tracks
To plot the track input variables, the following options are used.

```yaml
input_vars_trks_ttbar_loose_ptfrac:
  variables: "tracks"
  folder_to_save: input_vars_trks_ttbar_loose
  Datasets_to_plot:
    R21:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p3985/user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow_looser-track_selection.2020-07-01-T193555-R26654_output.h5/*
      label: "R21 Loose"
    R22:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p4441/user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4441.EMPFlow_loose.2021-04-20-T171733-R21211_output.h5/*
      label: "R22 Loose"
  plot_settings:
    sorting_variable: "ptfrac"
    n_Leading: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Log: True
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3M Jets"
    yAxisAtlasTag: 0.925
    yAxisIncrease: 2
    figsize: [7, 5]
    Ratio_Cut: [0.5, 1.5]
  binning:
    IP3D_signed_d0_significance: 100
    IP3D_signed_z0_significance: 100
    numberOfInnermostPixelLayerHits: [0, 4, 1]
    numberOfNextToInnermostPixelLayerHits: [0, 4, 1]
    numberOfInnermostPixelLayerSharedHits: [0, 4, 1]
    numberOfInnermostPixelLayerSplitHits: [0, 4, 1]
    numberOfPixelSharedHits: [0, 4, 1]
    numberOfPixelSplitHits: [0, 9, 1]
    numberOfSCTSharedHits: [0, 4, 1]
    ptfrac: [0, 5, 0.05]
    dr: 100
    numberOfPixelHits: [0, 11, 1]
    numberOfSCTHits: [0, 19, 1]
    btagIp_d0: 100
    btagIp_z0SinTheta: 100
  flavors:
    b: 5
    c: 4
    u: 0
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `input_vars_trks_ttbar_loose_ptfrac` | String | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | String | Necessary | Must be set to "tracks" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | String | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. |
| `nTracks` | Bool | Necessary | To plot the input variable distributions, this must be `False`. |  
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | String | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achive the number of jets given in the `Eval_parameters`. |
| `label` | String | Necessary | Plot label for the plot legend. |
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |
| `sorting_variable` | String | Optional | Variable Name to sort after. |
| `n_Leading` | List | Optional | List of the x leading tracks. If `None`, all tracks will be plotted. If `0` the leading tracks sorted after `sorting variable` will be plotted. You can add like `None`, `0` and `1` for example and it will plot all 3 of them, each in their own folders with according labeling. This must be a list! Even if there is only one option given. |  
| `Log` | Bool | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional | Two element list that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | List | Optional | If you add more then two models to plot, the comparison function is used with a small ratio plot at the bottom. Two element list that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
| `binning` | None | Necessary | Here starts the binning for each variable. If you give a `int`, there will be so much equal distant bins. You can also give a three element list which will be used in the `numpy.arange` function. The first element is start, second ist stop and third is the step width. The so aranged numbers are bin edges not bins! If `None` is given, the standard value is `100`. If a variable is not defined here, its not plotted. |
| `flavors` | None | Necessary | Here starts the flavors that are about to be plotted. Each entry name, for example `b` is also the label which is to be added to the plot legend. The number gives the particle ID of the wanted particle. |

#### Input Variables Jets
To plot the jet input variables, the following options are used.

```yaml
input_vars_ttbar_loose:
  variables: "jets"
  folder_to_save: input_vars_ttbar_loose
  Datasets_to_plot:
    R21:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p3985/user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow_looser-track_selection.2020-07-01-T193555-R26654_output.h5/*
      label: "R21 Loose"
    R22:
      files: /work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p4441/user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4441.EMPFlow_loose.2021-04-20-T171733-R21211_output.h5/*
      label: "R22 Loose"
  special_param_jets:
    IP2D_cu:
      lim_left: -30
      lim_right: 30                        
    IP2D_bu:   
      lim_left: -30
      lim_right: 30                          
    IP2D_bc:  
      lim_left: -30
      lim_right: 30                           
    IP3D_cu:   
      lim_left: -30
      lim_right: 30                          
    IP3D_bu:  
      lim_left: -30
      lim_right: 30                         
    IP3D_bc:     
      lim_left: -30
      lim_right: 30
    SV1_NGTinSvx:
      lim_left: 0
      lim_right: 19 
    JetFitterSecondaryVertex_nTracks:
      lim_left: 0
      lim_right: 17  
    JetFitter_nTracksAtVtx:
      lim_left: 0
      lim_right: 19 
    JetFitter_nSingleTracks:
      lim_left: 0
      lim_right: 18
    JetFitter_nVTX:
      lim_left: 0
      lim_right: 6
    JetFitter_N2Tpair:
      lim_left: 0
      lim_right: 200
  binning:
    IP2D_cu                           : 100
    IP2D_bu                           : 100
    IP2D_bc                           : 100
    IP2D_isDefaults                   : 2
    IP3D_cu                           : 100
    IP3D_bu                           : 100
    IP3D_bc                           : 100
    IP3D_isDefaults                   : 2
    JetFitter_mass                    : 100       
    JetFitter_energyFraction          : 100
    JetFitter_significance3d          : 100
    JetFitter_deltaR                  : 100
    JetFitter_nVTX                    : 7
    JetFitter_nSingleTracks           : 19
    JetFitter_nTracksAtVtx            : 20
    JetFitter_N2Tpair                 : 201
    JetFitter_isDefaults              : 2
    JetFitterSecondaryVertex_minimumTrackRelativeEta: 11
    JetFitterSecondaryVertex_averageTrackRelativeEta: 11
    JetFitterSecondaryVertex_maximumTrackRelativeEta: 11
    JetFitterSecondaryVertex_maximumAllJetTrackRelativeEta : 11
    JetFitterSecondaryVertex_minimumAllJetTrackRelativeEta : 11
    JetFitterSecondaryVertex_averageAllJetTrackRelativeEta : 11
    JetFitterSecondaryVertex_displacement2d : 100
    JetFitterSecondaryVertex_displacement3d : 100
    JetFitterSecondaryVertex_mass           : 100
    JetFitterSecondaryVertex_energy         : 100
    JetFitterSecondaryVertex_energyFraction : 100
    JetFitterSecondaryVertex_isDefaults     : 2
    JetFitterSecondaryVertex_nTracks        : 18
    pt_btagJes                        : 100
    absEta_btagJes                    : 100
    SV1_Lxy                           : 100
    SV1_N2Tpair                       : 8
    SV1_NGTinSvx                      : 20
    SV1_masssvx                       : 100
    SV1_efracsvx                      : 100
    SV1_significance3d                : 100
    SV1_deltaR                        : 10
    SV1_L3d                           : 100
    SV1_isDefaults                    : 2
    rnnip_pb                          : 50
    rnnip_pc                          : 50
    rnnip_pu                          : 50
  plot_settings:
    Log: True
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3M Jets"
    yAxisAtlasTag: 0.925
    yAxisIncrease: 2
    figsize: [7, 5]
  flavors:
    b: 5
    c: 4
    u: 0
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `input_vars_trks_ttbar_loose_ptfrac` | String | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | String | Necessary | Must be set to "jets" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | String | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. | 
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | String | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achive the number of jets given in the `Eval_parameters`. |
| `label` | String | Necessary | Plot label for the plot legend. |
| `special_param_jets` | None | Necessary | Here starts the special x axis limits for a variable. If you want to set the x range by hand, add the variable here and also the `lim_left` for xmin and `lift_right` for xmax. |
| `binning` | None | Necessary | Here starts the binning for each variable. If you give a `int`, there will be so much equal distant bins. You can also give a three element list which will be used in the `numpy.arange` function. The first element is start, second ist stop and third is the step width. The so aranged numbers are bin edges not bins! If `None` is given, the standard value is `100`. Variables that are not in here are not plotted! |
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |  
| `Log` | Bool | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | Bool | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | String | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | String | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | Float |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | Float | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | List | Optional | Two element list that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | List | Optional | If you add more then two models to plot, the comparison function is used with a small ratio plot at the bottom. Two element list that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
| `flavors` | None | Necessary | Here starts the flavors that are about to be plotted. Each entry name, for example `b` is also the label which is to be added to the plot legend. The number gives the particle ID of the wanted particle. |
