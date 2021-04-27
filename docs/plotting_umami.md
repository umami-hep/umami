# Plotting the evaluation results
The evaluation results can be plotted using different functions. There is the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py), [plotting_epoch_performance](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_epoch_performance.py) and the [plot_input_variables.py](). Each plotting script is explained in its dedicated section.

## plotting_umami.py
The plotting_umami.py is used to plot the results of the evaluation script. Different plots can be produced with it which are fully customizable. All plots that are defined in the `plotting_umami_config_X.yaml`. The `X` defines the tagger here but its just a name. All config files are usable with the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py) script.

### Yaml Config File
**Important: The indentation in this .yaml is important due to the way the files are read by the script.**   
A fully written one can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/examples/plotting_umami_config_dips.yaml).   
The config file starts with the `Eval_parameters`. Here the `Path_to_models_dir` is set, where the models are saved. Also the `model_name` and the `epoch` which is to be plotted is set. For example, this can look like this:

```yaml
Eval_parameters:
  Path_to_models_dir: /work/ws/nemo/fr_af1100-Training-Simulations-0/b-Tagging/packages/umami/umami
  model_name: dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full
  epoch: 59
```

In the different available plots, there are options that are available in mostly all of them. So they will be explained next. For specific options, look at the comment in the section of the plot.   

`Name_of_the_plot`: All plots start with no indentation and the name of plot. This will be the output name of the plot file and has no impact on the plot itself.   

`type`: This option specifies the plot function that is used.   

`data_set_name`: Decides which evaluated dataset (or file) is used. Options are `ttbar` for the `test_file`, `zpext` for the `add_test_file`, `ttbar_comparison` for the `comparison_file` and `zpext_comparison` for the `add_comparison_file`.   

`models_to_plot`: In the comparison plots, the models which are to be plotted needs to be defined in here. You can add as many models as you want. For example this can be used to plot the results of the different taggers in one plot.   

`plot_settings`: In this section, all optional plotting settings are defined. They don't need to be defined but you can. For the specific available options in each function, look in the corresponding section.

For plotting, these different plots are available:

#### Confusion Matrix
Plot a confusion matrix. For example:

```yaml
confusion_matrix_Dips_ttbar:
  type: "confusion_matrix"
  data_set_name: "ttbar"
  prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
```
`prediction_labels`, List: A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`.   
For DIPS: `["dips_pb", "dips_pc", "dips_pu"]`   
For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]`   
For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]`   
For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]`  

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

`prediction_labels`, List: A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`.   
For DIPS: `["dips_pb", "dips_pc", "dips_pu"]`   
For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]`   
For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]`   
For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]`   

`WorkingPoints`, List: The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP.   

`nBins`, Int: Number of bins that are used.   

`yAxisIncrease`, Float: Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot.   

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag.   

`yAxisAtlasTag`, Float: y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner).

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
`prediction_labels`, List: A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`.   
For DIPS: `["dips_pb", "dips_pc", "dips_pu"]`   
For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]`   
For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]`   
For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]`   

`dips_r21`, None: Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything.   

`label`, String: Label for the Legend in the plot.   

`WorkingPoints`, List: The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP.   

`nBins`, Int: Number of bins that are used.   

`yAxisIncrease`, Float: Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot.   

`figsize:`, List: A list of the width and hight of the plot.

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag.   

`yAxisAtlasTag`, Float: y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner).

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
`dips_r21`, None: Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything.   

`label`, String: Label for the Legend in the plot.   

`df_key`, String: Decide which rejection is plotted. The structure is like this: `model_Xrej`. The `X` defines the wanted rejection. `u` for light-, `c` for c-rejection.   

`ylabel:`, String: Set the ylabel of the X-rejection. For example: 'c' will output `c-flavor rejection`.   

`binomialErrors`, Bool: Plot binomial errors to plot.   

`xmin`, Float: Set the minimum b efficiency in the plot (which is the xmin limit).   

`ymax`, Float: The maximum y axis.   

`colors`, List: For each model in `models_to_plot`, there must be a color in this list. If leave it empty (=None), the colors will be set automatically.    

`WorkingPoints`, List: The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP.   

`yAxisIncrease`, Float: Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot.   

`figsize:`, List: A list of the width and hight of the plot.

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag.   

`yAxisAtlasTag`, Float: y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner).

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
`title`, String: Title which will be on top above the plot itself.   

`target_beff`, Float: The WP which needs to be passed/not passed.   

`jet_flavour`, Int: The jet flavor that will be plotted. Current possibilites: 2: b, 1: c, 0: light.   

`PassBool`, Bool: Decide if the b-tagging discriminant of the jets, which will be used, needs to be above the WP cut value or not.    

`FlipAxis`, Bool: If True, the y and x axis will be switched. Usefull for presenation plots. True: landscape format.   

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag.   

`yAxisAtlasTag`, Float: y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner).

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
    dips_r21:
      data_set_name: "zpext"
      label: "DIPS"
      prediction_labels: ["dips_pb", "dips_pc", "dips_pu"]
  plot_settings:
    bin_edges: [200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000]
    flavor: 2
    WP: 0.77
    WP_Line: True
    Fixed_WP_Bin: True
    binomialErrors: True
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
`dips_r21`, None: Name of the model which is to be plotted. Don't effect anything. Just for you. You can change dips_r21 to anything.  

`label`, String: Label for the Legend in the plot.   

`prediction_labels`, List: A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`.   
For DIPS: `["dips_pb", "dips_pc", "dips_pu"]`   
For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]`   
For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]`   
For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]`   

`bin_edges`, List: The pT bin edges that should be used. Don't forget the starting and the ending edge!   

`flavor`, Int: Decide which eff/rej will be plotted. 2: b, 1: c, 0: light.   

`WP`, Float: Which Working Point is used.   

`WP_Line`, Bool: Decide if a horizontal WP line at is added or not. (Only used for beff).   

`Fixed_WP_Bin`, Bool: If True, the b-Tagging discriminant cut value for the given WP is not calculated over all bins but seperatly for each bin.   

`binomialErrors`, Bool: Plot binomial errors to plot.   

`figsize:`, List: A list of the width and hight of the plot.

`Log`, Bool: Decide if the y axis is plotted as logarithmic or not.

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. Don't add fc value here! Its automatically added also the WP.   

`yAxisAtlasTag`, Float: y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner).   

`yAxisIncrease`, Float: Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot.

`ymin`, Float: Set the y axis minimum. Leave empty (=None) for automatically set border.   

`ymax`, Float: Set the y axis maximum. Leave empty (=None) for automatically set border.   

`alpha`, Float: The Alpha value of the plots.   

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
`flat_eff` bool: whether to use a flat b-tag b-jet efficiency per variable bin or a global one.

`efficiency`int: the desired b-jet b-tag efficiency in percent

`fc`: float (optional): the fc value to use

`prediction_labels`, List: A list of the probability outputs of a model. The order here is important! (pb, pc, pu). The different model outputs are maily build the same like `model_pX`.   
For DIPS: `["dips_pb", "dips_pc", "dips_pu"]`   
For UMAMI: `["umami_pb", "umami_pc", "umami_pu"]`   
For RNNIP: `["rnnip_pb", "rnnip_pc", "rnnip_pu"]`   
For DL1r: `["dl1r_pb", "dl1r_pc", "dl1r_pu"]`   

`variable` string: a variable contained in the h5 result file from `evaluate.py` (e.g., "pt"). 
Note! pt variable is automatically transformed in GeV (divide by 1000)!

`max_variable` float (optional): the maximum value to be considered for variable in the binning
Note! For pt, value of variable is in GeV.

`max_variable` float (optional): the minimum value to be considered for variable in the binning
Note! For pt, value of variable is in GeV.

`nbin` int (optional - default 100): the number of bin to be considered for variable in the binning

`var_bins` list of float (optional): the bins to use for variable. Overrides the three parameters above

`xticksval` list of float (optional): main ticks positions.
Note! For pt, values are in GeV.

`xticks` list of string: the ticks to write. Requires `xticksval` to work.

`xlabel` string: to write as name of the x label

`minor_ticks_frequency`: frequency of the minor ticks to draw
Note! For pt, values are in GeV.

`UseAtlasTag`, Bool: Decide if the ATLAS Tag is printed in the upper left corner of the plot or not.   

`AtlasTag`, String: The first line of text right behind the 'ATLAS'.   

`SecondTag`, String: Second line of text right below the 'ATLAS' and the AtlasTag. Don't add fc value nor efficiency here! They are automatically added to the third tag.

`ThirdTag`, String: Write this text on the upepr left corner. Usually meant to indicate efficiency format (global or flat) and the tagger used (DIPS, DL1r, ...). The fc value and the b-jet efficiency are automatically added to this tag. 

### Executing the Script
The script can be executed by using the following command:

```bash
plotting_umami.py -c ${EXAMPLES}/plotting_umami_config_dips.yaml -o dips_eval_plots
```

The `-o` option defines the name of the output directory. It will be added to the model folder where also the results are saved.
