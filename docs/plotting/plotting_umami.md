# Plotting the evaluation results
The evaluation results can be plotted using different functions. There is the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_umami.py), [plotting_epoch_performance](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/umami/plotting_epoch_performance.py) and the [plot_input_variables.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plot_input_variables.py). Each plotting script is explained in its dedicated section.

## plotting_umami.py
The plotting_umami.py is used to plot the results of the evaluation script. Different plots can be produced with it which are fully customizable. All plots that are defined in the `plotting_umami_config_X.yaml`. The `X` defines the tagger here but its just a name. All config files are usable with the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script.

### Yaml Config File
**Important: The indentation in this .yaml is important due to the way the files are read by the script.**
A fully written one can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_umami_config_dips.yaml).
The name of your freshly trained tagger, the `tagger_name` here in the config, is always the name of your model you have trained. The name is the value of `tagger` from the `nn_structure`. 

The config file starts with the `Eval_parameters`. Here the `Path_to_models_dir` is set, where the models are saved. Also the `model_name` and the `epoch` which is to be plotted is set. A boolean parameter can be §set here to add the epoch to the end of the plot name. This is `epoch_to_name`. For example, this can look like this:

```yaml
§§§examples/plotting_umami_config_dips.yaml:1:6§§§
```

In the different available plots, there are options that are available in mostly all of them. So they will be explained next. For specific options, look at the comment in the section of the plot.

| Options | Explanation |
|---------|-------------|
| `Name_of_the_plot` | All plots start with no indentation and the name of plot. This will be the output name of the plot file and has no impact on the plot itself. |
| `type` | This option specifies the plot function that is used. |
| `data_set_name` | Decides which evaluated dataset (or file) is used. This `data_set_name` are set in the `train_config` yaml file which is used in the evaluation of the model. There the different files are getting their own `data_set_name` which needs to be the same as here! |
| `class_labels` | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! (Possible entries are defined in the [global_config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/global_config.yaml))|
| `models_to_plot` | In the plots, the models which are to be plotted needs to be defined in here. You can add as many models as you want. For example this can be used to plot the results of the different taggers in one plot (e.g. for score or ROC curves). The different models can be assisted with `evaluation_file` to point to the results file you have created with `evaluate_model.py`. e.g.`evaluation_file: YOURMODEL/results/results-rej_per_eff-229.h5` |
| `plot_settings` | In this section, all optional plotting settings are defined. They don't need to be defined but you can. For the specific available options in each function, look in the corresponding section. |

In `plot_settings`, some general options can be set which are used in all of the
available plots. These are:

§§§docs/ci_assets/docstring_puma_PlotObject.md::§§§

For plotting, these different plots are available:

#### Confusion Matrix
Plot a confusion matrix. For example:

```yaml
§§§examples/plotting_umami_config_dips.yaml:142:148§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `colourbar` | `bool` | Optional | Define, if the colourbar on the side is shown or not. |

#### Probability
Plotting the DNN probability output for a specific class. For example:

```yaml
§§§examples/plotting_umami_config_dips.yaml:164:179§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | `str` | Necessary | This gives the type of plot function used. Must be `"probability"` here. |
| `prob_class` | `str` | Necessary | Class of the to be plotted probability. |
| `dips_r22` | `None` | Necessary | Internal naming of the model. This will not show up anywhere, but it must be unique! You can define multiple of these models. All of them will be plotted. The baseline for the ratio is the first model defined here. |
| `data_set_name` | `str` | Necessary | Name of the dataset that is used. This is the name of the test_file which you want to use. |
| `label` | `str` | Necessary | Legend label of the model. |
| `tagger_name` | `str` | Necessary | Name of the tagger which is to be plotted. This is the name of the tagger either from the `.h5` files or your freshly trained tagger (look [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/#yaml-config-file) for an explanation of the freshly trained tagger names). **IMPORTANT: If you want to use a tagger from the `.h5` files, you must run the `evaluate_model.py` script with the names of taggers in the train config in the `evaluation_settings` section. There you need to enter the name to the `tagger` list and the fraction values to the `frac_values_comp` dict. The key is the name of the tagger. |
| `class_labels` | `list` | Necessary | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |


#### Scores
Plotting the b-tagging discriminant scores for the different jet flavors. For example:

```yaml
§§§examples/plotting_umami_config_dips.yaml:36:51§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | `str` | Necessary | This gives the type of plot function used. Must be `"scores"` here. |
| `main_class` | `str` | Class which is to be tagged. |
| `dips_r21` | `None` | Necessary | Internal naming of the model. This will not show up anywhere, but it must be unique! You can define multiple of these models. All of them will be plotted. The baseline for the ratio is the first model defined here. |
| `data_set_name` | `str` | Necessary | Name of the dataset that is used. This is the name of the test_file which you want to use. |
| `tagger_name` | `str` | Necessary | Name of the tagger which is to be plotted. This is the name of the tagger either from the `.h5` files or your freshly trained tagger (look [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/#yaml-config-file) for an explanation of the freshly trained tagger names). **IMPORTANT: If you want to use a tagger from the `.h5` files, you must run the `evaluate_model.py` script with the names of taggers in the train config in the `evaluation_settings` section. There you need to enter the name to the `tagger` list and the fraction values to the `frac_values_comp` dict. The key is the name of the tagger. |
| `class_labels` | `list` | Necessary | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `label` | `str` | Necessary | Legend label of the model. |
| `working_points` | `list` | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |

#### ROC Curves
Plotting the ROC Curves of the rejection rates against the b-tagging efficiency. For example:

```yaml
§§§examples/plotting_umami_config_dips.yaml:101:117§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | `str` | Necessary | This gives the type of plot function used. Must be `"ROC"` here. |
| `main_class` | `str` | Class which is to be tagged. |
| `dips_r21` | `None` | Necessary | Internal naming of the model. This will not show up anywhere, but it must be unique! You can define multiple of these models. All of them will be plotted. The baseline for the ratio is the first model defined here. |
| `data_set_name` | `str` | Necessary | Name of the dataset that is used. This is the name of the test_file which you want to use. |
| `label` | `str` | Necessary | Legend label of the model. |
| `tagger_name` | `str` | Necessary | Name of the tagger which is to be plotted. This is the name of the tagger either from the `.h5` files or your freshly trained tagger (look [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/#yaml-config-file) for an explanation of the freshly trained tagger names). **IMPORTANT: If you want to use a tagger from the `.h5` files, you must run the `evaluate_model.py` script with the names of taggers in the train config in the `evaluation_settings` section. There you need to enter the name to the `tagger` list and the fraction values to the `frac_values_comp` dict. The key is the name of the tagger. |
| `rejection_class` | `str` | Necessary | Class which the main flavour is plotted against. |
| `draw_errors` | `bool` | Optional | Plot binomial errors to plot. |
| `xmin` | `float` | Optional | Set the minimum b efficiency in the plot (which is the xmin limit). |
| `ymax` | `float` | Optional | The maximum y axis. |
| `working_points` | `list` | Optional | The specified WPs are calculated and at the calculated b-tagging discriminant there will be a vertical line with a small label on top which prints the WP. |

You can plot two rejections at the same time with two subplots with the ratios. One for each rejection. An example for this can be seen here:

```yaml
§§§examples/plotting_umami_config_dips.yaml:119:140§§§
```

#### Variable vs Efficiency
Plot the b efficiency/c-rejection/light-rejection against the pT. For example:

```yaml
§§§examples/plotting_umami_config_dips.yaml:78:99§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | `str` | Necessary | This gives the type of plot function used. Must be `"var_vs_eff"` here. |
| `dips` | `None` | Necessary | Internal naming of the model. This will not show up anywhere, but it must be unique! You can define multiple of these models. All of them will be plotted. The baseline for the ratio is the first model defined here. |
| `data_set_name` | `str` | Necessary | Name of the dataset that is used. This is the name of the test_file which you want to use. |
| `label` | `str` | Necessary | Legend label of the model. |
| `tagger_name` | `str` | Necessary | Name of the tagger which is to be plotted. This is the name of the tagger either from the `.h5` files or your freshly trained tagger (look [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/#yaml-config-file) for an explanation of the freshly trained tagger names). **IMPORTANT: If you want to use a tagger from the `.h5` files, you must run the `evaluate_model.py` script with the names of taggers in the train config in the `evaluation_settings` section. There you need to enter the name to the `tagger` list and the fraction values to the `frac_values_comp` dict. The key is the name of the tagger. |
| `bin_edges` | `list` | Necessary | Setting the edges of the bins. Don't forget the first/last edge! |
| `flavour` | `str` | Necessary | Flavour class rejection which is to be plotted. |
| `variable` | `str` | Necessary | Variable against the efficiency/rejection is plotted against. |
| `class_labels` | `list` | Necessary | List of class labels that were used in the preprocessing/training. They must be the same in all three files! Order is important! |
| `main_class` | `str` | Necessary | Class which is to be tagged. |
| `working_point` | `float` | Necessary | Float of the working point that will be used. |
| `working_point_line` | `float` | Optional | Print a horizontal line at this value efficiency. |
| `fixed_eff_bin` | `bool` | Optional | Calculate the WP cut on the discriminant per bin. |

#### Saliency Plots
To evaluate the impact of the track variables to the final b-tagging discriminant can't be found using SHAPley. To make the impact visible (for each track of the jet), so-called Saliency maps are used. These maps are calculated when evaluating the model you have trained (if it is activated). A lot of different options can be set. An example is given here:

```yaml
§§§examples/plotting_umami_config_dips.yaml:150:161§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `type` | `str` | Necessary | This gives the type of plot function used. Must be `"saliency"` here. |
| `data_set_name` | `str` | Necessary | Name of the dataset that is used. This is the name of the test_file which you want to use. |
| `target_eff` | `float` | Necessary | Efficiency of the target flavour you want to use (Which WP you want to use). The value is given between 0 and 1. |
| `jet_flavour` | `str` | Necessary | Name of flavour you want to plot. |
| `PassBool` | `str` | Necessary | Decide if the jets need to pass the working point discriminant cut or not. `False` would give you, for example, truth b-jets which does not pass the working point discriminant cut and are therefore not tagged a b-jets. |
| `nFixedTrks` | `int` | Necessary | The saliency maps can only be calculated for jets with a fixed number of tracks. This number of tracks can be set with this parameter. For example, if this value is `8`, than only jets which have exactly 8 tracks are used for the saliency maps. This value needs to be set in the train config when you run the evaluation! If you run the evaluation with, for example `5`, you can't plot the saliency map for `8`. |

#### Fraction Contour Plot
Plot two rejections against each other for a given working point with different fraction values.

```yaml
§§§examples/plotting_umami_config_dips.yaml:9:33§§§
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `rejections` | `list` | Necessary | List with two items. These are the rejections that are plotted against each other. Only background classes can be plotted like this. |
| `tagger_name` | `str` | Necessary | Name of the tagger which is to be plotted. This is the name of the tagger either from the `.h5` files or your freshly trained tagger (look [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/#yaml-config-file) for an explanation of the freshly trained tagger names). **IMPORTANT: If you want to use a tagger from the `.h5` files, you must run the `evaluate_model.py` script with the names of taggers in the train config in the `evaluation_settings` section. There you need to enter the name to the `tagger` list and the fraction values to the `frac_values_comp` dict. The key is the name of the tagger. |
| `colour` | `str` | Optional | Give a specific colour to the tagger. |
| `linestyle` | `str` | Optional | Give a specific linestyle to the tagger. |
| `label` | `str` | Necessary | Give a label for the tagger that will be printed to the legend. |
| `data_set_name` | `str` | Necessary | The dataset to use from the dataframe as specified in evaluation. |
| `marker` | `dict` | Optional | You can set a marker (a x or something like that) at a certain fraction combination if you want to. All important information for that are added here. |
| `rejection` | `float` | Necessary (if `marker` is used) | Give two fraction values for your selected rejections. This is the position where the marker will be plotted. In the example, this is `cjets` and `ujets`. |
| `marker_style` | `str` | Optional | Give a marker style that is used for the marker. Default is "x". |
| `marker_label` | `str` | Optional | Give a custom marker legend label. Default is the tagger label + the fraction values. |
| `markersize` | `int` | Optional | Size of the marker. Default is `15`. |
| `markeredgewidth` | `int` | Optional | Size of the lines of the marker. Default is `2`. |

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
