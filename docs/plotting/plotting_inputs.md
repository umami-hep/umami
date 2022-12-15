# Plotting Input Variables
The input variables for different files can also be plotted using the `plot_input_variables.py` script. Its also steered by a yaml file. An example for such a file can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml). The structure is close to the one from `plotting_umami` but still a little bit different.
To start the plotting of the input variables, you need to run the following command

```bash
plot_input_vars.py -c <path/to/config> --tracks
```

or

```bash
plot_input_vars.py -c <path/to/config> --jets
```

which will plot either all plots defined using jet- or track variables. You can also give the `-f` or `--format` option where you can decide on a format for the plots. The default is `pdf`.

### Yaml File
In the following, the possible configration parameters are listed with a brief description. 

#### Number of jets
Here you can define the number of jets that are used.

??? example "Click to see corresponding code in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml
    §§§examples/plotting_input_vars.yaml:15:17§§§
    ```

#### Number of Tracks per Jet
The number of tracks per jet can be plotted for all different files. This can be given like this:

??? example "Click to see corresponding code in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml
    §§§examples/plotting_input_vars.yaml:110:128§§§
    ```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `nTracks_ttbar_loose` | `str` | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | `str` | Necessary | Must be set to "tracks" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | `str` | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. |
| `nTracks` | `bool` | Necessary | MUST BE TRUE HERE! Decide if the Tracks per Jets are plotted or the input variable. |
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | `str` | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achieve the number of jets given in the `Eval_parameters`. |
| `label` | `str` | Necessary | Plot label for the plot legend. |
| `tracks_name` | `str` | Necessary | Name of the tracks inside the h5 files you want to plot. |
| `cut_vars_dict` | `list` | Necessary | A dict with cuts on the jet variables that should be applied when creating the input variable plots. Technically, this is implemented as a list of `dict` entries, which have as the key the name of the variable which is used for the cut (e.g. `pt_btagJes`) and then as sub-entries the operator used for the cut (`operator`) and the condition used for the cut (`condition`). |
| `plot_settings` | `dict` | Necessary | Here starts the plot settings. See possible parameters in the section below. |

#### Input Variables Tracks
To plot the track input variables, the following options are used.

??? example "Click to see corresponding code in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml
    §§§examples/plotting_input_vars.yaml:130:170§§§
    ```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `input_vars_trks_ttbar_loose_ptfrac` | `str` | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | `str` | Necessary | Must be set to "tracks" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | `str` | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. |
| `nTracks` | `bool` | Necessary | To plot the input variable distributions, this must be `False`. |
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | `str` | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achieve the number of jets given in the `Eval_parameters`. |
| `label` | `str` | Necessary | Plot label for the plot legend. |
| `tracks_name` | `str` | Necessary | Name of the tracks inside the h5 files you want to plot. |
| `plot_settings` | `dict` | Necessary | Here starts the plot settings. See possible parameters in the section below. |
| `var_dict` | `dict` | Necessary | A dict with all the variables you want to plot inside. The key of the entry is the name of the variable you want to plot (how it is named in the files) and the entry itself is the binning. If you give an `int`, you will get your chosen number of equidistant bins. You can also give a three element `list` which will be used in the `numpy.arange` function. The first element is start, second is stop and third is number of bins. The so arranged numbers are bin edges not bins! If no value is given, the standard value is `100`. If you want, for example, plot the sum of `numberOfPixelHits` and `numberOfSCTHits`, the entry needs to be a dict itself with three entries. `variables`, which is a list of variables you want to add up for example. `operator` which is the operation how to merge them. Available are `"+"`, `"-"`, `"*"` and `"/"`. And last the binning. This is the same as explained before with the `int` and the `list`. An example is given in the config above. The variable is named `number_nPix_nSCT`. You can also apply the log to one variable. This can be done by defining only one variable in the dict and set the operator to `"log"`. |
| `cut_vars_dict` | `list` | Necessary | A dict with cuts on the jet variables that should be applied when creating the input variable plots. Technically, this is implemented as a list of `dict` entries, which have as the key the name of the variable which is used for the cut (e.g. `pt_btagJes`) and then as sub-entries the operator used for the cut (`operator`) and the condition used for the cut (`condition`). |
| `xlabels` | `dict` | Optional | Dict with custom xlabels |

#### Input Variables Jets
To plot the jet input variables, the following options are used.

??? example "Click to see corresponding code in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml
    §§§examples/plotting_input_vars.yaml:19:108§§§
    ```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `input_vars_trks_ttbar_loose_ptfrac` | `str` | Necessary | Name of the plots. This does not effect anything for the plots itself. |
| `variables` | `str` | Necessary | Must be set to "jets" for this function. Decides, which functions for plotting are used. |
| `folder_to_save` | `str` | Necessary | Path where the plots should be saved. This is a relative path. Add a folder name as path. |
| `Datasets_to_plot` | None | Necessary | Here the category starts of which plots shall be plotted. |
| `R21` | None | Necessary | Name of the fileset which is to be plotted. Does not effect anything! |
| `files` | `str` | Necessary | Path to a file which is to be used for plotting. Wildcard is supported. The function will load as much files as needed to achieve the number of jets given in the `Eval_parameters`. |
| `label` | `str` | Necessary | Plot label for the plot legend. |
| `special_param_jets` | None | Necessary | Here starts the special x axis limits for a variable. If you want to set the x range by hand, add the variable here and also the `lim_left` for xmin and `lift_right` for xmax. |
| `var_dict` | `dict` | Necessary | A dict with all the variables you want to plot inside. The key of the entry is the name of the variable you want to plot (how it is named in the files) and the entry itself is the binning. If you give an `int`, you will get your chosen number of equidistant bins. You can also give a three element `list` which will be used in the `numpy.arange` function. The first element is start, second is stop and third is number of bins. The so arranged numbers are bin edges not bins! If no value is given, the standard value is `100`. If you want, for example, plot the sum of `rnnip_pc` and `rnnip_pu`, the entry needs to be a dict itself with three entries. `variables`, which is a list of variables you want to add up for example. `operator` which is the operation how to merge them. Available are "+", "-", "*" and "/". And last the binning. This is the same as explained before with the `int` and the `list`. An example is given in the config above. The variable is named `combined_rnnip`. You can also apply the log to one variable. This can be done by defining only one variable in the dict and set the operator to `log`. |
| `cut_vars_dict` | `list` | Necessary | A dict with cuts on the jet variables that should be applied when creating the input variable plots. Technically, this is implemented as a list of `dict` entries, which have as the key the name of the variable which is used for the cut (e.g. `pt_btagJes`) and then as sub-entries the operator used for the cut (`operator`) and the condition used for the cut (`condition`). |
| `plot_settings` | `dict` | Necessary | Here starts the plot settings. See possible parameters in the section below. |
| `xlabels` | dict | Optional | Dict with custom xlabels |

#### Plot settings
The `plot_settings` section is similar for all three cases described above. 
In order to define some settings you want to apply to all plots, use yaml anchors
as shown here:

??? example "Click to see corresponding code in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml
    §§§examples/plotting_input_vars.yaml:1:13§§§
    ```

Most of the plot settings are valid for all types of input variable plots 
(i.e. jet variables, track variables and the n_tracks plot).
If a parameter is only valid for a certain type of plot, this is listed below.


## Plot settings
You can specify some parameters for the plots themselves. You can use the following parameters. Note that some parameters are not supported for all types of plots.

| Options | Plot Type | Data Type | Necessary/Optional | Explanation |
|---------|-----------|-----------|--------------------|-------------|
| `xlabels` | dict | Optional | Dict with custom xlabels |
| `sorting_variable` | Track variables | `str` | Optional | Variable Name to sort after. |
| `n_leading` | Track variables | `list` | Optional | `list` of the x leading tracks. If `None`, all tracks will be plotted. If `0` the leading tracks sorted after `sorting variable` will be plotted. You can add like `None`, `0` and `1` for example and it will plot all 3 of them, each in their own folders with according labeling. This must be a `list`! Even if there is only one option given. |
| `track_origins` | Track variables and n_tracks plot | `list` | Optional | `list` that gives the desired track origins when plotting. |

All remaining plot settings are parameters which are handed to `puma` (Plotting
UMami API) more specifically the `HistogramPlot` class.
Therefore, all parameters supported by the `HistogramPlot` class can be specified there.

[`puma` documentation](https://umami-hep.github.io/puma/)

### List of `puma` parameters

§§§docs/ci_assets/docstring_puma_HistogramPlot.md§§§
§§§docs/ci_assets/docstring_puma_PlotObject.md:3:§§§
<!-- in the docstring for the PlotObject class, start at line 3, since we don't want
the header to be included (column names of the md-table -->
