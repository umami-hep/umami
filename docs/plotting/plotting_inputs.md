# Plotting Input Variables
The input variables for different files can also be plotted using the `plot_input_variables.py` script. Its also steered by a yaml file. An example for such a file can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml). The structure is close to the one from `plotting_umami` but still a little bit different.

### Yaml File
In the following, the possible configration parameters are listed with a brief description. 

#### Variable dict and number of jets
Here you can define the number of jets that are used and also the variable dict, where all the variables that are available are saved.

??? example "Click to see corresponding code highlighted in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml linenums="1", hl_lines="9-14"
    §§§examples/plotting_input_vars.yaml§§§
    ```

#### Number of Tracks per Jet
The number of tracks per jet can be plotted for all different files. This can be given like this:

??? example "Click to see corresponding code highlighted in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml linenums="1", hl_lines="117-133"
    §§§examples/plotting_input_vars.yaml§§§
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
| `plot_settings` | `dict` | Necessary | Here starts the plot settings. See possible parameters in the section below. |

#### Input Variables Tracks
To plot the track input variables, the following options are used.

??? example "Click to see corresponding code highlighted in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml linenums="1", hl_lines="135-168"
    §§§examples/plotting_input_vars.yaml§§§
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

#### Input Variables Jets
To plot the jet input variables, the following options are used.

??? example "Click to see corresponding code highlighted in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml linenums="1", hl_lines="16-115"
    §§§examples/plotting_input_vars.yaml§§§
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
| `binning` | None | Necessary | Here starts the binning for each variable. If you give a `int`, there will be so much equal distant bins. You can also give a three element `list` which will be used in the `numpy.arange` function. The first element is start, second is stop and third is the step width. The so arranged numbers are bin edges not bins! If `None` is given, the standard value is `100`. Variables that are not in here are not plotted! |
| `plot_settings` | `dict` | Necessary | Here starts the plot settings. See possible parameters in the section below. |

#### Plot settings
The `plot_settings` section is similar for all three cases described above. 
In order to define some settings you want to apply to all plots, use yaml anchors
as shown here:

??? example "Click to see corresponding code highlighted in the [example config file](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_input_vars.yaml)"
    ```yaml linenums="1", hl_lines="1-7"
    §§§examples/plotting_input_vars.yaml§§§
    ```

Most of the plot settings are valid for all types of input variable plots 
(i.e. jet variables, track variables and the n_tracks plot).
If a parameter is only valid for a certain type of plot, this is listed below.


## Plot settings
You have to specify some parameters for the plots themselves, like for example the
`binning` of all the variables. *Note that the binning also indicates if a variable is 
plotted or not.*

You can use the following parameters. Note that some parameters are not supported for all types of plots.

| Options | Plot Type | Data Type | Necessary/Optional | Explanation |
|---------|-----------|-----------|--------------------|-------------|
| `binning` | All | `int`, `list` or empty | Necessary | Here starts the binning for each variable. If you give a `int`, there will be that many equal-width bins. You can also give a three element `list` which will be used in the `numpy.arange` function. The first element is start, second is stop and third is number of bins. The so arranged numbers are bin edges not bins! If no value is given, the standard value is `100`. If a variable is not defined here, its not plotted. |
| `sorting_variable` | Track variables | `str` | Optional | Variable Name to sort after. |
| `n_leading` | Track variables | `list` | Optional | `list` of the x leading tracks. If `None`, all tracks will be plotted. If `0` the leading tracks sorted after `sorting variable` will be plotted. You can add like `None`, `0` and `1` for example and it will plot all 3 of them, each in their own folders with according labeling. This must be a `list`! Even if there is only one option given. |
| `track_origins` | Track variables and n_tracks plot | `list` | Optional | `list` that gives the desired track origins when plotting. |

All remaining plot settings are parameters which are handed to the plotting API, 
more specifically the `histogram_plot` class.
Therefore, all parameters supported by the `histogram_plot` class can be specified there.

### List of plotting API parameters

§§§docs/ci_assets/docstring_input_var_plots_umami.plotting.histogram_plot.md§§§
§§§docs/ci_assets/docstring_input_var_plots_umami.plotting.plot_object.md:3:§§§
