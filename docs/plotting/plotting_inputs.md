# Plotting Input Variables
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
  track_origins: ['All', 'Pileup', 'Fake', 'Primary', 'FromB', 'FromBC', 'FromC', 'FromTau', 'OtherSecondary']
  Datasets_to_plot:
    R21:
      files: <path>/<to>/<h5>/<files>/*.h5
      label: "R21 Loose"
      tracks_name: "tracks"
    R22:
      files: <path>/<to>/<h5>/<files>/*.h5
      label: "R22 Loose"
      tracks_name: "tracks"
  plot_settings:
    Log: True
    UseAtlasTag: True
    AtlasTag: "Internal Simulation"
    SecondTag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow Jets \n3M Jets"
    yAxisAtlasTag: 0.925
    yAxisIncrease: 2
    figsize: [7, 5]
    Ratio_Cut: [0.5, 2]
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
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |
| `Log` | `bool` | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | `bool` | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | `str` | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | `str` | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | `float` |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | `float` | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | `list` | Optional | Two element `list` that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | `list` | Optional | Two element `list` that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
| `track_origins` | `list` | Optional | `list` that gives the desired track origins when plotting. |

#### Input Variables Tracks
To plot the track input variables, the following options are used.

```yaml
input_vars_trks_ttbar_loose_ptfrac:
  variables: "tracks"
  folder_to_save: input_vars_trks_ttbar_loose
  track_origins: ['All', 'Pileup', 'Fake', 'Primary', 'FromB', 'FromBC', 'FromC', 'FromTau', 'OtherSecondary']
  Datasets_to_plot:
    R21:
      files: <path>/<to>/<h5>/<files>/*.h5
      label: "R21 Loose"
      tracks_name: "tracks"
    R22:
      files: <path>/<to>/<h5>/<files>/*.h5
      label: "R22 Loose"
      tracks_name: "tracks"
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
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |
| `sorting_variable` | `str` | Optional | Variable Name to sort after. |
| `n_Leading` | `list` | Optional | `list` of the x leading tracks. If `None`, all tracks will be plotted. If `0` the leading tracks sorted after `sorting variable` will be plotted. You can add like `None`, `0` and `1` for example and it will plot all 3 of them, each in their own folders with according labeling. This must be a `list`! Even if there is only one option given. |
| `Log` | `bool` | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | `bool` | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | `str` | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | `str` | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | `float` |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
| `yAxisIncrease` | `float` | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | `list` | Optional | Two element `list` that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | `list` | Optional | If you add more then two models to plot, the comparison function is used with a small ratio plot at the bottom. Two element `list` that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
| `binning` | None | Necessary | Here starts the binning for each variable. If you give a `int`, there will be so much equal distant bins. You can also give a three element `list` which will be used in the `numpy.arange` function. The first element is start, second is stop and third is the step width. The so arranged numbers are bin edges not bins! If `None` is given, the standard value is `100`. If a variable is not defined here, its not plotted. |
| `track_origins` | `list` | Optional | `list` that gives the desired track origins when plotting. |

#### Input Variables Jets
To plot the jet input variables, the following options are used.

```yaml
input_vars_ttbar_loose:
  variables: "jets"
  folder_to_save: input_vars_ttbar_loose
  Datasets_to_plot:
    R21:
      files: <path>/<to>/<h5>/<files>/*.h5
      label: "R21 Loose"
    R22:
      files: <path>/<to>/<h5>/<files>/*.h5
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
| `plot_settings` | None | Necessary | Here starts the plot settings. Do not fill! |
| `Log` | `bool` | Optional | Decide if the plots are plotted with logarithmic y axis or without. |
| `UseAtlasTag` | `bool` | Optional | Decide if the ATLAS Tag is printed in the upper left corner of the plot or not. |
| `AtlasTag` | `str` | Optional | The first line of text right behind the 'ATLAS'. |
| `SecondTag` | `str` | Optional | Second line (if its starts with `\n`) of text right below the 'ATLAS' and the AtlasTag. |
| `yAxisAtlasTag` | `float` |  Optional | y-axis position of the ATLAS Tag in parts of the y-axis (0: lower left corner, 1: upper left corner). |
 | `yAxisIncrease` | `float` | Optional |Increase the y-axis by a given factor. Mainly used to fit in the ATLAS Tag without cutting the lines of the plot. |
| `figsize` | `list` | Optional | Two element `list` that gives the shape of the plot. (First is width, second is height). |
| `Ratio_Cut` | `list` | Optional | If you add more then two models to plot, the comparison function is used with a small ratio plot at the bottom. Two element `list` that gives the lower (first element) and upper (second element) y axis bound of the ratio plot below the main plot. |
