# Instructions to train DL1 with the umami framework

The following instructions are meant to give a guideline how to reproduce the DL1r results obtained in the last [retraining campaign](http://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/FTAG-2019-005/). It is focused on the PFlow training. The approach can also be used to train DL1d and, more generally, any variation to the list of input of the DL1 tagger.

## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [mc-samples.md](mc-samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [mc-samples.md](mc-samples.md) which can be used for training.

### Ntuple preparation

After the previous step the ntuples need to be further processed. We can use different resampling approaches to achieve the same pt and eta distribution for all of the used flavour categories. In order to reduce the memory usage, we first extract the different jet categories separately, since, e.g., c-jets only make up 8% of the ttbar sample.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](preprocessing.md) for additional information.
Note, that for running DL1 no tracks have to be stored in the output hybrid sample. Therefore, the `save_tracks` argument in the preprocessing config does not need to be set while preprocessing the samples.

Note that the training Variables for DL1r R21 and DL1r R22 are defined in [DL1r_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/DL1r_Variables.yaml) and [DL1r_Variables_R22.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/DL1r_Variables_R22.yaml).

Important note about the preprocessing: when using taus, it is not advised to use the `count` sampling method. Indeed, the tau statistics is much lower, which would result in throwing away far too many jets. Instead, use the `pdf` or the `probability_ratio` sampling methods to have an exact match of the jets flavour and a proportional distribution of taus.

If you don't want to process some samples yourself, you can use the already preprocessed samples uploaded to rucio in the datasets `user.mdraguet.dl1r.R21.PFlowJetsDemoSamples` for DL1r or `user.mdraguet.dl1d.R21.PFlowJetsDemoSamples` for DL1d (RNNIP replaced by DIPS). These datasets do not have taus included. Note that you need to download both the datasets and the associated dictionary with scaling factors (+ the dictionary of variable). There are two test samples available: an hybrid (ttbar + Z'-ext) and a Z'-ext solely. Each should be manually cut in 2 to get a test and validation file. The data comes from:
- mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p4060
- mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p4060

## Config File

After all the files are ready we can start with the training. The config file for the DL1r training is [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml). This will look for example like this:

```yaml
# Set modelname and path to Pflow preprocessing config file
model_name: <MODELNAME>
preprocess_config: <path>/<to>/<preprocessing>/<config>/PFlow-Preprocessing.yaml

# Add here a pretrained model to start with.
# Leave empty for a fresh start
model_file:

# Add training file
train_file: <path>/<to>/<train>/<samples>/train_file.h5

# Defining templates for the variable cuts
.variable_cuts_ttbar: &variable_cuts_ttbar
    variable_cuts:
        - pt_btagJes:
            operator: "<="
            condition: 2.5e5

.variable_cuts_zpext: &variable_cuts_zpext
    variable_cuts:
        - pt_btagJes:
            operator: ">"
            condition: 2.5e5


# Add validation files
validation_files:
    ttbar_r21_val:
        path: <path_palce_holder>/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5
        label: "$t\\bar{t}$ Release 21"
        <<: *variable_cuts_ttbar

    zprime_r21_val:
        path: <path_palce_holder>/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5
        label: "$Z'$ Release 21"
        <<: *variable_cuts_zpext

test_files:
    ttbar_r21:
        path: <path>/<to>/<preprocessed>/<samples>/ttbar_r21_test_file.h5
        <<: *variable_cuts_ttbar

    ttbar_r22:
        path: <path>/<to>/<preprocessed>/<samples>/ttbar_r22_test_file.h5
        <<: *variable_cuts_ttbar

    zpext_r21:
        path: <path>/<to>/<preprocessed>/<samples>/zpext_r21_test_file.h5
        <<: *variable_cuts_zpext

    zpext_r22:
        path: <path>/<to>/<preprocessed>/<samples>/zpext_r22_test_file.h5
        <<: *variable_cuts_zpext

# Path to Variable dict used in preprocessing
var_dict: <path>/<to>/<variables>/DL1r_Variables.yaml

# Variables or Variable Headers to exclude from the tagger
#      but contained in <path>/<to>/<variables>/DL1r_Variables.yaml
exclude: null

# Values for the neural network
NN_structure:
    # Decide, which tagger is used
    tagger: "dl1"

    # NN Training parameters
    lr: 0.001
    batch_size: 15000
    epochs: 200

    # Number of jets used for training
    # To use all: Fill nothing
    nJets_train:

    # Dropout rate. If = 0, dropout is disabled
    dropout: 0

    # Define which classes are used for training
    # These are defined in the global_config
    class_labels: ["ujets", "cjets", "bjets"]

    # Main class which is to be tagged
    main_class: "bjets"

    # Decide if Batch Normalisation is used
    Batch_Normalisation: True

    # Nodes per dense layer. Starting with first dense layer.
    dense_sizes: [256, 128, 60, 48, 36, 24, 12, 6]

    # Activations of the layers. Starting with first dense layer.
    activations: ["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"]

    # Variables to repeat in the last layer (example)
    repeat_end: ["pt_btagJes", "absEta_btagJes"]

    # Options for the Learning Rate reducer
    LRR: True

    # Option if you want to use sample weights for training
    use_sample_weights: False


# Plotting settings for training metrics plots
Validation_metrics_settings:
    # Define which taggers should also be plotted
    taggers_from_file: ["rnnip", "DL1r"]

    # Label for the freshly trained tagger
    tagger_label: "DL1r"

    # Enable/Disable atlas tag
    UseAtlasTag: True

    # fc_value and WP_b are autmoatically added to the plot label
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow jets"

    # Set the datatype of the plots
    plot_datatype: "pdf"

# Eval parameters for validation evaluation while training
Eval_parameters_validation:
    # Number of jets used for validation
    n_jets: 3e5

    # Define taggers that are used for comparison in evaluate_model
    # This can be a list or a string for only one tagger
    tagger: ["rnnip", "DL1r"]

    # Define fc values for the taggers
    frac_values_comp: {
        "rnnip": {
            "cjets": 0.08,
            "ujets": 0.92,
        },
        "DL1r": {
            "cjets": 0.018,
            "ujets": 0.982,
        },
    }

    # Charm fraction value used for evaluation of the trained model
    frac_values: {
        "cjets": 0.018,
        "ujets": 0.982,
    }

    # A list to add available variables to the evaluation files
    add_variables_eval: ["actualInteractionsPerCrossing"]

    # Working point used in the evaluation
    WP: 0.77

    # Minimal efficiency considered in ROC computation
    eff_min: 0.49


    # some properties for the feature importance explanation with SHAPley
    shapley:
        # Over how many full sets of features it should calculate over.
        # Corresponds to the dots in the beeswarm plot.
        # 200 takes like 10-15 min for DL1r on a 32 core-cpu
        feature_sets: 200

        # defines which of the model outputs (flavor) you want to explain
        # [tau,b,c,u] := [3, 2, 1, 0]
        model_output: 2

        # You can also choose if you want to plot the magnitude of feature
        # importance for all output nodes (flavors) in another plot. This
        # will give you a bar plot of the mean SHAP value magnitudes.
        bool_all_flavor_plot: False

        # as this takes much longer you can average the feature_sets to a
        # smaller set, 50 is a good choice for DL1r
        averaged_sets: 50

        # [11,11] works well for dl1r
        plot_size: [11, 11]

```

It contains the information about the neural network architecture and the training as well as about the files for training, validation and testing. Also evaluation parameters are given for the training evaluation which is performed by the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script.

The different options are briefly explained here:

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `model_name` | String | Necessary | Name of the model which is to be trained. Also the foldername where everything of the model will be saved. |
| `preprocess_config` | String | Necessary | Path to the `preprocess_config` which was used to produce the training samples. |
| `model_file` | String | Optional | If you already have a model and want to continue the training of this model, you can give the path to this model here. This model will be loaded and used instead of init a new one. |
| `train_file` | String | Necessary | Path to the training sample. This is given by the `preprocessing` step of Umami |
| `validation_files` | Dict | Optional | Here you can define different validation samples that are used in the training and the `plotting_epoch_performance.py` script. Those validation samples need to be defined in a dict structure shown in the example. The name of the dict entry is relevant and is the unique identifier for this sample (DO NOT USE IT MULTIPLE TIMES). `path` gives the path to the file. |
| `test_files` | Dict | Optional | Here you can define different test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is relevant and is the unique identifier in the results file which is produced by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). `Path` gives the path to the file. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `path` | String | Necessary | Path to the validation/test file which is to be used. Using wildcards is possible. |
| `variable_cuts` | Dict | Optional | Dict of cuts which are applied when loading the different test files. Only jet variables can be cut on. These are in this example defined as templates for the different samples types. |
| `var_dict` | String | Necessary | Path to the variable dict used in the `preprocess_config` to produce the train sample. |
| `exclude` | List | Necessary | List of variables that are excluded from training. Only compatible with DL1r training. To include all, just give an empty list. |
| `NN_structure` | None | Necessary | A dict where all important information for the training are defined. |
| `tagger` | String | Necessary | Name of the tagger that is used/to be trained. |
| `lr` | Float | Necessary | Learning rate which is used for training. |
| `batch_size` | Int | Necessary | Batch size which is used for training. |
| `epochs` | Int | Necessary | Number of epochs of the training. |
| `nJets_train` | Int | Necessary | Number of jets used for training. Leave empty to use all. |
| `dropout` | Float | Necessary | Dropout factor used in the network. If 0, dropout is not used. |
| `class_labels` | List | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | String | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. |
| `Batch_Normalisation` | Bool | Necessary | Decide, if batch normalisation is used in the network. |
| `activations` | List | Necessary | List of activation function for each layer of the network. Every entry is one layer. The entries need to be string recognised as activation functions! |
| `dense_sizes` | List | Necessary | List of nodes per layer of the network. Every entry is one layer. The numbers need to be ints! |
| `LRR` | Bool | Optional | Decide, if a Learning Rate Reducer (LRR) is used or not. If yes, the following options can be added. |
| `use_sample_weights` | Bool | Optional | Applies the weights, you calculated with the `--weighting` flag from the preprocessing to the training loss function. |
| `LRR_monitor` | String | Optional | Quantity to be monitored. Default: "loss" |
| `LRR_factor` | Float | Optional | Factor by which the learning rate will be reduced. `new_lr = lr * factor`. Default: 0.8 |
| `LRR_patience` | Int | Optional | Number of epochs with no improvement after which learning rate will be reduced. Default: 3 |
| `LRR_verbose` | Int | Optional | 0: Quiet, 1: Update messages. Default: 1 |
| `LRR_mode` | String | Optional | One of `{"auto", "min", "max"}`. In "min" mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in "max" mode it will be reduced when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Default: "auto" |
| `LRR_cooldown` | Int | Optional | Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 5 |
| `LRR_min_lr` | Float | Optional | Lower bound on the learning rate. Default: 0.000001 |
| `Validation_metrics_settings` | None | Necessary | Plotting settings for the validation plots which are produced by the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `taggers_from_file` | List | Optional | List of taggers that are available in the .h5 samples. The here given taggers are plotted as reference lines in the rejection per epoch plots. |
| `tagger_label` | String | Optional | Name for the legend of the freshly trained tagger for the rejection per epoch plots. |
| `trained_taggers` | Dict | Optional | A dict with local trained taggers which shall be plotted in the rejection per epoch plots. You need to provide a dict with a `path` and a `label`. The path is the path to the validation metrics .json file, where the rejections per epoch are saved. The `label` is the label which will be shown in the legend in the rejection per epoch plots. The `dipsReference` in the example here is just an internal naming. It will not be shown anywhere. |
| `UseAtlasTag` | Bool | Optional | Decide, if the ATLAS tag is printed at the top left of the plot. |
| `AtlasTag` | String | Optional | Main ATLAS tag which is right to "ATLAS" |
| `SecondTag` | String | Optional | Second line below the ATLAS tag |
| `plot_datatype` | String | Necessary | Datatype of the plots that are produced using the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `Eval_parameters_validation` | None | Necessary | A dict where all important information for the training are defined. |
| `n_jets` | Int | Necessary | Number of jets used for evaluation. This should not be to high, due to the fact that Callback function also uses this amount of jets after each epoch for validation. |
| `tagger` | List | Necessary | List of taggers used for comparison. This needs to be a list of string or a single string. The name of the taggers must be same as in the evaluation file. For example, if the DL1d probabilities in the test samples are called `DL1dLoose20210607_pb`, the name you need to add to the list is `DL1dLoose20210607`. |
| `frac_values_comp` | Dict | Necessary | Dict with the fraction values for the comparison taggers. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `frac_values` | Dict | Necessary | Dict with the fraction values for the freshly trained tagger. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `WP` | Float | Necessary | Working point which is used in the validation and evaluation. |
| `eff_min` | Float | Optional | Minimal main class efficiency considered for ROC. |

## Training

Before starting the training, you need to set some paths for the umami package to find all the tools. Change to the umami dir and run the `setup.py`.

```bash
python setup.py install
```

Note that with the `install` setup, changes that are performed to the scripts after setup are not included! For development and usage of changes without resetup everything, use

```bash
source run_setup.sh
```

This script sets the python path to a specific folder where the executables are directly the code you are working on.

After that, you can switch to the folder `umami/umami` and run the training, using the following command

```bash
train.py -c ${EXAMPLES}/DL1r-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml).

If you want instant performance checks of the model after each epoch during the training, you can use

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/DL1r-PFlow-Training-config.yaml
```

which will write out plots for the light- and c-rejection, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. In this form, the performance measurements, like light- and c-rejection, will be recalculated using the working point, the `frac_values` value and the number of validation jets defined in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml). If taus are used in the training too, they will be included in these plots.

If you don't want to recalculate it, you can give the path to the existing dict with the option `--dict`. For example:

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/DL1r-PFlow-Training-config.yaml --dict MODELNAME/validation_WP0p77_fc0p018_300000jets_Dict.json
```

## Evaluating the Results

After the training is over, the different epochs can be evaluated with ROC plots, output scores, variable vs efficiency, confusion matrices, etc. using the build-in scripts. Before plotting these, the model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script.

```bash
evaluate_model.py -c ${EXAMPLES}/DL1r-PFlow-Training-config.yaml -e 5
```

The `-e` options (here `5`) allows to set the training epoch which should be evaluated.
It will produce .h5 and .pkl files with the evaluations which will be saved in the model folder in an extra folder called `results/`. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. For an explanation, look in the [plotting_umami documentation](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md). The plots can then be retrieved running the following command

```bash
python umami/plotting_umami.py -c ${EXAMPLES}/plotting_umami_config_DL1r.yaml -o eval_plots
```
