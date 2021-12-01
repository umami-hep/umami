# Instructions to train DIPS with the umami framework

The following instructions are meant to give a guidline how to reproduce the DIPS results presented in the [DIPS Note](https://cds.cern.ch/record/2718948). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md) which can be used for training.

### Ntuple preparation

After the previous step the ntuples need to be further processed. We can use different resampling approaches to achieve the same pt and eta distribution for all of the used flavour categories.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](preprocessing.md) for additional information.
Note, that for running Dips tracks have to be stored in the output hybrid sample. Therefore, the `save_tracks` argument in the preprocessing config need to be set while the preprocessing the samples.

## Config File

After all the files are ready we can start with the training. The config file for the Dips training is [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). This will look for example like this:

```yaml
# Set modelname and path to Pflow preprocessing config file
model_name: <MODELNAME>
preprocess_config: <path>/<to>/<preprocessing>/<config>/PFlow-Preprocessing.yaml

# Add here a pretrained model to start with.
# Leave empty for a fresh start
model_file:

# Add training file
train_file: <path>/<to>/<train>/<samples>/train_file.h5

# Add validation files
# ttbar val
validation_file: <path>/<to>/<validation>/<samples>/ttbar_r21_validation_file.h5

# zprime val
add_validation_file:  <path>/<to>/<validation>/<samples>/zpext_r21_validation_file.h5

ttbar_test_files:
    ttbar_r21:
        Path: <path>/<to>/<preprocessed>/<samples>/ttbar_r21_test_file.h5
        data_set_name: "ttbar_r21"

    ttbar_r22:
        Path: <path>/<to>/<preprocessed>/<samples>/ttbar_r22_test_file.h5
        data_set_name: "ttbar_r22"

zpext_test_files:
    zpext_r21:
        Path: <path>/<to>/<preprocessed>/<samples>/zpext_r21_test_file.h5
        data_set_name: "zpext_r21"

    zpext_r22:
        Path: <path>/<to>/<preprocessed>/<samples>/zpext_r22_test_file.h5
        data_set_name: "zpext_r22"

# Path to Variable dict used in preprocessing
var_dict: <path>/<to>/<variables>/Dips_Variables.yaml

exclude: []

# Values for the neural network
NN_structure:
    # Decide, which tagger is used
    tagger: "dips"

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

    # Structure of the dense layers for each track
    ppm_sizes: [100, 100, 128]

    # Structure of the dense layers after summing up the track outputs
    dense_sizes: [100, 100, 100, 30]

    # Options for the Learning Rate reducer
    LRR: True

# Plotting settings for training metrics plots
Validation_metrics_settings:
    # Define which taggers should also be plotted
    taggers_from_file: ["rnnip", "DL1r"]

    # Define which freshly trained taggers should be plotted
    trained_taggers:
        dipsReference:
            path: "dips_Loose/validation_WP0p77_300000jets_Dict.json"
            label: "DIPS Reference"

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

    # Cuts which are applied to the different datasets used for evaluation
    variable_cuts:
        ttbar_r21:
            - pt_btagJes:
                operator: "<="
                condition: 250000

        ttbar_r22:
            - pt_btagJes:
                operator: "<="
                condition: 250000

        zpext_r21:
            - pt_btagJes:
                operator: ">"
                condition: 250000

        zpext_r22:
            - pt_btagJes:
                operator: ">"
                condition: 250000

    # Working point used in the evaluation
    WP: 0.77

    # Decide, if the Saliency maps are calculated or not.
    Calculate_Saliency: True
```

It contains the information about the neural network architecture and the training as well as about the files for training, validation and testing. Also evaluation parameters are given for the training evaluation which is performed by the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script.

The different options are briefly explained here:

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `model_name` | String | Necessary | Name of the model which is to be trained. Also the foldername where everything of the model will be saved. |
| `preprocess_config` | String | Necessary | Path to the `preprocess_config` which was used to produce the training samples. |
| `model_file` | String | Optional | If you already have a model and want to continue the training of this model, you can give the path to this model here. This model will be loaded and used instead of init a new one. |
| `train_file` | String | Necessary | Path to the training sample. This is given by the `preprocessing` step of Umami |
| `validation_file` | String | Necessary | Path to the validation sample (ttbar). This is given by the `preprocessing` step of Umami |
| `add_validation_file` | String | Necessary | Path to the validation sample (zpext). This is given by the `preprocessing` step of Umami |
| `ttbar_test_files` | Dict | Optional | Here you can define different ttbar test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is irrelevant while the `Path` and `data_set_name` are important. The `data_set_name` needs to be unique. Its the identifier/name of the dataset in the evaluation file which is used for plotting. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `zpext_test_files` | Dict | Optional | Here you can define different zpext test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is irrelevant while the `Path` and `data_set_name` are important. The `data_set_name` needs to be unique. Its the identifier/name of the dataset in the evaluation file which is used for plotting. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `var_dict` | String | Necessary | Path to the variable dict used in the `preprocess_config` to produce the train sample. |
| `exclude` | List | Necessary | List of variables that are excluded from training. Only compatible with DL1r training. To include all, just give an empty list. |
| `NN_structure` | None | Necessary | A dict where all important information for the training are defined. |
| `tagger` | String | Necessary | Name of the tagger that is used/to be trained. |
| `lr` | Float | Necessary | Learning rate which is used for training. |
| `batch_size` | Int | Necessary | Batch size which is used for training. |
| `epochs` | Int | Necessary | Number of epochs of the training. |
| `nJets_train` | Int | Necessary | Number of jets used for training. Leave empty to use all. |
| `dropout` | Float | Necessary | Dropout factor used in the _ϕ_ network. If 0, dropout is not used. |
| `class_labels` | List | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | String | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. |
| `Batch_Normalisation` | Bool | Necessary | Decide, if batch normalisation is used in the _ϕ_ network. |
| `ppm_sizes` | List | Necessary | List of nodes per layer of the _ϕ_ network. Every entry is one layer. The numbers need to be ints! |
| `dense_sizes` | List | Necessary | List of nodes per layer of the _F_ network. Every entry is one layer. The numbers need to be ints! |
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
| `variable_cuts` | Dict | Necessary | Dict of cuts which are applied when loading the different test files. Only jet variables can be cut on. |
| `WP` | Float | Necessary | Working point which is used in the validation and evaluation. |
| `Calculate_Saliency` | Bool | Optional | Decide, if the saliency maps are calculated or not. This takes a lot of time and resources! |

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
train.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). 

If you want instant performance checks of the model after each epoch during the training, you can use

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml
```

which will write out plots for the not- main flavour rejections, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. In this form, the performance measurements, like the rejection performances, will be recalculated using the working point, the `frac_values` and the number of validation jets defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). If you don't want to recalculate it, you can give the path to the existing dict with the option `--dict`. For example:

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dict dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full/validation_WP0p77_fc0p018_300000jets_Dict.json
```

### Train on Zeuthen Cluster

Alternatively, if you are working out of the DESY Zeuthen servers, `warp.zeuthen.desy.de`, you can train using the batch system via `qsub` and GPU support by giving it the `zeuthen` flag

```bash
train_Dips.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --zeuthen
```

The job will output a log to the current working directory and copy the results to the current working directory when it's done. The options for the job (time, memory, space, etc.) can be changed in `umami/institutes/zeuthen/train_job.sh`.

## Evaluating the results

After the training is over, the different epochs can be evaluated with ROC plots, output scores, saliency maps and confusion matrices etc. using the build-in scripts. Before plotting these, the model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py).

```bash
evaluate_model.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml -e 5
```

The `-e` options (here `5`) allows to set the training epoch which should be evaluated.
It will produce .h5 and .pkl files with the evaluations which will be saved in the model folder in an extra folder called `results/`. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. For an explanation, look in the [plotting_umami documentation](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md)
