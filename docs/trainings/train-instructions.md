# Instructions to Train the different Taggers with the UMAMI Framework
The following instructions are meant to give a guidline how to train the different taggers currently living in the UMAMI framework. Currently supported taggers are

* DL1r / DL1d
* DIPS
* Umami
* DIPS Attention / CADS

## Sample Preparation
Before we can start training the different taggers, we need to produce our training,
validation and test datasets. This done using the preprocessing, which is explained
[here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/preprocessing.md).

For the different taggers, different information need to be inside the preprocessed
samples. While DL1r / DL1d and DIPS work on one specific information set (DL1r/DL1d: jet information, DIPS/DIPS Attention: track information), Umami and CADS need both information. Due to the fact that the jet information are always preprocessed (due to truth info needed), you need to check if you need track information. If this is the case, you need to set the `save_tracks` option to `True`. The rest of the preprocessing (with example files etc.) is explained [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/preprocessing.md).

## Train Config

After all files are preprocessed, we can start with the training. The train config files for the different trainings can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/examples). While the basic options needed/provided inside this config files are the same for all taggers, some options are only available for some other. A list with all options/explanations, if the option is necessary or optional and for which tagger the option can be used, is provided here.

### Global Settings

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `model_name` | All | `str` | Necessary | Name of the model you want to train. This will be the name of the folder, where all results etc. will be saved in. This folder will automatically be created if not existing. |
| `preprocess_config` | All | `str` | Necessary | Path to your preprocess config you used producing your train datasets. When you start the training and the folder for the model is created, this file is copied to the `metadata/` folder inside the model folder. Also, the path here in the train config will be changed to the new path of the preprocess config inside the `metadata/` folder. |
| `model_file` | All | `str` | Optional | If you already have a model and want to use the weights of this model as start point, you can give the path to this model here. This model will be loaded and used instead of init a new one. If you don't set `load_optimiser` in `NN_structure`, the optimiser state will be resetted. If you just want to continue a specific training, use `continue_training` and leave this option empty. |
| `continue_training` | All | `bool` | Optional | If your training died due to time constrains of jobs and you just want to continue the training from the latest point on, set this value to `True`. |
| `train_file` | All | `str` | Necessary | Path to the training sample. This is given by the `preprocessing` step of Umami. If you want to use the tfrecords format to train, this must be the path to the folder where the tfrecords files are saved. |
| `var_dict` | All | `str` | Necessary | Path to the variable dict used in the `preprocess_config` to produce the train sample. |
| `exclude` | DL1r, DL1d | `list` | Necessary | List of variables that are excluded from training. Only compatible with DL1r training. To include all, just give an empty list. |
|`tracks_name`| DIPS, DIPS Attention, Umami, CADS | `str` | Necessary* | Name of the tracks data-set to use for training and evaluation, default is "tracks".  <br />* ***This option is necessary when using tracks, but, when working with old preprpocessed files (before January 2022, Tag 05 or older)  this option has to be removed form the config file to ensure compatibility*** |

### validation_files

Here you can define different validation samples that are used in the training and the `plotting_epoch_performance.py` script. Those validation samples need to be defined in a dict structure shown in the example. The name of the dict entry is relevant and is the unique identifier for this sample (DO NOT USE IT MULTIPLE TIMES). `path` gives the path to the file. If you don't want to use validation files, leave this blank.

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `path` | All | `str` | Necessary | Path to the validation/test file which is to be used. Using wildcards is possible. |
| `variable_cuts` | All | `dict` | Optional | `dict` of cuts which are applied when loading the different test files. Only jet variables can be cut on. These are in this example defined as templates for the different samples types. |

### test_files

Here you can define different test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is relevant and is the unique identifier in the results file which is produced by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). `Path` gives the path to the file. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. If you don't want to use test files, leave this blank.

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `path` | All | `str` | Necessary | Path to the validation/test file which is to be used. Using wildcards is possible. |
| `variable_cuts` | All | `dict` | Optional | `dict` of cuts which are applied when loading the different test files. Only jet variables can be cut on. These are in this example defined as templates for the different samples types. |

### NN_Structure

Here all important information about the tagger you want to train are defined. 

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `tagger` | All | `str` | Necessary | Name of the tagger that is used/to be trained. |
| `load_optimiser` | All | `bool` | Optional | When loading a model (via `model_file`), you can load the optimiser state for continuing a training (`True`) or initialize a new optimiser to use the model as a start point for a fresh training (`False`). |
| `lr` | All | `float` | Necessary | Learning rate which is used for training. |
| `batch_size` | All | `int` | Necessary | Batch size which is used for training. |
| `epochs` | All | `int` | Necessary | Number of epochs of the training. |
| `nJets_train` | All | `int` | Necessary | Number of jets used for training. Leave empty to use all. |
| `dropout` | All | `float` | Necessary | Dropout factor used in the _ϕ_ network. If 0, dropout is not used. |
| `class_labels` | All | `list` | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | All | `str` or `list` of `str` | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. This can either be one single class (`str`) or multiple classes (`list` of `str`). |
| `Batch_Normalisation` | All | `bool` | Necessary | Decide, if batch normalisation is used in the network. (Look in the model files where this is used for the specific models) |
| `ppm_sizes` | DIPS, DIPS Attention, Umami, CADS | `list` | Necessary | List of nodes per layer of the _ϕ_ network. Every entry is one layer. The numbers need to be ints! |
| `dense_sizes` | All | `list` | Necessary | List of nodes per layer of the _F_ network (DIPS/DIPS Attention/Umami/CADS). Every entry is one layer. The numbers need to be ints! For DL1r/DL1d, this is the number of nodes per layer. |
| `use_sample_weights` | All | `bool` | Optional | Applies the weights, you calculated with the `--weighting` flag from the preprocessing to the training loss function. |
| `nfiles_tfrecord` | All | `int` | Optional | Number of files that are loaded at the same time when using tfrecords for training. |
| `LRR` | All | `bool` | Optional | Decide, if a Learning Rate Reducer (LRR) is used or not. If yes, the following options can be added. |
| `LRR_monitor` | All | `str` | Optional | Quantity to be monitored. Default: "loss" |
| `LRR_factor` | All | `float` | Optional | Factor by which the learning rate will be reduced. `new_lr = lr * factor`. Default: 0.8 |
| `LRR_patience` | All | `int` | Optional | Number of epochs with no improvement after which learning rate will be reduced. Default: 3 |
| `LRR_verbose` | All | `int` | Optional | 0: Quiet, 1: Update messages. Default: 1 |
| `LRR_mode` | All | `str` | Optional | One of `{"auto", "min", "max"}`. In "min" mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in "max" mode it will be reduced when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Default: "auto" |
| `LRR_cooldown` | All | `int` | Optional | Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 5 |
| `LRR_min_lr` | All | `float` | Optional | Lower bound on the learning rate. Default: 0.000001 |

### Validation_metrics_settings

Here are all important settings defined for the validation process (Validation while training and validation via the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script.

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `n_jets` | All | `int` | Necessary | Number of jets to used for validation. |
| `taggers_from_file` | All | `dict` | Optional | Dict of taggers that are available in the .h5 samples. The here given taggers are plotted as reference lines in the rejection per epoch plots. The key of the dict is the name of the tagger inside the .h5 samples. The value of the key must be a string with the label for the tagger for the valdation plots. |
| `tagger_label` | All | `str` | Optional | Name for the legend of the freshly trained tagger for the rejection per epoch plots. |
| `trained_taggers` | All | `dict` | Optional | A dict with local trained taggers which shall be plotted in the rejection per epoch plots. You need to provide a dict with a `path` and a `label`. The path is the path to the validation metrics .json file, where the rejections per epoch are saved. The `label` is the label which will be shown in the legend in the rejection per epoch plots. The `dipsReference` in the example here is just an internal naming. It will not be shown anywhere. |
| `WP` | All | `float` | Necessary | Working point which is used in the validation. This value is used to calculate the validation json with the `MyCallback` functions or when recalculating the validation json with the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `UseAtlasTag` | All | `bool` | Optional | Decide, if the ATLAS tag is printed at the top left of the plot. |
| `AtlasTag` | All | `str` | Optional | Main ATLAS tag which is right to "ATLAS" |
| `SecondTag` | All | `str` | Optional | Second line below the ATLAS tag |
| `plot_datatype` | All | `str` | Necessary | Datatype of the plots that are produced using the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `val_batch_size` | All | `int` | Optional | Number of jets used per batch for the validation of the training. If not given, the batch size from `NN_structure` is used. |

### Eval_parameters_validation

Here are all important settings defined for the evaluation process (evaluating via the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script.

| Options | Tagger | Data Type | Necessary, Optional | Explanation |
|---------|--------|-----------|---------------------|-------------|
| `results_filename_extension` | All | `str` | Optional | String which is added to the filenames of the several files created when evaluating. This allows to re-evaluate without overwriting old results. Make sure you specify the `evaluation_file` when plotting the corresponding results, otherwise the plotting script will look for files without the extension. |
| `n_jets` | All | `int` | Necessary | Number of jets used for evaluation. |
| `tagger` | All | `list` | Necessary | List of taggers used for comparison. This needs to be a list of `str` or a single `str`. The name of the taggers must be same as in the evaluation file. For example, if the DL1d probabilities in the test samples are called `DL1dLoose20210607_pb`, the name you need to add to the list is `DL1dLoose20210607`. |
| `frac_values_comp` | All | `dict` | Necessary | `dict` with the fraction values for the comparison taggers. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `frac_values` | All | `dict` | Necessary | `dict` with the fraction values for the freshly trained tagger. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `WP` | All | `float` | Necessary | Working point which is used in the evaluation. In the evaluation step, this is the value used for the fraction scan. |
| `eff_min` | All | `float` | Optional | Minimal main class efficiency considered for ROC. |
| `eff_max` | All | `float` | Optional | Maximal main class efficiency considered for ROC. |
| `frac_step` | All | `float` | Optional | Step size of the fraction value scan. Please keep in mind that the fractions given to the background classes need to add up to one! All combinations that do not add up to one are ignored. If you choose a combination `frac_min`, `frac_max` or `frac_step` where the fractions of the brackground classes do not add up to one, you will get an error while running `evaluate_model.py` |
| `frac_min` | All | `float` | Optional | Minimal fraction value which is set for a background class in the fraction scan. |
| `frac_max` | All | `float` | Optional | Maximal fraction value which is set for a background class in the fraction scan. |
| `Calculate_Saliency` | DIPS | `bool` | Optional | Decide, if the saliency maps are calculated or not. This takes a lot of time and resources! |
| `add_variables_eval` | DL1r, DL1d | `list` | Optional | A list to add available variables to the evaluation files. |
| `shapley` | DL1r, DL1d | `dict` | Optional | `dict` with the options for the feature importance explanation with SHAPley |
| `feature_sets` | DL1r, DL1d | `int` | Optional | Over how many full sets of features it should calculate over. Corresponds to the dots in the beeswarm plot. 200 takes like 10-15 min for DL1r on a 32 core-cpu. |
| `model_output` | DL1r, DL1d | `int` | Optional | Defines which of the model outputs (flavour) you want to explain. This is the index of the flavour in `class_labels`. |
| `bool_all_flavor_plot` | DL1r, DL1d | `bool` | Optional | You can also choose if you want to plot the magnitude of feature importance for all output nodes (flavors) in another plot. This will give you a bar plot of the mean SHAP value magnitudes. |
| `averaged_sets` | DL1r, DL1d | `int` | Optional | As this takes much longer you can average the feature_sets to a smaller set, 50 is a good choice for DL1r. |
| `plot_size` | DL1r, DL1d | `list` | Optional | Figure size of the SHAPley plot. This is a list with `[width, height]` |
| `eval_batch_size` | All | `int` | Optional | Number of jets used per batch for the evaluation of the training. If not given, the batch size from `NN_structure` is used. |

## Training
Before starting the training, you need to set some paths for the umami package to find all the tools. How to set this all up is explained [here](https://umami-docs.web.cern.ch/installation/).

Note: When working with `Singularity`, the `python install` option is not writable and therefore will fail. In this case, switch to the umami folder and run the following command.

```bash
source run_setup.sh
```

This will create links to the executables. Note: If you now change something in the files, this will immediately become active. A more detailed explanation can be found [here](https://umami-docs.web.cern.ch/installation/).

After that, you can switch to the folder `umami/umami` and run the training, using the following command

```bash
train.py -c <path>/<to>/<train>/<config>
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the train config.

If you only want to produce the output folder (`umami/umami/MODELNAME/`) with the metadata already copied and ready to start (for cross-checking) but not start the training, you can give the `-p` or `--prepare` option like this:

```bash
train.py -c <path>/<to>/<train>/<config> -p
```

## Plotting the Training Metrics
If you want to check how your model is performing while training, you can use the following command:

```bash
plotting_epoch_performance.py -c <path>/<to>/<train>/<config>
```

This will write out plots for the non-main flavour rejections, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. The values are taken from the validation json (the name depends on the values for number of jets and working point). If you renamed the validation json or you want to use another one, you can run:

```bash
plotting_epoch_performance.py -c <path>/<to>/<train>/<config> --dict <path>/<to>/<validation>/<json>
```

where the argument given after `--dict` is the path to the validation json you want to plot. The validation json mentioned here will be produced by the `MyCallback` function which is running on each epoch end. As a result, a json file will be filled with different metrics. The validation json is updated after each epoch. The file will be stored in the `umami/umami/MODELNAME/` folder.

If you want to recalculate this dict (with more jets for example), you can run:

```bash
plotting_epoch_performance.py -c <path>/<to>/<train>/<config> --recalculate
```

This will recalculate the  the performance measurements, like the rejection performances, using the working point, the `frac_values` and the number of validation jets defined in the train config. This can take a long time without a GPU, because each saved model is loaded and evaluated with the validation files. We strongly advice you to only do that if you changed the validation files!

## Evaluating the results
After the training is over, the different epochs can be evaluated with ROC plots, output scores, saliency maps and confusion matrices etc. using the build-in scripts. Before plotting these, a model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py).

```bash
evaluate_model.py -c <path>/<to>/<train>/<config> -e 5
```

The `-e` options (here `5`) allows to set the training epoch which should be evaluated.
It will produce .h5 and .pkl files with the evaluations which will be saved in the model folder in an extra folder called `results/`. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. For an explanation, look in the [plotting_umami documentation](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md).
