## Start Training your Model

After all files are preprocessed, we can start with the training. In the following sections, the global settings and the neural network settings are explained more in detail. If you want to train completely fresh, you can just adapt one of the example config files. The creation of the model folder is taken care of when we start the actual training!

### Global Settings

The global settings are the base settings needed for every training. Here we define general things like the model name and which preprocessing config file was used to produce the train dataset.

```yaml
§§§examples/training/Dips-PFlow-Training-config.yaml:1:58§§§
```

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `model_name` | `str` | Necessary | Name of the model you want to train. This will be the name of the folder, where all results etc. will be saved in. This folder will automatically be created if not existing. |
| `preprocess_config` | `str` | Necessary | Path to the preprocessing config you used to produce the train datasets. When you start the training and the folder for the model is created, this file is copied to the `metadata/` folder inside the model folder. Also, the path here in the train config will be changed to the new path of the preprocessing config inside the `metadata/` folder. |
| `model_file` | `str` | Optional | If you already have a model and want to use the weights of this model as start point (maybe you have a R21 trained model and now you want to use the weights of that model as initial weights for your R22 training), you can give the path to this model here. This model will be loaded and used instead of init a new one. If you don't set `load_optimiser` in `nn_structure`, the optimiser state will be resetted. If you just want to continue a specific training, use `continue_training` and leave this option empty. |
| `train_file` | `str` | Necessary | Path to the training sample. This is given by the `preprocessing` step of Umami. One can also train with a TDD file format that is available after resampling but before writing step. The scaling and shifting dict will be taken from the path in preprocessing configs automatically. If you want to use the `TFRecords` format to train, this must be the path to the folder where the `TFRecords` files are saved. |
| `continue_training` | `bool` | Optional | If your training died due to time constrains of jobs and you just want to continue the training from the latest point on, set this value to `True` and restart the training. |
| `exclude` | `list` | Necessary | List of jet variables that are excluded from training. Only compatible with DL1* training!. To include all, just set this option to `null`. If you don't train DL1* also set this just to `null`. |
|`tracks_name`| `str` | Necessary | Name of the tracks data-set to use for training and evaluation, default is "tracks". If you are training DL1*, just remove this option. <br /> ***This option is necessary when using tracks, but, when working with old preprocessed files (before January 2022, Tag 05 or older) this option has to be removed form the config file to ensure compatibility*** |

Two uncovered options are the `validation_files` and the `test_files`. Here we simply define the files which are later used for the validation/evaluation step. It is possible to define multiple validation files using the dict structure shown in the example. Focusing on the `validation_files` first, each entry needs a unique name. This name is the internal identifier for this specific file. The options defined in this entry are explained here:

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `path` | `str` | Necessary | Path to the validation/test file which is to be used. Using wildcards is possible. |
| `label` | `str` | Necessary | Label which is used for this file when plotting the validation plots. |
| `variable_cuts` | `dict` | Optional | `dict` of cuts which are applied when loading the different test files. Only jet variables can be cut on. These are in this example defined as templates for the different samples types. |

Focusing now on the `test_files`, the options are the same but no `label` is needed, because the plotting of the evaluation results is an extra step covered [here](../plotting/plotting_umami.md).

Both options can also be added after the training is finished. For the training itself, you can leave this options blank.

### Network Settings

The next section in the train config is the `nn_structure`. Here we define all the information needed for building the model, like which tagger we want to use and also the number of hidden layer and nodes per hidden layer. The general options are shown next while the tagger dependant options are shown in their respective subsections.

```yaml
§§§examples/training/Dips-PFlow-Training-config.yaml:61:83§§§
§§§examples/training/Dips-PFlow-Training-config.yaml:88:97§§§
```

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `tagger` | `str` | Necessary | Name of the tagger that is used/to be trained. The currently supported taggers are `dips`, `dips_attention`, `cads`, `dl1`, `umami`, `umami` and `umami_cond_att`. **Note** All version of DL1* (like DL1r or DL1d) uses the `tagger` `dl1`! |
| `learning_rate` | `float` | Necessary | Learning rate which is used for training. |
| `batch_size` | `int` | Necessary | Batch size which is used for training. |
| `epochs` | `int` | Necessary | Number of epochs of the training. |
| `n_jets_train` | `int` | Necessary | Number of jets used for training. Leave empty to use all. |
| `class_labels` | `list` | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | `str` or `list` of `str` | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. This can either be one single class (`str`) or multiple classes (`list` of `str`). |
| `batch_normalisation` | `bool` | Necessary | Decide, if batch normalisation is used in the network. (Look in the model files where this is used for the specific models) |
| `dense_sizes` | `list` | Necessary | List of nodes per layer of the network. Every entry is one layer. The numbers need to be `int`! For DL1r/DL1d, this is the number of nodes per layer. For DIPS/DIPS Attention/Umami/CADS this is the number of nodes per layer for the _F_ network. |
| `dropout_rate` | `list` | List of dropout rates for the layers defined via `dense_sizes`. Has to be of the same length as the `dense_sizes` list. |
| `load_optimiser` | `bool` | Optional | When loading a model (via `model_file`), you can load the optimiser state for continuing a training (`True`) or initialize a new optimiser to use the model as a start point for a fresh training (`False`). |
| `use_sample_weights` | `bool` | Optional | Applies the weights, you calculated with the `--weighting` flag from the preprocessing to the training loss function. |
| `nfiles_tfrecord` | `int` | Optional | Number of files that are loaded at the same time when using `TFRecords` for training. |
| `lrr` | `bool` | Optional | Decide, if a Learning Rate Reducer (lrr) is used or not. If yes, the following options can be added. |
| `lrr_monitor` | `str` | Optional | Quantity to be monitored. Default: "loss" |
| `lrr_factor` | `float` | Optional | Factor by which the learning rate will be reduced. `new_lr = lr * factor`. Default: 0.8 |
| `lrr_patience` | `int` | Optional | Number of epochs with no improvement after which learning rate will be reduced. Default: 3 |
| `lrr_verbose` | `int` | Optional | 0: Quiet, 1: Update messages. Default: 1 |
| `lrr_mode` | `str` | Optional | One of `{"auto", "min", "max"}`. In "min" mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in "max" mode it will be reduced when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Default: "auto" |
| `lrr_cooldown` | `int` | Optional | Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 5 |
| `lrr_min_learning_rate` | `float` | Optional | Lower bound on the learning rate. Default: 0.000001 |

??? info "DIPS"
    #### DIPS

    ```yaml
    §§§examples/training/Dips-PFlow-Training-config.yaml:84:86§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `ppm_sizes` | `list` | Necessary | List of nodes per layer of the _ϕ_ network. Every entry is one layer. The numbers need to be `int`! |
    | `dropout_rate_phi` | `list` | Necessary | List of dropout rates in the _ϕ_ network. Has to be of the same length as the `ppm_sizes` list. |

??? info "DL1*"
    #### DL1*

    ```yaml
    §§§examples/training/DL1r-PFlow-Training-config.yaml:89:93§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `activations` | `list` | Necessary | List of activations per layer defined in `dense_sizes`. Every entry is the activation for one hidden layer. The entries must be `str` and activations supported by `Keras`. |
    | `repeat_end` | `list` | Optional | List of input variables that are folded into the output of the penultimate layer. This is then feeded into the last layer. |

??? info "Umami"
    #### Umami

    ```yaml
    §§§examples/training/umami-PFlow-Training-config.yaml:76:100§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `dips_ppm_units` | `list` | Necessary | Similar to DIPS `ppm_sizes`. List of nodes per layer of the _ϕ_ network. Every entry is one layer. The numbers need to be `int`! |
    | `dips_dense_units` | `list` | Necessary | Similar to DIPS `dense_sizes`. List of nodes per layer of the _F_ network. Every entry is one layer. The numbers need to be `int`! |
    | `intermediate_units` | `list` | Necessary | These are the layers that will be concatenated with the last layer of `dips_dense_units`. Every entry is one layer. The numbers need to be `int`! |
    | `dl1_units` | `list` | Necessary | Similar to DL1+ `dense_sizes`. List of nodes per layer of the DL1-like network. Every entry is one layer. The numbers need to be `int`! |
    | `dips_loss_weight` | `float` or `int` | Necessary | Loss weight $w_{\text{DIPS}}$ for the DIPS loss. While training Umami, two losses are obtained: The final Umami loss and the DIPS loss. This value is the factor how important the DIPS loss is for the final model loss. $\text{Loss}_{\text{Total}} = \text{Loss}_{\text{Umami}} + w_{\text{DIPS}} * \text{Loss}_{\text{DIPS}}$. |

??? info "DIPS Attention/CADS"
    #### DIPS Attention/CADS

    The different between the `tagger` here is the `*_condition` options. If all `*_condition` options are `False`, the tagger to use is `dips_attention` while if one `*_condition` option is `True`, the tagger to use is `cads`.

    ```yaml
    §§§examples/training/CADS-PFlow-Training-config.yaml:84:111§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `ppm_sizes` | `list` | Necessary | Similar to DIPS `ppm_sizes`. List of nodes per layer of the _ϕ_ network. Every entry is one layer. The numbers need to be `int`! |
    | `ppm_condition` | `bool` | Necessary | If you want to use/fold the conditional information into the input of the _ϕ_ network. |
    | `dense_sizes` | `list` | Necessary | Similar to DIPS `dense_sizes`. List of nodes per layer of the _F_ network. Every entry is one layer. The numbers need to be `int`! |
    | `dense_condition` | `bool` | Necessary | If you want to use/fold the conditional information into the input of the _F_ network. |
    | `n_conditions` | `int` | Necessary | Number of conditional jet input variables to use for CADS. |
    | `pooling` | `string` | Necessary | Pooling method that is used to pool the output of the _ϕ_ and _A_ networks. |
    | `attention_sizes` | `string` | Necessary | Similar to `ppm_sizes`. List of nodes per layer of the _A_ network. Every entry is one layer. The numbers need to be `int`! |
    | `attention_condition` | `bool` | Necessary | If you want to use/fold the conditional information into the input of the _A_ network. |

### Running the Training

After the global- and network settings are prepared, you can start training your model. To start the training, switch to the `umami/umami` folder and run the following command:

```bash
train.py -c <path to train config file> --prepare
```

This command will not directly start the training, but will prepare the model folder with all needed configs/scale dicts etc. Before starting the real training, you should check all config files again. The new folder will have the name given in `model_name`. In there is a folder called `metadata` in which all configs/scale dicts etc. were copied. Also, all paths in the config are adapted to the metadata folder, like the path to the preprocessing config etc.
After you checked everything, you can run the real training via the command:

```bash
train.py -c <path to train config file>
```

This will start the real training. Other command line arguments available for `train.py` are `-o` which will overwrite the configs/dicts in metadata if you run the training.

**Note** When training, the callback methods of the different taggers validate the training on the fly which can lead to memory issues. To deactivate the on the fly validation, what we recommend, just set the `n_jets` option in the `validation_settings` section of the train config to `0`.
