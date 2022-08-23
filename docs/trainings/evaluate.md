## Evaluate your Training

After you validated your training and found an epoch you want to use for a more detailed check, the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script comes in. It evaluates the given model with the samples defined in `test_samples` and writes the results in a `results/` folder in the model folder. The results can then be visualised/plotted using the [`plotting_umami.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script. A detailed explanation how to use this script is given [here](../plotting/plotting_umami.md).

### Config

The important part for the evaluation is the `evaluation_settings` section. In there are all options set to evaluate your model. In the following example, the different options are shown/explained. There are some specific options only available for some taggers. These options can be found in their respective subsections.

```yaml
§§§examples/training/Dips-PFlow-Training-config.yaml:124:143§§§
```

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `results_filename_extension` | `str` | Optional | String which is added to the filenames of the several files created when evaluating. This allows to re-evaluate without overwriting old results. Make sure you specify the `evaluation_file` when plotting the corresponding results, otherwise the plotting script will look for files without the extension. |
| `n_jets` | `int` | Necessary | Number of jets per sample used for evaluation. |
| `tagger` | `list` | Necessary | List of taggers used for comparison. This needs to be a list of `str` or a single `str`. The name of the taggers must be same as in the evaluation file. For example, if the DL1d probabilities in the test samples are called `DL1dLoose20210607_pb`, the name you need to add to the list is `DL1dLoose20210607`. |
| `frac_values_comp` | `dict` | Necessary | `dict` with the fraction values for the comparison taggers. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `frac_values` | `dict` | Necessary | `dict` with the fraction values for the freshly trained tagger. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `WP` | `float` | Necessary | Working point which is used in the evaluation. In the evaluation step, this is the value used for the fraction scan. |
| `eff_min` | `float` | Optional | Minimal main class efficiency considered for ROC. |
| `eff_max` | `float` | Optional | Maximal main class efficiency considered for ROC. |
| `frac_step` | `float` | Optional | Step size of the fraction value scan. Please keep in mind that the fractions given to the background classes need to add up to one! All combinations that do not add up to one are ignored. If you choose a combination `frac_min`, `frac_max` or `frac_step` where the fractions of the background classes do not add up to one, you will get an error while running `evaluate_model.py` |
| `frac_min` | `float` | Optional | Minimal fraction value which is set for a background class in the fraction scan. |
| `frac_max` | `float` | Optional | Maximal fraction value which is set for a background class in the fraction scan. |
| `add_eval_variables` | `list` | Optional | A list to add available variables to the evaluation files. With this, variables can be added for the variable vs eff/rejection plots. |
| `eval_batch_size` | `int` | Optional | Number of jets used per batch for the evaluation of the training. If not given, the batch size from `nn_structure` is used. |

??? info "DIPS"
    #### DIPS

    ```yaml
    §§§examples/training/Dips-PFlow-Training-config.yaml:145:146§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `calculate_saliency` | `bool` | Optional | Decide, if the saliency maps are calculated or not. This takes a lot of time and resources! |

??? info "DL1*"
    #### DL1*

    The following options are used in the feature importance check. A detailed description how to run this, please have a look [here](Feature_Importance.md)

    ```yaml
    §§§examples/training/DL1r-PFlow-Training-config.yaml:154:175§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `shapley` | `dict` | Optional | `dict` with the options for the feature importance explanation with SHAPley |
    | `feature_sets` | `int` | Optional | Over how many full sets of features it should calculate over. Corresponds to the dots in the bee swarm plot. 200 takes like 10-15 min for DL1r on a 32 core-cpu. |
    | `model_output` | `int` | Optional | Defines which of the model outputs (flavour) you want to explain. This is the index of the flavour in `class_labels`. |
    | `bool_all_flavor_plot` | `bool` | Optional | You can also choose if you want to plot the magnitude of feature importance for all output nodes (flavors) in another plot. This will give you a bar plot of the mean SHAPley value magnitudes. |
    | `averaged_sets` | `int` | Optional | As this takes much longer you can average the feature_sets to a smaller set, 50 is a good choice for DL1r. |
    | `plot_size` | `list` | Optional | Figure size of the SHAPley plot. This is a list with `[width, height]` |

### Running the Evaluation

After the config is prepared switch to the `umami/umami` folder and run the `evaluate_model.py` by executing the following command:

```bash
evaluate_model.py -c <path to train config file> -e <epoch to evaluate>
```

The `-e` options allows to define which epoch of the training is to be evaluated.

**Note** Depending on the number of jets which are used for evaluation, this can take some time to process! Also, the use of a GPU for evaluation is highly recommended to reduce the time needed for execution.