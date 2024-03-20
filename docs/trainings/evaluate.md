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
| `working_point` | `float` | Necessary | Working point which is used in the evaluation. In the evaluation step, this is the value used for the fraction scan. |
| `eff_min` | `float` | Optional | Minimal main class efficiency considered for ROC. |
| `eff_max` | `float` | Optional | Maximal main class efficiency considered for ROC. |
| `frac_step` | `float` | Optional | Step size of the fraction value scan. Please keep in mind that the fractions given to the background classes need to add up to one! All combinations that do not add up to one are ignored. If you choose a combination `frac_min`, `frac_max` or `frac_step` where the fractions of the background classes do not add up to one, you will get an error while running `evaluate_model.py` |
| `frac_min` | `float` | Optional | Minimal fraction value which is set for a background class in the fraction scan. |
| `frac_max` | `float` | Optional | Maximal fraction value which is set for a background class in the fraction scan. |
| `add_eval_variables` | `list` | Optional | A list of available variables which are to be added to the evaluation files. With this, variables can be added for the variable vs eff/rejection plots. |
| `eval_batch_size` | `int` | Optional | Number of jets used per batch for the evaluation of the training. If not given, the batch size from `nn_structure` is used. |
| `extra_classes_to_evaluate` | `list` | Optional | List with jet flavours that are also loaded for evaluation although the tagger is not trained on this class. With this option, you can test the taggers behaviour for classes it wasn't trained on. Note: This must be a `list` and you also only need to add extra classes that are not in `class_labels`! Also you need to add an entry to the `frac_values` for each class in this list with value `0` so the calculation of the discriminants work. |

??? info "DIPS"
    #### DIPS

    ```yaml
    §§§examples/training/Dips-PFlow-Training-config.yaml:145:146§§§
    ```

    | Options | Data Type | Necessary, Optional | Explanation |
    |---------|-----------|---------------------|-------------|
    | `calculate_saliency` | `bool` | Optional | Decide, if the saliency maps are calculated or not. This takes a lot of time and resources! |

### Running the Evaluation

After the config is prepared switch to the `umami/umami` folder and run the `evaluate_model.py` by executing the following command:

```bash
evaluate_model.py -c <path to train config file> -e <epoch to evaluate>
```

The `-e` options allows to define which epoch of the training is to be evaluated.

**Note** Depending on the number of jets which are used for evaluation, this can take some time to process! Also, the use of a GPU for evaluation is highly recommended to reduce the time needed for execution.

### Evaluate only the taggers inside the .h5 files (without a freshly trained model)

Although the UMAMI framework is made to evaluate and plot the results of the trainings of the taggers that are living inside of it, the framework can also evaluate and plot taggers that are already present in the files coming from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper).
The tagger results come from LWTNN models which are used to evaluate the jets in the derivations. The training-dataset-dumper applies these taggers and dumps the output probabilities for the different classes in the output .h5 files. These probabilities can be read by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script and can be evaluated like a freshly trained model.

To evaluate only the output files, there is a specific config file in the examples, which is called [evaluate_comp_taggers.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/training/evaluate_comp_taggers.yaml).

This file is shown here:

```yaml
§§§examples/training/evaluate_comp_taggers.yaml§§§
```

Most of the options are similar to the ones already explained, although a lot of them are missing because they are not needed here. The ones that are new are explained in the following table

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `evaluate_trained_model` | `bool` | Necessary | This options enables/disables the evaluation of a freshly trained model. By default, this value is `True` but if you want to evaluate only taggers already present in the `.h5` files, you need to set this option to `False`! |

Now you can simply run the `evaluate_model.py` script as described in the section above but without the `-e` option. The command would look like this:

```bash
evaluate_model.py -c <path to train config file>
```

The `evaluate_model.py` will now output a results file like the one from the "regular" usage of the scripts with the difference that only your defined taggers in `tagger` are present in the files and no freshly trained tagger. An explanation how to plot the results is given in [here](https://umami-docs.web.cern.ch/plotting/plotting_umami/).

### Explaining the importance of features with SHAPley (only for DL1*)

[SHAPley](https://github.com/slundberg/shap) is a framework that helps you understand how your training of a machine learning model is affected by the input variables, or in other words from which variables your model possibly learns the most. SHAPley is for now only usable when evaluating a DL1* version. You can run that by executing the command

```bash
evaluate_model.py -c <path to train config file> -e <epoch to evaluate> -s shapley
```

which will output a beeswarm plot into `modelname/plots/`. Each dot in this plot is for one whole set of features (or one jet). They are stacked vertically once there is no space horizontally anymore to indicate density. The colour map tells you what the actual value was that entered the model. The SHAP value is basically calculated by removing features, letting the model make a prediction and then observe what would happen if you introduce features again to your prediction. If you do this over all possible combinations you get estimates of a features impact to your model. This is what the x-axis (SHAP value) tells you: the on average(!) contribution of a variable to an output node you are interested in (default is the output node for $b$-jets). In practice, large magnitudes (which is also what these plots are ordered by default in umami) are great, as they give the model a better possibility to discriminate. Features with large negative SHAP values therefore will help the model to better reject, whereas features with large positive SHAP values helps the model to learn that these are most probably jets from the category of interest. If you want to know more about SHAPley values, here is a [talk](https://indico.cern.ch/event/1071129/#4-shapely-for-nn-input-ranking) from one of our FTAG algorithm meeting.

You have some options to play with in the `evaluation_settings` section in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/training/DL1r-PFlow-Training-config.yaml) shown here:

```yaml
§§§examples/training/DL1r-PFlow-Training-config.yaml:154:175§§§
```

The options are explained here:

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `shapley` | `dict` | Optional | `dict` with the options for the feature importance explanation with SHAPley |
| `feature_sets` | `int` | Optional | Over how many full sets of features it should calculate over. Corresponds to the dots in the bee swarm plot. 200 takes like 10-15 min for DL1r on a 32 core-cpu. |
| `model_output` | `int` | Optional | Defines which of the model outputs (flavour) you want to explain. This is the index of the flavour in `class_labels`. |
| `bool_all_flavor_plot` | `bool` | Optional | You can also choose if you want to plot the magnitude of feature importance for all output nodes (flavors) in another plot. This will give you a bar plot of the mean SHAPley value magnitudes. |
| `averaged_sets` | `int` | Optional | As this takes much longer you can average the feature_sets to a smaller set, 50 is a good choice for DL1r. |
| `plot_size` | `list` | Optional | Figure size of the SHAPley plot. This is a list with `[width, height]` |
