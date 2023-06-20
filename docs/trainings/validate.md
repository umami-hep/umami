## Validate your Training

After your training is finished, you first need to validate the overall training. For this specific task, Umami has a specific script called [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py). Depending if you ran the training with on the fly validation or without, the script validates the different models created after each epoch and plots the results in different plots.

### Config

The important part for the validation of your training is the `validation_settings` section. In there are all options set to validate and plot the validated results of your training. In the following example, the different options are shown/explained.
**Note** The validation plots are created using the [`puma package`](https://github.com/umami-hep/puma/). You can also give `puma` arguments, like `atlas_first_tag`, for the plots. These options are given to `puma` when the plotting is called.

```yaml
§§§examples/training/Dips-PFlow-Training-config.yaml:100:121§§§
```

| Options | Data Type | Necessary, Optional | Explanation |
|---------|-----------|---------------------|-------------|
| `n_jets` | `int` | Necessary | Number of jets to used per validation sample for validation. |
| `working_point` | `float` | Necessary | Working point which is used in the validation. This value is used to calculate the validation json with the `MyCallback` functions or when recalculating the validation json with the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `plot_datatype` | `str` | Necessary | Datatype of the plots that are produced using the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script. |
| `taggers_from_file` | `dict` | Optional | Dict of taggers that are available in the `.h5` samples. The here given taggers are plotted as reference lines in the rejection per epoch plots. The key of the dict is the name of the tagger inside the `.h5` samples. The value of the key must be a string with the label for the tagger for the validation plots. |
| `tagger_label` | `str` | Optional | Name for the legend of the freshly trained tagger for the validation plots. |
| `trained_taggers` | `dict` | Optional | A dict with local trained taggers which are to be plotted in the rejection per epoch plots. You need to provide a dict with a `path` and a `label`. The path is the path to the validation metrics `.json` file, where the rejections per epoch are saved. The `label` is the label which will be shown in the legend in the rejection per epoch plots. The `dipsReference` in the example here is just an internal naming. It will not be shown anywhere. |
| `use_atlas_tag` | `bool` | Optional | Decide, if the ATLAS tag is printed at the top left of the plot. |
| `atlas_first_tag` | `str` | Optional | Main ATLAS tag which is right to "ATLAS" |
| `atlas_second_tag` | `str` | Optional | Second line below the ATLAS tag |
| `val_batch_size` | `int` | Optional | Number of jets used per batch for the validation of the training. If not given, the batch size from `nn_structure` is used. |

### Running the Validation

Before running the validation, please ensure that the `frac_dict` and `frac_dict_comp` option in `evaluation_settings` is set for all taggers you want to plot! The validation uses this fraction values also for the rejection per epoch calculation! For a more detailed description of `frac_dict` and `frac_dict_comp`, please have a look [here](evaluate.md). Also, check that the `n_jets` are properly set to a value larger than 0! A zero will produce errors.

After the config is prepared switch to the `umami/umami` folder and run the `plotting_epoch_performance.py` by executing the following command:

```bash
plotting_epoch_performance.py -c <path to train config file> --recalculate
```

The `-r` or `--recalculate` command line option activates the validation of the samples defined in `validation_samples`. The script will calculate performance values for each epoch trained and will save the results in a specific `.json` file which has some parameters in the name. An example name is `validation_WP0p77_300000jets_Dict.json`, where `0p77` is the working point used and `300000` is the number of jets used. After the calculations are done, the script will plot different validation plots like the accuracy, loss and rejection per epoch plots.

Once the `validation_WP0p77_300000jets_Dict.json` is created (either by the on the fly validation or the `--recalculate` option), you can simply re-run the plotting via the command:

```bash
plotting_epoch_performance.py -c <path to train config file>
```

If you want to plot another `.json` file, you can give the command line option `-d` or `--dict` and the path to the specific file. But please keep in mind that the script extracts the working point and number of jets used from the name of the file!

**Note** If you want to plot another sample, just add it to `validation_samples` an run the script again with the `--recalculate` option. Also, if the on-the-fly validation was activated, you don't need to run the script with the `--recalculate`. You can simply run the plotting.


