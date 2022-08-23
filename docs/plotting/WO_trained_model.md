# Evaluate and Plotting without a freshly trained Model
Although the UMAMI framework is made to evaluate and plot the results of the trainings of the taggers that are living inside of it, the framework can also evaluate and plot taggers that are already present in the files coming from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper).
The tagger results come from LWTNN models which are used to evaluate the jets in the derivations. The training-dataset-dumper applies these taggers and dumps the output probabilities for the different classes in the output .h5 files. These probabilities can be read by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script and can be evaluated like a freshly trained model.

To evaluate only the output files, there is a specific config file in the examples, which is called [evaluate_comp_taggers.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/training/evaluate_comp_taggers.yaml).
These can look for example like this:

```yaml
# Set foldername (aka modelname)
model_name: Eval_results

# Set the option to evaluate a freshly trained model to False
evaluate_trained_model: False

test_files:
    ttbar_r21:
        path: <path>/<to>/<preprocessed>/<samples>/ttbar_r21_test_file.h5
        variable_cuts:
            - pt_btagJes:
                operator: "<="
                condition: 250000

    ttbar_r22:
        path: <path>/<to>/<preprocessed>/<samples>/ttbar_r22_test_file.h5
        variable_cuts:
            - pt_btagJes:
                operator: "<="
                condition: 250000

    zpext_r21:
        path: <path>/<to>/<preprocessed>/<samples>/zpext_r21_test_file.h5
        variable_cuts:
            - pt_btagJes:
                operator: ">"
                condition: 250000

    zpext_r22:
        path: <path>/<to>/<preprocessed>/<samples>/zpext_r22_test_file.h5
        variable_cuts:
            - pt_btagJes:
                operator: ">"
                condition: 250000

# Values for the neural network
nn_structure:
    # Use evaluated tagger scores in h5 file and not trained model
    tagger: None

    # Define which classes are used for training
    # These are defined in the global_config
    class_labels: ["ujets", "cjets", "bjets"]

    # Main class which is to be tagged
    main_class: "bjets"

# Plotting settings for training metrics plots.
# Those are not used here. Only when running plotting_epoch_performance.py
validation_settings:

# Eval parameters for validation evaluation while training
evaluation_settings:
    # Number of jets used for validation
    n_jets: 3e5

    # Define taggers that are used for comparison in evaluate_model
    # This can be a list or a string for only one tagger
    tagger: ["rnnip", "DL1r"]

    # Define fc values for the taggers
    frac_values_comp:
        {
            "rnnip": {"cjets": 0.07, "ujets": 0.93},
            "DL1r": {"cjets": 0.018, "ujets": 0.982},
        }

    # Charm fraction value used for evaluation of the trained model
    frac_values: {"cjets": 0.018, "ujets": 0.982}

    # Working point used in the evaluation
    WP: 0.77
```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `model_name` | String | Necessary | Name of the model which is to be trained. Also the foldername where everything of the model will be saved. |
| `evaluate_trained_model` | Bool | Necessary | Needs to be `False` here. Otherwise the script tries to load the freshly trained model
| `test_files` | Dict | Optional | Here you can define different test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is relevant and is the unique identifier in the results file which is produced by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). `Path` gives the path to the file. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `nn_structure` | None | Necessary | A dict where all important information for the training are defined. |
| `class_labels` | List | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | String | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. |
| `evaluation_settings` | None | Necessary | A dict where all important information for the training are defined. |
| `n_jets` | Int | Necessary | Number of jets used for evaluation. This should not be to high, due to the fact that Callback function also uses this amount of jets after each epoch for validation. |
| `tagger` | List | Necessary | List of taggers used for comparison. This needs to be a list of string or a single string. The name of the taggers must be same as in the evaluation file. For example, if the DL1d probabilities in the test samples are called `DL1dLoose20210607_pb`, the name you need to add to the list is `DL1dLoose20210607`. |
| `frac_values_comp` | Dict | Necessary | Dict with the fraction values for the comparison taggers. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `frac_values` | Dict | Necessary | Dict with the fraction values for the freshly trained tagger. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `WP` | Float | Necessary | Working point that is used for evaluation. |

To run the evaluation, you can now execute the following command in the `umami/umami` folder where the `evaluate_model.py` is:

```bash
evaluate_model.py -c ../examples/training/evaluate_comp_taggers.yaml
```

The `evaluate_model.py` will now output a results file which has the results of your defined taggers inside. You can now use it like a regular one with a freshly trained model inside. An explanation how to plot the results is given in the [plotting_umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md) documentation.
