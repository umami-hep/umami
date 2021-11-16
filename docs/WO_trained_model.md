# Evaluate and Plotting without a freshly trained Model
Although the UMAMI framework is made to evaluate and plot the results of the trainings of the taggers that are living inside of it, the framework can also evaluate and plot taggers that are already present in the files coming from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper).   
The tagger results come from LWTNN models which are used to evaluate the jets in the derivations. The training-dataset-dumper applies these taggers and dumps the output probabilities for the different classes in the output .h5 files. These probabilities can be read by the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py) script and can be evaluated like a freshly trained model.

To evaluate only the output files, there is a specific config file in the examples, which is called [evalute_comp_taggers.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/evalute_comp_taggers.yaml).   
These can look for example like this:

```yaml
# Set foldername (aka modelname)
model_name: Eval_results

# Set the option to evaluate a freshly trained model to False
evaluate_trained_model: False

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

# Values for the neural network
NN_structure:
    # Define which classes are used for training
    # These are defined in the global_config
    class_labels: ["ujets", "cjets", "bjets"]

    # Main class which is to be tagged
    main_class: "bjets"

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

    # Cuts which are applied to the different datasets used for evaluation
    variable_cuts: {
        "ttbar_r21": {
            "pt_btagJes": {
                "operator": "<=",
                "condition": 250000,
            }
        },
        "ttbar_r22": {
            "pt_btagJes": {
                "operator": "<=",
                "condition": 250000,
            }
        },
        "zpext_r21": {
            "pt_btagJes": {
                "operator": ">",
                "condition": 250000,
            }
        },
        "zpext_r22": {
            "pt_btagJes": {
                "operator": ">",
                "condition": 250000,
            }
        },
    }

# Plotting settings for training metrics plots
Plotting_settings:
    # Enable/Disable atlas tag
    UseAtlasTag: True

    # fc_value and WP_b are autmoatically added to the plot label
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow jets"

    # Set the datatype of the plots
    plot_datatype: "pdf"

```

| Options | Data Type | Necessary/Optional | Explanation |
|---------|-----------|--------------------|-------------|
| `model_name` | String | Necessary | Name of the model which is to be trained. Also the foldername where everything of the model will be saved. |
| `evaluate_trained_model` | Bool | Necessary | Needs to be `False` here. Otherwise the script tries to load the freshly trained model
| `ttbar_test_files` | Dict | Optional | Here you can define different ttbar test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is irrelevant while the `Path` and `data_set_name` are important. The `data_set_name` needs to be unique. Its the identifier/name of the dataset in the evaluation file which is used for plotting. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `zpext_test_files` | Dict | Optional | Here you can define different zpext test samples that are used in the [`evaluate_model.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). Those test samples need to be defined in a dict structure shown in the example. The name of the dict entry is irrelevant while the `Path` and `data_set_name` are important. The `data_set_name` needs to be unique. Its the identifier/name of the dataset in the evaluation file which is used for plotting. For test samples, all samples from the training-dataset-dumper can be used without preprocessing although the preprocessing of Umami produces test samples to ensure orthogonality of the jets with respect to the train sample. |
| `NN_structure` | None | Necessary | A dict where all important information for the training are defined. |
| `class_labels` | List | Necessary | List of flavours used in training. NEEDS TO BE THE SAME AS IN THE `preprocess_config`. Even the ordering needs to be the same! |
| `main_class` | String | Necessary | Main class which is to be tagged. Needs to be in `class_labels`. |
| `Eval_parameters_validation` | None | Necessary | A dict where all important information for the training are defined. |
| `n_jets` | Int | Necessary | Number of jets used for evaluation. This should not be to high, due to the fact that Callback function also uses this amount of jets after each epoch for validation. | 
| `tagger` | List | Necessary | List of taggers used for comparison. This needs to be a list of string or a single string. The name of the taggers must be same as in the evaluation file. For example, if the DL1d probabilities in the test samples are called `DL1dLoose20210607_pb`, the name you need to add to the list is `DL1dLoose20210607`. |
| `frac_values_comp` | Dict | Necessary | Dict with the fraction values for the comparison taggers. For all flavour (except the main flavour), you need to add values here which add up to one. |
| `variable_cuts` | Dict | Necessary | Dict of cuts which are applied when loading the different test files. Only jet variables can be cut on. |

To run the evaluation, you can now execute the following command in the `umami/umami` folder where the `evaluate_model.py` is:

```bash
evaluate_model.py -c ../examples/Dips-PFlow-Training-config.yaml
```

The `evaluate_model.py` will now output a results file which has the results of your defined taggers inside. You can now use it like a regular one with a freshly trained model inside. An explanation how to plot the results is given in the [plotting_umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md) documentation.
