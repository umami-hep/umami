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

# Decide, if taus are used or not
bool_use_taus: False

ttbar_test_files:
    ttbar_r21:
        Path: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_1.h5
        data_set_name: "ttbar"

    ttbar_r22:
        Path: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids_r22/MC16d_hybrid-r22_odd_100_PFlow-no_pTcuts-file_1.h5
        data_set_name: "ttbar_comparison"

zpext_test_files:
    zpext_r21: 
        Path: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_1.h5
        data_set_name: "zpext"

    zpext_r22:
        Path: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids_r22/MC16d_hybrid-r22-ext_odd_0_PFlow-no_pTcuts-file_1.h5
        data_set_name: "zpext_comparison"

    zpext_r22_no_QSP:
        Path: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids_r22/MC16d_hybrid-r22-ext_odd_0_PFlow-no_pTcuts_No_QSPI-file_1.h5
        data_set_name: "zpext_comparison_no_QSP"

# Eval parameters for validation evaluation while training
Eval_parameters_validation:
    # Number of jets used for validation
    n_jets: 3e5

    # Define taggers that are used for comparison in evaluate_model
    # This can be a list or a string for only one tagger
    tagger: ["rnnip", "DL1r"]

    # Define fc values for the taggers
    fc_values_comp: {
        "rnnip": 0.08,
        "DL1r": 0.018,
    }
```

The first option `model_name` sets the name of the output folder where the results of the evaluation are saved. Also you need to decide if you want to include taus (Note: You can only use taus if they are in the samples you want to evaluate!).   
The `evaluate_trained_model` needs to be `False` here! Otherwise the script will try to load and evaluate the freshly trained model that is not available.   
Very important are the `ttbar_test_files` and `zpext_test_files`. Here you can give the files you want to evaluate (Wildcarding is enabled) and give them a unique `data_set_name`. Which results to plot later will be decided by this names!

Now the important part, the `Eval_parameters_validation`. Here we can define the options for the evaluation.   
First, the number of jets `n_jets`. This is the max number of jets that are loaded (also from multiple files).   
The taggers you want to evaluate are defined in the `taggers` option. This needs to be a list of strings! Also the names in there need to be the same as in the files you want to evaluate. I.e. you have `DL1r_pb`, `DL1r_pc` and `DL1r_pu` in your files, the correct name of the tagger for the list would be `"DL1r"`. The script is loading the tagger probabilities like `TAGGER_pb`.   
Another part is the `fc_values_comp`. For each of your taggers, you can define a specific `fc` value (the "normal" `fc` value is not used if `evaluate_trained_model` is `False`)

To run the evaluation, you can now execute the following command in the `umami/umami` folder where the `evaluate_model.py` is:

```bash
evaluate_model.py -c ../examples/Dips-PFlow-Training-config.yaml
```

The `evaluate_model.py` will now output a results file which has the results of your defined taggers inside. You can now use it like a regular one with a freshly trained model inside. An explanation how to plot the results is given in the [plotting_umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md) documentation.
