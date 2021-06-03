# Instructions to train DIPS with the umami framework

The following instructions are meant to give a guidline how to reproduce the DIPS results presented in the [DIPS Note](https://cds.cern.ch/record/2718948). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets

After the previous step the ntuples need to be further processed. We use an undersampling approach to achieve the same pt and eta distribution for all three flavour categories.
In order to reduce the memory usage we first extract the 3 jet categories separately since e.g. c-jets only make up 8% of the ttbar sample. Note that tau jets are not yet supported in DIPS.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](preprocessing.md) for additional information.
Note, that for running Dips tracks have to be stored in the output hybrid sample. Therefore, the `--tracks` argument needs to be set.

## Training

After all the files are ready we can start with the training. The config file for the Dips training is [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). This will look for example like this:

```yaml
# Set modelname and path to Pflow preprocessing config file
model_name: dips_lr_0.001_bs_15000_epoch_200_nTrainJets_Full
preprocess_config: /work/ws/nemo/fr_af1100-Training-Simulations-0/b-Tagging/packages/umami/examples/PFlow-Preprocessing.yaml

# Add training file
train_file: /work/ws/nemo/fr_af1100-Training-Simulations-0/preprocessed/PFlow-hybrid-preprocessed_shuffled.h5

# Add validation files
# ttbar val
validation_file: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids/MC16d_hybrid_odd_100_PFlow-no_pTcuts-file_0.h5

# zprime val
add_validation_file: /work/ws/nemo/fr_af1100-Training-Simulations-0/hybrids/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts-file_0.h5

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

# Path to Variable dict used in preprocessing
var_dict: /work/ws/nemo/fr_af1100-Training-Simulations-0/b-Tagging/packages/umami/umami/configs/Dips_Variables.yaml

bool_use_taus: False

exclude: []

# Values for the neural network
NN_structure:
    # NN Training parameters
    lr: 0.001
    batch_size: 15000
    epochs: 200

    # Number of jets used for training
    # To use all: Fill nothing
    nJets_train:

    # Dropout rate. If = 0, dropout is disabled
    dropout: 0

    # Number of output classes
    nClasses: 3

    # Decide if Batch Normalisation is used
    Batch_Normalisation: True

    # Structure of the dense layers for each track
    ppm_sizes: [100, 100, 128]

    # Structure of the dense layers after summing up the track outputs
    dense_sizes: [100, 100, 100, 30]

# Eval parameters for validation evaluation while training
Eval_parameters_validation:
    # Number of jets used for validation
    n_jets: 3e5

    # Charm fraction value used for evaluation
    fc_value: 0.018

    # b Working point used in the evaluation
    WP_b: 0.77

    # Enable/Disable atlas tag
    UseAtlasTag: True

    # fc_value and WP_b are autmoatically added to the plot label
    AtlasTag: "Internal Simulation"
    SecondTag: "\n$\\sqrt{s}=13$ TeV, PFlow jets"

    # Set the datatype of the plots
    plot_datatype: "pdf"
```

It contains the information about the neural network architecture and the training as well as about the files for training, validation and testing. Also evaluation parameters are given for the training evaluation which is performed by the [plotting_epoch_performance.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_epoch_performance.py) script.

Different test files can also be added. These are used for evaluation of the model. They can be defined in their specific catagory. The filenames, i.e `ttbar_r21` are only here for separation. The important options are `Path` and `data_set_name`. First is the path to the test file. The `data_set_name` defines the name in the evaluation files, which are produced with `evaluate_model.py`. Also these names NEED to be unique. When plotting, these `data_set_name` defines which evaluation results are plotted. 

Before starting the training, you need to set some paths for the umami package to find all the tools. Change to the umami dir and run the `setup.py`.

```bash
python setup.py install
```

Note that with the `install` setup, changes that are performed to the scripts after setup are not included! For development and usage of changes without resetup everything, use 

```bash
python setup.py develop
```

After that, you can switch to the folder `umami/umami` and run the training, using the following command

```bash
train_Dips.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). If you want instant performance checks of the model after each epoch during the training, you can use

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips
```

which will write out plots for the light- and c-rejection, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. The `--dips` option defines that the dips tagger is used. In this form, the performance measurements, like light- and c-rejection, will be recalculated using the working point, the `fc` value and the number of validation jets defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). If you don't want to recalculate it, you can give the path to the existing dict with the option `--dict`. For example:

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips --dict dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full/validation_WP0p77_fc0p018_300000jets_Dict.json
```

Here, the `plotting_epoch_performance.py` will get the working point, `fc` value and number of validation jets from the filename. It will not recalculate something. The values are taken for labels in the plots.

## Evaluating the results

After the training is over, the different epochs can be evaluated with ROC plots, output scores, saliency maps and confusion matrices etc. using the build-in scripts. Before plotting these, the model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py).

```bash
evaluate_model.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips -e 5
```

The `-e` options (here `5`) allows to set the training epoch which should be evaluated. The `--dips` option defines that the dips tagger is used.
It will produce .h5 and .pkl files with the evaluations which will be saved in the model folder in an extra folder called `results/`. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. For an explanation, look in the [plotting_umami documentation](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/plotting_umami.md)
