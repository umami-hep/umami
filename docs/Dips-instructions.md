# Instructions to train DIPS with the umami framework

The following instructions are meant to give a guidline how to reproduce the DIPS results presented in the [DIPS Note](https://cds.cern.ch/record/2718948). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets

After the previous step the ntuples need to be further processed. We use an undersampling approach to achieve the same pt and eta distribution for all three flavour categories.
In order to reduce the memory usage we first extract the 3 jet categories separately since e.g. c-jets only make up 8% of the ttbar sample.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](preprocessing.md) for additional information.
Note, that for running Dips tracks have to be stored in the output hybrid sample. Therefore, the `--tracks` argument needs to be set.

There are several training and validation/test samples to produce. See below a list of all the necessary ones:

##### Training Samples (even EventNumber)

* ttbar (pT < 250 GeV)
    * b-jets
    * c-jets
    * light-jets
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined


##### Validation and Test Samples (odd EventNumber)

* ttbar
* Z' (extended and standard)

## Preprocessing

After the preparation of the samples, the next step is the processing for the training itself which is done with the script [preprocessing.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py).

The configurations for the preprocessing are defined in the config file [PFlow-Preprocessing.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml), you need to adapt it to your needs especially the `file_path`.

1. Running the undersampling

```
preprocessing.py -c ${EXAMPLES}/PFlow-Preprocessing.yaml --var_dict ${CONFIGS}/Dips_Variables.yaml --undersampling --tracks
```

2. Retrieving scaling and shifting factors

```
preprocessing.py -c ${EXAMPLES}/PFlow-Preprocessing.yaml --var_dict ${CONFIGS}/Dips_Variables.yaml --scaling --tracks
```

3. Applying shifting and scaling factors

```
preprocessing.py -c ${EXAMPLES}/PFlow-Preprocessing.yaml --var_dict ${CONFIGS}/Dips_Variables.yaml --apply_scales --tracks
```

4. Shuffling the samples and writing the samples to disk

```
preprocessing.py -c ${EXAMPLES}/PFlow-Preprocessing.yaml --var_dict ${CONFIGS}/Dips_Variables.yaml --write --tracks
```

With the `--tracks` option, the track informations are saved and `--var_dict` is used to give the yaml file with the used variables to the program.
The training variables for DIPS are defined in [Dips_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/Dips_Variables.yaml).

## Training

After all the files are ready we can start with the training. The config file for the Dips training is [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml).

It contains the information about the neural network architecture and the training as well as about the files for training, validation and testing.

Before starting the training, you need to set some paths for the umami package to find all the tools. Change to the umami dir and run the `setup.py`.

```bash
python setup.py install
```

After that, you can switch to the folder `umami/umami` and run the training, using the following command

```bash
train_Dips.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). If you want instant performance checks of the model after each epoch during the training, you can use

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips
```

which will write out plots for the light- and c-rejection, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. The `--dips` option defines that the dips tagger is used. In this form, the performance measurments, like light- and c-rejection, will be recalculated using the working point, the `fc` value and the number of validation jets defined in the [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml). If you don't want to recalculate it, you can give the path to the exisiting dict with the option `--dict`. For example:

```bash
plotting_epoch_performance.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips --dict dips_Loose_lr_0.001_bs_15000_epoch_200_nTrainJets_Full/validation_WP0p77_fc0p018_300000jets_Dict.json
```

Here, the `plotting_epoch_performance.py` will get the working point, `fc` value and number of validation jets from the filename. It will not recalculate something. The values are taken for labels in the plots.

## Evaluating the results

After the training is over, the different epochs can be evaluated with ROC plots, output scores, saliency maps and confusion matrices using the build-in scripts. Before plotting these, the model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py).

```bash
evaluate_model.py -c ${EXAMPLES}/Dips-PFlow-Training-config.yaml --dips -e 5
```

The `-e` options (here `5`) allows to set the training epoch which should be evaluated. The `--dips` option defines that the dips tagger is used.
It will produce .h5 and .pkl files with the evaluations which will be saved in the model folder in an extra folder called `results/`. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. To do this, use the following command.

```bash
plotting_umami.py -c ${EXAMPLES}/plotting_eval_config_dips.yaml -o dips_eval_plots
```

The `-o` option defines the name of the output directory. It will be added to the model folder where also the results are saved. The config file used here is a special plotting config which defines which plots will be generated and how they are labeled. Also in the [plotting_eval_config_dips.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_eval_config_dips.yaml) config file is a set of evaluation parameters. The path to the directory where the models are saved needs to be filled. Also the name and the epoch of the model which is to be evaluated needs to be filled.
