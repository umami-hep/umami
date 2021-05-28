# Instructions to train DL1r with the umami framework

The following instructions are meant to give a guideline how to reproduce the DL1r results obtained in the last [retraining campaign](http://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/FTAG-2019-005/). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [mc-samples.md](mc-samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [mc-samples.md](mc-samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets (+tau-jets)

After the previous step the ntuples need to be further processed. We use an undersampling approach to achieve the same pt and eta distribution for all three flavour categories.
In order to reduce the memory usage we first extract the 3 jet categories separately since e.g. c-jets only make up 8% of the ttbar sample.

This processing can be done using the preprocessing capabilities of Umami via the [`preprocessing.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py) script.

Please refer to the [documentation on preprocessing](preprocessing.md) for additional information.
Note, that for running DL1r no tracks have to be stored in the output hybrid sample. Therefore, the `--tracks` argument can be omitted.

There are several training and validation/test samples to produce. See below a list of all the necessary ones:

##### Training Samples (even EventNumber)

* ttbar (pT < 250 GeV)
    * b-jets
    * c-jets
    * light-jets
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined

Note: to uses taus, the parameter `bool_process_taus` in the preprocessing must be activated and a ttbar sample of taus (pT < 250 GeV) must be generated, similarly to the other flavours. In the configuration, (for the samples entry in preparation), the taus set can be collected by specifying `taujets` as `category` for ttbar (as shown [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/taus/examples/PFlow-Preprocessing.yaml#L59-69)).

##### Validation and Test Samples (odd EventNumber)

* ttbar
* Z' (extended and standard)

### Preprocessing

After you produced these samples during the preparation step and merged the output files, you can run the
preprocessing remaining four stages:


1. Running the undersampling
```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict umami/configs/DL1r_Variables.yaml --undersampling
```

2. Retrieving scaling and shifting factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict umami/configs/DL1r_Variables.yaml --scaling
```

3. Applying shifting and scaling factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict umami/configs/DL1r_Variables.yaml --apply_scales
```

4. Shuffling the samples and writing the samples to disk

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --var_dict umami/configs/DL1r_Variables.yaml --write
```

The training Variables for DL1r are defined in [DL1r_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/DL1r_Variables.yaml).

Important: when using taus, it is not advised to use the `count` sampling method, as defined in the [preprocessing configuration](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/taus/examples/PFlow-Preprocessing.yaml#L127-130). Indeed, the tau statistics is much lower in the data and this would throw away far too many jets. Instead, use `count_bcl_weight_tau` sampling method to have an exact match of the jets flavour and a proportional distribution of taus. 

If you don't want to process some samples yourself, you can use the already preprocessed samples uploaded to rucio in the datasets `user.mdraguet.dl1r.R21.PFlowJetsDemoSamples` for DL1r or `user.mdraguet.dl1d.R21.PFlowJetsDemoSamples` for DL1d (RNNIP replaced by DIPS). These datasets do not have taus included. Note that you need to download both the datasets and the associated dictionary with scaling factors (+ the dictionary of variable). There are two test samples available: an hybrid (ttbar + Z'-ext) and a Z'-ext solely. Each should be manually cut in 2 to get a test and validation file. The data comes from:
- mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p4060
- mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p4060



## Training

After all the files are ready we can start with the training. The config file for the DL1r training is [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml).

It contains the information about the neural network architecture as well as about the files for training, validation and testing. To use taus in training, the preprocessing given in the training config must have the parameter `bool_process_taus` set to `True` and the parameter `bool_use_taus` in the training config should also be set to `True`. 

To run the training, use the following command

```bash
train_DL1.py -c examples/DL1r-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/MODELNAME/` folder. The modelname is the name defined in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml). If you want instant performance checks of the model after each epoch during the training, you can use

```bash
plotting_epoch_performance.py -c examples/DL1r-PFlow-Training-config.yaml --dl1
```

which will write out plots for the light- and c-rejection, accuracy and loss per epoch to `umami/umami/MODELNAME/plots/`. In this form, the performance measurements, like light- and c-rejection, will be recalculated using the working point, the `fc` value and the number of validation jets defined in the [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml). If taus are used in the training too, they will be included in these plots. 

## Performance Evaluation

Finally we can evaluate our model. These scripts are being continuously updated.

You first need to choose which epoch you want to use for the evaluation (easiest by looking at the performance plots vs. epochs) and then run

```bash
python umami/evaluate_model.py -c examples/DL1r-PFlow-Training-config.yaml -e 230 --dl1
```

Next you need to adapt the plotting config file [plotting_umami_config_DL1r.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_umami_config_DL1r.yaml) providing the epoch and model name. You can use the parameter `bool_use_taus` to include taus in the plotting. When including taus, some plot configs have to been modified accordingly (e.g., `prediction_labels` must have a `ptau` as final entry in the list). The plots can then be retrieved running the following command

```bash
python umami/plotting_umami.py -c examples/plotting_umami_config_DL1r.yaml
```
