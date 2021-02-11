# Instructions to train DL1r with the umami framework

The following instructions are meant to give a guidline how to reproduce the DL1r results obtained in the last [retraining campaign](http://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/FTAG-2019-005/). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [mc-samples.md](mc-samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [mc-samples.md](mc-samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets

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


##### Validation and Test Samples (odd EventNumber)

* ttbar
* Z' (extended and standard)

### Preprocessing

After you produced these samples during the preparation step and merged the output files, you can run the
preprocessing remaining four stages:


1. Running the undersampling

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

If you don't want to process them all you can use the already processed samples uploaded to rucio in the dataset `user.mguth:user.mguth.dl1r.trainsamples`. Note, that in this case you need to supply the associated[dictionary with scaling factors](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/assets/PFlow-scale_dict-22M.json) by hand, which is otherwise creating during the preprocessing.


## Training

After all the files are ready we can start with the training. The config file for the DL1r training is [DL1r-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/DL1r-PFlow-Training-config.yaml).

It contains the information about the neural network architecture as well as about the files for training, validation and testing.

To run the training, use the following command

```bash
train_DL1.py -c examples/DL1r-PFlow-Training-config.yaml
```

You can check the performance of your model during the training via

```bash
train_DL1.py -c examples/DL1r-PFlow-Training-config.yaml -p
```

this will write out plots for the light- and c-rejection per epoch.

## Performance Evaluation

Finally we can evaluate our model. These scripts are in very work in progress state.

You first need to choose which epoch you want to use for the evaluation (easiest by looking at the performance plots vs. epochs) and then run

```bash
python umami/evaluate_model.py -c examples/DL1r-PFlow-Training-config.yaml -e 230 --dl1
```

Next you need to adapt the plotting config file [plotting_umami_config_DL1r.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/plotting_umami_config_DL1r.yaml) providing the epoch and model name. The plots can then be retrieved running the following command

```bash
python umami/plotting_umami.py -c examples/plotting_umami_config_DL1r.yaml
```
