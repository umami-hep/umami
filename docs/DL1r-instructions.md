# Instructions to train DL1r with the umami framework

The following instructions are meant to give a guidline how to reproduce the DL1r results obtained in the last [retraining campaign](http://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/FTAG-2019-005/). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](MC-Samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets

After the previous step the ntuples need to be further processed. We use an undersampling approach to achieve the same pt and eta distribution for all three flavour categories. 
In order to reduce the memory usage we first extract the 3 jet categories separately since e.g. c-jets only make up 8% of the ttbar sample.

This processing can be done using the script [`create_hybrid-large_files.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/blob/master/create_hybrid-large_files.py)


There are several training and validation/test samples to produce. See below a list of all the necessary ones

##### Training Samples (even EventNumber)

`{TTBAR}` are the names of the ttbar-files retrieved from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which can be passed with wildcards and the same for Z' `${ZPRIME}`.

* ttbar (pT < 250 GeV)
    * b-jets
        ```bash
        python create_hybrid-large_files.py --n_split 4 --even --bjets -Z ${ZPRIME} -t ${TTBAR} -n 10000000 -c 1.0 -o ${FPATH}/MC16d_hybrid-bjets_even_1_PFlow-merged.h5
        ```
    * c-jets
        ```bash
        python create_hybrid-large_files.py --n_split 4 --even --cjets -Z ${ZPRIME} -t ${TTBAR} -n 12745953 -c 1.0 -o ${FPATH}/MC16d_hybrid-cjets_even_1_PFlow-merged.h5
        ```
    * light-jets
        ```bash
        python create_hybrid-large_files.py --n_split 5 --even --ujets -Z ${ZPRIME} -t ${TTBAR} -n 20000000 -c 1.0 -o ${FPATH}/MC16d_hybrid-ujets_even_1_PFlow-merged.h5
        ```
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined 
        ```bash
        python create_hybrid-large_files.py --even -Z ${ZPRIME} -t ${TTBAR} -n 9593092 -c 0.0 -o ${FPATH}/MC16d_hybrid-ext_even_0_PFlow-merged.h5
        ```


##### Validation and Test Samples (odd EventNumber)

* ttbar
    ```
    python create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 1.0 -o ${FPATH}/MC16d_hybrid_odd_100_PFlow-no_pTcuts.h5
    ```
* Z' (extended and standard)
    ```
    python create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 0.0 -o ${FPATH}/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts.h5
    ```

The above script will output several files per sample, to not run into memory issues, which can be merged using the [`merge_big.py`](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_big.py) script.

## Preprocessing

After the preparation of the samples, the next step is the processing for the training itself which is done with the script [preprocessing.py](../preprocessing.py).

The configurations for the preprocessing are defined in the config file [PFlow-Preprocessing.yaml](../examples/PFlow-Preprocessing.yaml), you need to adapt it to your needs especially the `file_path`

1. Running the undersampling

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --undersampling
```

2. Retrieving scaling and shifting factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --scaling
```

3. Applying shifting and scaling factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --apply_scales
```

4. Shuffling the samples and writing the samples to disk

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml --write
```

The training Variables for DL1r are defined in [DL1r_Variables.yaml](../umami/configs/DL1r_Variables.yaml).

If you don't want to process them all you can use the already processed samples uploaded to rucio in the dataset `user.mguth:user.mguth.dl1r.trainsamples`.

## Training

After all the files are ready we can start with the training. The config file for the DL1r training is [DL1r-PFlow-Training-config.yaml](../examples/DL1r-PFlow-Training-config.yaml. It contains the information about the neural network architecture as well as about the files for training, validation and testing.

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
 
Next you need to adapt the plotting config file [comparison_dl1.yaml](examples/comparison_dl1.yaml) providing the epoch and model name. The plots can then be retrieved running the following command

```bash
python umami/plotting-DL1.py -c examples/comparison_dl1.yaml 
```
