# Instructions to train DIPS with the umami framework

The following instructions are meant to give a guidline how to reproduce the DIPS results presented in the [DIPS Note](https://cds.cern.ch/record/2718948). It is focused on the PFlow training.


## Sample Preparation

The first step is to obtain the samples for the training. All the samples are listed in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md). For the PFlow training only the ttbar and extended Z' samples from 2017 data taking period (MC16d) were used.

The training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps the jets from the FTAG1 derivations directly into hdf5 files. The processed ntuples are also listed in the table in [MC-Samples.md](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/MC-Samples.md) which can be used for training.

### Ntuple preparation for b-,c- & light-jets

After the previous step the ntuples need to be further processed. We use an undersampling approach to achieve the same pt and eta distribution for all three flavour categories. 
In order to reduce the memory usage we first extract the 3 jet categories separately since e.g. c-jets only make up 8% of the ttbar sample.

This processing can be done using the script [`create_hybrid-large_files.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/blob/master/create_hybrid-large_files.py)


There are several training(even) and validation/test(odd) samples to produce. See below a list of all the necessary ones

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
    ```bash
    python create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 1.0 -o ${FPATH}/MC16d_hybrid_odd_100_PFlow-no_pTcuts.h5
    ```
* Z' (extended and standard)
    ```bash
    python create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 0.0 -o ${FPATH}/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts.h5
    ```

The above script will output several files per sample, to not run into memory issues, which can be merged using the [`merge_big.py`](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_big.py) script.

## Preprocessing

After the preparation of the samples, the next step is the processing for the training itself which is done with the script [preprocessing.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/preprocessing.py).

The configurations for the preprocessing are defined in the config file [PFlow-Preprocessing.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/PFlow-Preprocessing.yaml), you need to adapt it to your needs especially the `file_path`. To run the Proprocessing for the ttbar-only files, just change to `PFlow-Preprocessing-ttbar.yaml`.

1. Running the undersampling

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml -t -v Dips_Variables.yaml --undersampling
```

2. Retrieving scaling and shifting factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml -t -v Dips_Variables.yaml --scaling
```

3. Applying shifting and scaling factors

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml -t -v Dips_Variables.yaml --apply_scales
```

4. Shuffling the samples and writing the samples to disk

```
preprocessing.py -c examples/PFlow-Preprocessing.yaml -t -v Dips_Variables.yaml --write
```

With the `-t` option, the track informations are saved and `-v` is used to give the yaml file with the used variables to the program.
The training Variables for DIPS are defined in [Dips_Variables.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/Dips_Variables.yaml).

If you don't want to process them all you can use the already processed samples uploaded to rucio in the dataset `user.mguth:user.mguth.dl1r.trainsamples`.

## Training

After all the files are ready we can start with the training. The config file for the Dips training is [Dips-PFlow-Training-config.yaml](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/Dips-PFlow-Training-config.yaml).

It contains the information about the neural network architecture and the training as well as about the files for training, validation and testing.

Before starting the training, you need to set some paths for the umami package to find all the tools. Change to the umami dir and run the `setup.py`.

```bash
python setup.py install
```

After that, you can switch to the folder `umami/umami` and run the training, using the following command

```bash
train_Dips.py -c examples/Dips-PFlow-Training-config.yaml
```

The results after each epoch will be saved to the `umami/umami/dips/` folder. If you want instant performance checks of the model after each epoch during the training, you can use

```bash
train_Dips.py -c examples/Dips-PFlow-Training-config.yaml -p
```

which will write out plots for the light- and c-rejection, accuracy and loss per epoch to `umami/umami/dips/plots/`.

## Evaluating the results

After the training is over, the different epochs can be evaluated with ROC plots and confusion matrices using the build-in scripts. Before plotting these, the model needs to be evaluated using the [evaluate_model.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/evaluate_model.py). 

```bash
evaluate_model.py -c examples/Dips-PFlow-Training-config.yaml -e 5 --dips
```

The 5 gives the epoch which is to evaluate. It will produce .h5 files with the evaluations. After, the [plotting_umami.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/plotting_umami.py) script can be used to plot the results. To do this, use the following command.

```bash
plotting_umami.py -c examples/plotting_umami_config_dips.yaml -o dips_eval_plots
```

The `-o` option defines the name of the output directory. It will be added to the model folder where also the results are saved. The config file used here is a special plotting config which defines which plots will be generated and how they are labeled.