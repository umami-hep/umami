# Umami

## Installation

### Docker image

```bash
singularity exec docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest bash
```

besides the CPU image, there is also a GPU image available which is especially useful for the training step

```bash
singularity exec --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami:latest-gpu bash
```

### Manual setup

ALternatively you can also check out this repository via `git clone` and then run

```bash
python setupy.py install
```

this will install the umami package

If you want to modify the code you should run instead

```bash
python setupy.py develop
```

which creates a symlink to the repository.

## Testing & Linter

The unit test you can be run via

```bash
pytest ./umami/tests/ -v
```

In order to run the code style checker `flake8` use the following command

```bash
flake8 ./umami
```

## DL1r instructions

If you want to train or evaluate DL1r please follow the [DL1r-instructions](docs/DL1r-instructions.md).


## Preprocessing

For the training of umami the ntuples are used as specified in the section [MC Samples](#mc-samples).

Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the table in the file [MC-Samples.md](docs/MC-Samples.md).

There are two different labeling available, the `HadronConeExclTruthLabelID` and the `HadronConeExclExtendedTruthLabelID` which includes extended jet categories:

| HadronConeExclExtendedTruthLabelID | Category    |
| ------------- | ---------------- |
| 0            | light jets   |
| 4            | c-jets   |
| 5, 54            | single b-jets   |
| 15            | tau-jets   |
| 44            | double c-jets   |
| 55            | double b-jets   |

For the `HadronConeExclTruthLabelID` labeling, the categories `4` and `44` as well as `5`, `54` and `55` are combined.

### Ntuple preparation for b-,c- & light-jets

These jets are taken from ttbar and Z' events.

After the ntuple production the samples have to be further processed using the script [`create_hybrid-large_files.py`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/blob/master/create_hybrid-large_files.py)

In case of the default umami (3 categories:b, c, light) the label `HadronConeExclTruthLabelID` is used.

There are several training and validation/test samples to produce. See below a list of all the necessary ones

##### Training Samples (even EventNumber)

* ttbar (pT < 250 GeV)
    * b-jets
        ```bash
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 4 --even --bjets -Z ${ZPRIME} -t ${TTBAR} -n 10000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-bjets_even_1_PFlow-merged.h5 --write_tracks 
        ```
    * c-jets
        ```bash
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 4 --even --cjets -Z ${ZPRIME} -t ${TTBAR} -n 12745953 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-cjets_even_1_PFlow-merged.h5 --write_tracks
        ```
    * light-jets
        ```bash
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 5 --even --ujets -Z ${ZPRIME} -t ${TTBAR} -n 20000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-ujets_even_1_PFlow-merged.h5 --write_tracks
        ```
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined 
        ```bash
        python ${SCRIPT}/create_hybrid-large_files.py --even -Z ${ZPRIME} -t ${TTBAR} -n 9593092 -c 0.0 -o ${FPATH}/hybrids/MC16d_hybrid-ext_even_0_PFlow-merged.h5 --write_tracks
        ```


##### Validation and Test Samples (odd EventNumber)

* ttbar
    ```
    python ${SCRIPT}/create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid_odd_100_PFlow-no_pTcuts.h5 --write_tracks
    ```
* Z' (extended and standard)
    ```
    python ${SCRIPT}/create_hybrid-large_files.py --n_split 2 --odd --no_cut -Z ${ZPRIME} -t ${TTBAR} -n 4000000 -c 0.0 -o ${FPATH}/hybrids/MC16d_hybrid-ext_odd_0_PFlow-no_pTcuts.h5 --write_tracks
    ```

The above script will output several files per sample which can be merged using the [`merge_big.py`](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_big.py) script.

### Ntuple Preparation for bb-jets

The double b-jets will be taken from Znunu and Zmumu samples.

Since the double b-jets represent only a fraction of the jets, they can be filtered out using the [`merge_ntuples.py`](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_ntuples.py) script from the [hdf5-manipulator](https://gitlab.cern.ch/mguth/hdf5_manipulator).
