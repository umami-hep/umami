Umami
==============

Installation
-------------

### Docker image

```
singularity exec docker://gitlab-registry.cern.ch/mguth/umami/umami-cpu:latest bash
```
besides the CPU image, there is also a GPU image available which is especially useful for the training step
```
singularity exec --nv docker://gitlab-registry.cern.ch/mguth/umami/umami-gpu:latest bash
```

### Environment setup

you can check out this repository via `git clone` and then run
```python setupy.py install```
this will install the umami package
if you want to modify the code you should install it via
```python setupy.py develop```

## Testing
The unit test you can run via
```
pytest ./umami/tests/ -v
```
Preprocessing
---------------
For the training of umami the ntuples are used as specified in the section [MC Samples](#mc-samples).

Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the table in the section [MC Samples](#mc-samples).

There are two different labeling available, the ```HadronConeExclTruthLabelID``` and the ```HadronConeExclExtendedTruthLabelID``` which includes extended jet categories:

| HadronConeExclExtendedTruthLabelID | Category    |
| ------------- | ---------------- |
| 0            | light jets   |
| 4            | c-jets   | 
| 5, 54            | single b-jets   | 
| 15            | tau-jets   | 
| 44            | double c-jets   | 
| 55            | double b-jets   | 

For the ```HadronConeExclTruthLabelID``` labeling, the categories `4` and `44` as well as `5`, `54` and `55` are combined.

### Ntuple preparation for b-,c- & light-jets
These jets are taken from ttbar and Z' events.

After the ntuple production the samples have to be further processed using the script [```create_hybrid-large_files.py```](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/blob/master/create_hybrid-large_files.py)

In case of the default umami (3 categories:b, c, light) the label ```HadronConeExclTruthLabelID``` is used.

There are several training and validation/test samples to produce. See below a list of all the necessary ones

##### Training Samples (even EventNumber)
* ttbar (pT < 250 GeV)
    * b-jets
        ```
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 4 --even --bjets -Z ${ZPRIME} -t ${TTBAR} -n 10000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-bjets_even_1_PFlow-merged.h5 --write_tracks 
        ```
    * c-jets
        ```
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 4 --even --cjets -Z ${ZPRIME} -t ${TTBAR} -n 12745953 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-cjets_even_1_PFlow-merged.h5 --write_tracks
        ```
    * light-jets
        ```
        python ${SCRIPT}/create_hybrid-large_files.py --n_split 5 --even --ujets -Z ${ZPRIME} -t ${TTBAR} -n 20000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-ujets_even_1_PFlow-merged.h5 --write_tracks
        ```
* Z' (pT > 250 GeV) -> extended Z'
    * b, c, light-jets combined 
        ```
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

The above script will output several files per sample which can be merged using the [```merge_big.py```](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_big.py) script.



### Ntuple Preparation for bb-jets
The double b-jets will be taken from Znunu and Zmumu samples.


Since the double b-jets represent only a fraction of the jets, they can be filtered out using the [```merge_ntuples.py```](https://gitlab.cern.ch/mguth/hdf5_manipulator/blob/master/merge_ntuples.py) script from the [hdf5-manipulator](https://gitlab.cern.ch/mguth/hdf5_manipulator).



## MC Samples

The full list of MC samples are located under [docs/MC-Samples.md](https://gitlab.cern.ch/mguth/umami/-/blob/master/docs/MC-Samples.md).
