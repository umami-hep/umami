Umami
==============

Installation
-------------

### Docker image

```
singularity exec docker://gitlab-registry.cern.ch/mguth/umami/umami-cpu:latest bash
```

### Manual installation

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
| 5            | b-jets   | 
| 15            | tau-jets   | 
| 44            | double c-jets   | 
| 55            | double b-jets   | 

For the ```HadronConeExclTruthLabelID``` labeling, the categories `4` and `44` as well as `5` and `55` are combined.

### Ntuple preparation for b-,c- & light-jets
These jets are taken from ttbar and Z' events.

After the ntuple production the samples have to be further processed using the script [```create_hybrid-large_files.py```](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/blob/master/create_hybrid-large_files.py)

In case of the default umami (3 categories:b, c, light) the label ```HadronConeExclTruthLabelID``` is used.

There are several training and validation/test samples to produce. See below a list of all the necessary ones

##### Training Samples (even EventNumber)
* ttbar (pT < 250 GeV)
    * b-jets
        ```
        python ${SCRIPT}/create_hybrid.py --n_split 4 --even --bjets -Z ${ZPRIME} -t ${TTBAR} -n 10000000 -c 1.0 -o ${FPATH}/hybrids/MC16d_hybrid-bjets_even_1_PFlow-merged.h5 --write_tracks 
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

The FTAG1 derivations and the most recent ntuples for PFlow with the new RNNIP, SMT and the latest DL1* recommendations inside are shown in the following table


| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|MC16a - ttbar             |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r9364_p3985 | |
|MC16a - Z'  |           |     mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r9364_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r9364 |
|MC16d - ttbar             | user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p3985 | |
|MC16d - Z'             | user.mguth.427080.btagTraining.e5362_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10201_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10201 |
|MC16d - Z' extended             |  user.mguth.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-15-T225316-R8334_output.h5 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.recon.AOD.e6928_s3126_r10201 |
|MC16e - ttbar             |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10724_p3985 | |
|MC16e - Z'      |   | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10724_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10724 |

The Z' FTAG 1 derivations were requested [here](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)

In addition there are the Z+jets (Z->mumu/nunu) samples for the bb extension

| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|Znunu | user.mguth.366010.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210   |
|Znunu | user.mguth.366011.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | user.mguth.366012.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | user.mguth.366013.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | user.mguth.366014.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | user.mguth.366015.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210  |
|Znunu | user.mguth.366016.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | user.mguth.366017.btagTraining.e6695_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|||||
|Zmumu | user.mguth.364102.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
|Zmumu | user.mguth.364105.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.merge.AOD.e5271_s3126_r10201_r10210  |
|Zmumu | user.mguth.364108.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.merge.AOD.e5271_s3126_r10201_r10210       |
|Zmumu | user.mguth.364111.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
|Zmumu | user.mguth.364112.btagTraining.e5271_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.merge.AOD.e5271_e5984_s3126_r10201_r10210 |
|Zmumu | user.mguth.364113.btagTraining.e5271_s3126_r10201_r10210_p3985.EMPFlow.2020-02-14-T235121-R31122_output.h5 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.merge.AOD.e5271_s3126_r10201_r10210 |




ntuples on slac cluster: `/u/ki/nhartman/gpfs/public/btag_hdf5/umami-stdTrkCuts`

ntuples on Freiburg cluster: `/work/ws/nemo/fr_mg1150-umami-0/ntuples-p3985/`
