# Umami

## Installation

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

## MC Samples

The FTAG1 derivations and the most recent ntuples for PFlow with the new RNNIP, SMT and the latest DL1* recommendations inside are shown in the following table


| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|MC16a - ttbar             |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r9364_p3985 | |
|MC16a - Z'  |           |     mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r9364_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r9364 |
|MC16d - ttbar             |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p3985 | |
|MC16d - Z'             |  | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10201_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10201 |
|MC16d - Z' extended             |  | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.recon.AOD.e6928_s3126_r10201 |
|MC16e - ttbar             |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10724_p3985 | |
|MC16e - Z'      |   | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10724_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10724 |

The Z' FTAG 1 derivations were requested [here](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)

In addition there are the Z+jets (Z->mumu/nunu) samples for the bb extension

| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|Znunu | | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210   |
|Znunu | | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210  |
|Znunu | | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|Znunu | | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.merge.AOD.e6695_e5984_s3126_r10201_r10210 |
|||||
|Zmumu | | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
|Zmumu | | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.merge.AOD.e5271_s3126_r10201_r10210  |
|Zmumu | | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.merge.AOD.e5271_s3126_r10201_r10210       |
|Zmumu | | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.merge.AOD.e5271_s3126_r10201_r10210 |
|Zmumu | | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.merge.AOD.e5271_e5984_s3126_r10201_r10210 |
|Zmumu | | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r10201_r10210_p3985 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.merge.AOD.e5271_s3126_r10201_r10210 |




ntuples on slac cluster: `/u/ki/nhartman/gpfs/public/btag_hdf5/umami-stdTrkCuts`
