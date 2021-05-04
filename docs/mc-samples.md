MC Samples
==============

The FTAG1 derivations and the most recent ntuples for PFlow with the new RNNIP, SMT and the latest DL1* recommendations inside are shown in the following table

## Default FTAG Samples (ttbar and Z')

| Sample | h5 ntuples | h5 ntuples (looser track selection)   |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
|MC16a - ttbar           |  |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r9364_p3985 | |
|MC16a - Z'  |         |  |     mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r9364_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r9364 |
|MC16d - ttbar             | user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5 | user.mguth.410470.btagTraining.e6337_s3126_r10201_p3985.EMPFlow_looser-track_selection.2020-07-01-T193555-R26654_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p3985 | |
|MC16d - Z'             | user.mguth.427080.btagTraining.e5362_s3126_r10201_p3985.EMPFlow.2020-02-14-T232210-R26303_output.h5 | user.mguth.427080.btagTraining.e5362_s3126_r10201_p3985.EMPFlow_looser-track_selection.2020-07-01-T193555-R26654_output.h5| mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10201_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10201 |
|MC16d - Z' extended             |  user.mguth.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow.2020-02-15-T225316-R8334_output.h5 | user.mguth.427081.btagTraining.e6928_e5984_s3126_r10201_r10210_p3985.EMPFlow_looser-track_selection.2020-07-01-T195748-R1_output.h5 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p3985 | mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.recon.AOD.e6928_s3126_r10201 |
|MC16e - ttbar           |  |  | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10724_p3985 | |
|MC16e - Z'      | |  | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.deriv.DAOD_FTAG1.e5362_s3126_r10724_p3985 | mc16_13TeV.427080.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime.recon.AOD.e5362_s3126_r10724 |




The Z' & Z+jets FTAG 1 derivations were requested [here](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)


## Z+jets Samples for bb category

### MC16d
For MC16d the p-tag p3985 was [requested](https://its.cern.ch/jira/browse/ATLFTAGDPD-216)

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


### MC16a

[Derivation request](https://prodtask-dev.cern.ch/prodtask/inputlist_with_request/30189/) for p4062

| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|Znunu |  | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.recon.AOD.e6695_s3126_r9364|
|Znunu |  | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r9364 |
|Znunu |  | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r9364 |
|Znunu |  | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r9364 |
|Znunu |  | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r9364 |
|Znunu |  | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r9364  |
|Znunu |  | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r9364 |
|Znunu |  | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r9364_p4062 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.recon.AOD.e6695_s3126_r9364 |
|||||
|Zmumu |  | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.recon.AOD.e5271_s3126_r9364 |
|Zmumu |  | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.recon.AOD.e5271_s3126_r9364 |
|Zmumu |  | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.recon.AOD.e5271_s3126_r9364 |
|Zmumu |  | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.recon.AOD.e5271_s3126_r9364 |
|Zmumu |  | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.recon.AOD.e5271_s3126_r9364 |
|Zmumu |  | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r9364_p4062 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.recon.AOD.e5271_s3126_r9364 |


### MC16e

[Derivation request](https://prodtask-dev.cern.ch/prodtask/inputlist_with_request/30189/) for p4062

| Sample | h5 ntuples    |  FTAG1 derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- |
|Znunu |  | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366010.Sh_221_NN30NNLO_Znunu_PTV70_100_BFilter.recon.AOD.e6695_s3126_r10724|
|Znunu |  | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366011.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r10724 |
|Znunu |  | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366012.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r10724 |
|Znunu |  | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366013.Sh_221_NN30NNLO_Znunu_PTV100_140_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r10724 |
|Znunu |  | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366014.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ0_500_BFilter.recon.AOD.e6695_s3126_r10724 |
|Znunu |  | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366015.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ500_1000_BFilter.recon.AOD.e6695_s3126_r10724  |
|Znunu |  | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366016.Sh_221_NN30NNLO_Znunu_PTV140_280_MJJ1000_E_CMS_BFilter.recon.AOD.e6695_s3126_r10724 |
|Znunu |  | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.deriv.DAOD_FTAG1.e6695_s3126_r10724_p4062 | mc16_13TeV.366017.Sh_221_NN30NNLO_Znunu_PTV280_500_BFilter.recon.AOD.e6695_s3126_r10724 |
|||||
|Zmumu |  | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364102.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_BFilter.recon.AOD.e5271_s3126_r10724 |
|Zmumu |  | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364105.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV70_140_BFilter.recon.AOD.e5271_s3126_r10724 |
|Zmumu |  | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364108.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV140_280_BFilter.recon.AOD.e5271_s3126_r10724 |
|Zmumu |  | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364111.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV280_500_BFilter.recon.AOD.e5271_s3126_r10724 |
|Zmumu |  | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364112.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV500_1000.recon.AOD.e5271_s3126_r10724 |
|Zmumu |  | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.deriv.DAOD_FTAG1.e5271_s3126_r10724_p4062 | mc16_13TeV.364113.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV1000_E_CMS.recon.AOD.e5271_s3126_r10724 |

## Release 22 Samples

| Sample | h5 ntuples | h5 ntuples (looser track selection)   |  DAOD_PHYSVAL derivations| AOD |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
|ttbar | user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.410470.btagTraining.e6337_s3126_r12305_r12253_r12305_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSVAL.e6337_s3126_r12305_r12253_r12305_p4505 | mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.merge.AOD.e6337_e5984_s3126_r12305_r12253_r12305_r12298 |
|Z' Extended (With QSPI, Yes shower weights) | user.alfroch.800030.btagTraining.e7954_s3582_r12305_r12253_r12305_r12298_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.800030.btagTraining.e7954_s3582_r12305_r12253_r12305_r12298_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3582_r12305_r12253_r12305_r12298_p4505 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.merge.AOD.e7954_s3582_r12305_r12253_r12305_r12298 |
|Z' Extended (No QSPI, Yes shower weights) | user.alfroch.800030.btagTraining.e7954_s3126_r12305_r12253_r12305_p4505.EMPFlow.2021-05-04-T093250-R3084_output.h5 | user.alfroch.800030.btagTraining.e7954_s3126_r12305_r12253_r12305_p4505.EMPFlow_loose.2021-05-04-T093534-R11612_output.h5 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_PHYSVAL.e7954_s3126_r12305_r12253_r12305_p4505 | mc16_13TeV.800030.Py8EG_A14NNPDF23LO_flatpT_Zprime_Extended.merge.AOD.e7954_e7400_s3126_r12305_r12253_r12305_r12298 |

# Ntuples on the different clusters

ntuples on slac cluster: `/u/ki/nhartman/gpfs/public/btag_hdf5/umami-stdTrkCuts`

R21 ntuples on Freiburg cluster: `/work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p3985/`    
R22 ntuples on Freiburg cluster: `/work/ws/nemo/fr_af1100-Training-Simulations-0/ntuples_p4505/`
